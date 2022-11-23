import argparse
import csv
import datetime
import json
import logging
import pickle
import os
import shutil
import time
from typing import Callable
from typing import NamedTuple
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from distiller import Distiller
from src.data.my_prepro import add_special_tokens_
from src.metrics.f1_score import f1_per_batch
from src.metrics.rougescore import rouge_per_batch
from torch.utils.tensorboard import SummaryWriter
from torch.distributed import get_rank, get_world_size
from tqdm import tqdm
from transformers import logging as trsfmr_logging
#trsfmr_logging.set_verbosity_warning()
#from src.model import GPT2Config, GPT2LMHeadModel, Adam
from transformers import GPT2Config
from transformers import GPT2Tokenizer

from transformers import GPT2LMHeadModel
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from src.utils import bool_flag
from src.utils import save_model 
from utils_comp import init_gpu_params, set_seed

from src.data.loader import dbBucketingDataLoader, DistributedBucketingDataLoader
from src.model.optim import Adam
from src.utils_.distributed import all_reduce_and_rescale_tensors, all_gather_list
from src.decoder_hugFace import greedy, sample_seq_batch

import warnings
warnings.filterwarnings("ignore")

import sys

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

SPECIAL_TOKENS = ['<speaker1>', '<speaker2>']
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>'}
MODEL_INPUTS = ["input_ids", "position_ids", "token_type_ids", "lm_labels"]
MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
}

global best_val_f1 
best_val_f1 = float("-inf")

def sanity_checks(args):
    """
    A bunch of args sanity checks to perform even starting...
    """
    assert (args.mlm > 0.0 and args.alpha_mlm > 0.0) or (not (args.mlm ==0.0 and args.alpha_mlm == 0.0))
    assert (args.alpha_mlm > 0.0 and args.alpha_clm == 0.0) or (args.alpha_mlm == 0.0 and args.alpha_clm > 0.0)
    if args.mlm:
        assert os.path.isfile(args.token_counts)
        assert (args.student_type in ["roberta", "distilbert"]) and (args.teacher_type in ["roberta", "bert"])
    else:
        assert (args.student_type in ["gpt2"]) and (args.teacher_type in ["gpt2"])

    assert args.teacher_type == args.student_type or (
        args.student_type == "distilbert" and args.teacher_type == "bert"
    )
    assert os.path.isfile(args.student_config)
    if args.student_pretrained_weights is not None:
        assert os.path.exists(args.student_pretrained_weights)

    if args.freeze_token_type_embds:
        assert args.student_type in ["roberta"]

    assert args.alpha_ce >= 0.0
    assert args.alpha_mlm >= 0.0
    assert args.alpha_clm >= 0.0
    assert args.alpha_mse >= 0.0
    assert args.alpha_cos >= 0.0
    assert args.alpha_ce + args.alpha_mlm + args.alpha_clm + args.alpha_mse + args.alpha_cos > 0.0

def freeze_pos_embeddings(student, args):
    if args.student_type == "roberta":
        student.roberta.embeddings.position_embeddings.weight.requires_grad = False
    elif args.student_type == "gpt2":
        student.transformer.wpe.weight.requires_grad = False

def freeze_token_type_embeddings(student, args):
    if args.student_type == "roberta":
        student.roberta.embeddings.token_type_embeddings.weight.requires_grad = False

def setup_args():
    parser = argparse.ArgumentParser(description="Train dialog model")

    # parser.add_argument("config_file", help="path to json file with the training configuration.") ... to be modified
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--val_batch_size", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.25e-4)
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--n_epochs", type=int, help="number of epochs", default=100)
    parser.add_argument("--max_grad_norm", type=int, help="max_norm for gradient clipping", default=1.0)
    parser.add_argument("--max_steps", type=int, help="max number of training steps", default=0.0)
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--log_every", type=int, default=1000, help="number of epochs that pass before saving model")
    parser.add_argument('--local_rank', type=int, default=-1,help='for torch.distributed')
    parser.add_argument('--grad_acc_steps', type=int, default=16, help='number of steps before parameter update')
    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument('--top-k', type=int, default=10, required=False, help='beam width')
    parser.add_argument('--top-p', type=int, default=0.9, required=False, help='top probabilities for filtering')
    parser.add_argument("--save_every", type=int, default=5, help="number of epochs that pass before saving model")
    parser.add_argument("--skip_eval", type=bool, default=False)
    parser.add_argument("--resume", type=bool_flag, required=True, default=True)
    parser.add_argument("--model_size", type=str, 
                        help="the size of the base model: small, medium or large")
    parser.add_argument("--model_folder", required=True,
                        help="directory where models(pretrained or finetuned) be saved ")
    parser.add_argument("--pretrained_tokenizer_directory",
                        type=str, help="path to the directory containing the pretrained tokenizer")
    parser.add_argument("--log_dir", type=str, required=True)  # default='./Experiments')
    parser.add_argument("--train_input_file", type=str, help="path to file containing training set")
    parser.add_argument("--val_input_file", type=str, help="path to file containing validation set")

    # distillation params
    parser.add_argument("--dump_path", type=str, required=True, default= '../models/distill_dump',
                        help="The output directory (log, checkpoints, parameters, etc.)"
    )
    parser.add_argument("--student_config", type=str, required=True, help="Path to the student configuration.")
    parser.add_argument("--student_pretrained_weights", default=None, type=str,
                        help="Load student initialization checkpoint."
    )
    parser.add_argument("--student_type", type=str, choices=["distilbert", "roberta", "gpt2"], required=True,
                        default="gpt2", help="The student type (DistilBERT, RoBERTa).",
    )
    parser.add_argument("--teacher_type", choices=["bert", "roberta", "gpt2"], default='gpt2',
                        required=True, help="Teacher type (BERT, RoBERTa)."
    )
    parser.add_argument("--teacher_name", type=str, required=True, help="The teacher model.")
    parser.add_argument("--teacher_pretrained_weights", type=str, required=True, help="path to the saved pretrained teacher model.")
    parser.add_argument("--temperature", default=2.0, type=float, help="Temperature for the softmax temperature.")
    parser.add_argument("--alpha_ce", default=0.5, type=float, 
                        help="Linear weight for the distillation loss. Must be >=0."
    )
    parser.add_argument("--alpha_mlm",default=0.0, type=float,
                        help="Linear weight for the MLM loss. Must be >=0. Should be used in coonjunction with `mlm` flag.",
    )
    parser.add_argument("--n_gpu", type=int, default=1, help="Number of GPUs in the node.")
    parser.add_argument("--alpha_clm", default=0.5, type=float, help="Linear weight for the CLM loss. Must be >=0.")
    parser.add_argument("--alpha_mse", default=0.0, type=float, help="Linear weight of the MSE loss. Must be >=0.")
    parser.add_argument("--alpha_cos", default=0.0, type=float, help="Linear weight of the cosine embedding loss. Must be >=0.")
    parser.add_argument("--mlm", action="store_true", help="The LM step: MLM or CLM. If `mlm` is True, the MLM is used over CLM.")
    parser.add_argument("--mlm_mask_prop", default=0.15, type=float, help="Proportion of tokens for which we need to make a prediction.",)
    parser.add_argument("--word_mask", default=0.8, type=float, help="Proportion of tokens to mask out.")
    parser.add_argument("--word_keep", default=0.1, type=float, help="Proportion of tokens to keep.")
    parser.add_argument("--word_rand", default=0.1, type=float, help="Proportion of tokens to randomly replace.")
    parser.add_argument("--mlm_smoothing", default=0.7, type=float,
                        help="Smoothing parameter to emphasize more rare tokens (see XLM, similar to word2vec).",
    )
    parser.add_argument("--token_counts", type=str, help="The token counts in the data_file for MLM.")

    parser.add_argument("--restrict_ce_to_mask",action="store_true",
                        help="If true, compute the distilation loss only the [MLM] prediction distribution.",
    )
    parser.add_argument("--freeze_pos_embs", action="store_true",
                        help="Freeze positional embeddings during distillation. For student_type in ['roberta', 'gpt2'] only.",)
    parser.add_argument("--freeze_token_type_embds", action="store_true",
                        help="Freeze token type embeddings during distillation if existent. For student_type in ['roberta'] only.",
    )
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size (for each process).") 
    parser.add_argument("--warmup_prop", default=0.05, type=float, help="Linear warmup proportion.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--initializer_range", default=0.02, type=float, help="Random initialization range.")

    parser.add_argument("--fp16", action="store_true", 
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                            "See details at https://nvidia.github.io/apex/amp.html",
                            )
    parser.add_argument("--log_interval", type=int, default=500, help="Tensorboard logging interval.")
    parser.add_argument("--checkpoint_interval", type=int, default=4000, help="Checkpoint interval.")

    args = parser.parse_args()
    return args

def main(args: Optional[NamedTuple]) -> None:
    """
    Train and evaluate a dialog model on specific training configuration.
    """
    # Initialize distributed training if needed
    args.distributed = bool(args.local_rank != -1)

    if not args.distributed: # cpu if no gpu else the only gpu available
        logger.info('CUDA available? {}'.format(str(torch.cuda.is_available())))
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #data

    # ARGS #from HF's distillation repo
    init_gpu_params(args)
    set_seed(args)
    if args.is_master:
        if not os.path.exists(args.dump_path):
            os.makedirs(args.dump_path)
        logger.info(f"Experiment will be dumped and logged in {args.dump_path}")

        # SAVE PARAMS #
        logger.info(f"Param: {args}")
        with open(os.path.join(args.dump_path, "parameters.json"), "w") as f:
            # lil hack
            _device = args.device
            del args.device
            json.dump(vars(args), f, indent=4)
            args.device=_device

    student_config_class, student_model_class, _ = MODEL_CLASSES[args.student_type]
    teacher_config_class, teacher_model_class, teacher_tokenizer_class = MODEL_CLASSES[args.teacher_type]

    # TOKENIZER #
    tokenizer = teacher_tokenizer_class.from_pretrained(args.teacher_name)
    #tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
    #tokenizer.add_tokens(SPECIAL_TOKENS)
    args.max_model_input_size = tokenizer.max_model_input_sizes[args.teacher_name]


    # Data Loading
    assert os.path.exists(args.train_input_file),  "Train input file does not exist"
    if args.mlm:
        logger.info(f"Loading token counts from {args.token_counts} (already pre-computed)")
        with open(args.token_counts, "rb") as fp:
            counts = pickle.load(fp)

        token_probs = np.maximum(counts, 1) ** -args.mlm_smoothing
        for idx in ATTR_TO_SPECIAL_TOKEN.values():
            token_probs[idx] = 0.0  # do not predict special tokens
        token_probs = torch.from_numpy(token_probs)
    else:
        token_probs = None

    if not args.distributed:
        train_dataloader = dbBucketingDataLoader(args.train_input_file,
                                                args.train_batch_size,
                                                args.max_seq_len)
        #val_dataloader = dbBucketingDataLoader(args.val_input_file,
        #                                        args.val_batch_size,
        #                                        args.max_seq_len)                            
    else: 
        logger.info("loading dist_data")
        train_dataloader = DistributedBucketingDataLoader(get_rank(), get_world_size(),
                                                        args.train_input_file, args.train_batch_size,
                                                        args.max_seq_len)
        #val_dataloader = DistributedBucketingDataLoader(get_rank(), get_world_size(),
        #                                                args.val_input_file,args.val_batch_size,
        #                                                args.max_seq_len)
        logger.info(f"Train loader size: {len(train_dataloader)} val loader size: {len(val_dataloader)}")
    
    # Model
    # STUDENT #
    logger.info(f"Loading student config from {args.student_config}")
    stu_architecture_config = student_config_class.from_pretrained(args.student_config)
    stu_architecture_config.output_hidden_states = True

    if args.student_pretrained_weights is not None:

        logger.info(f"Loading pretrained weights from {args.student_pretrained_weights}")
        student = student_model_class.from_pretrained(args.student_pretrained_weights, config=stu_architecture_config)
    else:
        student = student_model_class(stu_architecture_config)
    print(student)
    if args.n_gpu > 0:
        student.to(f"cuda:{args.local_rank}")
    logger.info("Student loaded.")

    # TEACHER #
    teacher = teacher_model_class.from_pretrained(args.teacher_name, output_hidden_states=True)
    if args.n_gpu > 0:
        teacher.to(f"cuda:{args.local_rank}")
    logger.info(f"Teacher loaded from {args.teacher_name}.")

    # resize model embeddings dim
    teacher.resize_token_embeddings(len(tokenizer))
    student.resize_token_embeddings(len(tokenizer))
    print(len(tokenizer), teacher.config.vocab_size)
    # FREEZING #
    if args.freeze_pos_embs:
        freeze_pos_embeddings(student, args)
    if args.freeze_token_type_embds:
        freeze_token_type_embeddings(student, args)

    # SANITY CHECKS #
    assert student.config.vocab_size == teacher.config.vocab_size
    assert student.config.hidden_size == teacher.config.hidden_size
    assert student.config.max_position_embeddings == teacher.config.max_position_embeddings
    if args.mlm:
        assert token_probs.size(0) == stu_architecture_config.vocab_size

    # DISTILLER #
    torch.cuda.empty_cache()
    distiller = Distiller(
        params=args, dataloader=train_dataloader, token_probs=token_probs, student=student, teacher=teacher
    )
    print(args.device)
    distiller.train()
    logger.info("Let's go get some drinks.")
   

if __name__ == "__main__":
    
    args = setup_args()
    # sanity check the args
    sanity_checks(args)

    tb_writer = SummaryWriter(log_dir=args.log_dir)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    main(args)
