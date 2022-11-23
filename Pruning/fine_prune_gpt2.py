import argparse
import logging
import numpy as np
import os
import sys
import torch
import torch.nn as nn
from modeling import MaskedGPT2Config, MaskedGPT2LMHeadModel
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from torch.distributed import get_rank, get_world_size
from tqdm import tqdm
from types import SimpleNamespace
from src.loader import Loader, DistLoader
from src.eval_utils import eval_model
from src.utils import (
    Adam,
    CEloss,
    set_seed,
    set_lr,
    _save_checkpoint,
    _load_checkpoint
)
# for fp16 training
try:
    from apex import amp
    amp.register_half_function(torch, "einsum")
except ImportError as no_apex:
    # error handling in TrainManager object construction
    pass
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "gpt2-medium": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    "masked_gpt2": (MaskedGPT2Config, MaskedGPT2LMHeadModel, GPT2Tokenizer)
}

global grad_step, global_step, epoch, best_f1
grad_step = 0
global_step = 0
best_f1 = -float('inf')


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default=None, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument('--model_path', type=str, required=False,
                        help='path to pretrained model')
    parser.add_argument("--config_name", default="", type=str, 
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument('--seed', type=int, default=42, required=False,
                        help='random seed')
    parser.add_argument("--resume", action='store_true', 
                        help='resume training from a ckpt file')
    parser.add_argument('--ckpt_file', type=str, required=False,
                        help='saved model if --resume is True')
    parser.add_argument('--warmup_steps', default=0, type=int, 
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--train_file', type=str, required=True,
                        help='path to the training corpus')
    parser.add_argument('--eval_file', type=str, required=True,
                        help='path to the validation corpus')
    parser.add_argument('--final_threshold', type=float, default=0.7,
                        help="Final value of the threshold (for scheduling).")
    parser.add_argument('--initial_threshold', type=float, default=1.0, 
                        help="Initial value of the threshold (for scheduling).")
    parser.add_argument('--final_lambda', type=float, default=0.0,
                        help="Regularization intensity (used in conjunction with `regularization`.",)
    parser.add_argument('--initial_warmup', default=1,  type=int,
                        help="Run `initial_warmup` * `warmup_steps` steps of threshold warmup during" 
                        "which threshold stays at its `initial_threshold` value (sparsity schedule).")
    parser.add_argument('--final_warmup', default=2, type=int,
                        help="Run `final_warmup` * `warmup_steps` steps of threshold cool-down during which"
                        "threshold stays at its final_threshold value (sparsity schedule).")
    parser.add_argument('--global_topk', action="store_true", help="Global TopK on the Scores.")
    parser.add_argument('--global_topk_frequency_compute', default=25, type=int,
                        help="Frequency at which we compute the TopK global threshold.")
    parser.add_argument("--pruning_method", default="topK", type=str, choices=['topK', 'l0', 'magnitude', 'sigmoied_threshold'],
                        help="Pruning Method (l0 = L0 regularization, magnitude = Magnitude pruning, topK = Movement pruning"
                             "sigmoied_threshold = Soft movement pruning).")
    parser.add_argument('--regularization', default=None,
                        help="Add L0 or L1 regularization to the mask scores.")
    parser.add_argument('--grad_acc_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--val_step', type=int, default=100, 
                        help="Number of grad_step before evaluating")
    parser.add_argument('--local_rank', type=int, default=-1, 
                        help="For distributed training: local_rank")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument('--lr', type=float, required=True, default=5e-5,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--lr_schedule", type=str, default='noam', 
                        help="learning rate schedule")
    parser.add_argument('--log_dir', type=str, required=True,
                        help='location for training and validation logs')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='location for saving model checkpoints and logs')
    parser.add_argument('--max_seq_len', type=int, default=256,
                        help="The maximum total input sequence length after WordPiece tokenization."
                        "Sequences longer than this will be truncated, and sequences shorter than this will be padded.",)
    parser.add_argument("--weight_decay", default=0.0, type=float, 
                        help="Weight decay if we apply some.")
    parser.add_argument("--mask_init", default="constant", type=str,
                        help="Initialization method for the mask scores. Choices: constant, uniform, kaiming.",)
    parser.add_argument("--mask_scale", default=0.0, type=float, 
                        help="Initialization parameter for the chosen initialization method.")
    parser.add_argument("--mask_scores_learning_rate", default=1e-2,
                        type=float, help="The Adam initial learning rate of the mask scores.")
    parser.add_argument("--max_steps", default=3000, required=True, type=int,
                        help="The Adam initial learning rate of the mask scores.")
    parser.add_argument("--use_fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--opt_level", type=str, default="O2", help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html",)
    
    args = parser.parse_args()
    return args


def schedule_threshold(global_step: int, total_steps: int, warmup_steps: int, 
                       initial_threshold: float, final_threshold: float, 
                       initial_warmup: int, final_warmup: int, final_lambda: float,):
    if global_step <= initial_warmup * warmup_steps:
        threshold = initial_threshold
    elif global_step > (total_steps - final_warmup * warmup_steps):
        threshold = final_threshold
    else:
        spars_warmup_steps = initial_warmup * warmup_steps
        spars_schedu_steps = (final_warmup + initial_warmup) * warmup_steps
        mul_coeff = 1 - (global_step - spars_warmup_steps) / (total_steps - spars_schedu_steps)
        threshold = final_threshold + (initial_threshold - final_threshold) * (mul_coeff ** 3)
    # print(f"\n'\nstep: {global_step}, thresh: {threshold}, final_th: {final_threshold}")
    regu_lambda = final_lambda * threshold / final_threshold
    return threshold, regu_lambda


def regularization(model: nn.Module, mode: str):
    regu, counter = 0, 0
    for name, param in model.named_parameters():
        if "mask_scores" in name:
            if mode == "l1":
                regu += torch.norm(torch.sigmoid(param), p=1) / param.numel()
            elif mode == "l0":
                regu += torch.sigmoid(param - 2 / 3 * np.log(0.1 / 1.1)).sum() / param.numel()
            else:
                ValueError("Don't know this mode.")
            counter += 1
    return regu / counter


def run_epoch(model, optimizer, device, config, t_total, train_ldr, args):
    global grad_step, global_step, epoch, best_f1

    model.train()
    for batch in tqdm(train_ldr, total=len(train_ldr)):
        batch = tuple(t.to(device) for t in batch)
        input_ids, token_ids, pos_ids, label_ids = batch
        threshold, regu_lambda = schedule_threshold(global_step=global_step,total_steps=t_total,warmup_steps=args.warmup_steps,
                                                    final_threshold=args.final_threshold,initial_threshold=args.initial_threshold, 
                                                    final_warmup=args.final_warmup,initial_warmup=args.initial_warmup, 
                                                    final_lambda=args.final_lambda,)

        # Global TopK
        if args.global_topk:
            threshold_mem = None
            if threshold == 1.0:
                threshold = -1e2  # Or an indefinitely low quantity
            else:
                if (threshold_mem is None) or (global_step % args.global_topk_frequency_compute == 0):
                    # Sort all the values to get the global topK
                    concat = torch.cat(
                        [param.view(-1) for name, param in model.named_parameters() if "mask_scores" in name]
                    )
                    n = concat.numel()
                    kth = max(n - (int(n * threshold) + 1), 1)
                    threshold_mem = concat.kthvalue(kth).values.item()
                    threshold = threshold_mem
                else:
                    threshold = threshold_mem
        inputs = {"input_ids": input_ids, 
                  "position_ids": pos_ids,
                  "label_ids": label_ids}

        if "masked" in args.model_type:
            inputs["threshold"] = threshold
        outputs = model(input_ids=inputs['input_ids'],
                        position_ids=inputs['position_ids'],
                        labels=inputs['label_ids'],
                        threshold=threshold)

        hf_loss, logits = outputs['loss'], outputs['logits']    
        loss, ppl = CEloss(logits, label_ids)

        # Regularization
        if args.regularization is not None:
            regu_ = regularization(model=model, mode=args.regularization)
            loss = loss + regu_lambda * regu_

        if args.local_rank != -1:
            loss = loss.mean()
        loss = loss / (args.grad_acc_steps / input_ids.size(0))
        
        if args.use_fp16:
            # print("\n\n\n backward pass")
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        global_step += 1

        if round(threshold, 4)*100 == 80 and threshold < 0.99: # < args.initial_threshold and (global_step+1) % args.grad_step == 0:
            # Evaluate model @@@@@@@@@
            print(f"Evaluating . . . thresh: {threshold}, gl_step: {global_step}")
            if args.local_rank==-1 or get_rank() == 0:
                logger.info("Evaluating model on dev set . . .")
                eval_args = {'val_data': '/home/aobaruwa/codebase/dd_data/val.txt', 'grad_step': grad_step,
                        'bm_size':1, 'do_sample': False, 'max_resp_len': 1000, 'epoch':epoch, 'device':'cuda',
                        'min_resp_len': 1, 'top_k': 0, 'top_p': 0.9, 'repetition_penalty': 1.2,
                        'temperature': 0.6, 'ngram_sz': 2, 'early_stopping': True,
                        'log_dir': '/home/aobaruwa/codebase/logs/pruning'}

                eval_f1, _, _, _, _, _, _ = eval_model(model,
                                                    SimpleNamespace(**eval_args),
                                                    args.pruning_method,
                                                    threshold)
                eval_f1 = round(eval_f1, 1)
                if eval_f1 > best_f1:
                    print(f"new best f1-{eval_f1}; old_best f1-{best_f1}")

                    logger.info("saving checkpoint ,{:4f} > {:4f} at step {}".format(eval_f1, best_f1, global_step))
                    _save_checkpoint(model, optimizer, grad_step, args)
                    best_f1 = eval_f1

        #if (global_step+1) % args.val_step == 0:
        #    print(f"Saving model at step {global_step}. . .")
        #    _save_checkpoint(model, optimizer, global_step, args)
        
        if global_step % args.grad_acc_steps == 0:
            set_lr(optimizer, global_step, args.lr_schedule,
                   args.lr, tot_steps=args.max_steps,
                   n_embd=config.n_embd)
            optimizer.step()
            optimizer.zero_grad()
            grad_step += 1

            # write to log
            if args.local_rank==-1 or get_rank() == 0:
                os.makedirs(args.log_dir, exist_ok=True)
                f = open(os.path.join(args.log_dir, 'train_log.txt'),
                            'a', buffering=1)
                print('{:4}\t{:4}\t{:.4f}\t{:.4f}\t{:.4f}\t{}'.format(epoch+1, global_step,
                                                                loss.detach().cpu().item(), ppl.detach().cpu().item(),
                                                                threshold, args.pruning_method),
                                                                file=f)


def main(args):
    global grad_step, global_step, epoch, best_f1
    set_seed(args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, distributed training: %s",
        args.local_rank,
        bool(args.local_rank != -1),
    )

    # Set seed
    set_seed(args)
    
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.model_type = args.model_type.lower()
    config_class, model_class, _ = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else 'gpt2-medium',
                                          pruning_method=args.pruning_method, mask_init=args.mask_init,
                                          mask_scale=args.mask_scale,)
    # tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_type,
    #                                            cache_dir=args.cache_dir if args.cache_dir else None,)
    logger.info("Loading {args.model_type} model . . .")
    model = model_class.from_pretrained(args.model_path, from_tf=False, config=config)
    # ckpt = torch.load(args.model_path)
    # model= model_class(config=config)
    # print(f"\n\nModel param names- {[n for n,p in model.named_parameters()]}")# if 'mask_scores' in n]}")
    model.to(args.device)
    optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters()
                if "mask_score" in n and p.requires_grad],
     'lr': args.mask_scores_learning_rate},
    {'params': [p for n, p in model.named_parameters() 
                if "mask_score" not in n and p.requires_grad and 
                not any(nd in n for nd in ["bias", "ln.weight"])],
     'lr': args.lr,
     'weight_decay': args.weight_decay},
    {'params': [p for n, p in model.named_parameters() 
                if "mask_score" not in n and p.requires_grad and 
                any(nd in n for nd in ["bias", "ln.weight"])],
    'lr': args.lr,
    'weight_decay': 0.0,},
    ]
    optimizer = Adam(optimizer_grouped_parameters, args.lr,
                     max_grad_norm=1.0)
    if args.use_fp16:
        model, optimizer = amp.initialize(model,
                                          optimizer,
                                          opt_level=args.opt_level,
                                          loss_scale='dynamic',
                                          min_loss_scale=1.0,
                                          max_loss_scale=2.**16)
    # multi-gpu training
    if args.local_rank != -1:
        logger.info("Use distributed training -- True . . .")
        torch.distributed.init_process_group(backend='nccl')
        model = nn.DataParallel(model)

    # Load data
    train_ldr = Loader(args.train_file, args.per_gpu_train_batch_size,
                        args.max_seq_len, shuffle=True) if args.local_rank == -1 \
                        else DistLoader(get_rank(), get_world_size(), args.train_file,
                                        args.per_gpu_train_batch_size, args.max_seq_len)
    # fix this block
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_ldr) // args.grad_acc_steps) + 1
    else:
        t_total = len(train_ldr) // args.grad_acc_steps * args.num_train_epochs

    logger.info(f"T_total: {t_total} n_epochs: {args.num_train_epochs}")

    if args.resume:
        _load_checkpoint(args.ckpt_file, model, optimizer, args)

    logging.info("\nTraining . . .")
    if args.resume is False:
        if args.local_rank==-1 or get_rank() == 0:
            f = open(os.path.join(args.log_dir, 'train_log.txt'),
                            'a', buffering=1)
            print('epoch,step, loss,\t\tppl,\t\t\tthreshold,  method', file=f)
    epoch = 0
    for _ in range(args.num_train_epochs):

        run_epoch(model, optimizer, args.device, config, t_total, train_ldr, args)
        
        epoch += 1
    # Regularization
    if args.regularization == "null":
        args.regularization = None

    return 


if __name__ == '__main__':
    args = setup_args()
    main(args)
