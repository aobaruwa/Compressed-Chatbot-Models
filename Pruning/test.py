from modeling import MaskedGPT2Config, MaskedGPT2LMHeadModel
from transformers import AutoTokenizer
import torch
import sys

def main():
    config = MaskedGPT2Config(pruning_method='topK')
    model = MaskedGPT2LMHeadModel(config=config, threshold=0.80)
    ckpt=torch.load('/home/aobaruwa/codebase/Pruning/Results/mvt/GPT2-finetune-step-1800-0.9.bin')# /Results/pruned_model_0.9.bin') #/out/GPT2-petrain-step-1800.pkl')
    # ckpt = torch.load('/home/aobaruwa/codebase/Pruning/out/pruned_model.bin')
    #for n, tensor in ckpt['model_state'].items().named_parameters():
    #    print(n, tensor, '\n\n\n')
    #sys.exit()
    model.load_state_dict(ckpt)
    model.to('cuda')
    toker = AutoTokenizer.from_pretrained('gpt2-medium')
    while True:
        input_ids = input('Human: ') #"hello bot, what's your name?<|endoftext|>"
        input_ids += '<|endoftext|>'
        input_ids = toker.encode(input_ids, return_tensors='pt')

        output_ids = model.generate(input_ids.to('cuda'), 
                                    do_sample=True, 
                                    num_beams=4, top_p=0.8, 
                                    temperature=0.7, 
                                    repetition_penalty=1.2,
                                    threshold=0.900,
                                    no_repeat_ngram_size=2)
        resp = toker.decode(output_ids[0])
        print('Bot: ', resp.strip().split('<|endoftext|>')[-2])


if __name__=='__main__':
    main()