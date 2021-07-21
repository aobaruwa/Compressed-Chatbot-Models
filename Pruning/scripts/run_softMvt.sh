#! bin/bash
python fine_prune_gpt2.py --output_dir /home/aobaruwa/codebase/Pruning/out \
                          --log_dir /home/aobaruwa/codebase/logs/pruning \
                          --train_file /home/aobaruwa/codebase/dd_data/train.256len.db \
                          --eval_file /home/aobaruwa/codebase/dd_data/val.txt \
                          --model_type masked_gpt2 \
                          --model_path /home/aobaruwa/codebase/model/medium_ft.pkl \
                          --ckpt_file /home/aobaruwa/codebase/model/medium_ft.pkl \
                          --per_gpu_train_batch_size 4 \
                          --grad_acc_steps 8 \
                          --val_step 600 \
                          --max_steps 3000 \
                          --max_seq_len 150 \
                          --warmup_steps 100 \
                          --initial_warmup 1 \
                          --final_warmup 2 \
                          --lr 3e-5 \
                          --local_rank -1 \
                          --initial_threshold 1.0 \
                          --final_threshold 0.90 \
                          --pruning_method sigmoied_threshold \
                          --mask_init constant \
                          --mask_scale 0.0 \
                          #--opt_level O2 \
                          #--use_fp16
