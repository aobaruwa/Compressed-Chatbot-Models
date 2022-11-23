python dataset.py --corpus_dir $DATA_DIR \
                  --chunk_size 2048\
                  --max_seq_len 256

python train.py --train_data $DB_FILE \
                --val_data $VAL_LOG_FILE \
                --model_type 'gpt2-medium' \
                --batch_size 8 \
                --epochs 100 \
                --grad_acc_steps 4 \
                --lr 1e-5 \
                --lm_coef 1.0 \
                --lr_schedule 'noam' \
                --local_rank -1 \
                --resume False \
                --ckpt_file $CKPT_FILE \
                --output_dir $MODEL_DIR \
                --log_dir $LOG_DIR \
                --use_fp16 False 

python src/distillation/scripts/extract.py --model_type 'gpt2' \
                                   --model_name $MODEL_DIR \
                                   --dump_checkpoint $STUDENT_CKPT_FILE \
                                   --vocab_transform 
                                   
python counts_parameters.py --pruning_method magnitude \
                            --threshold 0.90 \
                            --serialization_file $CKPT_FILE

python -m torch.distributed.launch fine_prune_gpt2.py --output_dir $OUTPUT_DIR \
                          --log_dir $PRUNING_DIR \
                          --train_file $DATA_DB_FILE \
                          --model_type masked_gpt2 \
                          --model_path /home/aobaruwa/codebase/model/medium_ft.pkl \
                          --ckpt_file /home/aobaruwa/codebase/model/medium_ft.pkl \
                          --per_gpu_train_batch_size 4 \
                          --grad_acc_steps 8  \
                          --max_steps 3000 \
                          --warmup_steps 1000 \
                          --initial_warmup 1 \
                          --final_warmup 2 \
                          --lr 5e-5 \
                          --local_rank 0 \
                          --initial_threshold 1.0 \
                          --final_threshold 0.9 \
                          --pruning_method magnitude \
                          --opt_level O1 \
                          --use_fp16

python fine_prune_gpt2.py --output_dir /home/aobaruwa/codebase/Pruning/out \
                          --log_dir /home/aobaruwa/codebase/logs/pruning \
                          --train_file /home/aobaruwa/codebase/dd_data/train.256len.db \
                          --eval_file /home/aobaruwa/codebase/dd_data/val.txt \
                          --model_type masked_gpt2 \
                          --model_path /home/aobaruwa/codebase/model/medium_ft.pkl \
                          --ckpt_file /home/aobaruwa/codebase/model/medium_ft.pkl \
                          --per_gpu_train_batch_size 4 \
                          --grad_acc_steps 8  \
                          --val_step 200 \
                          --max_steps 3000 \
                          --warmup_steps 200 \
                          --initial_warmup 1 \
                          --final_warmup 2 \
                          --lr 5e-5 \
                          --local_rank -1 \
                          --initial_threshold 1.0 \
                          --final_threshold 0.90 \
                          --pruning_method magnitude \
                          --opt_level O2 \
                          --use_fp16

Movt. Pruning
python fine_prune_gpt2.py --model_type masked_gpt2 \
                          --model_name_or_path gpt2-medium \
                          --per_gpu_train_batch_size 16 \
                          --warmup_steps 5400 \
                          --num_train_epochs 3 \
                          --lr 3e-5 
                          --mask_scores_learning_rate 1e-2 \
                          --initial_threshold 1 
                          --final_threshold 0.15 \
                          --initial_warmup 1 
                          --final_warmup 2 \
                          --pruning_method topK 
                          --mask_init constant 
                          --mask_scale 0.0
    

python src/distillation/scripts/token_counts.py --dataloader_path $TRAIN_DB_FILE \
                                                --vocab_size 50257

python train_comp.py --student_type gpt2 \
                     --student_config training_configs/distilgpt2.json \
                     --student_pretrained_weight /home/femi/compression/my_diacomp_dailyDialog/src/distillation/out/gpt2_student_seed.pth \
                     --train_batch_size 8 --val_batch_size 8 \
                     --teacher_type gpt2 --teacher_name gpt2 \
                     --teacher_pretrained_weights /home/femi/my_diacomp_dailyDialog/models/small \
                     --alpha_ce 5.0 --alpha_mlm 2.0 --alpha_cos 1.0 --alpha_clm 0.0 \
                     --freeze_pos_embs\
                     --dump_path ./out \
                     --token_counts ./out/counts.pkl \
                     --n_epochs 10 \
                     --n_gpu 0\
                     --local_rank -1 \
                     --max_seq_len 256 \
                     --grad_acc_steps 8 \
                     --log_every 500 \
                     --seed 52 \
                     --resume False \
                     --max_steps 5000 \
                     --warmup_step 1000 \
                     --model_folder '../models/medium' \
                     --save_every 4\
                     --train_input_file '/home/femi/my_diacomp_dailyDialog/dataset/validation/valid.128len.db' \
                     --val_input_file '/home/femi/my_diacomp_dailyDialog/dataset/validation/valid.128len.db' \
                     --log_dir '../dialoGPT_spk12' 

