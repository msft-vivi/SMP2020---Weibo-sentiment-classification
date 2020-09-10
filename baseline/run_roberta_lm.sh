#!/bin/bash
python roberta_lm_fineturning.py \
	--pre_train_path pretrain_model/roberta_large_added_tokens \
	--output_dir pretrain_model/clean_usual_roberta_wwm_ext_large_lm_2 \
  --data_dir weibo_data/weibo-all.txt \
	--max_seq_length 140 \
	--train_batch_size 16 \
  	--gradient_accumulation_steps 8 \
  	--num_train_epochs 2 \
	--learning_rate 2e-5 \
  	--warmup_rate 0 \
	--log_dir clean_usual_roberta_wwm_ext_large_lm_fineturning.log\
  	--do_train

python roberta_lm_fineturning.py \
    --pre_train_path pretrain_model/clean_usual_roberta_wwm_ext_large_lm_2 \
    --output_dir pretrain_model/clean_usual_roberta_wwm_ext_large_lm_4 \
  --data_dir weibo_data/weibo-all.txt \
    --max_seq_length 140 \
    --train_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 2 \
    --learning_rate 2e-5 \
    --warmup_rate 0 \
    --log_dir clean_usual_roberta_wwm_ext_large_lm_fineturning.log\
    --do_train

python roberta_lm_fineturning.py \
    --pre_train_path pretrain_model/clean_usual_roberta_wwm_ext_large_lm_4 \
    --output_dir pretrain_model/clean_usual_roberta_wwm_ext_large_lm_6 \
  --data_dir weibo_data/weibo-all.txt \
    --max_seq_length 140 \
    --train_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 2 \
    --learning_rate 2e-5 \
    --warmup_rate 0 \
    --log_dir clean_usual_roberta_wwm_ext_large_lm_fineturning.log\
    --do_train

python roberta_lm_fineturning.py \
    --pre_train_path pretrain_model/clean_usual_roberta_wwm_ext_large_lm_6 \
    --output_dir pretrain_model/clean_usual_roberta_wwm_ext_large_lm_8 \
  --data_dir weibo_data/weibo-all.txt \
    --max_seq_length 140 \
    --train_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 2 \
    --learning_rate 2e-5 \
    --warmup_rate 0 \
    --log_dir clean_usual_roberta_wwm_ext_large_lm_fineturning.log\
    --do_train

python roberta_lm_fineturning.py \
    --pre_train_path pretrain_model/clean_usual_roberta_wwm_ext_large_lm_8 \
    --output_dir pretrain_model/clean_usual_roberta_wwm_ext_large_lm_10 \
  --data_dir weibo_data/weibo-all.txt \
    --max_seq_length 140 \
    --train_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 2 \
    --learning_rate 2e-5 \
    --warmup_rate 0 \
    --log_dir clean_usual_roberta_wwm_ext_large_lm_fineturning.log\
    --do_train

#python roberta_lm_fineturning.py \
#	--pre_train_path 'roberta_wwm_ext_large' \
#	--output_dir 'pretrain_roberta' \
#  	--data_dir 'weibo_data' \
#  	--max_seq_length 140 \
#  	--train_batch_size 6 \
#  	--gradient_accumulation_steps 4 \
#  	--num_train_epochs 10 \
# 	--learning_rate 2e-5 \
#  	--warmup_rate 0 \
#  	--log_dir '训练过程的日志' \
#  	--do_train

