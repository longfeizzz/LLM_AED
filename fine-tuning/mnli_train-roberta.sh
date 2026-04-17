export TASK_NAME=mnli

python run.py \
  --model_name_or_path roberta-base \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --adam_beta1 0.9 \
  --adam_beta2 0.999 \
  --adam_epsilon 1e-8 \
  --weight_decay 0.0 \
  --lr_scheduler_type linear \
  --warmup_ratio 0.0 \
  --output_dir /root/MJD-fine-tuning/roberta_finetune \
  --load_best_model_at_end \
  --evaluation_strategy steps \
  --save_strategy steps \
  --save_total_limit 1 \
  --save_steps 500 \
  --eval_steps 250 \
  --seed 42 \
  --logging_steps 100 \
  --logging_dir /root/MJD-fine-tuning/logs \
  --metric_for_best_model eval_accuracy \
  --fp16 --overwrite_output_dir
