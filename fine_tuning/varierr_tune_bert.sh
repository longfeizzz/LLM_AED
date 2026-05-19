#!/bin/bash

CUDA_VISIBLE_DEVICES=0
BNB_CUDA_VERSION=122

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
directory="$REPO_ROOT/training_data"

find "$directory" -type f | while read file; do
    folder_path=$(dirname "$file")
    echo "Processing file: $file"
    TRAIN_FILE=$file
    VALIDATION_FILE="$REPO_ROOT/dataset/dev_test/dev_cleaned.json"
    TEST_FILE="$REPO_ROOT/dataset/dev_test/test_cleaned.json"
    OUTPUT="$REPO_ROOT/output/bert_tuned"
    mkdir -p "$OUTPUT"

    python "$SCRIPT_DIR/run.py" \
      --model_name_or_path "$SCRIPT_DIR/bert_finetune" \
      --train_file "$TRAIN_FILE" \
      --validation_file "$VALIDATION_FILE" \
      --test_file "$TEST_FILE" \
      --do_train --do_eval --do_predict \
      --output_dir "$OUTPUT" \
      --local_data_name chaosnli \
      --problem_type distribution_matching \
      --per_device_train_batch_size 4 \
      --learning_rate 2e-5 \
      --weight_decay 0.0 \
      --adam_beta1 0.9 \
      --adam_beta2 0.999 \
      --adam_epsilon 1e-8 \
      --lr_scheduler_type linear \
      --warmup_ratio 0.0 \
      --num_train_epochs 5 \
      --evaluation_strategy steps \
      --save_strategy steps \
      --eval_steps 20 \
      --logging_steps 20 \
      --load_best_model_at_end \
      --metric_for_best_model eval_macro_F1 \
      --overwrite_output_dir \
      --save_total_limit 1 \
      2>&1 | tee "$OUTPUT/run.log"

done
