#!/bin/bash
# run_all_validation.sh

MODELS=("llama_8b" "llama_70b" "qwen_7b" "qwen_72b")

for model in "${MODELS[@]}"; do
  echo "==== Running model: $model ===="
  
  DATA_FILE="/Users/phoebeeeee/ongoing/LLM_AED/new_processing/${model}_generation_raw.jsonl"
  SCORE_FILE="/Users/phoebeeeee/ongoing/LLM_AED/new_processing/validation_result/original/${model}_original/scores.json"
  
  echo "Data file: $DATA_FILE"
  echo "Score file: $SCORE_FILE"

  for thr in $(seq 0.1 0.1 0.9); do
    OUTPUT_FILE="/Users/phoebeeeee/ongoing/LLM_AED/new_processing/validation_result/original/${model}_original/with_validation_${thr}.jsonl"
    echo "  -> threshold $thr"
    
    python /Users/phoebeeeee/ongoing/LLM_AED/src/val_threshold.py \
      --score_file "$SCORE_FILE" \
      --data_file "$DATA_FILE" \
      --output_file "$OUTPUT_FILE" \
      --threshold "$thr"
  done
done

echo "Done."
