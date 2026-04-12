#!/bin/bash
# run_all_validation.sh

MODELS=("llama_8b")

for model in "${MODELS[@]}"; do
  echo "==== Running model: $model ===="
  
  DATA_FILE="/Users/phoebeeeee/ongoing/LLM_AED/new_processing/${model}_generation_raw.jsonl"
  SCORE_FILE="/Users/phoebeeeee/ongoing/LLM_AED/new_processing/validation_result/one_llm/${model}_all/scores.json"
  
  echo "Data file: $DATA_FILE"
  echo "Score file: $SCORE_FILE"

  for thr in $(seq 0.0 0.1 1.0); do
    OUTPUT_FILE="/Users/phoebeeeee/ongoing/LLM_AED/new_processing/validation_result/one_llm/${model}_all/threshold_2/with_validation_${thr}.jsonl"
    mkdir -p "$(dirname "$OUTPUT_FILE")"
    echo "  -> threshold $thr"
    
    python /Users/phoebeeeee/ongoing/LLM_AED/src/val_threshold.py \
      --score_file "$SCORE_FILE" \
      --data_file "$DATA_FILE" \
      --output_file "$OUTPUT_FILE" \
      --threshold "$thr"
  done
done

echo "Done."
