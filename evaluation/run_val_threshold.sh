#!/bin/bash

VALIDATION_ROOT="LLM_AED/validation/validation_results"
EVAL_ROOT="LLM_AED/evaluation"

for mode_dir in "$VALIDATION_ROOT"/*/; do
  mode=$(basename "$mode_dir")

  for model_dir in "$mode_dir"*/; do
    model=$(basename "$model_dir")

    SCORE_FILE="$model_dir/scores.json"
    DATA_FILE="../processing/${model}_generation_raw.jsonl"

    echo "==== mode: $mode | model: $model ===="

    for thr in $(seq 0.0 0.1 1.0); do
      OUTPUT_FILE="${EVAL_ROOT}/${mode}/${model}/threshold/with_validation_${thr}.jsonl"

      mkdir -p "$(dirname "$OUTPUT_FILE")"
      echo "  -> threshold $thr"

      python val_threshold.py \
        --score_file "$SCORE_FILE" \
        --data_file "$DATA_FILE" \
        --output_file "$OUTPUT_FILE" \
        --threshold "$thr"
    done
  done
done

echo "Done."