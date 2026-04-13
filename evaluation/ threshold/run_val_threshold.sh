#!/bin/bash

VALIDATION_ROOT="../validation/validation_results"

for mode_dir in "$VALIDATION_ROOT"/*/; do
  for model_dir in "$mode_dir"*/; do
    SCORE_FILE="$model_dir/scores.json"

    if [ ! -f "$SCORE_FILE" ]; then
      continue
    fi

    mode=$(basename "$mode_dir")
    model=$(basename "$model_dir")

    DATA_FILE="../processing/${model}_generation_raw.jsonl"

    if [ ! -f "$DATA_FILE" ]; then
      echo "Missing data file: $DATA_FILE, skipping $model"
      continue
    fi

    echo "==== mode: $mode | model: $model ===="

    for thr in $(seq 0.0 0.1 1.0); do
      OUTPUT_FILE="$model_dir/threshold/with_validation_${thr}.jsonl"
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