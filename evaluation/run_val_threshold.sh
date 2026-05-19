#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VALIDATION_ROOT="$REPO_ROOT/validation/validation_results"
EVAL_ROOT="$SCRIPT_DIR"
SCRIPT="$SCRIPT_DIR/val_threshold.py"

for mode_dir in "$VALIDATION_ROOT"/*/; do
  [ -d "$mode_dir" ] || continue
  mode=$(basename "$mode_dir")

  for model_dir in "$mode_dir"*/; do
    [ -d "$model_dir" ] || continue
    model=$(basename "$model_dir")

    SCORE_FILE="$model_dir/scores.json"
    if [ "$mode" = "all_llm" ]; then
      DATA_FILE="$REPO_ROOT/processing/generation_all.jsonl"
    else
      DATA_FILE="$REPO_ROOT/processing/${model}_generation_raw.jsonl"
    fi

    echo "==== mode: $mode | model: $model ===="

    for thr in $(seq 0.0 0.1 1.0); do
      OUTPUT_FILE="${EVAL_ROOT}/${mode}/${model}/threshold/with_validation_${thr}.jsonl"

      mkdir -p "$(dirname "$OUTPUT_FILE")"
      echo "  -> threshold $thr"

      python "$SCRIPT" \
        --score_file "$SCORE_FILE" \
        --data_file "$DATA_FILE" \
        --output_file "$OUTPUT_FILE" \
        --threshold "$thr"
    done
  done
done

echo "Done."
