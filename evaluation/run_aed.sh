#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$SCRIPT_DIR/../validation/validation_results"
SCRIPT_PATH="$SCRIPT_DIR/evaluation_protocol.py"
OUT_ROOT="$SCRIPT_DIR"

for mode_dir in "$BASE_DIR"/*; do
  [ -d "$mode_dir" ] || continue
  mode=$(basename "$mode_dir")

  for model_dir in "$mode_dir"/*; do
    [ -d "$model_dir" ] || continue
    model=$(basename "$model_dir")

    score_file="$model_dir/scores.json"

    if [ -f "$score_file" ]; then
      out_dir="$OUT_ROOT/$mode/$model"
      out_file="$out_dir/aed_results.txt"
      mkdir -p "$out_dir"

      echo "Running: mode=$mode | model=$model"
      python "$SCRIPT_PATH" --score "$score_file" | tee "$out_file"
      echo "Saved AED results to: $out_file"
      echo "-----------------------------------"
    else
      echo "Missing: $score_file"
    fi
  done
done
