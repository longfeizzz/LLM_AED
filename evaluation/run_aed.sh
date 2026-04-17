#!/bin/bash

BASE_DIR="../LLM_AED/validation/validation_results"
SCRIPT="evaluate.py"

for mode_dir in "$BASE_DIR"/*; do
  [ -d "$mode_dir" ] || continue
  mode=$(basename "$mode_dir")

  for model_dir in "$mode_dir"/*; do
    [ -d "$model_dir" ] || continue
    model=$(basename "$model_dir")

    score_file="$model_dir/scores.json"

    if [ -f "$score_file" ]; then
      echo "Running: mode=$mode | model=$model"
      python "$SCRIPT" --score "$score_file"
      echo "-----------------------------------"
    else
      echo "Missing: $score_file"
    fi
  done
done