#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCRIPT="$SCRIPT_DIR/similarity_within_llm.py"

# 1. BEFORE (4 files)
BEFORE_BASE="$REPO_ROOT/processing"

MODELS=("Llama-3.1-8B" "Llama-3.3-70B" "Qwen2.5-7B" "Qwen2.5-72B")

for model in "${MODELS[@]}"; do
    FILE="$BEFORE_BASE/${model}_generation_raw.jsonl"
    echo "Running BEFORE: $FILE"
    python "$SCRIPT" "$FILE"
done

# 2. AFTER (12 files)

AFTER_BASE="$SCRIPT_DIR"

declare -A THRESHOLD

THRESHOLD["Llama-3.1-8B,one_expl"]=0.8
THRESHOLD["Llama-3.1-8B,one_llm"]=0.2
THRESHOLD["Llama-3.1-8B,all_llm"]=0.2

THRESHOLD["Llama-3.3-70B,one_expl"]=0.9
THRESHOLD["Llama-3.3-70B,one_llm"]=0.6
THRESHOLD["Llama-3.3-70B,all_llm"]=0.6

THRESHOLD["Qwen2.5-7B,one_expl"]=0.7
THRESHOLD["Qwen2.5-7B,one_llm"]=0.2
THRESHOLD["Qwen2.5-7B,all_llm"]=0.2

THRESHOLD["Qwen2.5-72B,one_expl"]=0.8
THRESHOLD["Qwen2.5-72B,one_llm"]=0.7
THRESHOLD["Qwen2.5-72B,all_llm"]=0.7

for model in "${MODELS[@]}"; do
  for setting in one_expl one_llm all_llm; do

    MODEL_NAME=$model
    TH=${THRESHOLD["$model,$setting"]}

    FILE="${AFTER_BASE}/${setting}/${MODEL_NAME}/threshold/with_validation_${TH}.jsonl"

    echo "Running AFTER: $FILE"
    python "$SCRIPT" "$FILE"

  done
done
