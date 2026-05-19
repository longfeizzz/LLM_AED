#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCRIPT="$SCRIPT_DIR/llm_fine_tuning.py"

BASE="$REPO_ROOT/evaluation"

OUTPUT_BASE="$SCRIPT_DIR/processed_data/llm_cleaned"

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

mkdir -p "$OUTPUT_BASE"

for model in Llama-3.1-8B Llama-3.3-70B Qwen2.5-7B Qwen2.5-72B; do
  for setting in one_expl one_llm all_llm; do

    MODEL_DIR=$model
    TH=${THRESHOLD["$model,$setting"]}

    INPUT_FILE="${BASE}/${setting}/${MODEL_DIR}/threshold/with_validation_${TH}.jsonl"

    OUTPUT_DIR="${OUTPUT_BASE}/${setting}/${model}"
    mkdir -p "$OUTPUT_DIR"
 
    OUTPUT_FILE="${OUTPUT_DIR}/processed_data.jsonl"
 
    echo "Processing: $INPUT_FILE"
    echo "Output to: $OUTPUT_FILE"
    
    python "$SCRIPT" "$INPUT_FILE" "$OUTPUT_FILE"

  done
done

echo "All done."
