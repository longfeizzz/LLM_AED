#!/bin/bash

SCRIPT="llm_fine_tuning.py"

BASE="../LLM_AED/evaluation"

OUTPUT_BASE="../LLM_AED/fine-tuning/processed_data/llm_fine_tuning"

declare -A MODEL_PATH
MODEL_PATH["llama_8b"]="llama_8b_peer"
MODEL_PATH["llama_70b"]="llama_70b_peer"
MODEL_PATH["qwen_8b"]="qwen_8b_peer"
MODEL_PATH["qwen_72b"]="qwen_72b_peer"

declare -A THRESHOLD

THRESHOLD["llama_8b,one-expl"]=0.8
THRESHOLD["llama_8b,one-llm"]=0.2
THRESHOLD["llama_8b,all-llm"]=0.2

THRESHOLD["llama_70b,one-expl"]=0.9
THRESHOLD["llama_70b,one-llm"]=0.6
THRESHOLD["llama_70b,all-llm"]=0.6

THRESHOLD["qwen_8b,one-expl"]=0.7
THRESHOLD["qwen_8b,one-llm"]=0.2
THRESHOLD["qwen_8b,all-llm"]=0.2

THRESHOLD["qwen_72b,one-expl"]=0.8
THRESHOLD["qwen_72b,one-llm"]=0.7
THRESHOLD["qwen_72b,all-llm"]=0.7

mkdir -p "$OUTPUT_BASE"

for model in llama_8b llama_70b qwen_8b qwen_72b; do
  for setting in one-expl one-llm all-llm; do

    MODEL_DIR=${MODEL_PATH[$model]}
    TH=${THRESHOLD["$model,$setting"]}

    INPUT_FILE="${BASE}/${setting}/${MODEL_NAME}/threshold/with_validation_${TH}.jsonl"

    OUTPUT_DIR="${OUTPUT_BASE}/${setting}/${model}"
    mkdir -p "$OUTPUT_DIR"
 
    OUTPUT_FILE="${OUTPUT_DIR}/processed_data.jsonl"
 
    echo "Processing: $INPUT_FILE"
    echo "Output to: $OUTPUT_FILE"
    
    python "$SCRIPT" "$INPUT_FILE" "$OUTPUT_FILE"

  done
done

echo "All done."