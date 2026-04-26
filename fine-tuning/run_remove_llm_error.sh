#!/bin/bash

SCRIPT="remove_llm_error.py"

EVAL_BASE="LLM_AED/evaluation"

declare -A MODEL_PATH
MODEL_PATH["llama_8b"]="Llama-3.1-8B"
MODEL_PATH["llama_70b"]="Llama-3.1-70B"
MODEL_PATH["qwen_8b"]="Qwen2-7B"
MODEL_PATH["qwen_72b"]="Qwen2-72B"

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

for model in llama_8b llama_70b qwen_8b qwen_72b; do
  for setting in one-expl one-llm all-llm; do

    MODEL_NAME=${MODEL_PATH[$model]}
    TH=${THRESHOLD["$model,$setting"]}

    MODEL_FILE="${EVAL_BASE}/${setting}/${MODEL_NAME}/threshold/with_validation_${TH}.jsonl"

    OUT_DIR="${EVAL_BASE}/${setting}/${MODEL_NAME}/without_llm_error"

    echo "Processing: $MODEL_FILE"
    python $SCRIPT "$MODEL_FILE" "$OUT_DIR"

  done
done

echo "All done."