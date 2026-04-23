#!/bin/bash

SCRIPT="similarity_llm_human.py"
VAR_FILE="/LLM_AED/dataset/varierr.json"

# BEFORE (4 files)

BEFORE_BASE="/LLM_AED/processing"

for model in llama_8b llama_70b qwen_8b qwen_72b; do

    FILE="${BEFORE_BASE}/${model}_generation_raw.jsonl"

    echo "Running BEFORE (LLM vs VariErr): $FILE"

    python $SCRIPT "$FILE" "$VAR_FILE"

done


# AFTER (12 files)

AFTER_BASE="/Users/phoebeeeee/ongoing/LLM_AED/evaluation"

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

    FILE="${AFTER_BASE}/${setting}/${MODEL_NAME}/threshold/with_validation_${TH}.jsonl"

    echo "Running AFTER (LLM vs VariErr): $FILE"

    python $SCRIPT "$FILE" "$VAR_FILE"

  done
done

echo "All runs completed."