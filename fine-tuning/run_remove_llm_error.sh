#!/bin/bash

SCRIPT="remove_llm_error.py"

EVAL_BASE="../evaluation"

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

for model in Llama-3.1-8B Llama-3.3-70B Qwen2.5-7B Qwen2.5-72B; do
  for setting in one_expl one_llm all_llm; do

    MODEL_NAME=$model
    TH=${THRESHOLD["$model,$setting"]}

    MODEL_FILE="${EVAL_BASE}/${setting}/${MODEL_NAME}/threshold/with_validation_${TH}.jsonl"

    OUT_DIR="../fine-tuning/processed_data/without_llm_error/${setting}/${MODEL_NAME}"
    
    echo "Processing: $MODEL_FILE"
    python $SCRIPT "$MODEL_FILE" "$OUT_DIR"

  done
done

echo "All done."