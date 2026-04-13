#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=$1
export CUDA_VISIBLE_DEVICES

MODEL_TYPE="qwen"
MODELS=(
  "Qwen/Qwen2.5-7B-Instruct"
  "Qwen/Qwen2.5-72B-Instruct"
)

echo "============================================================"
echo "1. one-expl"
echo "============================================================"
for model in "${MODELS[@]}"; do
  echo ">> $model"
  python one-expl.py \
    --model_type "$MODEL_TYPE" \
    --model_name_or_path "$model"
done

echo "============================================================"
echo "2. one-llm"
echo "============================================================"
for model in "${MODELS[@]}"; do
  echo ">> $model"
  python one-llm.py \
    --model_type "$MODEL_TYPE" \
    --model_name_or_path "$model"
done

echo "============================================================"
echo "3. all-llm"
echo "============================================================"
for model in "${MODELS[@]}"; do
  echo ">> $model"
  python all-llm.py \
    --model_type "$MODEL_TYPE" \
    --model_name_or_path "$model" \
    --input_path ../processing/generation_all.jsonl
done

echo "All done."