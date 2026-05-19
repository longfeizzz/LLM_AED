#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=${1:-0}
export CUDA_VISIBLE_DEVICES
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_TYPE="llama"
MODELS=(
  "meta-llama/Llama-3.1-8B-Instruct"
  "meta-llama/Llama-3.3-70B-Instruct"
)

echo "============================================================"
echo "1. one-expl"
echo "============================================================"
for model in "${MODELS[@]}"; do
  echo ">> $model"
  python "$SCRIPT_DIR/one_expl.py" \
    --model_type "$MODEL_TYPE" \
    --model_name_or_path "$model"
done

echo "============================================================"
echo "2. one-llm"
echo "============================================================"
for model in "${MODELS[@]}"; do
  echo ">> $model"
  python "$SCRIPT_DIR/one_llm.py" \
    --model_type "$MODEL_TYPE" \
    --model_name_or_path "$model"
done

echo "============================================================"
echo "3. all-llm"
echo "============================================================"
for model in "${MODELS[@]}"; do
  echo ">> $model"
  python "$SCRIPT_DIR/all_llm.py" \
    --model_type "$MODEL_TYPE" \
    --model_name_or_path "$model" \
    --input_path "$REPO_ROOT/processing/generation_all.jsonl"
done

echo "All done."
