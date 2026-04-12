#!/usr/bin/env bash
set -euo pipefail

PAIRS=(
  "Qwen/Qwen2.5-7B-Instruct|/mounts/data/proj/zlongfei/ZLF/generation_all.jsonl|validation/qwen_7b_original_peer"
  "Qwen/Qwen2.5-72B-Instruct|/mounts/data/proj/zlongfei/ZLF/generation_all.jsonl|validation/qwen_72b_original_peer"
)

PY=/mounts/data/proj/zlongfei/ZLF/llm_validation_original_qwen_peer.py

export CUDA_VISIBLE_DEVICES="2,3"

for entry in "${PAIRS[@]}"; do
  IFS='|' read -r model input_path output_dir <<<"$entry"

  echo "============================================================"
  echo "[model name] model=$model"
  echo "[input path] input=$input_path"
  echo "[output dir] output=$output_dir"
  echo "============================================================"

  mkdir -p "$output_dir"

  python3 "$PY" \
    --model_name "$model" \
    --input_path "$input_path" \
    --output_dir "$output_dir"
done

echo "All done."
