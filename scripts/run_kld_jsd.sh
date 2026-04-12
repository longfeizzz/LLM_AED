#!/bin/bash
# run_kld_jsd.sh

set -euo pipefail

MODEL="${1:-qwen_72b}"

# SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMPARE_PY="/Users/phoebeeeee/ongoing/LLM_AED/src/kld_jsd.py"
PLOT_PY="/Users/phoebeeeee/ongoing/LLM_AED/src/plot.py"

BASE_DIR="/Users/phoebeeeee/ongoing/LLM_AED/new_processing/validation_result/one_llm"
MODEL_DIR="${BASE_DIR}/${MODEL}_all/threshold_2"
OUT_DIR="${MODEL_DIR}/kld_jsd_2"
mkdir -p "$OUT_DIR"

echo "Model: ${MODEL}"
echo "MODEL_DIR: ${MODEL_DIR}"
echo "OUT_DIR  : ${OUT_DIR}"

for thr in $(seq 0 0.1 1); do
  MODEL_JSONL="${MODEL_DIR}/with_validation_${thr}.jsonl"
  if [[ ! -f "$MODEL_JSONL" ]]; then
    echo "Missing: $MODEL_JSONL — skip this threshold."
    continue
  fi

  # round 1 = before
#   echo "${MODEL} | thr=${thr} | round=1 (before)"
  python "$COMPARE_PY" \
    --model_jsonl "$MODEL_JSONL" \
    --out_dir "$OUT_DIR" \
    --prefix "${MODEL}_all_before_${thr}" \
    --round_idx 1

  # round 2 = after
#   echo "${MODEL} | thr=${thr} | round=2 (after)"
  python "$COMPARE_PY" \
    --model_jsonl "$MODEL_JSONL" \
    --out_dir "$OUT_DIR" \
    --prefix "${MODEL}_all_after_${thr}" \
    --round_idx 2
done

# echo "Thresholds processed. Plotting lines..."
python "$PLOT_PY" --model "$MODEL" --base_dir "$BASE_DIR"

echo "Done."
