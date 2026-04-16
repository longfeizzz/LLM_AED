#!/bin/bash
set -euo pipefail

MODE="$1"
MODEL="$2"

COMPARE_PY="kld_jsd.py"

EVAL_ROOT="../evaluation"
MODEL_DIR="${EVAL_ROOT}/${MODE}/${MODEL}"
THR_DIR="${MODEL_DIR}/threshold"
OUT_DIR="${MODEL_DIR}/kld_jsd"
mkdir -p "$OUT_DIR"

echo "Mode:      $MODE"
echo "Model:     $MODEL"
echo "Input dir: $THR_DIR"
echo "Out dir:   $OUT_DIR"

for thr in $(seq 0 0.1 1); do
  MODEL_JSONL="${THR_DIR}/with_validation_${thr}.jsonl"
  if [[ ! -f "$MODEL_JSONL" ]]; then
    echo "Missing: $MODEL_JSONL — skipping"
    continue
  fi

  echo ">> threshold=$thr | round=1 (before)"
  python "$COMPARE_PY" \
    --model_jsonl "$MODEL_JSONL" \
    --out_dir "$OUT_DIR" \
    --prefix "${MODEL}_before_${thr}" \
    --round_idx 1

  echo ">> threshold=$thr | round=2 (after)"
  python "$COMPARE_PY" \
    --model_jsonl "$MODEL_JSONL" \
    --out_dir "$OUT_DIR" \
    --prefix "${MODEL}_after_${thr}" \
    --round_idx 2
done

echo "Done."