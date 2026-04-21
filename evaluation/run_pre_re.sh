#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$SCRIPT_DIR"
SCRIPT_PATH="$SCRIPT_DIR/precision_recall.py"

# Loop through all modes (all-llm, one-llm, etc.)
for mode_dir in "$BASE_DIR"/*; do
    if [ ! -d "$mode_dir" ]; then
        continue
    fi
    
    mode=$(basename "$mode_dir")
    echo "========================================"
    echo "Processing mode: $mode"
    echo "========================================"
    
    # Loop through all models in the mode directory
    for model_dir in "$mode_dir"/*; do
        if [ ! -d "$model_dir" ]; then
            continue
        fi
        
        model=$(basename "$model_dir")
        validation_dir="$model_dir/validated_overlap"
    
        echo "[Info] Processing $mode/$model"
        
        # Run the Python script for this directory
        python3 "$SCRIPT_PATH" "$validation_dir"
        
        if [ $? -eq 0 ]; then
            echo "Results saved to: $validation_dir/results_summary.csv"
        else
            echo "[Error] Failed to evaluate $mode/$model"
        fi
        
        echo ""
    done
done

echo "========================================"
echo "All evaluations completed!"
echo "========================================"
