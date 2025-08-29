import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import average_precision_score, precision_score, recall_score
import argparse

def recall_at_k(y_true, y_score, k):
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    idx = np.argsort(y_score)[::-1][:k]
    y_pred = np.zeros_like(y_true)
    y_pred[idx] = 1
    return recall_score(y_true, y_pred)

def precision_at_k(y_true, y_score, k):
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    idx = np.argsort(y_score)[::-1][:k]
    return precision_score(y_true[idx], np.ones(k))

def compute_metrics(id_to_score, id_to_gt):
    ids = [i for i in id_to_gt if i in id_to_score]
    if not ids:
        print("No matching IDs found")
        return {"ap": 0.0, "p@100": 0.0, "r@100": 0.0}
    y_true = np.array([id_to_gt[i] for i in ids])
    y_score = np.array([id_to_score[i] for i in ids])
    return {
        "ap": average_precision_score(y_true, y_score),
        "p@100": precision_at_k(y_true, y_score, 100),
        "r@100": recall_at_k(y_true, y_score, 100)
    }

def prettify_results(df: pd.DataFrame) -> str:
    df *= 100
    header = "Method\tAP\tP@100\tR@100"
    row = df.round(1).iloc[0]
    values = "\t".join([f"{v:.1f}" for v in row])
    return f"{header}\n{df.index[0]}\t{values}"

def get_ground_truth(instances):
    id_to_gt = {}
    for ex in instances:
        id_ = ex.get("id")
        if not id_:
            print("Warning: missing 'id' in instance:", ex)
            continue
        round1 = ex.get("label_count_round_1", {})
        errors = ex.get("error_labels", [])
        for label, count in round1.items():
            if count is None:
                continue
            label_id = f"{id_}-{label[0]}"
            id_to_gt[label_id] = label in errors
    return id_to_gt

def build_score_table(varierr_path, score_path):
    # Load ground truth
    with open(varierr_path) as f:
        data = [json.loads(line) for line in f]

    with open(score_path) as f:
        id_to_score = json.load(f)


    id_to_gt = get_ground_truth(data)
    id_to_score = {k: v for k, v in id_to_score.items() if k in id_to_gt}
    metrics = compute_metrics(id_to_score, id_to_gt)

    df = pd.DataFrame([metrics], index=[score_path.stem])
    return prettify_results(df)

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM error detection using AP, P@100, and R@100.")
    parser.add_argument("--varierr", required=True, type=Path, help="Path to varierr.json")
    parser.add_argument("--score", required=True, type=Path, help="Path to score file (JSON)")

    args = parser.parse_args()

    result = build_score_table(
        varierr_path=args.varierr,
        score_path=args.score
    )

    print("\nEvaluation Result:")
    print(result)

if __name__ == "__main__":
    main()
