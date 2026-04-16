import os
import json
import csv
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

LABEL_ORDER = ["e", "n", "c"]  


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def normalize(counts):
    arr = np.array(counts, dtype=np.float64)
    s = arr.sum()
    return arr / s if s > 0 else np.ones_like(arr) / len(arr)


def kl_divergence(p, q, eps=1e-12):
    p = np.array(p, dtype=np.float64) + eps
    q = np.array(q, dtype=np.float64) + eps
    return float(np.sum(p * np.log2(p / q)))


def write_merged_errors(model_jsonl, varierr_json, out_csv):
    with open(model_jsonl, "r", encoding="utf-8") as f:
        data_a = {json.loads(line)["id"]: json.loads(line) for line in f}
    with open(varierr_json, "r", encoding="utf-8") as f:
        data_b = {json.loads(line)["id"]: json.loads(line) for line in f}

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "llm_error", "varierr_error"])
        writer.writeheader()
        for id_ in data_a:
            if id_ not in data_b:
                print(f"ID {id_} not found in VariErr dataset.")
            writer.writerow({
                "id": id_,
                "llm_error": json.dumps(data_a[id_].get("error", []), ensure_ascii=False),
                "varierr_error": json.dumps(data_b.get(id_, {}).get("error_labels", []), ensure_ascii=False),
            })


def load_model_counts(model_jsonl, round_idx):  # 1 or 2
    dist_map = {}
    with open(model_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            counts = data.get(f"label_count_round_{round_idx}", {})
            dist_map[data["id"]] = {
                lbl: float(counts.get(lbl, 0) or 0) for lbl in LABEL_ORDER
            }
    return dist_map


def load_human_counts_chaos(chaos_jsonl):
    dist_map = {}
    with open(chaos_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            counts = r.get("chaosnli_labels", {})
            dist_map[r["id"]] = {
                lbl: float(counts.get(lbl, 0) or 0) for lbl in LABEL_ORDER
            }
    return dist_map


def load_human_counts_varierr(varierr_json, round_idx):
    with open(varierr_json, "r") as f:
        records = [json.loads(line) for line in f]  

    dist_map = {}
    for r in records:
        uid = r["id"]
        raw_counts = r.get(f"label_count_round_{round_idx}", {})   # change here 1 or 2

        counter = {
            "e": float(raw_counts.get("entailment") or 0),
            "n": float(raw_counts.get("neutral") or 0),
            "c": float(raw_counts.get("contradiction") or 0)
        }

        dist_map[uid] = counter

    return dist_map

def load_human_counts_varierr(varierr_json, round_idx):
    label_remap = {"entailment": "e", "neutral": "n", "contradiction": "c"}
    dist_map = {}
    with open(varierr_json, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            counts = r.get(f"label_count_round_{round_idx}", {})
            dist_map[r["id"]] = {
                short: float(counts.get(full, 0) or 0)
                for full, short in label_remap.items()
            }
    return dist_map


def compare_distributions(model_counts, human_counts, out_csv, title=""):
    rows = []
    shared_ids = set(model_counts.keys()) & set(human_counts.keys())
    print(f"[Info] {title} shared IDs: {len(shared_ids)}")

    for uid in shared_ids:
        model_vec = [model_counts[uid].get(lbl, 0) for lbl in LABEL_ORDER]
        human_vec = [human_counts[uid].get(lbl, 0) for lbl in LABEL_ORDER]

        model_dist = normalize(model_vec)
        human_dist = normalize(human_vec)

        jsd = float(jensenshannon(model_dist, human_dist, base=2.0) ** 2)
        kl = kl_divergence(model_dist, human_dist)

        rows.append({
            "uid": uid,
            "js_divergence": jsd,
            "kl_divergence": kl,
            "model_distribution": model_dist.tolist(),
            "human_distribution": human_dist.tolist(),
            "model_counts": model_vec,
            "human_counts": human_vec,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return out_csv


def summarize_means(csv_paths, out_csv):
    rows = []
    for name, path in csv_paths.items():
        if not os.path.exists(path):
            print(f"Skip missing: {name} ({path})")
            continue
        df = pd.read_csv(path)
        mean_jsd = float(df["js_divergence"].mean()) if "js_divergence" in df else float("nan")
        mean_kld = float(df["kl_divergence"].mean()) if "kl_divergence" in df else float("nan")
        print(f"[Mean] {name}: JSD={mean_jsd:.4f}, KL={mean_kld:.4f}")
        rows.append({"name": name, "mean_jsd": mean_jsd, "mean_kl": mean_kld})

    if rows:
        pd.DataFrame(rows).to_csv(out_csv, index=False)
    else:
        print("No rows for summary")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_jsonl", required=True)
    ap.add_argument("--varierr_json", default="../dataset/varierr.json")
    ap.add_argument("--chaos_jsonl", default="../dataset/chaosNLI_mnli_m.jsonl")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prefix", default="model")
    ap.add_argument("--round_idx", type=int, default=2)
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    # 1. merged errors
    merged_csv = os.path.join(args.out_dir, f"{args.prefix}_merged_errors.csv")
    write_merged_errors(args.model_jsonl, args.varierr_json, merged_csv)

    # 2. JSD/KLD - ChaosNLI
    model_counts = load_model_counts(args.model_jsonl, round_idx=args.round_idx)
    chaos_counts = load_human_counts_chaos(args.chaos_jsonl)
    chaos_csv = os.path.join(args.out_dir, f"{args.prefix}_chaos_jsd_kl.csv")
    compare_distributions(model_counts, chaos_counts, chaos_csv, title="ChaosNLI")

    # 3. JSD/KL - VariErr
    vari_counts = load_human_counts_varierr(args.varierr_json, round_idx=args.round_idx)
    vari_csv = os.path.join(args.out_dir, f"{args.prefix}_varierr_jsd_kl.csv")
    compare_distributions(model_counts, vari_counts, vari_csv, title="VariErr")

    # 3) average summary
    summarize_means(
        {"ChaosNLI": chaos_csv, "VariErr": vari_csv},
        os.path.join(args.out_dir, f"{args.prefix}_summary.csv")
    )


if __name__ == "__main__":
    main()
