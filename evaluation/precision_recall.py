import os
import sys
import json
import csv
import argparse
import pandas as pd
from ast import literal_eval
import re
from pathlib import Path
from collections import Counter

map_long = {
    "entailment": "e",
    "contradiction": "c",
    "neutral": "n",
    "e": "e",
    "c": "c",
    "n": "n",
}


def parse_list_cell(x):
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s:
        return []
    return literal_eval(s)


def evaluate_file(csv_path):
    df = pd.read_csv(csv_path)

    TP, FP, FN = 0, 0, 0
    for _, row in df.iterrows():
        llm = set(literal_eval(row["llm_validated"]))
        varierr = set(literal_eval(row["varierr_validated"]))

        TP += len(llm & varierr)
        FP += len(llm - varierr)
        FN += len(varierr - llm)

    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_folder(folder, out_csv="results_summary.csv"):
    results = []
    for fname in os.listdir(folder):
        if not fname.endswith(".csv"):
            continue
        if fname == out_csv:
            continue

        m = re.search(r"with_validation_([0-9.]+)_merged_validation", fname)
        if not m:
            print(f"[Debug] Skipping non-validation CSV: {fname}")
            continue

        threshold = m.group(1)
        fpath = os.path.join(folder, fname)
        print(f"[Info] Evaluating {fpath} with threshold {threshold}")
        metrics = evaluate_file(fpath)
        metrics["threshold"] = threshold
        results.append(metrics)

    if not results:
        print(f"[Warning] No CSV files found in {folder}")

    df_results = pd.DataFrame(results)
    out_path = os.path.join(folder, out_csv)
    df_results.to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")
    return df_results


def write_merged_errors_batch(model_dir, varierr_json, out_dir=None, suffix="_merged_validation.csv"):
    model_dir = Path(model_dir)
    varierr_json = Path(varierr_json)
    out_dir = Path(out_dir) if out_dir is not None else model_dir.parent / "validated_overlap"
    out_dir.mkdir(parents=True, exist_ok=True)

    abb_dict = {"entailment": "e", "contradiction": "c", "neutral": "n"}

    with varierr_json.open("r", encoding="utf-8") as f:
        data_b = {json.loads(line)["id"]: json.loads(line) for line in f}

    written = 0
    for model_jsonl in sorted(model_dir.glob("*.jsonl")):
        out_csv = out_dir / f"{model_jsonl.stem}{suffix}"

        with model_jsonl.open("r", encoding="utf-8") as f:
            data_a = {json.loads(line)["id"]: json.loads(line) for line in f}

        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["id", "llm_validated", "varierr_validated", "chaos_validated"]
            )
            writer.writeheader()

            for id_ in data_a:
                if id_ not in data_b:
                    print(f"[{model_jsonl.name}] ID {id_} not found in VariErr dataset.")

                raw_labels = data_b.get(id_, {}).get("label_set_round_2", [])
                mapped_labels = [abb_dict.get(lbl, lbl) for lbl in raw_labels]

                chaos_dict = data_b.get(id_, {}).get("chaosnli_labels", {})
                chaos_validated = [lbl for lbl, val in chaos_dict.items() if val >= 20]

                row = {
                    "id": id_,
                    "llm_validated": json.dumps(
                        data_a.get(id_, {}).get("label_set_round_2", []), ensure_ascii=False
                    ),
                    "varierr_validated": json.dumps(mapped_labels, ensure_ascii=False),
                    "chaos_validated": json.dumps(chaos_validated, ensure_ascii=False),
                }
                writer.writerow(row)

        print(f"Written: {out_csv}")
        written += 1

    return out_dir


def process_stats_folder(folder: Path, pattern: str = "*_merged_validation.csv"):
    files = sorted(folder.glob(pattern))
    if not files:
        print(f"No CSV matched in: {folder}")
        return

    for csv_path in files:
        try:
            process_csv(csv_path)
        except Exception as e:
            print(f"[ERROR] {csv_path.name}: {e}")


def process_csv(path: Path):
    df = pd.read_csv(path)

    overlap_23 = Counter()
    overlap_24 = Counter()
    total_col2 = Counter()
    total_col3 = Counter()
    total_col4 = Counter()

    for items2, items3, items4 in zip(
        df["llm_validated"].apply(parse_list_cell),
        df["varierr_validated"].apply(parse_list_cell),
        df["chaos_validated"].apply(parse_list_cell),
    ):
        set2 = {map_long.get(it) for it in items2 if map_long.get(it) in ("e", "n", "c")}
        set3 = {map_long.get(it) for it in items3 if map_long.get(it) in ("e", "n", "c")}
        set4 = {map_long.get(it) for it in items4 if map_long.get(it) in ("e", "n", "c")}

        for lab in (set2 & set3):
            overlap_23[lab] += 1
        for lab in (set2 & set4):
            overlap_24[lab] += 1

        for lab in set2:
            total_col2[lab] += 1
        for lab in set3:
            total_col3[lab] += 1
        for lab in set4:
            total_col4[lab] += 1

    return {
        "overlap_23": overlap_23,
        "overlap_24": overlap_24,
        "total_col2": total_col2,
        "total_col3": total_col3,
        "total_col4": total_col4,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("threshold_dir",)
    parser.add_argument("--varierr_json", default=str(Path(__file__).resolve().parent.parent / "dataset" / "varierr.json"))
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--skip_create_csv", action="store_true")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    threshold_dir = Path(args.threshold_dir)
    if not threshold_dir.is_dir():
        print(f"Error: threshold directory {threshold_dir} does not exist")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else threshold_dir.parent / "validated_overlap"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_create_csv:
        write_merged_errors_batch(threshold_dir, args.varierr_json, out_dir=output_dir)

    if not args.stats and not args.evaluate:
        args.evaluate = True

    if args.stats:
        process_stats_folder(output_dir)

    if args.evaluate:
        df = evaluate_folder(output_dir)

