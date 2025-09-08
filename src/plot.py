# -*- coding: utf-8 -*-

import os
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base_dir", default="/Users/phoebeeeee/ongoing/LLM_AED/new_processing/validation_result/original")
    args = ap.parse_args()

    model = args.model
    model_dir = os.path.join(args.base_dir, f"{model}_original")
    out_dir = os.path.join(model_dir, "kld_jsd")
    os.makedirs(out_dir, exist_ok=True)

    pat = re.compile(rf"^{re.escape(model)}_original_(before|after)_([0-9.]+)_summary\.csv$")

    records = []
    for fname in os.listdir(out_dir):
        if not fname.endswith("_summary.csv"):
            continue
        m = pat.match(fname)
        if not m:
            continue

        round_suffix, thr = m.group(1), float(m.group(2))
        df = pd.read_csv(os.path.join(out_dir, fname))

        if "name" not in df.columns:
            print(f"{fname} missing 'name'")
            continue

        for _, row in df.iterrows():
            dataset = str(row.get("name", "")).strip()
            if dataset not in ("ChaosNLI", "VariErr"):
                continue
            mean_jsd = float(row.get("mean_jsd"))
            mean_kl = float(row.get("mean_kl"))
            records.append({
                "threshold": thr,
                "round": round_suffix,       # before / after
                "dataset": dataset,          # ChaosNLI / VariErr
                "mean_jsd": mean_jsd,
                "mean_kl": mean_kl,
            })

    if not records:
        print("No summary files found or no valid rows. Abort.")
        return

    agg = pd.DataFrame(records).sort_values(["dataset", "round", "threshold"])
    agg_csv = os.path.join(out_dir, "summary_by_threshold_per_dataset.csv")
    agg.to_csv(agg_csv, index=False)
    # print("summary_by_threshold_per_dataset.csv written to", out_dir)

    def plot_metric_per_dataset(metric: str):
        for dataset, g_all in agg.groupby("dataset"):
            plt.figure()
            for rd, g in g_all.groupby("round"):
                g = g.sort_values("threshold")
                plt.plot(g["threshold"], g[metric], marker="o", label=rd)
            plt.xlabel("threshold")
            plt.ylabel(metric)
            plt.title(f"{dataset} {metric.upper()} vs Threshold ({model}: before vs after)")
            plt.legend(title="round")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            out_png = os.path.join(out_dir, f"{dataset.lower()}_{metric}_vs_threshold.png")
            plt.savefig(out_png, dpi=200)
            plt.close()

    def plot_metric_all(metric: str):
        plt.figure()
        for (dataset, rd), g in agg.groupby(["dataset", "round"]):
            g = g.sort_values("threshold")
            label = f"{dataset}-{rd}"
            plt.plot(g["threshold"], g[metric], marker="o", label=label)
        plt.xlabel("threshold")
        plt.ylabel(metric)
        plt.title(f"{metric.upper()} vs Threshold ({model})")
        plt.legend(title="series")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        out_png = os.path.join(out_dir, f"combined_{metric}_vs_threshold.png")
        plt.savefig(out_png, dpi=200)
        plt.close()

    for m in ("mean_jsd", "mean_kl"):
        plot_metric_per_dataset(m)
        plot_metric_all(m)

    print("Plots saved to:", out_dir)

if __name__ == "__main__":
    main()
