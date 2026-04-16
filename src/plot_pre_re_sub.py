# -*- coding: utf-8 -*-
import os, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


def classify_model_mode(path: str):
    parts = [p.lower() for p in os.path.normpath(path).split(os.sep) if p]
    last   = parts[-2]
    parent = parts[-3]

    if parent in {"one-expl", "one_expl"}: mode = "one-expl"
    elif parent in {"one-llm", "one_llm"}: mode = "one-llm"
    elif parent in {"all-llm", "all_llm"}: mode = "all-llm"
    else: mode = "unknown"

    if any(k in last for k in ["llama-8b","llama_8b","8b"]): model = "Llama-8B"
    elif any(k in last for k in ["llama-70b","llama_70b","70b"]): model = "Llama-70B"
    elif any(k in last for k in ["qwen-7b","qwen_7b","7b"]): model = "Qwen-7B"
    else: model = "Qwen-72B"

    return model, mode


def load_all(csv_files):
    rows = []
    for f in csv_files:
        df = pd.read_csv(f)

        need = {"precision","recall","f1","threshold"}
        if not need.issubset(df.columns):
            raise ValueError(f"{f} missing columns, got {df.columns.tolist()}")

        model, mode = classify_model_mode(os.path.dirname(f))
        df["model"] = model
        df["mode"]  = mode

        rows.append(df[["model","mode","threshold","precision","recall","f1"]])

    out = pd.concat(rows, ignore_index=True)

    model_order = ["Llama-8B","Llama-70B","Qwen-7B","Qwen-72B"]
    mode_order  = ["one-expl","one-llm","all-llm","unknown"]
    thr_order   = [round(x,1) for x in np.arange(0.0, 1.01, 0.1)]

    out["model"]     = pd.Categorical(out["model"], categories=model_order, ordered=True)
    out["mode"]      = pd.Categorical(out["mode"],  categories=mode_order,  ordered=True)
    out["threshold"] = pd.Categorical(out["threshold"], categories=thr_order, ordered=True)

    return out.sort_values(["model","mode","threshold"])


def plot_metric(df, metric, out_png):
    sns.set_theme(style="whitegrid", font="DejaVu Sans")

    model2color = {
        "Llama-8B": "#1f77b4",   # 蓝
        "Llama-70B": "#f1c40f",  # 橙
        "Qwen-7B":   "#2ca02c",  # 绿
        "Qwen-72B":  "#d62728"   # 红
    }

    mode2marker = {
        "one-expl":"o",
        "one-llm":"s",
        "all-llm":"^",
        "unknown":"x"
    }

    model_order = ["Llama-8B","Llama-70B","Qwen-7B","Qwen-72B"]
    mode_order  = ["one-expl","one-llm","all-llm","unknown"]

    # ====== 分组 ======
    grouped = defaultdict(list)
    for (model, mode), g in df.groupby(["model","mode"], sort=False):
        grouped[model].append((mode, g))

    # ====== 子图 ======
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)

    for i, model in enumerate(model_order):
        ax = axes[i]
        curves = grouped.get(model, [])

        color = model2color.get(model, "#444444")

        # 排序 mode
        curves.sort(key=lambda x: mode_order.index(x[0]) if x[0] in mode_order else 999)

        for mode, g in curves:
            g_sorted = g.sort_values("threshold")

            x = g_sorted["threshold"].astype(float)
            y = g_sorted[metric].astype(float)

            if x.empty:
                continue

            ax.plot(
                x, y,
                color=color,
                marker=mode2marker.get(mode, "x"),
                linewidth=2.5,
                markersize=8,
                label=mode
            )

        ax.set_title(model, fontsize=16, weight="bold",pad=12)
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks(np.arange(0.0, 1.01, 0.1))
        ax.tick_params(axis='both', labelsize=12)
        ax.set_xlabel("Threshold", fontsize=14)

        # y 轴
        if metric.lower() == "precision":
            ax.set_ylim(0.4, 1.0)
        else:
            ax.set_ylim(0.0, 1.0)

        if i == 0:
            ax.set_ylabel(metric.capitalize(), fontsize=18)

        # ax.legend(
        #     loc="lower left",      
        #     fontsize=10,
        #     frameon=True,
        #     facecolor="white",
        #     framealpha=0.8
        # )

    # fig.suptitle(f"{metric.capitalize()} vs. Threshold", fontsize=18, weight="bold",y=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.savefig(os.path.splitext(out_png)[0] + ".pdf", bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", required=True)
    ap.add_argument("--out_prefix", default="pr_curves")
    args = ap.parse_args()

    df = load_all(args.csvs)

    df.to_csv(args.out_prefix + "_merged.csv", index=False)

    plot_metric(df, "precision", args.out_prefix + "_precision.png")
    plot_metric(df, "recall",    args.out_prefix + "_recall.png")

    print("Saved plots:",
          args.out_prefix + "_precision.png",
          args.out_prefix + "_recall.png")


if __name__ == "__main__":
    main()