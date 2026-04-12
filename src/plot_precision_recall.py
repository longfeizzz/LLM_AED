# -*- coding: utf-8 -*-
import os, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def classify_model_mode(path: str):
    parts = [p.lower() for p in os.path.normpath(path).split(os.sep) if p]
    last   = parts[-2] 
    parent = parts[-3] 

    if parent in {"one-expl", "one_expl"}: mode = "one-expl"
    elif parent in {"one-llm", "one_llm"}: mode = "one-llm"
    elif parent in {"all-llm", "all_llm"}: mode = "all-llm"
    else: mode = "unknown"

    if any(k in last for k in ["llama-8b","llama_8b","8b"]): model = "LLaMA-8B"
    elif any(k in last for k in ["llama-70b","llama_70b","70b"]): model = "LLaMA-70B"
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
    model_order = ["LLaMA-8B","LLaMA-70B","Qwen-7B","Qwen-72B"]
    mode_order  = ["one-expl","one-llm","all-llm","unknown"]
    thr_order   = [round(x,1) for x in np.arange(0.0, 1.01, 0.1)]

    out["model"]     = pd.Categorical(out["model"], categories=model_order, ordered=True)
    out["mode"]      = pd.Categorical(out["mode"],  categories=mode_order,  ordered=True)
    out["threshold"] = pd.Categorical(out["threshold"], categories=thr_order, ordered=True)

    return out.sort_values(["model","mode","threshold"])

def plot_metric(df, metric, out_png):
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    model2color = {
        "LLaMA-8B": "#1f77b4",   # 蓝
        "LLaMA-70B": "#f1c40f",  # 橙
        "Qwen-7B":   "#2ca02c",  # 绿
        "Qwen-72B":  "#d62728"   # 红
    }

    mode2marker = {"one-expl":"o", "one-llm":"s", "all-llm":"^", "unknown":"x"}

    plt.figure(figsize=(12,8))

    # 逐组绘制并附标签
    for (model, mode), g in df.groupby(["model","mode"], sort=False):
        g_sorted = g.sort_values("threshold")
        x = g_sorted["threshold"].astype(float)
        y = g_sorted[metric].astype(float)
        if x.empty:  # 防御
            continue
        plt.plot(x, y,
                 color=model2color.get(model, "#444444"),
                 marker=mode2marker.get(mode, "x"),
                 linewidth=2,
                 label=f"{model} · {mode}")

    plt.xlabel("Threshold", fontsize=16)
    plt.ylabel(metric.capitalize(), fontsize=16)
    plt.title(f"{metric.capitalize()} vs. Threshold", fontsize=16, weight="bold",pad=18)
    plt.xlim(0.0, 1.0)
    plt.xticks([round(x,1) for x in np.arange(0.0, 1.01, 0.1)])

    # 仅 precision 放大
    if metric.lower() == "precision":
        plt.ylim(0.4, 1.0)
    else:
        plt.ylim(0.0, 1.0)
    # plt.ylim(0.0, 1.0)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=4, fontsize=14, frameon=True, facecolor="white", framealpha=0.7)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.savefig(os.path.splitext(out_png)[0] + ".pdf", bbox_inches="tight")
    plt.savefig(os.path.splitext(out_png)[0] + ".png", bbox_inches="tight")
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", required=True,
                    help="CSV files with columns: precision, recall, f1, threshold")
    ap.add_argument("--out_prefix", default="pr_curves")
    args = ap.parse_args()

    df = load_all(args.csvs)
    df.to_csv(args.out_prefix + "_merged.csv", index=False)

    plot_metric(df, "precision", args.out_prefix + "_precision.png")
    plot_metric(df, "recall",    args.out_prefix + "_recall.png")
    print("Saved plots:", args.out_prefix + "_precision.png", args.out_prefix + "_recall.png")

if __name__ == "__main__":
    main()
