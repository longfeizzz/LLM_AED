# -*- coding: utf-8 -*-
import os, re, argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict


def norm_dir(d: str) -> str:
    kd = os.path.join(d, "kld_jsd")
    return kd if os.path.isdir(kd) else d


def get_prefix(d: str) -> str:
    return os.path.basename(os.path.normpath(d))


def load_curve(d: str, round_tag: str, dataset_keep: str = "ChaosNLI") -> pd.DataFrame:
    d = norm_dir(d)

    raw_prefix = get_prefix(os.path.dirname(d)) if os.path.basename(d) == "kld_jsd" else get_prefix(d)
    rp = raw_prefix.lower()

    if "llama" in rp and "8b" in rp:
        prefix = "llama_8b"
    elif "llama" in rp and "70b" in rp:
        prefix = "llama_70b"
    elif "qwen" in rp and "7b" in rp:
        prefix = "qwen_7b"
    elif "qwen" in rp and "72b" in rp:
        prefix = "qwen_72b"
    else:
        prefix = rp

    pat = re.compile(rf"^{re.escape(prefix)}_{re.escape(round_tag)}_([0-9.]+)_summary\.csv$")

    rec = []
    for fname in os.listdir(d):
        m = pat.match(fname)
        if not m:
            continue

        thr = float(m.group(1))
        fpath = os.path.join(d, fname)

        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            print(f"[WARN] Failed to read {fpath}: {e}")
            continue

        if "name" not in df.columns or "mean_kl" not in df.columns:
            continue

        sub = df[df["name"].astype(str).str.strip() == dataset_keep]
        if sub.empty:
            continue

        val = float(sub["mean_kl"].iloc[0])
        rec.append({"threshold": thr, "mean_kl": val})

    return pd.DataFrame(rec).sort_values("threshold") if rec else pd.DataFrame()


def get_round_tag(mode: str) -> str:
    if mode == "one-expl":
        return "original_after"
    elif mode == "one-llm":
        return "all_after"
    elif mode == "all-llm":
        return "peer_after"
    else:
        return "after"


def classify_model_mode(path: str):
    norm = path
    if os.path.basename(os.path.normpath(path)) == "kld_jsd":
        norm = os.path.dirname(os.path.normpath(path))

    last = os.path.basename(os.path.normpath(norm)).lower()
    parent = os.path.basename(os.path.dirname(os.path.normpath(norm))).lower()

    if parent in {"one-expl", "one_expl"}:
        mode = "one-expl"
    elif parent in {"one-llm", "one_llm"}:
        mode = "one-llm"
    elif parent in {"all-llm", "all_llm"}:
        mode = "all-llm"
    else:
        mode = "unknown"

    if any(k in last for k in ["llama-8b", "llama_8b", "8b"]):
        model = "Llama-8B"
    elif any(k in last for k in ["llama-70b", "llama_70b", "70b"]):
        model = "Llama-70B"
    elif any(k in last for k in ["qwen-7b", "qwen_7b", "7b"]):
        model = "Qwen-7B"
    else:
        model = "Qwen-72B"

    return model, mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--after_dirs", nargs="+", required=True)
    ap.add_argument("--out", default="chaosnli_mean_kl_split.png")
    ap.add_argument("--title", default="ChaosNLI mean_KL vs Threshold")
    args = ap.parse_args()

    sns.set_theme(style="whitegrid", font="DejaVu Sans")

    # 颜色按模型
    model2color = {
        "Llama-8B": "#1f77b4",   # 蓝
        "Llama-70B": "#f1c40f",  # 橙
        "Qwen-7B":   "#2ca02c",  # 绿
        "Qwen-72B":  "#d62728"   # 红
    }

    # marker 按模式
    mode2marker = {
        "one-expl": "o",
        "one-llm": "s",
        "all-llm": "^",
        "unknown": "x"
    }

    after_curves = []

    for d in args.after_dirs:
        model, mode = classify_model_mode(d)
        round_tag = get_round_tag(mode)

        df_a = load_curve(d, round_tag, dataset_keep="ChaosNLI")
        if df_a.empty:
            print(f"[WARN] No AFTER data in {d}")
            continue

        label = get_prefix(d)
        after_curves.append((label, model, mode, df_a))

    if not after_curves:
        print("[ERROR] No AFTER data found.")
        return

    # ====== 分组 ======
    grouped = defaultdict(list)
    for item in after_curves:
        grouped[item[1]].append(item)

    model_order = ["Llama-8B", "Llama-70B", "Qwen-7B", "Qwen-72B"]
    mode_order = ["one-expl", "one-llm", "all-llm", "unknown"]

    # ====== 画图 ======
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)

    for i, model in enumerate(model_order):
        ax = axes[i]
        curves = grouped.get(model, [])

        color = model2color.get(model, "#444444")

        curves.sort(key=lambda x: mode_order.index(x[2]) if x[2] in mode_order else 999)

        for label, m, mode, df in curves:
            marker = mode2marker.get(mode, "x")

            ax.plot(
                df["threshold"],
                df["mean_kl"],
                linestyle="-",
                marker=marker,
                markersize=8,
                linewidth=2.5,
                color=color,
                label=mode
            )

        ax.set_title(model, fontsize=16, weight="bold")
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks(np.arange(0.0, 1.01, 0.1))
        ax.tick_params(axis='both', labelsize=12)
        ax.set_xlabel("Threshold", fontsize=14)
        # ax.set_xlabel("")  # 去掉 "Threshold"
        # ax.tick_params(axis='x', which='both', labelbottom=False)  # 去掉数字

        if i == 0:
            ax.set_ylabel("KL Divergence", fontsize=18)

        ax.legend(
            loc="lower left",      
            fontsize=10,
            frameon=True,
            facecolor="white",
            framealpha=0.8
        )


    fig.suptitle(args.title, fontsize=18, weight="bold", y=0.9)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    plt.savefig(args.out, dpi=220, bbox_inches="tight")

    pdf_out = os.path.splitext(args.out)[0] + ".pdf"
    plt.savefig(pdf_out, bbox_inches="tight", format="pdf")

    plt.close()

    print("Saved plot to:", args.out)


if __name__ == "__main__":
    main()