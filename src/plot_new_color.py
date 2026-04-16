# -*- coding: utf-8 -*-
import os, re, argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


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
        model = "LLaMA-8B"
    elif any(k in last for k in ["llama-70b", "llama_70b", "70b"]):
        model = "LLaMA-70B"
    elif any(k in last for k in ["qwen-7b", "qwen_7b", "7b"]):
        model = "Qwen-7B"
    else:
        model = "Qwen-72B"

    return model, mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--after_dirs", nargs="+", required=True)
    ap.add_argument("--out", default="chaosnli_mean_kl_after.png")
    ap.add_argument("--title", default="ChaosNLI mean_KL vs Threshold")
    args = ap.parse_args()

    sns.set_theme(style="whitegrid", font="DejaVu Sans")

    # ✅ 关键修改：8B 仍然蓝色，70B 改为紫色
    model2color = {
        "LLaMA-8B": "#1f77b4",   # 蓝（保持）
        "LLaMA-70B": "#ff7f0e",  # 橙（更标准的橙色）
        "Qwen-7B":   "#2ca02c",  # 绿（保持）
        "Qwen-72B":  "#8c564b"   # 棕（替换红色）
    }

    mode2alpha = {
        "one-expl": 1.0,
        "one-llm": 0.6,
        "all-llm": 0.2,
        "unknown": 0.3
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

    plt.figure(figsize=(10, 7))

    model_order = ["LLaMA-8B", "LLaMA-70B", "Qwen-7B", "Qwen-72B"]
    mode_order = ["one-expl", "one-llm", "all-llm", "unknown"]

    mrank = {m: i for i, m in enumerate(model_order)}
    orank = {m: i for i, m in enumerate(mode_order)}

    after_curves.sort(key=lambda x: (mrank.get(x[1], 1e9), orank.get(x[2], 1e9)))

    for label, model, mode, df in after_curves:
        c = model2color.get(model, "#444444")
        alpha = mode2alpha.get(mode, 0.6)

        plt.plot(
            df["threshold"],
            df["mean_kl"],
            linestyle="-",
            linewidth=3.5,
            color=c,
            alpha=alpha,
            label=f"{model} · {mode}"
        )

    plt.xlabel("Threshold", fontsize=18)
    plt.ylabel("KL Divergence", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.title(args.title, fontsize=18, weight="bold", pad=18)

    plt.xlim(0.0, 1.0)
    plt.xticks(np.arange(0.0, 1.01, 0.1))

    plt.legend(
        loc="best",
        fontsize=11,
        frameon=True,
        facecolor="white",
        framealpha=0.8
    )

    plt.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    plt.savefig(args.out, dpi=220, bbox_inches="tight")
    pdf_out = os.path.splitext(args.out)[0] + ".pdf"
    plt.savefig(pdf_out, bbox_inches="tight", format="pdf")

    plt.close()

    print("Saved plot to:", args.out)


if __name__ == "__main__":
    main()