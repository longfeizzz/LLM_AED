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
    prefix = get_prefix(os.path.dirname(d)) if os.path.basename(d) == "kld_jsd" else get_prefix(d)
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

def classify_model_mode(path: str):
    last = os.path.basename(os.path.normpath(path)).lower()          
    parent = os.path.basename(os.path.dirname(os.path.normpath(path))).lower() 

    if parent in {"one-expl", "one_expl"}: mode = "one-expl"
    elif parent in {"one-llm", "one_llm"}: mode = "one-llm"
    elif parent in {"all-llm", "all_llm", "all-llm/"} or parent == "all_llm": mode = "all-llm"
    else: mode = "unknown"

    if any(k in last for k in ["llama-8b", "llama_8b", "8b"]): model = "LLaMA-8B"
    elif any(k in last for k in ["llama-70b", "llama_70b", "70b"]): model = "LLaMA-70B"
    elif any(k in last for k in ["qwen-7b", "qwen_7b", "7b"]): model = "Qwen-7B"
    else: model = "Qwen-72B"

    return model, mode

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before_dir", required=True)
    ap.add_argument("--after_dirs", nargs="+", required=True)  # 12 个目录
    ap.add_argument("--out", default="chaosnli_mean_kl_before_vs_after.png")
    ap.add_argument("--title", default="ChaosNLI mean_KL vs Threshold")
    args = ap.parse_args()

    sns.set_theme(style="whitegrid", font="DejaVu Sans")

    # 颜色按模型，形状按场景
    # model_list = ["LLaMA-8B", "LLaMA-70B", "Qwen-7B", "Qwen-72B"]
    model2color = {
        "LLaMA-8B": "#1f77b4",   # 蓝
        "LLaMA-70B": "#ff7f0e",  # 橙
        "Qwen-7B":   "#2ca02c",  # 绿
        "Qwen-72B":  "#d62728"   # 红
    }
    mode2marker = {"one-expl": "o", "one-llm": "s", "all-llm": "^"}

    # before
    df_before = load_curve(args.before_dir, "before", dataset_keep="ChaosNLI")
    if df_before.empty:
        print("[ERROR] No BEFORE data found.")
        return

    # after
    after_curves = []
    for d in args.after_dirs:
        df_after = load_curve(d, "after", dataset_keep="ChaosNLI")
        if df_after.empty:
            print(f"[WARN] No AFTER data in {d}")
            continue
        label = get_prefix(d)
        model, mode = classify_model_mode(d) 
        after_curves.append((label, model, mode, df_after))

    if not after_curves:
        print("[ERROR] No AFTER data found.")
        return

    plt.figure(figsize=(10, 7))

    # before 曲线
    plt.plot(df_before["threshold"], df_before["mean_kl"],
             linestyle="--", marker="o", linewidth=2,
             color="black", label="before")

    # after 曲线：颜色=模型，点型=模式
    for label, model, mode, df in after_curves:
        c = model2color.get(model)
        m = mode2marker.get(mode, "x")
        plt.plot(df["threshold"], df["mean_kl"],
                 marker=m, linewidth=2, color=c,
                 label=f"{model} · {mode}")

    plt.xlabel("Threshold", fontsize=12)
    plt.ylabel("KL Divergence", fontsize=12)
    plt.title(args.title, fontsize=14, weight="bold")

    # x 轴 0.0 — 1.0（数据只有 0.1—0.9 也会完整显示）
    plt.xlim(0.0, 1.0)
    plt.xticks(np.arange(0.0, 1.01, 0.1))

    # 图例
    plt.legend(loc="upper left", fontsize=9, frameon=True, facecolor="white", framealpha=0.7, ncol=1)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=220, bbox_inches="tight")
    pdf_out = os.path.splitext(args.out)[0] + ".pdf"
    plt.savefig(pdf_out, bbox_inches="tight", format="pdf")
    plt.close()
    print("Saved plot to:", args.out)

if __name__ == "__main__":
    main()
