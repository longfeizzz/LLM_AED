# -*- coding: utf-8 -*-
import os, re, argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def norm_dir(d: str) -> str:
    kd = os.path.join(d, "kld_jsd_2")
    return kd if os.path.isdir(kd) else d

def get_prefix(d: str) -> str:
    return os.path.basename(os.path.normpath(d))

def load_curve(d: str, round_tag: str, dataset_keep: str = "ChaosNLI") -> pd.DataFrame:
    d = norm_dir(d)
    prefix = get_prefix(os.path.dirname(d)) if os.path.basename(d) == "kld_jsd_2" else get_prefix(d)
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
    elif parent in {"all-llm", "all_llm"} or parent == "all_llm": mode = "all-llm"
    else: mode = "unknown"

    if any(k in last for k in ["llama-8b", "llama_8b", "8b"]): model = "LLaMA-8B"
    elif any(k in last for k in ["llama-70b", "llama_70b", "70b"]): model = "LLaMA-70B"
    elif any(k in last for k in ["qwen-7b", "qwen_7b", "7b"]): model = "Qwen-7B"
    else: model = "Qwen-72B"

    return model, mode

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before_dirs", nargs="+", required=True)   # 改：多个 before
    ap.add_argument("--after_dirs", nargs="+", required=True)    # 12 个目录
    ap.add_argument("--out", default="chaosnli_mean_kl_before_vs_after_2.png")
    ap.add_argument("--title", default="ChaosNLI mean_KL vs Threshold")
    args = ap.parse_args()

    sns.set_theme(style="whitegrid", font="DejaVu Sans")

    # 颜色按模型，形状按场景
    model2color = {
        "LLaMA-8B": "#1f77b4",   # 蓝
        "LLaMA-70B": "#f1c40f",  # 橙
        "Qwen-7B":   "#2ca02c",  # 绿
        "Qwen-72B":  "#d62728"   # 红
    }

    # model2color = {
    #     "LLaMA-8B": "#9467bd",   # 蓝
    #     "LLaMA-70B": "#8c564b",  # 橙
    #     "Qwen-7B":   "#e377c2",  # 绿
    #     "Qwen-72B":  "#7f7f7f"   # 红
    # }

    mode2marker = {"one-expl": "o", "one-llm": "s", "all-llm": "^", "unknown": "x"}

    # BEFORE 多条曲线
    before_styles = [
        {"ls": "--", "c": "#000000"},
        {"ls": "-.", "c": "#555555"},
        {"ls": ":",  "c": "#888888"},
        {"ls": (0, (5, 2, 1, 2)), "c": "#BBBBBB"},  # 自定义虚线
    ]

    before_curves = []
    for d in args.before_dirs:
        df_b = load_curve(d, "before", dataset_keep="ChaosNLI")
        if df_b.empty:
            print(f"[WARN] No BEFORE data in {d}")
            continue
        label = get_prefix(d)
        model, mode = classify_model_mode(d)  # 若 before 也按目录层级编码模型/场景
        before_curves.append((label, model, mode, df_b))

    if not before_curves:
        print("[ERROR] No BEFORE data found.")
        return

    # AFTER
    after_curves = []
    for d in args.after_dirs:
        df_a = load_curve(d, "after", dataset_keep="ChaosNLI")
        if df_a.empty:
            print(f"[WARN] No AFTER data in {d}")
            continue
        label = get_prefix(d)
        model, mode = classify_model_mode(d)
        after_curves.append((label, model, mode, df_a))

    if not after_curves:
        print("[ERROR] No AFTER data found.")
        return

    plt.figure(figsize=(10, 7))

    # 画 BEFORE（灰度，不同虚线）
    # for label, model, mode, df in before_curves:
    #     c = model2color.get(model, "#444444")  # 模型颜色
    #     plt.plot(df["threshold"], df["mean_kl"],
    #             linestyle="--", marker=None, linewidth=2,
    #             color=c, label=f"before · {model}")

    # 画 AFTER（颜色=模型，标记=场景）
    # 定义想要的顺序
    model_order = ["LLaMA-8B", "LLaMA-70B", "Qwen-7B", "Qwen-72B"]
    mode_order  = ["one-expl", "one-llm", "all-llm", "unknown"]
    mrank = {m:i for i,m in enumerate(model_order)}
    orank = {m:i for i,m in enumerate(mode_order)}

    # 排序曲线列表：[(label, model, mode, df), ...]
    before_curves.sort(key=lambda x: (mrank.get(x[1], 1e9), orank.get(x[2], 1e9)))
    after_curves.sort(key=lambda x: (mrank.get(x[1], 1e9), orank.get(x[2], 1e9)))

    
    for label, model, mode, df in after_curves:
        c = model2color.get(model, "#444444")
        m = mode2marker.get(mode, "x")
        plt.plot(df["threshold"], df["mean_kl"],
                linestyle="-", marker=m, linewidth=2,
                color=c, label=f"{model} · {mode}")
        
    plt.xlabel("Threshold", fontsize=16)
    plt.ylabel("KL Divergence", fontsize=16)
    plt.title(args.title, fontsize=16, weight="bold",pad=18)

    # x 轴固定 0.0—1.0
    plt.xlim(0.0, 1.0)
    plt.xticks(np.arange(0.0, 1.01, 0.1))

    # 图例
    # plt.legend(loc="lower left", fontsize=9, frameon=True, facecolor="white", framealpha=0.7)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=220, bbox_inches="tight")
    pdf_out = os.path.splitext(args.out)[0] + ".pdf"
    plt.savefig(pdf_out, bbox_inches="tight", format="pdf")
    plt.close()
    print("Saved plot to:", args.out)

if __name__ == "__main__":
    main()
