#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import re
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---- seaborn 全局风格 ----
sns.set_theme(style="whitegrid", context="talk", palette="deep")  # 想更简洁可用 context="paper"

# --------- 工具函数 ---------

def parse_name_and_threshold(s: str) -> tuple[str, float | None]:
    if s is None:
        return "", None
    v = str(s).strip()
    m = re.search(r"(\d+(?:\.\d+)?)\s*$", v)  # 取结尾数字为阈值
    if not m:
        return v, None
    thr = float(m.group(1))
    prefix = v[:m.start(1)].strip()
    return prefix, thr

def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)

# --------- 数据读取：只取“第一列原本为空”的行，先前向填充 ---------

def read_single_csv_use_empty_rows(csv_path: str | Path) -> pd.DataFrame:
    p = Path(csv_path)
    df = pd.read_csv(p, header=None, dtype=str)
    if df.shape[1] < 4:
        raise ValueError("CSV 至少需要4列：名称+阈值, E, N, C")

    col0 = df.iloc[:, 0]
    empty_mask = col0.isna() | col0.str.strip().eq("")

    # 把空串当成 NaN，再前向填充
    df.iloc[:, 0] = col0.mask(col0.notna() & col0.str.strip().eq(""))
    df.iloc[:, 0] = df.iloc[:, 0].ffill()

    # 仅保留“原本为空”的行
    df = df[empty_mask].copy()

    prefixes, thrs = zip(*[parse_name_and_threshold(x) for x in df.iloc[:, 0]])
    out = pd.DataFrame({
        "ModelPrefix": prefixes,
        "Threshold": thrs,
        "E": pd.to_numeric(df.iloc[:, 1], errors="coerce"),
        "N": pd.to_numeric(df.iloc[:, 2], errors="coerce"),
        "C": pd.to_numeric(df.iloc[:, 3], errors="coerce"),
    })
    return out[out["Threshold"].notna()].reset_index(drop=True)

# --------- 画图（seaborn 折线，输出 PDF） ---------

def line_plot_pdf(df: pd.DataFrame, x: str, ys: Iterable[str], title: str, pdf_path: Path):
    # seaborn 更偏好“长表”，这里 melt 一下
    df_long = df.melt(id_vars=[x], value_vars=list(ys),
                      var_name="Label", value_name="Value")

    plt.figure()
    ax = sns.lineplot(data=df_long, x=x, y="Value", hue="Label", marker="o")
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel("value")
    ax.legend(title="")
    plt.tight_layout()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(pdf_path)   # 扩展名 .pdf 即输出 PDF
    plt.close()

def per_group_plots(tidy: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for prefix in sorted(tidy["ModelPrefix"].unique()):
        sub = tidy[tidy["ModelPrefix"] == prefix].copy()
        sub = sub.sort_values("Threshold")
        base = out_dir / safe_name(prefix)
        line_plot_pdf(sub, "Threshold", ["E", "N", "C"],
                      f"{prefix} — Error Overlap", base.with_name(base.name + "error_overlap.pdf"))

# --------- 主流程 ---------

def run(csv_path: str | Path, output_dir: str):
    tidy = read_single_csv_use_empty_rows(csv_path)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    per_group_plots(tidy, out)
    # tidy.to_csv(out / "summary_metrics.csv", index=False)
    print(f"Saved PDF & summary to: {out}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="单个 CSV 路径")
    ap.add_argument("--out", default="charts_out", help="输出目录")
    args = ap.parse_args()
    run(args.csv, args.out)
