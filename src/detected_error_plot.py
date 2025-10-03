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

# 全局 seaborn 风格
sns.set_theme(style="whitegrid", context="talk", palette="deep")

def parse_name_and_threshold(s: str) -> tuple[str, float | None]:
    if s is None:
        return "", None
    v = str(s).strip()
    m = re.search(r"(\d+(?:\.\d+)?)\s*$", v)
    if not m:
        return v, None
    thr = float(m.group(1))
    prefix = v[:m.start(1)].strip()
    return prefix, thr

def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)

def read_single_csv(csv_path: str | Path) -> pd.DataFrame:
    p = Path(csv_path)
    df = pd.read_csv(p, header=None)
    if df.shape[1] < 4:
        raise ValueError("CSV 至少需要4列：名称+阈值, E, N, C")
    prefixes, thrs = zip(*[parse_name_and_threshold(x) for x in df.iloc[:,0]])
    out = pd.DataFrame({
        "ModelPrefix": prefixes,
        "Threshold": thrs,
        "E": pd.to_numeric(df.iloc[:,1], errors="coerce"),
        "N": pd.to_numeric(df.iloc[:,2], errors="coerce"),
        "C": pd.to_numeric(df.iloc[:,3], errors="coerce"),
    })
    return out[out["Threshold"].notna()].reset_index(drop=True)

def line_plot(df: pd.DataFrame, x: str, ys: Iterable[str], title: str, path: Path):
    # seaborn 更偏好“长表” -> melt
    df_long = df.melt(id_vars=[x], value_vars=list(ys),
                      var_name="Label", value_name="Value")
    plt.figure()
    ax = sns.lineplot(data=df_long, x=x, y="Value", hue="Label", marker="o")
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel("value")
    ax.legend(title="")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)  # 扩展名 .pdf 即导出 PDF
    plt.close()

def per_group_plots(tidy: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for prefix in sorted(tidy["ModelPrefix"].unique()):
        sub = tidy[tidy["ModelPrefix"] == prefix].copy()
        sub = sub.sort_values("Threshold")
        base = out_dir / safe_name(prefix)
        line_plot(
            sub, "Threshold", ["E","N","C"],
            f"{prefix} — Detected Error",
            base.with_name(base.name + "counts.pdf")
        )

def run(csv_path: str | Path, output_dir: str):
    tidy = read_single_csv(csv_path)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    per_group_plots(tidy, out)
    tidy.to_csv(out/"summary_metrics.csv", index=False)
    print(f"Saved charts & summary to: {out}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="单个 CSV 路径")
    ap.add_argument("--out", default="charts_out", help="输出目录")
    args = ap.parse_args()
    run(args.csv, args.out)
