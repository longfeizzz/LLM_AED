import os
import pandas as pd
from ast import literal_eval
import re

def evaluate_file(csv_path):
    df = pd.read_csv(csv_path)

    TP, FP, FN = 0, 0, 0
    for _, row in df.iterrows():
        llm = set(literal_eval(row["llm_validated"]))
        varierr = set(literal_eval(row["varierr_validated"]))

        TP += len(llm & varierr)       # 预测对的
        FP += len(llm - varierr)       # 多预测的
        FN += len(varierr - llm)       # 漏掉的

    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}

def evaluate_folder(folder, out_csv="results_summary.csv"):
    results = []
    for fname in os.listdir(folder):
        if not fname.endswith(".csv"):
            continue

        m = re.search(r"with_validation_([0-9.]+)_merged_validation", fname)
        threshold = m.group(1) if m else "unknown"

        fpath = os.path.join(folder, fname)
        print(f"[Info] Evaluating {fpath} with threshold {threshold}")
        metrics = evaluate_file(fpath)
        # metrics["file"] = fname
        metrics["threshold"] = threshold 
        results.append(metrics)

    # 保存结果
    df_results = pd.DataFrame(results)
    out_path = os.path.join(folder, out_csv)
    df_results.to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")
    return df_results

if __name__ == "__main__":
    folder = "/Users/phoebeeeee/ongoing/LLM_AED/new_processing/validation_result/one_llm/qwen_72b_all/validated_overlap_2"  # 修改成你的文件夹路径
    df = evaluate_folder(folder) 
    print(df)
