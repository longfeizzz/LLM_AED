#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from collections import OrderedDict
from pathlib import Path

REQ_FIELDS = ["id", "premise", "hypothesis", "generated_explanations"]

# 你的四个绝对路径 -> 标签
SOURCE_BY_PATH = {
    str(Path("/Users/phoebeeeee/ongoing/Beyond-noise-MA-Zuo/dataset/llama-8b/llama_8b_explanation_raw.jsonl").resolve()): "llama8b",
    str(Path("/Users/phoebeeeee/ongoing/Beyond-noise-MA-Zuo/dataset/llama-70b/llama_70b_explanation_raw.jsonl").resolve()): "llama70b",
    str(Path("/Users/phoebeeeee/ongoing/Beyond-noise-MA-Zuo/dataset/gpt/gpt_explanation_raw.jsonl").resolve()): "gpt",
    str(Path("/Users/phoebeeeee/ongoing/Beyond-noise-MA-Zuo/EACL/qwen_7b_generation_raw.jsonl").resolve()): "qwen",
}

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def merge_by_id(files):
    merged = OrderedDict()  # {id: {...}}
    for fp in files:
        fp_resolved = str(Path(fp).resolve())
        tag = SOURCE_BY_PATH.get(fp_resolved, Path(fp).stem)  # 未匹配时用文件名stem当标签
        for obj in read_jsonl(fp):
            if not all(k in obj for k in REQ_FIELDS):
                print("缺少字段，已跳过:", obj)
                continue

            _id = obj["id"]
            if _id not in merged:
                merged[_id] = {
                    "id": _id,
                    "premise": obj["premise"],
                    "hypothesis": obj["hypothesis"],
                    "generated_explanations": []
                }
            else:
                # 如有不一致，保留首次出现的 premise/hypothesis
                if merged[_id]["premise"] != obj["premise"]:
                    print(f"[WARN] premise 不一致（id={_id}），保留首次版本")
                if merged[_id]["hypothesis"] != obj["hypothesis"]:
                    print(f"[WARN] hypothesis 不一致（id={_id}），保留首次版本")

            # 仅合并（不去重），并在每条 explanation 结尾加来源标签
            for e in obj.get("generated_explanations", []):
                if isinstance(e, (list, tuple)):
                    e_list = list(e) + [tag]
                else:
                    # 非列表数据也收下：用 [内容, tag] 的形式
                    e_list = [e, tag]
                merged[_id]["generated_explanations"].append(e_list)
    return merged

def write_jsonl(merged, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in merged.values():
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")

def main():
    parser = argparse.ArgumentParser(description="按 id 合并多个 JSONL，不去重；在每条 explanation 末尾添加来源标签。")
    parser.add_argument("files", nargs="+", help="一个或多个 JSONL 的绝对路径")
    parser.add_argument("-o", "--output", default="generation_all.jsonl", help="输出 JSONL 路径")
    args = parser.parse_args()

    merged = merge_by_id(args.files)
    write_jsonl(merged, args.output)
    print(f"完成：合并 {len(merged)} 个 id，已保存到 {args.output}")

if __name__ == "__main__":
    main()
