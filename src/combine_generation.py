#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from collections import OrderedDict
from pathlib import Path

REQ_FIELDS = ["id", "premise", "hypothesis", "generated_explanations"]

SOURCE_BY_PATH = {
    str(Path("/Users/phoebeeeee/ongoing/LLM_AED/no_preprocessing/llama_8b_generation_raw.jsonl").resolve()): "llama8b",
    str(Path("/Users/phoebeeeee/ongoing/LLM_AED/no_preprocessing/llama_70b_generation_raw.jsonl").resolve()): "llama70b",
    str(Path("/Users/phoebeeeee/ongoing/LLM_AED/no_preprocessing/qwen_7b_generation_raw.jsonl").resolve()): "qwen7b",
    str(Path("/Users/phoebeeeee/ongoing/LLM_AED/no_preprocessing/qwen_72b_generation_raw.jsonl").resolve()): "qwen72b",
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
        tag = SOURCE_BY_PATH.get(fp_resolved)  
        for obj in read_jsonl(fp):
            if not all(k in obj for k in REQ_FIELDS):
                print("Missing", obj)
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
                if merged[_id]["premise"] != obj["premise"]:
                    print(f"Wrong premise（id={_id}）")
                if merged[_id]["hypothesis"] != obj["hypothesis"]:
                    print(f"Wrong hypothesis（id={_id}）")

            for e in obj.get("generated_explanations", []):
                if isinstance(e, (list, tuple)):
                    e_list = list(e) + [tag]
                else:
                    e_list = [e, tag]
                merged[_id]["generated_explanations"].append(e_list)
    return merged

def write_jsonl(merged, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in merged.values():
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("-o", "--output", default="generation_all.jsonl")
    args = parser.parse_args()

    merged = merge_by_id(args.files)
    write_jsonl(merged, args.output)
    print(f"Merging done.")

if __name__ == "__main__":
    main()
