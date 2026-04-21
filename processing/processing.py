#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import argparse
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm

def clean_explanation(text: str) -> str:
    return re.sub(r"^\s*(?:[\d]+[\.\)]|[-•*]|[a-zA-Z][\.\)]|\(\w+\))\s*", "", text).strip()

label_map = {"E": "e", "N": "n", "C": "c"}


# raw generation to jsonl file
def inject_one_model(explanation_root: Path, input_jsonl: Path, output_jsonl: Path):
    with open(input_jsonl, "r", encoding="utf-8") as f:
        instances = [json.loads(line) for line in f]

    written = 0
    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for instance in tqdm(instances, desc=f"Inject {explanation_root.name}"):
            sample_id = str(instance["id"])
            subfolder = explanation_root / sample_id
            new_comments = []

            if not subfolder.exists():
                print(f"Missing folder: {subfolder}")
            else:
                for label in ["E", "N", "C"]:
                    file_path = subfolder / label
                    if file_path.exists():
                        with open(file_path, "r", encoding="utf-8") as f:
                            for raw in f:
                                if not raw.strip():
                                    continue
                                exp = clean_explanation(raw)
                                if exp:
                                    new_comments.append([exp, label_map[label]])
                    else:
                        print(f"No file found for {label} in {subfolder}")

            if new_comments:
                new_instance = {
                    "id": instance["id"],
                    "premise": instance["context"],
                    "hypothesis": instance["statement"],
                    "generated_explanations": new_comments,
                }
                fout.write(json.dumps(new_instance, ensure_ascii=False) + "\n")
                written += 1

    print(f"{written} instances written to {output_jsonl}")
    return output_jsonl


# merge all raw generation jsonl files into one for all_llm prompting
REQ_FIELDS = ["id", "premise", "hypothesis", "generated_explanations"]


def read_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def merge_jsonls(jsonl_files: list[Path], output_path: Path):
    merged = OrderedDict()

    for fp in jsonl_files:
        tag = fp.stem.replace("_generation_raw", "") 
        for obj in read_jsonl(fp):
            if not all(k in obj for k in REQ_FIELDS):
                print(f"Missing fields, skipping: {obj.get('id')}")
                continue

            _id = obj["id"]
            if _id not in merged:
                merged[_id] = {
                    "id": _id,
                    "premise": obj["premise"],
                    "hypothesis": obj["hypothesis"],
                    "generated_explanations": [],
                }
            else:
                if merged[_id]["premise"] != obj["premise"]:
                    print(f"Premise mismatch (id={_id})")
                if merged[_id]["hypothesis"] != obj["hypothesis"]:
                    print(f"Hypothesis mismatch (id={_id})")

            for e in obj.get("generated_explanations", []):
                e_list = list(e) if isinstance(e, (list, tuple)) else [e]
                e_list.append(tag)
                merged[_id]["generated_explanations"].append(e_list)

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in merged.values():
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")

    print(f"Merging completed")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation_dir", type=str, default="../generation")
    parser.add_argument("--input_jsonl", type=str, default="../dataset/varierr.json")
    parser.add_argument("--processing_dir", type=str, default="../processing")
    parser.add_argument("--all_dir", type=str, default="../processing/generation_all.jsonl")
    return parser.parse_args()


def main():
    args = parse_args()

    generation_dir = Path(args.generation_dir)
    input_jsonl = Path(args.input_jsonl)
    processing_dir = Path(args.processing_dir)
    processing_dir.mkdir(parents=True, exist_ok=True)

    raw_folders = sorted(generation_dir.glob("*_generation_raw"))
    if not raw_folders:
        print(f"No *_generation_raw folders found in {generation_dir}")
        return

    jsonl_files = []
    for folder in raw_folders:
        model_name = folder.name  
        output_jsonl = processing_dir / f"{model_name}.jsonl"
        inject_one_model(folder, input_jsonl, output_jsonl)
        jsonl_files.append(output_jsonl)

    all_path = processing_dir / args.all_dir
    merge_jsonls(jsonl_files, all_path)
    print("All done.")


if __name__ == "__main__":
    main()
