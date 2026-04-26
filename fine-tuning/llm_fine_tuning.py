# finetuning with LLM-detected error / removal
import json
import sys
from collections import Counter
from pathlib import Path

SHORT2LONG = {"e": "entailment", "n": "neutral", "c": "contradiction"}
ORDER = ["entailment", "neutral", "contradiction"]

BASE_EVAL = "/LLM_AED/evaluation"

def load_jsonl_by_id(path: Path):
    data = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            data[obj["id"]] = obj
    return data

def convert_label_list_to_dist(label_list):
    mapped = [SHORT2LONG.get(x, x) for x in (label_list or [])]
    uniq = set(mapped)
    counter = Counter(uniq)
    e = counter.get("entailment", 0)
    n = counter.get("neutral", 0)
    c = counter.get("contradiction", 0)
    total = e + n + c
    if total == 0:
        return [0.0, 0.0, 0.0]
    return [e / total, n / total, c / total]

def make_clean_record(entry, dist):
    return {
        "uid": entry.get("id"),
        "premise": entry.get("premise"),
        "hypothesis": entry.get("hypothesis"),
        "label": dist,  # [p(e), p(n), p(c)]
    }

def process_one_file(input_file: str, output_file: str = None):
    input_path = Path(input_file)

    if not input_path.exists():
        print(f"[ERROR] File not found: {input_path}")
        return False
    
    model_data = load_jsonl_by_id(input_path)

    cleaned_r1, cleaned_r2 = [], []
    for _, entry in model_data.items():
        # round_1
        r1_list = entry.get("label_set_round_1", [])
        dist_r1 = convert_label_list_to_dist(r1_list)
        cleaned_r1.append(make_clean_record(entry, dist_r1))

        # round_2
        r2_list = entry.get("label_set_round_2", [])
        dist_r2 = convert_label_list_to_dist(r2_list)
        cleaned_r2.append(make_clean_record(entry, dist_r2))

    if output_file:
        out_path = Path(output_file)
        out_dir = out_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        with out_path.open("w", encoding="utf-8") as f:
                for item in cleaned_r1:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                for item in cleaned_r2:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        print(f"[OK] Saved: {out_path}")
        return True
    else:
        print(f"[ERROR] output_file is required")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_file> [output_file]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
 
    success = process_one_file(input_file, output_file)
    sys.exit(0 if success else 1)
