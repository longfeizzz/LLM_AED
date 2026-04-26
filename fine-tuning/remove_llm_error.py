import json
from collections import Counter
from pathlib import Path
from tqdm import tqdm
import sys

VARIERR_FILE = "/LLM_AED/dataset/varierr/varierr.json"

label_map_short2long = {'e': 'entailment', 'n': 'neutral', 'c': 'contradiction'}

def load_jsonl_to_dict_by_id(path: Path):
    data = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            data[obj["id"]] = obj
    return data

def convert_label_list_to_dist(label_list):
    counter = Counter(label_list or [])
    e = counter.get("entailment", 0)
    n = counter.get("neutral", 0)
    c = counter.get("contradiction", 0)
    total = e + n + c
    if total == 0:
        return [0.0, 0.0, 0.0]
    return [e / total, n / total, c / total]

def make_clean_record(raw):
    return {
        "uid": raw.get("id", raw.get("uid")),
        "premise": raw.get("context"),
        "hypothesis": raw.get("statement"),
        "label": raw.get("label"),
    }

def process_one_file(model_file: str, out_dir: str):
    model_path = Path(model_file)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    varierr_data = load_jsonl_to_dict_by_id(Path(VARIERR_FILE))
    model_data = load_jsonl_to_dict_by_id(model_path)

    output_file = out_dir / "varierr_r2_cleaned.jsonl"

    merged_cleaned = []

    for uid, var_entry in varierr_data.items():
        var_entry = dict(var_entry)
        model_entry = model_data.get(uid, {})

        var_entry.pop('entailment', None)
        var_entry.pop('contradiction', None)
        var_entry.pop('neutral', None)
        var_entry.pop('idk', None)

        if 'error' in model_entry:
            error_raw = model_entry['error'] or []
            error_mapped = [label_map_short2long.get(lbl, lbl) for lbl in error_raw]
            var_entry['error_llm'] = error_mapped

        original_labels = set(var_entry.get('label_set_round_2', []))
        error_labels = set(var_entry.get('error_llm', []))
        label_set_llm = sorted(original_labels - error_labels)

        dist = convert_label_list_to_dist(label_set_llm)
        cleaned = make_clean_record(var_entry, dist)
        merged_cleaned.append(cleaned)

    with output_file.open("w", encoding="utf-8") as fout:
        for item in merged_cleaned:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[OK] Saved: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py model_file output_dir")
        sys.exit(1)

    process_one_file(sys.argv[1], sys.argv[2])