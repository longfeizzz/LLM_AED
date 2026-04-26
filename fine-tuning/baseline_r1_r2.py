import json
from collections import Counter
from pathlib import Path

VARIERR_FILE = "/LLM_AED/dataset/varierr/varierr.json"
OUT_DIR      = "LLM_AED/dataset"

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
    uniq = set(label_list or [])
    counter = Counter(uniq)

    e = counter.get("entailment", 0)
    n = counter.get("neutral", 0)
    c = counter.get("contradiction", 0)

    total = e + n + c
    if total == 0:
        return [0.0, 0.0, 0.0]
    return [e / total, n / total, c / total]

def make_clean_record(raw, dist):
    return {
        "uid": raw.get("id", raw.get("uid")),
        "premise": raw.get("context"),
        "hypothesis": raw.get("statement"),
        "label": dist,
    }

def main():
    varierr_data = load_jsonl_to_dict_by_id(Path(VARIERR_FILE))
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_r1 = out_dir / "varierr_r1.jsonl"
    out_r2 = out_dir / "varierr_r2.jsonl"

    cleaned_r1, cleaned_r2 = [], []

    for _, var_entry in varierr_data.items():
        # round1
        r1_list = var_entry.get('label_set_round_1', [])
        dist_r1 = convert_label_list_to_dist(r1_list)
        cleaned_r1.append(make_clean_record(var_entry, dist_r1))

        # round2
        r2_list = var_entry.get('label_set_round_2', [])
        dist_r2 = convert_label_list_to_dist(r2_list)
        cleaned_r2.append(make_clean_record(var_entry, dist_r2))

    with out_r1.open("w", encoding="utf-8") as f1:
        for item in cleaned_r1:
            f1.write(json.dumps(item, ensure_ascii=False) + "\n")

    with out_r2.open("w", encoding="utf-8") as f2:
        for item in cleaned_r2:
            f2.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[OK] Saved: {out_r1}")
    print(f"[OK] Saved: {out_r2}")

if __name__ == "__main__":
    main()