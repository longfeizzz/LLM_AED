# -*- coding: utf-8 -*-
import json
from collections import defaultdict
import argparse


def add_validation_tags(all_data, score_data, threshold):
    for instance in all_data:
        inst_id = instance["id"]
        new_comments = []
        for idx, (reason_text, label_code) in enumerate(instance["generated_explanations"]):
            reason_id = f"{inst_id}_{label_code}-{idx}"
            if reason_id in score_data:
                if score_data[reason_id] >= threshold:
                    tag = "validated"
                else:
                    tag = "not_validated"
            else:
                print("missing:", reason_id)
                tag = "not_scored"

            new_comments.append([reason_text, label_code, tag])
        instance["generated_explanations"] = new_comments
    return all_data


def add_label_counts(all_data):
    for instance in all_data:
        comments = instance.get("generated_explanations", [])

        count_r1 = defaultdict(int)
        count_r2 = defaultdict(int)

        for reason_text, label_code, tag in comments:
            count_r1[label_code] += 1
            if tag == "validated":
                count_r2[label_code] += 1

        set_r1 = set(count_r1.keys())
        set_r2 = set(count_r2.keys())

        instance["label_count_round_1"] = dict(count_r1)
        instance["label_set_round_1"] = list(set_r1)

        instance["label_count_round_2"] = dict(count_r2)
        instance["label_set_round_2"] = list(set_r2)

        instance["error"] = list(set_r1 - set_r2)
        not_validated_exp = {
            label: count_r1[label] - count_r2.get(label, 0)
            for label in count_r1
            if count_r1[label] - count_r2.get(label, 0) > 0
        }
        instance["not_validated_exp"] = not_validated_exp

    return all_data


def main(args):
    with open(args.score_file, "r", encoding="utf-8") as f:
        score_data = json.load(f)
    score_data = {
        k: float(v) if v is not None else 0.0
        for k, v in score_data.items()
    }

    with open(args.data_file, "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f]

    all_data = add_validation_tags(all_data, score_data, args.threshold)
    all_data = add_label_counts(all_data)

    with open(args.output_file, "w", encoding="utf-8") as f:
        for instance in all_data:
            f.write(json.dumps(instance, ensure_ascii=False) + "\n")

    print("Done. Output saved to", args.output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_file", type=str, required=True)
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.7)
    args = parser.parse_args()

    main(args)
