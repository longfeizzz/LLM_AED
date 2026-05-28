import json
from pathlib import Path
import argparse


def split_scores_by_model(input_json):
    input_json = Path(input_json)
    output_dir = input_json.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_json, "r") as f:
        data = json.load(f)

    grouped = {}

    for full_key, value in data.items():
        parts = full_key.split("_", 1)

        if len(parts) != 2:
            print(f"Skip invalid key: {full_key}")
            continue

        model_name, cleaned_key = parts

        if model_name not in grouped:
            grouped[model_name] = {}

        grouped[model_name][cleaned_key] = value

    for model_name, model_data in grouped.items():
        output_path = output_dir / f"{model_name}.json"

        with open(output_path, "w") as f:
            json.dump(model_data, f, indent=2)

        print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, required=True)
    args = parser.parse_args()
    split_scores_by_model(args.input_json)


if __name__ == "__main__":
    main()