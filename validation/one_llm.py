import json
import re
from pathlib import Path
from tqdm import tqdm
import argparse
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import datasets


label_map = {"e": "entailment", "n": "neutral", "c": "contradiction"}


def parse_json_predictions(text):
    obj = None
    try:
        obj = json.loads(text)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                pass
    if obj is None:
        return {}

    def to_prob(v):
        if isinstance(v, (int, float)):
            val = float(v)
        elif isinstance(v, str):
            mm = re.search(r"\b(?:0(?:\.\d+)?|1(?:\.0+)?)\b", v)
            if not mm:
                return None
            val = float(mm.group(0))
        else:
            return None
        return max(0.0, min(1.0, val))

    return {str(k): to_prob(v) for k, v in obj.items() if to_prob(v) is not None}

def load_model(model_id, model_type):
    if model_type == "llama":
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer,
            torch_dtype=torch.bfloat16, device_map="auto"
        )
        return {"pipe": pipe, "tokenizer": tokenizer}

    elif model_type == "qwen":
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto", trust_remote_code=True
        )
        return {"model": model, "tokenizer": tokenizer}


def generate(model_type, model_dict, prompt):
    tokenizer = model_dict["tokenizer"]

    if model_type == "llama":
        result = model_dict["pipe"](prompt, max_new_tokens=4096, return_full_text=False, do_sample=False)
        return result[0]["generated_text"]

    elif model_type == "qwen":
        model = model_dict["model"]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        gen = model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
        generated_ids = gen[0, inputs["input_ids"].shape[1]:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True)


def build_messages(premise, hypothesis, items):
    reasons_block = ""
    for idx, it in enumerate(items, start=1):
        reasons_block += f"Reason {idx} for label {it['label']}: {it['reason'].strip()}\n"

    user_content = (
        "We have collected annotations for an NLI instance together with explanations for the labels.\n"
        "You will first be shown all explanations together so that you understand the overall context, and then your task is to judge whether each reason makes sense for the label. "
        "You must output a single JSON object that maps each explanation's index (1,2,3,...) to its probability in one time.\n"
        "Provide the probability (0.0 - 1.0) that each reason makes sense for the label. "
        "Give ONLY the probability, no other words or explanation.\n\n"
        "Output example:\n"
        '{"1": 0.9, "2": 0.8, ...}\n\n'
        f"Context: {premise}\n"
        f"Statement: {hypothesis}\n\n"
        f"{reasons_block}\n"
        "Now output the JSON object ONLY."
    )

    return [
        {"role": "system", "content": "You are an expert linguistic annotator."},
        {"role": "user", "content": user_content},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True, choices=["llama", "qwen"])
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    model_short = args.model_name_or_path.split("/")[-1].replace("-Instruct", "")

    input_path = Path(args.input_path) if args.input_path else repo_root / "processing" / f"{model_short}_generation_raw.jsonl"
    output_dir = Path(args.output_dir) if args.output_dir else repo_root / "validation" / "validation_results" / "one-llm" / model_short
    output_path = output_dir / "scores.json"

    print(f"Model: {args.model_name_or_path}")
    print(f"Input: {input_path}")
    print(f"Output dir: {output_dir}")
    print(f"Output file: {output_path}")

    if not Path(input_path).exists():
        raise FileNotFoundError(f"No input path: {input_path}")

    model_dict = load_model(args.model_name_or_path, args.model_type)
    tokenizer = model_dict["tokenizer"]
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = datasets.Dataset.from_json(input_path)
    predictions = {}

    for instance in tqdm(dataset):
        instance_id = instance["id"]
        premise = instance["premise"]
        hypothesis = instance["hypothesis"]

        items = []
        index_to_reason_id = {}

        for idx, (reason_text, label_code) in enumerate(instance["generated_explanations"]):
            if label_code not in label_map:
                print(f"Unknown label code {label_code} in instance {instance_id}")
                continue
            label = label_map[label_code]
            reason_id = f"{instance_id}_{label_code}-{idx}"
            items.append({"reason_id": reason_id, "label": label, "reason": reason_text})
            index_to_reason_id[str(len(items))] = reason_id

        if not items:
            continue

        messages = build_messages(premise, hypothesis, items)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        output_text = generate(args.model_type, model_dict, prompt)

        per_instance = parse_json_predictions(output_text)
        print("per_instance:", per_instance)

        for idx_str, reason_id in index_to_reason_id.items():
            predictions[reason_id] = per_instance.get(idx_str, None)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    exists = output_path.exists()
    print(f"Done. Output saved to: {output_path}")
    print(f"Output file found: {exists}")


if __name__ == "__main__":
    main()