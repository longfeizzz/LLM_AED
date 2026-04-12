import json
import re
from pathlib import Path
from tqdm import tqdm
import argparse
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import datasets
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'

label_map = {"e": "entailment", "n": "neutral", "c": "contradiction"}

def parse_json_predictions(text, expected_keys=None):
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

    cleaned = {}
    for k, v in obj.items():
        k = str(k)
        if expected_keys is not None and k not in expected_keys:
            continue
        pv = to_prob(v)
        if pv is not None:
            cleaned[k] = pv
    return cleaned

def load_qwen(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    # if tokenizer.pad_token_id is None:
    #     tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

def build_messages_batch(premise, hypothesis, items):
    reasons_block = ""
    for idx, it in enumerate(items, start=1):
        label = it['label']
        reason = it['reason'].strip()
        reason_id = it['reason_id']
        reasons_block += f"Reason {idx} for label {label}: {reason}\n"

        user_content = (
            "We have collected annotations for an NLI instance together with explanations for the labels.\n"
            "You will first be shown all explanations together so that you understand the overall context, and then your task is to judge whether each reason makes sense for the label. "
            "You must output a single JSON object that maps each explanation's index (1,2,3,...) to its probability in one time.\n"
            "Provide the probability (0.0 - 1.0) that each reason makes sense for the label. "
            "Give ONLY the probability, no other words or explanation. \n\n"
            "Output example: \n"
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
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True) 
    # parser.add_argument("--max_model_len", type=int, default=8192, help="safety: roughly control max output tok")
    args = parser.parse_args()

    model, tokenizer = load_qwen(args.model_name_or_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = datasets.Dataset.from_json(args.input_path)
    predictions = {}

    for instance in tqdm(dataset):
        instance_id = instance["id"]
        premise = instance["premise"]
        hypothesis = instance["hypothesis"]

        items = []
        index_to_reason_id = {}
        # expected_keys = []
        # for idx, (reason_text, label_code, model_id) in enumerate(instance["generated_explanations"]):
        for idx, (reason_text, label_code) in enumerate(instance["generated_explanations"]):
            if label_code not in label_map:
                print(f"Unknown label code {label_code} in instance {instance_id}")
                continue
            label = label_map[label_code]
            # reason_id = f"{model_id}_{instance_id}_{label_code}-{idx}"
            reason_id = f"{instance_id}_{label_code}-{idx}"
            items.append({"reason_id": reason_id, "label": label, "reason": reason_text})
            index_to_reason_id[str(len(items))] = reason_id
            # expected_keys.append(reason_id)

        if not items:
            continue

        messages = build_messages_batch(premise, hypothesis, items)
        # print("messages are:", messages)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print("prompt:", prompt)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        gen = model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
        generated_ids = gen[0, inputs["input_ids"].shape[1]:]
        output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        # print("output_text:", output_text)

        per_instance = parse_json_predictions(output_text)
        print("per_instance:", per_instance)

        for idx_str, reason_id in index_to_reason_id.items():
            if idx_str in per_instance:
                predictions[reason_id] = per_instance[idx_str]
            else:
                predictions[reason_id] = None

        # for rid in expected_keys:
        #     if rid in per_instance:
        #         predictions[rid] = per_instance[rid]
        #     else:
        #         predictions[rid] = None

    with open(output_dir / "scores.json", "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"Done. Output saved to: {output_dir / 'scores.json'}")

if __name__ == "__main__":
    main()
