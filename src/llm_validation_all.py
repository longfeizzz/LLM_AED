import json
import re
from pathlib import Path
from tqdm import tqdm
import argparse
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import datasets
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

label_map = {"e": "entailment", "n": "neutral", "c": "contradiction"}

# 解析模型一次性输出的 JSON，并做稳健容错
def parse_json_predictions(text, expected_keys=None):
    obj = None
    try:
        obj = json.loads(text)
    except Exception:
        # 尝试从文本中抽出第一个大括号包裹的 JSON 对象
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
        if expected_keys is not None and k not in expected_keys:
            continue
        pv = to_prob(v)
        if pv is not None:
            cleaned[k] = pv
    return cleaned

def load_llama_pipeline(model_id="meta-llama/Llama-3.1-8B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    return pipe, tokenizer

def build_messages_batch(premise, hypothesis, items):
    reasons_block = ""
    for it in items:
        label = it['label']
        reason = it['reason'].strip()
        reason_id = it['reason_id']
        reasons_block += f"Reason for label {label}: {reason.strip()}\n"

        user_content = (
            "We have collected annotations for an NLI instance together with reasons for the labels.\n"
            "Your task is to judge whether the reasons make sense for the label. "
            "Provide the probability (0.0 - 1.0) that each reason makes sense for the label. "
            "Give ONLY the probability, no other words or explanation. \n\n"
            "You must output a json object that maps each explanation's id to its probability.\n"
            "Output example: \n"
            '{"reason_id_1": 0.9, "reason_id_2": 0.8, ...}\n'
            f"Reason_id: {reason_id}\n"
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
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--input_path", type=str, default="../ZLF/llama_explanation_raw.jsonl")
    parser.add_argument("--output_dir", type=str, default="predictions/llama3.2-3b_batchjson")
    # parser.add_argument("--max_model_len", type=int, default=8192, help="safety: roughly control max output tok")
    args = parser.parse_args()

    pipe, tokenizer = load_llama_pipeline(args.model_name_or_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = datasets.Dataset.from_json(args.input_path)
    predictions = {}

    for instance in tqdm(dataset):
        instance_id = instance["id"]
        premise = instance["premise"]
        hypothesis = instance["hypothesis"]

        items = []
        expected_keys = []
        for idx, (reason_text, label_code) in enumerate(instance["generated_explanations"]):
            if label_code not in label_map:
                print(f"[Warning] Unknown label code {label_code} in instance {instance_id}")
                continue
            label = label_map[label_code]
            reason_id = f"{instance_id}_{label_code}-{idx}"
            items.append({"reason_id": reason_id, "label": label, "reason": reason_text})
            expected_keys.append(reason_id)

        if not items:
            continue

        messages = build_messages_batch(premise, hypothesis, items)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # max_new_tokens = min(4096, 16 * len(items) + 64)

        result = pipe(
            prompt,
            max_new_tokens=4096,
            do_sample=False,
            return_full_text=False
        )
        output_text = result[0]["generated_text"]

        per_instance = parse_json_predictions(output_text, expected_keys=set(expected_keys))

        for rid in expected_keys:
            if rid in per_instance:
                predictions[rid] = per_instance[rid]
            else:
                predictions[rid] = None

    with open(output_dir / "scores.json", "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"Done. Output saved to: {output_dir / 'scores.json'}")

if __name__ == "__main__":
    main()
