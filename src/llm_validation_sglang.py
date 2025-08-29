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

def extract_probability(text: str):
    m = re.search(r"\b(?:0(?:\.\d+)?|1(?:\.0+)?)\b", text)
    return float(m.group(0)) if m else None

PROMPT_TEMPLATE = ( 
    "We have collected annotations for an NLI instance together with reasons for the labels. "
    "Your task is to judge whether the reasons make sense for the label. "
    "Provide the probability (0.0 - 1.0) that the reason makes sense for the label. "
    "Give ONLY the probability, no other words or explanation. \n"
    "For example: \nProbability: <the probability between 0.0 and 1.0 that the reason makes sense for the label, without any extra commentary whatsoever; just the probability!> \n\n"
)

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

def build_starting_messages(premise, hypothesis, items):
    # 大背景：模板 + 上下文 + 列出所有 reasons（含标签）
    reasons_block = ""
    for it in items:
        reasons_block += f"Reason for label {it['label']}: {it['reason'].strip()}\n"

    content = (
        f"{PROMPT_TEMPLATE}\n"
        f"Context: {premise}\n"
        f"Statement: {hypothesis}\n\n"
        f"{reasons_block.strip()}"
    )

    return [
        {"role": "system", "content": "You are an expert linguistic annotator."},
        {"role": "user", "content": content},
    ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--input_path", type=str, default="../ZLF/llama_explanation_raw.jsonl")
    parser.add_argument("--output_dir", type=str, default="predictions/llama3.2-3b-chase-pipeline")
    args = parser.parse_args()

    pipe, tokenizer = load_llama_pipeline(args.model_name_or_path)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    dataset = datasets.Dataset.from_json(args.input_path)
    predictions = {}

    for instance in tqdm(dataset):
        instance_id = instance["id"]
        premise = instance.get("premise")
        hypothesis = instance.get("hypothesis")

        items = []
        for idx, (reason_text, label_code) in enumerate(instance["generated_explanations"]):
            if label_code not in label_map:
                print(f"[Warning] Unknown label code {label_code} in instance {instance_id}")
                continue
            reason_id = f"{instance_id}_{label_code}-{idx}"
            items.append({"reason_id": reason_id, "label": label_map[label_code], "reason": reason_text})

        if not items:
            continue

        messages = build_starting_messages(premise, hypothesis, items)
        # print("starting messages:", messages)

        for it in items:
            reason_id = it["reason_id"]
            q = f"Label: {it['label']}\nReason: {it['reason'].strip()}\nProbability:"
            messages.append({"role": "user", "content": q})

            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            print("prompt:", prompt)

            result = pipe(
                prompt,
                max_new_tokens=16,
                do_sample=False,
                return_full_text=False
            )
            output_text = result[0]["generated_text"]
            print("output_text:",output_text)
            print("-" * 80)
            probability = extract_probability(output_text)
            print(probability)
            print("-" * 80)

            predictions[reason_id] = probability
            messages.append({"role": "assistant", "content": output_text.strip()})

    with open(outdir / "scores.json", "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"Done. Output saved to: {outdir / 'scores.json'}")

if __name__ == "__main__":
    main()
