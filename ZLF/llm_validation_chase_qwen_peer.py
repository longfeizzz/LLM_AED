import json
import re
from pathlib import Path
from tqdm import tqdm
import argparse
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import datasets
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

label_map = {"e": "entailment", "n": "neutral", "c": "contradiction"}

def extract_probability(text: str):
    m = re.search(r"\b(?:0(?:\.\d+)?|1(?:\.0+)?)\b", text)
    return float(m.group(0)) if m else None

PROMPT_TEMPLATE = ( 
    "We have collected annotations for an NLI instance together with explanations for the labels. \n"
    "You will first be shown all explanations together so that you understand the overall context, and then your task is to score ONE explanation at a time. "
    "Provide the probability score (0.0 - 1.0) that each explanation makes sense for the label. "
    "Previously scored explanations (together with their scores) will remain in the conversation as additional context. \n"
    "Give ONLY the probability, no other words or explanation. \n"
    "For example: \nProbability: <the probability between 0.0 and 1.0 that the explanation makes sense for the label, without any extra commentary whatsoever; just the probability!> \n"
)


def load_qwen(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    return model, tokenizer

def build_starting_messages(premise, hypothesis, items):
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
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--input_path", type=str, default="/mounts/data/proj/zlongfei/ZLF/qwen_72b_generation_raw.jsonl")
    parser.add_argument("--output_dir", type=str, default="validation/qwen_72b_chase")
    args = parser.parse_args()

    model, tokenizer = load_qwen(args.model_name)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = datasets.Dataset.from_json(args.input_path)
    predictions = {}

    for instance in tqdm(dataset):
        instance_id = instance["id"]
        premise = instance.get("premise")
        hypothesis = instance.get("hypothesis")

        items = []
        
        for idx, (reason_text, label_code, model_id) in enumerate(instance["generated_explanations"]):
        # for idx, (reason_text, label_code) in enumerate(instance["generated_explanations"]):
            if label_code not in label_map:
                print(f"Unknown label code {label_code} in instance {instance_id}")
                continue
            # reason_id = f"{instance_id}_{label_code}-{idx}"
            reason_id = f"{model_id}_{instance_id}_{label_code}-{idx}"
            items.append({"reason_id": reason_id, "label": label_map[label_code], "reason": reason_text})

        if not items:
            continue

        messages = build_starting_messages(premise, hypothesis, items)
        
        for it in items:
            reason_id = it["reason_id"]
            q = f"Label: {it['label']}\nReason: {it['reason'].strip()}\nProbability:"
            messages.append({"role": "user", "content": q})
            
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            print("prompt:", prompt)
            print("-" * 80)

            # prompt = tokenizer.apply_chat_template(
            #     messages, tokenize=False, add_generation_prompt=True
            # )

            # print("-" * 80)
            # print("prompt:", prompt)
            
            model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=16,   
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )

            # result = pipe(
            #     prompt,
            #     max_new_tokens=16,
            #     do_sample=False,
            #     return_full_text=False
            # )

            new_tokens = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            output_text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
            # print("output_text:", output_text)
            # print("-" * 80)

            probability = extract_probability(output_text)
            print(probability)
            print("-" * 80)
            predictions[reason_id] = probability

            messages.append({"role": "assistant", "content": output_text.strip()})

            # print("messages:", messages)
            # print("-" * 80)

            # output_text = result[0]["generated_text"]

    with open(output_dir / "scores.json", "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"Done. Output saved to: {output_dir / 'scores.json'}")

if __name__ == "__main__":
    main()
