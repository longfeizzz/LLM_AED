import json
import re
from pathlib import Path
from tqdm import tqdm
import argparse
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import datasets
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

label_map = {"e": "entailment", "n": "neutral", "c": "contradiction"}

def extract_probability(text):
    matches = re.findall(r"\b(?:0\.\d+|1\.0+)\b", text)
    if matches:
        return float(matches[-1]) 
    return None

def load_qwen(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    return model, tokenizer


def build_messages(premise, hypothesis, label, reason_text):
    return [
        {"role": "system", "content": "You are an expert linguistic annotator."},
        {"role": "user", "content":
            "We have collected annotations for a NLI instance together with reasons for the labels. "
            "Your task is to judge whether the reasons make sense for the label. "
            "Provide the probability (0.0 - 1.0) that the reason makes sense for the label. "
            "Give ONLY the probability, no other words or explanation. \n\n"
            # "Reason: <The verbatim copy of the reason>\n"
            "For example: \nProbability: <the probability between 0.0 and 1.0 that the reason makes sense for the label, without any extra commentary whatsoever; just the probability!> \n\n"
            f"Context: {premise}\nStatement: {hypothesis}\n"
            f"Reason for label {label}: {reason_text.strip()}\n"
            # f"Reason: {reason_text.strip()}\n"
            f"Probability:"
        }
    ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    model, tokenizer = load_qwen(args.model_name)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = datasets.Dataset.from_json(args.input_path)
    predictions = {}

    for instance in tqdm(dataset):
        instance_id = instance['id']
        premise = instance['premise']
        hypothesis = instance['hypothesis']

        for idx, (reason_text, label_code, model_id) in enumerate(instance["generated_explanations"]):
        # for idx, (reason_text, label_code) in enumerate(instance["generated_explanations"]):
            if label_code not in label_map:
                print(f"Unknown label code {label_code} in instance {instance_id}")
                continue
            label = label_map[label_code]
            # reason_id = f"{instance_id}_{label_code}-{idx}"
            reason_id = f"{model_id}_{instance_id}_{label_code}-{idx}"

            messages = build_messages(premise, hypothesis, label, reason_text)
            # print("messages:", messages)
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
            print("prompt:",prompt)
            print("-" * 80)

            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id
            )

            gen_ids = outputs[0][len(inputs.input_ids[0]):]
            output_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            # print("output_text:",output_text)
            # print("-" * 80)

            probability = extract_probability(output_text)
            print(probability)
            print("-" * 80)
            predictions[reason_id] = probability


    with open(output_dir / "scores.json", "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"Done. Output saved to: {output_dir / 'scores.json'}")

if __name__ == "__main__":
    main()