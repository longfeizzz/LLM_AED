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

def extract_probability(text):
    matches = re.findall(r"\b(?:0\.\d+|1\.0+)\b", text)
    if matches:
        return float(matches[-1]) 
    return None

def load_llama_pipeline(model_id="meta-llama/Llama-3.2-3B-Instruct"):
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

def build_messages(premise, hypothesis, label, reason_text):
    return [
        {"role": "system", "content": "You are an expert linguistic annotator."},
        {"role": "user", "content":
            "We have collected annotations for an NLI instance together with reasons for the labels. "
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
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--input_path", type=str, default="../ZLF/llama_explanation_raw.jsonl")
    parser.add_argument("--output_dir", type=str, default="predictions/llama3.2-3b")
    args = parser.parse_args()

    pipe, tokenizer = load_llama_pipeline(args.model_name_or_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = datasets.Dataset.from_json(args.input_path)
    predictions = {}

    for instance in tqdm(dataset):
        instance_id = instance['id']
        premise = instance['premise']
        hypothesis = instance['hypothesis']

        for idx, (reason_text, label_code) in enumerate(instance["generated_explanations"]):
            if label_code not in label_map:
                print(f"[Warning] Unknown label code {label_code} in instance {instance_id}")
                continue
            label = label_map[label_code]
            reason_id = f"{instance_id}_{label_code}-{idx}"

            messages = build_messages(premise, hypothesis, label, reason_text)
            # print("messages:", messages)
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            print("prompt:",prompt)

            result = pipe(prompt, max_new_tokens=32)
            print("result:",result)
            output_text = result[0]['generated_text']
            print("output_text:",output_text)
            print("-" * 80)

            probability = extract_probability(output_text)
            print(probability)
            print("-" * 80)
            predictions[reason_id] = probability


    with open(output_dir / "scores.json", "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"Done. Output saved to: {output_dir / 'scores.json'}")

if __name__ == "__main__":
    main()
