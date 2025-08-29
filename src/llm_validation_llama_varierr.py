import json
import re
from pathlib import Path
from tqdm import tqdm
import argparse
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

label_map = {"entailment": "entailment", "neutral": "neutral", "contradiction": "contradiction"}

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

def build_messages(context, statement, label, reason_text):
    return [
        {"role": "system", "content": "You are an expert linguistic annotator."},
        {"role": "user", "content":
            "We have collected annotations for a NLI instance together with reasons for the labels. "
            "Your task is to judge whether the reasons make sense for the label. "
            "Provide the probability (0.0 - 1.0) that the reason makes sense for the label. "
            "Give ONLY the probability, no other words or explanation. \n\n"
            "For example: \nProbability: <the probability between 0.0 and 1.0 that the reason makes sense for the label, without any extra commentary whatsoever; just the probability!> \n\n"
            f"Context: {context}\nStatement: {statement}\n"
            f"Reason for label {label}: {reason_text.strip()}\n"
            f"Probability:"
        }
    ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="predictions/llama3.2-3b")
    args = parser.parse_args()

    pipe, tokenizer = load_llama_pipeline(args.model_name_or_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = {}

    with open(args.input_path, "r") as f:
        for line in tqdm(f):
            instance = json.loads(line)
            instance_id = instance["id"]
            context = instance["context"]
            statement = instance["statement"]

            for label in ["entailment", "neutral", "contradiction"]:
                if label not in instance or not isinstance(instance[label], list):
                    continue
                label_code = label[0]  
                for idx, reason_entry in enumerate(instance[label]):
                    reason_id = f"{instance_id}-{label_code}-{idx}"
                    reason_text = reason_entry["reason"]

                    messages = build_messages(context, statement, label, reason_text)
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    print("prompt:",prompt)
                    try:
                        result = pipe(prompt, max_new_tokens=32)
                        print("result:",result)
                        output_text = result[0]['generated_text']
                        print("-" * 80)

                        probability = extract_probability(output_text)
                        print(probability)
                        print("-" * 80)
                        predictions[reason_id] = probability
                    except Exception as e:
                        print(f"Failed to process {reason_id}: {e}")
                        predictions[reason_id] = None


    with open(output_dir / "scores.json", "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"Done. Output saved to: {output_dir / 'scores.json'}")

if __name__ == "__main__":
    main()
