import json
import re
from pathlib import Path
from tqdm import tqdm
import argparse
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import datasets


label_map = {"e": "entailment", "n": "neutral", "c": "contradiction"}


def extract_probability(text):
    matches = re.findall(r"\b(?:0\.\d+|1\.0+)\b", text)
    return float(matches[-1]) if matches else None


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
        result = model_dict["pipe"](prompt, max_new_tokens=32, return_full_text=False, do_sample=False)
        return result[0]["generated_text"]

    elif model_type == "qwen":
        model = model_dict["model"]
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        gen_ids = outputs[0][len(inputs.input_ids[0]):]
        return tokenizer.decode(gen_ids, skip_special_tokens=True)


def build_messages(premise, hypothesis, label, reason_text):
    return [
        {"role": "system", "content": "You are an expert linguistic annotator."},
        {"role": "user", "content": (
            "We have collected annotations for a NLI instance together with reasons for the labels. "
            "Your task is to judge whether the reasons make sense for the label. "
            "Provide the probability (0.0 - 1.0) that the reason makes sense for the label. "
            "Give ONLY the probability, no other words or explanation.\n\n"
            "For example:\nProbability: <the probability between 0.0 and 1.0>\n\n"
            f"Context: {premise}\nStatement: {hypothesis}\n"
            f"Reason for label {label}: {reason_text.strip()}\n"
            "Probability:"
        )}
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True, choices=["llama", "qwen"])
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    model_short = args.model_name_or_path.split("/")[-1].replace("-Instruct", "")
    input_path = args.input_path or f"../processing/{model_short}_generation_raw.jsonl"
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"../validation/validation_results/one_expl/{model_short}")

    print(f"Model: {args.model_name_or_path}")
    print(f"Input: {input_path}")
    print(f"Output dir: {output_dir}")

    model_dict = load_model(args.model_name_or_path, args.model_type)
    tokenizer = model_dict["tokenizer"]
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = datasets.Dataset.from_json(input_path)
    predictions = {}

    for instance in tqdm(dataset):
        instance_id = instance["id"]
        premise = instance["premise"]
        hypothesis = instance["hypothesis"]

        for idx, (reason_text, label_code) in enumerate(instance["generated_explanations"]):
            if label_code not in label_map:
                print(f"Unknown label code {label_code} in instance {instance_id}")
                continue
            label = label_map[label_code]
            reason_id = f"{instance_id}_{label_code}-{idx}"

            messages = build_messages(premise, hypothesis, label, reason_text)
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            output_text = generate(args.model_type, model_dict, prompt)

            probability = extract_probability(output_text)
            print(f"{reason_id}: {probability}")
            predictions[reason_id] = probability

    with open(output_dir / "scores.json", "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"Done. Output saved to: {output_dir / 'scores.json'}")


if __name__ == "__main__":
    main()