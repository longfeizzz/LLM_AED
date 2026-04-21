import json
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def build_chat_messages(premise, hypothesis, relationship):
    sys_msg = {
        "role": "system",
        "content": (
            "You are an expert in Natural Language Inference (NLI). "
            f"List every distinct explanation for why the statement is {relationship} given the context below without introductory phrases. "
            f"If you think the relationship is false given the context, you can choose not to provide explanations.  "
            f"Do not repeat or paraphrase the same idea in different words. End your answer after all reasonable distinct explanations are listed.\n"
            f"Format your answer as a numbered list (e.g., 1., 2., 3.).\n\n"
        ),
    }
    user_msg = {
        "role": "user",
        "content": f"Context: {premise}\nStatement: {hypothesis}",
    }

    messages = [sys_msg,user_msg]

    return messages


def generate_response(model, tokenizer, messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    outputs = model.generate(
        **model_inputs,
        max_new_tokens=256
    )

    gen_ids = outputs[0][model_inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return response.strip()

def process_jsonl(model, tokenizer, jsonl_path, output_dir):
    relationships = {"E_0.txt": "true", "N_0.txt": "undetermined", "C_0.txt": "false"}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Generating Explanations"):
        item = json.loads(line)
        premise, hypothesis, sample_id = item["context"], item["statement"], item["id"]
        sample_folder = os.path.join(output_dir, sample_id)
        os.makedirs(sample_folder, exist_ok=True)

        for filename, relation in relationships.items():
            messages = build_chat_messages(premise, hypothesis, relation)
            response = generate_response(model, tokenizer, messages)
            with open(os.path.join(sample_folder, filename), "w", encoding="utf-8") as out_file:
                out_file.write(response)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, type=str, default=None)
    parser.add_argument("--jsonl_path", type=str, default="../dataset/varierr.json")
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.output_dir is None:
        model_short = args.model_name.split("/")[-1].replace("-Instruct", "")
        output_dir = f"../generation/{model_short}_generation_raw"
    else:
        output_dir = args.output_dir

    print(f"Model: {args.model_name}")
    print(f"Output dir: {output_dir}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    process_jsonl(model, tokenizer, args.jsonl_path, output_dir)
