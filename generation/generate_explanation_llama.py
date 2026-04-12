import json
import os
import torch
import argparse
from transformers import pipeline
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
    return [sys_msg, user_msg]


def generate_response(pipe, messages):
    out = pipe(messages, max_new_tokens=256)
    assistant_answer = out[0]["generated_text"][-1]["content"]
    return assistant_answer.strip()

def process_jsonl(pipe, jsonl_path, output_dir):
    relationships = {"E": "true", "N": "undetermined", "C": "false"}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Generating Explanations"):
        item = json.loads(line)
        premise, hypothesis, sample_id = item["context"], item["statement"], item["id"]
        sample_folder = os.path.join(output_dir, sample_id)
        os.makedirs(sample_folder, exist_ok=True)

        for filename, relation in relationships.items():
            messages = build_chat_messages(premise, hypothesis, relation)
            response = generate_response(pipe, messages)
            # print("response is:", response)
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
        model_id = args.model_id.split("/")[-1].replace("-Instruct", "")
        output_dir = f"../generation/{model_id}_generation_raw"
    else:
        output_dir = args.output_dir

    print(f"Model: {args.model_id}")
    print(f"Output dir: {output_dir}")

    pipe = pipeline(
        "text-generation",
        model=args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    process_jsonl(pipe, args.jsonl_path, output_dir)