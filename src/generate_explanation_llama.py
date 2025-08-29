import json
import os
import torch
from transformers import pipeline

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_CACHE"] = "../zlongfei/.cache"

# model_id = "meta-llama/Llama-3.1-8B-Instruct"
model_id = "meta-llama/Llama-3.3-70B-Instruct"


pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

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


def generate_response(messages):
    # outputs = pipe(prompt_str, max_new_tokens=256)
    # response = outputs[0]["generated_text"]
    # print("response:", response)
    # return response.strip()
    out = pipe(messages, max_new_tokens=256)
    # print("out is:", out)
    assistant_answer = out[0]["generated_text"][-1]["content"]
    # print("assistant_answer is:", assistant_answer)
    return assistant_answer.strip()

def process_jsonl(jsonl_path, output_dir):
    relationships = {"E": "true", "N": "undetermined", "C": "false"}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            premise, hypothesis, sample_id = item["context"], item["statement"], item["id"]
            sample_folder = os.path.join(output_dir, sample_id)
            os.makedirs(sample_folder, exist_ok=True)

            for filename, relation in relationships.items():
                messages = build_chat_messages(premise, hypothesis, relation)
                response = generate_response(messages)
                print("response is:", response)
                with open(os.path.join(sample_folder, filename), "w", encoding="utf-8") as out_file:
                    out_file.write(response)

jsonl_path = "../varierr.json"
output_dir = "../llama3.3_70b_generation"
process_jsonl(jsonl_path, output_dir)