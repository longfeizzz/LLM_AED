import json
import os
from pathlib import Path
from tqdm import tqdm

from sglang import function, user, system, assistant, gen, OpenAI, set_default_backend

set_default_backend(OpenAI(model_name="gpt-4.1"))


# def build_chat_messages(s, premise, hypothesis, relationship):
#     sys_msg = {
#         "role": "system",
#         "content": (
#             "You are an expert in Natural Language Inference (NLI). "
#             f"List every distinct explanation for why the statement is {relationship} given the context below without introductory phrases. "
#             f"If you think the relationship is false given the context, you can choose not to provide explanations.  "
#             f"Do not repeat or paraphrase the same idea in different words. End your answer after all reasonable distinct explanations are listed.\n"
#             f"Format your answer as a numbered list (e.g., 1., 2., 3.).\n\n"
#         ),
#     }
#     user_msg = {
#         "role": "user",
#         "content": f"Context: {premise}\nStatement: {hypothesis}",
#     }
    # return [sys_msg, user_msg]
@function
def generate_reason_list(s, premise, hypothesis, label):
    s += system(
        "You are an expert in Natural Language Inference (NLI). "
        f"List every distinct explanation for why the statement is {label} given the context below without introductory phrases. "
        f"If you think the relationship is false given the context, you can choose not to provide explanations. "
        f"Do not repeat or paraphrase the same idea in different words. "
        f"End your answer after all reasonable distinct explanations are listed.\n"
        f"Format your answer as a numbered list (e.g., 1., 2., 3.)."
    )
    s += user(f"Context: {premise}\nStatement: {hypothesis}")
    s += assistant(gen("response", max_tokens=256))




def process_jsonl(jsonl_path, output_dir):
    relationships = {"E": "true", "N": "undetermined", "C": "false"}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            item = json.loads(line)
            premise, hypothesis, sample_id = item["context"], item["statement"], item["id"]
            sample_folder = Path(output_dir) / sample_id
            sample_folder.mkdir(parents=True, exist_ok=True)

            for filename, label in relationships.items():
                state = generate_reason_list.run(premise, hypothesis, label)
                print(state)
                result = state["response"].strip()
                # print(f"[{sample_id} - {filename}] → {result[:60]}...")
                with open(sample_folder / filename, "w", encoding="utf-8") as f_out:
                    f_out.write(result)

jsonl_path = "../varierr.json"
output_dir = "../gpt-4.1/gpt_4.1_generation_raw"
process_jsonl(jsonl_path, output_dir)