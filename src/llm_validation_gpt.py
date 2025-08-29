import json
from pathlib import Path
import argparse
from tqdm import tqdm

from sglang import function, system, user, assistant, gen, set_default_backend, OpenAI
import datasets

PROMPT_TEMPLATE = f"We have collected annotations for an NLI instance together with reasons for the labels. Your task is to judge whether the reasons make sense for the label. Provide the probability (0.0 - 1.0) that the reason makes sense for the label. Give ONLY the probability, no other words or explanation. For example:\n\nProbability: <the probability between 0.0 and 1.0 that the reason makes sense for the label, without any extra commentary whatsoever; just the probability!>."

dataset = datasets.Dataset.from_json("../gpt-4.1/gpt_explanation_raw.jsonl")

label_map = {"e": "entailment", "n": "neutral", "c": "contradiction"}

@function
def score_single_reason(s, premise, hypothesis, label, reason_id, reason_text):
    s += system("You are an expert linguistic annotator.")
    s += user(PROMPT_TEMPLATE)
    s += user(f"Premise: {premise}\nHypothesis: {hypothesis}\n\nReason for label {label}: {reason_text.strip()}")
    s += user("Probability:")
    s += assistant(gen(reason_id, max_tokens=16)) 
    print(f"Prompt: {s}")

parser = argparse.ArgumentParser()
parser.add_argument("model_name", type=str)
args = parser.parse_args()

output_dir = Path(f"../gpt-4.1/test/{args.model_name}")
output_dir.mkdir(exist_ok=True, parents=True)
set_default_backend(OpenAI(model_name=args.model_name))

predictions = {}

for instance in tqdm(dataset, total=len(dataset), desc="Scoring instances", dynamic_ncols=True):
    premise = instance["premise"]
    hypothesis = instance["hypothesis"]
    instance_id = instance["id"]

    for idx, (reason_text, label_code) in enumerate(instance["generated_explanations"]):
        if label_code not in label_map:
            print(f" {label_code} not in label map.")
            continue
        label = label_map[label_code]
        reason_id = f"{instance_id}_{label_code}-{idx}"
        cache_file = (output_dir / reason_id).with_suffix(".json")

        if cache_file.exists():
            with open(cache_file, "r") as f:
                cached = json.load(f)
                predictions[reason_id] = cached["probability"]
                continue

        state = score_single_reason.run(
            premise=premise,
            hypothesis=hypothesis,
            label=label,
            reason_id=reason_id,
            reason_text=reason_text
        )

        probability = state[reason_id].strip()
        predictions[reason_id] = probability
        cacheable = {"messages": state.messages(), "probability": probability}
        with open(cache_file, "w") as f:
            json.dump(cacheable, f, indent=2)

with open(output_dir / "scores.json", "w") as f:
    json.dump(predictions, f, indent=2)
