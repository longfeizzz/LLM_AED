import os, glob, json, itertools, logging
from collections import Counter
import numpy as np
import spacy
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

ROOT_LLM_DIR = "/Users/phoebeeeee/ongoing/LLM_AED/new_processing/validation_result/llm_vs_human_before"
ROOT_VAR_FILE = "/Users/phoebeeeee/ongoing/LLM_AED/dataset/varierr/varierr.json" 
LOG_PATH = "./llm_varierr_similarity_before.log"
N_GRAMS = [1, 2, 3]

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s | %(message)s"
)
log = logging.getLogger(__name__)


nlp = spacy.load("en_core_web_md")

def spacy_tokens(text):
    doc = nlp(text)
    return [t.text.lower() for t in doc if not t.is_punct and not t.is_space]

def spacy_pos(text):
    doc = nlp(text)
    return [t.pos_ for t in doc]

def get_ngram_counts(tokens, n):
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

def overlap_ratio(ca, cb):
    count_1_in_2 = sum([1 if b in ca else 0 for b in cb])
    count_2_in_1 = sum([1 if a in cb else 0 for a in ca])
    combined_length = len(ca) + len(cb)
    return (
        (count_1_in_2 + count_2_in_1) / combined_length if combined_length > 0 else float("nan")
    )

def lexical_diversity(a_text, b_text, n):
    ca = get_ngram_counts(spacy_tokens(a_text), n)
    cb = get_ngram_counts(spacy_tokens(b_text), n)
    return overlap_ratio(ca, cb)

def syntactic_diversity(a_text, b_text, n):
    ca = get_ngram_counts(spacy_pos(a_text), n)
    cb = get_ngram_counts(spacy_pos(b_text), n)
    return overlap_ratio(ca, cb)

LABEL_MAP = {"e": "entailment", "n": "neutral", "c": "contradiction"}
LABELS = ["entailment", "neutral", "contradiction"]


def load_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def load_records_from_folder(folder_path):
    all_data = []
    for fname in sorted(os.listdir(folder_path)):
        if fname.endswith(".jsonl"):
            fpath = os.path.join(folder_path, fname)
            recs = load_records(fpath)
            all_data.append((fname, recs))
    return all_data


def extract_llm_buckets(rec):
    buckets = {lbl: [] for lbl in LABELS}
    ge = rec.get("generated_explanations") or []
    for item in ge:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            print(f"[WARN] Invalid generated_explanations item: {item}")
            continue
        reason, code = item
        if not isinstance(reason, str) or not reason.strip():
            print(f"[WARN] Invalid reason: {reason}")
            continue
        if not isinstance(code, str):
            print(f"[WARN] Invalid label code: {code}")
            continue
        lbl = LABEL_MAP.get(code.strip().lower())
        if lbl:
            buckets[lbl].append(reason.strip())

    return buckets

    #     reason, code, validation = item
    #     if not isinstance(reason, str) or not reason.strip():
    #         print(f"[WARN] Invalid reason: {reason}")
    #         continue
    #     if not isinstance(code, str):
    #         print(f"[WARN] Invalid label code: {code}")
    #         continue 
    #     if not isinstance(validation, str) or validation.strip().lower() != "validated":
    #         # print(f"[WARN] Invalid validation status: {validation}")
    #         continue
    #     lbl = LABEL_MAP.get(code.strip().lower())
    #     if lbl:
    #         buckets[lbl].append(reason.strip())
    # return buckets

def extract_varierr_reasons(rec: dict, key: str):
    lst = rec.get(key, []) or []
    reasons = []
    for item in lst:
        if isinstance(item, dict):
            r = item.get("reason")
            if isinstance(r, str) and r.strip():
                reasons.append(r.strip())
        elif isinstance(item, str) and item.strip():
            reasons.append(item.strip())
    return reasons

def extract_varierr_buckets(rec):
    return {lbl: extract_varierr_reasons(rec, lbl) for lbl in LABELS}


device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("sentence-transformers/all-distilroberta-v1")
model = AutoModel.from_pretrained("sentence-transformers/all-distilroberta-v1").to(device).eval()

def sentence_embedding(text):
    encoded = tok([text], padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**encoded, output_hidden_states=True)
    token_embeddings = out.hidden_states[-1]
    attn_mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
    emb = torch.sum(token_embeddings * attn_mask, 1) / torch.clamp(attn_mask.sum(1), min=1e-9)
    return F.normalize(emb, p=2, dim=1).cpu().numpy()[0]


def cosine_similarity(e1, e2):
    return float(np.dot(e1, e2))  
    # return -cosine(e1, e2) + 1

def euclidean_similarity(e1, e2):
    return 1.0 / (1.0 + np.linalg.norm(e1 - e2))


def main():
    llm_sets = load_records_from_folder(ROOT_LLM_DIR)
    var_records = load_records(ROOT_VAR_FILE)

    for fname, llm_records in llm_sets:
        if len(llm_records) != len(var_records):
            log.warning(f"[{fname}] 条目数不一致: LLM={len(llm_records)}, VariErr={len(var_records)}")
            continue

        totals = {
            "lexical": {n: [] for n in N_GRAMS},
            "syntactic": {n: [] for n in N_GRAMS},
            "cosine": [],
            "euclidean": []
        }
        log.info(f"Processing file: {fname} with {len(llm_records)} records")


        for rec_llm, rec_var in zip(llm_records, var_records):
            llm_buckets = extract_llm_buckets(rec_llm)
            var_buckets = extract_varierr_buckets(rec_var)
            for lbl in LABELS:
                llm_reasons = llm_buckets.get(lbl, [])
                var_reasons = var_buckets.get(lbl, [])
                if not llm_reasons or not var_reasons:
                    continue
                
                llm_embeds = [sentence_embedding(r) for r in llm_reasons]
                var_embeds = [sentence_embedding(r) for r in var_reasons]

                for i in range(len(llm_reasons)):
                    for j in range(len(var_reasons)):
                        a, b = llm_reasons[i], var_reasons[j]
                        for n in N_GRAMS:
                            totals["lexical"][n].append(lexical_diversity(a, b, n))
                            totals["syntactic"][n].append(syntactic_diversity(a, b, n))

                        cos = cosine_similarity(llm_embeds[i], var_embeds[j])
                        euc = euclidean_similarity(llm_embeds[i], var_embeds[j])
                        totals["cosine"].append(cos)
                        totals["euclidean"].append(euc)

        log.info("=== Global Average Results ===")
        for n in N_GRAMS:
            vals = totals["lexical"][n]
            log.info(f"Lexical {n}-gram: {np.mean(vals):.3f}" if vals else f"Lexical {n}-gram: None")
        for n in N_GRAMS:
            vals = totals["syntactic"][n]
            log.info(f"Syntactic {n}-gram: {np.mean(vals):.3f}" if vals else f"Syntactic {n}-gram: None")
        if totals["cosine"]:
            log.info(f"Semantic cosine similarity: {np.mean(totals['cosine']):.3f}")
            log.info(f"Semantic euclidean similarity: {np.mean(totals['euclidean']):.3f}")
        else:
            log.info("Semantic: None")

        print(f"Done. Log saved to: {os.path.abspath(LOG_PATH)}")

if __name__ == "__main__":
    main()