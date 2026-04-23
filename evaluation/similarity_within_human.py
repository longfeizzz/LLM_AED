import json, re, itertools
from collections import Counter
import numpy as np
import spacy
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from scipy.spatial.distance import cosine, euclidean

ROOT_FILE = "../dataset/varierr.json"
N_GRAMS = [1, 2, 3]

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

def load_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def extract_reasons(rec: dict, key: str):
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
    records = load_records(ROOT_FILE)

    totals = {
        "lexical": {n: [] for n in N_GRAMS},
        "syntactic": {n: [] for n in N_GRAMS},
        "cosine": [],
        "euclidean": []
    }

    for rec in records:
        for lbl in ["entailment", "neutral", "contradiction"]:
            reasons = extract_reasons(rec, lbl)
            # print(f"Processing {lbl} with {len(reasons)} reasons...")
            if len(reasons) < 2:
                continue

            # token_lists = [spacy_tokens(r) for r in reasons]
            embeds = [sentence_embedding(r) for r in reasons]

            for i, j in itertools.combinations(range(len(reasons)), 2):
                # lexical
                for n in N_GRAMS:
                    totals["lexical"][n].append(
                        lexical_diversity(reasons[i], reasons[j], n)
                    )
                # syntactic
                for n in N_GRAMS:
                    totals["syntactic"][n].append(
                        syntactic_diversity(reasons[i], reasons[j], n)
                    )
                # semantic similarities
                cos = cosine_similarity(embeds[i], embeds[j])
                euc = euclidean_similarity(embeds[i], embeds[j])
                totals["cosine"].append(cos)
                totals["euclidean"].append(euc)

    print("=== Global Average Results ===")
    print("Lexical:")
    for n in N_GRAMS:
        vals = totals["lexical"][n]
        print(f"  {n}-gram: {np.mean(vals):.3f}" if vals else f"  {n}-gram: None")

    print("Syntactic (POS):")
    for n in N_GRAMS:
        vals = totals["syntactic"][n]
        print(f"  {n}-gram: {np.mean(vals):.3f}" if vals else f"  {n}-gram: None")

    if totals["cosine"]:
        print(f"Semantic cosine similarity: {np.mean(totals['cosine']):.3f}")
        print(f"Semantic euclidean similarity: {np.mean(totals['euclidean']):.3f}")
    else:
        print("Semantic: None")

if __name__ == "__main__":
    main()
