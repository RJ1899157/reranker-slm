"""
baselines.py
------------
Evaluate BM25 and BGE-base-en-v1.5 baselines on MS MARCO's
NATURAL query-passage groups.  Unlike random negatives, these
are "hard negatives" — passages retrieved by a search engine
for the same query, so they're topically related but mostly
not relevant.  This makes the ranking task realistic.

Also saves the evaluation groups to data/splits/eval_golden.jsonl
so we can later evaluate our trained reranker on the same data.
"""

import json
import os
import random
import numpy as np
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics import ndcg_score
from tqdm import tqdm

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SEED               = 42
OUTPUT_PATH        = "eval/results/baselines.json"
GOLDEN_PATH        = "data/splits/eval_golden.jsonl"
BGE_MODEL_NAME     = "BAAI/bge-base-en-v1.5"
NUM_EVAL_QUERIES   = 500    # sample this many queries for evaluation

random.seed(SEED)
np.random.seed(SEED)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1. LOAD MS MARCO VALIDATION SPLIT (natural hard negatives)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#  We use the VALIDATION split (not train) to avoid data leakage.
#  Our training data came from the train split.
#
#  Each MS MARCO example has:
#    - query: the search query
#    - passages.passage_text: ~10 candidate passages
#    - passages.is_selected: 1 if relevant, 0 if not
#
#  These candidates were retrieved by Bing for the same query,
#  so the negatives are HARD — topically related but not the
#  right answer.

print("⏳ Loading MS MARCO v1.1 validation split ...")
dataset = load_dataset("microsoft/ms_marco", "v1.1", split="validation")
print(f"✅ Loaded {len(dataset)} examples")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  2. BUILD NATURAL QUERY GROUPS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("⏳ Building query groups ...")
eval_groups = []

for example in dataset:
    query   = example["query"]
    texts   = example["passages"]["passage_text"]
    labels  = [int(s) for s in example["passages"]["is_selected"]]

    num_pos = sum(labels)
    num_neg = len(labels) - num_pos

    # Keep only queries with at least 1 positive AND 1 negative
    # and at least 3 passages total (for meaningful ranking)
    if num_pos >= 1 and num_neg >= 1 and len(texts) >= 3:
        eval_groups.append({
            "query": query,
            "passages": texts,
            "labels": labels,
        })

# Shuffle and sample a fixed subset for speed
random.shuffle(eval_groups)
eval_groups = eval_groups[:NUM_EVAL_QUERIES]

avg_cand = np.mean([len(g["passages"]) for g in eval_groups])
avg_pos  = np.mean([sum(g["labels"]) for g in eval_groups])
avg_neg  = np.mean([len(g["labels"]) - sum(g["labels"]) for g in eval_groups])

print(f"📊 {len(eval_groups)} query groups selected")
print(f"   Avg candidates per query:  {avg_cand:.1f}")
print(f"   Avg positives per query:   {avg_pos:.1f}")
print(f"   Avg negatives per query:   {avg_neg:.1f}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  3. SAVE GOLDEN EVAL FILE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#  We save these exact groups so our trained reranker is
#  evaluated on the SAME queries and passages — fair comparison.

os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
with open(GOLDEN_PATH, "w") as f:
    for group in eval_groups:
        f.write(json.dumps(group) + "\n")
print(f"💾 Golden eval saved to {GOLDEN_PATH}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  4. METRIC FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def mrr_at_k(labels, scores, k=10):
    """
    Reciprocal Rank @ K for a single query.
    Sorts by score descending, finds the first relevant result
    in the top K, returns 1/rank.  Returns 0 if none found.
    """
    ranked_indices = np.argsort(scores)[::-1][:k]
    for rank, idx in enumerate(ranked_indices, start=1):
        if labels[idx] == 1:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(labels, scores, k=10):
    """
    NDCG @ K for a single query.
    Measures how well the entire top-K ranking matches ideal ordering.
    """
    labels = np.array(labels, dtype=float)
    scores = np.array(scores, dtype=float)
    if labels.sum() == 0:
        return 0.0
    return ndcg_score([labels], [scores], k=k)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  5. BM25 BASELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#  BM25 scores by word overlap.  With hard negatives (passages
#  about the same topic), BM25 struggles because many passages
#  share the same keywords as the query.


def bm25_score(query, passages):
    """Score each passage against the query using BM25."""
    tokenized_passages = [p.lower().split() for p in passages]
    bm25 = BM25Okapi(tokenized_passages)
    tokenized_query = query.lower().split()
    return bm25.get_scores(tokenized_query)


print("\n⏳ Running BM25 baseline ...")
bm25_mrr_list = []
bm25_ndcg_list = []

for group in tqdm(eval_groups, desc="BM25"):
    scores = bm25_score(group["query"], group["passages"])
    bm25_mrr_list.append(mrr_at_k(group["labels"], scores, k=10))
    bm25_ndcg_list.append(ndcg_at_k(group["labels"], scores, k=10))

bm25_mrr  = float(np.mean(bm25_mrr_list))
bm25_ndcg = float(np.mean(bm25_ndcg_list))
print(f"✅ BM25  →  MRR@10: {bm25_mrr:.4f}  |  NDCG@10: {bm25_ndcg:.4f}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  6. BGE BASELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#  BGE is a bi-encoder: it encodes query and passage into
#  separate vectors and scores by cosine similarity.
#  Better than BM25 on hard negatives because it captures
#  semantic meaning, but weaker than a cross-encoder because
#  it can't jointly attend across query + passage.

print(f"\n⏳ Loading BGE model ({BGE_MODEL_NAME}) ...")
bge_model = SentenceTransformer(BGE_MODEL_NAME)
print("✅ BGE model loaded")

print("⏳ Running BGE baseline ...")
bge_mrr_list = []
bge_ndcg_list = []

for group in tqdm(eval_groups, desc="BGE "):
    query    = group["query"]
    passages = group["passages"]

    q_emb  = bge_model.encode([query], normalize_embeddings=True)
    p_embs = bge_model.encode(passages, normalize_embeddings=True)
    scores = (p_embs @ q_emb.T).flatten()

    bge_mrr_list.append(mrr_at_k(group["labels"], scores, k=10))
    bge_ndcg_list.append(ndcg_at_k(group["labels"], scores, k=10))

bge_mrr  = float(np.mean(bge_mrr_list))
bge_ndcg = float(np.mean(bge_ndcg_list))
print(f"✅ BGE   →  MRR@10: {bge_mrr:.4f}  |  NDCG@10: {bge_ndcg:.4f}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  7. SAVE RESULTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

results = {
    "eval_source": "MS MARCO v1.1 validation split (natural hard negatives)",
    "num_queries": len(eval_groups),
    "avg_candidates_per_query": round(avg_cand, 1),
    "baselines": {
        "BM25": {
            "MRR@10": round(bm25_mrr, 4),
            "NDCG@10": round(bm25_ndcg, 4),
        },
        "BGE-base-en-v1.5": {
            "MRR@10": round(bge_mrr, 4),
            "NDCG@10": round(bge_ndcg, 4),
        },
    },
    "reference": "Nogueira & Cho 2019 (arXiv:1901.04085) — BERT-large MRR@10 ~0.365 on MS MARCO 1000-candidate",
}

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

# ── Final comparison ──────────────────────────────────────────
print(f"\n📊 Results saved to {OUTPUT_PATH}")
print(f"📊 Golden eval saved to {GOLDEN_PATH}")
print("\n┌─────────────────────────────────────────────────┐")
print("│     Baseline Comparison (hard negatives)        │")
print("├──────────────────────┬──────────┬───────────────┤")
print("│ Method               │  MRR@10  │   NDCG@10     │")
print("├──────────────────────┼──────────┼───────────────┤")
print(f"│ BM25                 │  {bm25_mrr:.4f}  │   {bm25_ndcg:.4f}       │")
print(f"│ BGE-base-en-v1.5     │  {bge_mrr:.4f}  │   {bge_ndcg:.4f}       │")
print("├──────────────────────┼──────────┼───────────────┤")
print("│ 🎯 Target: beat BGE with our fine-tuned Qwen    │")
print("└─────────────────────────────────────────────────┘")