import json
import random
import os
import yaml
from collections import Counter
from datasets import load_dataset

# ── Load settings from config.yaml ──────────────────────────
with open("model/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

TRAIN_SAMPLES = cfg["train_samples"]   # 80000
VAL_SAMPLES   = cfg["val_samples"]     # 5000
EVAL_SAMPLES  = cfg["eval_samples"]    # 2000
SEED          = 42                     # fixed seed for reproducibility

# ── Download MS MARCO v1.1 from HuggingFace ────────────────
print("⏳ Downloading MS MARCO v1.1 ...")
dataset = load_dataset("microsoft/ms_marco", "v1.1", split="train")
print(f"✅ Loaded {len(dataset)} examples from MS MARCO train split")

# ── Extract (query, passage, label) triples ─────────────────
print("⏳ Extracting query-passage-label triples ...")
positives = []
negatives = []

for example in dataset:
    query    = example["query"]
    passages = example["passages"]

    for text, is_selected in zip(passages["passage_text"],
                                  passages["is_selected"]):
        triple = {"query": query, "passage": text, "label": int(is_selected)}

        if is_selected == 1:
            positives.append(triple)
        else:
            negatives.append(triple)

print(f"✅ Extracted {len(positives)} positives, {len(negatives)} negatives")

# ── Shuffle both pools with a fixed seed ─────────────────────
random.seed(SEED)
random.shuffle(positives)
random.shuffle(negatives)

# ── Helper: carve a balanced slice from both pools ───────────
def make_split(pos_list, neg_list, total, start_pos, start_neg):
    """
    Take total/2 positives and total/2 negatives starting at the
    given offsets.  Returns the combined list (shuffled) and the
    new offsets so the next split starts where this one left off.
    """
    half = total // 2
    split = pos_list[start_pos : start_pos + half] \
          + neg_list[start_neg : start_neg + half]
    random.shuffle(split)          # mix pos and neg together
    return split, start_pos + half, start_neg + half

# ── Carve the three splits (no overlap between them) ─────────
pos_offset = 0
neg_offset = 0

train, pos_offset, neg_offset = make_split(positives, negatives,
                                            TRAIN_SAMPLES,
                                            pos_offset, neg_offset)

val,   pos_offset, neg_offset = make_split(positives, negatives,
                                            VAL_SAMPLES,
                                            pos_offset, neg_offset)

evl,   pos_offset, neg_offset = make_split(positives, negatives,
                                            EVAL_SAMPLES,
                                            pos_offset, neg_offset)

# ── Save each split as a JSONL file ──────────────────────────
def save_jsonl(data, filepath):
    """Write a list of dicts to a .jsonl file (one JSON object per line)."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

splits = {
    "train": train,
    "val":   val,
    "eval":  evl,
}

for name, data in splits.items():
    path = f"data/splits/{name}.jsonl"
    save_jsonl(data, path)

    labels = Counter(item["label"] for item in data)
    print(f"\n📁 {path}")
    print(f"   Total:    {len(data)}")
    print(f"   Label 1 (relevant):     {labels[1]}")
    print(f"   Label 0 (irrelevant):   {labels[0]}")

print("\n🎉 Done! All splits saved to data/splits/")