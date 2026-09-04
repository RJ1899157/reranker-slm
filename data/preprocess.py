"""
preprocess.py
-------------
Read JSONL splits, format into prompt templates, tokenize with
the Qwen tokenizer, and return HuggingFace Dataset objects.
"""

import json
import yaml
from datasets import Dataset
from transformers import AutoTokenizer

# ── Load settings from config.yaml ───────────────────────────
with open("model/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

MODEL_NAME = cfg["model_name"]       # Qwen/Qwen2.5-0.5B-Instruct
MAX_LENGTH = cfg["max_length"]       # 512

# ── Load the tokenizer ──────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Qwen's tokenizer has no default pad token — set it to eos_token
# so padding works correctly during batching
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ── Read a JSONL file into a list of dicts ───────────────────
def load_jsonl(filepath):
    """Read a .jsonl file and return a list of dicts."""
    with open(filepath, "r") as f:
        return [json.loads(line) for line in f]


# ── Format a single record into our prompt template ──────────
LABEL_MAP = {1: "relevant", 0: "irrelevant"}

def format_example(record):
    """
    Convert a dict with {query, passage, label} into the
    prompt-completion format the model will learn to generate.

    Input record:
        {"query": "what is gravity", "passage": "Gravity is ...", "label": 1}

    Output:
        {
          "text": "[QUERY]: what is gravity\n[PASSAGE]: Gravity is ...\n
                   Is this passage relevant...?\nrelevant",
          "label": 1
        }
    """
    prompt = (
        f"[QUERY]: {record['query']}\n"
        f"[PASSAGE]: {record['passage']}\n"
        f"Is this passage relevant to the query? "
        f"Answer with relevant or irrelevant."
    )
    answer = LABEL_MAP[record["label"]]

    return {
        "text": f"{prompt}\n{answer}",   # full sequence the model learns
        "label": record["label"],         # keep numeric label for eval metrics
    }


# ── Tokenize a HuggingFace Dataset ──────────────────────────
def tokenize(dataset):
    """
    Tokenize the 'text' field of every example in the dataset.
    Returns a new dataset with columns: input_ids, attention_mask, label.
    """
    def _tokenize_fn(batch):
        tokens = tokenizer(
            batch["text"],
            max_length=MAX_LENGTH,
            truncation=True,        # chop if longer than 512 tokens
            padding="max_length",   # pad shorter sequences to exactly 512
        )
        tokens["label"] = batch["label"]
        return tokens

    # batched=True → process many examples at once (much faster)
    # remove_columns → drop the raw "text" column, keep only numeric tensors
    return dataset.map(_tokenize_fn, batched=True, remove_columns=["text"])


# ── Main: load → format → tokenize → return datasets ────────
def prepare_datasets():
    """
    Full pipeline: read JSONL splits, format prompts, tokenize.
    Returns a dict of HuggingFace Datasets ready for training.
    """
    split_files = {
        "train": "data/splits/train.jsonl",
        "val":   "data/splits/val.jsonl",
        "eval":  "data/splits/eval.jsonl",
    }

    datasets = {}
    for name, path in split_files.items():
        # 1. Load raw JSONL
        raw = load_jsonl(path)
        print(f"📂 Loaded {len(raw)} records from {path}")

        # 2. Format into prompt template
        formatted = [format_example(r) for r in raw]

        # 3. Convert to HuggingFace Dataset
        hf_dataset = Dataset.from_list(formatted)

        # 4. Tokenize
        tokenized = tokenize(hf_dataset)
        datasets[name] = tokenized

        print(f"   ✅ Tokenized → columns: {tokenized.column_names}")
        print(f"   Example input_ids length: {len(tokenized[0]['input_ids'])}")

    return datasets


# ── Run as standalone script to verify everything works ──────
if __name__ == "__main__":
    ds = prepare_datasets()

    print("\n── Summary ────────────────────────────────────────")
    for name, d in ds.items():
        print(f"  {name}: {len(d)} examples, columns={d.column_names}")

    # Show one decoded example so you can visually verify the format
    print("\n── Sample (decoded) ───────────────────────────────")
    sample_ids = ds["train"][0]["input_ids"]
    # Strip padding (zeros) for a clean printout
    sample_ids_clean = [t for t in sample_ids if t != tokenizer.pad_token_id]
    print(tokenizer.decode(sample_ids_clean))