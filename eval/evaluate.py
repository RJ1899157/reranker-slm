"""
evaluate.py
-----------
Load the fine-tuned reranker (LoRA adapter + classification head),
run inference on eval_golden.jsonl, compute NDCG@10 and MRR@10,
and print a comparison table against all baselines.
"""

import json
import os
import yaml
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import PeftModel
from sklearn.metrics import ndcg_score
from tqdm import tqdm

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1. CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with open("model/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

MODEL_NAME   = cfg["model_name"]       # Qwen/Qwen2.5-0.5B-Instruct
MAX_LENGTH   = cfg["max_length"]       # 512
OUTPUT_DIR   = cfg["output_dir"]       # model/adapter

GOLDEN_PATH    = "data/splits/eval_golden.jsonl"
BASELINES_PATH = "eval/results/baselines.json"
METRICS_PATH   = "eval/results/metrics.json"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  2. DEVICE DETECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

USE_4BIT = torch.cuda.is_available()

if USE_4BIT:
    DEVICE = "cuda"
    print("🟢 CUDA detected → using 4-bit quantization")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
    print("🟡 Apple MPS detected → loading in float32")
else:
    DEVICE = "cpu"
    print("🟡 CPU only → loading in float32")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  3. PRE-FLIGHT CHECK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

adapter_config = os.path.join(OUTPUT_DIR, "adapter_config.json")
classifier_file = os.path.join(OUTPUT_DIR, "classifier_head.pt")

if not os.path.exists(adapter_config):
    print(f"\n❌ No trained adapter found at {OUTPUT_DIR}/")
    print(f"   Run 'python model/train.py' first to train the model.")
    exit(1)

if not os.path.exists(classifier_file):
    print(f"\n❌ No classifier head found at {classifier_file}")
    print(f"   Run 'python model/train.py' first to train the model.")
    exit(1)

if not os.path.exists(GOLDEN_PATH):
    print(f"\n❌ No golden eval file found at {GOLDEN_PATH}")
    print(f"   Run 'python eval/baselines.py' first to generate it.")
    exit(1)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  4. LOAD TOKENIZER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print(f"\n⏳ Loading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  5. LOAD BASE MODEL + LoRA ADAPTER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#  Reconstruction:
#    1. Load the original Qwen base model (same as training)
#    2. Attach the saved LoRA adapter on top
#    3. Load the classification head weights
#  This gives us the exact model we had at the end of training.

print(f"⏳ Loading base model ({MODEL_NAME}) ...")
if USE_4BIT:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )
elif DEVICE == "mps":
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
    ).to(DEVICE)
else:
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
    )

print(f"⏳ Loading LoRA adapter from {OUTPUT_DIR}/ ...")
peft_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
peft_model.eval()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  6. RECONSTRUCT RERANKER MODEL (same architecture as train.py)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HIDDEN_SIZE = peft_model.config.hidden_size   # 896 for Qwen2.5-0.5B


class RerankerModel(nn.Module):
    """Same architecture as in train.py — must match exactly."""

    def __init__(self, peft_model, hidden_size, num_labels=2):
        super().__init__()
        self.peft_model = peft_model
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.peft_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(input_ids.shape[0], device=input_ids.device)
        pooled = hidden_states[batch_idx, seq_lengths]
        logits = self.classifier(pooled.float())
        return SequenceClassifierOutput(logits=logits)


model = RerankerModel(peft_model, HIDDEN_SIZE)

# Load the trained classification head weights
device = next(peft_model.parameters()).device
model.classifier.load_state_dict(
    torch.load(classifier_file, map_location=device, weights_only=True)
)
model.classifier = model.classifier.to(device=device, dtype=torch.float32)
model.eval()
print("✅ Model fully loaded: base + LoRA adapter + classifier head")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  7. LOAD GOLDEN EVAL DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


eval_groups = load_jsonl(GOLDEN_PATH)
print(f"📊 Loaded {len(eval_groups)} query groups from {GOLDEN_PATH}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  8. PROMPT FORMATTING (same template as train.py)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def format_prompt(query, passage):
    """Build the classification prompt (no answer appended)."""
    return (
        f"[QUERY]: {query}\n"
        f"[PASSAGE]: {passage}\n"
        f"Is this passage relevant to the query? "
        f"Answer with relevant or irrelevant."
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  9. METRIC FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def mrr_at_k(labels, scores, k=10):
    """Reciprocal rank of the first relevant result in top K."""
    ranked_indices = np.argsort(scores)[::-1][:k]
    for rank, idx in enumerate(ranked_indices, start=1):
        if labels[idx] == 1:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(labels, scores, k=10):
    """NDCG @ K for a single query."""
    labels = np.array(labels, dtype=float)
    scores = np.array(scores, dtype=float)
    if labels.sum() == 0:
        return 0.0
    return ndcg_score([labels], [scores], k=k)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  10. RUN INFERENCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#  For each query group:
#    1. Format all passages into prompts
#    2. Tokenize the whole batch at once
#    3. Forward pass → logits → softmax → P(relevant)
#    4. Use P(relevant) as the relevance score for ranking

print("\n⏳ Running reranker inference ...")
reranker_mrr_list = []
reranker_ndcg_list = []

with torch.no_grad():
    for group in tqdm(eval_groups, desc="Evaluating"):
        query    = group["query"]
        passages = group["passages"]
        labels   = group["labels"]

        # Format all passages for this query into prompts
        prompts = [format_prompt(query, p) for p in passages]

        # Tokenize as a batch (all passages for this query at once)
        tokens = tokenizer(
            prompts,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids      = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        # Forward pass
        output = model(input_ids=input_ids, attention_mask=attention_mask)

        # Softmax → P(class 1 = relevant) as relevance score
        probs = torch.softmax(output.logits, dim=-1)
        scores = probs[:, 1].cpu().numpy()    # (num_passages,)

        # Compute metrics for this query
        reranker_mrr_list.append(mrr_at_k(labels, scores, k=10))
        reranker_ndcg_list.append(ndcg_at_k(labels, scores, k=10))

reranker_mrr  = float(np.mean(reranker_mrr_list))
reranker_ndcg = float(np.mean(reranker_ndcg_list))
print(f"✅ Reranker-SLM →  MRR@10: {reranker_mrr:.4f}  |  NDCG@10: {reranker_ndcg:.4f}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  11. LOAD BASELINES FOR COMPARISON
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

baselines = {}
if os.path.exists(BASELINES_PATH):
    with open(BASELINES_PATH) as f:
        baselines = json.load(f)

bm25_mrr  = baselines.get("baselines", {}).get("BM25", {}).get("MRR@10", "N/A")
bm25_ndcg = baselines.get("baselines", {}).get("BM25", {}).get("NDCG@10", "N/A")
bge_mrr   = baselines.get("baselines", {}).get("BGE-base-en-v1.5", {}).get("MRR@10", "N/A")
bge_ndcg  = baselines.get("baselines", {}).get("BGE-base-en-v1.5", {}).get("NDCG@10", "N/A")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  12. SAVE RESULTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Merge with any existing metrics (e.g. training metrics)
metrics = {}
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH) as f:
        metrics = json.load(f)

metrics["eval_golden"] = {
    "num_queries": len(eval_groups),
    "reranker_slm": {
        "MRR@10": round(reranker_mrr, 4),
        "NDCG@10": round(reranker_ndcg, 4),
    },
    "baselines": {
        "BM25":             {"MRR@10": bm25_mrr,  "NDCG@10": bm25_ndcg},
        "BGE-base-en-v1.5": {"MRR@10": bge_mrr,   "NDCG@10": bge_ndcg},
        "BERT-large-ref":   {"MRR@10": 0.365, "NDCG@10": "N/A (1000-candidate)"},
    },
}

os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  13. COMPARISON TABLE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fmt(val):
    """Format a metric value for display."""
    if isinstance(val, (int, float)):
        return f"{val:.4f}"
    return str(val)

print(f"\n📊 Results saved to {METRICS_PATH}")
print("\n╔════════════════════════════════════════════════════════╗")
print("║         Reranker Comparison (eval_golden)              ║")
print("╠════════════════════════╦══════════╦════════════════════╣")
print("║ Method                 ║  MRR@10  ║  NDCG@10           ║")
print("╠════════════════════════╬══════════╬════════════════════╣")
print(f"║ BM25                   ║  {fmt(bm25_mrr):>6}  ║  {fmt(bm25_ndcg):>6}             ║")
print(f"║ BGE-base-en-v1.5       ║  {fmt(bge_mrr):>6}  ║  {fmt(bge_ndcg):>6}             ║")
print(f"║ BERT-large (ref*)      ║  0.3650  ║  N/A (1000-cand)   ║")
print(f"║ reranker-slm (ours) 🔥 ║  {reranker_mrr:>6.4f}  ║  {reranker_ndcg:>6.4f}             ║")
print("╠════════════════════════╩══════════╩════════════════════╣")

# Verdict
if isinstance(bge_mrr, (int, float)) and reranker_mrr > bge_mrr:
    delta_mrr = reranker_mrr - bge_mrr
    print(f"║ 🎉 reranker-slm BEATS BGE by +{delta_mrr:.4f} MRR@10!          ║")
elif isinstance(bge_mrr, (int, float)):
    delta_mrr = bge_mrr - reranker_mrr
    print(f"║ 📈 BGE still leads by {delta_mrr:.4f} — try more epochs/data   ║")
else:
    print("║ ⚠️  Run baselines.py first for comparison                ║")

print("║ * BERT-large ref is on 1000-candidate setup (harder)    ║")
print("╚════════════════════════════════════════════════════════╝")