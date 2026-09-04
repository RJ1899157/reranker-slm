"""
train.py
--------
Fine-tune Qwen2.5-0.5B-Instruct with QLoRA for binary passage
relevance classification.  Saves LoRA adapter weights, the
classification head, and training metrics when done.
"""

import os
import json
import yaml
import torch
import torch.nn as nn
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score
from datasets import Dataset

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1. LOAD CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with open("model/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

MODEL_NAME        = cfg["model_name"]          # Qwen/Qwen2.5-0.5B-Instruct
MAX_LENGTH        = int(cfg["max_length"])     # 512
LORA_R            = int(cfg["lora_r"])         # 16
LORA_ALPHA        = int(cfg["lora_alpha"])     # 32
LORA_DROPOUT      = float(cfg["lora_dropout"]) # 0.05
TARGET_MODULES    = cfg["target_modules"]      # [q_proj, v_proj]
BATCH_SIZE        = int(cfg["batch_size"])     # 8
GRAD_ACCUMULATION = int(cfg["grad_accumulation"]) # 4
LEARNING_RATE     = float(cfg["learning_rate"]) # 0.0002
EPOCHS            = int(cfg["epochs"])         # 2
OUTPUT_DIR        = cfg["output_dir"]          # model/adapter
TRAIN_SAMPLES     = int(cfg.get("train_samples", 10000))
VAL_SAMPLES       = int(cfg.get("val_samples", 1000))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  2. DEVICE DETECTION — CUDA vs MPS vs CPU
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

USE_CUDA = torch.cuda.is_available()
USE_MPS  = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

if USE_CUDA:
    print("🟢 CUDA detected → using 4-bit quantization (QLoRA)")
    DEVICE = "cuda"
elif USE_MPS:
    print("🟢 Apple Silicon MPS detected → training locally on Mac GPU")
    DEVICE = "mps"
else:
    print("🟡 CPU only detected")
    DEVICE = "cpu"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  3. QUANTIZATION CONFIG (only used with CUDA)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#  4-bit quantization stores each weight in 4 bits instead of 16.
#  This cuts model memory by ~4×.
#
#  - nf4 (normalized float4):  a special 4-bit format optimized for
#    normally-distributed neural network weights — better than plain int4.
#  - double quantization:  even the quantization constants get quantized,
#    saving an extra ~0.4 GB on a 0.5B model.
#  - compute_dtype=float16:  math is done in 16-bit for speed,
#    weights are stored in 4-bit for memory savings.

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  4. LOAD TOKENIZER + BASE MODEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print(f"\n⏳ Loading tokenizer for {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"⏳ Loading model ({MODEL_NAME}) ...")

if USE_CUDA:
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
    base_model = prepare_model_for_kbit_training(base_model)
elif USE_MPS:
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
    ).to(DEVICE)
    base_model.gradient_checkpointing_enable()
    base_model.enable_input_require_grads()
else:
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
    )

print(f"✅ Model loaded on {DEVICE} (memory-optimized with float16 + gradient checkpointing)")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,                        # rank of adapter matrices
    lora_alpha=LORA_ALPHA,           # scaling factor (alpha/r = 2×)
    lora_dropout=LORA_DROPOUT,       # regularization
    target_modules=TARGET_MODULES,   # attach to q_proj and v_proj
)

peft_model = get_peft_model(base_model, lora_config)
peft_model.print_trainable_parameters()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  6. CLASSIFICATION HEAD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#  The LM produces a hidden state vector (896 dims for Qwen 0.5B)
#  for every token.  We take the LAST real token's hidden state
#  (it has "seen" the whole input via causal attention) and map it
#  through a Linear(896 → 2) layer to get class logits.
#
#  Class 0 = irrelevant,  Class 1 = relevant.

HIDDEN_SIZE = peft_model.config.hidden_size   # 896 for Qwen2.5-0.5B


class RerankerModel(nn.Module):
    """
    PEFT causal-LM  +  binary classification head.

    Forward:
      tokens → Qwen+LoRA → hidden states → last-token pooling
             → Linear(896, 2) → cross-entropy loss
    """

    def __init__(self, peft_model, hidden_size, num_labels=2):
        super().__init__()
        self.peft_model = peft_model
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

        # Put the classifier head on the same device as the model
        device = next(peft_model.parameters()).device
        self.classifier = self.classifier.to(device=device, dtype=torch.float32)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Step 1 — Run through Qwen + LoRA, get all hidden states
        outputs = self.peft_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Step 2 — Grab the last transformer layer's output
        #          Shape: (batch_size, seq_len, 896)
        hidden_states = outputs.hidden_states[-1]

        # Step 3 — Pool: pick the LAST non-padding token per example
        #          attention_mask sums to the number of real tokens
        seq_lengths = attention_mask.sum(dim=1) - 1        # 0-indexed
        batch_idx = torch.arange(input_ids.shape[0], device=input_ids.device)
        pooled = hidden_states[batch_idx, seq_lengths]     # (batch, 896)

        # Step 4 — Classify (cast to float32 for numerical stability)
        logits = self.classifier(pooled.float())           # (batch, 2)

        # Step 5 — Compute loss if labels are provided (training + eval)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)

    def save_all(self, output_dir):
        """Save LoRA adapter weights + classification head separately."""
        os.makedirs(output_dir, exist_ok=True)
        # Save LoRA adapter (small — just the A and B matrices)
        self.peft_model.save_pretrained(output_dir)
        # Save the classification head (tiny — 896×2 + bias = 1794 params)
        torch.save(
            self.classifier.state_dict(),
            os.path.join(output_dir, "classifier_head.pt"),
        )
        print(f"💾 Adapter + classifier head saved to {output_dir}/")


model = RerankerModel(peft_model, HIDDEN_SIZE)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  7. PREPARE DATA (classification — prompt only, no answer)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#  Note: preprocess.py appended the answer ("relevant"/"irrelevant")
#  to the text for generative training.  For classification, we
#  only feed the PROMPT — the label is the classification target.


def load_jsonl(path, limit=None):
    """Read a .jsonl file into a list of dicts."""
    records = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            records.append(json.loads(line))
    return records


def format_for_classification(record):
    """Build the prompt WITHOUT the answer appended."""
    text = (
        f"[QUERY]: {record['query']}\n"
        f"[PASSAGE]: {record['passage']}\n"
        f"Is this passage relevant to the query? "
        f"Answer with relevant or irrelevant."
    )
    return {"text": text, "label": record["label"]}


def prepare_split(path, limit=None):
    """Load JSONL → format prompts → tokenize → HF Dataset."""
    raw = load_jsonl(path, limit=limit)
    formatted = [format_for_classification(r) for r in raw]
    ds = Dataset.from_list(formatted)

    def tokenize_fn(batch):
        tokens = tokenizer(
            batch["text"],
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
        )
        tokens["labels"] = batch["label"]   # "labels" (plural) for Trainer
        return tokens

    return ds.map(tokenize_fn, batched=True, remove_columns=["text", "label"])


print("\n⏳ Preparing datasets ...")
train_ds = prepare_split("data/splits/train.jsonl", limit=TRAIN_SAMPLES)
val_ds   = prepare_split("data/splits/val.jsonl", limit=VAL_SAMPLES)
print(f"✅ Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples")
print(f"   Columns: {train_ds.column_names}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  8. METRICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def compute_metrics(eval_pred):
    """Called by Trainer after each eval epoch."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)   # pick the class with higher score
    return {"accuracy": accuracy_score(labels, preds)}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  9. TRAINING ARGUMENTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

training_args = TrainingArguments(
    output_dir="model/checkpoints",              # intermediate checkpoints
    num_train_epochs=EPOCHS,                     # 2 passes
    per_device_train_batch_size=BATCH_SIZE,       # 8
    per_device_eval_batch_size=BATCH_SIZE,        # 8
    gradient_accumulation_steps=GRAD_ACCUMULATION, # effective batch = 8×4 = 32
    learning_rate=LEARNING_RATE,                  # 0.0002
    eval_strategy="epoch",                       # evaluate after every epoch
    save_strategy="no",                          # Do not save base model checkpoints (avoids tied weights crash)
    load_best_model_at_end=False,
    fp16=USE_CUDA,                               # only on CUDA
    logging_steps=25,                            # log train loss every 25 steps
    dataloader_pin_memory=False,                 # clean up MPS pin_memory warning
    report_to="none",                            # disable wandb/mlflow
    remove_unused_columns=False,                 # keep all columns in the batch
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  10. TRAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

print("\n🚀 Starting training ...")
print(f"   Epochs:           {EPOCHS}")
print(f"   Batch size:       {BATCH_SIZE}")
print(f"   Grad accumulation: {GRAD_ACCUMULATION}")
print(f"   Effective batch:  {BATCH_SIZE * GRAD_ACCUMULATION}")
print(f"   Learning rate:    {LEARNING_RATE}")
print(f"   Device:           {DEVICE}\n")

train_result = trainer.train()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  11. SAVE ADAPTER + CLASSIFIER HEAD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

model.save_all(OUTPUT_DIR)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  12. LOG METRICS TO eval/results/metrics.json
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

os.makedirs("eval/results", exist_ok=True)

# Extract per-epoch metrics from trainer's log history
metrics = {
    "train_loss_per_step": [],
    "val_accuracy_per_epoch": [],
}

for entry in trainer.state.log_history:
    # Logging-step entries have "loss" (train loss at that step)
    if "loss" in entry and "epoch" in entry:
        metrics["train_loss_per_step"].append({
            "step": entry.get("step"),
            "epoch": round(entry["epoch"], 2),
            "train_loss": entry["loss"],
        })
    # Eval entries have "eval_accuracy" (computed by compute_metrics)
    if "eval_accuracy" in entry:
        metrics["val_accuracy_per_epoch"].append({
            "epoch": round(entry["epoch"], 2),
            "val_accuracy": entry["eval_accuracy"],
            "val_loss": entry.get("eval_loss"),
        })

metrics["final_train_loss"] = train_result.metrics.get("train_loss")
metrics["best_val_accuracy"] = max(
    (e["val_accuracy"] for e in metrics["val_accuracy_per_epoch"]),
    default=None,
)

with open("eval/results/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n📊 Metrics saved to eval/results/metrics.json")
print(f"   Final train loss:   {metrics['final_train_loss']:.4f}")
print(f"   Best val accuracy:  {metrics['best_val_accuracy']:.4f}")
print("\n🎉 Training complete!")