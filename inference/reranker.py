"""
reranker.py
-----------
Self-contained reranker module.  Loads a fine-tuned QLoRA model
(base LM + LoRA adapter + classification head) and exposes a
clean API for scoring and reranking query-passage pairs.

Usage as a library (e.g. from HopRAG-CR):
    from inference.reranker import Reranker

    critic = Reranker("model/adapter")
    ranked = critic.rerank("what is gravity", [passage1, passage2, ...])
    # → [(best_passage, 0.97), (okay_passage, 0.45), (bad_passage, 0.02)]

Usage as a script (smoke test):
    python inference/reranker.py
"""

import json
import os
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import PeftModel


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MODEL ARCHITECTURE (must match train.py exactly)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class _RerankerModel(nn.Module):
    """
    PEFT causal-LM + binary classification head.
    Internal class — use the Reranker wrapper below.
    """

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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PUBLIC API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class Reranker:
    """
    Loads a fine-tuned QLoRA reranker and provides a simple API
    to score and rerank passages for a given query.

    Parameters
    ----------
    adapter_path : str
        Path to the directory containing:
          - adapter_config.json   (written by peft)
          - adapter_model.*       (LoRA weights)
          - classifier_head.pt    (classification head weights)
        The base model name is auto-detected from adapter_config.json,
        so you only need to point to the adapter directory.

    max_length : int, optional
        Maximum number of tokens per prompt (default: 512).
    """

    def __init__(self, adapter_path: str, max_length: int = 512):
        self.max_length = max_length

        # ── Read base model name from adapter config ─────────
        config_path = os.path.join(adapter_path, "adapter_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"No adapter_config.json in {adapter_path}. "
                f"Train the model first with 'python model/train.py'."
            )

        with open(config_path) as f:
            adapter_cfg = json.load(f)
        base_model_name = adapter_cfg["base_model_name_or_path"]

        # ── Device detection ─────────────────────────────────
        if torch.cuda.is_available():
            self.device = "cuda"
            use_4bit = True
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
            use_4bit = False
        else:
            self.device = "cpu"
            use_4bit = False

        # ── Load tokenizer ───────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── Load base model ──────────────────────────────────
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        elif self.device == "mps":
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
            ).to(self.device)
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float32,
            )

        # ── Attach LoRA adapter ──────────────────────────────
        peft_model = PeftModel.from_pretrained(base_model, adapter_path)
        peft_model.eval()

        # ── Build full model with classifier head ────────────
        hidden_size = peft_model.config.hidden_size
        self.model = _RerankerModel(peft_model, hidden_size)

        # ── Load classifier head weights ─────────────────────
        classifier_path = os.path.join(adapter_path, "classifier_head.pt")
        if not os.path.exists(classifier_path):
            raise FileNotFoundError(
                f"No classifier_head.pt in {adapter_path}. "
                f"Train the model first with 'python model/train.py'."
            )

        device = next(peft_model.parameters()).device
        self.model.classifier.load_state_dict(
            torch.load(classifier_path, map_location=device, weights_only=True)
        )
        self.model.classifier = self.model.classifier.to(
            device=device, dtype=torch.float32
        )
        self.model.eval()
        self._device = device

    def score(self, query: str, passages: list[str]) -> list[float]:
        """
        Score each passage for relevance to the query.

        Returns a list of floats (one per passage) in the SAME order
        as the input.  Each score is P(relevant) ∈ [0, 1].
        """
        if not passages:
            return []

        # Format prompts
        prompts = [
            f"[QUERY]: {query}\n"
            f"[PASSAGE]: {passage}\n"
            f"Is this passage relevant to the query? "
            f"Answer with relevant or irrelevant."
            for passage in passages
        ]

        # Tokenize as a batch
        tokens = self.tokenizer(
            prompts,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = tokens["input_ids"].to(self._device)
        attention_mask = tokens["attention_mask"].to(self._device)

        # Forward pass (no gradient needed for inference)
        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(output.logits, dim=-1)
            scores = probs[:, 1].cpu().tolist()   # P(relevant)

        return scores

    def rerank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        """
        Score and sort passages by relevance to the query.

        Returns a list of (passage_text, score) tuples sorted
        by score descending (most relevant first).
        """
        scores = self.score(query, passages)

        # Pair each passage with its score, sort descending
        ranked = sorted(
            zip(passages, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SMOKE TEST
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    ADAPTER_PATH = "model/adapter"

    print("=" * 60)
    print("  reranker-slm  —  Smoke Test")
    print("=" * 60)

    print(f"\n⏳ Loading reranker from {ADAPTER_PATH}/ ...")
    reranker = Reranker(ADAPTER_PATH)
    print("✅ Reranker loaded!\n")

    # ── Test query and passages ──────────────────────────────
    query = "What causes the seasons on Earth?"

    passages = [
        # Passage A — clearly relevant (correct answer)
        "The seasons are caused by the tilt of Earth's axis at 23.5 "
        "degrees relative to its orbital plane around the Sun. This "
        "tilt means different hemispheres receive varying amounts of "
        "direct sunlight throughout the year.",

        # Passage B — topically related but wrong (common misconception)
        "Earth's distance from the Sun varies throughout the year due "
        "to its elliptical orbit. At perihelion in January, Earth is "
        "about 147 million km from the Sun, and at aphelion in July, "
        "about 152 million km.",

        # Passage C — completely irrelevant
        "The Great Wall of China stretches over 13,000 miles and was "
        "built over many centuries to protect against invasions from "
        "northern nomadic tribes.",
    ]

    print(f"Query: {query}\n")

    # ── Score + rerank ───────────────────────────────────────
    ranked = reranker.rerank(query, passages)

    print("Ranked results:")
    print("-" * 60)
    for rank, (passage, score) in enumerate(ranked, start=1):
        # Show first 80 chars of each passage
        snippet = passage[:80].replace("\n", " ") + "..."
        print(f"  #{rank}  score={score:.4f}  │ {snippet}")

    # ── Verify ordering ─────────────────────────────────────
    print("-" * 60)
    top_passage = ranked[0][0]
    if "tilt" in top_passage.lower():
        print("✅ Correct! The relevant passage about axial tilt ranked #1")
    elif "distance" in top_passage.lower() or "elliptical" in top_passage.lower():
        print("⚠️  The misconception passage ranked #1 — model needs more training")
    else:
        print("❌ The irrelevant passage ranked #1 — something is wrong")

    print("\n🎉 Smoke test complete!")