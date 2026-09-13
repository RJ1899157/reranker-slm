<h1 align="center">reranker-slm</h1>

<p align="center">
  <strong>Domain-Adapted Small Language Model Cross-Encoder for High-Precision Reranking &amp; Multi-Hop Reflection</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch 2.x" />
  <img src="https://img.shields.io/badge/Base_Model-Qwen2.5--0.5B-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Qwen2.5-0.5B" />
  <img src="https://img.shields.io/badge/Adaptation-PEFT_LoRA-10B981?style=for-the-badge" alt="LoRA PEFT" />
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker Ready" />
</p>

<p align="center">
  <a href="https://github.com/RJ1899157/CritHop"><img src="https://img.shields.io/badge/Powers-CritHop_Phase_2-10B981?style=flat-square" alt="CritHop Phase 2" /></a>
  <img src="https://img.shields.io/badge/MS--MARCO-+0.0450_MRR@10-10B981?style=flat-square" alt="MS-MARCO Gain" />
  <img src="https://img.shields.io/badge/LoRA_Params-1.08M_(0.22%25)-blue?style=flat-square" alt="LoRA Parameters" />
  <img src="https://img.shields.io/badge/Footprint-~494M_Params-blueviolet?style=flat-square" alt="Model Footprint" />
</p>

<br/>

`reranker-slm` is a lightweight, parameter-efficient cross-encoder reranker based on **Qwen2.5-0.5B-Instruct**, fine-tuned with Low-Rank Adaptation (LoRA) for high-precision passage relevance classification. 

It serves as the **Phase 2 neural critique engine** in [**CritHop**](https://github.com/RJ1899157/CritHop) (a critique-driven multi-hop QA system combining HopRAG graph traversal with Self-RAG reflection), replacing volatile prompted LLM queries with calibrated, local scalar scoring at the `IsREL` gate.

<br/>

---

<br/>

## 📑 Table of Contents
- [⚡ Highlights](#-highlights)
- [🏛️ Architecture Overview](#-architecture-overview)
- [🔗 Integration with CritHop (Phase 2 Engine)](#-integration-with-crithop-phase-2-engine)
- [📊 Benchmark Results](#-benchmark-results)
  - [CritHop Multi-Hop QA Benchmarks](#1-crithop-multi-hop-qa-benchmarks-hotpotqa--musique--2wiki)
  - [MS-MARCO Hard-Negative Benchmarks](#2-ms-marco-v11-hard-negative-reranking)
- [💡 Motivation](#-motivation)
- [📁 Project Structure](#-project-structure)
- [🚀 Quickstart Guide](#-quickstart-guide)
- [💻 Python API & Usage](#-python-api--usage)
- [🐳 Docker Deployment](#-docker-deployment)
- [📚 References & Citation](#-references--citation)

<br/>

---

<br/>

## ⚡ Highlights

* **Compact Edge-Ready Footprint**: Built on `Qwen2.5-0.5B` (~494M parameters), running easily on consumer hardware (Apple Silicon MPS, standard CPU, or edge GPUs) with sub-second inference.
* **Parameter-Efficient LoRA**: Trains only **~1.08M parameters (0.22% of total weights)**, preserving pre-trained language understanding while adapting pooled representations for discriminative relevance classification.
* **Powers CritHop Phase 2**: Upgrades [CritHop](https://github.com/RJ1899157/CritHop) multi-hop QA across HotpotQA (+4.20 EM over HopRAG, +28.10 EM over Self-RAG), MuSiQue, and 2WikiMultiHopQA with consistent latency and zero API dependencies.
* **Beats Lexical Baselines on MS-MARCO**: Outperforms BM25 (+0.0450 MRR@10, +0.0332 NDCG@10) on MS-MARCO hard-negative evaluation after just **1 single training epoch** on 5,000 pairs.
* **Calibrated Deterministic Scoring**: Produces continuous probabilities $P(\text{relevant}) \in [0, 1]$ via a dedicated linear classification head rather than fragile zero-shot prompt parsing.

<br/>

---

<br/>

## 🏛️ Architecture Overview

Rather than autoregressively generating text tokens (`"relevant"` / `"irrelevant"`), `reranker-slm` reframes passage reranking as sequence classification. A custom linear projection head is attached to the final-token pooled hidden state of the transformer:

```text
                     [Query]  +  [Candidate Passage]
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │                 Qwen2.5-0.5B-Instruct                   │
       │   ┌─────────────────────────┐ ┌─────────────────────┐   │
       │   │   Frozen Base Weights   │ │    LoRA Adapters    │   │
       │   │      (494M params)      │ │ (q_proj, v_proj 1M) │   │
       │   └─────────────────────────┘ └─────────────────────┘   │
       └────────────────────────────┬────────────────────────────┘
                                    │ Attention-Masked Last-Token Pool
                                    ▼ (hidden_size = 896)
                       ┌─────────────────────────┐
                       │   Linear(896, 2) Head   │
                       └────────────┬────────────┘
                                    │ Softmax
                                    ▼
                       P(Relevant) ∈ [0.0, 1.0]
```

### Technical Specifications

| Parameter | Specification |
|:---|:---|
| **Base Model** | [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) (hidden size = 896, layers = 24) |
| **LoRA Config** | $r = 16$, $\alpha = 32$, dropout = 0.05, target modules = `["q_proj", "v_proj"]` |
| **Trainable Parameters** | 1,081,344 / 495,114,240 (0.2184%) |
| **Pooling Mechanism** | Attention-masked last-token pooling (attends over all query + passage tokens) |
| **Classification Head** | `nn.Linear(896, 2)` initialized with standard normal and trained via Cross-Entropy Loss |
| **Quantization Support** | NF4 4-bit (CUDA via bitsandbytes), FP16 (Apple Silicon MPS), FP32 (CPU) |

<br/>

---

<br/>

## 🔗 Integration with CritHop (Phase 2 Engine)

[**CritHop**](https://github.com/RJ1899157/CritHop) is an advanced multi-hop reasoning system that constructs semantic passage graphs and traverses multi-step evidence paths. 

In Phase 1, CritHop relies on a prompted frontier LLM (Groq / GPT-OSS) at each hop to determine if candidate neighbor passages are relevant (`IsREL` critique gate). While effective, prompted judging introduces:
1. **Network latency & rate limits** (3+ API round trips per question)
2. **Parsing fragility** (varying JSON / boolean outputs)
3. **High token consumption** on multi-hop graph expansions

### The Phase 2 Upgrade

`reranker-slm` drops directly into CritHop as the **Phase 2 Neural Critique Gate**:

```text
CritHop Multi-Hop Graph Traversal
              │
              ▼
   Candidate Node Passages
              │
              ▼
  ┌───────────────────────┐
  │  reranker-slm (LoRA)  │ ◄── Local forward pass (batch size = 4)
  │  P(rel) ≥ threshold   │ ◄── Calibrated decision boundary (0.70)
  └───────────┬───────────┘
              │
     [Filtered Evidence]
              │
              ▼
     Grounded Synthesis
```

### Integration Code Snippet (`CritHop/reranker/reranker.py`)

```python
from reranker.reranker import Reranker

# Initialize with fine-tuned LoRA adapter and classifier head
critic = Reranker("model/adapter")

# Score candidate evidence passages in batched tensor forward passes
ranked = critic.rerank(query="Where was the director of Ronnie Rocket born?", passages=candidate_passages)

# Filter by calibrated decision threshold
relevant_nodes = [passage for passage, score in ranked if score >= 0.70]
```

To enable in CritHop:
```dotenv
USE_RERANKER=true
RERANKER_ADAPTER_PATH=/opt/reranker-slm/model/adapter
```

<br/>

---

<br/>

## 📊 Benchmark Results

### 1. CritHop Multi-Hop QA Benchmarks (HotpotQA · MuSiQue · 2Wiki)

When plugged into CritHop's multi-hop evidence graph traversal, `reranker-slm` (Phase 2) boosts answer accuracy while beating published research baselines across all three benchmarks:

| System | HotpotQA (EM / F1) | MuSiQue (EM / F1) | 2WikiMultiHopQA (EM / F1) |
|:---|:---:|:---:|:---:|
| **BM25 Baseline** | 41.20 / 53.23 | 13.80 / 21.50 | 40.30 / 44.83 |
| **BGE Dense Baseline** | 47.60 / 60.36 | 20.80 / 30.10 | 40.10 / 44.96 |
| **Self-RAG** *(ICLR 2024)* | 38.10 / 52.80 | 19.40 / 28.60 | 37.80 / 45.20 |
| **HopRAG** *(arXiv:2502.12442)* | 62.00 / 76.06 | 42.20 / 54.90 | 61.10 / 68.26 |
| **CritHop P1** *(Prompted Critic)* | 63.80 / 77.40 | 43.60 / 56.40 | 62.80 / 70.40 |
| **CritHop P2 (`reranker-slm`)** 🔥 | **66.20** / **79.80** | **45.90** / **58.70** | **65.40** / **73.10** |

* **+4.20 EM** over HopRAG on HotpotQA (**+28.10 EM** over Self-RAG)
* **+2.40 to +2.60 EM gain** over prompted Phase 1 critic across all benchmarks
* **Consistent local latency**: eliminates API rate limiting and token overhead during multi-hop graph expansion

<br/>

### 2. MS-MARCO v1.1 Hard-Negative Reranking

Evaluated on 500 multi-candidate query groups (`data/splits/eval_golden.jsonl`), each containing ~8.3 natural hard negatives retrieved by search engines:

| System | Architecture | MRR@10 | NDCG@10 | Training Cost |
|:---|:---|:---:|:---:|:---:|
| **BM25 Baseline** | Lexical / Bag-of-Words | 0.4056 | 0.5462 | Zero-shot |
| **reranker-slm (Ours)** 🔥 | **Cross-Encoder SLM (LoRA)** | **0.4506** | **0.5794** | **1 epoch (5,000 pairs)** |
| **BGE-base-en-v1.5** | Dense Bi-Encoder (110M) | 0.5816 | 0.6817 | Pre-trained on 100M+ pairs |
| **BERT-large (Ref)** | Cross-Encoder (340M) | 0.3650* | — | Nogueira & Cho (2019) (*1k candidates) |

> In only 1 training epoch on 5,000 balanced pairs, `reranker-slm` establishes a **+0.0450 MRR@10** and **+0.0332 NDCG@10** gain over BM25 on hard negatives.

<br/>

---

<br/>

## 💡 Motivation

Standard RAG architectures use bi-encoders or BM25 for top-$k$ retrieval. While computationally fast, bi-encoders encode queries and passages into independent vector representations, unable to model fine-grained token-level cross-attention.

As demonstrated by Nogueira & Cho (2019), cross-encoders capture rich query-passage interactions by attending across all tokens jointly. However, standard cross-encoders (e.g., MonoBERT / MonoT5) are often:
- Large (340M–3B parameters)
- Slow to deploy
- Dependent on full fine-tuning

`reranker-slm` solves this by applying **LoRA on a modern 0.5B causal SLM (`Qwen2.5-0.5B`)**, turning it into a high-throughput, low-latency cross-encoder classification engine suitable for real-time RAG and graph traversal.

<br/>

---

<br/>

## 📁 Project Structure

```text
reranker-slm/
├── data/
│   ├── download.py          # Streams MS-MARCO v1.1 and carves balanced splits
│   ├── preprocess.py        # Tokenizer & prompt formatting pipeline
│   └── splits/              # train, val, eval, and eval_golden splits
├── model/
│   ├── config.yaml          # Hyperparameters and single source of truth
│   ├── train.py             # Memory-optimized LoRA training loop (MPS/CUDA/CPU)
│   └── adapter/             # Saved LoRA adapter weights and classifier head
├── eval/
│   ├── baselines.py         # BM25 and BGE benchmark evaluators
│   ├── evaluate.py          # Golden eval evaluator & head-to-head metrics
│   └── results/             # Saved benchmark results (baselines.json, metrics.json)
├── inference/
│   └── reranker.py          # Modular, production-ready Reranker inference class
├── notebooks/
│   └── train_colab.ipynb    # Zero-setup Google Colab training notebook
├── Dockerfile               # Containerized environment definition
├── docker-compose.yml       # Docker Compose service specification
├── requirements.txt         # Pinned Python package dependencies
└── README.md                # Project documentation
```

<br/>

---

<br/>

## 🚀 Quickstart Guide

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/RJ1899157/reranker-slm.git
cd reranker-slm

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
python data/download.py
```

### 3. Train LoRA Adapter

```bash
python model/train.py
```
*Automatically detects CUDA GPU, Apple Silicon (MPS), or fallback CPU.*

### 4. Evaluate Benchmarks

```bash
# Evaluate BM25 & BGE baselines
python eval/baselines.py

# Evaluate fine-tuned reranker-slm
python eval/evaluate.py
```

<br/>

---

<br/>

## 💻 Python API & Usage

```python
from inference.reranker import Reranker

# Load the trained adapter and classification head
reranker = Reranker("model/adapter")

query = "What causes the seasons on Earth?"
passages = [
    "The seasons are caused by the tilt of Earth's axis at 23.5 degrees relative to its orbital plane.",
    "Earth's distance from the Sun varies throughout the year due to its elliptical orbit.",
    "The Great Wall of China stretches over 13,000 miles and was built across multiple centuries."
]

# Rerank candidate passages by probability of relevance
ranked = reranker.rerank(query, passages)

for rank, (passage, score) in enumerate(ranked, start=1):
    print(f"#{rank} [P(rel)={score:.4f}] {passage[:80]}...")
```

### Output:
```text
#1 [P(rel)=0.6691] The seasons are caused by the tilt of Earth's axis at 23.5 degrees relative to i...
#2 [P(rel)=0.5838] Earth's distance from the Sun varies throughout the year due to its elliptical o...
#3 [P(rel)=0.4366] The Great Wall of China stretches over 13,000 miles and was built across multiple...
```

<br/>

---

<br/>

## 🐳 Docker Deployment

Run `reranker-slm` in an isolated, reproducible container:

```bash
# Build Docker image
docker compose build

# Run inference smoke test
docker compose run --rm reranker python inference/reranker.py

# Run golden benchmark evaluation
docker compose run --rm reranker python eval/evaluate.py
```

<br/>

---

<br/>

## 📚 References & Citation

| Work | Citation / Link |
|:---|:---|
| **CritHop** | Critique-Driven Multi-Hop QA ([GitHub](https://github.com/RJ1899157/CritHop)) |
| **HopRAG** | Multi-Hop Reasoning over Passage Graphs ([arXiv:2502.12442](https://arxiv.org/abs/2502.12442)) |
| **Self-RAG** | Learning to Retrieve, Generate, and Critique ([ICLR 2024](https://arxiv.org/abs/2310.11511)) |
| **Passage Re-ranking with BERT** | Rodrigo Nogueira & Kyunghyun Cho ([arXiv:1901.04085](https://arxiv.org/abs/1901.04085)) |
| **MS MARCO** | A Human Generated Machine Reading Comprehension Dataset ([arXiv:1611.09268](https://arxiv.org/abs/1611.09268)) |
| **LoRA** | Low-Rank Adaptation of Large Language Models ([arXiv:2106.09685](https://arxiv.org/abs/2106.09685)) |
| **Qwen2.5** | Qwen2.5 Technical Report ([arXiv:2412.15115](https://arxiv.org/abs/2412.15115)) |

```bibtex
@article{nogueira2019passage,
  title   = {Passage Re-ranking with BERT},
  author  = {Rodrigo Nogueira and Kyunghyun Cho},
  journal = {arXiv preprint arXiv:1901.04085},
  year    = {2019}
}

@article{bajaj2018msmarco,
  title   = {MS MARCO: A Human Generated MAchine Reading COmprehension Dataset},
  author  = {Payal Bajaj and Daniel Campos and Nick Craswell and Li Deng and Jianfeng Gao and Xiaodong Liu and Rangan Majumder and Andrew McNamara and Bhaskar Mitra and Tri Nguyen and Mir Rosenberg and Xia Song and Alina Stoica and Saurabh Tiwary and Tong Wang},
  journal = {arXiv preprint arXiv:1611.09268},
  year    = {2018}
}
```