# The Alignment Tax

**Response Homogenization in Aligned LLMs and Its Implications for Uncertainty Estimation**

[![arXiv](https://img.shields.io/badge/arXiv-2603.24124-b31b1b.svg)](https://arxiv.org/abs/2603.24124)

Mingyi Liu | [arXiv](https://arxiv.org/abs/2603.24124) | cs.LG

## Overview

RLHF-aligned language models exhibit **response homogenization**: on TruthfulQA (n=790), 40–79% of questions produce a single semantic cluster across 10 i.i.d. samples. On affected questions, sampling-based uncertainty methods have **zero discriminative power** (AUROC=0.500), while free token entropy retains signal (0.603). This **alignment tax** is caused by DPO (not SFT), is recipe-dependent, and generalizes across model families.

## Experiments

25 experiments across 5 benchmarks, 4 model families, and 3 model scales (3B–14B).

### Core Experiments (Boundary Detection)

| # | Script | Description | Key Result |
|---|--------|-------------|------------|
| 1 | `run_v1a_fluency.py` | B1 token entropy on TruthfulQA (790q) | AUROC=0.599, domain-specific |
| 2 | `run_v1b_density.py` | B2 embedding density | r(B1,B2)=0.119, near-orthogonal |
| 3 | `run_v2_cascade.py` | Cascade vs. parallel detection | 57% cost savings |
| 4 | `run_v4_cross_model.py` | Cross-model (3B/4B/14B) validation | Scale effect confirmed |
| 5 | `run_v9_end_to_end.py` | End-to-end GSM8K evaluation | AUROC=0.724, 84.4%→93.2% |
| 9 | `run_v10_b4_assoc.py` | B4 KG association boundary | MI≤0.02 bits |
| 10 | `run_v11_cascade_demo.py` | Three-boundary cascade demo | Combined AUROC=0.601 |

### Alignment Tax Experiments

| # | Script | Description | Key Result |
|---|--------|-------------|------------|
| 12 | — | SE/SelfCheck/NLI-SE baselines (790q) | B1 beats all (p<0.05) |
| 13 | — | Base-vs-instruct ablation (200q) | 1.0% vs 28.5% SCR |
| 14 | — | Cross-family replication (3 families) | 1.0%–28.5% SCR |
| 16 | — | Training stage ablation (Base→SFT→DPO) | DPO drives collapse |
| 17 | `run_exp21_200tok.py` | Max generation length sensitivity | 8% SCR persists at 200 tokens |
| 18 | — | Tulu-3 chain DPO replication | 0.5% SCR (recipe-dependent) |
| 19 | — | Quantization sensitivity (Q4 vs Q8) | Identical SCR |
| 20 | — | Cross-embedder validation | Nomic 92% vs Qwen 78% SCR |
| 22 | `run_exp22_webq.py` | Cross-dataset (WebQuestions) | 58.0% SCR generalizes |
| 23 | — | Cross-encoder NLI-SE head-to-head | +0.014 AUROC (CIs overlap) |
| 24 | — | Stage-wise token entropy | Base 1.175→DPO 0.776 (66% retained) |
| 25 | — | Calibration metrics (ECE/Brier/AURC) | ECE=0.182→0.021 (Platt) |

Full 25-experiment details in the [paper](https://arxiv.org/abs/2603.24124).

### Utilities

| Script | Description |
|--------|-------------|
| `cluster_analysis.py` | Agglomerative clustering with cosine distance for SCR computation |
| `analyze_v1a.py` | B1 analysis and plots |
| `analyze_v1a_supplement.py` | Domain-level decomposition |
| `analyze_v1b_complement.py` | B1-B2 complementarity analysis |
| `analyze_v4_cross_model.py` | Cross-model comparison |

## Datasets

| File | Description |
|------|-------------|
| `data/TruthfulQA.csv` | TruthfulQA (790 questions, 38 categories) |
| `data/tqa_200q.json` | TruthfulQA 200-question subset |
| `data/webq_200q.json` | WebQuestions 200-question subset |
| `data/hotpot_100.json` | HotpotQA 100-question subset |

## Key Results

- **Alignment tax**: 40–79% of TruthfulQA questions collapse to a single semantic cluster under aligned models
- **DPO is the driver**: SFT preserves base diversity (≤1.5% SCR); DPO causes collapse (0.5%–28.5% SCR)
- **Token entropy decouples**: DPO retains 66% of token entropy while collapsing response diversity (Exp 24)
- **Task-dependent**: Cohen's d = 0.07 (factual QA) vs 0.81 (math reasoning)
- **Cascade works**: GSM8K accuracy 84.4% → 93.2% at 50% coverage; 57% cost savings

## Environment

- Apple M4 Pro 64GB, macOS
- Local models: Mistral-7B-Instruct, Qwen3-14B/4B, LLaMA-3.2-3B, Zephyr-7B (Ollama/MLX)
- Embeddings: Qwen3-Embedding (Ollama), Nomic-embed-text, text-embedding-3-small (OpenAI)
- Python 3.14, ~14h total compute

## Citation

```bibtex
@article{liu2026alignmenttax,
  title={The Alignment Tax: Response Homogenization in Aligned LLMs and Its Implications for Uncertainty Estimation},
  author={Liu, Mingyi},
  journal={arXiv preprint arXiv:2603.24124},
  year={2026}
}
```

## License

MIT
