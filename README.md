# The Alignment Tax: Response Homogenization in Aligned LLMs

Experiment code for the paper:

**"The Alignment Tax: Response Homogenization in Aligned LLMs and Its Implications for Uncertainty Estimation"**

Mingyi Liu | [arXiv](https://arxiv.org/abs/2603.xxxxx) | cs.LG

## Overview

RLHF-aligned language models exhibit **response homogenization**: on TruthfulQA (n=790), 40-79% of questions produce a single semantic cluster across 10 i.i.d. samples. On affected questions, sampling-based uncertainty methods have zero discriminative power (AUROC=0.500), while free token entropy retains signal (0.603).

## Experiments

22 experiments across 5 benchmarks, 4 model families, and 3 model scales (3B-14B).

### Core Experiments (Boundary Detection)

| # | Script | Description |
|---|--------|-------------|
| 1 | `run_v1a_fluency.py` | B1 (Fluency) boundary detection — domain-specific entropy signal |
| 2 | `run_v1b_density.py` | B2 (Density) boundary detection — embedding clustering |
| 3 | `run_v2_cascade.py` | Cascade vs. parallel detection — Pareto efficiency |
| 4 | `run_v4_cross_model.py` | Cross-model validation (Qwen3-14B/4B, LLaMA-3.2-3B) |

### Alignment Tax Experiments

| # | Script | Description |
|---|--------|-------------|
| 5-16 | `logprobs_entropy_experiment.py` | Token entropy via API logprobs (GPT-4o-mini, DeepSeek, Gemini, Together AI) |
| 17 | `run_exp21_200tok.py` | Max-tokens sensitivity (200 tokens, 200 questions) |
| 18 | `exp19_quant_scr.py` | Quantization sensitivity (Q4_K_M vs Q8_0) |
| 19 | `exp20_logtoku_headtohead.py` | LogTokU head-to-head (entropy vs neg-log-prob AUROC) |
| 20 | `run_exp22_webq.py` | Cross-dataset validation (WebQuestions, 200 questions) |
| 21 | `download_triviaqa.py` | TriviaQA dataset preparation |

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

- **Response Homogenization**: 40-79% SCR on TruthfulQA across model families
- **Alignment Tax**: Sampling AUROC collapses to 0.500 on homogenized questions
- **Token Entropy**: Retains signal (AUROC 0.603-0.724) — free, single-pass
- **Causal Evidence**: Base 1.0% SCR vs Instruct 28.5% (p < 10^-6); DPO is the cause, not SFT
- **Cascade (UCBD)**: 84.4% -> 93.2% accuracy at 50% coverage; 57% cost savings

## Environment

- Apple M4 Pro, macOS
- Local models: Mistral-7B-Instruct, Qwen3-14B/4B, LLaMA-3.2-3B (Ollama/MLX)
- Embeddings: Qwen3-Embedding (Ollama), text-embedding-3-small (OpenAI)
- APIs: OpenAI, DeepSeek, Google Gemini, Together AI

## Citation

```bibtex
@article{liu2026alignmenttax,
  title={The Alignment Tax: Response Homogenization in Aligned LLMs and Its Implications for Uncertainty Estimation},
  author={Liu, Mingyi},
  journal={arXiv preprint arXiv:2603.xxxxx},
  year={2026}
}
```

## License

MIT
