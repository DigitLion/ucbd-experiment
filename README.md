# UCBD Experiment

Experimental validation code and data for the paper:

**"Unified Cognitive Boundary Detection for AI Agents: A Five-Boundary Framework with Cascaded Meta-Cognitive Dispatch"**

## Experiments

| Experiment | Script | Description |
|---|---|---|
| V1-1A | `run_v1a_fluency.py` | Fluency boundary effectiveness and domain specificity |
| V1-1B | `run_v1b_density.py` | Density boundary independence and complementarity |
| V2 | `run_v2_cascade.py` | Cascade vs. parallel detection Pareto efficiency |
| V4 | `run_v4_cross_model.py` | Cross-model domain-specificity validation |

## Analysis

| Script | Description |
|---|---|
| `analyze_v1a.py` | Fluency boundary analysis and plots |
| `analyze_v1a_supplement.py` | Domain-level decomposition and grouped ROC |
| `analyze_v1b_complement.py` | B1-B2 complementarity and oracle routing |
| `analyze_v4_cross_model.py` | Cross-model comparison and scale effect |

## Dataset

- TruthfulQA (790 questions, 38 categories) — `data/TruthfulQA.csv`

## Key Findings

- B1 (Fluency) is domain-specific: AUC=0.623 in effective domains, inverted in blind zones
- B1 and B2 are nearly orthogonal: Pearson r=0.018
- Cascade detection Pareto-dominates parallel, saving 57.4% computation
- Counter-intuitive scale effect: smaller models have stronger B1 signals (3B AUC=0.676 vs 14B AUC=0.537)

## Environment

- Apple M4 Pro, macOS
- Qwen3-14B-4bit, Qwen3-4B, LLaMA-3.2-3B (MLX local inference)
- Qwen3-Embedding (Ollama, 4096 dimensions)
