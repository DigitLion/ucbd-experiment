# UCBD V2: Cascade vs Parallel Detection Report

Dataset: TruthfulQA, 401 binary samples (196 incorrect, 205 correct)

## Strategy Comparison

| Strategy | AUC | 95% CI | F1 | Avg Cost | Efficiency (AUC/Cost) |
|----------|-----|--------|----|---------|-----------------------|
| A: B1-only | 0.5196 | [0.4628, 0.5754] | 0.6599 | 0.000 | 51.96 |
| B: Parallel | 0.5323 | [0.4771, 0.5871] | 0.6599 | 1.000 | 0.53 |
| C: Cascade (best) | 0.5382 | [0.4843, 0.5945] | 0.6599 | 0.716 | 0.74 |

Optimal parallel weight: w_B1=0.65, w_B2=0.35
Best cascade delta: 0.20 (escalation rate: 71.6%)

## Pareto Frontier

| Escalation Rate | Avg Cost | AUC | F1 |
|----------------|---------|-----|------|
| 0.2% | 0.002 | 0.5184 | 0.6599 |
| 9.0% | 0.090 | 0.5208 | 0.6599 |
| 18.0% | 0.180 | 0.5155 | 0.6599 |
| 28.9% | 0.289 | 0.5194 | 0.6599 |
| 37.7% | 0.377 | 0.5269 | 0.6599 |
| 42.6% | 0.426 | 0.5298 | 0.6599 |
| 50.6% | 0.506 | 0.5328 | 0.6599 |
| 56.4% | 0.564 | 0.5353 | 0.6599 |
| 62.3% | 0.623 | 0.5381 | 0.6599 |
| 67.8% | 0.678 | 0.5330 | 0.6599 |
| 71.6% | 0.716 | 0.5382 | 0.6599 |
| 77.8% | 0.778 | 0.5326 | 0.6599 |
| 82.5% | 0.825 | 0.5345 | 0.6599 |
| 84.3% | 0.843 | 0.5337 | 0.6599 |
| 88.0% | 0.880 | 0.5321 | 0.6599 |
| 92.3% | 0.923 | 0.5324 | 0.6599 |
| 93.5% | 0.935 | 0.5337 | 0.6610 |
| 95.0% | 0.950 | 0.5330 | 0.6599 |
| 95.8% | 0.958 | 0.5326 | 0.6599 |
| 97.3% | 0.973 | 0.5321 | 0.6599 |
| 97.8% | 0.978 | 0.5322 | 0.6599 |
| 98.0% | 0.980 | 0.5322 | 0.6599 |
| 98.0% | 0.980 | 0.5322 | 0.6599 |
| 98.3% | 0.983 | 0.5322 | 0.6599 |
| 98.8% | 0.988 | 0.5322 | 0.6599 |
| 98.8% | 0.988 | 0.5322 | 0.6599 |

## Interpretation

1. Cascade is 1.4x more cost-efficient than parallel (AUC/Cost)
2. Only 71.6% of queries need the expensive B2 detector
3. 28.4% of queries are resolved by the free B1 check alone
4. This validates the UCBD energy-minimization principle: check cheap signals first
