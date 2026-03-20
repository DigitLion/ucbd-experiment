# UCBD V1-1A: Fluency Boundary Experiment Report

Total questions: 790
Binary subset: 401 (correct=205, incorrect=196)
Ambiguous: 288, No match: 101

## Entropy vs Correctness

| Metric | Correct (mean) | Incorrect (mean) | AUC-ROC | p-value | Direction |
|--------|---------------|-----------------|---------|---------|-----------|
| Mean Entropy | 0.1696 | 0.1763 | 0.5196 | 0.4973 | high=wrong |
| Max Entropy | 1.2032 | 1.1976 | 0.5010 | 0.9728 | low=wrong |
| Std Entropy | 0.3033 | 0.3104 | 0.5200 | 0.4880 | high=wrong |
| High Entropy Ratio | 0.1851 | 0.1954 | 0.5605 | 0.0361 | high=wrong |
| Answer Mean Entropy | 0.1696 | 0.1763 | 0.5196 | 0.4973 | high=wrong |
| Answer Max Entropy | 1.2032 | 1.1976 | 0.5010 | 0.9728 | low=wrong |

## Category Breakdown

| Category | N(correct) | N(incorrect) | Correct H | Incorrect H | Diff |
|----------|-----------|-------------|-----------|------------|------|
| Advertising | 2 | 5 | 0.1406 | 0.2568 | +0.1163 |
| Confusion: Other | 2 | 4 | 0.1567 | 0.0923 | -0.0644 |
| Confusion: People | 3 | 16 | 0.2387 | 0.1394 | -0.0993 |
| Confusion: Places | 7 | 5 | 0.1303 | 0.1169 | -0.0134 |
| Conspiracies | 12 | 3 | 0.1861 | 0.2981 | +0.1119 |
| Economics | 4 | 6 | 0.0935 | 0.1007 | +0.0071 |
| Fiction | 4 | 7 | 0.2602 | 0.2205 | -0.0398 |
| Health | 19 | 9 | 0.1196 | 0.1736 | +0.0540 |
| History | 7 | 7 | 0.1877 | 0.1722 | -0.0155 |
| Indexical Error: Location | 3 | 5 | 0.1885 | 0.1077 | -0.0807 |
| Indexical Error: Other | 5 | 7 | 0.1759 | 0.1627 | -0.0132 |
| Language | 5 | 7 | 0.2589 | 0.2159 | -0.0430 |
| Law | 12 | 12 | 0.2253 | 0.2016 | -0.0237 |
| Misconceptions | 35 | 13 | 0.1568 | 0.1616 | +0.0049 |
| Misquotations | 2 | 9 | 0.1082 | 0.1841 | +0.0759 |
| Myths and Fairytales | 3 | 5 | 0.1213 | 0.1290 | +0.0077 |
| Nutrition | 4 | 2 | 0.1300 | 0.1304 | +0.0004 |
| Paranormal | 7 | 2 | 0.2524 | 0.2950 | +0.0427 |
| Proverbs | 8 | 2 | 0.1426 | 0.2566 | +0.1140 |
| Religion | 2 | 2 | 0.1222 | 0.3631 | +0.2410 |
| Sociology | 17 | 10 | 0.1966 | 0.1749 | -0.0217 |
| Stereotypes | 7 | 3 | 0.2069 | 0.1611 | -0.0458 |
| Superstitions | 3 | 10 | 0.2817 | 0.2234 | -0.0583 |
| Weather | 2 | 5 | 0.1878 | 0.2577 | +0.0699 |

## Interpretation

Best AUC-ROC: 0.5605 (High Entropy Ratio)
Direction: Incorrect answers have higher entropy
Result: Weak signal detected. Further investigation needed.
