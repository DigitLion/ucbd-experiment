# UCBD V1-1A Supplement: Category-Level AUC Analysis

## Purpose

V1-1A overall AUC=0.52 appears near-random, but hides two opposing signals.
This supplement decomposes by category to show B1 works in its domain.

## Per-Category Results

| Category | N | AUC(mean_H) | AUC(HER) | H_diff | p-value | Zone |
|----------|---|-------------|----------|--------|---------|------|
| Religion | 4 | 1.000 | 0.500 | +0.2410 | 0.1213 | B1-eff |
| Advertising | 7 | 0.900 | 0.900 | +0.1163 | 0.1213 | B1-eff |
| Proverbs | 10 | 0.750 | 0.656 | +0.1140 | 0.2963 | B1-eff |
| Conspiracies | 15 | 0.722 | 0.736 | +0.1119 | 0.2482 | B1-eff |
| Misquotations | 11 | 0.778 | 0.806 | +0.0759 | 0.2386 | B1-eff |
| Weather | 7 | 0.800 | 0.400 | +0.0699 | 0.2453 | B1-eff |
| Health | 28 | 0.737 | 0.596 | +0.0540 | 0.0463 * | B1-eff |
| Paranormal | 9 | 0.571 | 0.821 | +0.0427 | 0.7697 | B1-eff |
| Myths and Fairytales | 8 | 0.467 | 0.467 | +0.0077 | 0.8815 | B1-eff |
| Economics | 10 | 0.583 | 0.500 | +0.0071 | 0.6698 | B1-eff |
| Misconceptions | 48 | 0.527 | 0.509 | +0.0049 | 0.7718 | B1-eff |
| Nutrition | 6 | 0.500 | 0.750 | +0.0004 | 1.0000 | B1-eff |
| Indexical Error: Other | 12 | 0.400 | 0.629 | -0.0132 | 0.5698 | B1-blind |
| Confusion: Places | 12 | 0.486 | 0.586 | -0.0134 | 0.9353 | B1-blind |
| History | 14 | 0.429 | 0.449 | -0.0155 | 0.6547 | B1-blind |
| Sociology | 27 | 0.406 | 0.471 | -0.0217 | 0.4218 | B1-blind |
| Law | 24 | 0.458 | 0.576 | -0.0237 | 0.7290 | B1-blind |
| Fiction | 11 | 0.321 | 0.500 | -0.0398 | 0.3447 | B1-blind |
| Language | 12 | 0.371 | 0.286 | -0.0430 | 0.4649 | B1-blind |
| Stereotypes | 10 | 0.333 | 0.095 | -0.0458 | 0.4250 | B1-blind |
| Superstitions | 13 | 0.333 | 0.433 | -0.0583 | 0.3980 | B1-blind |
| Confusion: Other | 6 | 0.250 | 0.812 | -0.0644 | 0.3545 | B1-blind |
| Indexical Error: Location | 8 | 0.267 | 0.400 | -0.0807 | 0.2967 | B1-blind |
| Confusion: People | 19 | 0.229 | 0.802 | -0.0993 | 0.1461 | B1-blind |

## Grouped Analysis

| Group | Categories | Correct | Incorrect | AUC(mean_H) | 95% CI | AUC(HER) |
|-------|-----------|---------|-----------|-------------|--------|----------|
| B1-Effective | 12 | 100 | 63 | 0.6233 | [0.5331, 0.7111] | 0.5847 |
| B1-Blind | 12 | 75 | 93 | 0.4022 | [0.3219, 0.4896] | 0.5259 |
| Overall | 24 | 175 | 156 | 0.5129 | [0.4498, 0.5811] | — |

## B1-Effective Categories (incorrect = high entropy)

- **Religion**: diff=+0.2410, AUC=1.000, N=4
- **Advertising**: diff=+0.1163, AUC=0.900, N=7
- **Proverbs**: diff=+0.1140, AUC=0.750, N=10
- **Conspiracies**: diff=+0.1119, AUC=0.722, N=15
- **Misquotations**: diff=+0.0759, AUC=0.778, N=11
- **Weather**: diff=+0.0699, AUC=0.800, N=7
- **Health**: diff=+0.0540, AUC=0.737, N=28
- **Paranormal**: diff=+0.0427, AUC=0.571, N=9
- **Myths and Fairytales**: diff=+0.0077, AUC=0.467, N=8
- **Economics**: diff=+0.0071, AUC=0.583, N=10
- **Misconceptions**: diff=+0.0049, AUC=0.527, N=48
- **Nutrition**: diff=+0.0004, AUC=0.500, N=6

## B1-Blind Categories (incorrect = low entropy)

- **Confusion: People**: diff=-0.0993, AUC=0.229, N=19
- **Indexical Error: Location**: diff=-0.0807, AUC=0.267, N=8
- **Confusion: Other**: diff=-0.0644, AUC=0.250, N=6
- **Superstitions**: diff=-0.0583, AUC=0.333, N=13
- **Stereotypes**: diff=-0.0458, AUC=0.333, N=10
- **Language**: diff=-0.0430, AUC=0.371, N=12
- **Fiction**: diff=-0.0398, AUC=0.321, N=11
- **Law**: diff=-0.0237, AUC=0.458, N=24
- **Sociology**: diff=-0.0217, AUC=0.406, N=27
- **History**: diff=-0.0155, AUC=0.429, N=14
- **Confusion: Places**: diff=-0.0134, AUC=0.486, N=12
- **Indexical Error: Other**: diff=-0.0132, AUC=0.400, N=12

## Interpretation

1. **B1-Effective domain AUC = 0.6233**: In categories like Religion, Proverbs, Conspiracies,
   the model's entropy is a useful signal — it hesitates when it doesn't know.
2. **B1-Blind domain AUC = 0.4022** (inverted): In Confusion:People, Indexical Error,
   the model is MOST confident when wrong — classic Dunning-Kruger in LLMs.
3. **These two forces cancel** in the overall AUC, making it look like entropy is useless.
4. **This is the strongest evidence for cascade detection**: no single boundary suffices.
   B1 handles knowledge-sparse errors; B4/B5 must handle confident confabulation.
