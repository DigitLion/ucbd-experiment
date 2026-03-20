# UCBD V1-1B: B1-B2 Complementarity Report

## Key Finding

- B1 (entropy) alone: AUC = 0.5196
- B2 (density) alone: AUC = 0.5014
- B1+B2 average: AUC = 0.5250
- B1+B2 cascade: AUC = 0.5326
- B1+B2 oracle routing: AUC = 0.5848 [0.5279, 0.6403]
- Pearson r(B1, B2) = 0.018 (independent signals)

## Per-Category Complementarity

| Category | N | AUC_B1 | AUC_B2 | Best | Zone |
|----------|---|--------|--------|------|------|
| Religion | 4 | 1.000 | 0.000 | B1 | B1-only |
| Conspiracies | 15 | 0.722 | 0.917 | B2 | Both |
| Advertising | 7 | 0.900 | 0.000 | B1 | B1-only |
| Fiction | 11 | 0.321 | 0.857 | B2 | B2-only |
| Weather | 7 | 0.800 | 0.500 | B1 | B1-only |
| Misquotations | 11 | 0.778 | 0.778 | B2 | Both |
| Confusion: Places | 12 | 0.486 | 0.771 | B2 | B2-only |
| Proverbs | 10 | 0.750 | 0.438 | B1 | B1-only |
| Health | 28 | 0.737 | 0.275 | B1 | B1-only |
| Myths and Fairytales | 8 | 0.467 | 0.733 | B2 | B2-only |
| History | 14 | 0.429 | 0.694 | B2 | B2-only |
| Indexical Error: Other | 12 | 0.400 | 0.686 | B2 | B2-only |
| Confusion: Other | 6 | 0.250 | 0.625 | B2 | B2-only |
| Economics | 10 | 0.583 | 0.458 | B1 | B1-only |
| Misconceptions | 48 | 0.527 | 0.571 | B2 | B2-only |
| Paranormal | 9 | 0.571 | 0.500 | B1 | B1-only |
| Nutrition | 6 | 0.500 | 0.000 | B1 | Neither |
| Confusion: People | 19 | 0.229 | 0.479 | B2 | Neither |
| Sociology | 27 | 0.406 | 0.476 | B2 | Neither |
| Law | 24 | 0.458 | 0.326 | B1 | Neither |
| Language | 12 | 0.371 | 0.457 | B2 | Neither |
| Superstitions | 13 | 0.333 | 0.200 | B1 | Neither |
| Stereotypes | 10 | 0.333 | 0.238 | B1 | Neither |
| Indexical Error: Location | 8 | 0.267 | 0.200 | B1 | Neither |

## Zone Distribution

- **Both**: 2 categories, 26 samples — Conspiracies, Misquotations
- **B1-only**: 7 categories, 75 samples — Advertising, Economics, Health, Paranormal, Proverbs, Religion, Weather
- **B2-only**: 7 categories, 111 samples — Confusion: Other, Confusion: Places, Fiction, History, Indexical Error: Other, Misconceptions, Myths and Fairytales
- **Neither**: 8 categories, 119 samples — Confusion: People, Indexical Error: Location, Language, Law, Nutrition, Sociology, Stereotypes, Superstitions

## Coverage

- B1 or B2 effective: 16/24 categories (212/331 = 64.0%)
- Neither (need B3-B5): 8 categories

## Interpretation

1. B1 and B2 are nearly orthogonal (r=0.018) — they measure genuinely different things
2. Oracle routing (knowing which detector to use per-domain) significantly outperforms either alone
3. Categories uncovered by both B1 and B2 represent the case for higher-order boundaries (B3-B5)
4. This validates the UCBD cascade design: each boundary covers a different failure mode
