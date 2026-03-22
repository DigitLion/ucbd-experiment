#!/usr/bin/env python3
"""
Experiment 10: B4 Association Rupture (Toy Validation)
======================================================
Tests whether entity-pair embedding distance (proxy for KG incompleteness)
predicts model errors, especially in B1+B2 blind zones.

Method:
  1. Extract named entities from TruthfulQA questions (regex-based NER)
  2. Embed each entity with qwen3-embedding (Ollama)
  3. B4_score = mean pairwise cosine distance between entity embeddings
  4. Correlate B4_score with correctness (from V1-1A word-overlap judge)
  5. Analyze B4 effectiveness in B1+B2 blind zones

Author: Mingyi Liu
Date: 2026-03-23
"""

import json, re, time, sys
import numpy as np
import requests
from pathlib import Path
from collections import defaultdict

OLLAMA_URL = "http://127.0.0.1:11434/api/embed"
RESULTS_DIR = Path(__file__).parent / "results"
DATA_DIR = Path(__file__).parent / "data"

# B1+B2 blind zone categories (from V1-1B complement analysis)
B1B2_BLIND = {
    "Confusion: People", "Stereotypes", "Law", "Superstitions",
    "Logical Falsehood", "Psychology", "Sociology", "Education"
}

# B1-effective categories
B1_EFFECTIVE = {
    "Religion", "Advertising", "Misquotations", "Proverbs",
    "Health", "Conspiracies", "Nutrition", "Weather"
}


STOPWORDS = {
    'the', 'a', 'an', 'in', 'on', 'at', 'for', 'and', 'but', 'or', 'is', 'are',
    'was', 'were', 'has', 'have', 'do', 'does', 'did', 'can', 'could', 'should',
    'would', 'will', 'may', 'might', 'if', 'so', 'yet', 'not', 'no', 'yes',
    'its', 'it', 'he', 'she', 'they', 'we', 'you', 'my', 'your', 'his', 'her',
    'our', 'their', 'this', 'that', 'these', 'those', 'some', 'any', 'all',
    'each', 'every', 'many', 'much', 'most', 'few', 'more', 'less', 'very',
    'also', 'just', 'only', 'even', 'still', 'how', 'why', 'what', 'when',
    'where', 'who', 'which', 'been', 'being', 'had', 'having', 'get', 'got',
    'than', 'then', 'there', 'here', 'with', 'from', 'about', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'between', 'under', 'over',
    'to', 'of', 'by', 'up', 'down', 'out', 'off', 'such', 'other', 'because',
    'while', 'until', 'against', 'both', 'same', 'own', 'too', 'well', 'back',
    'really', 'actually', 'true', 'false', 'people', 'person', 'thing', 'things',
    'way', 'time', 'year', 'day', 'world', 'life', 'something', 'nothing',
    'everything', 'anything', 'someone', 'everyone', 'anyone', 'nobody',
}

# Common question words/phrases to strip
Q_STARTERS = re.compile(
    r'^(What|Who|Where|When|Why|How|Which|Is|Are|Do|Does|Did|Can|Could|Should|'
    r'Was|Were|Has|Have|Will|Would|If)\b\s*', re.I
)


def extract_entities(question: str) -> list[str]:
    """Extract named entities + key concepts from a question using multi-strategy NER."""
    entities = []

    # Strategy 1: Multi-word capitalized phrases (proper nouns)
    for match in re.finditer(
        r'\b([A-Z][a-zA-Z\']+(?:\s+(?:of|the|and|de|von|van|al|el|la|le|for|in)\s+)?'
        r'(?:[A-Z][a-zA-Z\']+)(?:\s+[A-Z][a-zA-Z\']+)*)\b', question
    ):
        ent = match.group(1).strip()
        if len(ent) > 2 and ent.lower() not in STOPWORDS:
            entities.append(ent)

    # Strategy 2: Single capitalized words anywhere in question
    for match in re.finditer(r'\b([A-Z][a-z]{2,})\b', question):
        word = match.group(1)
        if word.lower() not in STOPWORDS and word not in entities:
            # Check it's not a sub-entity of an already captured multi-word
            if not any(word in e for e in entities):
                entities.append(word)

    # Strategy 3: Quoted phrases
    for match in re.finditer(r'"([^"]+)"', question):
        entities.append(match.group(1))

    # Strategy 4: Key noun phrases — extract subject + object of the question
    # Remove question starter and extract remaining noun-like chunks
    stripped = Q_STARTERS.sub('', question).rstrip('?').strip()
    # Split on common delimiters and take meaningful chunks
    chunks = re.split(r'\s+(?:if|when|that|which|who|because|and|or|but|than)\s+', stripped, flags=re.I)
    for chunk in chunks:
        # Extract noun phrases: sequences of non-stopwords
        words = chunk.split()
        phrase = []
        for w in words:
            clean = re.sub(r'[^\w\'-]', '', w)
            if clean.lower() not in STOPWORDS and len(clean) > 2:
                phrase.append(clean)
            else:
                if len(phrase) >= 2:
                    np_text = ' '.join(phrase)
                    if np_text not in entities and not any(np_text in e for e in entities):
                        entities.append(np_text)
                phrase = []
        if len(phrase) >= 2:
            np_text = ' '.join(phrase)
            if np_text not in entities:
                entities.append(np_text)

    # Strategy 5: If still < 2 entities, extract individual content words as concepts
    if len(entities) < 2:
        stripped = Q_STARTERS.sub('', question).rstrip('?').strip()
        for w in stripped.split():
            clean = re.sub(r'[^\w\'-]', '', w)
            if clean.lower() not in STOPWORDS and len(clean) > 3 and clean not in entities:
                entities.append(clean)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for e in entities:
        key = e.lower()
        if key not in seen:
            seen.add(key)
            unique.append(e)

    return unique[:8]  # Cap at 8 entities per question


def embed_text(text: str) -> np.ndarray:
    """Get embedding from Ollama qwen3-embedding."""
    resp = requests.post(OLLAMA_URL, json={"model": "qwen3-embedding", "input": text})
    resp.raise_for_status()
    return np.array(resp.json()["embeddings"][0])


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance = 1 - cosine_similarity."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 1.0
    return 1.0 - dot / norm


def compute_b4_score(entity_embeddings: list[np.ndarray]) -> float:
    """Compute B4 association rupture score as mean pairwise cosine distance."""
    if len(entity_embeddings) < 2:
        return 0.0  # Can't compute pairwise distance with < 2 entities

    distances = []
    for i in range(len(entity_embeddings)):
        for j in range(i + 1, len(entity_embeddings)):
            distances.append(cosine_distance(entity_embeddings[i], entity_embeddings[j]))

    return float(np.mean(distances))


def load_v1a_data() -> list[dict]:
    """Load V1-1A fluency experiment results."""
    records = []
    with open(RESULTS_DIR / "v1a_fluency.jsonl") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def analyze_results(records: list[dict]):
    """Analyze B4 association rupture results."""
    # Filter to records with B4 scores (need >= 2 entities)
    valid = [r for r in records if r.get("b4_score", 0) > 0]
    print(f"\n{'='*60}")
    print(f"B4 Association Rupture Analysis")
    print(f"{'='*60}")
    print(f"Total questions: {len(records)}")
    print(f"Questions with ≥2 entities: {len(valid)} ({100*len(valid)/len(records):.1f}%)")

    if len(valid) < 20:
        print("Too few valid questions for analysis")
        return {}

    # Basic stats
    b4_scores = np.array([r["b4_score"] for r in valid])
    correct = np.array([1 if r.get("correct_score", 0) > r.get("incorrect_score", 0) else 0 for r in valid])

    print(f"\nB4 score: mean={np.mean(b4_scores):.4f}, std={np.std(b4_scores):.4f}")
    print(f"Correct: {np.sum(correct)}/{len(correct)} ({100*np.mean(correct):.1f}%)")

    # B4 score by correctness
    correct_b4 = b4_scores[correct == 1]
    incorrect_b4 = b4_scores[correct == 0]
    print(f"\nB4 score (correct):   mean={np.mean(correct_b4):.4f}")
    print(f"B4 score (incorrect): mean={np.mean(incorrect_b4):.4f}")
    print(f"Δ = {np.mean(incorrect_b4) - np.mean(correct_b4):.4f} (positive = B4 higher for wrong)")

    # AUC computation
    overall_auc = compute_auc(b4_scores, 1 - correct)  # Higher B4 → more likely wrong
    print(f"\nOverall B4 AUC (wrong detection): {overall_auc:.3f}")

    # Point-biserial correlation
    # Manual computation (avoid scipy dependency)
    x = 1 - correct
    y = b4_scores
    n = len(x)
    n1 = np.sum(x == 1)
    n0 = n - n1
    mean1 = np.mean(y[x == 1]) if n1 > 0 else 0
    mean0 = np.mean(y[x == 0]) if n0 > 0 else 0
    sy = np.std(y, ddof=1)
    r_pb = (mean1 - mean0) / sy * np.sqrt(n1 * n0 / (n * n)) if sy > 0 else 0
    # t-test for significance
    t_stat = r_pb * np.sqrt((n - 2) / (1 - r_pb**2)) if abs(r_pb) < 1 else 0
    # Approximate p-value using normal distribution for large n
    p_pb = 2 * (1 - 0.5 * (1 + np.sign(abs(t_stat)) * (1 - np.exp(-0.7 * abs(t_stat)))))
    print(f"Point-biserial r: {r_pb:.4f}, p={p_pb:.4f}")

    # B1+B2 blind zone analysis
    blind_records = [r for r in valid if r["category"] in B1B2_BLIND]
    effective_records = [r for r in valid if r["category"] in B1_EFFECTIVE]

    results = {
        "n_total": len(records),
        "n_valid": len(valid),
        "overall_auc": overall_auc,
        "r_pb": r_pb,
        "p_pb": p_pb,
    }

    if len(blind_records) >= 10:
        blind_b4 = np.array([r["b4_score"] for r in blind_records])
        blind_correct = np.array([1 if r.get("correct_score", 0) > r.get("incorrect_score", 0) else 0 for r in blind_records])
        blind_auc = compute_auc(blind_b4, 1 - blind_correct)
        print(f"\nB1+B2 blind zone ({len(blind_records)} questions):")
        print(f"  B4 AUC: {blind_auc:.3f}")
        print(f"  Categories: {set(r['category'] for r in blind_records)}")
        results["blind_zone_auc"] = blind_auc
        results["blind_zone_n"] = len(blind_records)

    if len(effective_records) >= 10:
        eff_b4 = np.array([r["b4_score"] for r in effective_records])
        eff_correct = np.array([1 if r.get("correct_score", 0) > r.get("incorrect_score", 0) else 0 for r in effective_records])
        eff_auc = compute_auc(eff_b4, 1 - eff_correct)
        print(f"\nB1-effective zone ({len(effective_records)} questions):")
        print(f"  B4 AUC: {eff_auc:.3f}")
        results["effective_zone_auc"] = eff_auc
        results["effective_zone_n"] = len(effective_records)

    # Independence from B1
    b1_scores = np.array([r.get("mean_entropy", 0) for r in valid])
    # Manual Pearson r
    def pearson_r(x, y):
        n = len(x)
        mx, my = np.mean(x), np.mean(y)
        sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)
        if sx == 0 or sy == 0:
            return 0.0
        return np.sum((x - mx) * (y - my)) / ((n - 1) * sx * sy)

    r_b1b4 = pearson_r(b1_scores, b4_scores)
    print(f"\nB1-B4 independence: Pearson r={r_b1b4:.4f}")
    results["b1_b4_pearson_r"] = r_b1b4

    # B2-B4 independence (load B2 density data)
    try:
        b2_data = {}
        with open(RESULTS_DIR / "v1b_density.jsonl") as f:
            for line in f:
                rec = json.loads(line)
                b2_data[rec["question_idx"]] = rec.get("density_score", 0)

        b2_scores = np.array([b2_data.get(r["question_idx"], 0) for r in valid])
        r_b2b4 = pearson_r(b2_scores, b4_scores)
        print(f"B2-B4 independence: Pearson r={r_b2b4:.4f}")
        results["b2_b4_pearson_r"] = r_b2b4
    except Exception as e:
        print(f"B2 data not available: {e}")

    # Per-category analysis
    print(f"\n{'='*60}")
    print(f"Per-Category B4 AUC")
    print(f"{'='*60}")
    cat_data = defaultdict(lambda: {"b4": [], "correct": []})
    for r in valid:
        cat = r["category"]
        cat_data[cat]["b4"].append(r["b4_score"])
        cat_data[cat]["correct"].append(1 if r.get("correct_score", 0) > r.get("incorrect_score", 0) else 0)

    cat_aucs = {}
    for cat in sorted(cat_data.keys()):
        b4 = np.array(cat_data[cat]["b4"])
        cor = np.array(cat_data[cat]["correct"])
        n = len(b4)
        if n >= 5 and len(set(cor)) > 1:
            auc = compute_auc(b4, 1 - cor)
            zone = "BLIND" if cat in B1B2_BLIND else ("EFF" if cat in B1_EFFECTIVE else "other")
            print(f"  {cat:30s}  n={n:3d}  AUC={auc:.3f}  [{zone}]")
            cat_aucs[cat] = {"auc": auc, "n": n, "zone": zone}

    results["category_aucs"] = cat_aucs

    # Coverage: B4-effective categories (AUC > 0.55)
    b4_effective = [cat for cat, info in cat_aucs.items() if info["auc"] > 0.55]
    b4_blind_cats = [cat for cat, info in cat_aucs.items() if info["auc"] < 0.45]
    print(f"\nB4-effective categories (AUC>0.55): {len(b4_effective)}/{len(cat_aucs)}")
    print(f"B4-blind categories (AUC<0.45): {len(b4_blind_cats)}/{len(cat_aucs)}")

    # Key finding: B4 effective in B1+B2 blind zone?
    blind_effective = [cat for cat in b4_effective if cat in B1B2_BLIND]
    print(f"\nB4-effective AND in B1+B2 blind zone: {blind_effective}")
    results["b4_effective_in_blind"] = blind_effective

    # Union coverage: B1 ∪ B2 ∪ B4
    b1b2_covered = B1_EFFECTIVE | {
        "Fiction", "Confusion: Places", "Myths and Fairytales", "History",
        "Conspiracies", "Misquotations"
    }  # From V1-1B analysis
    b4_covered = set(b4_effective)
    union = b1b2_covered | b4_covered
    print(f"\nB1∪B2 coverage: {len(b1b2_covered)} categories")
    print(f"B4 coverage: {len(b4_covered)} categories")
    print(f"B1∪B2∪B4 coverage: {len(union)} categories")
    print(f"New categories from B4: {b4_covered - b1b2_covered}")
    results["union_coverage"] = len(union)
    results["new_from_b4"] = list(b4_covered - b1b2_covered)

    return results


# Simple AUC implementation (avoid sklearn dependency)
class sklearn_auc:
    @staticmethod
    def compute_auc(scores, labels):
        """Compute AUC using the trapezoidal rule."""
        # Sort by score descending
        order = np.argsort(-scores)
        labels_sorted = labels[order]

        n_pos = np.sum(labels)
        n_neg = len(labels) - n_pos

        if n_pos == 0 or n_neg == 0:
            return 0.5

        tp = 0
        fp = 0
        auc = 0.0
        prev_fp = 0

        for i in range(len(labels_sorted)):
            if labels_sorted[i] == 1:
                tp += 1
            else:
                fp += 1
                auc += tp

        return auc / (n_pos * n_neg)


# Make compute_auc available at module level
def compute_auc(scores, labels):
    return sklearn_auc.compute_auc(scores, np.array(labels))


def main():
    print("="*60)
    print("Experiment 10: B4 Association Rupture (Toy Validation)")
    print("="*60)

    # Load V1-1A data
    records = load_v1a_data()
    print(f"Loaded {len(records)} V1-1A records")

    # Phase 1: Extract entities from all questions
    print("\nPhase 1: Entity extraction...")
    entity_counts = []
    for r in records:
        entities = extract_entities(r["question"])
        r["entities"] = entities
        entity_counts.append(len(entities))

    ec = np.array(entity_counts)
    print(f"  Entity count: mean={np.mean(ec):.1f}, median={np.median(ec):.0f}")
    print(f"  Questions with ≥2 entities: {np.sum(ec >= 2)} ({100*np.mean(ec >= 2):.1f}%)")
    print(f"  Questions with 0 entities: {np.sum(ec == 0)} ({100*np.mean(ec == 0):.1f}%)")

    # Phase 2: Embed entities
    print("\nPhase 2: Embedding entities...")
    entity_cache = {}
    all_entities = set()
    for r in records:
        for e in r["entities"]:
            all_entities.add(e)

    print(f"  Unique entities: {len(all_entities)}")

    t0 = time.time()
    batch_size = 10
    entity_list = list(all_entities)

    for i in range(0, len(entity_list), batch_size):
        batch = entity_list[i:i+batch_size]
        for entity in batch:
            try:
                emb = embed_text(entity)
                entity_cache[entity] = emb
            except Exception as e:
                print(f"  Error embedding '{entity}': {e}")

        if (i + batch_size) % 100 == 0 or i + batch_size >= len(entity_list):
            elapsed = time.time() - t0
            print(f"  Embedded {min(i+batch_size, len(entity_list))}/{len(entity_list)} "
                  f"({elapsed:.1f}s)")

    print(f"  Total embedding time: {time.time()-t0:.1f}s")

    # Phase 3: Compute B4 scores
    print("\nPhase 3: Computing B4 scores...")
    for r in records:
        embeddings = [entity_cache[e] for e in r["entities"] if e in entity_cache]
        r["b4_score"] = compute_b4_score(embeddings)
        r["n_entities"] = len(r["entities"])

    # Phase 4: Analysis
    results = analyze_results(records)

    # Save results
    output_path = RESULTS_DIR / "v10_b4_assoc.jsonl"
    with open(output_path, "w") as f:
        for r in records:
            out = {
                "question_idx": r["question_idx"],
                "category": r["category"],
                "question": r["question"],
                "entities": r["entities"],
                "n_entities": r["n_entities"],
                "b4_score": r["b4_score"],
                "correct_score": r.get("correct_score", 0),
                "incorrect_score": r.get("incorrect_score", 0),
                "mean_entropy": r.get("mean_entropy", 0),
            }
            f.write(json.dumps(out) + "\n")

    summary_path = RESULTS_DIR / "v10_summary.json"
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(summary_path, "w") as f:
        json.dump({k: convert(v) if not isinstance(v, dict) else
                   {k2: convert(v2) if not isinstance(v2, dict) else v2
                    for k2, v2 in v.items()}
                   for k, v in results.items()}, f, indent=2, default=convert)

    print(f"\nResults saved to {output_path}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
