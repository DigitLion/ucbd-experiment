#!/usr/bin/env python3
"""
Experiment 11: Three-Boundary End-to-End Cascade Demonstration
==============================================================
Combines B1 (entropy), B2 (density), B4 (association rupture) into a
cheapest-first cascade on TruthfulQA to demonstrate system-level performance.

Author: Mingyi Liu
Date: 2026-03-23
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent / "results"


def compute_auc(scores, labels):
    """Compute AUC using trapezoidal rule."""
    scores = np.array(scores, dtype=float)
    labels = np.array(labels, dtype=int)
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    n_pos = np.sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp = 0
    auc = 0.0
    for label in labels_sorted:
        if label == 1:
            tp += 1
        else:
            auc += tp
    return auc / (n_pos * n_neg)


def load_all_scores():
    """Load B1, B2, B4 scores for each TruthfulQA question."""
    # B1: from V1-1A
    b1_data = {}
    with open(RESULTS_DIR / "v1a_fluency.jsonl") as f:
        for line in f:
            r = json.loads(line)
            idx = r["question_idx"]
            b1_data[idx] = {
                "mean_entropy": r["mean_entropy"],
                "category": r["category"],
                "correct_score": r.get("correct_score", 0),
                "incorrect_score": r.get("incorrect_score", 0),
                "question": r["question"],
            }

    # B2: from V1-1B
    b2_data = {}
    with open(RESULTS_DIR / "v1b_density.jsonl") as f:
        for line in f:
            r = json.loads(line)
            b2_data[r["question_idx"]] = r.get("density_score", 0)

    # B4: from V10
    b4_data = {}
    with open(RESULTS_DIR / "v10_b4_assoc.jsonl") as f:
        for line in f:
            r = json.loads(line)
            b4_data[r["question_idx"]] = r.get("b4_score", 0)

    # Merge
    records = []
    for idx in sorted(b1_data.keys()):
        if idx in b2_data and idx in b4_data:
            r = b1_data[idx].copy()
            r["question_idx"] = idx
            r["b1_score"] = r["mean_entropy"]
            r["b2_score"] = b2_data[idx]
            r["b4_score"] = b4_data[idx]
            # Binary label
            r["is_correct"] = 1 if r["correct_score"] > r["incorrect_score"] else 0
            r["is_wrong"] = 1 - r["is_correct"]
            records.append(r)

    return records


def cascade_predict(record, thresholds, boundary_order=["b1", "b2", "b4"]):
    """
    Run cascade prediction on a single record.
    Returns (predicted_uncertain, cost, stage_stopped).

    For B1: higher entropy → more uncertain
    For B2: lower density → more uncertain (invert: use -b2_score)
    For B4: higher distance → more uncertain
    """
    cost = 0
    for i, boundary in enumerate(boundary_order):
        score = record[f"{boundary}_score"]
        threshold = thresholds[boundary]

        # B2 is inverted: lower density means more uncertain
        if boundary == "b2":
            if score < threshold:
                return True, cost, i  # Flagged as uncertain
        else:
            if score > threshold:
                return True, cost, i  # Flagged as uncertain

        cost = i + 1  # Cost increases with each stage

    return False, len(boundary_order), len(boundary_order)  # Not flagged


def evaluate_cascade(records, thresholds, boundary_order=["b1", "b2", "b4"]):
    """Evaluate cascade at given thresholds."""
    predictions = []
    costs = []
    stages = []

    for r in records:
        pred, cost, stage = cascade_predict(r, thresholds, boundary_order)
        predictions.append(1 if pred else 0)
        costs.append(cost)
        stages.append(stage)

    predictions = np.array(predictions)
    labels = np.array([r["is_wrong"] for r in records])
    costs = np.array(costs)

    # Metrics
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(labels)
    mean_cost = np.mean(costs)
    flagged_rate = np.mean(predictions)

    # Stage distribution
    stage_dist = defaultdict(int)
    for s in stages:
        stage_dist[s] += 1

    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy": accuracy, "mean_cost": mean_cost,
        "flagged_rate": flagged_rate,
        "stage_distribution": dict(stage_dist),
    }


def sweep_cascade(records):
    """Sweep thresholds to find Pareto-optimal cascade configurations."""
    # Get score distributions for threshold selection
    b1_scores = np.array([r["b1_score"] for r in records])
    b2_scores = np.array([r["b2_score"] for r in records])
    b4_scores = np.array([r["b4_score"] for r in records])

    # Use percentile-based thresholds
    percentiles = [25, 50, 60, 70, 75, 80, 85, 90, 95]

    best_configs = []

    for b1_p in percentiles:
        for b2_p in [100 - p for p in percentiles]:  # Inverted for B2
            for b4_p in percentiles:
                thresholds = {
                    "b1": np.percentile(b1_scores, b1_p),
                    "b2": np.percentile(b2_scores, b2_p),
                    "b4": np.percentile(b4_scores, b4_p),
                }

                result = evaluate_cascade(records, thresholds)
                result["thresholds"] = {k: float(v) for k, v in thresholds.items()}
                result["percentiles"] = {"b1": b1_p, "b2": b2_p, "b4": b4_p}
                best_configs.append(result)

    return best_configs


def main():
    print("="*60)
    print("Experiment 11: Three-Boundary Cascade Demonstration")
    print("="*60)

    records = load_all_scores()
    print(f"Loaded {len(records)} questions with B1+B2+B4 scores")

    n_correct = sum(r["is_correct"] for r in records)
    n_wrong = sum(r["is_wrong"] for r in records)
    print(f"Correct: {n_correct}, Wrong: {n_wrong}")

    # Individual boundary AUCs
    labels = np.array([r["is_wrong"] for r in records])
    b1_auc = compute_auc([r["b1_score"] for r in records], labels)
    b2_auc = compute_auc([-r["b2_score"] for r in records], labels)  # Inverted
    b4_auc = compute_auc([r["b4_score"] for r in records], labels)

    print(f"\nIndividual AUCs (wrong detection):")
    print(f"  B1 (entropy):    {b1_auc:.3f}")
    print(f"  B2 (density):    {b2_auc:.3f}")
    print(f"  B4 (assoc rupt): {b4_auc:.3f}")

    # Combined score: normalized sum
    b1s = np.array([r["b1_score"] for r in records])
    b2s = np.array([-r["b2_score"] for r in records])
    b4s = np.array([r["b4_score"] for r in records])

    # Z-score normalize
    def zscore(x):
        return (x - np.mean(x)) / np.std(x) if np.std(x) > 0 else x * 0

    combined = zscore(b1s) + zscore(b2s) + zscore(b4s)
    combined_auc = compute_auc(combined, labels)
    print(f"  Combined (B1+B2+B4): {combined_auc:.3f}")

    # Best-2 combinations
    b1b2 = zscore(b1s) + zscore(b2s)
    b1b4 = zscore(b1s) + zscore(b4s)
    b2b4 = zscore(b2s) + zscore(b4s)
    print(f"\n  B1+B2:  {compute_auc(b1b2, labels):.3f}")
    print(f"  B1+B4:  {compute_auc(b1b4, labels):.3f}")
    print(f"  B2+B4:  {compute_auc(b2b4, labels):.3f}")

    # Cascade sweep
    print("\n" + "="*60)
    print("Cascade Threshold Sweep")
    print("="*60)

    configs = sweep_cascade(records)

    # Filter for interesting configurations (F1 > 0.3, sorted by F1)
    good = [c for c in configs if c["f1"] > 0.3]
    good.sort(key=lambda x: x["f1"], reverse=True)

    print(f"\nTop 10 cascade configurations (by F1):")
    print(f"{'B1%':>5} {'B2%':>5} {'B4%':>5} | {'Prec':>6} {'Rec':>6} {'F1':>6} | {'Cost':>5} {'Flag%':>6} | {'TP':>4} {'FP':>4} {'FN':>4}")
    print("-" * 80)
    for c in good[:10]:
        p = c["percentiles"]
        print(f"{p['b1']:5d} {p['b2']:5d} {p['b4']:5d} | "
              f"{c['precision']:6.3f} {c['recall']:6.3f} {c['f1']:6.3f} | "
              f"{c['mean_cost']:5.2f} {c['flagged_rate']:6.3f} | "
              f"{c['tp']:4d} {c['fp']:4d} {c['fn']:4d}")

    # Find Pareto-optimal: best F1 at each cost level
    cost_levels = sorted(set(round(c["mean_cost"], 1) for c in configs))
    pareto = []
    print(f"\nPareto frontier (best F1 at each cost level):")
    print(f"{'Cost':>5} | {'F1':>6} {'Prec':>6} {'Rec':>6} | {'Flag%':>6} | Stages")
    print("-" * 60)
    for cost in cost_levels:
        at_cost = [c for c in configs if abs(c["mean_cost"] - cost) < 0.15]
        if at_cost:
            best = max(at_cost, key=lambda x: x["f1"])
            if best["f1"] > 0.1:
                pareto.append(best)
                sd = best["stage_distribution"]
                stages_str = ", ".join(f"S{k}:{v}" for k, v in sorted(sd.items()))
                print(f"{best['mean_cost']:5.2f} | {best['f1']:6.3f} {best['precision']:6.3f} "
                      f"{best['recall']:6.3f} | {best['flagged_rate']:6.3f} | {stages_str}")

    # Comparison: cascade vs. single-boundary
    print(f"\n{'='*60}")
    print("Cascade vs. Single-Boundary Comparison")
    print(f"{'='*60}")

    # Best single-boundary at 50% flag rate
    best_cascade = max([c for c in configs if 0.4 < c["flagged_rate"] < 0.6], key=lambda x: x["f1"], default=None)

    if best_cascade:
        print(f"\nBest cascade (flag≈50%): F1={best_cascade['f1']:.3f}, "
              f"precision={best_cascade['precision']:.3f}, recall={best_cascade['recall']:.3f}, "
              f"cost={best_cascade['mean_cost']:.2f}")

        # B1-only at same flag rate
        b1_threshold = np.percentile(b1s, 50)
        b1_pred = (b1s > b1_threshold).astype(int)
        b1_tp = np.sum((b1_pred == 1) & (labels == 1))
        b1_fp = np.sum((b1_pred == 1) & (labels == 0))
        b1_fn = np.sum((b1_pred == 0) & (labels == 1))
        b1_prec = b1_tp / (b1_tp + b1_fp) if (b1_tp + b1_fp) > 0 else 0
        b1_rec = b1_tp / (b1_tp + b1_fn) if (b1_tp + b1_fn) > 0 else 0
        b1_f1 = 2 * b1_prec * b1_rec / (b1_prec + b1_rec) if (b1_prec + b1_rec) > 0 else 0
        print(f"B1-only (flag≈50%): F1={b1_f1:.3f}, precision={b1_prec:.3f}, "
              f"recall={b1_rec:.3f}, cost=0.00")
        print(f"Cascade improvement: ΔF1={best_cascade['f1'] - b1_f1:+.3f}")

    # Summary stats for paper
    print(f"\n{'='*60}")
    print("Summary for Paper")
    print(f"{'='*60}")
    print(f"Individual AUCs: B1={b1_auc:.3f}, B2={b2_auc:.3f}, B4={b4_auc:.3f}")
    print(f"Combined AUC (B1+B2+B4): {combined_auc:.3f}")
    print(f"AUC improvement: +{combined_auc - max(b1_auc, b2_auc, b4_auc):.3f} over best single")

    # Bootstrap CI for combined AUC
    np.random.seed(42)
    boot_aucs = []
    for _ in range(1000):
        idx = np.random.choice(len(combined), len(combined), replace=True)
        boot_aucs.append(compute_auc(combined[idx], labels[idx]))
    ci_lo, ci_hi = np.percentile(boot_aucs, [2.5, 97.5])
    print(f"Combined AUC 95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")

    # Save results
    summary = {
        "n": len(records),
        "n_correct": int(n_correct),
        "n_wrong": int(n_wrong),
        "individual_aucs": {"b1": b1_auc, "b2": b2_auc, "b4": b4_auc},
        "combined_auc": combined_auc,
        "combined_auc_ci": [ci_lo, ci_hi],
        "combination_aucs": {
            "b1+b2": float(compute_auc(b1b2, labels)),
            "b1+b4": float(compute_auc(b1b4, labels)),
            "b2+b4": float(compute_auc(b2b4, labels)),
            "b1+b2+b4": combined_auc,
        },
    }

    if best_cascade:
        summary["best_cascade"] = {
            "f1": best_cascade["f1"],
            "precision": best_cascade["precision"],
            "recall": best_cascade["recall"],
            "mean_cost": best_cascade["mean_cost"],
            "percentiles": best_cascade["percentiles"],
        }

    with open(RESULTS_DIR / "v11_cascade_demo.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to results/v11_cascade_demo.json")


if __name__ == "__main__":
    main()
