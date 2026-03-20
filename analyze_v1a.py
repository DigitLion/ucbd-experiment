#!/usr/bin/env python3
"""
UCBD V1-1A: Analyze fluency boundary experiment results.
Computes AUC-ROC, generates summary statistics, and creates visualizations.

Usage:
  python3 analyze_v1a.py                    # full analysis
  python3 analyze_v1a.py --no-plot          # stats only
"""

import json
import math
import argparse
from pathlib import Path
from collections import Counter

RESULTS_PATH = Path(__file__).parent / "results" / "v1a_fluency.jsonl"
REPORT_PATH = Path(__file__).parent / "results" / "v1a_report.md"


def load_results():
    results = []
    with open(RESULTS_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def compute_auc_roc(scores, labels):
    """
    Compute AUC-ROC from score-label pairs.
    Higher score = more likely positive (incorrect).
    Uses trapezoidal rule on the ROC curve.
    """
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    tp = fp = 0
    prev_score = None
    roc_points = [(0.0, 0.0)]

    for score, label in pairs:
        if score != prev_score and prev_score is not None:
            fpr = fp / n_neg
            tpr = tp / n_pos
            roc_points.append((fpr, tpr))
        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score

    roc_points.append((fp / n_neg, tp / n_pos))

    # Trapezoidal AUC
    auc = 0.0
    for i in range(1, len(roc_points)):
        dx = roc_points[i][0] - roc_points[i - 1][0]
        dy = (roc_points[i][1] + roc_points[i - 1][1]) / 2
        auc += dx * dy
    return auc


def mann_whitney_u(group_a, group_b):
    """Simple Mann-Whitney U test (two-sided)."""
    combined = [(v, "a") for v in group_a] + [(v, "b") for v in group_b]
    combined.sort(key=lambda x: x[0])

    # Assign ranks (handle ties with average rank)
    ranks = {}
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2  # 1-indexed average
        for k in range(i, j):
            ranks[id(combined[k])] = avg_rank
        i = j

    # Sum ranks for group a
    rank_sum_a = sum(ranks[id(x)] for x in combined if x[1] == "a")
    n_a = len(group_a)
    n_b = len(group_b)
    u_a = rank_sum_a - n_a * (n_a + 1) / 2
    u_b = n_a * n_b - u_a

    # Normal approximation for p-value
    mu = n_a * n_b / 2
    sigma = math.sqrt(n_a * n_b * (n_a + n_b + 1) / 12)
    if sigma == 0:
        return min(u_a, u_b), 1.0
    z = (min(u_a, u_b) - mu) / sigma
    # Two-sided p-value approximation using error function
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return min(u_a, u_b), p


def analyze(do_plot=True):
    results = load_results()
    print(f"=== UCBD V1-1A Analysis ===")
    print(f"Total results: {len(results)}")
    print()

    # Label distribution
    labels = Counter(r["label"] for r in results)
    print("Label distribution:")
    for k, v in labels.most_common():
        print(f"  {k}: {v} ({v/len(results)*100:.1f}%)")
    print()

    # Filter to binary labels for AUC analysis
    binary = [r for r in results if r["label"] in ("correct", "incorrect")]
    n_correct = sum(1 for r in binary if r["label"] == "correct")
    n_incorrect = sum(1 for r in binary if r["label"] == "incorrect")
    print(f"Binary subset: {len(binary)} ({n_correct} correct, {n_incorrect} incorrect)")
    print()

    if len(binary) < 10:
        print("ERROR: Too few binary samples for meaningful analysis.")
        return

    # Entropy statistics by label
    correct_data = [r for r in binary if r["label"] == "correct"]
    incorrect_data = [r for r in binary if r["label"] == "incorrect"]

    metrics = [
        ("mean_entropy", "Mean Entropy"),
        ("max_entropy", "Max Entropy"),
        ("std_entropy", "Std Entropy"),
        ("high_entropy_ratio", "High Entropy Ratio"),
        ("answer_mean_entropy", "Answer Mean Entropy"),
        ("answer_max_entropy", "Answer Max Entropy"),
    ]

    print("=" * 70)
    print(f"{'Metric':<25s} {'Correct':>10s} {'Incorrect':>10s} {'AUC-ROC':>10s} {'p-value':>10s}")
    print("=" * 70)

    report_lines = []
    report_lines.append("# UCBD V1-1A: Fluency Boundary Experiment Report\n")
    report_lines.append(f"Total questions: {len(results)}")
    report_lines.append(f"Binary subset: {len(binary)} (correct={n_correct}, incorrect={n_incorrect})")
    report_lines.append(f"Ambiguous: {labels.get('ambiguous',0)}, No match: {labels.get('no_match',0)}\n")
    report_lines.append("## Entropy vs Correctness\n")
    report_lines.append(f"| Metric | Correct (mean) | Incorrect (mean) | AUC-ROC | p-value | Direction |")
    report_lines.append(f"|--------|---------------|-----------------|---------|---------|-----------|")

    best_auc = 0.0
    best_metric = ""

    for key, name in metrics:
        c_vals = [r[key] for r in correct_data if key in r]
        i_vals = [r[key] for r in incorrect_data if key in r]

        if not c_vals or not i_vals:
            continue

        c_mean = sum(c_vals) / len(c_vals)
        i_mean = sum(i_vals) / len(i_vals)

        # AUC-ROC: score = metric value, label = 1 if incorrect
        scores = [r[key] for r in binary if key in r]
        bin_labels = [1 if r["label"] == "incorrect" else 0 for r in binary if key in r]
        auc = compute_auc_roc(scores, bin_labels)

        # If AUC < 0.5, the signal is inverted (low entropy = incorrect)
        # Report the "better" direction
        effective_auc = max(auc, 1 - auc)
        direction = "high=wrong" if auc >= 0.5 else "low=wrong"

        # Mann-Whitney U test
        _, p_val = mann_whitney_u(c_vals, i_vals)

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

        print(f"{name:<25s} {c_mean:>10.4f} {i_mean:>10.4f} {effective_auc:>10.4f} {p_val:>9.4f} {sig}")
        report_lines.append(
            f"| {name} | {c_mean:.4f} | {i_mean:.4f} | {effective_auc:.4f} | {p_val:.4f} | {direction} |"
        )

        if effective_auc > best_auc:
            best_auc = effective_auc
            best_metric = name

    print("=" * 70)
    print(f"\nBest discriminator: {best_metric} (AUC={best_auc:.4f})")
    print(f"Significance: * p<0.05, ** p<0.01, *** p<0.001")

    # Key finding
    c_mean_all = sum(r["mean_entropy"] for r in correct_data) / len(correct_data)
    i_mean_all = sum(r["mean_entropy"] for r in incorrect_data) / len(incorrect_data)
    direction = "higher" if i_mean_all > c_mean_all else "lower"
    print(f"\nKey finding: Incorrect answers have {direction} mean entropy than correct answers")
    print(f"  Correct: {c_mean_all:.4f}, Incorrect: {i_mean_all:.4f}")

    # Category-level analysis
    print("\n--- Category Breakdown ---")
    cats = {}
    for r in binary:
        cat = r["category"]
        if cat not in cats:
            cats[cat] = {"correct": [], "incorrect": []}
        cats[cat][r["label"]].append(r["mean_entropy"])

    report_lines.append("\n## Category Breakdown\n")
    report_lines.append("| Category | N(correct) | N(incorrect) | Correct H | Incorrect H | Diff |")
    report_lines.append("|----------|-----------|-------------|-----------|------------|------|")

    for cat in sorted(cats.keys()):
        c = cats[cat]["correct"]
        i = cats[cat]["incorrect"]
        if len(c) < 2 or len(i) < 2:
            continue
        c_m = sum(c) / len(c)
        i_m = sum(i) / len(i)
        diff = i_m - c_m
        print(f"  {cat:30s} C={len(c):3d} I={len(i):3d}  H_c={c_m:.4f}  H_i={i_m:.4f}  diff={diff:+.4f}")
        report_lines.append(f"| {cat} | {len(c)} | {len(i)} | {c_m:.4f} | {i_m:.4f} | {diff:+.4f} |")

    # Write report
    report_lines.append(f"\n## Interpretation\n")
    report_lines.append(f"Best AUC-ROC: {best_auc:.4f} ({best_metric})")
    report_lines.append(f"Direction: Incorrect answers have {direction} entropy")
    if best_auc >= 0.65:
        report_lines.append("**Result: SUPPORTS the fluency boundary hypothesis** (AUC > 0.65)")
    elif best_auc >= 0.55:
        report_lines.append("Result: Weak signal detected. Further investigation needed.")
    else:
        report_lines.append("Result: No clear signal. Hypothesis needs revision or different approach.")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\nReport saved: {REPORT_PATH}")

    # Visualization
    if do_plot:
        try:
            plot_results(correct_data, incorrect_data, binary)
        except Exception as e:
            print(f"Plotting failed: {e}")


def plot_results(correct_data, incorrect_data, binary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("UCBD V1-1A: Fluency Boundary — Entropy vs Correctness", fontsize=14, fontweight="bold")

    # 1. Histogram: mean entropy distribution
    ax = axes[0, 0]
    c_vals = [r["mean_entropy"] for r in correct_data]
    i_vals = [r["mean_entropy"] for r in incorrect_data]
    bins = 30
    ax.hist(c_vals, bins=bins, alpha=0.6, label=f"Correct (n={len(c_vals)})", color="#2ecc71", density=True)
    ax.hist(i_vals, bins=bins, alpha=0.6, label=f"Incorrect (n={len(i_vals)})", color="#e74c3c", density=True)
    ax.set_xlabel("Mean Entropy")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Mean Entropy")
    ax.legend()

    # 2. Box plot: entropy by label
    ax = axes[0, 1]
    data_box = [c_vals, i_vals]
    bp = ax.boxplot(data_box, labels=["Correct", "Incorrect"], patch_artist=True)
    bp["boxes"][0].set_facecolor("#2ecc71")
    bp["boxes"][1].set_facecolor("#e74c3c")
    ax.set_ylabel("Mean Entropy")
    ax.set_title("Entropy by Label")

    # 3. Scatter: mean entropy vs max entropy, colored by label
    ax = axes[1, 0]
    for label, data, color, marker in [
        ("Correct", correct_data, "#2ecc71", "o"),
        ("Incorrect", incorrect_data, "#e74c3c", "x"),
    ]:
        ax.scatter(
            [r["mean_entropy"] for r in data],
            [r["max_entropy"] for r in data],
            c=color, marker=marker, alpha=0.5, s=20, label=label,
        )
    ax.set_xlabel("Mean Entropy")
    ax.set_ylabel("Max Entropy")
    ax.set_title("Mean vs Max Entropy")
    ax.legend()

    # 4. ROC-like curve for mean_entropy
    ax = axes[1, 1]
    scores = [r["mean_entropy"] for r in binary]
    labels = [1 if r["label"] == "incorrect" else 0 for r in binary]
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    tp = fp = 0
    fprs, tprs = [0], [0]
    for score, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        fprs.append(fp / n_neg)
        tprs.append(tp / n_pos)
    ax.plot(fprs, tprs, color="#3498db", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    auc = compute_auc_roc(scores, labels)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve (AUC={auc:.4f})")
    ax.set_aspect("equal")

    plt.tight_layout()
    plot_path = Path(__file__).parent / "results" / "v1a_fluency_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plots saved: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    analyze(do_plot=not args.no_plot)
