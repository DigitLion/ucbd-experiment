#!/usr/bin/env python3
"""
UCBD V4: Cross-Model Domain Specificity Analysis
==================================================
Compares B1 fluency patterns across models to test if
domain specificity is a structural (model-invariant) property.

Key metrics:
  1. Per-category entropy_diff direction consistency across models
  2. Spearman rank correlation of delta_H across models
  3. B1-effective/blind domain overlap (Jaccard)
  4. Grouped AUC comparison

Usage:
  python3 analyze_v4_cross_model.py
"""

import json
import math
import random
from pathlib import Path
from collections import defaultdict
from itertools import combinations

RESULTS_DIR = Path(__file__).parent / "results"
REPORT_PATH = RESULTS_DIR / "v4_cross_model_report.md"
PLOT_PATH = RESULTS_DIR / "v4_cross_model_plots.png"

MODEL_FILES = {
    "qwen3-14b": "v4_qwen3-14b_fluency.jsonl",
    "qwen3-4b": "v4_qwen3-4b_fluency.jsonl",
    "llama3.2-3b": "v4_llama3.2-3b_fluency.jsonl",
}

MODEL_DISPLAY = {
    "qwen3-14b": "Qwen3-14B",
    "qwen3-4b": "Qwen3-4B",
    "llama3.2-3b": "LLaMA-3.2-3B",
}


def load_model_data(model_key: str) -> list[dict]:
    path = RESULTS_DIR / MODEL_FILES[model_key]
    if not path.exists():
        return []
    results = []
    with open(path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def compute_auc_roc(scores, labels):
    if not scores or not labels:
        return float("nan")
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
            roc_points.append((fp / n_neg, tp / n_pos))
        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score
    roc_points.append((fp / n_neg, tp / n_pos))
    auc = 0.0
    for i in range(1, len(roc_points)):
        dx = roc_points[i][0] - roc_points[i - 1][0]
        dy = (roc_points[i][1] + roc_points[i - 1][1]) / 2
        auc += dx * dy
    return auc


def bootstrap_auc(scores, labels, n_boot=2000, seed=42):
    rng = random.Random(seed)
    n = len(scores)
    aucs = []
    for _ in range(n_boot):
        idx = [rng.randint(0, n - 1) for _ in range(n)]
        s = [scores[i] for i in idx]
        l = [labels[i] for i in idx]
        if sum(l) == 0 or sum(l) == len(l):
            continue
        aucs.append(compute_auc_roc(s, l))
    aucs.sort()
    if len(aucs) < 100:
        return float("nan"), float("nan")
    return aucs[int(len(aucs) * 0.025)], aucs[int(len(aucs) * 0.975)]


def spearman_rank(x, y):
    """Spearman rank correlation coefficient."""
    if len(x) != len(y) or len(x) < 3:
        return float("nan")
    n = len(x)
    rx = rank_data(x)
    ry = rank_data(y)
    d_sq = sum((a - b) ** 2 for a, b in zip(rx, ry))
    rho = 1 - (6 * d_sq) / (n * (n ** 2 - 1))
    return rho


def rank_data(vals):
    indexed = sorted(enumerate(vals), key=lambda x: x[1])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 1) / 2
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def get_category_stats(data: list[dict]) -> dict[str, dict]:
    """Compute per-category entropy stats from model data."""
    binary = [r for r in data if r["label"] in ("correct", "incorrect")]
    cats = defaultdict(list)
    for r in binary:
        cats[r["category"]].append(r)

    stats = {}
    for cat, recs in cats.items():
        n_c = sum(1 for r in recs if r["label"] == "correct")
        n_i = sum(1 for r in recs if r["label"] == "incorrect")
        if n_c < 2 or n_i < 2:
            continue
        c_vals = [r["mean_entropy"] for r in recs if r["label"] == "correct"]
        i_vals = [r["mean_entropy"] for r in recs if r["label"] == "incorrect"]
        diff = sum(i_vals) / len(i_vals) - sum(c_vals) / len(c_vals)
        scores = [r["mean_entropy"] for r in recs]
        labels = [1 if r["label"] == "incorrect" else 0 for r in recs]
        auc = compute_auc_roc(scores, labels)
        stats[cat] = {
            "n_correct": n_c,
            "n_incorrect": n_i,
            "entropy_diff": diff,
            "auc": auc,
            "zone": "effective" if diff > 0 else "blind",
        }
    return stats


def analyze():
    # Load available models
    all_data = {}
    for model_key in MODEL_FILES:
        data = load_model_data(model_key)
        if data:
            all_data[model_key] = data
            n_binary = sum(1 for r in data if r["label"] in ("correct", "incorrect"))
            print(f"  {MODEL_DISPLAY[model_key]}: {len(data)} total, {n_binary} binary")

    if len(all_data) < 2:
        print(f"\nERROR: Need at least 2 models with data. Found: {list(all_data.keys())}")
        print("Run the missing models first:")
        for k in MODEL_FILES:
            if k not in all_data:
                print(f"  python3 run_v4_cross_model.py --model {k}")
        return

    print(f"\nModels loaded: {len(all_data)}")
    print()

    # Per-model category stats
    model_stats = {}
    for model_key, data in all_data.items():
        model_stats[model_key] = get_category_stats(data)

    # Find common categories across all models
    common_cats = None
    for stats in model_stats.values():
        if common_cats is None:
            common_cats = set(stats.keys())
        else:
            common_cats &= set(stats.keys())
    common_cats = sorted(common_cats)
    print(f"Common categories (>= 2 correct + 2 incorrect in all models): {len(common_cats)}")

    # === Analysis 1: Domain specificity consistency ===
    print("\n" + "=" * 100)
    print("1. PER-CATEGORY ENTROPY DIFF DIRECTION ACROSS MODELS")
    print("=" * 100)
    header = f"{'Category':<32s}"
    for mk in all_data:
        header += f" {MODEL_DISPLAY[mk]:>14s}"
    header += "  Consistent?"
    print(header)
    print("-" * 100)

    consistent_count = 0
    for cat in common_cats:
        row = f"  {cat:<30s}"
        signs = []
        for mk in all_data:
            diff = model_stats[mk][cat]["entropy_diff"]
            sign = "+" if diff > 0 else "-"
            signs.append(sign)
            row += f"  {diff:>+10.4f}{sign}"
        is_consistent = len(set(signs)) == 1
        if is_consistent:
            consistent_count += 1
        row += f"  {'YES' if is_consistent else 'NO':>8s}"
        print(row)

    consistency_rate = consistent_count / len(common_cats) if common_cats else 0
    print(f"\nDirection consistency: {consistent_count}/{len(common_cats)} = {consistency_rate:.1%}")

    # === Analysis 2: Spearman rank correlation ===
    print("\n" + "=" * 100)
    print("2. SPEARMAN RANK CORRELATION OF ENTROPY_DIFF ACROSS MODELS")
    print("=" * 100)

    model_keys = list(all_data.keys())
    for m1, m2 in combinations(model_keys, 2):
        diffs_1 = [model_stats[m1][cat]["entropy_diff"] for cat in common_cats]
        diffs_2 = [model_stats[m2][cat]["entropy_diff"] for cat in common_cats]
        rho = spearman_rank(diffs_1, diffs_2)
        print(f"  {MODEL_DISPLAY[m1]} vs {MODEL_DISPLAY[m2]}: rho = {rho:.4f}", end="")
        if abs(rho) > 0.6:
            print("  ** STRONG — domain specificity is model-invariant")
        elif abs(rho) > 0.3:
            print("  * MODERATE — partial structural consistency")
        else:
            print("    WEAK — domain patterns are model-specific")

    # === Analysis 3: B1 zone overlap ===
    print("\n" + "=" * 100)
    print("3. B1-EFFECTIVE / B1-BLIND ZONE OVERLAP (JACCARD)")
    print("=" * 100)

    model_zones = {}
    for mk in all_data:
        effective = {cat for cat in common_cats if model_stats[mk][cat]["zone"] == "effective"}
        blind = {cat for cat in common_cats if model_stats[mk][cat]["zone"] == "blind"}
        model_zones[mk] = {"effective": effective, "blind": blind}
        print(f"  {MODEL_DISPLAY[mk]}: {len(effective)} effective, {len(blind)} blind")

    print()
    for m1, m2 in combinations(model_keys, 2):
        eff_1, eff_2 = model_zones[m1]["effective"], model_zones[m2]["effective"]
        jaccard_eff = len(eff_1 & eff_2) / len(eff_1 | eff_2) if eff_1 | eff_2 else 0
        bld_1, bld_2 = model_zones[m1]["blind"], model_zones[m2]["blind"]
        jaccard_bld = len(bld_1 & bld_2) / len(bld_1 | bld_2) if bld_1 | bld_2 else 0
        print(f"  {MODEL_DISPLAY[m1]} vs {MODEL_DISPLAY[m2]}:")
        print(f"    Effective zone Jaccard: {jaccard_eff:.3f} (overlap: {len(eff_1 & eff_2)}/{len(eff_1 | eff_2)})")
        print(f"    Blind zone Jaccard:     {jaccard_bld:.3f} (overlap: {len(bld_1 & bld_2)}/{len(bld_1 | bld_2)})")

    # === Analysis 4: Grouped AUC comparison ===
    print("\n" + "=" * 100)
    print("4. GROUPED AUC BY MODEL (EFFECTIVE / BLIND / OVERALL)")
    print("=" * 100)

    print(f"  {'Model':<20s} {'Eff AUC':>10s} {'95% CI':>20s} {'Blind AUC':>10s} {'95% CI':>20s} {'Overall':>10s}")
    print("-" * 100)

    for mk, data in all_data.items():
        binary = [r for r in data if r["label"] in ("correct", "incorrect") and r["category"] in common_cats]
        stats = model_stats[mk]
        eff_cats = {cat for cat in common_cats if stats.get(cat, {}).get("zone") == "effective"}
        bld_cats = {cat for cat in common_cats if stats.get(cat, {}).get("zone") == "blind"}

        def pool_auc(cat_set):
            recs = [r for r in binary if r["category"] in cat_set]
            if not recs:
                return float("nan"), (float("nan"), float("nan"))
            scores = [r["mean_entropy"] for r in recs]
            labels = [1 if r["label"] == "incorrect" else 0 for r in recs]
            auc = compute_auc_roc(scores, labels)
            ci = bootstrap_auc(scores, labels)
            return auc, ci

        eff_auc, eff_ci = pool_auc(eff_cats)
        bld_auc, bld_ci = pool_auc(bld_cats)
        all_auc, all_ci = pool_auc(set(common_cats))

        print(f"  {MODEL_DISPLAY[mk]:<20s} {eff_auc:>10.4f} [{eff_ci[0]:.3f}, {eff_ci[1]:.3f}] {bld_auc:>10.4f} [{bld_ci[0]:.3f}, {bld_ci[1]:.3f}] {all_auc:>10.4f}")

    # === Analysis 5: Categories that flip zone across models ===
    print("\n" + "=" * 100)
    print("5. ZONE-FLIPPING CATEGORIES (effective in one model, blind in another)")
    print("=" * 100)
    flip_cats = []
    for cat in common_cats:
        zones = set()
        for mk in all_data:
            zones.add(model_stats[mk][cat]["zone"])
        if len(zones) > 1:
            flip_cats.append(cat)
            details = ", ".join(f"{MODEL_DISPLAY[mk]}={model_stats[mk][cat]['zone']}({model_stats[mk][cat]['entropy_diff']:+.4f})" for mk in all_data)
            print(f"  {cat}: {details}")
    print(f"\nFlipping categories: {len(flip_cats)}/{len(common_cats)} = {len(flip_cats)/len(common_cats):.1%}")
    print(f"Stable categories:  {len(common_cats)-len(flip_cats)}/{len(common_cats)} = {(len(common_cats)-len(flip_cats))/len(common_cats):.1%}")

    # === Write report ===
    write_report(all_data, model_stats, common_cats, model_zones, consistency_rate,
                 flip_cats, model_keys)

    # === Plot ===
    plot_cross_model(all_data, model_stats, common_cats, model_keys)


def write_report(all_data, model_stats, common_cats, model_zones, consistency_rate,
                 flip_cats, model_keys):
    lines = []
    lines.append("# UCBD V4: Cross-Model Domain Specificity Analysis\n")
    lines.append(f"Models: {', '.join(MODEL_DISPLAY[mk] for mk in all_data)}")
    lines.append(f"Common categories: {len(common_cats)}")
    lines.append(f"Direction consistency: {consistency_rate:.1%}\n")

    lines.append("## Key Findings\n")
    if consistency_rate > 0.7:
        lines.append("1. **Domain specificity is largely model-invariant.** "
                      f"{consistency_rate:.0%} of categories maintain the same B1 effective/blind "
                      "classification across models.")
    elif consistency_rate > 0.5:
        lines.append("1. **Partial structural consistency.** "
                      f"{consistency_rate:.0%} of categories are consistent, suggesting a mix of "
                      "model-invariant and model-specific domain patterns.")
    else:
        lines.append("1. **Domain specificity appears model-specific.** "
                      f"Only {consistency_rate:.0%} consistency — pointer models likely need "
                      "model-specific training.")

    # Spearman correlations
    lines.append("\n## Spearman Rank Correlations\n")
    lines.append("| Model Pair | rho | Interpretation |")
    lines.append("|-----------|-----|----------------|")
    from itertools import combinations as combs
    for m1, m2 in combs(model_keys, 2):
        diffs_1 = [model_stats[m1][cat]["entropy_diff"] for cat in common_cats]
        diffs_2 = [model_stats[m2][cat]["entropy_diff"] for cat in common_cats]
        rho = spearman_rank(diffs_1, diffs_2)
        interp = "Strong" if abs(rho) > 0.6 else ("Moderate" if abs(rho) > 0.3 else "Weak")
        lines.append(f"| {MODEL_DISPLAY[m1]} vs {MODEL_DISPLAY[m2]} | {rho:.4f} | {interp} |")

    lines.append(f"\n## Zone-Flipping Categories\n")
    lines.append(f"{len(flip_cats)}/{len(common_cats)} categories flip between effective/blind across models:\n")
    for cat in flip_cats:
        details = ", ".join(f"{MODEL_DISPLAY[mk]}={model_stats[mk][cat]['zone']}" for mk in all_data)
        lines.append(f"- **{cat}**: {details}")

    lines.append("\n## Implication for UCBD\n")
    lines.append("If domain specificity is largely model-invariant, the Pointer Model can be trained once "
                 "and applied across model families. If model-specific, each deployment needs its own "
                 "pointer model calibration — but the *cascade structure* itself remains universally valid.")
    lines.append("")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport: {REPORT_PATH}")


def plot_cross_model(all_data, model_stats, common_cats, model_keys):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    n_models = len(model_keys)
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle("UCBD V4: Cross-Model Domain Specificity", fontsize=15, fontweight="bold")

    colors = {"qwen3-14b": "#2563eb", "qwen3-4b": "#7c3aed", "llama3.2-3b": "#ea580c"}

    # 1. Entropy diff comparison (bar chart, grouped by category)
    ax = axes[0, 0]
    cats_sorted = sorted(common_cats, key=lambda c: model_stats[model_keys[0]][c]["entropy_diff"])
    x = range(len(cats_sorted))
    bar_w = 0.8 / n_models
    for i, mk in enumerate(model_keys):
        diffs = [model_stats[mk][cat]["entropy_diff"] for cat in cats_sorted]
        ax.barh([xi + i * bar_w for xi in x], diffs, height=bar_w,
                color=colors.get(mk, "#888"), alpha=0.8, label=MODEL_DISPLAY[mk])
    ax.set_yticks([xi + bar_w * (n_models - 1) / 2 for xi in x])
    short_names = [c.replace("Indexical Error: ", "Idx:").replace("Confusion: ", "C:") for c in cats_sorted]
    ax.set_yticklabels(short_names, fontsize=6)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Entropy Diff (incorrect - correct)")
    ax.set_title("Per-Category Entropy Diff by Model")
    ax.legend(fontsize=8, loc="lower right")

    # 2. Scatter: Model A diff vs Model B diff (first two models)
    ax = axes[0, 1]
    if len(model_keys) >= 2:
        m1, m2 = model_keys[0], model_keys[1]
        diffs_1 = [model_stats[m1][cat]["entropy_diff"] for cat in common_cats]
        diffs_2 = [model_stats[m2][cat]["entropy_diff"] for cat in common_cats]
        rho = spearman_rank(diffs_1, diffs_2)

        for i, cat in enumerate(common_cats):
            d1, d2 = diffs_1[i], diffs_2[i]
            same_sign = (d1 > 0 and d2 > 0) or (d1 <= 0 and d2 <= 0)
            c = "#16a34a" if same_sign else "#dc2626"
            ax.scatter(d1, d2, c=c, s=40, alpha=0.7, edgecolors="white", linewidth=0.5)
            if abs(d1) > 0.06 or abs(d2) > 0.06:
                ax.annotate(cat[:12], (d1, d2), fontsize=5, alpha=0.7)

        # Quadrant lines
        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
        ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)
        ax.set_xlabel(f"{MODEL_DISPLAY[m1]} entropy diff")
        ax.set_ylabel(f"{MODEL_DISPLAY[m2]} entropy diff")
        ax.set_title(f"Domain Specificity Correlation\nSpearman rho = {rho:.4f}")

        # Add quadrant labels
        ax.text(0.95, 0.95, "Both\nEffective", transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color="#16a34a", alpha=0.5)
        ax.text(0.05, 0.05, "Both\nBlind", transform=ax.transAxes, ha="left", va="bottom",
                fontsize=8, color="#16a34a", alpha=0.5)
        ax.text(0.05, 0.95, "Flipped", transform=ax.transAxes, ha="left", va="top",
                fontsize=8, color="#dc2626", alpha=0.5)
        ax.text(0.95, 0.05, "Flipped", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color="#dc2626", alpha=0.5)

    # 3. AUC comparison across models
    ax = axes[1, 0]
    for mk in model_keys:
        data = all_data[mk]
        binary = [r for r in data if r["label"] in ("correct", "incorrect") and r["category"] in common_cats]
        cat_aucs = []
        cat_names = []
        for cat in common_cats:
            recs = [r for r in binary if r["category"] == cat]
            if len(recs) < 4:
                continue
            scores = [r["mean_entropy"] for r in recs]
            labels = [1 if r["label"] == "incorrect" else 0 for r in recs]
            auc = compute_auc_roc(scores, labels)
            if not math.isnan(auc):
                cat_aucs.append(auc)
                cat_names.append(cat)
        ax.scatter(range(len(cat_aucs)),
                   sorted(cat_aucs),
                   c=colors.get(mk, "#888"), s=30, alpha=0.7,
                   label=f"{MODEL_DISPLAY[mk]} (mean={sum(cat_aucs)/len(cat_aucs):.3f})")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    ax.set_xlabel("Category rank")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Per-Category AUC Distribution by Model")
    ax.legend(fontsize=8)

    # 4. Grouped AUC bar chart
    ax = axes[1, 1]
    groups = ["B1-Effective", "B1-Blind", "Overall"]
    x_pos = range(len(groups))
    bar_w = 0.8 / n_models

    for i, mk in enumerate(model_keys):
        data = all_data[mk]
        binary = [r for r in data if r["label"] in ("correct", "incorrect") and r["category"] in common_cats]
        stats = model_stats[mk]
        eff_cats = {cat for cat in common_cats if stats.get(cat, {}).get("zone") == "effective"}
        bld_cats = {cat for cat in common_cats if stats.get(cat, {}).get("zone") == "blind"}

        aucs = []
        for cat_set in [eff_cats, bld_cats, set(common_cats)]:
            recs = [r for r in binary if r["category"] in cat_set]
            if recs:
                scores = [r["mean_entropy"] for r in recs]
                labels = [1 if r["label"] == "incorrect" else 0 for r in recs]
                aucs.append(compute_auc_roc(scores, labels))
            else:
                aucs.append(0.5)

        ax.bar([xi + i * bar_w for xi in x_pos], aucs, width=bar_w,
               color=colors.get(mk, "#888"), alpha=0.8, label=MODEL_DISPLAY[mk])

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xticks([xi + bar_w * (n_models - 1) / 2 for xi in x_pos])
    ax.set_xticklabels(groups)
    ax.set_ylabel("Pooled AUC-ROC")
    ax.set_title("Grouped AUC: B1-Effective vs B1-Blind")
    ax.legend(fontsize=8)
    ax.set_ylim(0.2, 0.85)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    print(f"Plots: {PLOT_PATH}")


if __name__ == "__main__":
    print("=== UCBD V4: Cross-Model Domain Specificity Analysis ===\n")
    print("Loading model data...")
    analyze()
