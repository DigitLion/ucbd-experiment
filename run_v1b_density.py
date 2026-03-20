#!/usr/bin/env python3
"""
UCBD V1-1B: Density Boundary Experiment.

Tests whether embedding-space density correlates with LLM correctness.
Low density = question is in a sparse region of the model's knowledge = more likely wrong.

Method:
1. Embed all TruthfulQA questions via Ollama qwen3-embedding (4096d)
2. For each question, compute k-NN density (mean cosine similarity to k nearest neighbors)
3. Analyze density vs correctness (AUC-ROC)
4. Test independence from B1 (entropy) signal
5. Test combined B1+B2 discriminator

Usage:
  python3 run_v1b_density.py                # full run
  python3 run_v1b_density.py --limit 20     # test with 20 questions
  python3 run_v1b_density.py --skip-embed   # reuse cached embeddings
"""

import json
import math
import time
import argparse
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
V1A_RESULTS = RESULTS_DIR / "v1a_fluency.jsonl"
EMBEDDINGS_CACHE = RESULTS_DIR / "v1b_embeddings.json"
V1B_RESULTS = RESULTS_DIR / "v1b_density.jsonl"

OLLAMA_URL = "http://127.0.0.1:11434/api/embed"
EMBED_MODEL = "qwen3-embedding"
K_NEIGHBORS = 10  # k for k-NN density


def load_v1a():
    """Load V1-1A results (has questions, answers, labels, entropy metrics)."""
    results = []
    with open(V1A_RESULTS, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def embed_text(text: str) -> list[float]:
    """Get embedding from Ollama qwen3-embedding."""
    payload = json.dumps({"model": EMBED_MODEL, "input": text}).encode()
    req = urllib.request.Request(OLLAMA_URL, data=payload,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data["embeddings"][0]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_knn_density(embeddings: list[list[float]], k: int = K_NEIGHBORS):
    """
    For each embedding, compute mean cosine similarity to k nearest neighbors.
    Higher value = denser region = more knowledge coverage.
    """
    n = len(embeddings)
    densities = []

    for i in range(n):
        sims = []
        for j in range(n):
            if i == j:
                continue
            sims.append(cosine_similarity(embeddings[i], embeddings[j]))
        sims.sort(reverse=True)
        top_k = sims[:k]
        density = sum(top_k) / len(top_k) if top_k else 0.0
        densities.append(density)

    return densities


def compute_knn_density_fast(embeddings: list[list[float]], k: int = K_NEIGHBORS):
    """
    k-NN density via numpy matrix ops.
    Computes all-pairs cosine similarity in one shot.
    """
    import numpy as np

    X = np.array(embeddings, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    X_normed = X / norms

    # All-pairs cosine similarity matrix
    sim_matrix = X_normed @ X_normed.T  # (n, n)
    np.fill_diagonal(sim_matrix, -np.inf)  # exclude self

    # Top-k neighbors per row
    if k >= sim_matrix.shape[1]:
        k = sim_matrix.shape[1] - 1
    top_k_indices = np.argpartition(sim_matrix, -k, axis=1)[:, -k:]
    top_k_sims = np.take_along_axis(sim_matrix, top_k_indices, axis=1)
    densities = top_k_sims.mean(axis=1)

    return densities.tolist()


def run_embeddings(v1a_data, limit=None):
    """Embed all questions and cache results."""
    items = v1a_data[:limit] if limit else v1a_data
    n = len(items)
    print(f"Embedding {n} questions via {EMBED_MODEL}...")

    embeddings = []
    t0 = time.time()
    batch_size = 10

    for i in range(0, n, batch_size):
        batch = items[i:i + batch_size]
        texts = [r["question"] for r in batch]

        # Ollama supports batch embedding
        payload = json.dumps({"model": EMBED_MODEL, "input": texts}).encode()
        req = urllib.request.Request(OLLAMA_URL, data=payload,
                                    headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())

        for emb in data["embeddings"]:
            embeddings.append(emb)

        done = min(i + batch_size, n)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (n - done) / rate if rate > 0 else 0
        print(f"  [{done}/{n}] {elapsed:.1f}s elapsed, {eta:.0f}s ETA")

    # Cache
    cache = {
        "model": EMBED_MODEL,
        "dims": len(embeddings[0]) if embeddings else 0,
        "n": len(embeddings),
        "question_indices": [r["question_idx"] for r in items],
        "embeddings": embeddings,
    }
    with open(EMBEDDINGS_CACHE, "w") as f:
        json.dump(cache, f)
    print(f"Cached {len(embeddings)} embeddings to {EMBEDDINGS_CACHE}")

    return embeddings


def load_cached_embeddings():
    """Load cached embeddings."""
    with open(EMBEDDINGS_CACHE) as f:
        cache = json.load(f)
    print(f"Loaded {cache['n']} cached embeddings ({cache['dims']}d, model={cache['model']})")
    return cache["embeddings"], cache["question_indices"]


def compute_auc_roc(scores, labels):
    """AUC-ROC via trapezoidal rule."""
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


def pearson_r(x, y):
    """Pearson correlation coefficient."""
    n = len(x)
    if n < 3:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / n
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x) / n)
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y) / n)
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)


def mann_whitney_u(group_a, group_b):
    """Mann-Whitney U test (two-sided)."""
    if len(group_a) < 2 or len(group_b) < 2:
        return 0, 1.0
    combined = [(v, "a") for v in group_a] + [(v, "b") for v in group_b]
    combined.sort(key=lambda x: x[0])

    ranks = {}
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2
        for k in range(i, j):
            ranks[id(combined[k])] = avg_rank
        i = j

    rank_sum_a = sum(ranks[id(x)] for x in combined if x[1] == "a")
    n_a, n_b = len(group_a), len(group_b)
    u_a = rank_sum_a - n_a * (n_a + 1) / 2
    u_b = n_a * n_b - u_a
    mu = n_a * n_b / 2
    sigma = math.sqrt(n_a * n_b * (n_a + n_b + 1) / 12)
    if sigma == 0:
        return min(u_a, u_b), 1.0
    z = (min(u_a, u_b) - mu) / sigma
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return min(u_a, u_b), p


def analyze(v1a_data, embeddings, question_indices, k=K_NEIGHBORS):
    """Full analysis: density vs correctness, independence from B1, combined model."""

    # Build index mapping
    idx_to_pos = {qi: pos for pos, qi in enumerate(question_indices)}

    # Filter to binary-labeled items that have embeddings
    binary = []
    for r in v1a_data:
        if r["label"] in ("correct", "incorrect") and r["question_idx"] in idx_to_pos:
            binary.append(r)

    n_c = sum(1 for r in binary if r["label"] == "correct")
    n_i = sum(1 for r in binary if r["label"] == "incorrect")
    print(f"\nBinary samples with embeddings: {len(binary)} (correct={n_c}, incorrect={n_i})")

    # Compute k-NN density for all embedded questions
    print(f"\nComputing {k}-NN density for {len(embeddings)} embeddings...")
    t0 = time.time()
    densities = compute_knn_density_fast(embeddings, k)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Also compute density using only binary subset embeddings
    binary_positions = [idx_to_pos[r["question_idx"]] for r in binary]
    binary_embeddings = [embeddings[p] for p in binary_positions]
    print(f"\nComputing {k}-NN density for {len(binary_embeddings)} binary-subset embeddings...")
    t0 = time.time()
    binary_densities = compute_knn_density_fast(binary_embeddings, k)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Attach density to each binary record
    for i, r in enumerate(binary):
        pos = idx_to_pos[r["question_idx"]]
        r["density_all"] = densities[pos]       # density in full embedding space
        r["density_binary"] = binary_densities[i]  # density in binary subset

    # --- Core analysis ---
    print("\n" + "=" * 80)
    print("V1-1B DENSITY BOUNDARY RESULTS")
    print("=" * 80)

    for density_key, density_name in [("density_all", "Density (all 790)"), ("density_binary", "Density (binary 401)")]:
        c_dens = [r[density_key] for r in binary if r["label"] == "correct"]
        i_dens = [r[density_key] for r in binary if r["label"] == "incorrect"]

        c_mean = sum(c_dens) / len(c_dens)
        i_mean = sum(i_dens) / len(i_dens)

        # Low density = more likely incorrect → invert for AUC (higher score = more likely incorrect)
        inv_scores = [-r[density_key] for r in binary]
        labels = [1 if r["label"] == "incorrect" else 0 for r in binary]
        auc_inv = compute_auc_roc(inv_scores, labels)

        # Also try raw (high density = incorrect? unlikely but test both)
        raw_scores = [r[density_key] for r in binary]
        auc_raw = compute_auc_roc(raw_scores, labels)

        effective_auc = max(auc_inv, auc_raw)
        direction = "low=wrong" if auc_inv >= auc_raw else "high=wrong"

        _, p_val = mann_whitney_u(c_dens, i_dens)

        print(f"\n  {density_name}:")
        print(f"    Correct mean:   {c_mean:.6f}")
        print(f"    Incorrect mean: {i_mean:.6f}")
        print(f"    Diff:           {i_mean - c_mean:+.6f}")
        print(f"    AUC-ROC:        {effective_auc:.4f} ({direction})")
        print(f"    p-value:        {p_val:.4f}")

    # --- Independence from B1 ---
    print("\n--- B1 (Entropy) vs B2 (Density) Independence ---")
    entropies = [r["mean_entropy"] for r in binary]
    dens_vals = [r["density_binary"] for r in binary]
    r_corr = pearson_r(entropies, dens_vals)
    print(f"  Pearson r(entropy, density) = {r_corr:.4f}")
    print(f"  {'Independent (|r|<0.3)' if abs(r_corr) < 0.3 else 'Correlated (|r|>=0.3)'}")

    # --- Combined B1+B2 discriminator ---
    print("\n--- Combined B1+B2 Discriminator ---")

    # Normalize both signals to [0,1] range
    def normalize(vals):
        lo, hi = min(vals), max(vals)
        rng = hi - lo
        if rng == 0:
            return [0.5] * len(vals)
        return [(v - lo) / rng for v in vals]

    norm_entropy = normalize([r["mean_entropy"] for r in binary])
    norm_density = normalize([-r["density_binary"] for r in binary])  # invert: low density = high score

    labels = [1 if r["label"] == "incorrect" else 0 for r in binary]

    # Individual AUCs
    auc_b1 = compute_auc_roc(norm_entropy, labels)
    auc_b2 = compute_auc_roc(norm_density, labels)

    # Combined: simple average
    combined_avg = [(e + d) / 2 for e, d in zip(norm_entropy, norm_density)]
    auc_combined_avg = compute_auc_roc(combined_avg, labels)

    # Combined: max (either detector fires)
    combined_max = [max(e, d) for e, d in zip(norm_entropy, norm_density)]
    auc_combined_max = compute_auc_roc(combined_max, labels)

    # Combined: weighted (optimize weight by grid search)
    best_w, best_auc_w = 0.5, auc_combined_avg
    for w in [i * 0.05 for i in range(21)]:
        combined_w = [w * e + (1 - w) * d for e, d in zip(norm_entropy, norm_density)]
        auc_w = compute_auc_roc(combined_w, labels)
        if auc_w > best_auc_w:
            best_w = w
            best_auc_w = auc_w

    print(f"  B1 only (entropy):      AUC = {auc_b1:.4f}")
    print(f"  B2 only (density):      AUC = {auc_b2:.4f}")
    print(f"  B1+B2 average:          AUC = {auc_combined_avg:.4f}")
    print(f"  B1+B2 max:              AUC = {auc_combined_max:.4f}")
    print(f"  B1+B2 best weight:      AUC = {best_auc_w:.4f} (w_B1={best_w:.2f})")
    print(f"  Improvement over best single: {best_auc_w - max(auc_b1, auc_b2):+.4f}")

    # --- Category-level B2 analysis ---
    print("\n--- Category-Level B2 Density ---")
    from collections import defaultdict
    cats = defaultdict(list)
    for r in binary:
        cats[r["category"]].append(r)

    cat_results = []
    for cat, recs in sorted(cats.items()):
        n_c_cat = sum(1 for r in recs if r["label"] == "correct")
        n_i_cat = sum(1 for r in recs if r["label"] == "incorrect")
        if n_c_cat < 2 or n_i_cat < 2:
            continue

        c_d = [r["density_binary"] for r in recs if r["label"] == "correct"]
        i_d = [r["density_binary"] for r in recs if r["label"] == "incorrect"]
        diff = sum(i_d) / len(i_d) - sum(c_d) / len(c_d)

        inv_scores = [-r["density_binary"] for r in recs]
        lab = [1 if r["label"] == "incorrect" else 0 for r in recs]
        auc = compute_auc_roc(inv_scores, lab)

        cat_results.append({
            "category": cat, "n": len(recs),
            "density_diff": diff, "auc_density": auc,
        })
        print(f"  {cat:30s} N={len(recs):3d}  diff={diff:+.6f}  AUC={auc:.3f}")

    # --- Save detailed results ---
    print("\nSaving results...")
    with open(V1B_RESULTS, "w", encoding="utf-8") as f:
        for r in binary:
            out = {
                "question_idx": r["question_idx"],
                "category": r["category"],
                "question": r["question"],
                "label": r["label"],
                "mean_entropy": r["mean_entropy"],
                "high_entropy_ratio": r.get("high_entropy_ratio", 0),
                "density_all": r["density_all"],
                "density_binary": r["density_binary"],
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"Results: {V1B_RESULTS}")

    # Return for plotting
    return {
        "binary": binary,
        "auc_b1": auc_b1, "auc_b2": auc_b2,
        "auc_combined_avg": auc_combined_avg,
        "auc_combined_max": auc_combined_max,
        "best_w": best_w, "best_auc_w": best_auc_w,
        "r_correlation": r_corr,
        "norm_entropy": norm_entropy,
        "norm_density": norm_density,
        "labels": labels,
        "cat_results": cat_results,
    }


def plot_v1b(stats):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return

    binary = stats["binary"]
    labels = stats["labels"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("UCBD V1-1B: Density Boundary Analysis", fontsize=14, fontweight="bold")

    # 1. Density distribution by label
    ax = axes[0, 0]
    c_d = [r["density_binary"] for r in binary if r["label"] == "correct"]
    i_d = [r["density_binary"] for r in binary if r["label"] == "incorrect"]
    ax.hist(c_d, bins=30, alpha=0.6, label=f"Correct (n={len(c_d)})", color="#2ecc71", density=True)
    ax.hist(i_d, bins=30, alpha=0.6, label=f"Incorrect (n={len(i_d)})", color="#e74c3c", density=True)
    ax.set_xlabel("k-NN Density (cosine sim)")
    ax.set_ylabel("Density")
    ax.set_title("Embedding Density Distribution")
    ax.legend()

    # 2. Entropy vs Density scatter
    ax = axes[0, 1]
    for label_name, color, marker in [("correct", "#2ecc71", "o"), ("incorrect", "#e74c3c", "x")]:
        pts = [r for r in binary if r["label"] == label_name]
        ax.scatter([r["mean_entropy"] for r in pts], [r["density_binary"] for r in pts],
                   c=color, marker=marker, alpha=0.4, s=15, label=label_name)
    ax.set_xlabel("B1: Mean Entropy")
    ax.set_ylabel("B2: k-NN Density")
    ax.set_title(f"B1 vs B2 (r={stats['r_correlation']:.3f})")
    ax.legend()

    # 3. ROC curves: B1, B2, Combined
    ax = axes[0, 2]
    for name, scores_list, color, ls in [
        (f"B1 Entropy (AUC={stats['auc_b1']:.3f})", stats["norm_entropy"], "#3498db", "--"),
        (f"B2 Density (AUC={stats['auc_b2']:.3f})", stats["norm_density"], "#e67e22", "--"),
        (f"B1+B2 Best (AUC={stats['best_auc_w']:.3f})", None, "#8e44ad", "-"),
    ]:
        if scores_list is None:
            w = stats["best_w"]
            scores_list = [w * e + (1 - w) * d for e, d in zip(stats["norm_entropy"], stats["norm_density"])]
        pairs = sorted(zip(scores_list, labels), key=lambda x: -x[0])
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
        ax.plot(fprs, tprs, color=color, linewidth=2, linestyle=ls, label=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC: B1 vs B2 vs Combined")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")

    # 4. Box plot: density by label
    ax = axes[1, 0]
    bp = ax.boxplot([c_d, i_d], labels=["Correct", "Incorrect"], patch_artist=True)
    bp["boxes"][0].set_facecolor("#2ecc71")
    bp["boxes"][1].set_facecolor("#e74c3c")
    ax.set_ylabel("k-NN Density")
    ax.set_title("Density by Label")

    # 5. Combined score distribution
    ax = axes[1, 1]
    w = stats["best_w"]
    c_combined = [w * stats["norm_entropy"][i] + (1 - w) * stats["norm_density"][i]
                  for i in range(len(binary)) if binary[i]["label"] == "correct"]
    i_combined = [w * stats["norm_entropy"][i] + (1 - w) * stats["norm_density"][i]
                  for i in range(len(binary)) if binary[i]["label"] == "incorrect"]
    ax.hist(c_combined, bins=25, alpha=0.6, label="Correct", color="#2ecc71", density=True)
    ax.hist(i_combined, bins=25, alpha=0.6, label="Incorrect", color="#e74c3c", density=True)
    ax.set_xlabel(f"Combined Score (w_B1={w:.2f})")
    ax.set_ylabel("Density")
    ax.set_title("Combined B1+B2 Score Distribution")
    ax.legend()

    # 6. Category-level B2 AUC
    ax = axes[1, 2]
    cat_results = sorted(stats["cat_results"], key=lambda x: x["auc_density"])
    names = [c["category"][:20] for c in cat_results]
    aucs = [c["auc_density"] for c in cat_results]
    colors = ["#2ecc71" if a >= 0.5 else "#e74c3c" for a in aucs]
    ax.barh(range(len(names)), aucs, color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("AUC-ROC (low density = incorrect)")
    ax.set_title("Per-Category B2 Density AUC")

    plt.tight_layout()
    plot_path = RESULTS_DIR / "v1b_density_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plots: {plot_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-embed", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("-k", type=int, default=K_NEIGHBORS, help="k for k-NN")
    args = parser.parse_args()

    v1a_data = load_v1a()
    print(f"Loaded {len(v1a_data)} V1-1A results")

    if args.skip_embed and EMBEDDINGS_CACHE.exists():
        embeddings, question_indices = load_cached_embeddings()
    else:
        items = v1a_data[:args.limit] if args.limit else v1a_data
        embeddings = run_embeddings(items, args.limit)
        question_indices = [r["question_idx"] for r in items]

    stats = analyze(v1a_data, embeddings, question_indices, k=args.k)

    if not args.no_plot:
        plot_v1b(stats)

    print("\nDone.")


if __name__ == "__main__":
    main()
