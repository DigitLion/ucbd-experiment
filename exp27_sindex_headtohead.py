#!/usr/bin/env python3
"""Exp 27: SINdex head-to-head comparison
Computes SINdex (full metric with intra-cluster coherence weighting),
SE-Embedding, SE-Jaccard, and B1 entropy on the same 200 TruthfulQA questions.
Reports AUROC for each method under LLM-judge labels.
"""
import json, sys, time, requests, math
import numpy as np
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import roc_auc_score

# === Config ===
EMBEDDING_URL = "http://127.0.0.1:11434/api/embed"
EMBEDDING_MODEL = "qwen3-embedding"
JACCARD_THRESH = 0.4
EMB_THRESH = 0.85
SINDEX_DIST_THRESH = 0.05  # SINdex paper: cosine distance <= 0.05 (sim >= 0.95)

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"

# === Load data ===
with open(DATA_DIR / "exp21_200tok_samples.json") as f:
    sample_data = json.load(f)

# Load LLM-judge labels from v1a
labels_map = {}
with open(RESULTS_DIR / "v1a_fluency.jsonl") as f:
    for line in f:
        rec = json.loads(line)
        labels_map[rec["question"].strip()] = {
            "label": rec["label"],
            "mean_entropy": rec["mean_entropy"],
            "category": rec.get("category", ""),
        }

# === Embedding cache ===
_emb_cache = {}

def get_embedding(text):
    if text in _emb_cache:
        return _emb_cache[text]
    resp = requests.post(EMBEDDING_URL, json={
        "model": EMBEDDING_MODEL, "input": text
    }, timeout=60)
    emb = np.array(resp.json()["embeddings"][0])
    _emb_cache[text] = emb
    return emb

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

# === Jaccard bigram clustering ===
def jaccard_bigrams(a, b):
    def bigrams(s):
        toks = s.lower().split()
        return set(zip(toks, toks[1:])) if len(toks) > 1 else {(s.lower(),)}
    sa, sb = bigrams(a), bigrams(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)

def cluster_jaccard(samples, threshold):
    clusters = []
    for s in samples:
        placed = False
        for c in clusters:
            if jaccard_bigrams(s, c[0]) >= threshold:
                c.append(s)
                placed = True
                break
        if not placed:
            clusters.append([s])
    return clusters

# === Embedding agglomerative clustering (our SE-Embedding) ===
def cluster_embedding_agglom(embeddings, threshold):
    n = len(embeddings)
    if n <= 1:
        return [list(range(n))]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normed = embeddings / norms
    sim = normed @ normed.T
    dist = np.clip(1 - sim, 0, 2)
    np.fill_diagonal(dist, 0)
    clust = AgglomerativeClustering(
        n_clusters=None, distance_threshold=1 - threshold,
        metric="precomputed", linkage="average"
    )
    lab = clust.fit_predict(dist)
    groups = {}
    for i, l in enumerate(lab):
        groups.setdefault(l, []).append(i)
    return list(groups.values())

# === SINdex: greedy clustering + coherence-weighted entropy ===
def compute_sindex(question, samples, embeddings):
    """Full SINdex metric (Abdaljalil et al., arXiv:2503.05980)"""
    P = len(samples)
    if P <= 1:
        return 0.0

    # Step 1: greedy single-pass clustering (distance threshold = 0.05)
    clusters = []  # list of lists of indices
    for i in range(P):
        assigned = False
        for c in clusters:
            total_dis = 0.0
            for j in c:
                total_dis += (1.0 - cosine_sim(embeddings[i], embeddings[j]))
            avg_dis = total_dis / len(c)
            if avg_dis <= SINDEX_DIST_THRESH:
                c.append(i)
                assigned = True
                break
        if not assigned:
            clusters.append([i])

    # Step 2: compute adjusted probabilities and entropy
    sindex = 0.0
    for c in clusters:
        p_i = len(c) / P
        # Intra-cluster average pairwise cosine similarity
        if len(c) == 1:
            cos_bar = 1.0
        else:
            pairs = []
            for a in range(len(c)):
                for b in range(a + 1, len(c)):
                    pairs.append(cosine_sim(embeddings[c[a]], embeddings[c[b]]))
            cos_bar = np.mean(pairs)
        # Adjusted proportion (NOT renormalized per paper)
        p_prime = p_i * cos_bar
        if p_prime > 0:
            sindex -= p_prime * math.log(p_prime)
    return sindex

# === Standard SE (Shannon entropy over cluster proportions) ===
def shannon_entropy(cluster_sizes, total):
    h = 0.0
    for sz in cluster_sizes:
        p = sz / total
        if p > 0:
            h -= p * math.log(p)
    return h

# === Main experiment ===
results_per_q = []
t0 = time.time()
questions = sample_data["per_question"]

for qi, q in enumerate(questions):
    qtext = q["question"].strip()
    samples = [s for s in q["samples"] if s and not s.startswith("ERROR")]
    if len(samples) < 2:
        continue

    label_info = labels_map.get(qtext)
    if not label_info:
        continue

    # Get embeddings for all samples
    embeddings = np.array([get_embedding(s) for s in samples])

    # 1. Jaccard SE
    jac_clusters = cluster_jaccard(samples, JACCARD_THRESH)
    se_jaccard = shannon_entropy([len(c) for c in jac_clusters], len(samples))
    n_jac = len(jac_clusters)

    # 2. Embedding SE (agglomerative, tau=0.85)
    emb_clusters = cluster_embedding_agglom(embeddings, EMB_THRESH)
    se_emb = shannon_entropy([len(c) for c in emb_clusters], len(samples))
    n_emb = len(emb_clusters)

    # 3. SINdex (full metric)
    sindex = compute_sindex(qtext, samples, embeddings)

    # 4. B1 token entropy (from v1a data)
    b1_entropy = label_info["mean_entropy"]

    # Binary label: 1=incorrect (hallucination), 0=correct
    y_true = 1 if label_info["label"] == "incorrect" else 0

    results_per_q.append({
        "qi": qi,
        "question": qtext,
        "category": label_info["category"],
        "y_true": y_true,
        "n_samples": len(samples),
        "n_clusters_jaccard": n_jac,
        "n_clusters_emb": n_emb,
        "se_jaccard": se_jaccard,
        "se_emb": se_emb,
        "sindex": sindex,
        "b1_entropy": b1_entropy,
        "scr_jaccard": 1 if n_jac == 1 else 0,
        "scr_emb": 1 if n_emb == 1 else 0,
    })

    elapsed = time.time() - t0
    eta = elapsed / (qi + 1) * (len(questions) - qi - 1)
    print(f"[{qi+1}/{len(questions)}] {elapsed:.0f}s ETA {eta:.0f}s | "
          f"Jac:{n_jac}c Emb:{n_emb}c SINdex:{sindex:.3f} B1:{b1_entropy:.3f} | "
          f"{qtext[:50]}...")
    sys.stdout.flush()

# === Compute AUROCs ===
y = np.array([r["y_true"] for r in results_per_q])
n_total = len(y)
n_pos = int(y.sum())
n_neg = n_total - n_pos

print(f"\n{'='*60}")
print(f"Exp 27: SINdex Head-to-Head (n={n_total}, pos={n_pos}, neg={n_neg})")
print(f"{'='*60}")

# Higher uncertainty → more likely incorrect → use as-is for AUROC
# For SE/SINdex: higher = more uncertain (good)
# For B1 entropy: higher = more uncertain (good)
methods = {
    "B1 mean entropy": [r["b1_entropy"] for r in results_per_q],
    "SE-Jaccard (tau=0.4)": [r["se_jaccard"] for r in results_per_q],
    "SE-Embedding (tau=0.85)": [r["se_emb"] for r in results_per_q],
    "SINdex (full, tau=0.95)": [r["sindex"] for r in results_per_q],
}

# Bootstrap AUROC with 95% CI
def bootstrap_auroc(y_true, scores, n_boot=10000):
    scores = np.array(scores)
    base = roc_auc_score(y_true, scores)
    rng = np.random.RandomState(42)
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(set(y_true[idx])) < 2:
            continue
        boots.append(roc_auc_score(y_true[idx], scores[idx]))
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return base, lo, hi

print(f"\n{'Method':<30} {'AUROC':>7} {'95% CI':>18} {'Cost':>10}")
print("-" * 70)
for name, scores in methods.items():
    auc, lo, hi = bootstrap_auroc(y, scores)
    cost = "Free" if "B1" in name else "11x"
    print(f"{name:<30} {auc:>7.3f}   [{lo:.3f}, {hi:.3f}]   {cost:>10}")

# SCR stats
scr_jac = sum(r["scr_jaccard"] for r in results_per_q) / n_total
scr_emb = sum(r["scr_emb"] for r in results_per_q) / n_total
print(f"\nSCR Jaccard: {scr_jac:.1%}")
print(f"SCR Embedding (tau=0.85): {scr_emb:.1%}")

# SINdex on single-cluster vs multi-cluster
sc_mask = np.array([r["scr_emb"] == 1 for r in results_per_q])
mc_mask = ~sc_mask
sindex_scores = np.array([r["sindex"] for r in results_per_q])

print(f"\nSINdex on single-cluster (emb): mean={sindex_scores[sc_mask].mean():.4f}, "
      f"std={sindex_scores[sc_mask].std():.4f}, n={sc_mask.sum()}")
if mc_mask.sum() > 0:
    print(f"SINdex on multi-cluster (emb):  mean={sindex_scores[mc_mask].mean():.4f}, "
          f"std={sindex_scores[mc_mask].std():.4f}, n={mc_mask.sum()}")

# AUROC on subsets
for subset_name, mask in [("Single-cluster (emb)", sc_mask), ("Multi-cluster (emb)", mc_mask)]:
    if mask.sum() < 5 or len(set(y[mask])) < 2:
        print(f"\n{subset_name}: too few samples or single class")
        continue
    print(f"\n{subset_name} (n={mask.sum()}):")
    for name, scores in methods.items():
        s = np.array(scores)
        try:
            auc = roc_auc_score(y[mask], s[mask])
            print(f"  {name:<30} AUROC={auc:.3f}")
        except:
            print(f"  {name:<30} AUROC=N/A")

# Save results
output = {
    "experiment": "exp27_sindex_headtohead",
    "n_questions": n_total,
    "n_positive": n_pos,
    "n_negative": n_neg,
    "scr_jaccard": scr_jac,
    "scr_emb": scr_emb,
    "per_question": results_per_q,
}
for name, scores in methods.items():
    auc, lo, hi = bootstrap_auroc(y, scores)
    output[f"auroc_{name}"] = {"auroc": auc, "ci_lo": lo, "ci_hi": hi}

outpath = RESULTS_DIR / "exp27_sindex_headtohead.json"
with open(outpath, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to {outpath}")
print(f"Total time: {time.time()-t0:.0f}s")
