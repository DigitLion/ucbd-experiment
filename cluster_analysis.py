#!/usr/bin/env python3
"""Cluster analysis for Exp 21/22 — compute SCR at multiple thresholds using embeddings"""
import json, time, requests, sys, numpy as np
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering

EMBEDDING_URL = "http://127.0.0.1:11434/api/embed"
EMBEDDING_MODEL = "qwen3-embedding"
THRESHOLDS = [0.80, 0.85, 0.90]

def get_embedding(text):
    resp = requests.post(EMBEDDING_URL, json={
        "model": EMBEDDING_MODEL,
        "input": text
    }, timeout=60)
    return np.array(resp.json()["embeddings"][0])

def compute_clusters(embeddings, threshold):
    """Agglomerative clustering with cosine distance threshold"""
    n = len(embeddings)
    if n <= 1:
        return 1
    # Compute cosine similarity matrix
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normed = embeddings / norms
    sim_matrix = normed @ normed.T
    # Convert to distance
    dist_matrix = 1 - sim_matrix
    np.fill_diagonal(dist_matrix, 0)
    dist_matrix = np.clip(dist_matrix, 0, 2)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1 - threshold,
        metric='precomputed',
        linkage='average'
    )
    labels = clustering.fit_predict(dist_matrix)
    return len(set(labels))

def analyze_samples(input_file, output_file):
    with open(input_file) as f:
        data = json.load(f)

    results = {
        "experiment": data["experiment"] + "_clustered",
        "model": data["model"],
        "n_questions": data["n_questions"],
        "thresholds": THRESHOLDS,
        "per_question": []
    }

    scr = {t: 0 for t in THRESHOLDS}
    t0 = time.time()

    for qi, q in enumerate(data["per_question"]):
        samples = [s for s in q["samples"] if not s.startswith("ERROR")]
        if len(samples) < 2:
            q_result = {"qi": qi, "question": q["question"], "n_valid": len(samples)}
            for t in THRESHOLDS:
                q_result[f"nc_{t}"] = 1
                scr[t] += 1
            results["per_question"].append(q_result)
            continue

        # Get embeddings
        embeddings = np.array([get_embedding(s) for s in samples])

        q_result = {
            "qi": qi,
            "question": q["question"],
            "n_valid": len(samples)
        }

        for t in THRESHOLDS:
            nc = compute_clusters(embeddings, t)
            q_result[f"nc_{t}"] = nc
            if nc == 1:
                scr[t] += 1

        results["per_question"].append(q_result)

        elapsed = time.time() - t0
        eta = elapsed / (qi + 1) * (len(data["per_question"]) - qi - 1)
        nc_str = " | ".join(f"τ={t}: {q_result[f'nc_{t}']}c" for t in THRESHOLDS)
        print(f"[{qi+1}/{len(data['per_question'])}] {elapsed:.0f}s, ETA {eta:.0f}s | {nc_str} | {q['question'][:50]}...")
        sys.stdout.flush()

    # Summary
    n = len(data["per_question"])
    results["summary"] = {}
    for t in THRESHOLDS:
        results["summary"][f"SCR_{t}"] = scr[t] / n
        results["summary"][f"SCR_{t}_count"] = f"{scr[t]}/{n}"

    results["total_time_s"] = time.time() - t0

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== RESULTS ===")
    for t in THRESHOLDS:
        print(f"SCR@{t}: {scr[t]/n:.1%} ({scr[t]}/{n})")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python cluster_analysis.py <input.json> <output.json>")
        sys.exit(1)
    analyze_samples(sys.argv[1], sys.argv[2])
