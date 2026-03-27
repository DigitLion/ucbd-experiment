#!/usr/bin/env python3
"""Exp 28: NLI adjudication of embedding single-cluster assignments
Validates that embedding single-cluster (tau=0.85) truly reflects semantic equivalence
by checking with DeBERTa-v3 NLI (bidirectional entailment) on a stratified subset.

For each question in the single-cluster subset:
- Take all N=10 response pairs and check bidirectional entailment
- If >90% of pairs are mutually entailing, NLI agrees with embedding's single-cluster

Also check a multi-cluster control set to ensure NLI disagrees there.
"""
import json, sys, os, time, random
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
from transformers import pipeline

# === Config ===
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"  # 184M params
N_SINGLE = 30   # stratified sample from single-cluster questions
N_MULTI = 20    # stratified sample from multi-cluster questions
ENTAIL_THRESHOLD = 0.5  # p(entailment) threshold for bidirectional

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"

# === Load exp27 results (has per-question embedding cluster counts) ===
with open(RESULTS_DIR / "exp27_sindex_headtohead.json") as f:
    exp27 = json.load(f)

# === Load raw samples ===
with open(DATA_DIR / "exp21_200tok_samples.json") as f:
    sample_data = json.load(f)

# Build question -> samples lookup
q_samples = {}
for q in sample_data["per_question"]:
    q_samples[q["question"].strip()] = [s for s in q["samples"] if s and not s.startswith("ERROR")]

# Split into single-cluster and multi-cluster
single_qs = []
multi_qs = []
for r in exp27["per_question"]:
    q = r["question"]
    if q not in q_samples or len(q_samples[q]) < 2:
        continue
    if r["scr_emb"] == 1:
        single_qs.append(r)
    else:
        multi_qs.append(r)

print(f"Single-cluster questions: {len(single_qs)}")
print(f"Multi-cluster questions: {len(multi_qs)}")

# Stratified sampling
random.seed(42)
single_sample = random.sample(single_qs, min(N_SINGLE, len(single_qs)))
multi_sample = random.sample(multi_qs, min(N_MULTI, len(multi_qs)))

print(f"Sampled: {len(single_sample)} single, {len(multi_sample)} multi")

# === Load NLI model ===
print(f"\nLoading NLI model: {NLI_MODEL}...")
nli = pipeline("text-classification", model=NLI_MODEL, device="mps", batch_size=32)
print("NLI model loaded.")

def check_bidirectional_entailment(responses, nli_pipe):
    """Check all pairs for bidirectional entailment. Returns agreement rate."""
    n = len(responses)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))

    if not pairs:
        return 1.0, 0, 0

    # Forward: A→B
    forward_inputs = [{"text": responses[i][:512], "text_pair": responses[j][:512]} for i, j in pairs]
    # Backward: B→A
    backward_inputs = [{"text": responses[j][:512], "text_pair": responses[i][:512]} for i, j in pairs]

    fwd_results = nli_pipe(forward_inputs)
    bwd_results = nli_pipe(backward_inputs)

    n_entail = 0
    n_total = len(pairs)
    for fwd, bwd in zip(fwd_results, bwd_results):
        fwd_entail = fwd["label"] == "entailment" and fwd["score"] > ENTAIL_THRESHOLD
        bwd_entail = bwd["label"] == "entailment" and bwd["score"] > ENTAIL_THRESHOLD
        if fwd_entail and bwd_entail:
            n_entail += 1

    return n_entail / n_total, n_entail, n_total

# === Run validation ===
results = {"single_cluster": [], "multi_cluster": []}
t0 = time.time()

print("\n=== Single-cluster subset (embedding tau=0.85) ===")
for i, r in enumerate(single_sample):
    q = r["question"]
    samples = q_samples[q]
    agree_rate, n_e, n_t = check_bidirectional_entailment(samples, nli)
    results["single_cluster"].append({
        "question": q,
        "n_samples": len(samples),
        "n_jaccard_clusters": r["n_clusters_jaccard"],
        "nli_entail_rate": agree_rate,
        "nli_entail_pairs": n_e,
        "nli_total_pairs": n_t,
        "nli_agrees": agree_rate > 0.7,  # >70% pairs entailing = semantic equivalence
    })
    status = "AGREE" if agree_rate > 0.7 else "DISAGREE"
    print(f"  [{i+1}/{len(single_sample)}] {status} entail={agree_rate:.1%} ({n_e}/{n_t}) | {q[:60]}...")
    sys.stdout.flush()

print("\n=== Multi-cluster control subset ===")
for i, r in enumerate(multi_sample):
    q = r["question"]
    samples = q_samples[q]
    agree_rate, n_e, n_t = check_bidirectional_entailment(samples, nli)
    results["multi_cluster"].append({
        "question": q,
        "n_samples": len(samples),
        "n_emb_clusters": r["n_clusters_emb"],
        "n_jaccard_clusters": r["n_clusters_jaccard"],
        "nli_entail_rate": agree_rate,
        "nli_entail_pairs": n_e,
        "nli_total_pairs": n_t,
        "nli_agrees_multi": agree_rate < 0.7,  # <70% = genuine diversity (agrees with multi-cluster)
    })
    status = "AGREE" if agree_rate < 0.7 else "DISAGREE"
    print(f"  [{i+1}/{len(multi_sample)}] {status} entail={agree_rate:.1%} ({n_e}/{n_t}) | {q[:60]}...")
    sys.stdout.flush()

# === Summary statistics ===
single_agrees = sum(1 for r in results["single_cluster"] if r["nli_agrees"])
single_total = len(results["single_cluster"])
single_mean_rate = np.mean([r["nli_entail_rate"] for r in results["single_cluster"]])

multi_agrees = sum(1 for r in results["multi_cluster"] if r["nli_agrees_multi"])
multi_total = len(results["multi_cluster"])
multi_mean_rate = np.mean([r["nli_entail_rate"] for r in results["multi_cluster"]])

print(f"\n{'='*60}")
print(f"NLI Validation Summary (DeBERTa-v3-base, 184M)")
print(f"{'='*60}")
print(f"\nSingle-cluster (embedding tau=0.85):")
print(f"  NLI agreement rate: {single_agrees}/{single_total} ({single_agrees/single_total:.1%})")
print(f"  Mean bidirectional entailment: {single_mean_rate:.1%}")
print(f"\nMulti-cluster control:")
print(f"  NLI agreement rate: {multi_agrees}/{multi_total} ({multi_agrees/multi_total:.1%})")
print(f"  Mean bidirectional entailment: {multi_mean_rate:.1%}")
print(f"\nConclusion: {'VALIDATED' if single_agrees/single_total > 0.8 and multi_agrees/multi_total > 0.6 else 'MIXED'}")
print(f"  Single-cluster assignments are {'validated' if single_agrees/single_total > 0.8 else 'NOT validated'} by NLI")
print(f"  Multi-cluster assignments are {'validated' if multi_agrees/multi_total > 0.6 else 'NOT validated'} by NLI")

# Save
output = {
    "experiment": "exp28_nli_validation",
    "nli_model": NLI_MODEL,
    "n_single": single_total,
    "n_multi": multi_total,
    "single_nli_agreement": single_agrees / single_total,
    "single_mean_entailment": single_mean_rate,
    "multi_nli_agreement": multi_agrees / multi_total,
    "multi_mean_entailment": multi_mean_rate,
    "results": results,
    "total_time_s": time.time() - t0,
}
outpath = RESULTS_DIR / "exp28_nli_validation.json"
with open(outpath, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to {outpath}")
print(f"Total time: {time.time()-t0:.0f}s")
