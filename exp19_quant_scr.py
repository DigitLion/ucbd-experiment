#!/usr/bin/env python3
"""Exp 19: Quantization sensitivity — Q4_K_M vs Q8_0 on Mistral-7B-Instruct.
Protocol: N=10 samples per question, T=1.0.
Measures: Mean pairwise Jaccard (character bigram), number of clusters at multiple thresholds.
Uses 30-question subset.
"""
import json, time, sys, os, requests
import statistics

OLLAMA_URL = "http://127.0.0.1:11434"
QUESTIONS_FILE = "/Users/john/Downloads/files/tqa_200q.json"
OUTPUT_FILE = "/Users/john/Downloads/files/exp19_results.json"
N_SAMPLES = 10
TEMPERATURE = 1.0
N_QUESTIONS = 30

MODELS = {
    "Q4_K_M": "mistral:7b-instruct",
    "Q8_0": "mistral:7b-instruct-q8_0",
}

# Multiple thresholds for robustness
THRESHOLDS = [0.4, 0.5, 0.6, 0.7]


def generate_response(model: str, question: str) -> str:
    prompt = f"[INST] {question} [/INST]"
    r = requests.post(f"{OLLAMA_URL}/api/generate", json={
        "model": model,
        "prompt": prompt,
        "options": {"temperature": TEMPERATURE},
        "stream": False,
    }, timeout=120)
    r.raise_for_status()
    return r.json().get("response", "").strip()


def char_bigram_jaccard(a: str, b: str) -> float:
    """Jaccard similarity on character bigrams."""
    def bigrams(s):
        s = s.lower()
        return set(s[i:i+2] for i in range(len(s)-1)) if len(s) > 1 else set()
    ba, bb = bigrams(a), bigrams(b)
    if not ba and not bb:
        return 1.0
    if not ba or not bb:
        return 0.0
    return len(ba & bb) / len(ba | bb)


def cluster_count(similarities, n: int, threshold: float) -> int:
    """Single-linkage clustering. Returns number of clusters."""
    parent = list(range(n))
    def find(x):
        r = x
        while parent[r] != r:
            r = parent[r]
        parent[x] = r
        return r
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    for (i, j), sim in similarities.items():
        if sim >= threshold:
            union(i, j)
    return len(set(find(i) for i in range(n)))


def run_model(model_key: str, model_name: str, questions: list[str]) -> dict:
    print(f"\n{'='*60}")
    print(f"Running {model_key} ({model_name}) on {len(questions)} questions")
    print(f"{'='*60}")

    print("Warming up model...")
    try:
        generate_response(model_name, "Hello")
    except Exception as e:
        print(f"Warmup failed: {e}")
        return {}

    results = []
    t_start = time.time()

    for qi, question in enumerate(questions):
        responses = []
        for si in range(N_SAMPLES):
            try:
                resp = generate_response(model_name, question)
                responses.append(resp)
            except Exception as e:
                print(f"  Q{qi} S{si} error: {e}")
                responses.append("")

        # Compute all pairwise similarities
        sims = {}
        all_sims = []
        for i in range(N_SAMPLES):
            for j in range(i+1, N_SAMPLES):
                s = char_bigram_jaccard(responses[i], responses[j])
                sims[(i,j)] = s
                all_sims.append(s)

        mean_sim = statistics.mean(all_sims) if all_sims else 0
        avg_words = statistics.mean(len(r.split()) for r in responses)

        # Cluster at multiple thresholds
        nc_by_thresh = {}
        scr_by_thresh = {}
        for t in THRESHOLDS:
            nc = cluster_count(sims, N_SAMPLES, t)
            nc_by_thresh[str(t)] = nc
            scr_by_thresh[str(t)] = 1 if nc == 1 else 0

        results.append({
            "question_idx": qi,
            "mean_pairwise_jaccard": round(mean_sim, 4),
            "avg_words": round(avg_words, 1),
            "nc_by_threshold": nc_by_thresh,
        })

        elapsed = time.time() - t_start
        eta = (elapsed / (qi+1)) * (len(questions) - qi - 1)
        running_mean_sim = statistics.mean(r["mean_pairwise_jaccard"] for r in results)
        nc_04 = nc_by_thresh["0.4"]
        print(f"  Q{qi:3d}/{len(questions)} | J={mean_sim:.3f} | NC@0.4={nc_04:2d} | AvgW={avg_words:.0f} | MeanJ={running_mean_sim:.3f} | ETA={eta:.0f}s", flush=True)

    total_time = time.time() - t_start
    mean_jac = statistics.mean(r["mean_pairwise_jaccard"] for r in results)

    print(f"\n{model_key} RESULTS:")
    print(f"  Mean pairwise Jaccard: {mean_jac:.4f}")
    for t in THRESHOLDS:
        scr = sum(1 for r in results if r["nc_by_threshold"][str(t)] == 1) / len(results)
        mean_nc = statistics.mean(r["nc_by_threshold"][str(t)] for r in results)
        print(f"  Threshold {t}: SCR={scr:.1%}, Mean NC={mean_nc:.2f}")
    print(f"  Time: {total_time:.0f}s")

    return {
        "model_key": model_key,
        "model_name": model_name,
        "n_questions": len(questions),
        "mean_pairwise_jaccard": round(mean_jac, 4),
        "total_time_s": round(total_time, 1),
        "per_threshold": {
            str(t): {
                "scr": sum(1 for r in results if r["nc_by_threshold"][str(t)] == 1) / len(results),
                "mean_nc": statistics.mean(r["nc_by_threshold"][str(t)] for r in results),
            }
            for t in THRESHOLDS
        },
        "per_question": results,
    }


def main():
    with open(QUESTIONS_FILE) as f:
        all_questions = json.load(f)

    questions = all_questions[:N_QUESTIONS]
    print(f"Loaded {len(questions)} questions")

    all_results = {}
    for model_key, model_name in MODELS.items():
        result = run_model(model_key, model_name, questions)
        all_results[model_key] = result

    # Summary
    print(f"\n{'='*60}")
    print("QUANTIZATION SENSITIVITY COMPARISON")
    print(f"{'='*60}")

    q4 = all_results.get("Q4_K_M", {})
    q8 = all_results.get("Q8_0", {})

    if q4 and q8:
        print(f"\n  Mean pairwise Jaccard:")
        print(f"    Q4_K_M: {q4['mean_pairwise_jaccard']:.4f}")
        print(f"    Q8_0:   {q8['mean_pairwise_jaccard']:.4f}")
        print(f"    Delta:  {q8['mean_pairwise_jaccard'] - q4['mean_pairwise_jaccard']:+.4f}")

        print(f"\n  SCR and NC at each threshold:")
        for t in THRESHOLDS:
            ts = str(t)
            q4t = q4["per_threshold"][ts]
            q8t = q8["per_threshold"][ts]
            print(f"    t={t}: Q4 SCR={q4t['scr']:.1%} NC={q4t['mean_nc']:.2f} | Q8 SCR={q8t['scr']:.1%} NC={q8t['mean_nc']:.2f} | dSCR={q8t['scr']-q4t['scr']:+.1%} dNC={q8t['mean_nc']-q4t['mean_nc']:+.2f}")

        # Per-question Jaccard correlation
        q4_jacs = [r["mean_pairwise_jaccard"] for r in q4["per_question"]]
        q8_jacs = [r["mean_pairwise_jaccard"] for r in q8["per_question"]]
        # Pearson correlation
        n = len(q4_jacs)
        mean4 = sum(q4_jacs)/n
        mean8 = sum(q8_jacs)/n
        cov = sum((a-mean4)*(b-mean8) for a,b in zip(q4_jacs, q8_jacs)) / n
        std4 = (sum((a-mean4)**2 for a in q4_jacs) / n) ** 0.5
        std8 = (sum((b-mean8)**2 for b in q8_jacs) / n) ** 0.5
        r = cov / (std4 * std8) if std4 > 0 and std8 > 0 else 0
        print(f"\n  Per-question Jaccard correlation (Q4 vs Q8): r={r:.3f}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
