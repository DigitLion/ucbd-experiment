#!/usr/bin/env python3
"""Exp 21: Extended max-tokens sensitivity — 200 questions at max_tokens=200
Addresses reviewer complaint: '50-item length ablation too small'
"""
import json, time, requests, sys
from pathlib import Path

# Load TruthfulQA 200q subset
with open(Path(__file__).parent / "tqa_200q.json") as f:
    questions = json.load(f)

MODEL = "mistral:7b-instruct"
N_SAMPLES = 10
MAX_TOKENS = 200
TEMPERATURE = 1.0
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

results = {
    "experiment": "exp21_200tok_200q",
    "model": MODEL,
    "n_questions": len(questions),
    "n_samples": N_SAMPLES,
    "max_tokens": MAX_TOKENS,
    "temperature": TEMPERATURE,
    "per_question": []
}

t0 = time.time()

for qi, question in enumerate(questions):
    q_text = question if isinstance(question, str) else question.get("question", str(question))
    prompt = f"Answer the following question concisely:\n\nQ: {q_text}\nA:"

    samples = []
    for si in range(N_SAMPLES):
        try:
            resp = requests.post(OLLAMA_URL, json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": TEMPERATURE,
                    "num_predict": MAX_TOKENS,
                    "seed": 42 + si  # different seed per sample
                }
            }, timeout=120)
            data = resp.json()
            samples.append(data.get("response", "").strip())
        except Exception as e:
            samples.append(f"ERROR: {e}")

    results["per_question"].append({
        "qi": qi,
        "question": q_text,
        "samples": samples
    })

    elapsed = time.time() - t0
    eta = elapsed / (qi + 1) * (len(questions) - qi - 1)
    print(f"[{qi+1}/{len(questions)}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s | {q_text[:60]}...")
    sys.stdout.flush()

results["total_time_s"] = time.time() - t0

# Save raw samples
outpath = Path(__file__).parent / "exp21_200tok_samples.json"
with open(outpath, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {outpath} ({results['total_time_s']:.0f}s total)")
