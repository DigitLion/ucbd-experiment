#!/usr/bin/env python3
"""Exp 22: WebQuestions SCR measurement — new dataset validation
Addresses reviewer complaint: 'Heavy reliance on TruthfulQA'
"""
import json, time, requests, sys
from pathlib import Path

with open(Path(__file__).parent / "webq_200q.json") as f:
    questions = json.load(f)

MODEL = "mistral:7b-instruct"
N_SAMPLES = 10
MAX_TOKENS = 100  # standard length
TEMPERATURE = 1.0
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

results = {
    "experiment": "exp22_webq_200q",
    "model": MODEL,
    "n_questions": len(questions),
    "n_samples": N_SAMPLES,
    "max_tokens": MAX_TOKENS,
    "temperature": TEMPERATURE,
    "per_question": []
}

t0 = time.time()

for qi, q in enumerate(questions):
    q_text = q["question"]
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
                    "seed": 42 + si
                }
            }, timeout=120)
            data = resp.json()
            samples.append(data.get("response", "").strip())
        except Exception as e:
            samples.append(f"ERROR: {e}")

    results["per_question"].append({
        "qi": qi,
        "question": q_text,
        "answers": q.get("answers", []),
        "samples": samples
    })

    elapsed = time.time() - t0
    eta = elapsed / (qi + 1) * (len(questions) - qi - 1)
    print(f"[{qi+1}/{len(questions)}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s | {q_text[:60]}...")
    sys.stdout.flush()

results["total_time_s"] = time.time() - t0

outpath = Path(__file__).parent / "exp22_webq_samples.json"
with open(outpath, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {outpath} ({results['total_time_s']:.0f}s total)")
