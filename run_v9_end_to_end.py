#!/usr/bin/env python3
"""
UCBD V9: End-to-End Mixed-Stream Experiment (MLX Direct)
========================================================
Tests whether B1 entropy predicts RAG benefit on HotpotQA.
Uses MLX directly (not oMLX API) to get real logprobs.

Design:
  - 100 HotpotQA questions (bridge + comparison, all "hard")
  - For each question: generate answer WITHOUT context → record entropy
  - For each question: generate answer WITH gold context → record answer
  - Compute F1 for both conditions
  - Analyze: does entropy predict RAG_benefit (F1_with - F1_without)?
  - Routing experiment: threshold-based selective retrieval

Usage:
  python3 run_v9_end_to_end.py              # full run (100 questions)
  python3 run_v9_end_to_end.py --limit 10   # quick test
"""

import json
import math
import re
import sys
import time
import argparse
import statistics
from pathlib import Path
from collections import defaultdict

import mlx.core as mx
from mlx_lm import load, stream_generate

# === Config ===
MODEL_PATH = str(Path.home() / "Models" / "Qwen3-14B-4bit")
DATA_PATH = Path(__file__).parent / "data" / "hotpot_100.json"
RESULTS_PATH = Path(__file__).parent / "results" / "v9_end_to_end.jsonl"
SUMMARY_PATH = Path(__file__).parent / "results" / "v9_summary.json"
MAX_TOKENS = 150


def compute_entropy(logprobs: mx.array) -> float:
    """Compute Shannon entropy from log-probability vector."""
    mask = logprobs > -1e6
    lp = mx.where(mask, logprobs, mx.zeros_like(logprobs))
    p = mx.exp(lp)
    H = -mx.sum(p * lp)
    return float(H.item())


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> block if present."""
    if "<think>" in text:
        end = text.find("</think>")
        if end >= 0:
            return text[end + len("</think>"):].strip()
        return text[text.find("<think>") + len("<think>"):].strip()
    return text.strip()


def normalize_answer(s: str) -> str:
    """Normalize answer for F1 computation."""
    s = s.lower().strip()
    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # Remove punctuation
    s = re.sub(r'[^\w\s]', '', s)
    # Collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_em(prediction: str, ground_truth: str) -> float:
    """Compute exact match."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def get_gold_context(question_data: dict) -> str:
    """Extract gold supporting paragraphs from HotpotQA question."""
    supporting = set()
    for title, sent_idx in question_data.get("supporting_facts", []):
        supporting.add(title)

    context_parts = []
    for title, sentences in question_data.get("context", []):
        if title in supporting:
            context_parts.append(f"{title}: {' '.join(sentences)}")

    return "\n".join(context_parts)


def generate_with_entropy(model, tokenizer, prompt, max_tokens=MAX_TOKENS):
    """Generate text and collect per-token entropy."""
    entropies = []
    text_parts = []
    greedy = lambda logits: mx.argmax(logits, axis=-1)

    for resp in stream_generate(
        model, tokenizer, prompt,
        max_tokens=max_tokens,
        sampler=greedy,
    ):
        if resp.logprobs is not None:
            H = compute_entropy(resp.logprobs)
            entropies.append(H)
        text_parts.append(resp.text)
        if resp.finish_reason:
            break

    full_text = "".join(text_parts)
    answer = strip_thinking(full_text)
    return answer, entropies


def build_prompt(tokenizer, question: str, context: str = None):
    """Build chat prompt with optional context."""
    if context:
        user_msg = f"Based on the following information, answer the question briefly.\n\nContext:\n{context}\n\nQuestion: {question}"
    else:
        user_msg = f"Answer the following question briefly in 1-2 sentences.\n\nQuestion: {question}"

    messages = [
        {"role": "system", "content": "Answer briefly and directly. Give only the answer, no explanation."},
        {"role": "user", "content": user_msg},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def run_experiment(limit=None):
    print("=== UCBD V9: End-to-End Mixed-Stream Experiment ===")
    print(f"Model: {MODEL_PATH}")
    print()

    # Load model
    t0 = time.time()
    print("Loading model...", end=" ", flush=True)
    model, tokenizer = load(MODEL_PATH)
    print(f"done ({time.time() - t0:.1f}s)")

    # Load HotpotQA
    with open(DATA_PATH) as f:
        questions = json.load(f)
    total = min(len(questions), limit) if limit else len(questions)
    print(f"HotpotQA questions: {total}")

    # Resume support
    done = set()
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    done.add(d["question_idx"])
        print(f"Resuming: {len(done)} already completed")

    remaining = total - len(done)
    if remaining <= 0:
        print("All questions already processed.")
        analyze_results()
        return

    print(f"Running {remaining} questions (2 generations each)...")
    print()

    t_start = time.time()
    processed = 0

    for idx in range(total):
        if idx in done:
            continue

        q = questions[idx]
        question_text = q["question"]
        gold_answer = q["answer"]
        q_type = q.get("type", "unknown")
        gold_context = get_gold_context(q)

        # === Condition A: No RAG (with entropy tracking) ===
        prompt_no_rag = build_prompt(tokenizer, question_text)
        answer_no_rag, entropies = generate_with_entropy(model, tokenizer, prompt_no_rag)

        # Entropy stats
        if entropies:
            mean_H = sum(entropies) / len(entropies)
            max_H = max(entropies)
            std_H = (sum((h - mean_H)**2 for h in entropies) / len(entropies)) ** 0.5
        else:
            mean_H = max_H = std_H = 0.0

        # === Condition B: With RAG (gold paragraphs) ===
        prompt_with_rag = build_prompt(tokenizer, question_text, context=gold_context)
        answer_with_rag, _ = generate_with_entropy(model, tokenizer, prompt_with_rag)

        # Compute F1 and EM
        f1_no_rag = compute_f1(answer_no_rag, gold_answer)
        f1_with_rag = compute_f1(answer_with_rag, gold_answer)
        em_no_rag = compute_em(answer_no_rag, gold_answer)
        em_with_rag = compute_em(answer_with_rag, gold_answer)
        rag_benefit = f1_with_rag - f1_no_rag

        result = {
            "question_idx": idx,
            "question": question_text[:200],
            "gold_answer": gold_answer,
            "type": q_type,
            "answer_no_rag": answer_no_rag[:300],
            "answer_with_rag": answer_with_rag[:300],
            "f1_no_rag": round(f1_no_rag, 4),
            "f1_with_rag": round(f1_with_rag, 4),
            "em_no_rag": round(em_no_rag, 4),
            "em_with_rag": round(em_with_rag, 4),
            "rag_benefit": round(rag_benefit, 4),
            "mean_entropy": round(mean_H, 4),
            "max_entropy": round(max_H, 4),
            "std_entropy": round(std_H, 4),
            "num_tokens": len(entropies),
        }

        with open(RESULTS_PATH, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        processed += 1
        elapsed = time.time() - t_start
        rate = processed / (elapsed / 60) if elapsed > 0 else 0

        if processed % 5 == 0 or processed <= 3:
            print(
                f"[{processed}/{remaining}] Q{idx} type={q_type:10s} "
                f"F1: {f1_no_rag:.2f}→{f1_with_rag:.2f} (Δ={rag_benefit:+.2f}) "
                f"H={mean_H:.3f} ({rate:.1f} q/min)"
            )

    elapsed = time.time() - t_start
    print(f"\n=== Done in {elapsed/60:.1f} min ===")
    analyze_results()


def analyze_results():
    """Analyze completed results and generate summary."""
    print("\n=== V9 Analysis ===\n")

    results = []
    with open(RESULTS_PATH) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    n = len(results)
    if n < 5:
        print(f"Only {n} results, skipping analysis")
        return

    # Basic stats
    f1_no = [r["f1_no_rag"] for r in results]
    f1_rag = [r["f1_with_rag"] for r in results]
    benefits = [r["rag_benefit"] for r in results]
    entropies = [r["mean_entropy"] for r in results]

    print(f"N = {n}")
    print(f"No-RAG F1:  {statistics.mean(f1_no):.3f} ± {statistics.stdev(f1_no):.3f}")
    print(f"With-RAG F1: {statistics.mean(f1_rag):.3f} ± {statistics.stdev(f1_rag):.3f}")
    print(f"RAG benefit: {statistics.mean(benefits):.3f} ± {statistics.stdev(benefits):.3f}")
    print(f"Mean entropy: {statistics.mean(entropies):.3f} ± {statistics.stdev(entropies):.3f}")

    # Pearson correlation: entropy vs RAG_benefit
    mean_e = statistics.mean(entropies)
    mean_b = statistics.mean(benefits)
    cov = sum((e - mean_e) * (b - mean_b) for e, b in zip(entropies, benefits)) / n
    std_e = statistics.stdev(entropies)
    std_b = statistics.stdev(benefits)
    r = cov / (std_e * std_b) if std_e > 0 and std_b > 0 else 0
    print(f"\nEntropy-RAG_benefit Pearson r = {r:.3f}")

    # By question type
    by_type = defaultdict(list)
    for r_ in results:
        by_type[r_["type"]].append(r_)

    print("\nBy question type:")
    for qtype, items in sorted(by_type.items()):
        f1_no_t = statistics.mean([i["f1_no_rag"] for i in items])
        f1_rag_t = statistics.mean([i["f1_with_rag"] for i in items])
        h_t = statistics.mean([i["mean_entropy"] for i in items])
        print(f"  {qtype:12s} (n={len(items):3d}): F1 {f1_no_t:.3f}→{f1_rag_t:.3f} (Δ={f1_rag_t-f1_no_t:+.3f}), H={h_t:.3f}")

    # Quartile analysis: split by entropy
    sorted_by_entropy = sorted(results, key=lambda x: x["mean_entropy"])
    q_size = n // 4

    print("\nQuartile analysis (by entropy):")
    for qi in range(4):
        start = qi * q_size
        end = start + q_size if qi < 3 else n
        quartile = sorted_by_entropy[start:end]
        if not quartile:
            continue
        q_entropies = [q_["mean_entropy"] for q_ in quartile]
        q_benefits = [q_["rag_benefit"] for q_ in quartile]
        q_f1_no = [q_["f1_no_rag"] for q_ in quartile]
        q_f1_rag = [q_["f1_with_rag"] for q_ in quartile]
        print(
            f"  Q{qi+1} (H={statistics.mean(q_entropies):.3f}): "
            f"No-RAG F1={statistics.mean(q_f1_no):.3f}, "
            f"RAG F1={statistics.mean(q_f1_rag):.3f}, "
            f"Δ={statistics.mean(q_benefits):+.3f}"
        )

    # Routing experiment: sweep entropy thresholds
    print("\nRouting experiment (retrieve if entropy > threshold):")
    print(f"  {'Pct':>5s}  {'Threshold':>9s}  {'F1':>6s}  {'Cost':>5s}  {'Recovery':>9s}")

    f1_always = statistics.mean(f1_rag)
    f1_never = statistics.mean(f1_no)
    f1_gap = f1_always - f1_never

    best_efficiency = 0
    best_row = None

    for pct in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        if pct == 0:
            thresh = float('inf')
        elif pct == 100:
            thresh = -float('inf')
        else:
            idx = int(n * (100 - pct) / 100)
            thresh = sorted_by_entropy[min(idx, n-1)]["mean_entropy"]

        # Route: retrieve if entropy > threshold
        routed_f1s = []
        retrieval_count = 0
        for r_ in results:
            if r_["mean_entropy"] > thresh:
                routed_f1s.append(r_["f1_with_rag"])
                retrieval_count += 1
            else:
                routed_f1s.append(r_["f1_no_rag"])

        routed_f1 = statistics.mean(routed_f1s)
        cost = retrieval_count / n
        recovery = (routed_f1 - f1_never) / f1_gap if f1_gap > 0 else 0
        efficiency = recovery / cost if cost > 0 else 0

        if efficiency > best_efficiency and 0 < cost < 1:
            best_efficiency = efficiency
            best_row = (pct, thresh, routed_f1, cost, recovery)

        print(f"  {pct:4d}%  {thresh:9.3f}  {routed_f1:.3f}  {cost:5.1%}  {recovery:8.1%}")

    if best_row:
        print(f"\n  Best efficiency: P{best_row[0]} → F1={best_row[2]:.3f} at {best_row[3]:.0%} cost ({best_row[4]:.0%} recovery)")

    # AUC: does entropy discriminate "RAG helps" vs "RAG doesn't help"?
    # Define "RAG helps" as rag_benefit > 0.1
    rag_helps = [(r_["mean_entropy"], 1 if r_["rag_benefit"] > 0.1 else 0) for r_ in results]
    rag_helps.sort(key=lambda x: -x[0])  # descending entropy
    n_pos = sum(x[1] for x in rag_helps)
    n_neg = n - n_pos
    if n_pos > 0 and n_neg > 0:
        auc_sum = 0
        tp = 0
        for entropy_val, label in rag_helps:
            if label == 1:
                tp += 1
            else:
                auc_sum += tp
        auc = auc_sum / (n_pos * n_neg)
        print(f"\nB1 AUC for 'RAG helps' prediction: {auc:.3f} (n_pos={n_pos}, n_neg={n_neg})")
    else:
        auc = 0.5
        print(f"\nCannot compute AUC: n_pos={n_pos}, n_neg={n_neg}")

    # Save summary
    summary = {
        "n": n,
        "f1_no_rag": round(statistics.mean(f1_no), 4),
        "f1_with_rag": round(statistics.mean(f1_rag), 4),
        "rag_gap": round(f1_gap, 4),
        "mean_entropy": round(statistics.mean(entropies), 4),
        "entropy_rag_pearson_r": round(r, 4),
        "b1_auc_rag_helps": round(auc, 4),
        "best_routing": {
            "percentile": best_row[0] if best_row else None,
            "f1": round(best_row[2], 4) if best_row else None,
            "cost": round(best_row[3], 4) if best_row else None,
            "recovery": round(best_row[4], 4) if best_row else None,
        } if best_row else None,
    }

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {SUMMARY_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    if args.analyze_only:
        analyze_results()
    else:
        run_experiment(limit=args.limit)
