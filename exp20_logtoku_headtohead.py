#!/usr/bin/env python3
"""Exp 20: LogTokU head-to-head — token entropy vs negative log-probability
on TruthfulQA using Mistral-7B-Instruct with logprobs from Ollama.

Both metrics are logit-based, single-pass (free). We compare AUROC on
a 100-question subset. Labels from TruthfulQA gold answer templates.
"""
import json, time, math, requests, re, csv, io, urllib.request
from sklearn.metrics import roc_auc_score

OLLAMA_URL = "http://127.0.0.1:11434"
MODEL = "mistral:7b-instruct"
QUESTIONS_FILE = "/Users/john/Downloads/files/tqa_200q.json"
OUTPUT_FILE = "/Users/john/Downloads/files/exp20_results.json"
N_QUESTIONS = 100
TQA_CSV_URL = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"


def load_tqa_with_answers():
    """Load TruthfulQA questions with best_answer and correct_answers for labeling."""
    req = urllib.request.Request(TQA_CSV_URL)
    req.add_header('User-Agent', 'Mozilla/5.0')
    data = urllib.request.urlopen(req, timeout=30).read().decode('utf-8')
    reader = csv.DictReader(io.StringIO(data))
    rows = list(reader)
    return rows


def generate_with_logprobs(question: str) -> dict:
    """Generate greedy response with per-token logprobs via Ollama OpenAI-compat API."""
    r = requests.post(f"{OLLAMA_URL}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [{"role": "user", "content": question}],
        "max_tokens": 100,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 1,
    }, timeout=60)
    r.raise_for_status()
    data = r.json()
    choice = data["choices"][0]
    content = choice["message"]["content"]
    logprobs_data = choice.get("logprobs", {})
    content_logprobs = logprobs_data.get("content", [])

    token_logprobs = []
    token_entropies = []

    for tok_info in content_logprobs:
        lp = tok_info.get("logprob", 0)
        token_logprobs.append(lp)

        # Approximate entropy from top_logprobs
        # With only top-1, entropy ≈ -log(p_top) as lower bound
        # More accurate with more top_logprobs, but this is a reasonable proxy
        p_top = math.exp(lp) if lp > -50 else 0
        # Entropy contribution from top token
        if p_top > 0 and p_top < 1:
            # Assume remaining mass is uniform over remaining vocab
            # Simple approximation: H ≈ -p*log(p) - (1-p)*log((1-p)/V)
            # But with only top-1, just use -log(p) as uncertainty signal
            token_entropies.append(-lp)
        else:
            token_entropies.append(0)

    n_tokens = len(token_logprobs)
    if n_tokens == 0:
        return {"content": content, "mean_neglogprob": 0, "mean_entropy_proxy": 0, "n_tokens": 0}

    mean_neglogprob = -sum(token_logprobs) / n_tokens  # LogTokU-style
    mean_entropy_proxy = sum(token_entropies) / n_tokens  # B1 proxy (same as neglogprob with top-1)
    total_logprob = sum(token_logprobs)  # Sequence log-prob (PRO-style)

    return {
        "content": content,
        "mean_neglogprob": mean_neglogprob,      # LogTokU: mean(-log p_chosen)
        "mean_entropy_proxy": mean_entropy_proxy,  # Proxy entropy (same as neglogprob with top-1)
        "total_logprob": total_logprob,            # PRO: sum(log p_chosen)
        "norm_logprob": total_logprob / n_tokens,  # Normalized log-prob
        "n_tokens": n_tokens,
    }


def label_response(response: str, tqa_row: dict) -> int:
    """Label response as correct (1) or incorrect (0) using TruthfulQA gold answers."""
    best = tqa_row.get("Best Answer", "").strip().lower()
    correct_answers = tqa_row.get("Correct Answers", "").strip().lower()
    incorrect_answers = tqa_row.get("Incorrect Answers", "").strip().lower()

    resp_lower = response.strip().lower()

    # Check if response contains the best answer or any correct answer
    if best and best in resp_lower:
        return 1

    # Check correct answers (semicolon-separated)
    for ans in correct_answers.split(";"):
        ans = ans.strip()
        if ans and len(ans) > 3 and ans in resp_lower:
            return 1

    # Check if response contains any incorrect answer
    for ans in incorrect_answers.split(";"):
        ans = ans.strip()
        if ans and len(ans) > 3 and ans in resp_lower:
            return 0

    # Fallback: check if key terms from best answer appear
    best_words = set(best.split())
    resp_words = set(resp_lower.split())
    if len(best_words) > 2:
        overlap = len(best_words & resp_words) / len(best_words)
        return 1 if overlap > 0.5 else 0

    return 0  # Default to incorrect if no match


def main():
    print("Loading TruthfulQA with gold answers...")
    tqa_rows = load_tqa_with_answers()
    print(f"Loaded {len(tqa_rows)} questions with gold answers")

    questions = tqa_rows[:N_QUESTIONS]

    print(f"\nGenerating greedy responses with logprobs for {len(questions)} questions...")
    print(f"Model: {MODEL}")

    results = []
    t_start = time.time()

    for qi, row in enumerate(questions):
        question = row["Question"]
        try:
            gen = generate_with_logprobs(question)
            label = label_response(gen["content"], row)

            results.append({
                "question_idx": qi,
                "question": question,
                "response": gen["content"][:200],
                "label": label,
                "mean_neglogprob": gen["mean_neglogprob"],
                "total_logprob": gen["total_logprob"],
                "norm_logprob": gen["norm_logprob"],
                "n_tokens": gen["n_tokens"],
            })

            elapsed = time.time() - t_start
            eta = (elapsed / (qi+1)) * (len(questions) - qi - 1)
            print(f"  Q{qi:3d}/{len(questions)} | NLP={gen['mean_neglogprob']:.3f} | L={label} | {gen['n_tokens']}tok | ETA={eta:.0f}s", flush=True)

        except Exception as e:
            print(f"  Q{qi:3d} ERROR: {e}")

    total_time = time.time() - t_start

    # Compute AUROC
    labels = [r["label"] for r in results]
    neglogprobs = [r["mean_neglogprob"] for r in results]
    normlogprobs = [-r["norm_logprob"] for r in results]  # Negate: lower logprob = higher uncertainty

    n_correct = sum(labels)
    n_incorrect = len(labels) - n_correct
    print(f"\nLabels: {n_correct} correct, {n_incorrect} incorrect ({n_correct/len(labels):.0%} accuracy)")

    if n_correct > 0 and n_incorrect > 0:
        # LogTokU-style: mean(-log p) as uncertainty (higher = more uncertain)
        auroc_logtoku = roc_auc_score(
            [1 - l for l in labels],  # 1=incorrect (uncertain), 0=correct
            neglogprobs  # Higher neglogprob = more uncertain
        )

        # PRO-style: normalized log-prob (lower = more uncertain)
        auroc_pro = roc_auc_score(
            [1 - l for l in labels],
            normlogprobs  # Higher = more uncertain (negated)
        )

        print(f"\n{'='*50}")
        print("HEAD-TO-HEAD RESULTS (Mistral-7B-Instruct)")
        print(f"{'='*50}")
        print(f"  LogTokU (mean -log p):    AUROC = {auroc_logtoku:.3f}")
        print(f"  PRO (norm log-prob):       AUROC = {auroc_pro:.3f}")
        print(f"  [Paper B1 entropy on Qwen]: AUROC = 0.599")
        print(f"  Note: LogTokU ≈ PRO here because with top-1 logprobs,")
        print(f"  both reduce to mean negative log-probability.")
        print(f"  Time: {total_time:.0f}s")
    else:
        auroc_logtoku = None
        auroc_pro = None
        print("Cannot compute AUROC: need both correct and incorrect labels")

    # Save
    output = {
        "model": MODEL,
        "n_questions": len(questions),
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "accuracy": n_correct / len(labels) if labels else 0,
        "auroc_logtoku": auroc_logtoku,
        "auroc_pro": auroc_pro,
        "total_time_s": round(total_time, 1),
        "per_question": results,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
