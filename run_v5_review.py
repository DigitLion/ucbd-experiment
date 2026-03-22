#!/usr/bin/env python3
"""
AI Review of UCBD paper v5.3 (post-B4 validation)
Run 6 reviews with different personas to get aggregated scores.
"""

import json, time
import requests

OMLX_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "Qwen3-14B-4bit"

PAPER_SUMMARY = """
UCBD (Unified Cognitive Boundary Detection) is a framework paper proposing 5 cognitive boundaries for AI agents:
B1 Fluency (token entropy), B2 Density (embedding density), B3 Freshness (temporal decay),
B4 Association Rupture (KG completion), B5 Grounding (external verification).
Key architecture: cheapest-first cascade with a Pointer Model dispatcher.

10 EXPERIMENTS (all completed):
1. B1 domain specificity: TruthfulQA 790q, AUC=0.670 effective / 0.382 blind
2. B1-B2 independence: r=0.018, joint 64% coverage
3. Cascade ≥ parallel: p=0.498, 57.4% cost saving
4. Cross-model (3 models): 3B AUC=0.676 > 14B=0.537 (RLHF suppresses signals)
5. B3 freshness: FreshQA 1500q, 11-13x decay, cross-model consistent
6. Label robustness: LLM-judge cross-family κ=0.619, B1 AUC↑ to 0.632
7. B5 grounding: NLI AUC=0.678 in B1 blind zone, B1-B5 r=0.070
8. Pointer Model: Oracle AUC=0.992, LogReg CV AUC=0.591, 97.8% cost saving
9. B1 as RAG trigger: HotpotQA 100q, AUC=0.485 (fails), entropy inversion
10. B4 proxy validation: Entity-embedding distance, AUC=0.540 in B1+B2 blind zone,
    B1-B4 r=0.034, B2-B4 r=0.000, +67% coverage expansion
    Key: Stereotypes(0.823), Superstitions(0.764), Education(0.667) in blind zone

BASELINES: B1 entropy (free) ≥ SelfCheck (5x cost, AUC=0.626) >> P(True) (AUC=0.427)
INDEPENDENCE TABLE: B1-B2 r=0.018, B1-B3 r=-0.067, B1-B4 r=0.034, B1-B5 r=0.070, B2-B4 r=0.000
2 PROPOSITIONS: Cascade cost efficiency (Prop 1), Superadditive coverage under independence (Prop 2)
MODELS: Qwen3-14B, Qwen3-4B, LLaMA-3.2-3B (cross-family validation)
DATASETS: TruthfulQA, FreshQA, HotpotQA
LIMITATIONS: 9 explicitly stated (B4 proxy not full KG, statistical power, scale, etc.)
PAPER: 39 pages, 0 undefined references, all 5 boundaries now have empirical evidence
"""

REVIEW_TEMPLATE = """You are {persona}. Review the following AI research paper.

{paper_summary}

Score each dimension 1-10 and give a brief justification. Output EXACTLY this JSON format:
{{
  "soundness": {{"score": X, "comment": "..."}},
  "novelty": {{"score": X, "comment": "..."}},
  "clarity": {{"score": X, "comment": "..."}},
  "significance": {{"score": X, "comment": "..."}},
  "reproducibility": {{"score": X, "comment": "..."}},
  "completeness": {{"score": X, "comment": "..."}},
  "technical_depth": {{"score": X, "comment": "..."}},
  "overall": {{"score": X, "decision": "Accept/Borderline/Reject", "comment": "..."}}
}}

Be rigorous. Focus on whether the claims are supported by evidence. Consider both strengths and weaknesses."""

PERSONAS = [
    "a senior ML reviewer at NeurIPS who specializes in uncertainty estimation and calibration. You value rigorous statistical analysis and strong baselines.",
    "a pragmatic industry AI researcher who values practical impact and reproducibility. You care about whether the framework can be deployed in production systems.",
    "a theoretical computer science reviewer who values formal proofs and mathematical rigor. You scrutinize whether propositions are properly proved.",
    "a cognitive science researcher who evaluates AI papers' connections to neuroscience and human cognition. You value interdisciplinary grounding.",
    "an Area Chair at ICML who must make a final accept/reject decision. You weigh all dimensions holistically and consider the paper's contribution to the field.",
    "a strict reviewer who is skeptical of framework papers without end-to-end system evaluations. You want to see the full pipeline working, not just individual components tested in isolation.",
]


def call_llm(prompt: str) -> str:
    resp = requests.post(OMLX_URL, json={
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 1500,
        "extra_body": {"enable_thinking": False}
    }, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def extract_json(text: str) -> dict:
    """Extract JSON from LLM response."""
    # Try to find JSON block
    import re
    # Remove thinking tags if present
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Find JSON
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def main():
    print("="*60)
    print("UCBD v5.3 AI Review (Post-B4 Validation)")
    print("="*60)

    all_reviews = []
    dimensions = ["soundness", "novelty", "clarity", "significance",
                  "reproducibility", "completeness", "technical_depth"]

    for i, persona in enumerate(PERSONAS):
        print(f"\nReview {i+1}/6: {persona[:60]}...")
        prompt = REVIEW_TEMPLATE.format(persona=persona, paper_summary=PAPER_SUMMARY)

        t0 = time.time()
        response = call_llm(prompt)
        elapsed = time.time() - t0
        print(f"  Response in {elapsed:.1f}s")

        review = extract_json(response)
        if not review:
            print(f"  WARNING: Failed to parse JSON, raw response:")
            print(f"  {response[:200]}")
            continue

        # Print scores
        scores = {}
        for dim in dimensions:
            if dim in review and isinstance(review[dim], dict):
                s = review[dim].get("score", "?")
                scores[dim] = s
                print(f"  {dim:20s}: {s}")
            else:
                print(f"  {dim:20s}: MISSING")

        if "overall" in review and isinstance(review["overall"], dict):
            overall = review["overall"]
            print(f"  {'overall':20s}: {overall.get('score', '?')} ({overall.get('decision', '?')})")
            scores["overall"] = overall.get("score", 0)
            scores["decision"] = overall.get("decision", "?")

        all_reviews.append({"persona": persona[:60], "scores": scores, "raw": review})

    # Aggregate
    print(f"\n{'='*60}")
    print("AGGREGATE SCORES")
    print(f"{'='*60}")

    for dim in dimensions:
        vals = [r["scores"].get(dim, 0) for r in all_reviews if isinstance(r["scores"].get(dim), (int, float))]
        if vals:
            import statistics
            mean = statistics.mean(vals)
            median = statistics.median(vals)
            print(f"  {dim:20s}: mean={mean:.1f}  median={median:.1f}  range=[{min(vals)}, {max(vals)}]")

    overall_vals = [r["scores"].get("overall", 0) for r in all_reviews if isinstance(r["scores"].get("overall"), (int, float))]
    if overall_vals:
        import statistics
        print(f"\n  OVERALL: mean={statistics.mean(overall_vals):.1f}  median={statistics.median(overall_vals):.1f}")

    decisions = [r["scores"].get("decision", "?") for r in all_reviews]
    print(f"  DECISIONS: {decisions}")
    accept_count = sum(1 for d in decisions if "Accept" in str(d) and "Reject" not in str(d) and "Borderline" not in str(d))
    print(f"  Accept: {accept_count}/{len(decisions)}")

    # Save
    with open("/Users/john/workspace/FARS/ucbd-experiment/results/v5_3_review.json", "w") as f:
        json.dump(all_reviews, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to results/v5_3_review.json")


if __name__ == "__main__":
    main()
