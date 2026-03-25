#!/usr/bin/env python3
"""Download 200 TriviaQA questions for Exp 22"""
import json
from datasets import load_dataset

# Load TriviaQA unfiltered validation set
ds = load_dataset("trivia_qa", "unfiltered", split="validation", trust_remote_code=True)

# Take first 200 questions (systematic, not cherry-picked)
questions = []
for i, item in enumerate(ds):
    if i >= 200:
        break
    questions.append({
        "qi": i,
        "question": item["question"],
        "answer": item["answer"]["value"],  # canonical answer
        "aliases": item["answer"].get("aliases", [])
    })

outpath = "/Users/john/Downloads/files/triviaqa_200q.json"
with open(outpath, "w") as f:
    json.dump(questions, f, indent=2)
print(f"Saved {len(questions)} TriviaQA questions to {outpath}")
print(f"Sample: {questions[0]['question']} -> {questions[0]['answer']}")
