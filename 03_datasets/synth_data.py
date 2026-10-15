"""Lab 03-B: synthetic data with a strong model — generate → judge → filter → dedup.

Run:  python synth_data.py --n 20              # SFT examples → out/synthetic.jsonl
      python synth_data.py --dpo --n 10        # preference pairs → out/dpo.jsonl
Needs GEMINI_API_KEY.

Diversity is forced with a grid of (topic × persona × difficulty) instead of asking "generate 100 examples".
Note: check your model provider's terms before using generated outputs to train other models.
"""
import argparse
import itertools
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.llm import parse  # noqa: E402
from build_sft_dataset import SYSTEM_PROMPT, near_dup_clusters  # noqa: E402

HERE = Path(__file__).resolve().parent
TOPICS = ["LoRA / QLoRA training", "GPU out-of-memory", "dataset quality and leakage", "chat templates and tokenizers",
          "evaluating fine-tuned models", "serving with vLLM", "agent tool design", "RAG retrieval quality"]
PERSONAS = ["junior backend engineer new to ML", "data scientist moving to LLMs", "impatient team lead who wants a yes/no",
            "Vietnamese-speaking engineer who writes in Vietnamese"]
DIFFICULTY = ["basic", "intermediate", "tricky edge case"]


class Example(BaseModel):
    question: str = Field(description="What the persona would actually type: realistic, specific, sometimes messy")
    answer: str = Field(description="Ideal helpdesk answer: correct, practical, step-by-step when useful, same language as the question")


class Judgement(BaseModel):
    correctness: int = Field(ge=1, le=5)
    helpfulness: int = Field(ge=1, le=5)
    realism: int = Field(ge=1, le=5, description="Does the question sound like a real user?")
    issue: str = Field(description="Main problem, or 'none'")


class Pair(BaseModel):
    chosen: str = Field(description="Excellent answer")
    rejected: str = Field(description="Plausible but worse answer: vague, subtly wrong, or ignores a constraint")


def generate(topic, persona, difficulty) -> Example:
    prompt = (f"Write ONE realistic question an internal ML-platform helpdesk receives, and the ideal answer.\n"
              f"Topic: {topic}\nAsker: {persona}\nDifficulty: {difficulty}\n"
              f"The answer is from this assistant: \"{SYSTEM_PROMPT}\". Keep the answer under 180 words.")
    return parse(prompt, Example)


def judge(ex: Example) -> Judgement:
    prompt = (f"Grade this helpdesk training example strictly (1=bad, 5=excellent).\n\n"
              f"QUESTION:\n{ex.question}\n\nANSWER:\n{ex.answer}")
    return parse(prompt, Judgement, fast=True)   # judging is high-volume → cheap model


def make_pair(question: str) -> Pair:
    prompt = (f"For this helpdesk question, write a 'chosen' answer that is excellent and a 'rejected' answer that "
              f"looks reasonable but is clearly worse (vague, partially wrong, or unhelpful). Question:\n{question}")
    return parse(prompt, Pair)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--min_score", type=int, default=4)
    ap.add_argument("--dpo", action="store_true")
    a = ap.parse_args()
    out = HERE / "out"
    out.mkdir(exist_ok=True)

    grid = list(itertools.product(TOPICS, PERSONAS, DIFFICULTY))
    random.Random(0).shuffle(grid)
    cells = grid[: a.n]

    with ThreadPoolExecutor(max_workers=8) as pool:  # parallel requests; the SDK retries 429/5xx
        examples = list(pool.map(lambda c: generate(*c), cells))

        if a.dpo:
            pairs = list(pool.map(lambda ex: make_pair(ex.question), examples))
            with open(out / "dpo.jsonl", "w") as f:
                for ex, p in zip(examples, pairs):
                    f.write(json.dumps({
                        "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ex.question}],
                        "chosen": [{"role": "assistant", "content": p.chosen}],
                        "rejected": [{"role": "assistant", "content": p.rejected}],
                    }, ensure_ascii=False) + "\n")
            print(f"wrote {len(pairs)} pairs → {out / 'dpo.jsonl'}")
            print("Better pairs in practice: sample 2–4 answers from YOUR SFT model and let a judge pick best/worst.")
            return

        grades = list(pool.map(judge, examples))

    keep = []
    for (topic, persona, diff), ex, g in zip(cells, examples, grades):
        score = min(g.correctness, g.helpfulness, g.realism)
        flag = "KEEP" if score >= a.min_score else "drop"
        print(f"[{flag}] c={g.correctness} h={g.helpfulness} r={g.realism} | {topic} / {diff} | {ex.question[:70]!r}"
              + (f"\n        issue: {g.issue}" if flag == "drop" else ""))
        if score >= a.min_score:
            keep.append((topic, persona, diff, ex))

    clusters = near_dup_clusters([ex.question + " " + ex.answer for *_, ex in keep], threshold=0.6)
    seen = set()
    with open(out / "synthetic.jsonl", "w") as f:
        for (topic, persona, diff, ex), c in zip(keep, clusters):
            if c in seen:
                continue
            seen.add(c)
            f.write(json.dumps({"messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ex.question},
                {"role": "assistant", "content": ex.answer},
            ], "meta": {"topic": topic, "persona": persona, "difficulty": diff, "synthetic": True}}, ensure_ascii=False) + "\n")
    print(f"\ngenerated={len(examples)} passed_judge={len(keep)} after_dedup={len(seen)} → {out / 'synthetic.jsonl'}")
    print("Now READ 10 of them yourself. If >10% are bad, fix the generate() prompt.")


if __name__ == "__main__":
    main()
