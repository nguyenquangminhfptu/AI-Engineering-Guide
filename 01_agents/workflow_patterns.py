"""Lab 01-C: workflows — when YOUR CODE decides the steps (cheaper and more predictable than an agent).

Run:  python workflow_patterns.py chain
      python workflow_patterns.py route "My fine-tuned model outputs garbage after epoch 3"
      python workflow_patterns.py refine "Explain LoRA to a backend engineer in 5 sentences"
"""
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.llm import ask, parse  # noqa: E402


# ---------- Pattern 1: prompt chaining (A → check → B) ----------
def chain(raw_notes: str) -> str:
    outline = ask(f"Turn these notes into a 3-bullet outline. Output only the bullets.\n\n{raw_notes}")
    if outline.count("\n") < 2:  # programmatic gate between steps — a key benefit of workflows
        raise ValueError(f"Outline step failed: {outline!r}")
    return ask(f"Write a short, clear paragraph from this outline:\n\n{outline}")


# ---------- Pattern 2: routing (classify → specialist) ----------
class Route(BaseModel):
    category: Literal["training_bug", "data_issue", "agent_issue", "other"]
    reason: str


SPECIALISTS = {
    "training_bug": "You are an ML training expert. Diagnose loss/optimizer/LR/precision problems step by step.",
    "data_issue": "You are a dataset engineer. Focus on data quality, leakage, formatting and chat templates.",
    "agent_issue": "You are an agent engineer. Focus on tool design, prompts, context and loop control.",
    "other": "You are a helpful AI engineering mentor.",
}


def route(question: str) -> str:
    # Structured output: the response is validated against the Pydantic schema.
    decision = parse(f"Classify this AI-engineering question:\n\n{question}", Route, fast=True)  # easy → cheap model
    print(f"[router] → {decision.category} ({decision.reason})")
    return ask(question, system=SPECIALISTS[decision.category])


# ---------- Pattern 3: evaluator–optimizer (generate → critique → revise) ----------
class Critique(BaseModel):
    passes: bool
    problems: list[str]


def refine(task: str, max_rounds: int = 3) -> str:
    draft = ask(task)
    for i in range(max_rounds):
        critique = parse(
            f"Task: {task}\n\nDraft:\n{draft}\n\n"
            "Judge strictly: is it accurate, does it follow every constraint in the task, is it clear? "
            "List concrete problems. passes=true only if there are none.",
            Critique,
        )
        print(f"[round {i + 1}] passes={critique.passes} problems={critique.problems}")
        if critique.passes:
            break
        draft = ask(f"Task: {task}\n\nPrevious draft:\n{draft}\n\nFix these problems:\n- " + "\n- ".join(critique.problems))
    return draft


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "chain"
    arg = " ".join(sys.argv[2:])
    if mode == "chain":
        print(chain(arg or "lora = low rank adapters; freeze base weights; train small A,B matrices; cheap; merge after"))
    elif mode == "route":
        print(route(arg or "My SFT loss drops to 0.01 but the model just repeats the system prompt. Why?"))
    elif mode == "refine":
        print(refine(arg or "Explain QLoRA to a backend engineer in exactly 4 sentences, no jargon without definition."))
