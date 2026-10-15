"""Lab 05: a small but real eval harness.

    python eval_harness.py selftest
    python eval_harness.py run eval_cases.jsonl [--system "..."] [--baseline results/x.jsonl]
    python eval_harness.py compare ../04_finetuning/out/compare.jsonl

Case format (JSONL):
    {"id": "...", "input": "...", "tags": ["topic:ft"], "reference": "...optional...",
     "checks": [{"type": "contains", "value": "..."}, {"type": "regex", "value": "..."},
                {"type": "not_contains", "value": "..."}, {"type": "max_words", "value": 150},
                {"type": "json"}, {"type": "rubric", "value": "Gives at least two concrete steps"}]}
A case passes only if ALL its checks pass.
To evaluate YOUR system (agent, RAG, fine-tuned model), replace `predict()`.
"""
import argparse
import json
import math
import random
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HERE = Path(__file__).resolve().parent
DEFAULT_SYSTEM = "You are the internal ML platform helpdesk. Give practical, step-by-step answers."


# ---------------- system under test ----------------
def predict(case: dict, system: str) -> str:
    from common.llm import ask
    return ask(case["input"], system=system)


# ---------------- graders ----------------
class Verdict(BaseModel):
    reason: str
    passed: bool


def judge_rubric(output: str, criterion: str, case: dict) -> tuple[bool, str]:
    from common.llm import parse
    ref = f"\nREFERENCE ANSWER (for facts):\n{case['reference']}\n" if case.get("reference") else ""
    prompt = (f"You are grading one criterion of an AI answer. Be strict and literal.\n\n"
              f"QUESTION:\n{case['input']}\n{ref}\nANSWER TO GRADE:\n{output}\n\n"
              f"CRITERION: {criterion}\n\nExplain briefly, then decide if the criterion is met.")
    v = parse(prompt, Verdict)
    return v.passed, v.reason


def run_check(check: dict, output: str, case: dict) -> tuple[bool, str]:
    t, v = check["type"], check.get("value")
    low = output.lower()
    if t == "contains":
        return str(v).lower() in low, f"contains {v!r}"
    if t == "not_contains":
        return str(v).lower() not in low, f"not_contains {v!r}"
    if t == "regex":
        return re.search(v, output, re.I | re.S) is not None, f"regex {v!r}"
    if t == "max_words":
        n = len(output.split())
        return n <= int(v), f"{n} words (max {v})"
    if t == "json":
        try:
            json.loads(output)
            return True, "valid json"
        except json.JSONDecodeError as e:
            return False, f"invalid json: {e}"
    if t == "rubric":
        return judge_rubric(output, v, case)
    raise ValueError(f"unknown check type {t}")


def wilson(passed: int, n: int, z: float = 1.96):
    """95% confidence interval for a pass rate. Small eval sets → wide intervals → don't over-read small changes."""
    if n == 0:
        return 0.0, 0.0
    p = passed / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


# ---------------- run mode ----------------
def evaluate_case(case, system):
    t0 = time.time()
    try:
        output = predict(case, system)
        error = None
    except Exception as e:  # a crash is a failed case, not a crashed eval run
        output, error = "", f"{type(e).__name__}: {e}"
    latency = time.time() - t0
    results = [] if error else [(c, *run_check(c, output, case)) for c in case.get("checks", [])]
    passed = error is None and all(ok for _, ok, _ in results)
    return {"id": case["id"], "tags": case.get("tags", []), "passed": passed, "latency_s": round(latency, 2),
            "error": error, "output": output,
            "checks": [{"type": c["type"], "value": c.get("value"), "ok": ok, "detail": d} for c, ok, d in results]}


def cmd_run(a):
    cases = [json.loads(l) for l in open(a.cases) if l.strip()]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda c: evaluate_case(c, a.system), cases))

    n, k = len(results), sum(r["passed"] for r in results)
    lo, hi = wilson(k, n)
    print(f"\nPASS {k}/{n} = {k / n:.0%}   (95% CI {lo:.0%}–{hi:.0%})   "
          f"p50 latency {sorted(r['latency_s'] for r in results)[n // 2]}s")
    by_tag = defaultdict(lambda: [0, 0])
    for r in results:
        for t in r["tags"]:
            by_tag[t][0] += r["passed"]
            by_tag[t][1] += 1
    for t, (p, m) in sorted(by_tag.items()):
        print(f"  {t:<22} {p}/{m}")
    print("\nFailures:")
    for r in results:
        if not r["passed"]:
            why = r["error"] or "; ".join(f"{c['type']}: {c['detail']}" for c in r["checks"] if not c["ok"])
            print(f"  ✗ {r['id']}: {why[:160]}")

    if a.baseline:
        base = {r["id"]: r["passed"] for r in map(json.loads, open(a.baseline))}
        fixed = [r["id"] for r in results if r["passed"] and base.get(r["id"]) is False]
        broke = [r["id"] for r in results if not r["passed"] and base.get(r["id"]) is True]
        print(f"\nvs baseline: fixed {fixed}  REGRESSED {broke}")

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    path = out / f"run_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nsaved {path}  (pass it as --baseline next time)")


# ---------------- compare mode: pairwise judge, both orders ----------------
class Pairwise(BaseModel):
    reason: str
    winner: Literal["A", "B", "tie"]


def pairwise(question, reference, a_text, b_text):
    from common.llm import parse
    prompt = (f"Which answer better serves the user? Judge correctness first (use the reference for facts), "
              f"then helpfulness and clarity. Ignore length unless it hurts clarity.\n\n"
              f"QUESTION:\n{question}\n\nREFERENCE:\n{reference}\n\nANSWER A:\n{a_text}\n\nANSWER B:\n{b_text}")
    return parse(prompt, Pairwise).winner


def cmd_compare(a):
    rows = [json.loads(l) for l in open(a.file) if l.strip()]

    def score(r):
        # judge both orders to cancel position bias; disagreement → tie
        first = pairwise(r["question"], r["reference"], r["base"], r["tuned"])
        second = pairwise(r["question"], r["reference"], r["tuned"], r["base"])
        v1 = {"A": "base", "B": "tuned", "tie": "tie"}[first]
        v2 = {"A": "tuned", "B": "base", "tie": "tie"}[second]
        return v1 if v1 == v2 else "tie"

    with ThreadPoolExecutor(max_workers=8) as pool:
        verdicts = list(pool.map(score, rows))
    n = len(verdicts)
    w, t, l = verdicts.count("tuned"), verdicts.count("tie"), verdicts.count("base")
    print(f"tuned vs base over {n} eval questions:  WIN {w}  TIE {t}  LOSS {l}   win-rate (ties=½) {(w + t / 2) / n:.0%}")
    for r, v in zip(rows, verdicts):
        if v == "base":
            print(f"  base better on: {r['question'][:90]!r}")


# ---------------- selftest: graders only, offline ----------------
def cmd_selftest(_):
    case = {"input": "q"}
    tests = [
        ({"type": "contains", "value": "LoRA"}, "Use lora with r=16", True),
        ({"type": "not_contains", "value": "as an ai"}, "As an AI, I can't", False),
        ({"type": "regex", "value": r"\b1e-4|2e-4\b"}, "try lr 2e-4", True),
        ({"type": "max_words", "value": 3}, "one two three four", False),
        ({"type": "json"}, '{"label": "billing"}', True),
        ({"type": "json"}, "label: billing", False),
    ]
    for check, output, expected in tests:
        ok, detail = run_check(check, output, case)
        print(f"{'✓' if ok == expected else '✗ WRONG'} {check['type']:<13} → {ok} ({detail})")
    print("wilson(7,10) =", tuple(round(x, 2) for x in wilson(7, 10)), "← 70% on 10 cases could really be 40%–89%")
    random.seed(0)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("cases")
    r.add_argument("--system", default=DEFAULT_SYSTEM)
    r.add_argument("--baseline")
    c = sub.add_parser("compare")
    c.add_argument("file")
    sub.add_parser("selftest")
    a = ap.parse_args()
    {"run": cmd_run, "compare": cmd_compare, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
