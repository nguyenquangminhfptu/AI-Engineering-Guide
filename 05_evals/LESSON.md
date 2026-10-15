# Module 05 — Evals (Week 8)

**Goal:** turn "it seems better" into a number you trust, for prompts, skills, agents and fine-tuned models alike.

> Evals are the test suite of AI engineering. As a software engineer you already know this mindset — this module just adapts it to non-deterministic outputs.

---

## 1. The eval loop

```
define success → build eval set → choose graders → run → read failures → change ONE thing → re-run → compare
```
Without a fixed eval set, every change is a guess, and every demo is cherry-picked.

## 2. Building the eval set

- **Source**: real user inputs first (logs, tickets). Synthetic only to fill gaps.
- **Size**: 50 is enough to catch big changes; 200–500 to detect small ones. (With 100 cases, a 95% confidence interval on a 70% pass rate is about ±9 points — small differences are noise!)
- **Slices / tags**: topic, difficulty, language, "known hard". Report per slice: an average can hide a collapse in one slice.
- **Freeze it** and version it. Add new failure cases from production over time (regression suite).
- **Never train on it.** (Module 03 leakage check.)

## 3. Graders — pick the cheapest one that's reliable

| Grader | Use for | Example |
|---|---|---|
| **Exact / contains / regex** | classification, extraction, required keywords | label == "billing"; answer mentions "gradient_accumulation_steps" |
| **Schema / parse check** | JSON / tool-call outputs | `json.loads` + Pydantic validate |
| **Execution** | code, SQL, agents | run unit tests; check final file/DB state |
| **Reference similarity** | short factual answers | normalized match, F1 overlap |
| **LLM-as-judge (rubric)** | open-ended quality | "Does the answer give at least 2 concrete steps? yes/no" |
| **LLM-as-judge (pairwise)** | A vs B (base vs tuned, prompt v1 vs v2) | "Which answer is better for this question?" |
| **Human** | calibration, final sign-off | 50–100 examples |

## 4. LLM-as-judge done right

- **Specific rubrics** with binary or 1–5 anchored criteria. "Is it good?" is useless; "Does it mention the root cause? Does it give a command the user can run?" works.
- **Pairwise > absolute** when comparing two systems.
- **Randomize A/B position** — judges have position bias. Better: judge both orders, count ties when they disagree.
- **Give the judge the reference answer** when you have one.
- **Ask for reasoning before the verdict** (or use structured output with a `reason` field).
- **Calibrate**: label 50 examples yourself, measure judge agreement. Below ~80% → fix the rubric before trusting it.
- Use a strong judge; don't let a model judge only itself.

## 5. What to measure besides quality

- **Cost** per task (tokens × price), **latency** (p50/p95), **tool calls** per agent run, **failure modes** (refusals, format errors, timeouts).
- For agents: task success rate, steps to success, and % of runs needing human help.

## 6. Error analysis — where the improvements actually come from

1. Pull 20–50 failures.
2. Write a one-line note for each ("retrieved wrong doc", "ignored format", "hallucinated param").
3. Group into categories, count them.
4. Fix the biggest category (data? prompt? retrieval? tool description?), re-run, and check that nothing else regressed.

## 7. Evals in CI

Treat prompts, skills, tool descriptions, model versions and datasets like code: any change → run the eval suite → block merge on regression beyond a threshold. `eval_harness.py --baseline` shows exactly which cases flipped.

---

## Labs
```bash
python eval_harness.py selftest                                   # graders only, no API key needed
python eval_harness.py run eval_cases.jsonl                       # run Claude (system prompt v1) on the suite
python eval_harness.py run eval_cases.jsonl --system "…v2…" --baseline results/<previous>.jsonl
python eval_harness.py compare ../04_finetuning/out/compare.jsonl # pairwise judge: base vs fine-tuned
```

## Exercises
1. Run the suite. Read every failure. Is the grader wrong or the answer wrong?
2. Change the system prompt; re-run with `--baseline`. Which cases flipped? Is the net change bigger than the confidence interval?
3. Add 5 cases from a real task at work, with at least one deterministic check each.
4. Label 20 judge decisions yourself; compute agreement.
5. Run `compare` on your fine-tune output. Report win/tie/loss of tuned vs base.

## At work
Put the evals for your skill/agent/model into one harness that runs with one command, and add it to CI.

## Read
- Chip Huyen, *AI Engineering* — chapters 3–4 (evaluation)
- Hamel Husain — "Your AI product needs evals" and "Creating a LLM-as-a-judge that drives business results" (blog)
- Anthropic docs — "Define success criteria" and "Create strong empirical evaluations"
