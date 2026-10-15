---
name: sft-dataset-checker
description: Validate and diagnose a JSONL chat dataset before LLM fine-tuning (SFT or DPO). Use when the user asks to check, validate, audit, clean or sanity-check a training/eval dataset, a .jsonl file with "messages", or asks why a fine-tune might be failing because of data. Not for general CSV analysis.
---

# SFT dataset checker

Goal: tell the user whether a dataset is safe to train on, and what to fix first.

## Steps

1. Run the deterministic checker first. It's faster and more reliable than reading the file yourself:
   ```bash
   python scripts/check_dataset.py <train.jsonl> [--eval <eval.jsonl>]
   ```
   It reports format errors, role-order errors, empty turns, exact duplicates, train/eval overlap (leakage) and length outliers.

2. Read 5–10 random rows (`python scripts/check_dataset.py <file> --sample 8`) and judge what the script can't:
   - Are assistant replies actually the behavior we want the model to learn? (tone, format, correctness)
   - Is there a systematic issue (all replies start the same way, boilerplate, leaked system text, wrong language)?
   - Is the task coverage balanced, or dominated by one kind of request?

3. Report in this order:
   - **Verdict**: ready / fix first / do not train.
   - **Blocking issues** (format errors, leakage, >5% duplicates) with counts and example line numbers.
   - **Quality issues** from manual review, each with one quoted example.
   - **Next actions**, most impactful first.

## Why these checks matter
- Train/eval overlap makes eval scores meaningless, so it's always blocking.
- Duplicates make the model over-weight a few examples and memorize them.
- Wrong role order or empty assistant turns silently break chat-template formatting and loss masking.
- Very long rows get truncated at `max_length`, which can cut off the answer the model is supposed to learn.

Don't modify the user's files. Suggest fixes, or write a cleaned copy only if asked.
