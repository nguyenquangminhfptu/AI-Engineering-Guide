"""Deterministic checks for a JSONL chat dataset: {"messages": [{"role": ..., "content": ...}, ...]} per line.

Usage: python check_dataset.py train.jsonl [--eval eval.jsonl] [--sample N] [--max-chars 12000]
"""
import argparse
import hashlib
import json
import random
import statistics
import sys

VALID_ROLES = {"system", "user", "assistant"}


def load(path):
    rows, errors = [], []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append((i, json.loads(line)))
            except json.JSONDecodeError as e:
                errors.append(f"line {i}: invalid JSON ({e.msg})")
    return rows, errors


def structure_errors(lineno, row):
    msgs = row.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return [f"line {lineno}: missing or empty 'messages' list"]
    errs = []
    turns = [m for m in msgs if m.get("role") != "system"]
    if msgs[0].get("role") not in ("system", "user"):
        errs.append(f"line {lineno}: first message must be system or user")
    for j, m in enumerate(msgs):
        if m.get("role") not in VALID_ROLES:
            errs.append(f"line {lineno}: message {j} has invalid role {m.get('role')!r}")
        if not str(m.get("content", "")).strip():
            errs.append(f"line {lineno}: message {j} ({m.get('role')}) is empty")
        if m.get("role") == "system" and j != 0:
            errs.append(f"line {lineno}: system message not at position 0")
    for a, b in zip(turns, turns[1:]):
        if a.get("role") == b.get("role"):
            errs.append(f"line {lineno}: two consecutive '{a.get('role')}' turns")
            break
    if turns and turns[-1].get("role") != "assistant":
        errs.append(f"line {lineno}: last turn is not assistant (nothing to learn)")
    return errs


def key(row):
    """Hash of the user-visible prompt side — used for duplicate and leakage detection."""
    prompt = " ".join(m.get("content", "") for m in row.get("messages", []) if m.get("role") == "user")
    return hashlib.md5(" ".join(prompt.lower().split()).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train")
    ap.add_argument("--eval")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--max-chars", type=int, default=12000)
    a = ap.parse_args()

    rows, errors = load(a.train)
    if a.sample:
        for lineno, r in random.sample(rows, min(a.sample, len(rows))):
            print(f"--- line {lineno}\n{json.dumps(r, ensure_ascii=False, indent=1)}")
        return

    for lineno, r in rows:
        errors += structure_errors(lineno, r)

    seen, dups = {}, []
    for lineno, r in rows:
        k = key(r)
        if k in seen:
            dups.append((seen[k], lineno))
        else:
            seen[k] = lineno

    lengths = [(lineno, sum(len(m.get("content", "")) for m in r.get("messages", []))) for lineno, r in rows]
    too_long = [ln for ln, n in lengths if n > a.max_chars]

    leaks = []
    if a.eval:
        eval_rows, eval_err = load(a.eval)
        errors += [f"[eval] {e}" for e in eval_err]
        leaks = [ln for ln, r in eval_rows if key(r) in seen]

    n = len(rows)
    print(f"rows: {n}")
    if lengths:
        print(f"chars per example: median={statistics.median(n for _, n in lengths):.0f} max={max(n for _, n in lengths)}")
    print(f"format/structure errors: {len(errors)}")
    for e in errors[:15]:
        print("  ", e)
    print(f"exact duplicate prompts: {len(dups)} ({len(dups) / max(n, 1):.1%})" + (f"  e.g. lines {dups[:5]}" if dups else ""))
    print(f"examples over {a.max_chars} chars: {len(too_long)}" + (f"  e.g. lines {too_long[:5]}" if too_long else ""))
    if a.eval:
        print(f"train/eval LEAKAGE (eval prompts also in train): {len(leaks)}" + (f"  eval lines {leaks[:5]}" if leaks else ""))

    blocking = errors or leaks or len(dups) / max(n, 1) > 0.05
    print("\nVERDICT:", "FIX FIRST" if blocking else "no blocking issues found — now review samples manually")
    sys.exit(1 if blocking else 0)


if __name__ == "__main__":
    main()
