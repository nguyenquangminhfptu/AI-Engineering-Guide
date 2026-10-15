"""Lab 03-A: a real SFT dataset pipeline.

raw → normalize → messages → validate → filter → PII mask → exact dedup → near-dup dedup → group split → card

Run:  python build_sft_dataset.py                                   # uses sample_raw.jsonl
      python build_sft_dataset.py --input my_export.jsonl --q question --a answer --group topic
Input: JSONL (or CSV) with one question/answer per row. Adapt `to_messages()` for multi-turn data.
"""
import argparse
import csv
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
SYSTEM_PROMPT = "You are the internal ML platform helpdesk. Give practical, step-by-step answers."

# ---------------- 1. load & normalize ----------------
def load(path):
    if path.endswith(".csv"):
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")   # unify Vietnamese accent encodings
    text = re.sub(r"<[^>]+>", " ", text)              # strip HTML tags
    text = re.sub(r"[ \t]+", " ", text)               # collapse spaces, keep newlines
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ---------------- 2. convert ----------------
def to_messages(q: str, a: str) -> dict:
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": q},
        {"role": "assistant", "content": a},
    ]}


# ---------------- 3. filters: each returns a reason string to DROP, or None to keep ----------------
BOILERPLATE = re.compile(r"\b(as an ai( language model)?|i cannot access|i'm sorry, but i can't|lorem ipsum)\b", re.I)


def filter_reason(q: str, a: str, min_q=8, min_a=40, max_chars=8000):
    if not q or not a:
        return "empty_field"
    if len(q) < min_q:
        return "question_too_short"
    if len(a) < min_a:
        return "answer_too_short"
    if len(q) + len(a) > max_chars:
        return "too_long"
    if BOILERPLATE.search(a):
        return "boilerplate_refusal"
    if "�" in q + a:
        return "broken_encoding"
    return None


# ---------------- 4. PII masking (extend with your company's ID / key formats) ----------------
PII = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
    (re.compile(r"(?<!\d)(?:\+?84|0)(?:[\s.-]?\d){8,10}(?!\d)"), "[PHONE]"),  # Vietnamese phone numbers
    (re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"), "[API_KEY]"),
]


def mask_pii(text: str):
    hits = 0
    for pattern, repl in PII:
        text, n = pattern.subn(repl, text)
        hits += n
    return text, hits


# ---------------- 5. dedup ----------------
def canon(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", "", text.lower()).split())


def shingles(text: str, n=5):
    t = " ".join(re.sub(r"[^\w\s]", "", text.lower()).split())
    return {t[i:i + n] for i in range(max(1, len(t) - n + 1))}


def minhash(sh, num_perm=64):
    """MinHash signature: P(sig_a[i] == sig_b[i]) == Jaccard(a, b). Lets you compare millions of docs cheaply."""
    return [min(int.from_bytes(hashlib.blake2b(f"{seed}:{s}".encode(), digest_size=8).digest(), "big") for s in sh)
            for seed in range(num_perm)]


def near_dup_clusters(texts, threshold=0.7, num_perm=64, bands=16):
    """LSH: rows sharing any identical band of their signature become candidates; confirm with estimated Jaccard."""
    sigs = [minhash(shingles(t), num_perm) for t in texts]
    rows_per_band = num_perm // bands
    buckets = defaultdict(list)
    for idx, sig in enumerate(sigs):
        for b in range(bands):
            buckets[(b, tuple(sig[b * rows_per_band:(b + 1) * rows_per_band]))].append(idx)
    parent = list(range(len(texts)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for members in buckets.values():
        for i in members[1:]:
            j = members[0]
            sim = sum(a == b for a, b in zip(sigs[i], sigs[j])) / num_perm
            if sim >= threshold:
                parent[find(i)] = find(j)
    return [find(i) for i in range(len(texts))]


# ---------------- 6. split by group (prevents leakage between near-identical items) ----------------
def group_split(items, group_key, eval_frac, seed):
    groups = sorted({it[group_key] for it in items})
    if len(groups) < 4:   # too few groups (e.g. topics) to split by group → stable hash split per row
        return ([it for it in items if int(hashlib.md5(it["id"].encode()).hexdigest(), 16) % 100 >= eval_frac * 100],
                [it for it in items if int(hashlib.md5(it["id"].encode()).hexdigest(), 16) % 100 < eval_frac * 100])
    random.Random(seed).shuffle(groups)
    eval_groups = set(groups[: max(1, round(len(groups) * eval_frac))])
    return [it for it in items if it[group_key] not in eval_groups], [it for it in items if it[group_key] in eval_groups]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(HERE / "sample_raw.jsonl"))
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--q", default="question")
    ap.add_argument("--a", default="answer")
    ap.add_argument("--group", default="cluster", help="column to split on; 'cluster' = near-dup cluster id")
    ap.add_argument("--eval_frac", type=float, default=0.2)
    ap.add_argument("--near_dup", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    raw = load(a.input)
    dropped = Counter()
    kept, pii_hits = [], 0
    for i, row in enumerate(raw):
        q, ans = normalize(row.get(a.q, "")), normalize(row.get(a.a, ""))
        reason = filter_reason(q, ans)
        if reason:
            dropped[reason] += 1
            print(f"  drop [{reason}] {row.get('id', i)}: {q[:60]!r}")
            continue
        q, h1 = mask_pii(q)
        ans, h2 = mask_pii(ans)
        pii_hits += h1 + h2
        kept.append({**row, "id": str(row.get("id", i)), "q": q, "a": ans})

    # exact dedup on normalized prompt+answer
    seen, unique = set(), []
    for it in kept:
        k = hashlib.md5(canon(it["q"] + " || " + it["a"]).encode()).hexdigest()
        if k in seen:
            dropped["exact_duplicate"] += 1
            print(f"  drop [exact_duplicate] {it['id']}: {it['q'][:60]!r}")
            continue
        seen.add(k)
        unique.append(it)

    # near-dup dedup: keep the first member of each cluster
    clusters = near_dup_clusters([it["q"] + " " + it["a"] for it in unique], threshold=a.near_dup)
    final, first_of = [], {}
    for it, c in zip(unique, clusters):
        if c in first_of:
            dropped["near_duplicate"] += 1
            print(f"  drop [near_duplicate of {first_of[c]}] {it['id']}: {it['q'][:60]!r}")
            continue
        first_of[c] = it["id"]
        final.append({**it, "cluster": str(c)})

    train, evl = group_split(final, a.group, a.eval_frac, a.seed)
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    for name, rows in (("train", train), ("eval", evl)):
        with open(out / f"{name}.jsonl", "w") as f:
            for it in rows:
                rec = to_messages(it["q"], it["a"])
                rec["meta"] = {"id": it["id"], "topic": it.get("topic"), "source": it.get("source")}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    topics = Counter(it.get("topic") for it in final)
    card = f"""# Dataset card — helpdesk SFT v1

- **Built**: {date.today()} by `build_sft_dataset.py`
- **Source**: `{Path(a.input).name}` ({len(raw)} raw rows)
- **Purpose**: teach the model to answer ML-platform helpdesk questions in a practical, step-by-step style
- **Format**: chat `messages` (system / user / assistant), JSONL
- **Splits**: train {len(train)} · eval {len(evl)} (split key: `{a.group}`, eval fraction {a.eval_frac})
- **Topics**: {dict(topics)}

## Processing
Unicode NFC → HTML strip → filters → PII masking ({pii_hits} replacements) → exact dedup → MinHash near-dup (threshold {a.near_dup}).

| Drop reason | Rows |
|---|---|
""" + "\n".join(f"| {k} | {v} |" for k, v in dropped.most_common()) + f"""
| **kept** | **{len(final)}** |

## Known gaps
- Small sample; topic coverage is uneven. Single-turn only.
- License / consent: internal helpdesk export — confirm approval before external use.
"""
    (out / "DATASET_CARD.md").write_text(card)
    print(f"\nraw={len(raw)} kept={len(final)} train={len(train)} eval={len(evl)} pii_masked={pii_hits}")
    print("dropped:", dict(dropped))
    print(f"wrote {out}/train.jsonl, eval.jsonl, DATASET_CARD.md")


if __name__ == "__main__":
    main()
