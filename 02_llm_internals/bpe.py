"""Lab 02-A: Byte-Pair Encoding from scratch (the algorithm behind GPT / Llama / Qwen tokenizers).

Run:  python bpe.py [--merges 300] [--text some_file.txt]
"""
import argparse
from collections import Counter
from pathlib import Path


def pair_counts(ids):
    return Counter(zip(ids, ids[1:]))


def merge(ids, pair, new_id):
    """Replace every occurrence of `pair` in ids with new_id."""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPE:
    def train(self, text: str, num_merges: int):
        ids = list(text.encode("utf-8"))          # start: 256 byte tokens, works for ANY language
        self.merges = {}                          # (a, b) -> new token id
        self.vocab = {i: bytes([i]) for i in range(256)}
        for k in range(num_merges):
            counts = pair_counts(ids)
            if not counts:
                break
            pair, freq = counts.most_common(1)[0]  # most frequent adjacent pair
            if freq < 2:
                break
            new_id = 256 + k
            ids = merge(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
        return self

    def encode(self, text: str):
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            # apply the earliest-learned merge available (same order as training)
            pair = min(pair_counts(ids), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--merges", type=int, default=300)
    ap.add_argument("--text", help="training text file (default: this course's lesson files)")
    a = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    text = Path(a.text).read_text() if a.text else "\n".join(p.read_text() for p in sorted(root.rglob("LESSON.md")))
    tok = BPE().train(text, a.merges)

    print("First 15 merges learned:")
    for (x, y), nid in list(tok.merges.items())[:15]:
        print(f"  {tok.vocab[x]!r:>12} + {tok.vocab[y]!r:<12} -> {tok.vocab[nid]!r}")
    print("Longest tokens:", sorted((tok.vocab[i] for i in tok.merges.values()), key=len)[-8:])

    sample = "Fine-tuning a language model with LoRA adapters."
    ids = tok.encode(sample)
    assert tok.decode(ids) == sample
    print(f"\n{sample!r}\n  bytes={len(sample.encode())} tokens={len(ids)} compression={len(sample.encode()) / len(ids):.2f}x")
    print("  pieces:", [tok.decode([i]) for i in ids])
    vi = "Tinh chỉnh mô hình ngôn ngữ lớn."
    print(f"{vi!r}: bytes={len(vi.encode())} tokens={len(tok.encode(vi))}  ← trained on English, so Vietnamese compresses poorly")
