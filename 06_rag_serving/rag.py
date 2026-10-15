"""Lab 06: RAG with measured retrieval — BM25 vs dense embeddings vs hybrid (RRF), then a cited answer.

Run:  python rag.py                                   # retrieval eval only (downloads a ~90MB embedding model once)
      python rag.py --ask "How do I choose LoRA rank?"  # + Gemini answer with citations (needs GEMINI_API_KEY)
Corpus = this course's LESSON.md files. Swap `load_chunks()` for your company docs.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Labeled retrieval eval: question → module folder that contains the answer.
EVAL = [
    ("How does the model know which earlier tokens to pay attention to?", "02_llm_internals"),
    ("What do Q, K and V mean?", "02_llm_internals"),
    ("Why does my model never stop generating after fine-tuning?", "04_finetuning"),
    ("How much GPU memory does QLoRA need for an 8B model?", "04_finetuning"),
    ("What value of lora_alpha should I use?", "04_finetuning"),
    ("How do I remove near-duplicate examples from a dataset?", "03_datasets"),
    ("Why normalize Vietnamese text to NFC?", "03_datasets"),
    ("How can I make synthetic data more diverse?", "03_datasets"),
    ("What is progressive disclosure in Claude skills?", "01_agents"),
    ("How do I protect an agent from prompt injection?", "01_agents"),
    ("When should I build an agent instead of a workflow?", "01_agents"),
    ("How do I reduce position bias in an LLM judge?", "05_evals"),
    ("How many eval cases do I need to detect a small improvement?", "05_evals"),
    ("What is continuous batching?", "06_rag_serving"),
    ("Should I merge my LoRA adapter before serving?", "06_rag_serving"),
]


def load_chunks():
    """Chunk by markdown heading (structure-aware), keep the source + section as metadata."""
    chunks = []
    for path in sorted(ROOT.glob("0*/LESSON.md")):
        section, buf = "intro", []
        for line in path.read_text().splitlines() + ["## END"]:
            if line.startswith("#"):
                text = "\n".join(buf).strip()
                if len(text) > 80:
                    chunks.append({"id": len(chunks), "module": path.parent.name, "section": section,
                                   "text": f"{section}\n{text}"})
                section, buf = line.lstrip("# ").strip(), []
            else:
                buf.append(line)
    return chunks


def tokenize(s):
    return re.findall(r"\w+", s.lower())


class Retriever:
    def __init__(self, chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
        self.encoder = SentenceTransformer(model_name)
        self.emb = self.encoder.encode([c["text"] for c in chunks], normalize_embeddings=True)

    def bm25_rank(self, q):
        return list(np.argsort(-self.bm25.get_scores(tokenize(q))))

    def dense_rank(self, q):
        qv = self.encoder.encode([q], normalize_embeddings=True)[0]
        return list(np.argsort(-(self.emb @ qv)))   # cosine similarity (vectors are normalized)

    def hybrid_rank(self, q, k=60):
        """Reciprocal Rank Fusion: robust, no score calibration needed."""
        scores = {}
        for ranking in (self.bm25_rank(q), self.dense_rank(q)):
            for r, idx in enumerate(ranking):
                scores[idx] = scores.get(idx, 0) + 1 / (k + r + 1)
        return sorted(scores, key=scores.get, reverse=True)


def recall_at_k(retriever, method, k):
    hits = 0
    for q, module in EVAL:
        top = getattr(retriever, f"{method}_rank")(q)[:k]
        hits += any(retriever.chunks[i]["module"] == module for i in top)
    return hits / len(EVAL)


def answer(question, retriever, k=5):
    from common.llm import ask
    top = [retriever.chunks[i] for i in retriever.hybrid_rank(question)[:k]]
    context = "\n\n".join(f"[{c['id']}] ({c['module']} › {c['section']})\n{c['text']}" for c in top)
    return ask(
        f"<context>\n{context}\n</context>\n\nQuestion: {question}",
        system="Answer using only the context. Cite chunk ids like [12] after each claim. "
               "If the context doesn't contain the answer, say so instead of guessing.",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ask")
    a = ap.parse_args()
    chunks = load_chunks()
    print(f"{len(chunks)} chunks from {len({c['module'] for c in chunks})} lessons")
    r = Retriever(chunks)

    print(f"\n{'method':<8} recall@1  recall@3  recall@5   (n={len(EVAL)} labeled questions)")
    for m in ("bm25", "dense", "hybrid"):
        print(f"{m:<8} " + "  ".join(f"{recall_at_k(r, m, k):>7.0%} " for k in (1, 3, 5)))

    if a.ask:
        print("\n" + answer(a.ask, r))


if __name__ == "__main__":
    main()
