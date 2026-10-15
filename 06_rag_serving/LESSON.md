# Module 06 — RAG, Serving & LLMOps (Weeks 9–11)

**Goal:** give models the right knowledge at runtime (RAG), put your fine-tuned model behind a fast API, and run it like a production service.

---

## Part A — RAG (Retrieval-Augmented Generation)

### 1. Why
Models don't know your internal docs, and fine-tuning is a poor way to teach facts. RAG = **search first, then answer using what was found**, with citations.

```
offline:  docs → clean → chunk → embed → index (vector DB + keyword index)
online:   question → retrieve top-k → (rerank) → put chunks in prompt → answer with citations
```

### 2. The parts and their knobs

| Part | Knob | Advice |
|---|---|---|
| **Chunking** | size, overlap, boundaries | split on structure (headings, paragraphs), ~200–800 tokens; keep title/section as metadata |
| **Embeddings** | model | multilingual model if you have Vietnamese (e.g. `bge-m3`, `multilingual-e5`); measure, don't assume |
| **Keyword search (BM25)** | — | catches exact terms (error codes, function names, IDs) that embeddings miss |
| **Hybrid** | fusion | combine BM25 + vectors with Reciprocal Rank Fusion: `score = Σ 1/(60 + rank)` |
| **Reranker** | cross-encoder | re-score the top 20–50 with a stronger model, keep top 3–8 |
| **Generation** | prompt | "answer only from the context; say if it's not there; cite chunk ids" |

### 3. Evaluate retrieval separately from generation
- **Retrieval**: labeled (question → relevant chunk/doc) pairs → **recall@k** ("is a relevant chunk in the top k?") and MRR. If recall@5 is 60%, no prompt will save you.
- **Generation**: faithfulness (claims supported by context?), answer correctness, citation correctness → rubric judge (module 05).

👉 Lab: [rag.py](rag.py) — BM25 vs embeddings vs hybrid on this course's lessons, with recall@k, and an optional Claude answer with citations.

### 4. RAG vs fine-tuning vs long context
- **Knowledge that changes / must be cited** → RAG.
- **Behavior, format, style, narrow skill** → fine-tune.
- **Small, stable corpus (< a few hundred pages)** → just put it in the long context window with prompt caching; simplest of all.
- Often combined: fine-tune a small model to *use retrieved context well*.

---

## Part B — Serving your model

### 5. Merge or not
- **Merge** adapter into base (`model.merge_and_unload()`) → one standalone model, simplest deploy.
- **Keep adapters separate** → one base model on the GPU, many LoRA adapters swapped per request (vLLM `--enable-lora`). Great when each team/customer has its own adapter.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype="auto")
merged = PeftModel.from_pretrained(base, "out/sft-lora").merge_and_unload()
merged.save_pretrained("out/merged"); AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct").save_pretrained("out/merged")
```

### 6. vLLM (NVIDIA GPU)
```bash
pip install vllm
vllm serve out/merged --max-model-len 4096                                   # OpenAI-compatible API on :8000
vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-lora --lora-modules helpdesk=out/sft-lora
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"helpdesk","messages":[{"role":"user","content":"LoRA OOM on 24GB?"}]}'
```
Why vLLM is fast: **continuous batching** (new requests join a running batch), **PagedAttention** (KV cache memory without fragmentation), optimized kernels, quantization support.

Local/Mac alternatives: **Ollama** or **llama.cpp** (GGUF quantized models), **mlx-lm** on Apple Silicon.

### 7. Serving metrics
| Metric | Meaning |
|---|---|
| TTFT (time to first token) | prefill cost; grows with prompt length |
| TPOT / tokens per second | decode speed per request |
| Throughput | total tokens/s across all users — what batching improves |
| p95 latency | what your users actually feel |
| Cost per 1k requests | GPU $/hour ÷ requests/hour |

Load-test before launch (e.g. `vllm bench serve`, `locust`, `k6`).

### 8. Production wrapper
FastAPI (or your existing backend) in front of the model server: auth, rate limits, timeouts, input size limits, request logging (with PII rules), and a fallback model if the primary is down. Package with Docker; deploy like any other service.

---

## Part C — LLMOps

| Practice | Tools | Why |
|---|---|---|
| Experiment tracking | W&B, MLflow | which data + hyperparams produced which model and score |
| Data & model versioning | HF Hub (private), DVC, S3 + manifest | reproducibility, rollback |
| Tracing | Langfuse, LangSmith, Arize Phoenix, OpenTelemetry | see every prompt, tool call, latency, cost in production |
| Evals in CI | your harness (module 05) | block regressions when prompts/models/data change |
| Online monitoring | dashboards + alerts | drift in inputs, rising refusal/error rate, cost spikes |
| Feedback loop | thumbs up/down, human review queue | new eval cases + new training data from real failures |
| Safety | guardrails, red-teaming, approvals for agent actions | prompt injection, data leakage |

**The flywheel**: production traces → find failures → add to eval set → fix (prompt/data/fine-tune) → eval in CI → deploy → repeat.

---

## Labs
```bash
python rag.py                      # builds index over course lessons, prints recall@k for bm25 / dense / hybrid
python rag.py --ask "How do I choose LoRA rank?"     # retrieval + Claude answer with citations (needs API key)
```

## Exercises
1. Change chunk size (by heading vs fixed 300 chars). How does recall@k change?
2. Add 5 labeled questions that use exact technical terms. Which retriever wins on those?
3. Make the generator answer "not in the docs" when retrieval misses. Test with an off-topic question.
4. Merge your module-04 adapter and serve it with Ollama or vLLM; measure tokens/second.
5. Turn on `report_to="wandb"` (or mlflow) in `sft_lora.py` and compare two runs side by side.

## At work
Design the production path for your fine-tuned model: merge vs multi-adapter, serving engine, metrics, eval-in-CI, tracing. Write it as a one-page design doc.

## Read
- Chip Huyen, *AI Engineering* — chapters 6 (RAG & agents), 9 (inference optimization), 10 (architecture & feedback)
- vLLM docs: https://docs.vllm.ai · Qdrant docs: https://qdrant.tech/documentation/
- Anthropic — "Contextual Retrieval" (engineering blog)
