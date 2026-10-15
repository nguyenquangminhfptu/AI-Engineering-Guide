# AI Engineering Fast Track — Course Material

Hands-on material for the 12-week plan in [../ai_engineering_roadmap.html](../ai_engineering_roadmap.html).
Every module = **one lesson you read (LESSON.md)** + **labs you run and modify**.

> Rule: don't just read. Run every lab, break it, change one thing, and write down what happened.

## Modules

| # | Module | Weeks | Read | Run |
|---|--------|-------|------|-----|
| 01 | Agents, Claude Skills & MCP | 1–2 | [01_agents/LESSON.md](01_agents/LESSON.md) | `agent_loop.py`, `agent_runner.py`, `workflow_patterns.py`, `mcp_server.py`, `skills/` |
| 02 | How LLMs work inside | 3–4 | [02_llm_internals/LESSON.md](02_llm_internals/LESSON.md) | `bpe.py`, `mini_gpt.py` |
| 03 | Dataset engineering | 5 | [03_datasets/LESSON.md](03_datasets/LESSON.md) | `build_sft_dataset.py`, `synth_data.py` |
| 04 | Fine-tuning (LoRA, QLoRA, SFT, DPO) | 6–7 | [04_finetuning/LESSON.md](04_finetuning/LESSON.md) | `lora_from_scratch.py`, `sft_lora.py`, `dpo.py` |
| 05 | Evals | 8 | [05_evals/LESSON.md](05_evals/LESSON.md) | `eval_harness.py` |
| 06 | RAG, serving & LLMOps | 9–11 | [06_rag_serving/LESSON.md](06_rag_serving/LESSON.md) | `rag.py` |

Week 12 = capstone: combine 03 → 04 → 01 → 05 → 06 on a task from your job.

## Setup

```bash
cd course
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# For the LLM labs (module 01, synth_data.py, LLM-judge in 05, rag.py --ask):
export GEMINI_API_KEY=...     # free key: https://aistudio.google.com
```
All LLM calls go through `common/llm.py` (Gemini).

What runs where:

| Lab | Your M1 Mac (16 GB) | Needs NVIDIA GPU (Colab / cloud) | Needs API key |
|-----|:-:|:-:|:-:|
| 01 agents / MCP / skills | ✅ | | ✅ |
| 02 `bpe.py`, `mini_gpt.py` | ✅ (MPS, minutes) | | |
| 03 `build_sft_dataset.py` | ✅ | | |
| 03 `synth_data.py` | ✅ | | ✅ |
| 04 `lora_from_scratch.py` | ✅ | | |
| 04 `sft_lora.py` | ⚠️ 0.5B model, `hf` engine, no 4-bit | ✅ recommended (TRL + QLoRA) | |
| 04 `dpo.py` | ❌ (TRL needs Triton → Linux/NVIDIA) | ✅ | |
| 05 `eval_harness.py` | ✅ | | ✅ for LLM-judge |
| 06 `rag.py` | ✅ | | |

For fine-tuning on a GPU: a free Google Colab T4 is enough for a 0.5B–1.5B model with QLoRA; rent an A10/L4/A100 (RunPod, Lambda, Modal) for 7–8B models.

## How to study one module (≈ 1 week)

1. **Read** the LESSON.md once fast (1 hour). Don't try to memorize.
2. **Run** the labs as-is. Confirm you get the output described.
3. **Do the exercises** at the bottom of each lesson. This is where learning happens.
4. **Apply** it to one real task at work (the "At work" box in each lesson).
5. **Write** 5–10 lines in a log: what you built, the eval number, what broke.

## The one idea that ties everything together

```
            ┌──────────── eval set (fixed, held-out) ────────────┐
            ▼                                                     │
 data ──► model (prompt / fine-tune / RAG) ──► agent / skill ──► measure ──► change ONE thing ──┐
  ▲                                                                                            │
  └──────────────────────────────── error analysis tells you what to change ◄──────────────────┘
```

Prompting, RAG, fine-tuning and agents are all just different knobs. The eval loop is what makes you an engineer instead of someone trying things.
