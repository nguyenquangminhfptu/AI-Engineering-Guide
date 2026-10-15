# Module 03 — Dataset Engineering (Week 5)

**Goal:** build fine-tuning datasets that actually improve a model, plus an eval set you can trust.

> Data quality beats data quantity. 1,000 excellent, diverse, consistent examples usually beat 50,000 noisy ones for SFT (see the LIMA paper). Most "fine-tuning didn't work" problems are data problems.

---

## 1. Decide what you're teaching

Before collecting anything, write one paragraph: **what behavior should change after fine-tuning?** Fine-tuning is good at:
- **format / style / tone** (always JSON in schema X, company voice, Vietnamese register)
- **narrow task skill** (classify tickets into 40 internal categories, extract fields from your documents)
- **distillation** (make a small cheap model imitate a big model on *your* task)
- **tool-call formats** for agents

It is bad at **adding fresh knowledge** (facts change, the model hallucinates around them) → use RAG for knowledge (module 06).

## 2. Formats

**SFT (chat format)** — one JSON object per line (JSONL):
```json
{"messages": [
  {"role": "system", "content": "You are the ML platform helpdesk..."},
  {"role": "user", "content": "My LoRA run OOMs on a 24GB GPU"},
  {"role": "assistant", "content": "Try, in order: 1) ..."}
]}
```

**Preference (DPO)**:
```json
{"prompt": [{"role": "user", "content": "..."}],
 "chosen": [{"role": "assistant", "content": "better answer"}],
 "rejected": [{"role": "assistant", "content": "worse answer"}]}
```

**Tool-use / agent traces**: messages that include assistant tool calls and tool results, in the target model's tool format.

The trainer turns `messages` into tokens with the model's **chat template**. Never hand-format `<|im_start|>` strings yourself.

## 3. Where data comes from

| Source | Pros | Watch out |
|---|---|---|
| Real logs (tickets, chats, emails) | realistic distribution | PII, consent/licensing, bad answers mixed in |
| Expert-written | highest quality | slow, expensive → use for eval set + seeds |
| **Synthetic (strong LLM)** | fast, scalable, controllable | low diversity, model's own mistakes, style collapse; check the model provider's terms |
| Public HF datasets | free | license, contamination with benchmarks, domain mismatch |

The usual recipe at work: **real prompts + strong-model or expert answers + human review of a sample**.

## 4. The pipeline (what `build_sft_dataset.py` does)

```
raw ─► normalize ─► convert to messages ─► validate ─► filter ─► PII mask ─► exact dedup ─► near-dup dedup ─► split ─► card
```

1. **Normalize**: Unicode NFC (very important for Vietnamese — the same letter can be encoded two ways), collapse whitespace, strip HTML.
2. **Validate**: roles alternate, ends with assistant, nothing empty.
3. **Filter** with explicit, logged rules: too short / too long, boilerplate ("As an AI…"), placeholder text, wrong language, broken encoding. **Log why each row was dropped** so you can audit the filters themselves.
4. **PII**: mask emails, phones, IDs, keys before data leaves your control.
5. **Dedup**: exact (hash of normalized text) then **near-duplicate** (MinHash over character n-grams — catches "same ticket, different greeting").
6. **Split**: train / eval. Split **by group** (customer, document, conversation thread, template) so near-identical things can't land on both sides.
7. **Dataset card**: source, size, filters, splits, known gaps, license. Future-you will need it.

## 5. The eval set comes FIRST

- Freeze 100–300 examples **before** you start iterating. Never train on them, never tune prompts by staring at them too much.
- Make it **representative** of real traffic + include a **hard/edge slice** (long inputs, ambiguous requests, adversarial).
- Tag examples (`topic`, `difficulty`) so you can see per-slice scores.
- Check for **leakage**: no eval prompt (or near-dup) in train. The checker skill in module 01 does this.

## 6. Synthetic data that doesn't suck (`synth_data.py`)

LLM-generated data tends to be repetitive. Force diversity explicitly:
- **Seed × persona × difficulty × format** grid → generate across combinations.
- Use **real prompts as seeds** whenever you have them; let the model generate answers or variations.
- **Generate → judge → filter**: a second pass scores each example with a rubric; keep only high scores.
- Dedup again after generation.
- **Human-review a random 5%**. If >10% are bad, fix the generator prompt, don't just filter harder.

## 7. Diagnostics that tell you the data is the problem

| Symptom after fine-tuning | Likely data cause |
|---|---|
| Repeats same opening phrase | low diversity in assistant replies |
| Answers cut off mid-sentence | examples truncated by `max_length` |
| Never stops / rambles | missing EOS / wrong chat template |
| Great on eval, bad in prod | eval leaked into train, or eval not representative |
| Forgot general abilities | dataset too narrow + too many epochs → mix in general data, lower LR |
| Confidently wrong facts | teaching knowledge via SFT → use RAG |

---

## Labs
```bash
python build_sft_dataset.py                        # runs on sample_raw.jsonl → out/train.jsonl, out/eval.jsonl, out/DATASET_CARD.md
python ../01_agents/skills/sft-dataset-checker/scripts/check_dataset.py out/train.jsonl --eval out/eval.jsonl
python synth_data.py --n 20                        # needs GEMINI_API_KEY → out/synthetic.jsonl
python synth_data.py --dpo --n 10                  # preference pairs → out/dpo.jsonl
```

## Exercises
1. Read the drop report from `build_sft_dataset.py`. Which rule dropped the most? Is each drop correct?
2. Lower the near-dup threshold from 0.7 to 0.5. What gets merged that shouldn't?
3. Add a filter rule: drop rows where the answer is not in the expected language.
4. Generate 50 synthetic examples. Read 10. Write down 3 failure patterns and fix the generator prompt.
5. Change the split to be by `topic` instead of random. Why can random splitting inflate eval scores?

## At work
Take your current company dataset. Run the pipeline + checker on it. Write its dataset card. Build (or freeze) an eval set of 100+ examples with tags.

## Read
- Hugging Face LLM Course — fine-tuning chapters: https://huggingface.co/learn/llm-course
- Chip Huyen, *AI Engineering* — chapter 8 "Dataset Engineering"
- LIMA: Less Is More for Alignment (2023)
