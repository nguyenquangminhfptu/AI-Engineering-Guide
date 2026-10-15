# Module 04 — Fine-tuning: SFT, LoRA, QLoRA, DPO (Weeks 6–7)

**Goal:** fine-tune an open model on your dataset, prove it improved with numbers, and know which knob to turn when it didn't.

---

## 1. Should you fine-tune at all?

Try in this order and stop as soon as the eval says "good enough":
1. **Better prompt** (+ few-shot examples) on a strong model
2. **RAG** if the problem is missing knowledge
3. **Fine-tune** when you need: a consistent format/style, a narrow skill at high volume, lower latency/cost (small model), on-prem/privacy, or distilling a big model's behavior into a small one

A fine-tuned 1–8B model that matches a frontier model *on your one task* at 1/50th the cost is the classic win.

## 2. Full fine-tuning vs LoRA vs QLoRA

**Full fine-tuning** updates all weights. A 7B model in bf16 = 14 GB of weights, but training also needs gradients + Adam states → roughly **16 bytes/param ≈ 112 GB+**. Not on one GPU.

**LoRA** (Low-Rank Adaptation): freeze the original weight `W` (d×k). Learn a small update:

```
W' = W + (α / r) · B·A        A: r×k,  B: d×r,  r ≪ d (e.g. 16)
```
- Trainable params for one 4096×4096 layer: 16.7M → with r=16: 131k (**0.8%**).
- `B` starts at zero → at step 0 the model is exactly the original model (safe start).
- After training you can **merge** `B·A` into `W` → zero inference overhead. Or keep adapters separate and hot-swap many adapters on one base model.
- Optimizer state only for A, B → memory ≈ weights + activations.

**QLoRA**: store the frozen base in **4-bit (NF4)**, train LoRA adapters in bf16 on top. 7–8B model trains on a single 24 GB GPU (even a 16 GB T4 for 1–3B). Small quality cost, huge memory savings. Needs `bitsandbytes` → NVIDIA GPU.

👉 Lab: [lora_from_scratch.py](lora_from_scratch.py) implements LoRA in ~30 lines on the mini-GPT from module 02, then merges it and proves the merge is exact.

## 3. LoRA knobs

| Knob | Typical | Notes |
|---|---|---|
| `r` (rank) | 8–64 | higher = more capacity; 16 is a fine default |
| `lora_alpha` | r to 2r | scales the update; keep α/r fixed when you change r |
| `target_modules` | `"all-linear"` | attention-only (q,k,v,o) is cheaper; all-linear (incl. MLP) usually better |
| `lora_dropout` | 0–0.1 | |
| learning rate | **1e-4 – 2e-4** | ~10× higher than full fine-tuning |
| epochs | 1–3 | more → memorization, forgetting |
| effective batch | 16–64 | `per_device_batch × grad_accum × num_gpus` |
| `max_length` | covers ~95–99% of examples | check token-length distribution first |

## 4. SFT mechanics you must get right

- **Chat template**: the trainer applies `tokenizer.apply_chat_template` to `messages`. Train and serve with the **same** template.
- **Loss masking**: ideally compute loss only on assistant tokens (`assistant_only_loss=True` in TRL — requires a chat template that marks assistant spans; or `completion_only_loss` for prompt/completion datasets). Otherwise the model also learns to imitate user questions.
- **Packing**: concatenate short examples into full-length sequences for speed. Fine for SFT; check your trainer handles attention boundaries.
- **EOS**: every assistant turn must end with the end-of-turn token or the model won't stop.
- **Eval during training**: log train + val loss every N steps. Save checkpoints; the best one is often not the last.

👉 Lab: [sft_lora.py](sft_lora.py) — TRL `SFTTrainer` + PEFT LoRA (+ `--qlora` on NVIDIA), trains on module 03's dataset, then compares base vs tuned answers on the eval set.

## 5. Preference tuning: DPO

SFT teaches *what a good answer looks like*. DPO teaches *which of two answers is better* — useful for tone, conciseness, refusing correctly, avoiding a specific failure mode.

- Data: `(prompt, chosen, rejected)` pairs. Best source: sample 2–4 answers **from your SFT model**, have humans or a calibrated judge pick best/worst.
- Loss: increase log-prob of `chosen` vs `rejected`, **relative to a frozen reference model** (usually the SFT model). `beta` (0.05–0.5) controls how far you may drift from the reference.
- Always do SFT first, then DPO on top. LR is low (5e-7 – 5e-6 full; ~5e-6 – 5e-5 with LoRA).
- Watch `rewards/margins` and `rewards/accuracies` going up; watch for answers getting weirdly long (length hacking).

👉 Lab: [dpo.py](dpo.py)

## 6. Memory cheat-sheet (rough, for planning)

| Model | Full FT (bf16 + Adam) | LoRA (bf16 base) | QLoRA (4-bit base) |
|---|---|---|---|
| 0.5B | ~8 GB | ~3 GB | ~2 GB |
| 1.5B | ~24 GB | ~6 GB | ~4 GB |
| 7–8B | ~110+ GB | ~20 GB | ~10 GB |

Plus activations, which grow with batch × sequence length → use gradient checkpointing.

Where to train:
- **Your M1 Mac**: 0.5B LoRA in bf16/fp32 on MPS — slow but works for learning. (Apple's `mlx-lm` also supports LoRA on Mac and is faster.)
- **Free Colab T4 (16 GB)**: QLoRA up to ~3B.
- **Rented L4/A10 (24 GB)**: QLoRA 7–8B. **A100/H100**: bigger, faster, multi-GPU.
- **Unsloth** (library): ~2× faster, less memory, drop-in for single-GPU LoRA/QLoRA.

## 7. Debugging fine-tunes

| Symptom | Check |
|---|---|
| Loss is NaN / spikes | LR too high, fp16 overflow (use bf16), bad data rows |
| Loss doesn't decrease | LR too low, LoRA not applied (trainable params = 0?), labels all masked |
| Train ↓, val ↑ | overfitting → fewer epochs, more data, lower LR |
| Model repeats / never stops | EOS / chat template mismatch |
| Good loss, bad outputs | loss on wrong tokens, eval generation settings differ, data teaches the wrong thing → **read outputs** |
| Lost general skills | too many epochs/LR, mix in 10–30% general instruction data |

**Loss is not the goal. The eval score is.** Always generate on the eval set and compare base vs tuned (module 05).

---

## Labs
```bash
python lora_from_scratch.py                                  # needs ../02_llm_internals/mini_gpt.pt (run mini_gpt.py first)
python sft_lora.py --max_steps 30                            # Mac smoke test with Qwen2.5-0.5B-Instruct (engine=hf)
python sft_lora.py --qlora --model Qwen/Qwen2.5-1.5B-Instruct --epochs 2   # Colab / NVIDIA (engine=trl)
python dpo.py --adapter out/sft-lora                         # NVIDIA only; after sft_lora.py and synth_data.py --dpo
```
`sft_lora.py` has two engines. `trl` (SFTTrainer) is what you'll use at work on GPUs. `hf` writes the chat-template + assistant-only loss mask by hand in `tokenize_with_assistant_mask()` — **read that function**: it is exactly what SFTTrainer does for you, in 8 lines.

## Exercises
1. In `lora_from_scratch.py`, try r=1, 4, 16, 64. Plot new-domain val loss vs trainable params.
2. Compare forgetting (old-domain loss) for LoRA vs unfreezing everything (`--full`).
3. Run `sft_lora.py` with 1 vs 3 epochs on your data. Compare eval-set outputs, not just loss.
4. Count tokens of your dataset with the model's tokenizer; pick `max_length` to cover 99%.
5. Make 20 DPO pairs where `rejected` is "too long", run DPO, and measure average answer length before/after.

## At work
Run your team's fine-tune again with: a frozen eval set, experiment tracking (W&B or MLflow), and a base-vs-tuned comparison table. Write down every hyperparameter and why.

## Read
- LoRA paper (Hu et al., 2021) — sections 1–4. QLoRA paper (Dettmers et al., 2023) — section 2.
- Hugging Face TRL docs (SFTTrainer, DPOTrainer): https://huggingface.co/docs/trl
- PEFT docs: https://huggingface.co/docs/peft · Unsloth docs: https://docs.unsloth.ai
- Hugging Face smol-course: https://github.com/huggingface/smol-course
