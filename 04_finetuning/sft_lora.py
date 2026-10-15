"""Lab 04-B: real SFT with LoRA / QLoRA using Hugging Face PEFT (+ TRL on NVIDIA GPUs).

Mac smoke test (0.5B, a few minutes):
    python sft_lora.py --max_steps 30
Colab / NVIDIA GPU (QLoRA):
    pip install bitsandbytes
    python sft_lora.py --qlora --model Qwen/Qwen2.5-1.5B-Instruct --epochs 2

Two engines, same result:
  --engine trl  TRL SFTTrainer: the industry-standard tool. Default on CUDA. (TRL ≥1.x needs Triton → Linux/NVIDIA only.)
  --engine hf   plain transformers.Trainer + PEFT with the chat template and assistant-only loss mask
                written out by hand, so you can see exactly what SFTTrainer does for you. Default on Mac/CPU.

Data: module 03 output (out/train.jsonl, out/eval.jsonl, + out/synthetic.jsonl if present).
Output: out/sft-lora/ (adapter) and out/compare.jsonl (base vs tuned answers on the eval set → grade in module 05).
"""
import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "03_datasets" / "out"
OUT = Path(__file__).resolve().parent / "out"


def pick_device_dtype(force=None):
    if force:
        return force, (torch.bfloat16 if force == "cuda" else torch.float32)
    if torch.cuda.is_available():
        return "cuda", (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    if torch.backends.mps.is_available():
        return "mps", torch.float32   # fp32 is the safe choice on Apple GPUs for training
    return "cpu", torch.float32


def tokenize_with_assistant_mask(messages, tok, max_length):
    """What SFTTrainer does internally: chat template → token ids, loss only on the final assistant reply.
    Prompt tokens get label -100 (= ignored by cross-entropy), so the model learns to answer, not to ask."""
    prompt_ids = tok.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=True, return_dict=False)
    full_ids = tok.apply_chat_template(messages, tokenize=True, return_dict=False)   # ends with <|im_end|> = learn to STOP
    assert full_ids[:len(prompt_ids)] == prompt_ids, "chat template is not prefix-consistent; mask per turn instead"
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return {"input_ids": full_ids[:max_length], "attention_mask": [1] * min(len(full_ids), max_length),
            "labels": labels[:max_length]}


@torch.no_grad()
def answer(model, tok, messages, max_new_tokens=256):
    inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)   # greedy → reproducible comparison
    return tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--qlora", action="store_true", help="4-bit base model (NVIDIA + bitsandbytes only)")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--max_steps", type=int, default=-1, help="override epochs for quick tests")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--engine", choices=["trl", "hf"], help="default: trl on CUDA, hf elsewhere")
    ap.add_argument("--device", choices=["cuda", "mps", "cpu"], help="override auto-detection (e.g. cpu if MPS swaps)")
    ap.add_argument("--eval_limit", type=int, default=0, help="only generate base-vs-tuned for the first N eval rows (0 = all)")
    a = ap.parse_args()

    device, dtype = pick_device_dtype(a.device)
    engine = a.engine or ("trl" if device == "cuda" else "hf")
    print(f"device={device} dtype={dtype} model={a.model} qlora={a.qlora} engine={engine}")

    # ---- data: chat "messages" format; the trainer applies the model's chat template ----
    train_files = [str(DATA / "train.jsonl")] + ([str(DATA / "synthetic.jsonl")] if (DATA / "synthetic.jsonl").exists() else [])
    ds = load_dataset("json", data_files={"train": train_files, "eval": str(DATA / "eval.jsonl")})
    ds = ds.select_columns(["messages"])
    print(f"train={len(ds['train'])} eval={len(ds['eval'])}")

    tok = AutoTokenizer.from_pretrained(a.model)
    lengths = sorted(len(tok.apply_chat_template(r["messages"], tokenize=True, return_dict=False)) for r in ds["train"])
    print(f"tokens/example: median={lengths[len(lengths) // 2]} p99={lengths[int(len(lengths) * 0.99) - 1]} max={lengths[-1]} (max_length={a.max_length})")

    # ---- model ----
    model_kwargs = {"dtype": dtype}
    if a.qlora:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(a.model, **model_kwargs)
    if not a.qlora:
        model.to(device)

    peft_config = LoraConfig(
        r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05,
        target_modules="all-linear",   # attention + MLP projections
        task_type="CAUSAL_LM",
    )

    common = dict(
        output_dir=str(OUT / "sft-lora"),
        num_train_epochs=a.epochs,
        max_steps=a.max_steps,
        per_device_train_batch_size=a.batch,
        gradient_accumulation_steps=a.grad_accum,     # effective batch = batch × grad_accum
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="no",
        gradient_checkpointing=device == "cuda",
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        report_to="none",                           # set "wandb" or "mlflow" for experiment tracking
        seed=42,
    )
    if engine == "trl":
        from trl import SFTConfig, SFTTrainer
        # SFTTrainer applies the chat template, masks the loss, packs/pads and wraps the model with LoRA for you.
        trainer = SFTTrainer(model=model, args=SFTConfig(max_length=a.max_length, **common),
                             train_dataset=ds["train"], eval_dataset=ds["eval"],
                             processing_class=tok, peft_config=peft_config)
    else:
        model = get_peft_model(model, peft_config)  # freeze base, inject LoRA A/B matrices
        if a.qlora or device == "cuda":
            model.enable_input_require_grads()      # needed with gradient checkpointing on a frozen base
        tokenized = ds.map(lambda r: tokenize_with_assistant_mask(r["messages"], tok, a.max_length),
                           remove_columns=["messages"])
        trainer = Trainer(model=model, args=TrainingArguments(**common),
                          train_dataset=tokenized["train"], eval_dataset=tokenized["eval"],
                          data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
    trainer.model.print_trainable_parameters()
    trainer.train()
    trainer.save_model(str(OUT / "sft-lora"))   # saves only the adapter (a few MB)
    print("final eval:", trainer.evaluate())

    # ---- base vs tuned on the held-out eval set (same weights, adapter on/off) ----
    peft_model = trainer.model
    peft_model.eval()
    peft_model.config.use_cache = True
    rows = []
    eval_rows = load_dataset("json", data_files=str(DATA / "eval.jsonl"))["train"]
    if a.eval_limit:
        eval_rows = eval_rows.select(range(min(a.eval_limit, len(eval_rows))))
    for r in eval_rows:
        prompt, reference = r["messages"][:-1], r["messages"][-1]["content"]
        with peft_model.disable_adapter():
            base = answer(peft_model, tok, prompt)
        tuned = answer(peft_model, tok, prompt)
        rows.append({"question": prompt[-1]["content"], "reference": reference, "base": base, "tuned": tuned})
        print(f"\nQ: {prompt[-1]['content'][:100]}\n  BASE : {base[:200]!r}\n  TUNED: {tuned[:200]!r}")
    with open(OUT / "compare.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nwrote {OUT / 'compare.jsonl'} → grade it: python ../05_evals/eval_harness.py compare {OUT / 'compare.jsonl'}")


if __name__ == "__main__":
    main()
