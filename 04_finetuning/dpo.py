"""Lab 04-C: DPO (preference tuning) on top of your SFT adapter.

Runs on an NVIDIA GPU (Colab / cloud): TRL ≥1.x trainers use a Triton kernel, which is Linux-only.

Steps:
    python ../03_datasets/synth_data.py --dpo --n 50      # or build pairs from your SFT model's own samples
    python sft_lora.py ...                                # produces out/sft-lora
    python dpo.py --adapter out/sft-lora [--max_steps 20]

Recipe: merge the SFT adapter into the base → that merged model is the policy's starting point AND the
frozen reference (TRL uses the model with the new LoRA disabled as reference). Train a fresh LoRA with DPO.
"""
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "out"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", help="SFT LoRA adapter dir to start from (recommended)")
    ap.add_argument("--data", default=str(ROOT / "03_datasets" / "out" / "dpo.jsonl"))
    ap.add_argument("--beta", type=float, default=0.1, help="higher = stay closer to the reference model")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--max_steps", type=int, default=-1)
    a = ap.parse_args()

    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float32

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype)
    if a.adapter:
        model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()   # SFT model = new starting point
    model.to(device)

    ds = load_dataset("json", data_files=a.data)["train"]                         # columns: prompt, chosen, rejected
    split = ds.train_test_split(test_size=0.1, seed=0) if len(ds) >= 20 else {"train": ds, "test": ds}

    args = DPOConfig(
        output_dir=str(OUT / "dpo-lora"),
        beta=a.beta,
        learning_rate=a.lr,
        num_train_epochs=a.epochs,
        max_steps=a.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_length=1024,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="no",
        bf16=dtype == torch.bfloat16,
        report_to="none",
    )
    trainer = DPOTrainer(
        model=model, args=args, train_dataset=split["train"], eval_dataset=split["test"],
        processing_class=tok,
        peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM"),
    )
    trainer.train()
    trainer.save_model(str(OUT / "dpo-lora"))
    m = trainer.evaluate()
    print({k: round(v, 4) for k, v in m.items() if "rewards" in k or "loss" in k})
    print("Healthy run: eval_rewards/accuracies ↑ (>0.5) and eval_rewards/margins ↑. Then check answers on the eval set!")


if __name__ == "__main__":
    main()
