"""Lab 04-A: LoRA from scratch on the mini-GPT from module 02.

Story: the model was pretrained on Shakespeare (domain A). We adapt it to technical English (domain B:
this course's lessons) by training a small fraction of the parameters, measure forgetting on A, then merge the adapter.

Run:  python ../02_llm_internals/mini_gpt.py      # once, creates mini_gpt.pt
      python lora_from_scratch.py [--rank 8] [--full]
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "02_llm_internals"))
from mini_gpt import GPT, Config, estimate_loss, load_text, pick_device, train  # noqa: E402


class LoRALinear(nn.Module):
    """y = x·Wᵀ + (α/r)·x·Aᵀ·Bᵀ   with W frozen, only A and B trained."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False                                       # freeze original weights
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)    # small random
        self.B = nn.Parameter(torch.zeros(base.out_features, r))          # zero → starts as the original model
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale

    def merged(self) -> nn.Linear:
        """Fold B·A into W: same outputs, no extra cost at inference."""
        lin = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        lin.weight.data = self.base.weight.data + (self.B @ self.A) * self.scale
        if self.base.bias is not None:
            lin.bias.data = self.base.bias.data.clone()
        return lin.to(self.base.weight.device)


def apply_lora(model, r, alpha, targets=("qkv", "proj", "up", "down")):
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and name in targets:
                setattr(module, name, LoRALinear(child, r, alpha))


def merge_lora(model):
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, name, child.merged())


def encode(text, chars):
    stoi = {c: i for i, c in enumerate(chars)}
    return torch.tensor([stoi[c] for c in text if c in stoi], dtype=torch.long)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "02_llm_internals" / "mini_gpt.pt"))
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--full", action="store_true", help="full fine-tuning instead of LoRA, for comparison")
    a = ap.parse_args()

    torch.manual_seed(0)
    device = pick_device()
    ck = torch.load(a.ckpt, map_location=device)
    model = GPT(Config(**ck["config"])).to(device)
    model.load_state_dict(ck["model"])
    chars, bs = ck["chars"], model.config.block_size

    old = encode(load_text(), chars)                                            # domain A
    new = encode("\n".join(p.read_text() for p in sorted(ROOT.rglob("LESSON.md"))), chars)  # domain B
    n = int(0.9 * len(new))
    new_train, new_val = new[:n], new[n:]
    old_val = old[int(0.9 * len(old)):]

    def report(tag):
        l = estimate_loss(model, {"old(A)": old_val, "new(B)": new_val}, bs, 32, device)
        print(f"{tag:<22} old-domain loss {l['old(A)']:.3f} | new-domain loss {l['new(B)']:.3f}")
        return l

    report("before fine-tuning")

    total = sum(p.numel() for p in model.parameters())
    if not a.full:
        for p in model.parameters():
            p.requires_grad = False
        apply_lora(model, a.rank, a.alpha)
        model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"mode={'FULL' if a.full else f'LoRA r={a.rank}'}  trainable {n_train:,} / {total:,} params ({n_train / total:.1%})")

    train(model, new_train, new_val, a.steps, a.lr, 32, device, log_every=150, params=trainable)
    after = report("after fine-tuning")

    if not a.full:
        model.eval()
        x = torch.randint(len(chars), (2, bs), device=device)
        with torch.no_grad():
            y_lora = model(x)[0]
            merge_lora(model)
            y_merged = model(x)[0]
        print(f"merged adapter into weights: max |difference| = {(y_lora - y_merged).abs().max().item():.2e}  (≈0 → merge is exact)")
        report("after merge")

    start = torch.tensor([encode("\n## ", chars).tolist()], device=device)
    print("\n=== sample after adaptation ===")
    print("".join(chars[i] for i in model.generate(start, 300)[0].tolist()))


if __name__ == "__main__":
    main()
