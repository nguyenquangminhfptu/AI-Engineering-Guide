"""Lab 02-B: a complete GPT (decoder-only Transformer) you can pretrain on a laptop.

Run:  python mini_gpt.py                      # ~1M params, a few minutes on an M1
      python mini_gpt.py --steps 3000 --n_layer 6 --n_embd 192
      python mini_gpt.py --data my_text.txt
      python mini_gpt.py --device cpu --n_layer 2 --n_embd 64 --steps 100

Same architecture family as GPT-2/Llama, just tiny and character-level.
Module 04 imports GPT from this file to teach LoRA.
"""
import argparse
import math
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
SHAKESPEARE = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


@dataclass
class Config:
    vocab_size: int = 65
    block_size: int = 128   # context length
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.n_head = c.n_head
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd)   # Wq, Wk, Wv fused into one matrix
        self.proj = nn.Linear(c.n_embd, c.n_embd)      # Wo
        self.drop = nn.Dropout(c.dropout)
        mask = torch.tril(torch.ones(c.block_size, c.block_size)).view(1, 1, c.block_size, c.block_size)
        self.register_buffer("mask", mask)             # lower-triangular = can't see the future

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, heads, T, head_dim): each head attends independently
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))   # (B, h, T, T) relevance scores
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))  # causal mask
        att = self.drop(F.softmax(att, dim=-1))
        y = att @ v                                                # weighted mix of values
        y = y.transpose(1, 2).contiguous().view(B, T, C)           # re-join heads
        return self.drop(self.proj(y))
        # (Production code uses F.scaled_dot_product_attention — same math, fused & faster.)


class MLP(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.up = nn.Linear(c.n_embd, 4 * c.n_embd)
        self.down = nn.Linear(4 * c.n_embd, c.n_embd)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x):
        return self.drop(self.down(F.gelu(self.up(x))))


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(c.n_embd), CausalSelfAttention(c)
        self.ln2, self.mlp = nn.LayerNorm(c.n_embd), MLP(c)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # residual: communicate across tokens
        x = x + self.mlp(self.ln2(x))    # residual: think per token
        return x


class GPT(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.config = c
        self.tok_emb = nn.Embedding(c.vocab_size, c.n_embd)
        self.pos_emb = nn.Embedding(c.block_size, c.n_embd)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln_f = nn.LayerNorm(c.n_embd)
        self.head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight          # weight tying (GPT-2 trick: fewer params)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        # GPT-2 init: small weights → near-uniform predictions at step 0 → initial loss ≈ ln(vocab).
        # PyTorch's default Embedding init is N(0, 1); with tied weights that makes logits huge (initial loss ~80).
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))               # (B, T, vocab)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.config.block_size:])   # crop to context window
            logits = logits[:, -1, :] / temperature              # only the last position predicts next
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            next_id = torch.multinomial(F.softmax(logits, dim=-1), 1)  # sample
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# ---------------- data: character-level tokenizer ----------------
class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)


def load_text(path=None):
    if path:
        return Path(path).read_text()
    cache = HERE / "data" / "tinyshakespeare.txt"
    if not cache.exists():
        try:
            cache.parent.mkdir(exist_ok=True)
            urllib.request.urlretrieve(SHAKESPEARE, cache)
        except Exception as e:
            print(f"download failed ({e}); training on the course lesson files instead")
            return "\n".join(p.read_text() for p in sorted(HERE.parent.rglob("LESSON.md")))
    return cache.read_text()


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])   # targets = inputs shifted by one
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, splits, block_size, batch_size, device, iters=20):
    model.eval()
    out = {}
    for name, data in splits.items():
        out[name] = sum(model(*get_batch(data, block_size, batch_size, device))[1].item() for _ in range(iters)) / iters
    model.train()
    return out


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(model, train_data, val_data, steps, lr, batch_size, device, log_every=200, params=None):
    """Generic loop reused by module 04 (pass `params` to train only some weights)."""
    opt = torch.optim.AdamW(params if params is not None else model.parameters(), lr=lr, weight_decay=0.01)
    warmup = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(  # linear warmup, then cosine decay to 10%
        opt, lambda s: min((s + 1) / warmup, 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s / steps, 1.0)))))
    bs = model.config.block_size
    t0 = time.time()
    for step in range(steps + 1):
        if step % log_every == 0 or step == steps:
            l = estimate_loss(model, {"train": train_data, "val": val_data}, bs, batch_size, device)
            print(f"step {step:5d} | train {l['train']:.3f} | val {l['val']:.3f} | lr {sched.get_last_lr()[0]:.1e} | {time.time() - t0:.0f}s")
        if step == steps:
            break
        x, y = get_batch(train_data, bs, batch_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_embd", type=int, default=128)
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--sample_tokens", type=int, default=400)
    ap.add_argument("--out", default=str(HERE / "mini_gpt.pt"))
    a = ap.parse_args()

    if min(a.n_layer, a.n_head, a.n_embd, a.block_size, a.batch_size, a.steps, a.log_every) < 1:
        ap.error("layers, heads, embedding size, context, batch size, steps and log interval must be positive")
    if a.n_embd % a.n_head != 0:
        ap.error("--n_embd must be divisible by --n_head (e.g. 64 / 4 = 16 dimensions per head)")
    if not 0 <= a.dropout < 1 or a.lr <= 0 or a.sample_tokens < 0:
        ap.error("require 0 <= --dropout < 1, --lr > 0 and --sample_tokens >= 0")
    if a.device == "cuda" and not torch.cuda.is_available():
        ap.error("CUDA is unavailable; use --device cpu or auto")
    if a.device == "mps" and not torch.backends.mps.is_available():
        ap.error("MPS is unavailable; use --device cpu or auto")

    torch.manual_seed(1337)
    device = pick_device() if a.device == "auto" else a.device
    text = load_text(a.data)
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))                       # 90/10 split: val text is NEVER trained on
    train_data, val_data = data[:n], data[n:]
    if min(len(train_data), len(val_data)) < a.block_size + 2:
        ap.error(f"each 90/10 data split needs at least {a.block_size + 2} characters; "
                 "provide more text or reduce --block_size")

    cfg = Config(vocab_size=len(tok.chars), block_size=a.block_size, n_layer=a.n_layer,
                 n_head=a.n_head, n_embd=a.n_embd, dropout=a.dropout)
    model = GPT(cfg).to(device)
    print(f"device={device} chars={len(text):,} vocab={cfg.vocab_size} params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"expected initial loss ≈ ln(vocab) = {math.log(cfg.vocab_size):.3f}")

    train(model, train_data, val_data, a.steps, a.lr, a.batch_size, device, log_every=a.log_every)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg.__dict__, "chars": tok.chars}, out)
    print(f"saved {a.out}\n\n=== sample ===")
    model.eval()  # disable dropout while sampling
    start = torch.tensor([[tok.stoi.get("\n", 0)]], dtype=torch.long, device=device)
    print(tok.decode(model.generate(start, a.sample_tokens)[0].tolist()))


if __name__ == "__main__":
    main()
