# Module 02 — How LLMs Work Inside (Weeks 3–4)

**Goal:** understand every line of a GPT training script well enough to make fine-tuning decisions on purpose (learning rate, context length, loss, tokenizer, chat template) instead of copying settings.

You don't need a math month. You need the ~8 ideas below, and the code in `mini_gpt.py` shows each one.

---

## 1. An LLM is a next-token predictor

Given tokens `t1 … tn`, the model outputs a probability for **every token in the vocabulary** being `t(n+1)`.
Generation = predict → sample one token → append → repeat. That's all. Chat, reasoning and tool calls are learned *formats* on top of this.

## 2. Tokens and tokenizers

Text → integers. Modern LLMs use **BPE** (byte-pair encoding): start from bytes, repeatedly merge the most frequent adjacent pair into a new token. Common words become 1 token; rare words split into pieces.

Why you care at work:
- **Cost and context limits** are in tokens, not characters. Vietnamese text often uses more tokens per word than English.
- **Chat templates**: a chat is flattened into one token stream with special tokens, e.g. (Qwen)
  ```
  <|im_start|>system\nYou are…<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello!<|im_end|>
  ```
  If you fine-tune with the wrong template, or run inference with a different template than training, quality collapses. **Always use `tokenizer.apply_chat_template`.**
- The EOS token teaches the model when to **stop**. Missing EOS in training data → model that rambles forever.

👉 Lab: [bpe.py](bpe.py) trains a tiny BPE tokenizer from scratch in ~60 lines.

## 3. Embeddings

Each token id looks up a learned vector (e.g. 4096 numbers). Positions also get information (learned position embeddings in GPT-2; **RoPE** rotations in modern models). Meaning lives in these vectors: similar tokens → nearby vectors.

## 4. Attention — the core idea

For each token, the model asks: "which earlier tokens matter for me?"

```
Q = x·Wq   (what am I looking for?)
K = x·Wk   (what do I contain?)
V = x·Wv   (what do I pass on?)
scores  = Q·Kᵀ / √d           # how relevant is each token to each other token
scores  = mask(scores)        # CAUSAL: a token can't see the future → set future to -inf
weights = softmax(scores)     # turn scores into probabilities that sum to 1
out     = weights · V         # weighted mix of the values
```

**Multi-head** = do this several times in parallel with smaller dims, so different heads learn different relations (syntax, coreference, copying…).

**Cost**: scores are n×n → memory/compute grows with the **square of context length**. That's why long context is expensive, and why the **KV cache** (storing K and V of past tokens during generation) matters for serving.

## 5. The Transformer block

```
x = x + Attention(LayerNorm(x))     # mix information ACROSS tokens
x = x + MLP(LayerNorm(x))           # process each token independently (most parameters live here)
```
- The `x + …` are **residual connections**: they let gradients flow through deep stacks.
- **LayerNorm/RMSNorm** keeps activations in a stable range.
- Stack N blocks → final norm → linear layer to vocab size → **logits**.

The linear layers here (`Wq, Wk, Wv, Wo`, MLP up/down) are exactly what **LoRA** targets in module 04.

## 6. Loss: cross-entropy

For each position: `loss = -log(probability the model gave to the correct next token)`.
- Perfect prediction → loss 0. Uniform guessing over V tokens → loss = ln(V) (≈ 4.17 for 65 chars, ≈ 11.9 for a 150k vocab).
- **Perplexity** = e^loss ≈ "how many tokens the model is choosing between".
- In SFT, we usually compute loss **only on assistant tokens** (mask the prompt) — we want the model to learn to answer, not to predict the user's question.

## 7. Training loop: gradient descent

```
for step:
    batch  = random chunks of text            # inputs x, targets y = x shifted by 1
    logits = model(x)
    loss   = cross_entropy(logits, y)
    loss.backward()                           # backprop: gradient of loss w.r.t. every weight
    clip_grad_norm(model, 1.0)                # stop rare huge updates
    optimizer.step(); optimizer.zero_grad()   # AdamW: weight -= lr * (smoothed gradient)
```

Hyperparameters you'll touch when fine-tuning:

| Knob | What it does | Symptom when wrong |
|---|---|---|
| Learning rate | step size | too high: loss spikes / NaN / model forgets everything. too low: loss barely moves |
| LR schedule (warmup + cosine) | small LR at start, decay at end | no warmup → early instability |
| Batch size (× grad accumulation) | examples per update | too small → noisy loss |
| Epochs | passes over data | too many → train loss ↓ but val loss ↑ (**overfitting**, memorization) |
| Context length (`max_length`) | tokens per example | too short → answers truncated, model learns cut-off replies |
| Weight decay, dropout | regularization | mostly leave defaults |

**Always watch train AND validation loss.** Val loss turning upward = stop.

## 8. Sampling

- **Temperature**: divide logits before softmax. <1 = sharper/safer, >1 = more random. 0 ≈ greedy.
- **Top-k / top-p**: only sample from the k most likely tokens / the smallest set covering p probability.

## 9. Pretraining → SFT → preference tuning

| Stage | Data | Result |
|---|---|---|
| Pretraining | trillions of tokens of raw text | "base model": knows language & facts, just continues text |
| SFT (instruction tuning) | 10k–1M (prompt, ideal response) chats | follows instructions, uses chat format |
| Preference tuning (RLHF / DPO) | pairs: chosen vs rejected answer | prefers helpful/safe/styled answers |
| (Reasoning RL) | verifiable problems + rewards | thinks step by step before answering |

At your company you'll almost always **start from an instruct model and do SFT (+ maybe DPO) with LoRA**. Pretraining from scratch is what `mini_gpt.py` does at toy scale — so you understand it, not so you do it at work.

---

## Labs

```bash
python bpe.py                       # train a tiny BPE tokenizer, see merges and compression
python mini_gpt.py                  # train a ~1M-param GPT on your Mac (MPS) in a few minutes
python mini_gpt.py --steps 3000 --n_layer 6 --n_embd 192   # bigger
```
`mini_gpt.py` downloads Tiny Shakespeare once (falls back to the course's own text if offline), trains, prints train/val loss, saves `mini_gpt.pt` and samples text.

## Exercises
1. Before training, predict the initial loss (hint: ln(vocab_size)). Check it. Then comment out `self.apply(self._init_weights)` in `GPT.__init__` and run again: the initial loss jumps to ~80. Why? (Hint: print the std of `tok_emb.weight`, and remember it's tied to the output head.) An initial loss far above ln(V) is a classic sign of bad initialization.
2. Set `lr=1e-2`, then `1e-5`. Describe the loss curves.
3. Remove the causal mask. Train loss collapses to near zero — why is that cheating, and why does generation become garbage?
4. Train for 10× more steps on a tiny dataset (`--data` a 5 KB file). Watch val loss go up: that's overfitting.
5. In `bpe.py`, train on Vietnamese text vs English text with the same number of merges. Compare compression ratios.
6. Load a real tokenizer (`AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")`) and print `apply_chat_template` of a 2-turn chat with `tokenize=False`. Identify every special token.

## At work
Open the training script your team uses. For every hyperparameter, write one line: what it does and why it has that value. Check which chat template the model uses and confirm training + inference use the same one.

## Watch / read
- Karpathy — *Neural Networks: Zero to Hero* (micrograd, makemore 1, Let's build GPT, GPT tokenizer): https://karpathy.ai/zero-to-hero.html
- Jay Alammar — *The Illustrated Transformer*
- 3Blue1Brown — *Neural networks* series (chapters on attention and transformers)
