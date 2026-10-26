"""Try a saved mini GPT: text continuation, next-token probabilities, and validation metrics.

Run from course/:
    python 02_llm_internals/probe_model.py --checkpoint path/to/model.pt --prompt "hello "
    python 02_llm_internals/probe_model.py --checkpoint path/to/model.pt
    python 02_llm_internals/probe_model.py --checkpoint path/to/model.pt --eval-text path/to/corpus.txt
"""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from mini_gpt import Config, GPT


def load_model(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    chars = saved["chars"]
    model = GPT(Config(**saved["config"])).eval()
    model.load_state_dict(saved["model"])
    if len(chars) != model.config.vocab_size or len(set(chars)) != len(chars):
        raise ValueError("Vocabulary không khớp checkpoint.")
    return model, chars


def encode_known(text, chars):
    """Reject unknown characters instead of silently dropping them."""
    missing = sorted(set(text) - set(chars))
    if missing:
        raise ValueError(f"Ký tự ngoài vocabulary: {missing!r}. Dùng ký tự có trong corpus đã train.")
    stoi = {char: i for i, char in enumerate(chars)}
    return torch.tensor([stoi[char] for char in text], dtype=torch.long)


@torch.no_grad()
def continue_text(model, chars, prompt, new_tokens=80, temperature=0.8, top_k=None, seed=42):
    if not prompt:
        raise ValueError("Prompt cần ít nhất một ký tự.")
    if new_tokens < 0 or temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("Số token mới phải không âm, temperature phải hữu hạn và lớn hơn 0.")
    if top_k is not None and not 1 <= top_k <= len(chars):
        raise ValueError(f"top-k phải từ 1 đến {len(chars)}.")
    inputs = encode_known(prompt, chars).unsqueeze(0)
    model.eval()
    logits, _ = model(inputs[:, -model.config.block_size:])
    probabilities = F.softmax(logits[0, -1], dim=-1)
    values, indices = probabilities.topk(min(5, len(chars)))
    next_tokens = [{"token": chars[i], "probability": float(p)} for p, i in zip(values.tolist(), indices.tolist())]
    torch.manual_seed(seed)
    generated = model.generate(inputs, new_tokens, temperature=temperature, top_k=top_k)
    return {
        "prompt": prompt,
        "text": "".join(chars[i] for i in generated[0].tolist()),
        "new_tokens": new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "seed": seed,
        "next_tokens_before_sampling_filters": next_tokens,
    }


@torch.no_grad()
def evaluate(model, ids):
    """Score each next-token target once, using up to block_size preceding tokens."""
    if len(ids) < 2:
        raise ValueError("Đoạn đánh giá cần ít nhất hai token.")
    model.eval()
    total_loss = 0.0
    correct = 0
    top_correct = 0
    k = min(5, model.config.vocab_size)
    for position in range(1, len(ids)):
        context = ids[max(0, position - model.config.block_size):position].unsqueeze(0)
        logits, _ = model(context)
        next_logits = logits[:, -1, :]
        if not torch.isfinite(next_logits).all():
            raise ValueError("Model trả logits không hữu hạn.")
        target = ids[position:position + 1]
        total_loss += F.cross_entropy(next_logits, target).item()
        correct += int(next_logits.argmax(dim=-1).item() == target.item())
        top_correct += int(target.item() in next_logits.topk(k, dim=-1).indices[0].tolist())
    count = len(ids) - 1
    loss = total_loss / count
    return {
        "tokens_scored": count,
        "cross_entropy": loss,
        "perplexity": math.exp(loss),
        "top1_accuracy": correct / count,
        "top_k": k,
        "top_k_accuracy": top_correct / count,
    }


def print_generation(result):
    print("\n5 ký tự dự đoán cao nhất (phân phối gốc, trước temperature/top-k):")
    for item in result["next_tokens_before_sampling_filters"]:
        print(f"  {item['token']!r:8s} {item['probability']:7.2%}")
    print("\n=== Văn bản: prompt + phần sinh thêm ===")
    print(result["text"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", help="Đoạn để model viết tiếp; bỏ qua để nhập tương tác.")
    parser.add_argument("--new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-text", type=Path, help="Corpus UTF-8 để đo khả năng dự đoán ký tự.")
    parser.add_argument("--split", choices=("val", "all"), default="val",
                        help="val: 10%% cuối corpus gốc; all: toàn bộ file độc lập (mặc định val).")
    parser.add_argument("--out", type=Path, help="Lưu prompt, mẫu sinh và metrics vào JSON.")
    args = parser.parse_args()
    if args.new_tokens < 0 or args.temperature <= 0 or not math.isfinite(args.temperature):
        parser.error("require --new-tokens >= 0 and finite --temperature > 0")
    try:
        model, chars = load_model(args.checkpoint)
        if args.top_k is not None and not 1 <= args.top_k <= len(chars):
            parser.error(f"--top-k phải từ 1 đến {len(chars)}")
        report = {"checkpoint": str(args.checkpoint), "config": model.config.__dict__,
                  "parameters": sum(p.numel() for p in model.parameters()), "generations": []}
        print(f"Loaded {report['parameters']:,} parameters | vocab={len(chars)} | context={model.config.block_size} | CPU")
        if args.eval_text:
            text = args.eval_text.read_text(encoding="utf-8")
            selected = text[int(0.9 * len(text)):] if args.split == "val" else text
            metrics = evaluate(model, encode_known(selected, chars))
            metrics.update({"source": str(args.eval_text), "split": args.split,
                            "uniform_loss": math.log(len(chars)), "uniform_perplexity": len(chars)})
            report["evaluation"] = metrics
            print(f"\nScored {metrics['tokens_scored']} next-character targets ({args.split})")
            print(f"Cross-entropy: {metrics['cross_entropy']:.4f} | Perplexity: {metrics['perplexity']:.3f}")
            print(f"Top-1: {metrics['top1_accuracy']:.2%} | Top-{metrics['top_k']}: {metrics['top_k_accuracy']:.2%}")
            print(f"Uniform baseline: loss={metrics['uniform_loss']:.4f}, perplexity={len(chars)}")
        if args.prompt is not None:
            result = continue_text(model, chars, args.prompt, args.new_tokens, args.temperature, args.top_k, args.seed)
            report["generations"].append(result)
            print_generation(result)
        elif not args.eval_text:
            print("\nNhập đoạn để model viết tiếp; /quit để thoát. Ví dụ: hello hoặc the model")
            while True:
                try:
                    prompt = input("\nPrompt> ")
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if prompt.strip() == "/quit":
                    break
                try:
                    result = continue_text(model, chars, prompt, args.new_tokens, args.temperature, args.top_k, args.seed)
                    report["generations"].append(result)
                    print_generation(result)
                except ValueError as error:
                    print(error)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print("\nSaved:", args.out)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
