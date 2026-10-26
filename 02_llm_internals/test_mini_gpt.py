"""Offline checks for decoder-only behavior, training, CLI, and checkpoints."""
import sys

import pytest
import torch

import mini_gpt


@pytest.mark.parametrize("text", ["hello AI", "xin chào AI"])
def test_character_tokenizer_round_trip(text):
    tokenizer = mini_gpt.CharTokenizer(text)
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_batch_targets_are_next_tokens():
    sequence = torch.arange(100)
    inputs, targets = mini_gpt.get_batch(sequence, block_size=8, batch_size=4, device="cpu")
    assert inputs.shape == targets.shape == (4, 8)
    torch.testing.assert_close(targets, inputs + 1)


def test_future_tokens_do_not_change_prefix_logits():
    torch.manual_seed(42)
    model = mini_gpt.GPT(mini_gpt.Config(
        vocab_size=11, block_size=8, n_layer=2, n_head=2, n_embd=16, dropout=0.0,
    )).eval()
    inputs = torch.tensor([[1, 2, 3, 4, 5]])
    changed_future = torch.tensor([[1, 2, 3, 9, 10]])
    with torch.no_grad():
        original_logits, _ = model(inputs)
        changed_logits, _ = model(changed_future)
    torch.testing.assert_close(original_logits[:, :3], changed_logits[:, :3], atol=1e-6, rtol=1e-5)
    assert not torch.allclose(original_logits[:, 3:], changed_logits[:, 3:])


def test_loss_backpropagates_through_embeddings_attention_and_ffn():
    torch.manual_seed(42)
    model = mini_gpt.GPT(mini_gpt.Config(
        vocab_size=11, block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0.0,
    ))
    _, loss = model(torch.tensor([[1, 2, 3, 4]]), torch.tensor([[2, 3, 4, 5]]))
    loss.backward()
    weights = [model.tok_emb.weight, model.pos_emb.weight,
               model.blocks[0].attn.qkv.weight, model.blocks[0].mlp.up.weight]
    for weight in weights:
        assert weight.grad is not None
        assert torch.isfinite(weight.grad).all()
        assert weight.grad.abs().sum() > 0


def test_cli_trains_and_samples_text_without_newline(tmp_path, monkeypatch, capsys):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello AI. A model predicts tokens. " * 20)
    checkpoint = tmp_path / "new_output_directory" / "model.pt"
    monkeypatch.setattr(sys, "argv", [
        "mini_gpt.py", "--data", str(corpus), "--out", str(checkpoint),
        "--device", "cpu", "--steps", "1", "--log_every", "1",
        "--n_layer", "1", "--n_head", "2", "--n_embd", "16",
        "--block_size", "8", "--batch_size", "2", "--sample_tokens", "4",
    ])
    original_generate = mini_gpt.GPT.generate

    def generate_in_eval_mode(model, *args, **kwargs):
        assert not model.training  # inference must disable training-time dropout
        return original_generate(model, *args, **kwargs)

    monkeypatch.setattr(mini_gpt.GPT, "generate", generate_in_eval_mode)
    mini_gpt.main()

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restored = mini_gpt.GPT(mini_gpt.Config(**saved["config"])).eval()
    restored.load_state_dict(saved["model"])
    logits, _ = restored(torch.zeros((1, 2), dtype=torch.long))
    assert torch.isfinite(logits).all()
    output = capsys.readouterr().out
    sample = output.split("=== sample ===\n", 1)[1].removesuffix("\n")
    assert len(sample) == 5  # one starting character plus four new characters


@pytest.mark.parametrize(("text", "options", "message"), [
    ("hello world " * 100, ["--n_embd", "30", "--n_head", "4"], "must be divisible"),
    ("hello", ["--block_size", "32"], "provide more text or reduce --block_size"),
])
def test_cli_explains_invalid_setup(tmp_path, monkeypatch, capsys, text, options, message):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(text)
    monkeypatch.setattr(sys, "argv", ["mini_gpt.py", "--data", str(corpus), "--device", "cpu", *options])
    with pytest.raises(SystemExit) as error:
        mini_gpt.main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err
