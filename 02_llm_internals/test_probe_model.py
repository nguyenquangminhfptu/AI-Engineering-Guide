"""Checks for truthful next-token evaluation and saved-model probing."""
import math
import sys
from types import SimpleNamespace

import pytest
import torch

import probe_model
from mini_gpt import Config, GPT


def test_uniform_predictions_have_expected_loss_and_never_see_the_target():
    class UniformModel:
        config = SimpleNamespace(block_size=2, vocab_size=3)

        def __init__(self):
            self.contexts = []

        def eval(self):
            return self

        def __call__(self, context):
            self.contexts.append(context[0].tolist())
            return torch.zeros(1, context.shape[1], 3), None

    model = UniformModel()
    result = probe_model.evaluate(model, torch.tensor([0, 1, 0, 2]))
    assert model.contexts == [[0], [0, 1], [1, 0]]
    assert result["tokens_scored"] == 3
    assert result["cross_entropy"] == pytest.approx(math.log(3))
    assert result["perplexity"] == pytest.approx(3)
    assert result["top1_accuracy"] == pytest.approx(1 / 3)
    assert result["top_k_accuracy"] == 1


def test_unknown_characters_are_not_silently_dropped():
    with pytest.raises(ValueError, match="ngoài vocabulary"):
        probe_model.encode_known("hello?", list("helo"))


def test_cli_loads_checkpoint_and_records_a_reproducible_continuation(tmp_path, monkeypatch):
    import json

    chars = sorted(set("hello AI. "))
    model = GPT(Config(vocab_size=len(chars), block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0.0))
    checkpoint = tmp_path / "model.pt"
    torch.save({"model": model.state_dict(), "config": model.config.__dict__, "chars": chars}, checkpoint)
    output = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["probe_model.py", "--checkpoint", str(checkpoint),
                                    "--prompt", "hello ", "--new-tokens", "8", "--out", str(output)])
    probe_model.main()
    first = json.loads(output.read_text())["generations"][0]
    probe_model.main()
    second = json.loads(output.read_text())["generations"][0]
    assert first == second
    assert first["text"].startswith("hello ")
    assert len(first["text"]) == len("hello ") + 8
    assert sum(item["probability"] for item in first["next_tokens_before_sampling_filters"]) <= 1.000001
