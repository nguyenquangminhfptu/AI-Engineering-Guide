"""Offline tests for agent_loop.py — no API key, no cost, runs in about a second.

Run:  cd course && python -m pytest 01_agents/test_agent_loop.py -v

Two levels, like any software:
  1. Unit tests: the tools are plain functions → test them directly.
  2. Loop tests: replace Gemini with a FakeGemini that returns SCRIPTED replies,
     then check that the loop runs tools, sends results back correctly, handles errors and stops.
"""
import os
from types import SimpleNamespace

os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used")  # the client is created at import time

import pytest
from google.genai import types

import agent_loop as agent


def call(id, name, args):
    return types.FunctionCall(id=id, name=name, args=args)


def reply(*, calls=(), text=None):
    parts = ([types.Part(text=text)] if text else []) + [types.Part(function_call=c) for c in calls]
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=types.Content(role="model", parts=parts),
                                    finish_reason="STOP")],
        function_calls=list(calls) or None,
        text=text,
        usage_metadata=SimpleNamespace(prompt_token_count=0, candidates_token_count=0),
        prompt_feedback=None,
    )


class FakeGemini:
    def __init__(self, *replies):
        self.replies, self.requests = list(replies), []
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, **kwargs):
        self.requests.append({**kwargs, "contents": list(kwargs["contents"])})
        return self.replies.pop(0) if self.replies else reply(calls=[call("x", "list_dir", {})])


@pytest.fixture
def fake(monkeypatch):
    def install(*replies):
        client = FakeGemini(*replies)
        monkeypatch.setattr(agent, "client", client)
        return client
    return install


def responses_in(content):
    return [p.function_response for p in content.parts]


# ---------------- 1. unit tests: tools ----------------
def test_safe_path_allows_files_inside_sandbox():
    assert agent.safe_path("README.md") == agent.SANDBOX / "README.md"


@pytest.mark.parametrize("bad", ["../../.zshrc", "/etc/passwd", "01_agents/../../secret.txt"])
def test_safe_path_blocks_escape(bad):
    with pytest.raises(PermissionError):
        agent.safe_path(bad)


def test_read_file_truncates_and_says_so():
    out = agent.read_file("README.md", max_chars=50)
    assert "[truncated," in out and len(out) < 120


def test_every_schema_has_a_matching_function():
    names = {d.name for d in agent.TOOLS[0].function_declarations}
    assert names == set(agent.TOOL_FUNCS)


# ---------------- 2. loop tests with a fake Gemini ----------------


def test_loop_runs_tool_then_returns_final_answer(fake):
    client = fake(reply(calls=[call("c1", "find_files", {"pattern": "LESSON.md"})]),
                  reply(text="There are 6 lessons."))
    assert agent.run_agent("How many lessons?") == "There are 6 lessons."
    second = client.requests[1]["contents"]
    assert [c.role for c in second] == ["user", "model", "user"]   # Gemini says "model", not "assistant"
    (resp,) = responses_in(second[2])
    assert resp.id == "c1" and resp.name == "find_files"
    assert "01_agents/LESSON.md" in resp.response["result"]


def test_tool_error_goes_back_as_error_field(fake):
    client = fake(reply(calls=[call("c1", "read_file", {"path": "../../.zshrc"})]),
                  reply(text="I can't read that."))
    agent.run_agent("read my zshrc")
    (resp,) = responses_in(client.requests[1]["contents"][-1])
    assert "outside the sandbox" in resp.response["error"]


def test_parallel_calls_return_in_one_message(fake):
    client = fake(reply(calls=[call("a", "list_dir", {"path": "."}), call("b", "find_files", {"pattern": "*.py"})]),
                  reply(text="ok"))
    agent.run_agent("explore")
    assert [r.id for r in responses_in(client.requests[1]["contents"][-1])] == ["a", "b"]


def test_step_budget(fake, monkeypatch):
    monkeypatch.setattr(agent, "MAX_STEPS", 3)
    client = fake()
    assert agent.run_agent("loop forever") == "[stopped: step budget exhausted]"
    assert len(client.requests) == 3
