"""Lab 01-B: the same agent, but the SDK runs the loop for you — Gemini's "automatic function calling".

Run:  export GEMINI_API_KEY=...
      python agent_runner.py "Which lab files use PyTorch?"

Compare with agent_loop.py: same loop, but here you only write the tool functions.
The SDK builds the schema from type hints + docstring, calls the functions, feeds results back, and loops.
"""
import sys
from pathlib import Path

from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.llm import MODEL, client  # noqa: E402

SANDBOX = Path(__file__).resolve().parents[1]


def find_files(pattern: str) -> str:
    """Recursively find files in the course project matching a glob pattern.

    Args:
        pattern: Glob such as '*.py' or 'LESSON.md'.
    """
    hits = [str(p.relative_to(SANDBOX)) for p in SANDBOX.rglob(pattern) if ".venv" not in p.parts]
    return "\n".join(sorted(hits)) or "no matches"


def grep(text: str, pattern: str = "*.py") -> str:
    """Return files (matching a glob) whose content contains the given text, with the first matching line.

    Args:
        text: Exact substring to search for, e.g. 'import torch'.
        pattern: Glob for which files to search. Defaults to '*.py'.
    """
    out = []
    for p in SANDBOX.rglob(pattern):
        if ".venv" in p.parts or not p.is_file():
            continue
        for line in p.read_text(errors="replace").splitlines():
            if text in line:
                out.append(f"{p.relative_to(SANDBOX)}: {line.strip()}")
                break
    return "\n".join(out) or "no matches"


if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) or "Which lab files use PyTorch? One line each."
    response = client.models.generate_content(
        model=MODEL,
        contents=task,
        config=types.GenerateContentConfig(
            tools=[find_files, grep],   # plain functions → the SDK generates the schemas AND runs the loop
            automatic_function_calling=types.AutomaticFunctionCallingConfig(maximum_remote_calls=10),  # budget
        ),
    )
    # The SDK keeps the intermediate turns so you can still trace what happened.
    for content in response.automatic_function_calling_history or []:
        for part in content.parts or []:
            if part.function_call:
                print(f"🔧 {part.function_call.name}({dict(part.function_call.args or {})})")
    print("\n=== FINAL ANSWER ===")
    print(response.text)
