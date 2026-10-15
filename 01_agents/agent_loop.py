"""Lab 01-A: an agent from scratch — the tool-calling loop, written by hand (Gemini API).

Run:  export GEMINI_API_KEY=...
      python agent_loop.py "How many .md files are in this course and which is the longest?"

Read this file top to bottom. Everything an "agent framework" does is here:
tool schemas → loop → execute tools → feed results back → stop.
"""
import sys
from pathlib import Path

from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.llm import MODEL, client  # noqa: E402

SANDBOX = Path(__file__).resolve().parents[1]  # the agent may only touch files inside course/
MAX_STEPS = 15


def safe_path(p: str) -> Path:
    """Least privilege: refuse anything outside the sandbox (blocks ../../etc/passwd)."""
    full = (SANDBOX / p).resolve()
    if not full.is_relative_to(SANDBOX):
        raise PermissionError(f"{p} is outside the sandbox")
    return full


# ---------- 1. Tools: plain Python functions ----------
def list_dir(path: str = ".") -> str:
    d = safe_path(path)
    return "\n".join(sorted(f"{x.name}/" if x.is_dir() else x.name for x in d.iterdir() if not x.name.startswith(".")))


def read_file(path: str, max_chars: int = 4000) -> str:
    text = safe_path(path).read_text(errors="replace")
    if len(text) > max_chars:  # don't flood the context window — and tell the model you truncated
        return text[:max_chars] + f"\n...[truncated, {len(text)} chars total]"
    return text


def find_files(pattern: str) -> str:
    hits = [str(p.relative_to(SANDBOX)) for p in SANDBOX.rglob(pattern) if ".venv" not in p.parts]
    return "\n".join(sorted(hits)) or "no matches"


TOOL_FUNCS = {"list_dir": list_dir, "read_file": read_file, "find_files": find_files}

# ---------- 2. Tool schemas: this is ALL the model knows about your tools ----------
TOOLS = [types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="list_dir",
        description="List files and folders in a directory of the course project. Folders end with '/'. "
                    "Use this to explore structure before reading files.",
        parameters_json_schema={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Directory relative to the project root, e.g. '.' or '01_agents'"}},
            "required": [],
        },
    ),
    types.FunctionDeclaration(
        name="read_file",
        description="Read a text file from the course project. Long files are truncated (the result says so).",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to the project root"},
                "max_chars": {"type": "integer", "description": "Max characters to return (default 4000)"},
            },
            "required": ["path"],
        },
    ),
    types.FunctionDeclaration(
        name="find_files",
        description="Recursively find files matching a glob pattern such as '*.py' or 'LESSON.md'. "
                    "Returns paths relative to the project root.",
        parameters_json_schema={
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
        },
    ),
])]

SYSTEM = (
    "You are a helpful assistant that answers questions about a local course project using the tools provided. "
    "Explore with tools instead of guessing. When you have enough information, answer concisely and cite file paths."
)

CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM,
    tools=TOOLS,
    # The SDK can run tools automatically (see agent_runner.py). Turn that OFF: we write the loop ourselves.
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)


# ---------- 3. The loop ----------
def run_agent(task: str) -> str:
    contents = [types.Content(role="user", parts=[types.Part(text=task)])]
    for step in range(1, MAX_STEPS + 1):  # budget: never loop forever
        response = client.models.generate_content(model=MODEL, contents=contents, config=CONFIG)

        if not response.candidates:  # blocked by a safety filter before any output
            return f"[blocked] {response.prompt_feedback}"
        candidate = response.candidates[0]
        usage = response.usage_metadata
        print(f"\n--- step {step} · finish_reason={candidate.finish_reason} · "
              f"in={usage.prompt_token_count} out={usage.candidates_token_count} tokens")

        # Always append the model's FULL content (text + function calls + hidden thought signatures),
        # not just the text. Gemini rejects the next request if parts of it are missing.
        contents.append(candidate.content)

        calls = response.function_calls or []
        if not calls:  # no function calls → the model is done
            return response.text or ""

        parts = []
        for call in calls:
            args = dict(call.args or {})
            print(f"  🔧 {call.name}({args})")  # TRACE: log every call
            try:
                output = TOOL_FUNCS[call.name](**args)
                result = {"result": output}
                print(f"     ↳ {output[:150]!r}")
            except Exception as e:  # errors go BACK to the model so it can recover
                result = {"error": str(e)}
                print(f"     ↳ ERROR {e}")
            # `id` links this result to the call it answers
            parts.append(types.Part(function_response=types.FunctionResponse(id=call.id, name=call.name, response=result)))
        # All results from this turn go back in ONE message.
        contents.append(types.Content(role="user", parts=parts))

    return "[stopped: step budget exhausted]"


if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) or "What modules does this course have, and what does module 02 teach? Be brief."
    print("\n=== FINAL ANSWER ===\n" + run_agent(task))
