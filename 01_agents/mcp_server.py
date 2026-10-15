"""Lab 01-D: an MCP server exposing dataset tools to ANY MCP client (Claude Code, Claude Desktop, Agent SDK…).

Register in Claude Code:
    claude mcp add dataset-tools -- python /ABSOLUTE/PATH/course/01_agents/mcp_server.py
Then ask Claude Code: "Use dataset-tools to give me stats for course/03_datasets/out/train.jsonl"

Test locally without a client:
    python mcp_server.py --selftest
"""
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from mcp.server.mcpserver import MCPServer  # MCP Python SDK 2.x (was FastMCP in 1.x)

mcp = MCPServer("dataset-tools")


def _load(path: str) -> list[dict]:
    rows = []
    with open(Path(path).expanduser()) as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"line {i}: invalid JSON ({e})") from e
    return rows


@mcp.tool()
def dataset_stats(path: str) -> str:
    """Summary statistics for a JSONL chat dataset (one {"messages": [...]} object per line):
    row count, turns per example, role distribution, and character lengths of assistant replies.
    Use this before fine-tuning to sanity-check a dataset."""
    rows = _load(path)
    turns = [len(r.get("messages", [])) for r in rows]
    roles = Counter(m.get("role") for r in rows for m in r.get("messages", []))
    reply_lens = [len(m.get("content", "")) for r in rows for m in r.get("messages", []) if m.get("role") == "assistant"]
    return json.dumps({
        "rows": len(rows),
        "turns_per_example": {"min": min(turns, default=0), "median": statistics.median(turns) if turns else 0, "max": max(turns, default=0)},
        "roles": dict(roles),
        "assistant_chars": {"min": min(reply_lens, default=0), "median": statistics.median(reply_lens) if reply_lens else 0, "max": max(reply_lens, default=0)},
    }, indent=2)


@mcp.tool()
def sample_rows(path: str, n: int = 3, offset: int = 0) -> str:
    """Return n example rows from a JSONL dataset starting at offset, for manual inspection."""
    rows = _load(path)[offset: offset + max(1, min(n, 20))]
    return "\n\n".join(json.dumps(r, ensure_ascii=False, indent=1) for r in rows)


@mcp.resource("datasets://list")
def list_datasets() -> str:
    """JSONL files found under the course folder."""
    root = Path(__file__).resolve().parents[1]
    return "\n".join(str(p) for p in root.rglob("*.jsonl") if ".venv" not in p.parts)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sample = Path(__file__).resolve().parents[1] / "03_datasets" / "sample_raw.jsonl"
        print(list_datasets())
        print(sample_rows(str(sample), n=1))
    else:
        mcp.run()  # stdio transport by default
