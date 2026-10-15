# Module 01 — Agents, Claude Skills & MCP (Weeks 1–2)

**Goal:** be able to design, build and *measure* an agent, a Claude skill and an MCP server — and know when NOT to build an agent.

---

## 1. The ladder of LLM systems

Always pick the lowest rung that solves the problem. Each rung up costs more, is slower and fails in more ways.

| Rung | What it is | Who decides the steps | Example |
|---|---|---|---|
| 1. Single call | prompt → answer | you | classify a ticket, extract fields |
| 2. Workflow | fixed chain of calls in code | **your code** | draft → critique → rewrite |
| 3. Agent | LLM in a loop, picks tools itself | **the model** | "fix the failing test in this repo" |

Use an agent only when (a) the steps can't be known in advance, (b) the task is worth the cost, (c) mistakes can be caught (tests, review, rollback).

## 2. What an agent actually is (it's ~30 lines)

```
messages = [user task]
loop:
    response = LLM(messages, tools)
    if response has no tool calls: return response.text      # done
    for each tool call: result = run_tool(name, input)        # YOU run it, not the model
    messages += [assistant response, tool results]
```

Key facts:
- The model **never runs anything**. It returns a *function call* (name + JSON arguments). Your code executes it and sends back a *function response*. (Gemini calls these `function_call` / `function_response`; Claude calls them `tool_use` / `tool_result` — same idea.)
- The model only knows a tool through its **name, description and input_schema**. Tool descriptions are prompts — write them like docs for a new teammate: what it does, when to use it, what it returns, edge cases.
- Return errors as results (`is_error: true`) instead of crashing. The model can recover: "file not found → 👉 Lab: [agent_loop.py](agent_loop.py) builds this loop by hand. [agent_runner.py](agent_runner.py) does the same with the SDK's automatic function calling (what you'd use in practice). [test_agent_loop.py](test_agent_loop.py) tests the loop offline with a fake model.ilds this loop by hand. [agent_runner.py](agent_runner.py) does the same with the SDK's tool runner (what you'd use in practice).

## 3. Workflow patterns (from Anthropic's "Building effective agents")

| Pattern | Shape | Use when |
|---|---|---|
| **Prompt chaining** | A → B → C, with checks between | Task splits cleanly into steps |
| **Routing** | classify → send to specialist prompt/model | Distinct input categories |
| **Parallelization** | N calls at once → combine (sectioning or voting) | Independent subtasks, or need confidence |
| **Orchestrator–workers** | LLM plans subtasks → workers do them → synthesize | Subtasks not known in advance |
| **Evaluator–optimizer** | generate → critique → revise loop | Clear quality criteria exist |

👉 Lab: [workflow_patterns.py](workflow_patterns.py) implements chaining, routing and evaluator–optimizer.

## 4. Context engineering — the real skill

An agent's quality is mostly determined by what's in its context window at each step.

- **System prompt**: role, goal, constraints, what "done" means. Explain *why* behind rules; modern models follow reasons better than SHOUTED RULES.
- **Tools**: few, well-named, non-overlapping. 5 sharp tools beat 20 fuzzy ones.
- **Tool results**: return what the model needs, not everything. Truncate huge outputs and say you truncated.
- **Long tasks**: context fills up. Options: clear old tool results, compact (summarize) history, or give the agent a memory/notes file.
- **Subagents**: delegate a reading-heavy subtask to a fresh agent that returns only a summary → keeps the main context clean.

## 5. Claude Agent Skills

A **skill** is a folder that teaches Claude a procedure, loaded only when relevant:

```
my-skill/
├── SKILL.md          # required: YAML frontmatter (name, description) + instructions
├── scripts/          # optional: code Claude can run (deterministic work)
└── references/       # optional: docs Claude reads only if needed
```

**Progressive disclosure** is the core idea:
1. Only `name` + `description` of every skill sit in context (cheap).
2. When a task matches the description, Claude reads `SKILL.md`.
3. `SKILL.md` points to scripts/references that are read/run only when needed.

So: the **description decides whether the skill triggers** — make it specific about *when* to use it ("Use when the user asks to validate or clean a JSONL chat dataset for fine-tuning…"). Put deterministic steps in scripts (cheaper, reliable), judgment in instructions.

Where skills run: Claude Code (`~/.claude/skills/` or `.claude/skills/` in a repo), claude.ai, the Claude Agent SDK, and the API (via the Skills API + code execution).

👉 Lab: [skills/sft-dataset-checker/](skills/sft-dataset-checker/) — a real skill for your job: validates fine-tuning datasets. Copy it into `.claude/skills/` and ask Claude Code "check my dataset data/train.jsonl".

## 6. MCP (Model Context Protocol)

MCP is a standard plug: write a tool server **once**, use it from Claude Code, Claude Desktop, the Agent SDK, other IDEs, other agents.

- **Server** exposes: `tools` (actions), `resources` (readable data), `prompts` (templates).
- **Transport**: `stdio` (local process) or HTTP (remote, needs auth).
- **Skill vs MCP**: MCP = *capability* (access to a system: DB, Jira, internal API). Skill = *know-how* (procedure for using capabilities well). They combine: a skill can say "use the `query_db` MCP tool, then…".

👉 Lab: [mcp_server.py](mcp_server.py) — exposes dataset stats tools. Register in Claude Code with:
```bash
claude mcp add dataset-tools -- python /absolute/path/to/course/01_agents/mcp_server.py
```

## 7. Claude Agent SDK vs Tool Runner vs Managed Agents

| | You write | Harness (loop) | Hosting | Built-in tools |
|---|---|---|---|---|
| Manual loop (API) | everything | you | you | none |
| SDK auto tool-calling (Gemini `automatic_function_calling`, Anthropic Tool Runner) | tool functions | SDK | you | none |
| **Claude Agent SDK** (`claude-agent-sdk`) | prompt + options | Claude Code's harness | you | Read/Write/Edit/Bash/Grep/Web + MCP + subagents + skills |
| Managed Agents (API) | config | Anthropic | Anthropic sandbox | bash, files, code exec, skills, MCP |

Rule of thumb at work: custom business tools → Tool Runner. Coding/file-system agent → Claude Agent SDK. Long-running hosted/scheduled agent → Managed Agents.

## 8. Safety for agents (non-negotiable at work)

- **Least privilege**: give read-only tools first. Write/delete/send tools need explicit human approval.
- **Sandbox** anything that executes code or shell (container, restricted dir — see `safe_path()` in the lab).
- **Prompt injection**: any text the agent reads (web pages, files, emails, tool output) may contain instructions. Treat it as data. Never let tool output grant new permissions.
- **Budgets**: max iterations, max tokens, timeouts.
- **Trace everything**: log every tool call + result. You can't debug what you can't see.

## 9. Measuring agents

An agent without an eval is a demo. Minimum:
1. 20–50 realistic tasks with a checkable outcome (final answer, file state, test passes).
2. Run the agent on all of them → pass rate, avg tool calls, avg cost, avg latency.
3. Read the **failed traces**. Classify failures (wrong tool, bad args, gave up, hallucinated). Fix the biggest bucket. Re-run.

Module 05 gives you the harness.

---

## Exercises

1. Run `agent_loop.py`. Add a 4th tool `count_lines(path)`. Ask a question that needs 3 tools. Read the trace.
2. Make a tool fail on purpose (bad path). Watch how the model recovers from `is_error`.
3. Write a *bad* tool description (vague) and a good one. Compare behavior on 5 questions.
4. In `workflow_patterns.py`, add a 4th route to the router and test it.
5. Install the skill in Claude Code. Write 3 prompts that SHOULD trigger it and 3 that should NOT. Tune the description until 6/6.
6. Register the MCP server in Claude Code and ask it about a dataset.

## At work (this week)
Pick one repeated task in your team. Decide: single call, workflow or agent? Build it as a skill (+ MCP if it needs system access). Write 20 test cases and record the pass rate **before** and **after** your improvements.

## Read
- Anthropic — *Building effective agents*: https://www.anthropic.com/engineering/building-effective-agents
- Anthropic — *Writing tools for agents* & *Effective context engineering* (engineering blog)
- Agent Skills docs: https://docs.claude.com/en/docs/agents-and-tools/agent-skills/overview
- Claude Agent SDK: https://code.claude.com/docs/en/agent-sdk
- MCP: https://modelcontextprotocol.io/introduction
