# browseragent

Two-agent web automation: a **Scout** explores pages with [agent-browser](https://agent-browser.dev)
and turns a natural-language goal into a semantic checklist, and an **Operator**
executes the checklist step by step against the live browser. Orchestration is
[CrewAI](https://docs.crewai.com), the backend is FastAPI, and the LLM is GLM
(via Z.ai's OpenAI-compatible endpoint).

## Architecture

```
NL goal + URL
      │
      ▼
POST /runs            FastAPI (app/main.py)
      │ spawns thread
      ▼
Crew, sequential          (app/crew.py)
├── Scout            read-only tools: navigate, snapshot, screenshot
│     └─ output: Checklist {steps[], observations[], risks}   # no refs, semantic
└── Operator         write tools: click, fill, press, scroll, verify
      └─ output: OperatorReport {status, outcomes[], follow_ups[]}
      │
      ▼  every tool call
agent-browser CLI  ──daemon──▶ Chrome   (app/tools/browser.py)
      │  --session <run_id>
      ▼
SQLite transcript + SSE stream   (app/store.py, app/stream.py → GET /runs/{id}/stream)
```

Key design points:
- **Ref lifecycle**: accessibility-tree refs (`@e3`) go stale when the DOM
  changes. The Scout deliberately never emits refs — only semantic steps
  ("fill the search box labelled Quick find"). The Operator resolves each to a
  fresh-snapshot ref at execution time, and `click` auto-snapshots after acting.
- **Budgets**: `MAX_BROWSER_CALLS` hard-stops runaway loops mid-run.
- **Isolation**: one browser session per API run; concurrent runs don't collide.
- **Recovery**: the Operator verifies after every significant action and retries
  a mismatch exactly once before reporting the step failed.

## CLI (no server)

```bash
uv run -m app.run "look up the pricing page" --url https://agent-browser.dev
```

Same crew/tools/store as the API, streamed to your terminal.

## Web UI

```bash
uv run uvicorn app.main:app --reload   # then open http://localhost:8000
```

## Valid model ids

`LLM_MODEL` must be an **exact gateway id** (lowercase-hyphenated). The
authoritative list, via the key in `.env`:

```bash
KEY=$(grep '^LLM_API_KEY=' .env | cut -d= -f2-)
curl -s -H "Authorization: Bearer $KEY" https://opencode.ai/inference/v1/models
```

Error-code cheat sheet for `/chat/completions`:

| HTTP | error.type | meaning |
|---|---|---|
| 401 | auth | wrong/expired key |
| 429 | FreeUsageLimitError | workspace free quota exhausted — wait or subscribe |
| 402 | server_error | *Insufficient account funds* — paid model, no balance |
| 400 | server_error | *Model is unavailable* — **invalid model id** (check the list) |
| 403 | server_error | *Model access is disabled* — that model is off on your plan |

## Setup

```bash
# 1. CLI + bundled Chrome
npm install -g agent-browser
agent-browser install

# 2. Python env
uv sync
cp .env.example .env     # add your Z.ai API key

# 3. Run the API
uv run uvicorn app.main:app --reload
```

## Usage

```bash
# Start a run
curl -X POST localhost:8000/runs \
  -H 'content-type: application/json' \
  -d '{"goal": "look up the pricing page", "start_url": "https://agent-browser.dev"}'

# Watch it live (SSE)
curl -N localhost:8000/runs/<id>/stream

# Inspect the transcript afterwards
curl localhost:8000/runs/<id>
```

## Tests

```bash
uv run pytest           # layer 1 runs offline; layer 2 needs LLM_API_KEY in .env
```

Layer 1 verifies the agent-browser CLI round-trip through our subprocess wrapper
(session isolation included). Layer 2 is the critical tool-calling probe: one
agent, one tool, one call through the Z.ai endpoint — if it fails, no CrewAI
feature in this codebase is trustworthy yet, fix that first.

## Config knobs

All in `.env` (see `.env.example`): model id/base URL, temperature, LLM retries,
browser-call budget, CLI timeout, per-agent `max_iter`, DB path.
