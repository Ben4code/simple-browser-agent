"""Process layer: drives the `agent-browser` CLI via subprocess.

One browser session per API run, isolated by `--session <run_id>`. The agent-browser
daemon persists browser state between commands, so Scout and Operator see the same
browser without any shared in-memory state. Explicit `close_session` in a finally
block prevents headless-browser leaks on the server.
"""

import subprocess
import threading
import time
from dataclasses import dataclass, field

from app import store, stream
from app.config import settings


class ToolError(RuntimeError):
    """Raised when a browser tool call fails or a safety cap trips."""


@dataclass
class RunHandle:
    """Per-run state: session name and step budget."""

    run_id: str
    register_time: float = field(default_factory=time.time)
    session: str = ""  # equals run_id
    browser_calls: int = 0
    thread_done: threading.Event = field(default_factory=threading.Event)


# run_id -> RunHandle
registry: dict[str, RunHandle] = {}
_registry_lock = threading.Lock()


def register(run_id: str) -> RunHandle:
    with _registry_lock:
        handle = RunHandle(run_id=run_id)
        handle.session = run_id
        registry[run_id] = handle
        return handle


def get(run_id: str) -> RunHandle:
    handle = registry.get(run_id)
    if handle is None:
        raise ToolError(f"unknown run {run_id}")
    return handle


def close(run_id: str) -> None:
    """Close the browser session and stop event streaming."""
    with _registry_lock:
        handle = registry.get(run_id)
        if handle is None:
            return
        try:
            _exec(["close", "--session", handle.session])
        except ToolError:
            pass  # session may already be gone; cleanup is best-effort
        registry.pop(run_id, None)
    stream.close(run_id)


def emit(run_id: str, kind: str, payload: dict) -> None:
    """Record a step event in the transcript and push it to the SSE stream."""
    store.record_step(run_id, kind, payload)
    stream.get_stream(run_id).push({"kind": kind, **payload})


# --- subprocess core -------------------------------------------------------


def _exec(args: list[str], timeout: int | None = None) -> str:
    """Run `agent-browser <args>` and return stdout. Raises ToolError on
    nonzero exit or timeout."""
    cmd = ["agent-browser", *args]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout or settings.cli_timeout_seconds,
        )
    except FileNotFoundError as e:
        raise ToolError(
            "agent-browser CLI not found. Install: npm install -g agent-browser"
        ) from e
    except subprocess.TimeoutExpired:
        raise ToolError(f"agent-browser timed out after {timeout}s: {' '.join(args)}")
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if proc.returncode != 0:
        raise ToolError(f"agent-browser failed ({proc.returncode}): {err or out}")
    return out or err  # some informational commands (e.g. close) print to stderr


def exec_tool(run_id: str, args: list[str], truncate: int = 40_000) -> str:
    """Budgeted variant used by all LLM-facing tools: counts calls against
    max_browser_calls, records the step, truncates oversized output."""
    handle = get(run_id)
    if handle.browser_calls >= settings.max_browser_calls:
        raise ToolError(
            f"browser call budget exhausted ({settings.max_browser_calls}); "
            "stop and report what you have."
        )
    started = time.time()
    try:
        out = _exec([*args, "--session", handle.session])
    except ToolError as e:
        emit(
            run_id,
            "tool_error",
            {"command": " ".join(args), "error": str(e), "duration_ms": int((time.time() - started) * 1000)},
        )
        # Re-raise as a soft failure the agent can read and recover from,
        # except on budget exhaustion which must actually halt the crew.
        if "budget exhausted" in str(e):
            raise
        raise ToolError(str(e)) from e
    handle.browser_calls += 1
    emit(
        run_id,
        "tool",
        {
            "command": " ".join(args),
            "duration_ms": int((time.time() - started) * 1000),
            "bytes": len(out),
        },
    )
    if len(out) > truncate:
        out = out[:truncate] + "\n… [truncated]"
    return out


def session_arg(run_id: str) -> list[str]:
    return ["--session", run_id]
