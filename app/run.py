"""Local TTY runner: drive one crew run straight from the terminal.

    uv run -m app.run "look up the pricing page" --url https://agent-browser.dev

Reuses the exact same code path as the FastAPI backend (crew + tools + store),
so a green CLI run means the server flow works too.
"""

import argparse
import time
import uuid
from queue import Empty

from app import store, stream
from app.config import settings
from app.crew import run_in_background
from app.tools import browser


def _print_event(event: dict) -> None:
    kind = event.get("kind", "?")
    ts = time.strftime("%H:%M:%S")
    if kind == "run_started":
        print(f"\n[{ts}] ▶ {event['goal']}  (start: {event['start_url']})\n")
    elif kind == "tool":
        print(f"[{ts}] ⚙ {event['command']}  ({event['duration_ms']}ms)")
    elif kind == "tool_error":
        print(f"[{ts}] ✗ {event['command']}\n      {event['error']}")
    elif kind == "run_completed":
        print(f"\n[{ts}] ✔ {event['summary']}")
        for f in event.get("follow_ups", []):
            print(f"      ⚠ follow-up needed: {f}")
    elif kind == "run_failed":
        print(f"\n[{ts}] ✖ run failed: {event['error']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one browser-agent crew locally")
    parser.add_argument("goal", help="Natural-language goal")
    parser.add_argument(
        "--url",
        dest="start_url",
        default="https://example.com",
        help="URL to begin from (default: https://example.com)",
    )
    args = parser.parse_args()

    if not settings.llm_api_key:
        parser.error("LLM_API_KEY not set — add it to .env")

    run_id = f"cli{uuid.uuid4().hex[:8]}"
    store.init()
    store.create_run(run_id, args.goal, args.start_url)
    thread = run_in_background(run_id, args.goal, args.start_url)
    run_started_at = time.time()

    events = stream.get_stream(run_id).queue
    try:
        while True:
            try:
                _print_event(events.get(timeout=1.0))
            except Empty:
                if not thread.is_alive():
                    drain = []
                    try:
                        while True:
                            drain.append(events.get_nowait())
                    except Empty:
                        pass
                    for e in drain:
                        _print_event(e)
                    break
            if thread.is_alive() and time.time() > run_started_at + settings.run_timeout_seconds:
                print("✖ run timed out")
                browser.close(run_id)
                break
    except KeyboardInterrupt:
        print("\nkeyboard interrupt — closing browser session")
        browser.close(run_id)
    finally:
        thread.join(timeout=5)
        stream.pop_stream(run_id)

    row = store.get_run(run_id)
    status = row["status"] if row else "error"
    print(f"\n[{run_id}] final status: {status}  (transcript in {settings.db_path})")
    return 0 if status in ("completed", "partial") else 1


if __name__ == "__main__":
    raise SystemExit(main())
