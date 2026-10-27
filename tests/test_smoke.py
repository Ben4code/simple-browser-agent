"""Smoke tests that decide whether the stack is safe to build on.

Layer 1 (offline): agent-browser CLI round-trip through our subprocess wrapper.
Layer 2 (needs LLM_API_KEY): tool-calling through CrewAI against an
openai-compatible endpoint — the single most important unknown.
"""

import os
import time

import pytest

from app.tools import browser


# --- Layer 1: browser CLI through the wrapper -------------------------------


@pytest.mark.timeout(120)
def test_browser_cli_roundtrip():
    out = browser._exec(["open", "https://example.com", "--wait", "networkidle"])
    assert "example" in out.lower()
    snap = browser._exec(["snapshot", "-i", "--wait", "1s"])
    assert "@e" in snap or "ref=e" in snap, "snapshot must list element refs"
    browser._exec(["close"])


@pytest.mark.timeout(60)
def test_session_isolation():
    browser.register("smoke-run")
    try:
        browser._exec(["open", "https://example.com", "--wait", "1s"])
        browser.close("smoke-run")  # must not raise even after manual close
    finally:
        browser.close("smoke-run")


# --- Layer 2: GLM tool-calling through CrewAI --------------------------------

has_llm_key = bool(os.environ.get("LLM_API_KEY", "").strip())


@pytest.mark.skipif(not has_llm_key, reason="LLM_API_KEY not set")
@pytest.mark.timeout(180)
def test_llm_single_tool_roundtrip():
    """One agent, one bound tool, one call. Passes iff the model reliably
    emits valid OpenAI-format tool calls through the compatible endpoint."""
    from crewai import Agent, Crew, LLM, Process, Task

    from app.config import settings
    from app.tools.browser_tools import NavigateTool

    seen = {}

    class ProbedNavigate(NavigateTool):
        def _run(self, url: str) -> str:
            seen["url"] = url
            return "navigated ok"

    agent = Agent(
        role="tester",
        goal="Navigate to the target page.",
        backstory="You navigate web pages when asked.",
        tools=[ProbedNavigate(run_id="smoke-glm-test")],
        llm=LLM(**settings.llm_kwargs()),
        max_iter=4,
        allow_delegation=False,
    )
    Crew(
        agents=[agent],
        tasks=[
            Task(
                description="Navigate the browser to https://example.com",
                expected_output="Confirmation of navigation.",
                agent=agent,
            )
        ],
        process=Process.sequential,
    ).kickoff()

    deadline = time.time() + 5
    while "url" not in seen and time.time() < deadline:
        time.sleep(0.5)
    assert "url" in seen, f"tool was never called; seen={seen}"
    assert "example.com" in seen["url"]
