"""Typed CrewAI tools over the agent-browser CLI.

Two sets, matching the two agents:
- Scout (read-only): navigate, snapshot, screenshot, read_page
- Operator (write): click, fill, press, scroll, plus snapshot/verify

Why typed tools instead of one generic CLI passthrough: parameter schemas teach
the model the syntax (no flag hallucination), validation catches mistakes before
execution, the surface is exactly what we permit, and every step is a structured
event for the SSE stream and replay.

Class-based (not @tool functions) so instances can bind to a run_id at crew
assembly time without baking run_id into the LLM-visible schema.
"""

from crewai.tools import BaseTool

from app.tools import browser


class _BrowserTool(BaseTool):
    run_id: str
    # BaseTool requires a package attribute for module resolution.
    package: str = "app.tools.browser_tools"


# --- Scout (read-only) ------------------------------------------------------


class NavigateTool(_BrowserTool):
    name: str = "navigate"
    description: str = (
        "Open a URL in the browser session. Input: the full URL starting with "
        "http:// or https://. Use this to move between pages, never for in-page clicks."
    )

    def _run(self, url: str) -> str:
        url = url.strip()
        if not url.startswith(("http://", "https://")):
            return "ERROR: url must start with http:// or https://"
        return browser.exec_tool(self.run_id, ["open", url, "--wait", "networkidle"])


class SnapshotTool(_BrowserTool):
    name: str = "snapshot"
    description: str = (
        "Take an accessibility-tree snapshot of the current page: compact text with "
        "per-element refs like @e3. Reading this is how you see the page. Input: none."
    )

    def _run(self) -> str:
        return browser.exec_tool(self.run_id, ["snapshot", "-i", "--wait", "1s"])


class ScreenshotTool(_BrowserTool):
    name: str = "screenshot"
    description: str = (
        "Capture the current page to a PNG (visual check only; you receive a path, "
        "not pixels). Input: no args."
    )

    def _run(self, path: str = "") -> str:
        import hashlib

        digest = hashlib.sha1(f"{self.run_id}{path}".encode()).hexdigest()[:8]
        out = f"screenshots/{digest}.png"
        return browser.exec_tool(self.run_id, ["screenshot", out])


# --- Operator (writes + verification) ---------------------------------------


class ClickTool(_BrowserTool):
    name: str = "click"
    description: str = (
        "Click an element by its snapshot ref, e.g. @e3. Refs come from a snapshot; "
        "if the page changed since, take a fresh snapshot first. Input: ref like @e3."
    )

    def _run(self, ref: str) -> str:
        ref = ref.strip()
        if not ref.startswith("@"):
            return "ERROR: ref must look like @e3 (from the latest snapshot)"
        out = browser.exec_tool(self.run_id, ["click", ref, "--wait", "1s"])
        # Refs go stale after DOM updates; produce a fresh one immediately.
        due = browser.exec_tool(self.run_id, ["snapshot", "-i", "--wait", "1s"])
        return f"{out}\n\nSnapshot after click:\n{due}"


class FillTool(_BrowserTool):
    name: str = "fill"
    description: str = (
        "Type text into an input element by its snapshot ref (clears existing value). "
        "Input: ref like @e2, one space, then the text to type (may contain spaces)."
    )

    def _run(self, input: str) -> str:
        ref, _, text = input.strip().partition(" ")
        if not ref.startswith("@") or not text:
            return "ERROR: format is '<ref> <text>' — e.g. '@e5 flights to LAX'"
        out = browser.exec_tool(self.run_id, ["fill", ref, text])
        return f"{out}\n\nFilled {ref}." + "\nTake a fresh snapshot before relying on old refs."


class PressTool(_BrowserTool):
    name: str = "press"
    description: str = (
        "Press a keyboard key on the focused element: Enter, Tab, Escape, ArrowDown, "
        "Combination chars use plus, e.g. Control+A. Input: the key name."
    )

    def _run(self, key: str) -> str:
        return browser.exec_tool(self.run_id, ["press", key.strip(), "--wait", "1s"])


class ScrollTool(_BrowserTool):
    name: str = "scroll"
    description: str = (
        "Scroll the page. Input: 'up', 'down', 'bottom' or 'top'."
    )

    def _run(self, direction: str) -> str:
        d = direction.strip().lower()
        if d not in ("up", "down", "bottom", "top"):
            return "ERROR: direction must be up | down | bottom | top"
        return browser.exec_tool(self.run_id, ["scroll", d])


class VerifyTool(_BrowserTool):
    name: str = "verify"
    description: str = (
        "Snapshot the page and check the last action worked the way you expected. "
        "Input: no args. Call after every significant action; if reality does not "
        "match, retry once, then mark the step failed and continue."
    )

    def _run(self) -> str:
        return browser.exec_tool(self.run_id, ["snapshot", "-i", "--wait", "1s"])


def scout_tools(run_id: str) -> list[BaseTool]:
    if run_id not in browser.registry:
        browser.register(run_id)
    return [
        NavigateTool(run_id=run_id),
        SnapshotTool(run_id=run_id),
        ScreenshotTool(run_id=run_id),
    ]


def operator_tools(run_id: str) -> list[BaseTool]:
    if run_id not in browser.registry:
        browser.register(run_id)
    return [
        SnapshotTool(run_id=run_id),
        ClickTool(run_id=run_id),
        FillTool(run_id=run_id),
        PressTool(run_id=run_id),
        ScrollTool(run_id=run_id),
        VerifyTool(run_id=run_id),
    ]
