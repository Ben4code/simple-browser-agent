"""Crew assembly and background run orchestration.

Two agents, sequential process (the browser session is a single serial resource):
- Scout: read-only exploration -> Checklist (semantic, ref-free, Pydantic-validated)
- Operator: executes checklist steps against fresh snapshots -> OperatorReport
"""

import threading
import traceback

from crewai import Agent, Crew, LLM, Process, Task

from app import store, stream
from app.config import settings
from app.models import Checklist, OperatorReport, RunResultStatus
from app.tools import browser
from app.tools.browser_tools import operator_tools, scout_tools


def _llm() -> LLM:
    return LLM(**settings.llm_kwargs())


def _scout(run_id: str) -> Agent:
    return Agent(
        role="Scout",
        goal=(
            "Turn the user's natural-language goal into a precise, ordered "
            "checklist of browser actions grounded in what the target page actually contains."
        ),
        backstory=(
            "You are an expert web-map reader. You explore pages via the snapshot "
            "tool (compact accessibility trees), never guess structure, and write "
            "instructions a less capable colleague can follow verbatim."
        ),
        tools=scout_tools(run_id),
        llm=_llm(),
        max_iter=settings.scout_max_iter,
        allow_delegation=False,
        verbose=False,
    )


def _operator(run_id: str) -> Agent:
    return Agent(
        role="Operator",
        goal="Execute the Scout's checklist step by step, verifying each action.",
        backstory=(
            "You are a meticulous browser operator. You resolve each semantic "
            "instruction to concrete element refs from a FRESH snapshot before "
            "acting, verify after every significant action, never push through a "
            "mismatch more than once, and report honestly failures and manual "
            "blockers (logins, CAPTCHAs, unusual confirmations)."
        ),
        tools=operator_tools(run_id),
        llm=_llm(),
        max_iter=settings.operator_max_iter,
        allow_delegation=False,
        verbose=False,
    )


def build_crew(run_id: str, goal: str, start_url: str) -> Crew:
    scout = _scout(run_id)
    operator = _operator(run_id)

    plan_task = Task(
        description=(
            "A user wants to accomplish this on the web:\n"
            "\"\"\"\n{goal}\n\"\"\"\n"
            "Begin at: {start_url}\n\n"
            "1. Navigate to the start URL and snapshot the page.\n"
            "2. Explore ONLY as much as needed to make the checklist concrete "
            "(typically one or two pages; at most three before committing).\n"
            f"3. Produce a Checklist: ordered imperative steps with the actual "
            f"value(s) to enter baked in (search terms, dates, filters). "
            f"Steps must NOT use element refs (@eN) — describe page anchors in "
            f"words instead, because refs go stale. Note popups, login walls, "
            f"or CAPTCHAs under observations, and fragile-looking steps under risks."
        ),
        expected_output=(
            f"A Checklist object (JSON) with title, goal_summary, steps, observations, risks."
        ),
        agent=scout,
        output_pydantic=Checklist,
    )

    execute_task = Task(
        description=(
            "A Scout produced the checklist below (it is appended as context).\n"
            "Execute it step by step against the live browser session:\n"
            "- The browser is already open at the start page.\n"
            "- For each step: take a fresh snapshot, resolve the semantic anchor "
            "to a ref, act, then verify with `verify`.\n"
            "- On a mismatch, retry ONCE with a fresh snapshot; still wrong -> mark "
            f"the step failed and move on.\n"
            "- If you hit a login wall, CAPTCHA, or payment confirmation, stop \n"
            f"that step, log it under follow_ups for the user, and continue if the \n"
            f"rest is still reachable.\n"
            f"Finish with an OperatorReport covering every step."
        ),
        expected_output=(
            f"An OperatorReport object (JSON) with status, summary, per-step outcomes, follow_ups."
        ),
        agent=operator,
        context=[plan_task],
        output_pydantic=OperatorReport,
    )

    return Crew(
        agents=[scout, operator],
        tasks=[plan_task, execute_task],
        process=Process.sequential,
        verbose=False,
    )


def run_in_background(run_id: str, goal: str, start_url: str) -> threading.Thread:
    """Spawn the crew on a thread; FastAPI returns immediately."""
    store.set_status(run_id, "running")
    stream.get_stream(run_id).push({"kind": "run_started", "goal": goal, "start_url": start_url})

    def target() -> None:
        handle = None
        try:
            handle = browser.register(run_id)
            crew = build_crew(run_id, goal, start_url)
            result = crew.kickoff(inputs={"goal": goal, "start_url": start_url})
            # kickoff returns the final (execute) Task output; prefer its
            # structured pydantic model.
            raw = getattr(result, "pydantic", None)
            report: OperatorReport | None = None
            if isinstance(raw, OperatorReport):
                report = raw
            elif raw is not None:
                try:
                    report = OperatorReport.model_validate(raw.model_dump())
                except Exception:
                    report = None
            if report is None:
                report = OperatorReport(
                    status=RunResultStatus.completed,
                    summary="Crew finished but returned no structured report.",
                )
            store.set_status(run_id, report.status.value, report.summary)
            browser.emit(
                run_id,
                "run_completed",
                {"summary": report.summary, "follow_ups": report.follow_ups},
            )
        except Exception as e:
            status = (
                RunResultStatus.cancelled.value
                if isinstance(e, KeyboardInterrupt)
                else RunResultStatus.error.value
            )
            store.set_status(run_id, status, str(e)[:500])
            browser.emit(
                run_id,
                "run_failed",
                {"error": str(e)[:500], "trace": traceback.format_exc()[:2000]},
            )
        finally:
            browser.close(run_id)
            if handle is not None:
                handle.thread_done.set()

    thread = threading.Thread(target=target, name=f"run-{run_id[:8]}", daemon=True)
    thread.start()
    return thread
