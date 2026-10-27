"""Pydantic schemas shared by the crew, tools, API, and store."""

from enum import Enum

from pydantic import BaseModel, Field


class Checklist(BaseModel):
    """Scout output: a semantic, ref-free checklist the Operator will resolve
    against fresh snapshots. Steps must describe page anchors in words, not
    element refs, because refs go stale as the DOM changes."""

    title: str = Field(description="Short name for the task")
    goal_summary: str = Field(description="Restatement of the user goal in one sentence")
    steps: list[str] = Field(
        description=(
            "Ordered imperative steps phrased against what is on the page, e.g. "
            "'fill the search box labelled Quick find with <value>'. No element refs."
        )
    )
    observations: list[str] = Field(
        default_factory=list,
        description="Anything unusual seen during exploration (popups, login walls, CAPTCHAs)",
    )
    risks: list[str] = Field(
        default_factory=list,
        description="Steps likely to fail or need care (timeouts, dynamic content)",
    )


class StepStatus(str, Enum):
    ok = "ok"
    failed = "failed"
    skipped = "skipped"


class StepOutcome(BaseModel):
    index: int
    instruction: str
    status: StepStatus
    detail: str = ""


class RunResultStatus(str, Enum):
    completed = "completed"
    partial = "partial"  # some steps failed but the run is reported
    failed = "failed"
    cancelled = "cancelled"
    timed_out = "timed_out"
    error = "error"  # framework/LLM error, not step failure


class OperatorReport(BaseModel):
    """Final output of the Operator agent."""

    status: RunResultStatus
    summary: str = Field(description="Plain-language account of what happened")
    outcomes: list[StepOutcome] = Field(default_factory=list)
    follow_ups: list[str] = Field(
        default_factory=list,
        description="Manual actions the user must take (logins, CAPTCHAs, confirmations)",
    )


class RunSpec(BaseModel):
    goal: str = Field(min_length=1, description="Natural-language goal")
    start_url: str = Field(description="URL to begin navigation from")


class RunOut(BaseModel):
    id: str
    goal: str
    start_url: str
    status: RunResultStatus
    summary: str | None = None
    created_at: str
    finished_at: str | None = None
