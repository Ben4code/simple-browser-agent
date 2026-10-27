"""FastAPI backend: POST /runs triggers a crew run; SSE streams live events."""

import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from app import crew, store, stream
from app.models import RunOut, RunSpec
from app.tools import browser

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.init()
    yield


app = FastAPI(title="browseragent", version="0.1.0", lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.post("/runs", response_model=RunOut, status_code=201)
def create_run(spec: RunSpec):
    if not settings_ready():
        raise HTTPException(
            503,
            "LLM_API_KEY not configured — set it in .env and restart the server.",
        )
    run_id = uuid.uuid4().hex[:12]
    store.init()
    store.create_run(run_id, spec.goal, spec.start_url)
    crew.run_in_background(run_id, spec.goal, spec.start_url)
    row = store.get_run(run_id)
    # 201 with the record; clients subscribe to /runs/{id}/stream for live events.
    return record_to_out(row)


def settings_ready() -> bool:
    from app.config import settings

    return bool(settings.llm_api_key)


def record_to_out(row: dict) -> RunOut:
    return RunOut(
        id=row["id"],
        goal=row["goal"],
        start_url=row["start_url"],
        status=row["status"],
        summary=row.get("summary"),
        created_at=row["created_at"],
        finished_at=row.get("finished_at"),
    )


@app.get("/runs")
def list_runs():
    return [record_to_out(r) for r in store.list_runs()]


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    row = store.get_run(run_id)
    if row is None:
        raise HTTPException(404, "unknown run")
    return {**record_to_out(row).model_dump(), "steps": store.get_steps(run_id)}


@app.get("/runs/{run_id}/stream")
def stream_run(run_id: str):
    if store.get_run(run_id) is None:
        raise HTTPException(404, "unknown run")
    return EventSourceResponse(stream.sse_events(run_id))


@app.delete("/runs/{run_id}", status_code=204)
def cancel_run(run_id: str):
    row = store.get_run(run_id)
    if row is None:
        raise HTTPException(404, "unknown run")
    if row["status"] in ("completed", "failed", "cancelled", "error"):
        return  # already final; keep transcript intact
    # Best-effort: closes the browser session and marks the run cancelled.
    # The crew thread keeps running until its next browser call fails.
    _finalize_cancel(run_id)


def _finalize_cancel(run_id: str) -> None:
    store.set_status(run_id, "cancelled", "cancelled by request")
    stream.close(run_id)
    browser.close(run_id)
