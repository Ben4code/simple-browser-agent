"""SQLite transcript store: runs + per-step events."""

import json
import sqlite3
import threading
from datetime import datetime, timezone

from app.config import settings

_local = threading.local()


def _db() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(settings.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        _local.conn = conn
    return conn


def init() -> None:
    db = _db()
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY,
            goal TEXT NOT NULL,
            start_url TEXT NOT NULL,
            status TEXT NOT NULL,
            summary TEXT,
            created_at TEXT NOT NULL,
            finished_at TEXT
        );
        CREATE TABLE IF NOT EXISTS steps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            payload TEXT NOT NULL,
            ts TEXT NOT NULL
        );
        """
    )
    db.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def create_run(run_id: str, goal: str, start_url: str) -> None:
    _db().execute(
        "INSERT INTO runs (id, goal, start_url, status, created_at) VALUES (?, ?, ?, 'planning', ?)",
        (run_id, goal, start_url, _now()),
    )
    _db().commit()


def set_status(run_id: str, status: str, summary: str | None = None) -> None:
    _db().execute(
        "UPDATE runs SET status = ?, summary = ?, finished_at = ? WHERE id = ?",
        (status, summary, _now(), run_id),
    )
    _db().commit()


def record_step(run_id: str, kind: str, payload: dict) -> None:
    try:
        _db().execute(
            "INSERT INTO steps (run_id, kind, payload, ts) VALUES (?, ?, ?, ?)",
            (run_id, kind, json.dumps(payload, default=str, ensure_ascii=False), _now()),
        )
        _db().commit()
    except sqlite3.Error:
        pass  # transcript is best-effort; never break the run over storage


def get_run(run_id: str) -> dict | None:
    row = _db().execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    return dict(row) if row else None


def get_steps(run_id: str, limit: int = 500) -> list[dict]:
    rows = _db().execute(
        "SELECT kind, payload, ts FROM steps WHERE run_id = ? ORDER BY id LIMIT ?",
        (run_id, limit),
    ).fetchall()
    return [{"kind": r["kind"], "ts": r["ts"], "payload": json.loads(r["payload"])} for r in rows]


def list_runs() -> list[dict]:
    rows = _db().execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]
