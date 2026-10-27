"""Per-run SSE event streaming.

Browser tools push events into a bounded per-run queue (each run keeps the last
MAX_BUFFERED events so late-joining SSE clients get recent context); the SSE
endpoint drains it, with heartbeats to keep connections alive.
"""

import asyncio
import json
import threading
from collections import deque
from queue import Empty, Queue
from typing import AsyncIterator

_HEARTBEAT_SECONDS = 10.0
_MAX_BUFFERED = 500


class _RunStream:
    """Queue-based stream with replay buffer for late subscribers."""

    def __init__(self) -> None:
        self.queue: Queue = Queue()
        self.buffer: deque[dict] = deque(maxlen=_MAX_BUFFERED)
        self.closed = threading.Event()

    def push(self, event: dict) -> None:
        self.queue.put(event)
        self.buffer.append(event)


_streams: dict[str, _RunStream] = {}
_lock = threading.Lock()


def get_stream(run_id: str) -> _RunStream:
    with _lock:
        stream = _streams.get(run_id)
        if stream is None:
            stream = _RunStream()
            _streams[run_id] = stream
        return stream


def close(run_id: str) -> None:
    stream = _streams.get(run_id)
    if stream is not None:
        stream.closed.set()
        stream.push({"kind": "run_end"})  # wake the SSE reader


async def sse_events(run_id: str) -> AsyncIterator[dict]:
    """Async iterator of run events for EventSourceResponse.

    Replays the buffer for late subscribers, then drains the live queue.
    Pushes heartbeats while idle. Ends when the run closes and the queue
    is empty. Draining uses a worker thread so the subprocess/blocking side
    and the async endpoint never interfere.
    """
    stream = get_stream(run_id)
    for event in stream.buffer:
        yield {"event": "update", "data": json.dumps(event, ensure_ascii=False, default=str)}

    loop = asyncio.get_running_loop()
    last_event = asyncio.get_event_loop().time()
    while True:
        try:
            event = await loop.run_in_executor(None, stream.queue.get, True, 0.2)
            last_event = loop.time()
        except Empty:
            if stream.closed.is_set():
                break
            if loop.time() - last_event > _HEARTBEAT_SECONDS:
                last_event = loop.time()
                yield {"event": "ping", "data": ""}
            continue
        yield {"event": "update", "data": json.dumps(event, ensure_ascii=False, default=str)}
        if event.get("kind") == "run_end":
            break
