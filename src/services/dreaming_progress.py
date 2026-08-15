"""Dreaming progress tracker — singleton, polled by the UI.

Mirror of ``mining_progress`` but for the dreaming pass, which is a flat
batch (N drafts, each a 1-2 call grounding audit + optional rewrite) rather
than a weighted multi-phase pipeline — so this tracker is deliberately
simpler: one running total, one completed counter, a skip counter.

``dream_all_pending`` drives it; ``GET /dreaming-progress`` reads it; the
Dream page polls it while a run is in flight to render a live bar that
increments per draft. Singleton for the same reason as mining_progress:
single-user local app, one dreaming batch at a time.
"""
from __future__ import annotations

import time
from threading import Lock

_lock = Lock()
_state: dict = {
    "status": "idle",       # idle | running | done | error
    "completed": 0,         # drafts dreamed so far
    "skipped": 0,           # drafts skipped (already enriched)
    "total": 0,             # drafts that will be dreamed this run
    "current": "",          # id/title of the draft in flight (best-effort)
    "started_at": 0.0,
    "error": None,
}


def start(total: int) -> None:
    """Begin a dreaming batch with a known count of drafts to process."""
    with _lock:
        _state.update({
            "status": "running",
            "completed": 0,
            "skipped": 0,
            "total": max(0, total),
            "current": "",
            "started_at": time.time(),
            "error": None,
        })


def set_current(label: str) -> None:
    """Best-effort marker of the draft currently being dreamed."""
    with _lock:
        _state["current"] = label or ""


def increment(skipped: bool = False) -> None:
    """Advance the completed (or skipped) counter. Lock-serialized so
    concurrent dreams (PS_DREAMING_PARALLEL) don't lose updates."""
    with _lock:
        if skipped:
            _state["skipped"] = _state.get("skipped", 0) + 1
        else:
            _state["completed"] = _state.get("completed", 0) + 1


def mark_done() -> None:
    with _lock:
        _state["status"] = "done"
        _state["current"] = ""


def mark_error(error: str) -> None:
    with _lock:
        _state["status"] = "error"
        _state["error"] = error


def get() -> dict:
    """Snapshot for the GET endpoint. Adds elapsed_s + pct so the UI needs
    no server-side math. Returns a copy so callers can't mutate live state."""
    with _lock:
        snap = dict(_state)
    started = snap.get("started_at", 0.0)
    snap["elapsed_s"] = round(time.time() - started, 2) if started else 0.0
    total = snap.get("total", 0) or 0
    done = snap.get("completed", 0) + snap.get("skipped", 0)
    snap["pct"] = round(100.0 * done / total, 1) if total else (100.0 if snap.get("status") == "done" else 0.0)
    return snap
