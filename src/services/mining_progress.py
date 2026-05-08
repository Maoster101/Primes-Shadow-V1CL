"""Mining progress tracker — module-level state, polled by the UI.

The miners (narrative + convo) update this from inside their async
work loops; the GET /mining-progress endpoint reads from it; the
frontend polls every couple of seconds while a mine is in flight to
render a progress bar with phase + count + ETA.

Singleton design: this is a single-user local app, only one mine
runs at a time. A run_id-based design would be cleaner for
multi-tenancy but adds wiring overhead the project doesn't need yet.
If you start two mines simultaneously, the second's reset() wipes
the first's state — accept that constraint and don't do that.

The state is plain dict-y under a lock, not a dataclass that gets
swapped, so updates are point-mutations rather than reassignments.
The lock matters because asyncio gathers can interleave with the
endpoint reader at any await point.
"""
from __future__ import annotations

import time
from threading import Lock
from typing import Optional


_lock = Lock()
_state: dict = {
    "miner_kind": "idle",
    "phase": "idle",
    "completed": 0,
    "total": 0,
    "started_at": 0.0,
    "error": None,
    "phase_history": [],  # list of {phase, completed, total, secs}
}


def reset(miner_kind: str) -> None:
    """Wipe state and mark a new mine as starting.

    Called once at the top of each miner's mine() method, before any
    work begins. Sets started_at so the endpoint can compute elapsed.
    """
    with _lock:
        _state.clear()
        _state.update({
            "miner_kind": miner_kind,
            "phase": "starting",
            "completed": 0,
            "total": 0,
            "started_at": time.time(),
            "error": None,
            "phase_history": [],
        })


def set_phase(phase: str, total: int = 1) -> None:
    """Transition to a new phase with a known total count.

    Records the previous phase's wall time into phase_history so the
    UI can show a per-phase breakdown post-mine if desired.
    """
    with _lock:
        # Stash the just-finished phase
        prior_phase = _state.get("phase")
        prior_started = _state.get("phase_started_at", _state.get("started_at", 0.0))
        if prior_phase and prior_phase not in ("idle", "starting"):
            _state["phase_history"].append({
                "phase": prior_phase,
                "completed": _state.get("completed", 0),
                "total": _state.get("total", 0),
                "secs": round(time.time() - prior_started, 2),
            })
        _state["phase"] = phase
        _state["completed"] = 0
        _state["total"] = max(1, total)  # avoid div-by-zero on UI
        _state["phase_started_at"] = time.time()


def increment(n: int = 1) -> None:
    """Advance the completed counter for the current phase.

    Called from inside async work units after each LLM call / segment
    completes. Safe to call concurrently — the lock serializes the
    increment so no updates get lost even when asyncio gathers
    multiple coroutines completing simultaneously.
    """
    with _lock:
        _state["completed"] = _state.get("completed", 0) + n


def mark_error(error: str) -> None:
    """Mark the mine as failed; phase stays where it was when it crashed."""
    with _lock:
        _state["error"] = error
        _state["phase"] = "error"


def mark_done() -> None:
    """Mark the mine as complete. Final phase transition before idle."""
    with _lock:
        # Stash the final phase too
        prior_phase = _state.get("phase")
        prior_started = _state.get("phase_started_at", _state.get("started_at", 0.0))
        if prior_phase and prior_phase not in ("idle", "starting", "done"):
            _state["phase_history"].append({
                "phase": prior_phase,
                "completed": _state.get("completed", 0),
                "total": _state.get("total", 0),
                "secs": round(time.time() - prior_started, 2),
            })
        _state["phase"] = "done"


def get() -> dict:
    """Snapshot of current state for the GET endpoint.

    Returns a copy so callers can't mutate the live state through the
    returned reference. Adds elapsed_s so the UI doesn't need server-
    side time math, and a rough overall_pct based on phase weights.
    """
    with _lock:
        snap = dict(_state)
    started = snap.get("started_at", 0.0)
    snap["elapsed_s"] = round(time.time() - started, 2) if started else 0.0
    snap["overall_pct"] = _compute_overall_pct(snap)
    return snap


# Per-phase weights used to compute a rough overall %. The weights
# don't have to sum to 1 — they're normalized at compute time. They
# reflect the relative wall-time each phase typically takes on a
# medium doc, so the overall bar moves at roughly steady pace.
_PHASE_WEIGHTS = {
    # narrative miner phases
    "segmenting": 0.01,
    "anchor_extraction": 0.40,
    "bundle_synthesis": 0.05,
    "slab_extraction": 0.45,
    "consolidation": 0.05,
    # convo miner phases
    "normalizing": 0.01,
    "chunking": 0.01,
    "extracting": 0.85,
    "deduplicating": 0.01,
    "edge_extraction": 0.10,
    "starting": 0.0,
    "idle": 0.0,
    "error": 0.0,
    "done": 1.0,
}


def _compute_overall_pct(snap: dict) -> float:
    """Best-effort overall % across all phases.

    Sums weights of completed phases (from phase_history) plus the
    fraction-completed of the current phase, normalized by the total
    weight of all phases this miner_kind passes through.
    """
    miner_kind = snap.get("miner_kind", "narrative")
    phase = snap.get("phase", "idle")
    if phase == "done":
        return 100.0
    if phase in ("idle", "starting", "error"):
        return 0.0

    # Determine which phase set applies based on miner_kind
    if miner_kind == "convo":
        order = ["normalizing", "chunking", "extracting", "deduplicating",
                 "edge_extraction", "consolidation"]
    else:
        order = ["segmenting", "anchor_extraction", "bundle_synthesis",
                 "slab_extraction", "consolidation"]

    total_weight = sum(_PHASE_WEIGHTS.get(p, 0.0) for p in order)
    if total_weight <= 0:
        return 0.0

    # Walk the order, sum completed-phase weights up to (but not
    # including) the current one, then add the current's fraction.
    accumulated = 0.0
    for p in order:
        if p == phase:
            break
        accumulated += _PHASE_WEIGHTS.get(p, 0.0)
    completed = snap.get("completed", 0)
    total = snap.get("total", 1) or 1
    cur_frac = min(1.0, completed / total)
    accumulated += _PHASE_WEIGHTS.get(phase, 0.0) * cur_frac
    return round(100.0 * accumulated / total_weight, 1)
