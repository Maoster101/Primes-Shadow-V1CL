"""§17.6-17.10 — User Drift Monitoring.

Two-way monitoring: the mirror tracks the user's epistemic state across turns.
Model proposes per-turn signals; code computes the rolling window, applies
recency scalar, domain modifier, and determines severity.

LLM = sensor (per-turn estimates), code = actuator (windowed severity + dampening).
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

from ..models.schemas import DriftEstimate
from ..models.enums import DriftSeverity, DomainMode, DampeningLevel
from .event_log import EventLog
from .policy import policy

_event_log = EventLog()

# All thresholds now read from frame_policy.yaml via the policy singleton.
# See app/corpus/state/frame_policy.yaml [drift] section.

# ── Adaptive session duration baseline ─────────────────────────
# Tracks turn counts from past sessions so the duration signal
# reflects actual usage patterns instead of a static guess.

_SESSION_HISTORY_PATH = Path("app/corpus/state/session_history.jsonl")

# Cache of parsed session durations, keyed on the file's (mtime, size).
# _load_session_durations runs on EVERY turn (via compute() →
# record_and_compute), but the file is append-only and changes only at
# session end. Re-reading and json-parsing every line each turn was pure
# waste; this reads from disk only when the file actually changes.
_DURATIONS_CACHE: Optional[list[int]] = None
_DURATIONS_CACHE_KEY: Optional[tuple[float, int]] = None


def _load_session_durations() -> list[int]:
    """Load recent session turn counts from disk.

    Cached on the file's (mtime, size). record_session_end() appends to
    the file, which changes both — so the next call re-reads naturally.
    """
    global _DURATIONS_CACHE, _DURATIONS_CACHE_KEY
    try:
        st = _SESSION_HISTORY_PATH.stat()
    except OSError:
        # Missing file → no history. Drop any stale cache.
        _DURATIONS_CACHE = None
        _DURATIONS_CACHE_KEY = None
        return []

    key = (st.st_mtime, st.st_size)
    if key == _DURATIONS_CACHE_KEY and _DURATIONS_CACHE is not None:
        return _DURATIONS_CACHE

    durations: list[int] = []
    try:
        for line in _SESSION_HISTORY_PATH.read_text(encoding="utf-8").strip().split("\n"):
            if not line.strip():
                continue
            entry = json.loads(line)
            durations.append(int(entry.get("turns", 0)))
    except Exception:
        # Cache the empty result under this key too, so a transiently
        # unreadable file isn't re-parsed every turn until it changes.
        _DURATIONS_CACHE = []
        _DURATIONS_CACHE_KEY = key
        return []

    _DURATIONS_CACHE = durations
    _DURATIONS_CACHE_KEY = key
    return durations


def _compute_adaptive_baseline() -> float:
    """Compute session duration baseline from recent history.

    Returns the rolling average of the last N sessions (where N =
    policy.drift.session_duration_history_size), falling back to
    the static baseline from frame_policy.yaml if no history exists.
    """
    durations = _load_session_durations()
    if not durations:
        return float(policy.drift.session_duration_baseline)
    n = policy.drift.session_duration_history_size
    recent = durations[-n:]
    return sum(recent) / len(recent)


def record_session_end(session_id: str, total_turns: int) -> None:
    """Record a completed session's turn count for adaptive baseline.

    Call this when a session ends (chat closed, session timed out, etc.).
    """
    if total_turns < 1:
        return
    _SESSION_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {"session_id": session_id, "turns": total_turns}
    with open(_SESSION_HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


class DriftWindow:
    """Rolling window of per-turn drift estimates with linear recency weighting."""

    def __init__(self):
        self._history: list[DriftEstimate] = []
        self._session_start_turn: int = 0

    def push(self, estimate: DriftEstimate, turn: int) -> None:
        """Add a per-turn estimate to the window."""
        self._history.append(estimate)
        if len(self._history) > policy.drift.window_size:
            self._history = self._history[-policy.drift.window_size:]
        if self._session_start_turn == 0:
            self._session_start_turn = turn

    def compute(self, current_turn: int) -> dict:
        """Compute windowed composite score with severity.

        Returns dict with: composite, severity, dampening, signals, domain_mode.
        """
        if not self._history:
            return {
                "composite": 0.0,
                "severity": DriftSeverity.LOW,
                "dampening": DampeningLevel.NONE,
                "signals": {"affect": 0, "volatility": 0, "rigor_drop": 0, "duration": 0},
                "domain_mode": DomainMode.EXTERNAL,
                "window_size": 0,
            }

        n = len(self._history)

        # §17.7 — Linear recency scalar: most recent = 1.0, oldest in window = 1/n
        # Sum of weights: 1 + 2 + ... + n = n(n+1)/2
        weight_sum = n * (n + 1) / 2

        affect_weighted = 0.0
        volatility_weighted = 0.0
        rigor_weighted = 0.0

        for i, est in enumerate(self._history):
            w = (i + 1) / weight_sum  # linear: oldest=1/sum, newest=n/sum
            affect_weighted += est.affect_density * w
            volatility_weighted += est.claim_volatility * w
            rigor_weighted += est.rigor_drop * w

        # §17.7 — Session duration signal (code-computed, adaptive baseline)
        session_turns = current_turn - self._session_start_turn
        adaptive_baseline = _compute_adaptive_baseline()
        duration_threshold = adaptive_baseline * 0.7
        duration_signal = min(1.0, session_turns / max(1, duration_threshold * 2)) if duration_threshold > 0 else 0.0

        # §17.7 — Domain mode (most recent estimate)
        domain_mode = self._history[-1].domain_mode if self._history else DomainMode.EXTERNAL

        # §17.7 — Domain modifier: internal work reads hotter
        dt = policy.drift.thresholds
        thresholds_map = {
            DomainMode.EXTERNAL: {"low": dt.external.low, "medium": dt.external.medium},
            DomainMode.INTERNAL: {"low": dt.internal.low, "medium": dt.internal.medium},
        }
        thresholds = thresholds_map.get(domain_mode, thresholds_map[DomainMode.EXTERNAL])

        # §17.8 — Composite: weighted average of three model signals + duration
        cw = policy.drift.composite_weights
        composite = (
            affect_weighted * cw.affect_density +
            volatility_weighted * cw.claim_volatility +
            rigor_weighted * cw.rigor_drop +
            duration_signal * cw.session_duration
        )
        composite = max(0.0, min(1.0, composite))

        # §17.8 — Severity bucketing
        if composite <= thresholds["low"]:
            severity = DriftSeverity.LOW
        elif composite <= thresholds["medium"]:
            severity = DriftSeverity.MEDIUM
        else:
            severity = DriftSeverity.HIGH

        # §17.8 — Dampening response
        if severity == DriftSeverity.LOW:
            dampening = DampeningLevel.NONE
        elif severity == DriftSeverity.MEDIUM:
            dampening = DampeningLevel.MEDIUM
        else:
            dampening = DampeningLevel.HIGH

        return {
            "composite": round(composite, 4),
            "severity": severity,
            "dampening": dampening,
            "signals": {
                "affect": round(affect_weighted, 4),
                "volatility": round(volatility_weighted, 4),
                "rigor_drop": round(rigor_weighted, 4),
                "duration": round(duration_signal, 4),
            },
            "domain_mode": domain_mode,
            "window_size": n,
        }


class DriftMonitor:
    """Per-session drift monitor. Manages rolling windows."""

    def __init__(self):
        self._windows: dict[str, DriftWindow] = {}  # session_id -> DriftWindow

    def get_window(self, session_id: str) -> DriftWindow:
        if session_id not in self._windows:
            self._windows[session_id] = DriftWindow()
        return self._windows[session_id]

    def record_and_compute(
        self,
        session_id: str,
        estimate: DriftEstimate,
        turn: int,
    ) -> dict:
        """Record a per-turn estimate and compute windowed severity.

        Returns the full drift assessment (composite, severity, dampening, signals).
        """
        window = self.get_window(session_id)
        window.push(estimate, turn)
        result = window.compute(turn)

        # Update the estimate's window_composite for the runtime header
        estimate.window_composite = result["composite"]

        # Log drift event if medium or high
        if result["severity"] != DriftSeverity.LOW:
            _event_log.log_drift_event(
                session_id=session_id,
                turn=turn,
                composite=result["composite"],
                severity=result["severity"].value,
                dampening=result["dampening"].value,
                signals=result["signals"],
                domain_mode=result["domain_mode"].value,
                window_size=result["window_size"],
            )

        # §OLI-5 — DEGRADATION_FLAG surface.
        # MEDIUM = user choice (RESOLVE_NOW | DEFER). Default on DEFER: narrow scope
        #          + reduce abstraction velocity.
        # HIGH   = forced clamp. No user choice. Apply full clamp:
        #          narrow_scope + reduce_abstraction_velocity +
        #          increase_uncertainty_surface + drop_nonessential_speculation.
        # Read back via EventLog.read_recent("degradation_flags.jsonl", ...).
        sev = result["severity"]
        if sev == DriftSeverity.MEDIUM:
            _event_log.log_degradation_flag(
                session_id=session_id,
                turn=turn,
                level="MEDIUM",
                reason=_summarise_drift_reason(result),
                composite=result["composite"],
                signals=result["signals"],
                domain_mode=result["domain_mode"].value,
                choice_required=True,
                actions_available=["RESOLVE_NOW", "DEFER"],
                default_on_defer=["narrow_scope", "reduce_abstraction_velocity"],
                forced_clamp=False,
                resolved=False,  # UI flips to True on user action
            )
        elif sev == DriftSeverity.HIGH:
            _event_log.log_degradation_flag(
                session_id=session_id,
                turn=turn,
                level="HIGH",
                reason=_summarise_drift_reason(result),
                composite=result["composite"],
                signals=result["signals"],
                domain_mode=result["domain_mode"].value,
                choice_required=False,
                actions_available=[],
                forced_clamp=True,
                clamp_actions=[
                    "narrow_scope",
                    "reduce_abstraction_velocity",
                    "increase_uncertainty_surface",
                    "drop_nonessential_speculation",
                ],
                resolved=True,  # forced — no user action needed
            )

        return result


def _summarise_drift_reason(result: dict) -> str:
    """Build a terse human-readable reason string from drift signals.

    Picks the dominant contributing signal(s) so the event reader can
    see at a glance what triggered the flag without parsing the full
    composite breakdown.
    """
    signals = result.get("signals", {})
    if not signals:
        return "drift composite threshold exceeded"
    # Find the top 2 signals by magnitude
    top = sorted(signals.items(), key=lambda kv: kv[1], reverse=True)[:2]
    parts = [f"{name}={val:.2f}" for name, val in top if val > 0]
    if not parts:
        return "drift composite threshold exceeded"
    return f"composite={result['composite']:.2f}; top: {', '.join(parts)}"
