"""§17.6-17.10 — User Drift Monitoring.

Two-way monitoring: the mirror tracks the user's epistemic state across turns.
Model proposes per-turn signals; code computes the rolling window, applies
recency scalar, domain modifier, and determines severity.

LLM = sensor (per-turn estimates), code = actuator (windowed severity + dampening).
"""
from __future__ import annotations
from typing import Optional

from ..models.schemas import DriftEstimate
from ..models.enums import DriftSeverity, DomainMode, DampeningLevel
from .event_log import EventLog

_event_log = EventLog()

# §17.7 — Rolling window config
WINDOW_SIZE = 50  # turns
SESSION_DURATION_BASELINE = 5  # avg session length in recent chats (turns, will be dynamic)

# §17.8 — Severity thresholds (from frame_policy.yaml, hardcoded for now)
THRESHOLDS = {
    DomainMode.EXTERNAL: {"low": 0.39, "medium": 0.69},
    DomainMode.INTERNAL: {"low": 0.54, "medium": 0.79},
}


class DriftWindow:
    """Rolling window of per-turn drift estimates with linear recency weighting."""

    def __init__(self):
        self._history: list[DriftEstimate] = []
        self._session_start_turn: int = 0

    def push(self, estimate: DriftEstimate, turn: int) -> None:
        """Add a per-turn estimate to the window."""
        self._history.append(estimate)
        if len(self._history) > WINDOW_SIZE:
            self._history = self._history[-WINDOW_SIZE:]
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

        # §17.7 — Session duration signal (code-computed)
        session_turns = current_turn - self._session_start_turn
        duration_threshold = SESSION_DURATION_BASELINE * 0.7
        duration_signal = min(1.0, session_turns / max(1, duration_threshold * 2)) if duration_threshold > 0 else 0.0

        # §17.7 — Domain mode (most recent estimate)
        domain_mode = self._history[-1].domain_mode if self._history else DomainMode.EXTERNAL

        # §17.7 — Domain modifier: internal work reads hotter
        thresholds = THRESHOLDS.get(domain_mode, THRESHOLDS[DomainMode.EXTERNAL])

        # §17.8 — Composite: weighted average of three model signals + duration
        composite = (
            affect_weighted * 0.30 +
            volatility_weighted * 0.25 +
            rigor_weighted * 0.25 +
            duration_signal * 0.20
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

        return result
