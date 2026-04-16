"""Event logging service — §5 events/ directory.

Append-only JSONL logs for push events, degradation flags,
gate events, verification calls, and drift events.
"""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

EVENTS_ROOT = Path("app/events")


class EventLog:
    """Append-only JSONL event logger."""

    def __init__(self, root: Optional[Path] = None):
        self.root = root or EVENTS_ROOT
        self.root.mkdir(parents=True, exist_ok=True)

    def _append(self, filename: str, event: dict) -> None:
        event["_logged_at"] = datetime.utcnow().isoformat()
        path = self.root / filename
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str, ensure_ascii=False) + "\n")

    def log_gate_event(self, **kwargs) -> None:
        self._append("gate_events.jsonl", kwargs)

    def log_push_event(self, **kwargs) -> None:
        self._append("push_events.jsonl", kwargs)

    def log_push_resolution(self, **kwargs) -> None:
        """Post-hoc resolution of a prior push event.

        Written to a separate append-only log so `push_events.jsonl` stays
        pure as the detection record. Join on `push_event_id` at read time.
        Schema: {push_event_id, resolution, detector_tier, observed_at_turn,
                 delay_turns, next_user_turn_preview, similarity_to_claim}.
        """
        self._append("push_resolutions.jsonl", kwargs)

    def log_degradation_flag(self, **kwargs) -> None:
        self._append("degradation_flags.jsonl", kwargs)

    def log_verification(self, **kwargs) -> None:
        self._append("verification_log.jsonl", kwargs)

    def log_drift_event(self, **kwargs) -> None:
        self._append("drift_events.jsonl", kwargs)

    def log_match_event(self, **kwargs) -> None:
        self._append("match_events.jsonl", kwargs)

    def log_frame_event(self, **kwargs) -> None:
        self._append("frame_events.jsonl", kwargs)

    def log_proposal_event(self, **kwargs) -> None:
        self._append("proposal_events.jsonl", kwargs)

    def read_recent(self, filename: str, last_n: int = 50) -> list[dict]:
        """Read the last N events from a log file."""
        path = self.root / filename
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        return [json.loads(line) for line in lines[-last_n:] if line]
