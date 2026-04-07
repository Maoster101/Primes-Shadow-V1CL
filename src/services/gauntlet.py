"""§17.1-17.5 — Pushback Gauntlet (Absence-of-Resistance Detector).

When the model detects that a claim advanced without expected resistance,
the gauntlet fires: pause smooth synthesis, run structured adversarial
interrogation, and record the outcome.

The gauntlet does NOT block the user. It injects friction into the
response — surfacing alternatives, counterfactuals, and contamination
checks. The user can acknowledge and continue.

LLM = sensor (proposes gauntlet check results), code = actuator
(decides when to fire, caps frequency, logs PushEvents).
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from typing import Optional
from dataclasses import dataclass, field

from ..models.schemas import FrameState, DriftEstimate
from ..models.enums import OLIMode
from .policy import policy
from . import ollama
from .event_log import EventLog

_event_log = EventLog()


# ── Gauntlet prompt ──────────────────────────────────────────────
# Sent to the model when gauntlet fires. The model runs the 7 steps
# and returns structured JSON. Code decides what to surface.

GAUNTLET_PROMPT = """You are an adversarial reasoning auditor. A claim advanced through
conversation without the expected resistance. Your job: stress-test it.

Run these 7 checks on the claim below and return ONLY valid JSON:

{
  "claim_summary": "1-sentence summary of the claim being tested",
  "alternatives": ["1-3 plausible alternative explanations"],
  "counterfactuals": ["1-3 conditions that would invalidate this claim"],
  "anchor_conflicts": ["any contradictions with known corpus anchors/bundles, or empty"],
  "time_scale_check": "is this urgency-driven or structurally sound? 1 sentence",
  "contamination_check": "is affect, incentive, or social pressure driving this? 1 sentence",
  "verdict": "passed | needs_friction | suspicious",
  "friction_note": "if verdict != passed, a 1-2 sentence note to surface to the user"
}

Active corpus anchors (for conflict check): $ANCHORS
Recent conversation context: $CONTEXT
Claim to interrogate: $CLAIM
"""


@dataclass
class GauntletResult:
    """Output of a gauntlet run."""
    fired: bool = False
    verdict: str = "passed"           # passed | needs_friction | suspicious
    claim_summary: str = ""
    alternatives: list[str] = field(default_factory=list)
    counterfactuals: list[str] = field(default_factory=list)
    anchor_conflicts: list[str] = field(default_factory=list)
    time_scale_check: str = ""
    contamination_check: str = ""
    friction_note: str = ""           # surfaced to user if non-empty
    push_event_id: Optional[str] = None


@dataclass
class GauntletState:
    """Per-session gauntlet tracking."""
    last_fired_turn: int = 0
    total_fires: int = 0
    fire_history: list[dict] = field(default_factory=list)


class GauntletEngine:
    """Manages the pushback gauntlet lifecycle per session."""

    def __init__(self, corpus):
        self.corpus = corpus
        self._sessions: dict[str, GauntletState] = {}

    def _get_state(self, session_id: str) -> GauntletState:
        if session_id not in self._sessions:
            self._sessions[session_id] = GauntletState()
        return self._sessions[session_id]

    def should_fire(
        self,
        session_id: str,
        turn: int,
        frame: FrameState,
        drift: DriftEstimate,
        oli_mode: OLIMode,
    ) -> tuple[bool, str]:
        """Determine if the gauntlet should fire this turn.

        Returns (should_fire, reason).
        Conditions (any one triggers):
          1. Mismatch score above escalation threshold (from frame_manager)
          2. Drift affect_density + claim_volatility high while rigor low
          3. Explicit pushback slab is active in the frame
        Cooldown: must wait gauntlet_cooldown_turns between fires.
        """
        state = self._get_state(session_id)
        cooldown = policy.pushback.gauntlet_cooldown_turns

        # Cooldown check (skip if gauntlet has never fired in this session)
        if state.total_fires > 0 and turn - state.last_fired_turn < cooldown:
            return False, "cooldown"

        # Condition 1: High mismatch (absence-of-resistance signal)
        sensitivity = policy.pushback.absence_sensitivity
        if frame.mismatch_score > policy.frame.activation.mismatch_escalation:
            return True, f"mismatch_score={frame.mismatch_score:.2f}"

        # Condition 2: Affect-volatility spike with rigor drop
        # This catches "the user is emotionally invested AND making claims
        # AND rigor is dropping" — a classic smooth-agreement trap
        if (drift.affect_density > 0.5
            and drift.claim_volatility > 0.4
            and drift.rigor_drop > 0.3):
            return True, (
                f"affect={drift.affect_density:.2f} "
                f"volatility={drift.claim_volatility:.2f} "
                f"rigor_drop={drift.rigor_drop:.2f}"
            )

        # Condition 3: Pushback slab explicitly active
        for slab_id in frame.active_slabs:
            if "pushback" in slab_id.lower():
                return True, f"pushback_slab_active:{slab_id}"

        return False, "no_trigger"

    async def run(
        self,
        session_id: str,
        turn: int,
        user_text: str,
        recent_context: str,
        frame: FrameState,
        drift: DriftEstimate,
        oli_mode: OLIMode,
    ) -> GauntletResult:
        """Run the full gauntlet if conditions are met.

        Returns GauntletResult. If fired=False, the gauntlet didn't
        trigger and the result is empty.
        """
        should, reason = self.should_fire(session_id, turn, frame, drift, oli_mode)
        if not should:
            return GauntletResult(fired=False)

        state = self._get_state(session_id)

        # Build anchor context for the model
        active_anchors = [
            f"{a.canonical_phrase} -> {', '.join(a.invokes)}"
            for aid in frame.active_anchors
            if (a := self.corpus.anchors.get(aid))
        ]
        anchor_text = "\n".join(active_anchors) if active_anchors else "(none active)"

        # Build the gauntlet prompt
        prompt = (
            GAUNTLET_PROMPT
            .replace("$ANCHORS", anchor_text)
            .replace("$CONTEXT", recent_context[-2000:])
            .replace("$CLAIM", user_text[-1000:])
        )

        # Step 1-6: Model runs adversarial checks
        try:
            result_raw = await ollama.structured_extract(prompt)
            result = GauntletResult(
                fired=True,
                verdict=result_raw.get("verdict", "needs_friction"),
                claim_summary=result_raw.get("claim_summary", ""),
                alternatives=result_raw.get("alternatives", []),
                counterfactuals=result_raw.get("counterfactuals", []),
                anchor_conflicts=result_raw.get("anchor_conflicts", []),
                time_scale_check=result_raw.get("time_scale_check", ""),
                contamination_check=result_raw.get("contamination_check", ""),
                friction_note=result_raw.get("friction_note", ""),
            )
        except Exception as e:
            # Gauntlet failure is non-fatal — log and return soft result
            result = GauntletResult(
                fired=True,
                verdict="needs_friction",
                friction_note=f"Gauntlet check failed ({e}). Proceeding with caution.",
            )

        # Step 7: Record PushEvent
        push_event_id = f"PUSH_{session_id}_{turn}"
        result.push_event_id = push_event_id

        _event_log.log_push_event(
            id=push_event_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            trigger=reason,
            confidence=drift.claim_volatility,
            expected_resistance=policy.pushback.absence_sensitivity,
            observed_resistance=max(0, policy.pushback.absence_sensitivity - frame.mismatch_score),
            resolution=result.verdict,
            verdict=result.verdict,
            claim_summary=result.claim_summary,
            alternatives_count=len(result.alternatives),
            counterfactuals_count=len(result.counterfactuals),
            anchor_conflicts_count=len(result.anchor_conflicts),
        )

        # Update session state
        state.last_fired_turn = turn
        state.total_fires += 1
        state.fire_history.append({
            "turn": turn,
            "reason": reason,
            "verdict": result.verdict,
        })

        return result

    def format_friction(self, result: GauntletResult) -> str:
        """Format gauntlet results as friction text to inject into the response.

        This is appended to the system prompt or surfaced in the UI,
        NOT inserted into the model's response text.
        """
        if not result.fired or result.verdict == "passed":
            return ""

        lines = ["\n[GAUNTLET — Absence-of-Resistance Check]"]
        if result.claim_summary:
            lines.append(f"Claim: {result.claim_summary}")

        if result.alternatives:
            lines.append("Alternatives:")
            for alt in result.alternatives[:3]:
                lines.append(f"  - {alt}")

        if result.counterfactuals:
            lines.append("Would break if:")
            for cf in result.counterfactuals[:3]:
                lines.append(f"  - {cf}")

        if result.anchor_conflicts:
            conflict_text = [c for c in result.anchor_conflicts if c]
            if conflict_text:
                lines.append("Corpus conflicts:")
                for c in conflict_text[:3]:
                    lines.append(f"  - {c}")

        if result.time_scale_check:
            lines.append(f"Time scale: {result.time_scale_check}")
        if result.contamination_check:
            lines.append(f"Contamination: {result.contamination_check}")

        if result.friction_note:
            lines.append(f">> {result.friction_note}")

        lines.append(f"Verdict: {result.verdict}")
        lines.append("[/GAUNTLET]")
        return "\n".join(lines)
