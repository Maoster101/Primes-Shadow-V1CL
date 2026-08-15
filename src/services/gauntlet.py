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
import re
from datetime import datetime, timezone
from typing import Optional
from dataclasses import dataclass, field

from ..models.schemas import FrameState, DriftEstimate
from ..models.enums import OLIMode
from .policy import policy
from . import ollama
from .event_log import EventLog

_event_log = EventLog()


# ── Resolution detector (tier 1.5) ──────────────────────────────
# Post-hoc categorisation of a prior push event based on the user's
# next turn. Runs at the START of turn N+1, looking back at any
# unresolved push event from turn N. No LLM call — regex + optional
# embedding similarity. See §8 feedback loop.
#
# KNOWN FALSE-POSITIVE RISK (tune with data):
#   "wait" in _REJECT_PAT matches at start-of-turn, so a curious
#   interruption like "wait, this is fascinating, tell me more" is
#   classified as explicit_reject. Same risk for "hmm" (could be
#   thoughtful engagement, not disagreement). Once push_resolutions.jsonl
#   accumulates real data, measure the FP rate — if a meaningful share
#   of "wait"/"hmm" openers turn out to be followed by continued
#   engagement (verdict-at-detection=passed, long subsequent turn),
#   either:
#     1. Remove those two tokens from _REJECT_PAT, OR
#     2. Require a stronger negation pattern after them
#        (e.g. r"^wait\b.{0,40}(no|but|actually|wrong)")
#   Until then the conservative behaviour (treat as reject) is fine —
#   false positives on the reject side are cheaper than false negatives
#   for calibrating the gauntlet (an over-flagged reject is easy to
#   notice in aggregates; a silent acceptance labelled "rejected" is not).

_REJECT_PAT = re.compile(
    r"^(no\b|actually\b|but\s|wrong\b|incorrect\b|disagree|that'?s\s+not|"
    r"not\s+quite|hmm|hold\s+on|wait\b)",
    re.IGNORECASE,
)
_ACCEPT_PAT = re.compile(
    r"^(good\b|great\b|yes\b|yeah\b|yep\b|right\b|correct\b|exactly\b|"
    r"agreed\b|perfect\b|makes\s+sense|got\s+it|understood)",
    re.IGNORECASE,
)


async def detect_resolution(
    next_user_turn: str,
    claim_summary: str,
) -> tuple[str, str, Optional[float]]:
    """Classify a user turn's relationship to a prior flagged claim.

    Returns (resolution, detector_tier, similarity_or_none).

    Resolution classes:
      explicit_reject  — user pushed back in words
      explicit_accept  — user explicitly praised/accepted
      implicit_accept  — short turn with low topical similarity (moved on)
      unresolved       — still engaged with the same topic (kept probing)
      ambiguous        — no strong signal
    """
    stripped = next_user_turn.strip()
    if not stripped:
        return "ambiguous", "tier1", None

    # Tier 1a: explicit keyword prefix match (strongest signal)
    if _REJECT_PAT.match(stripped):
        return "explicit_reject", "tier1", None
    if _ACCEPT_PAT.match(stripped):
        return "explicit_accept", "tier1", None

    # Tier 1b: embedding similarity — topical continuity vs topic change
    sim: Optional[float] = None
    if claim_summary:
        try:
            from . import embeddings
            sim = await embeddings.cosine_similarity(claim_summary, stripped)
        except Exception:
            sim = None

    short_turn = len(stripped) < 80
    if sim is not None:
        if sim < 0.3 and short_turn:
            return "implicit_accept", "tier1.5", sim
        if sim > 0.5:
            return "unresolved", "tier1.5", sim
        return "ambiguous", "tier1.5", sim

    # Fallback when embedding unavailable: length-only heuristic
    if short_turn:
        return "implicit_accept", "tier1", None
    return "unresolved", "tier1", None


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
    # Set when a push event is logged; cleared once the resolution detector
    # categorises the user's next turn. Holds the minimum info needed to
    # retroactively label the push event in push_resolutions.jsonl.
    pending_push: Optional[dict] = None


class GauntletEngine:
    """Manages the pushback gauntlet lifecycle per session."""

    def __init__(self, corpus):
        self.corpus = corpus
        self._sessions: dict[str, GauntletState] = {}

    def _get_state(self, session_id: str) -> GauntletState:
        if session_id not in self._sessions:
            self._sessions[session_id] = GauntletState()
        return self._sessions[session_id]

    async def detect_and_log_resolution(
        self,
        session_id: str,
        turn: int,
        user_text: str,
    ) -> Optional[str]:
        """Categorise the user's current turn against any pending push event.

        Called at the START of each new user turn, before the gauntlet itself
        runs for the new turn. If there's a push event awaiting resolution
        from a prior turn, detect the user's response type (accept / reject /
        implicit / unresolved) and write a record to push_resolutions.jsonl.

        Returns the resolution label, or None if there was nothing pending.
        """
        state = self._get_state(session_id)
        pending = state.pending_push
        if not pending:
            return None

        resolution, tier, sim = await detect_resolution(
            user_text, pending.get("claim_summary", "")
        )

        _event_log.log_push_resolution(
            push_event_id=pending["push_event_id"],
            resolution=resolution,
            detector_tier=tier,
            verdict_at_detection=pending["verdict"],
            fired_turn=pending["fired_turn"],
            observed_at_turn=turn,
            delay_turns=turn - pending["fired_turn"],
            similarity_to_claim=round(sim, 3) if sim is not None else None,
            next_user_turn_preview=user_text[:200],
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        state.pending_push = None
        return resolution

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

        # Condition 3: High truth pressure on any active node
        # Even if global mismatch is low, concentrated epistemic tension
        # on a single node should trigger targeted interrogation.
        for node_id, pressure in frame.truth_pressure.items():
            if pressure > 0.6 and node_id in frame.active_nodes:
                return True, f"truth_pressure={pressure:.2f} on {node_id}"

        # Condition 4: Pushback slab explicitly active
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
            # resolution is intentionally NOT written here. It is logged
            # post-hoc to push_resolutions.jsonl by detect_and_log_resolution()
            # when the user's next turn arrives. Previously this field was
            # aliased to verdict, which made the feedback loop a tautology.
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
        # Arm the resolution detector for the user's next turn.
        state.pending_push = {
            "push_event_id": push_event_id,
            "fired_turn": turn,
            "verdict": result.verdict,
            "claim_summary": result.claim_summary,
        }

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
