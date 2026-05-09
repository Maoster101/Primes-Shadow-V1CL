"""Live mining — per-turn extraction + end-of-conversation trajectory canonicalization.

The architectural complement to the document miners (convo_miner,
narrative_miner). Where document mining does spatial consolidation
(which anchors are central across the document), live mining does
temporal consolidation (which positions actually held by the end of
the conversation).

The flow has two halves:

  Half 1 — per-turn extraction (already exists):
    DraftManager.extract_proposals fires on each chat turn, produces
    tentative drafts with epistemic_status=NEWLY_RAISED and source
    turn provenance. Each draft is a position the conversation just
    surfaced — possibly to be confirmed, possibly to be retracted,
    possibly to drift into a different position.

  Half 2 — end-of-conversation canonicalization (this module):
    Walks the ghost stack with full transcript context. For each
    tentative draft:
      - CONFIRMED   (referenced back / validated) → promote to corpus
      - DRIFTED     (position evolved into another draft) → emit
                     TENSIONS edge with subtype=DRIFT, demote original
      - RETRACTED   (explicitly withdrawn) → emit TENSIONS edge with
                     subtype=RETRACTION, demote to historical
      - UNTOUCHED + low salience → drop
      - UNTOUCHED + high salience → preserve as standing-but-isolated

Crucially: this is an ANALYSIS pass, not a re-extraction. The drafts
were already mined turn-by-turn. The transcript provides temporal
context for trajectory classification, not raw text for fresh
extraction. That's what distinguishes this from the rejected
"end-of-session re-mine" approach — re-mining loses the live-
correction signal because user retractions don't survive in raw
text. Per-turn drafts captured those events when they happened;
this pass just classifies them.

This module is the data-structure scaffold. Implementation of the
trajectory classifier, cross-turn reference matcher, and end-of-
conversation canonicalization actions follow in subsequent commits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..models.enums import EpistemicStatus, DialecticSubtype


@dataclass
class TrajectoryVerdict:
    """Per-draft classification produced by the canonicalizer.

    Mirrors the AnchorVerdict shape from anchor_consolidation —
    holds the decision plus diagnostic fields that explain it.
    Translated into action by apply_trajectory_plan() at canonicalize
    time.
    """
    draft_id: str
    canonical_phrase: str         # for display + log lines
    final_status: EpistemicStatus  # CONFIRMED / DRIFTED / RETRACTED / UNTOUCHED
    reason: str                    # human-readable why-this-classification

    # CONFIRMED + DRIFTED + RETRACTED carry counts of supporting evidence
    reference_count: int = 0       # turns that referenced this draft after raise
    raised_at_turn: Optional[int] = None
    last_referenced_turn: Optional[int] = None

    # When DRIFTED / RETRACTED, the draft that superseded this one
    superseded_by_id: Optional[str] = None
    superseded_at_turn: Optional[int] = None

    # Subtype hint for the dialectic edge that this verdict implies.
    # DRIFT for trajectories that evolved through related positions;
    # RETRACTION for explicit withdrawal; LIVE_CORRECTION for mid-
    # turn rephrasing without retraction (rare in practice).
    edge_subtype: Optional[DialecticSubtype] = None

    # Salience signal — peak heat reached during the chat. Used by
    # apply_trajectory_plan to decide UNTOUCHED-but-high-salience
    # vs UNTOUCHED-low-salience.
    peak_salience: float = 0.0


@dataclass
class CanonicalizationPlan:
    """The plan produced by canonicalize() — feeds the apply step.

    Pure data; no I/O performed during planning. Lets us preview
    the canonicalization decisions before mutating the corpus,
    same pattern as anchor_consolidation's dry-run preview.
    """
    chat_id: str
    session_id: str
    confirmed: list[TrajectoryVerdict] = field(default_factory=list)
    drifted: list[TrajectoryVerdict] = field(default_factory=list)
    retracted: list[TrajectoryVerdict] = field(default_factory=list)
    untouched_keep: list[TrajectoryVerdict] = field(default_factory=list)
    untouched_drop: list[TrajectoryVerdict] = field(default_factory=list)

    # Edges that the canonicalizer will emit (DRIFT / RETRACTION /
    # LIVE_CORRECTION subtype). Pre-built so the apply step is just
    # a write loop, no further classification work.
    edges_to_emit: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        """Compact summary suitable for API responses + UI display."""
        total = (
            len(self.confirmed) + len(self.drifted) + len(self.retracted)
            + len(self.untouched_keep) + len(self.untouched_drop)
        )
        return {
            "total_drafts": total,
            "confirmed": len(self.confirmed),
            "drifted": len(self.drifted),
            "retracted": len(self.retracted),
            "untouched_keep": len(self.untouched_keep),
            "untouched_drop": len(self.untouched_drop),
            "edges_to_emit": len(self.edges_to_emit),
        }


# ── Stubs for the analysis pipeline (filled in subsequent commits) ─


async def update_reference_history(
    session_id: str, turn: int, turn_content: str,
) -> int:
    """Per-turn cross-reference matcher — STUB.

    When a new turn lands, walk the ghost stack and check whether
    any tentative draft is being referenced by the new turn's
    content. If so, append the turn to that draft's
    referenced_in_turns list.

    Implementation will use embedding similarity (anchor_matcher
    infrastructure) to detect when new turn content semantically
    matches existing drafts, and a small LLM call to detect
    retraction language ("actually no", "wait, that's wrong",
    "I take that back" — explicit withdrawal markers).

    Returns: number of drafts whose reference history was updated.
    """
    # TODO: implementation in next commit
    return 0


async def canonicalize(
    session_id: str, chat_id: str,
) -> CanonicalizationPlan:
    """End-of-conversation trajectory analysis — STUB.

    Walks the ghost stack for the session, classifies each draft's
    epistemic trajectory, returns a CanonicalizationPlan ready for
    apply_trajectory_plan().

    Pure analysis: no corpus mutation, no draft state change. The
    plan is reviewable before commit (dry-run pattern, mirrors
    anchor_consolidation.analyze).

    Implementation:
      1. Load all drafts for session (chat_store + session_store)
      2. For each draft:
         - Walk referenced_in_turns to compute confirmation density
         - Detect drift / retraction via superseded_by chains
         - Score peak salience from heat trajectory
         - Apply trajectory rules → final_status
      3. For DRIFTED / RETRACTED drafts, generate edge specs
         (TENSIONS with appropriate dialectic_subtype, source_turn
         set to the turn the trajectory transition happened)
      4. Return CanonicalizationPlan

    Returns the plan; caller decides whether to apply.
    """
    # TODO: implementation in subsequent commit
    return CanonicalizationPlan(chat_id=chat_id, session_id=session_id)


async def apply_trajectory_plan(
    plan: CanonicalizationPlan,
) -> dict:
    """Apply a canonicalization plan — STUB.

    Mutates corpus state per the verdicts:
      - CONFIRMED drafts: lifecycle.promote_draft_corpus
      - DRIFTED + RETRACTED: emit edge, mark draft status, optionally
        promote to a "historical" lifecycle bucket
      - UNTOUCHED_KEEP: lifecycle.promote_draft_tentative (preserves
        on disk without committing to corpus)
      - UNTOUCHED_DROP: lifecycle.discard_draft

    Returns counts for diagnostic reporting (mirror
    anchor_consolidation.apply_plan's return shape).
    """
    # TODO: implementation in subsequent commit
    return {
        "confirmed_promoted": 0,
        "drifted_demoted": 0,
        "retracted_demoted": 0,
        "untouched_preserved": 0,
        "untouched_discarded": 0,
        "edges_emitted": 0,
    }
