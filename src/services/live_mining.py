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

  Half 1.5 — per-turn cross-reference matcher (this module):
    On each turn, check whether the new content semantically
    references any existing tentative draft. If so, append the turn
    number to draft.referenced_in_turns. Pure positive-signal
    capture — doesn't try to classify confirm-vs-retract here.

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
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..models.enums import EpistemicStatus, DialecticSubtype, DraftStatus
from ..models.schemas import DraftPacket

logger = logging.getLogger(__name__)


# Cosine similarity threshold for "this turn semantically references
# this draft". Tuned for nomic-embed-text on chat-shaped text:
# - 0.50 catches casual rewording (lots of false positives)
# - 0.55 is a reasonable working threshold (current default)
# - 0.65 is more conservative, may miss paraphrased references
# Trajectory analysis at end-of-conversation can re-validate flagged
# references via LLM, so leaning generous here is safer than stingy
# — under-flagging means UNTOUCHED-misclassification later.
REFERENCE_THRESHOLD = 0.55


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


def _draft_text_for_matching(packet: DraftPacket) -> str:
    """Extract a comparable text representation from a draft packet.

    Anchors → canonical_phrase + aliases (compact identity).
    Slabs   → title + first 200 chars of canonical_text (the title
              alone often isn't enough for embedding match; the lead
              of the canonical_text disambiguates similar titles).
    Bundles → joined intent strings (the bundle's purpose statement).

    Returns "" when the draft has no usable text — caller skips
    those entries rather than embedding empty strings.
    """
    if packet.packet_type == "anchor" and packet.anchor:
        phrase = packet.anchor.get("canonical_phrase", "") or ""
        aliases = packet.anchor.get("aliases") or []
        return (phrase + " " + " ".join(aliases)).strip()
    if packet.packet_type == "slab" and packet.slab:
        title = packet.slab.get("title", "") or ""
        body = (packet.slab.get("canonical_text") or "")[:200]
        return (title + " " + body).strip()
    if packet.packet_type == "bundle" and packet.bundle:
        payload = packet.bundle.get("payload") or {}
        intent = payload.get("intent") or []
        return " ".join(intent).strip()
    return ""


async def update_reference_history(
    session_id: str, turn: int, turn_content: str,
) -> int:
    """Per-turn cross-reference matcher.

    For each tentative draft (DRAFT_UNAUTHORIZED + status NEWLY_RAISED)
    in the session's ghost stack, check whether the current turn
    semantically references it. If similarity >= REFERENCE_THRESHOLD,
    append the turn number to draft.referenced_in_turns.

    Pure positive-signal capture — does not classify whether the
    reference is a confirmation, retraction, or drift. The trajectory
    canonicalizer at end-of-conversation makes that determination
    with full context.

    Skips drafts where ``turn`` is already in source_turns (the turn
    that *raised* the draft can't *reference* it — that would create
    a self-loop in the trajectory analysis).

    Returns: number of drafts whose reference_in_turns list was updated.
    """
    from . import ollama
    from ..api import deps

    if not turn_content or not turn_content.strip():
        return 0

    # Load and filter candidate drafts
    packets = deps.session_store.list_draft_packets(session_id)
    candidates: list[DraftPacket] = []
    candidate_texts: list[str] = []
    for p in packets:
        if p.status != DraftStatus.DRAFT_UNAUTHORIZED:
            continue
        if p.epistemic_status != EpistemicStatus.NEWLY_RAISED:
            continue
        if turn in p.source_turns:
            continue  # don't self-match the raising turn
        if turn in p.referenced_in_turns:
            continue  # already recorded for this turn
        text = _draft_text_for_matching(p)
        if not text:
            continue
        candidates.append(p)
        candidate_texts.append(text)

    if not candidates:
        return 0

    # One Ollama batch call: turn_content + all candidate texts.
    # Batching avoids N+1 round-trips through nomic-embed-text.
    try:
        vecs = await ollama.embed([turn_content] + candidate_texts)
    except Exception as exc:
        logger.warning(
            "[LIVE-MINE] Embedding for reference matching failed (turn %d): %r",
            turn, exc,
        )
        return 0
    if not vecs or len(vecs) != len(candidates) + 1:
        logger.warning(
            "[LIVE-MINE] Embed returned %d vecs for %d inputs (turn %d)",
            len(vecs) if vecs else 0, len(candidates) + 1, turn,
        )
        return 0

    # L2-normalize once for cosine via dot product. nomic-embed-text
    # vectors aren't pre-normalized so we have to do this ourselves.
    turn_arr = np.array(vecs[0], dtype=float)
    turn_norm = float(np.linalg.norm(turn_arr))
    if turn_norm == 0.0:
        return 0
    turn_arr = turn_arr / turn_norm

    updated_count = 0
    for packet, candidate_vec in zip(candidates, vecs[1:]):
        cand_arr = np.array(candidate_vec, dtype=float)
        cand_norm = float(np.linalg.norm(cand_arr))
        if cand_norm == 0.0:
            continue
        cand_arr = cand_arr / cand_norm
        sim = float(np.dot(turn_arr, cand_arr))
        if sim < REFERENCE_THRESHOLD:
            continue
        # Append turn + persist the updated packet. Idempotent against
        # double-fire (the `turn in p.referenced_in_turns` guard above
        # filters re-runs).
        packet.referenced_in_turns.append(turn)
        deps.session_store.save_draft_packet(session_id, packet)
        updated_count += 1
        logger.debug(
            "[LIVE-MINE] turn %d references draft %s (sim=%.2f)",
            turn, packet.id, sim,
        )

    if updated_count:
        logger.info(
            "[LIVE-MINE] Turn %d cross-referenced %d existing draft(s)",
            turn, updated_count,
        )
    return updated_count


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
