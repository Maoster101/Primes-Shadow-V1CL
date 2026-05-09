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


# Trajectory thresholds. A reference COUNT (length of
# referenced_in_turns) of >= 2 is the confirmation floor — single
# references are too noisy on the 0.55 similarity threshold to
# trust without ambiguity. Tunable.
CONFIRM_REFERENCE_FLOOR = 2

# How many turns must elapse from raise without a reference before
# UNTOUCHED becomes a stable verdict. Prevents prematurely dropping
# drafts the conversation simply hasn't returned to yet.
UNTOUCHED_GRACE_TURNS = 3

# UNTOUCHED-keep vs UNTOUCHED-drop threshold. Peak salience scores
# above this preserve the draft as standing-but-isolated; below,
# discard. Salience values are 0-1 from the frame manager. 0.4 is
# the existing FRAME_ACTIVATION_THRESHOLD, so anything that hit the
# active-frame floor is "preserve worthy".
UNTOUCHED_KEEP_SALIENCE = 0.4


def _classify_one_draft(
    packet: DraftPacket,
    final_turn: int,
) -> TrajectoryVerdict:
    """Pure-heuristic trajectory classifier for a single draft.

    Returns a TrajectoryVerdict with final_status set. Does NOT
    detect drift/retraction relationships across drafts — that's
    a second pass over the population (drift requires comparing
    pairs of drafts, not single-draft inspection). Drafts that
    might be drifted/retracted come out as UNTOUCHED here and get
    re-classified by the population pass.

    Decision rules:
      reference_count >= CONFIRM_REFERENCE_FLOOR → CONFIRMED
      reference_count == 0 AND grace expired      → UNTOUCHED
      reference_count == 1 (ambiguous, single hit)→ UNTOUCHED
        (single reference may be a confirm, drift, or retract — the
        population pass + LLM disambiguation handles these; for now
        treat single as untouched and let promote-criteria filter)
      reference_count == 0 AND grace not expired  → UNTOUCHED
        (conversation might still come back to it; canonicalizer
        running before grace expiry will leave these UNTOUCHED with
        flag indicating "too early to drop")
    """
    # Pull display info
    canonical = ""
    if packet.packet_type == "anchor" and packet.anchor:
        canonical = packet.anchor.get("canonical_phrase", "") or ""
    elif packet.packet_type == "slab" and packet.slab:
        canonical = packet.slab.get("title", "") or ""
    elif packet.packet_type == "bundle" and packet.bundle:
        intent = (packet.bundle.get("payload") or {}).get("intent") or []
        canonical = intent[0] if intent else ""

    raised_at = packet.source_turns[0] if packet.source_turns else None
    refs = list(packet.referenced_in_turns)
    last_ref = max(refs) if refs else None
    peak_salience = max(packet.salience_trajectory) if packet.salience_trajectory else 0.0

    if len(refs) >= CONFIRM_REFERENCE_FLOOR:
        return TrajectoryVerdict(
            draft_id=packet.id,
            canonical_phrase=canonical,
            final_status=EpistemicStatus.CONFIRMED,
            reason=f"Referenced in {len(refs)} subsequent turns",
            reference_count=len(refs),
            raised_at_turn=raised_at,
            last_referenced_turn=last_ref,
            peak_salience=peak_salience,
        )

    grace_expired = (
        raised_at is not None
        and (final_turn - raised_at) >= UNTOUCHED_GRACE_TURNS
    )
    if len(refs) == 0:
        return TrajectoryVerdict(
            draft_id=packet.id,
            canonical_phrase=canonical,
            final_status=EpistemicStatus.UNTOUCHED,
            reason=(
                f"No subsequent references; raised {final_turn - (raised_at or final_turn)} turns ago"
                if grace_expired
                else f"No references yet (within {UNTOUCHED_GRACE_TURNS}-turn grace)"
            ),
            reference_count=0,
            raised_at_turn=raised_at,
            last_referenced_turn=None,
            peak_salience=peak_salience,
        )

    # Single reference (len(refs) == 1) — too ambiguous for heuristic
    # alone. Treat as UNTOUCHED for now; future LLM pass + population
    # pass can lift to CONFIRMED / DRIFTED / RETRACTED with stronger
    # signal. Tagged with reason so downstream UI can flag it.
    return TrajectoryVerdict(
        draft_id=packet.id,
        canonical_phrase=canonical,
        final_status=EpistemicStatus.UNTOUCHED,
        reason=f"Single reference at turn {refs[0]} (ambiguous — needs review)",
        reference_count=1,
        raised_at_turn=raised_at,
        last_referenced_turn=last_ref,
        peak_salience=peak_salience,
    )


async def canonicalize(
    session_id: str, chat_id: str,
) -> CanonicalizationPlan:
    """End-of-conversation trajectory analysis.

    Walks the ghost stack for the session, classifies each tentative
    draft's epistemic trajectory using pure heuristics (no LLM
    calls), and returns a CanonicalizationPlan ready for
    apply_trajectory_plan().

    Pure analysis: no corpus mutation, no draft state change. The
    plan is reviewable before commit (dry-run pattern mirrors
    anchor_consolidation.analyze).

    First-cut implementation is heuristic-only: reference count +
    salience peak determine CONFIRMED / UNTOUCHED-keep /
    UNTOUCHED-drop. The DRIFTED and RETRACTED branches require
    cross-draft comparison + LLM disambiguation — wired in but
    return empty for now. Subsequent commits add:
      - Population-level drift detection (anchors with high embedding
        similarity raised in adjacent turns where the later one
        succeeds the earlier in confirmation density)
      - LLM-based retraction-language detection on the transcript
        spans bracketing each draft's source_turn

    Drafts already in non-DRAFT_UNAUTHORIZED state (already promoted,
    already discarded) are skipped — they've been canonicalized
    through other paths.
    """
    from ..api import deps

    plan = CanonicalizationPlan(chat_id=chat_id, session_id=session_id)

    packets = deps.session_store.list_draft_packets(session_id)
    candidates = [
        p for p in packets
        if p.status == DraftStatus.DRAFT_UNAUTHORIZED
        and p.epistemic_status == EpistemicStatus.NEWLY_RAISED
    ]
    if not candidates:
        logger.info(
            "[LIVE-MINE] canonicalize: no candidates for session %s", session_id,
        )
        return plan

    # Determine the conversation's final turn — used to compute "how
    # many turns elapsed since this draft was raised" for grace-period
    # logic. Falls back to max source_turn across all drafts when no
    # explicit final-turn signal is available.
    final_turn = max(
        (max(p.source_turns) for p in candidates if p.source_turns),
        default=0,
    )
    for p in candidates:
        for t in p.referenced_in_turns:
            if t > final_turn:
                final_turn = t

    for packet in candidates:
        verdict = _classify_one_draft(packet, final_turn)
        if verdict.final_status == EpistemicStatus.CONFIRMED:
            plan.confirmed.append(verdict)
        elif verdict.final_status == EpistemicStatus.UNTOUCHED:
            if verdict.peak_salience >= UNTOUCHED_KEEP_SALIENCE:
                plan.untouched_keep.append(verdict)
            else:
                plan.untouched_drop.append(verdict)
        elif verdict.final_status == EpistemicStatus.DRIFTED:
            plan.drifted.append(verdict)
        elif verdict.final_status == EpistemicStatus.RETRACTED:
            plan.retracted.append(verdict)

    logger.info(
        "[LIVE-MINE] canonicalize plan for session %s: %s",
        session_id, plan.summary(),
    )
    return plan


async def apply_trajectory_plan(
    plan: CanonicalizationPlan,
) -> dict:
    """Apply a canonicalization plan — mutates corpus state per verdicts.

    Lifecycle mapping:
      CONFIRMED       → lifecycle.promote_draft_corpus
                        (commits to the merged corpus, fires the
                        usual promote-time machinery: warm_cache,
                        edge auto-accept, etc.)
      UNTOUCHED_KEEP  → lifecycle.promote_draft_tentative
                        (PROVISIONAL on disk — preserved but not in
                        the active corpus, reasoner can't see it.
                        Standing-but-isolated bucket from the design.)
      UNTOUCHED_DROP  → lifecycle.discard_draft
                        (audit trail kept on disk, status REJECTED.
                        Doesn't appear in the worklist anymore.)
      DRIFTED         → not yet implemented (heuristic-only canon
                        doesn't produce these; they require LLM
                        disambiguation in a follow-up commit)
      RETRACTED       → same as DRIFTED — emit-edge path stubbed

    Each transition also flips the draft's epistemic_status field
    to CANONICALIZED so subsequent passes see it as "trajectory
    classified, don't reclassify". Idempotent against re-runs.

    Returns count breakdown for the API response and diagnostic logs.
    """
    from ..api import deps

    counts = {
        "confirmed_promoted": 0,
        "drifted_demoted": 0,
        "retracted_demoted": 0,
        "untouched_preserved": 0,
        "untouched_discarded": 0,
        "edges_emitted": 0,
        "errors": 0,
    }

    # --- CONFIRMED → promote to corpus ---
    for verdict in plan.confirmed:
        try:
            result = await deps.lifecycle.promote_draft_corpus(
                plan.session_id, verdict.draft_id,
            )
            if isinstance(result, dict) and result.get("status") == "COMMITTED":
                counts["confirmed_promoted"] += 1
                _mark_canonicalized(plan.session_id, verdict.draft_id)
            else:
                counts["errors"] += 1
                logger.warning(
                    "[LIVE-MINE] promote_draft_corpus(%s) returned non-COMMITTED: %s",
                    verdict.draft_id, result,
                )
        except Exception as exc:
            counts["errors"] += 1
            logger.warning(
                "[LIVE-MINE] CONFIRMED promote failed for %s: %r",
                verdict.draft_id, exc,
            )

    # --- UNTOUCHED_KEEP → tentative library ---
    for verdict in plan.untouched_keep:
        try:
            await deps.lifecycle.promote_draft_tentative(
                plan.session_id, verdict.draft_id,
            )
            counts["untouched_preserved"] += 1
            _mark_canonicalized(plan.session_id, verdict.draft_id)
        except Exception as exc:
            counts["errors"] += 1
            logger.warning(
                "[LIVE-MINE] UNTOUCHED_KEEP tentative-promote failed for %s: %r",
                verdict.draft_id, exc,
            )

    # --- UNTOUCHED_DROP → discard ---
    for verdict in plan.untouched_drop:
        try:
            await deps.lifecycle.discard_draft(
                plan.session_id, verdict.draft_id,
            )
            counts["untouched_discarded"] += 1
            # Discarded packets don't get marked CANONICALIZED — their
            # status flips to REJECTED via the discard path, which is
            # the terminal state for these drafts.
        except Exception as exc:
            counts["errors"] += 1
            logger.warning(
                "[LIVE-MINE] UNTOUCHED_DROP discard failed for %s: %r",
                verdict.draft_id, exc,
            )

    # --- DRIFTED + RETRACTED — pending follow-up commit ---
    # The heuristic canonicalizer doesn't produce these verdicts yet
    # (drift/retraction detection requires either LLM disambiguation
    # or population-level pair comparison). Once the second-pass
    # classifier lands, these branches:
    #   1. Emit a TENSIONS edge with dialectic_subtype=DRIFT/RETRACTION
    #      and source_turn set to the trajectory transition turn
    #   2. Move the original draft to a "historical" lifecycle bucket
    #      (preserved on disk, marked superseded_by → the surviving draft)
    if plan.drifted or plan.retracted:
        logger.warning(
            "[LIVE-MINE] DRIFTED/RETRACTED branches not yet implemented "
            "(verdicts present: %d drifted, %d retracted) — skipping",
            len(plan.drifted), len(plan.retracted),
        )

    logger.info("[LIVE-MINE] apply_trajectory_plan counts: %s", counts)
    return counts


def _mark_canonicalized(session_id: str, draft_id: str) -> None:
    """Flip a draft's epistemic_status to CANONICALIZED.

    Called after a successful trajectory action so subsequent
    canonicalize() passes see this draft as already-classified
    and skip it. Idempotent. Best-effort: failure to persist
    the flip doesn't unwind the upstream lifecycle action.
    """
    from ..api import deps
    try:
        packets = deps.session_store.list_draft_packets(session_id)
        for p in packets:
            if p.id == draft_id:
                p.epistemic_status = EpistemicStatus.CANONICALIZED
                deps.session_store.save_draft_packet(session_id, p)
                return
    except Exception as exc:
        logger.warning(
            "[LIVE-MINE] _mark_canonicalized(%s) failed: %r",
            draft_id, exc,
        )
