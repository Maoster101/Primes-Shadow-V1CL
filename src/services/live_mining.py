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
import re
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


# Prompt for the LLM disambiguation pass — fires only on drafts with
# reference_count == 1 where heuristics can't decide CONFIRMED vs
# DRIFTED vs RETRACTED vs PASSING_MENTION. Compact enough for a
# small LLM (gemma3:12b or smaller) to handle reliably; gives the
# model the source turn + reference turn + draft phrase as context.
_DISAMBIGUATION_PROMPT = """You're analyzing how a position raised in a conversation evolved across turns.

A tentative position was raised at turn {raised_at}, then mentioned once at turn {reference_turn}. Classify how the reference relates to the original position.

DRAFT POSITION:
  Phrase: {canonical_phrase}
  Context: {justification}

ORIGINAL TURN (turn {raised_at}, where the position was raised):
{source_turn_content}

REFERENCE TURN (turn {reference_turn}, where the concept came up again):
{reference_turn_content}

Classify the reference turn's relationship to the original position. Choose ONE:

- CONFIRMATION: the reference turn agrees with, restates, or validates the original position.
  Examples: "yes, exactly", "right, that's the point", "this is why X matters", "I keep coming back to X".

- RETRACTION: the reference turn explicitly withdraws the original position.
  Examples: "actually no, X is wrong", "wait, I take that back", "scratch X, I was wrong", "X doesn't hold up".

- DRIFT: the reference turn pivots to a related but revised position — partial agreement, modified framing, or a follow-on refinement.
  Examples: "more precisely Y", "closer to Y than X", "X but with these caveats", "X works for case A, but case B needs Y".

- PASSING_MENTION: the reference turn mentions the concept in passing without committing to it. The mention exists but doesn't engage with the position.
  Examples: a casual reference in a list of topics, an aside, a clarifying mention while discussing something else.

Return ONLY raw JSON:
{{
  "verdict": "CONFIRMATION" | "RETRACTION" | "DRIFT" | "PASSING_MENTION",
  "justification": "short reason — quote the operative phrase from the reference turn if helpful"
}}
"""


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


def _phrase_appears_in_text(phrase: str, text: str) -> bool:
    """Word-bounded case-insensitive substring check.

    Used as a verbatim-match precheck before falling back to
    embedding cosine. Catches the calibration miss where a short
    phrase (e.g. "Neural Gravity") appears literally in a long turn
    but the long-turn embedding dilutes the cosine below threshold.

    Word-bounded so "log" doesn't false-match "Private Logic" and
    "NG" doesn't false-match "singing". Empty inputs are no-match.
    """
    if not phrase or not text:
        return False
    pattern = r"\b" + re.escape(phrase) + r"\b"
    return bool(re.search(pattern, text, re.IGNORECASE))


def _draft_phrase_appears_in_turn(packet: DraftPacket, turn_content: str) -> bool:
    """True if the draft's canonical_phrase OR any alias appears
    word-bounded in turn_content.

    For anchors: check canonical_phrase + each alias.
    For slabs:   check title only (full canonical_text would
                 false-match too readily — slab content is prose).
    For bundles: check each intent string.
    """
    if packet.packet_type == "anchor" and packet.anchor:
        if _phrase_appears_in_text(
            packet.anchor.get("canonical_phrase", "") or "", turn_content,
        ):
            return True
        for alias in (packet.anchor.get("aliases") or []):
            if _phrase_appears_in_text(alias, turn_content):
                return True
        return False
    if packet.packet_type == "slab" and packet.slab:
        return _phrase_appears_in_text(
            packet.slab.get("title", "") or "", turn_content,
        )
    if packet.packet_type == "bundle" and packet.bundle:
        intent = (packet.bundle.get("payload") or {}).get("intent") or []
        for item in intent:
            if _phrase_appears_in_text(item, turn_content):
                return True
        return False
    return False


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
    initial_candidates: list[DraftPacket] = []
    for p in packets:
        if p.status != DraftStatus.DRAFT_UNAUTHORIZED:
            continue
        if p.epistemic_status != EpistemicStatus.NEWLY_RAISED:
            continue
        if turn in p.source_turns:
            continue  # don't self-match the raising turn
        if turn in p.referenced_in_turns:
            continue  # already recorded for this turn
        if not _draft_text_for_matching(p):
            continue  # nothing embeddable / no canonical text
        initial_candidates.append(p)

    if not initial_candidates:
        return 0

    # Two-signal matching:
    #   1. Verbatim string match (cheap, catches "user names the
    #      draft phrase explicitly in this turn"). Short-circuits
    #      embedding for matched drafts.
    #   2. Embedding cosine (semantic match for paraphrased
    #      references). Falls back for drafts that didn't verbatim-
    #      match.
    # Either signal suffices; defense-in-depth handles the
    # calibration case where long-turn embeddings dilute verbatim-
    # phrase cosines below threshold (we measured 0.51 for a verbatim
    # "Neural Gravity" appearance — below the 0.55 cutoff).
    verbatim_matches: list[DraftPacket] = []
    embedding_candidates: list[DraftPacket] = []
    for p in initial_candidates:
        if _draft_phrase_appears_in_turn(p, turn_content):
            verbatim_matches.append(p)
        else:
            embedding_candidates.append(p)

    updated_count = 0

    # --- Path 1: verbatim hits — record + persist directly ---
    for packet in verbatim_matches:
        packet.referenced_in_turns.append(turn)
        deps.session_store.save_draft_packet(session_id, packet)
        updated_count += 1
        logger.debug(
            "[LIVE-MINE] turn %d references draft %s (verbatim)",
            turn, packet.id,
        )

    # --- Path 2: embedding cosine for the rest ---
    if embedding_candidates:
        candidate_texts = [_draft_text_for_matching(p) for p in embedding_candidates]
        try:
            vecs = await ollama.embed([turn_content] + candidate_texts)
        except Exception as exc:
            logger.warning(
                "[LIVE-MINE] Embedding for reference matching failed (turn %d): %r",
                turn, exc,
            )
            # Verbatim hits already counted; just return that count.
            return updated_count
        if not vecs or len(vecs) != len(embedding_candidates) + 1:
            logger.warning(
                "[LIVE-MINE] Embed returned %d vecs for %d inputs (turn %d)",
                len(vecs) if vecs else 0, len(embedding_candidates) + 1, turn,
            )
            return updated_count

        # L2-normalize once for cosine via dot product. nomic-embed-text
        # vectors aren't pre-normalized so we have to do this ourselves.
        turn_arr = np.array(vecs[0], dtype=float)
        turn_norm = float(np.linalg.norm(turn_arr))
        if turn_norm == 0.0:
            return updated_count
        turn_arr = turn_arr / turn_norm

        for packet, candidate_vec in zip(embedding_candidates, vecs[1:]):
            cand_arr = np.array(candidate_vec, dtype=float)
            cand_norm = float(np.linalg.norm(cand_arr))
            if cand_norm == 0.0:
                continue
            cand_arr = cand_arr / cand_norm
            sim = float(np.dot(turn_arr, cand_arr))
            if sim < REFERENCE_THRESHOLD:
                continue
            packet.referenced_in_turns.append(turn)
            deps.session_store.save_draft_packet(session_id, packet)
            updated_count += 1
            logger.debug(
                "[LIVE-MINE] turn %d references draft %s (sim=%.2f)",
                turn, packet.id, sim,
            )

    if updated_count:
        logger.info(
            "[LIVE-MINE] Turn %d cross-referenced %d existing draft(s) "
            "(%d verbatim, %d via embedding)",
            turn, updated_count,
            len(verbatim_matches), updated_count - len(verbatim_matches),
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


async def _llm_disambiguate_reference(
    packet: DraftPacket,
    raised_at: int,
    reference_turn: int,
    source_turn_content: str,
    reference_turn_content: str,
) -> tuple[EpistemicStatus, Optional[DialecticSubtype], str]:
    """LLM call to classify how a reference turn relates to a draft.

    Fires only on drafts with reference_count == 1 where the
    heuristic classifier flagged the relationship as ambiguous.
    Returns a tuple of (final_status, dialectic_subtype, reason)
    where dialectic_subtype is set for RETRACTED / DRIFTED and None
    for the others.

    Verdict mapping:
      CONFIRMATION    → CONFIRMED          (no edge subtype — promote)
      RETRACTION      → RETRACTED          (subtype=RETRACTION)
      DRIFT           → DRIFTED            (subtype=DRIFT)
      PASSING_MENTION → UNTOUCHED          (no edge — leave as is)

    Failure mode: LLM call exception or unparseable response leaves
    the draft as UNTOUCHED with a "(LLM disambiguation failed)" note.
    Worst-case behaviour is "fall back to heuristic" which is the
    same as not running this pass at all.
    """
    from . import ollama

    canonical = ""
    justification = ""
    if packet.packet_type == "anchor" and packet.anchor:
        canonical = packet.anchor.get("canonical_phrase", "") or ""
        justification = packet.anchor.get("notes", "") or ""
    elif packet.packet_type == "slab" and packet.slab:
        canonical = packet.slab.get("title", "") or ""
        justification = (packet.slab.get("canonical_text") or "")[:200]
    elif packet.packet_type == "bundle" and packet.bundle:
        intent = (packet.bundle.get("payload") or {}).get("intent") or []
        canonical = intent[0] if intent else ""
        justification = " | ".join(intent[1:]) if len(intent) > 1 else ""

    prompt = _DISAMBIGUATION_PROMPT.format(
        raised_at=raised_at,
        reference_turn=reference_turn,
        canonical_phrase=canonical or "(no canonical phrase)",
        justification=justification or "(no additional context)",
        source_turn_content=source_turn_content[:1500],
        reference_turn_content=reference_turn_content[:1500],
    )

    try:
        resp = await ollama.structured_extract(prompt)
    except Exception as exc:
        logger.warning(
            "[LIVE-MINE] LLM disambiguation failed for %s: %r",
            packet.id, exc,
        )
        return EpistemicStatus.UNTOUCHED, None, "LLM disambiguation failed (fell back to UNTOUCHED)"

    if not isinstance(resp, dict):
        return EpistemicStatus.UNTOUCHED, None, "LLM returned non-JSON (fell back to UNTOUCHED)"

    verdict = (resp.get("verdict") or "").strip().upper()
    reason = (resp.get("justification") or "").strip() or "(no reason given)"

    if verdict == "CONFIRMATION":
        return EpistemicStatus.CONFIRMED, None, f"LLM: {reason}"
    if verdict == "RETRACTION":
        return EpistemicStatus.RETRACTED, DialecticSubtype.RETRACTION, f"LLM: {reason}"
    if verdict == "DRIFT":
        return EpistemicStatus.DRIFTED, DialecticSubtype.DRIFT, f"LLM: {reason}"
    if verdict == "PASSING_MENTION":
        return EpistemicStatus.UNTOUCHED, None, f"LLM: passing mention only — {reason}"

    # Unknown verdict label — treat as ambiguous, leave UNTOUCHED.
    logger.warning(
        "[LIVE-MINE] LLM returned unknown verdict %r for %s — keeping UNTOUCHED",
        verdict, packet.id,
    )
    return EpistemicStatus.UNTOUCHED, None, f"LLM returned unknown verdict {verdict!r}"


async def canonicalize(
    session_id: str, chat_id: str,
    *,
    use_llm_disambiguation: bool = True,
) -> CanonicalizationPlan:
    """End-of-conversation trajectory analysis.

    Two-pass classifier:
      1. Pure-heuristic pass — reference count + salience peak.
         Drafts with >= CONFIRM_REFERENCE_FLOOR references → CONFIRMED.
         Drafts with 0 references after grace → UNTOUCHED.
         Drafts with exactly 1 reference → UNTOUCHED with ambiguous flag.
      2. LLM disambiguation pass (only on the ambiguous singles) —
         compares source_turn vs reference_turn content and classifies
         the relationship as CONFIRMATION / RETRACTION / DRIFT /
         PASSING_MENTION. Reclassifies the verdict accordingly.

    Pure analysis: no corpus mutation, no draft state change. The
    plan is reviewable before commit (dry-run pattern mirrors
    anchor_consolidation.analyze).

    Args:
      use_llm_disambiguation: when True (default), single-reference
        ambiguous drafts go through the LLM pass. When False, they
        stay UNTOUCHED — useful for fast canonicalization without
        burning LLM calls (e.g. for previewing).

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

    # ── Pass 1: heuristic classification ──
    # Sort each verdict into its bucket. Single-reference drafts go
    # to untouched_* buckets here but get re-routed by the LLM pass
    # below if it's enabled.
    verdict_by_packet: dict[str, TrajectoryVerdict] = {}
    for packet in candidates:
        verdict = _classify_one_draft(packet, final_turn)
        verdict_by_packet[packet.id] = verdict
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

    # ── Pass 2: LLM disambiguation for single-reference ambiguity ──
    # The heuristic put reference_count==1 drafts into UNTOUCHED. The
    # LLM call inspects each one's source_turn vs reference_turn
    # content and reclassifies as CONFIRMED / DRIFTED / RETRACTED /
    # leaves-as-UNTOUCHED-passing-mention.
    if use_llm_disambiguation:
        ambiguous_candidates = [
            (p, verdict_by_packet[p.id])
            for p in candidates
            if verdict_by_packet[p.id].reference_count == 1
        ]
        if ambiguous_candidates:
            await _run_llm_disambiguation_pass(
                plan, ambiguous_candidates, chat_id,
            )

    logger.info(
        "[LIVE-MINE] canonicalize plan for session %s: %s",
        session_id, plan.summary(),
    )
    return plan


async def _run_llm_disambiguation_pass(
    plan: CanonicalizationPlan,
    ambiguous: list[tuple[DraftPacket, TrajectoryVerdict]],
    chat_id: str,
) -> None:
    """Iterate ambiguous drafts, call the LLM disambiguator on each,
    and move verdicts to their correct buckets in `plan`.

    Loads chat messages once and indexes by turn so each LLM call
    only fans out the exact two turn contents it needs (source +
    reference). Mutates `plan` in place — verdicts get removed from
    untouched_* buckets and reinserted into confirmed/drifted/
    retracted as the LLM decides.
    """
    from ..api import deps
    messages = deps.chat_store.get_messages(chat_id)
    by_turn: dict[int, str] = {m.turn: m.content for m in messages}

    for packet, verdict in ambiguous:
        raised_at = packet.source_turns[0] if packet.source_turns else None
        reference_turn = (
            packet.referenced_in_turns[0]
            if packet.referenced_in_turns else None
        )
        if raised_at is None or reference_turn is None:
            continue  # shouldn't happen given filter, defensive
        source_content = by_turn.get(raised_at) or ""
        ref_content = by_turn.get(reference_turn) or ""
        if not source_content or not ref_content:
            # Can't disambiguate without both turn contents; leave
            # the heuristic verdict in place.
            continue

        new_status, edge_subtype, llm_reason = await _llm_disambiguate_reference(
            packet, raised_at, reference_turn,
            source_content, ref_content,
        )
        if new_status == verdict.final_status:
            # No reclassification needed (still UNTOUCHED).
            verdict.reason = llm_reason
            continue

        # Capture previous status BEFORE mutation so the log line
        # can show the actual transition.
        prior_status = verdict.final_status

        # Remove from current bucket
        for bucket in (
            plan.untouched_keep, plan.untouched_drop,
            plan.confirmed, plan.drifted, plan.retracted,
        ):
            if verdict in bucket:
                bucket.remove(verdict)
                break

        # Update verdict + reinsert into the right bucket
        verdict.final_status = new_status
        verdict.reason = llm_reason
        verdict.edge_subtype = edge_subtype

        if new_status == EpistemicStatus.CONFIRMED:
            plan.confirmed.append(verdict)
        elif new_status == EpistemicStatus.RETRACTED:
            plan.retracted.append(verdict)
        elif new_status == EpistemicStatus.DRIFTED:
            plan.drifted.append(verdict)
        else:
            # PASSING_MENTION-like outcome — back to untouched buckets
            # by salience.
            if verdict.peak_salience >= UNTOUCHED_KEEP_SALIENCE:
                plan.untouched_keep.append(verdict)
            else:
                plan.untouched_drop.append(verdict)

        logger.info(
            "[LIVE-MINE] LLM reclassified %s: %s → %s",
            packet.id, prior_status.value, new_status.value,
        )


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
