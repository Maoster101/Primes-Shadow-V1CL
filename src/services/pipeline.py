"""Core runtime pipeline — §6.

On each user turn: ingest → classify → match → frame+drift → header → respond.
LLM = sensor, code = actuator throughout.
"""
from __future__ import annotations
import asyncio
import json
import logging
from typing import Optional, AsyncIterator, TYPE_CHECKING

from ..models.schemas import (
    ChatMessage, MessageClassification, DriftEstimate,
    RuntimeHeader, FrameState, FrameStateSummary, GateState,
    EnforcementFlags,
)
from ..models.enums import MessageFunction, OLIMode, DampeningLevel, ValidationStatus
from ..prompts.classification import (
    FUNCTION_GATE_PROMPT, DRIFT_ESTIMATE_PROMPT, COMBINED_CLASSIFY_DRIFT_PROMPT,
)
from . import ollama
from .context_packer import build_system_prompt, build_messages
from .oli_validator import validate_output
from .event_log import EventLog
from .policy import policy

if TYPE_CHECKING:
    from .anchor_matcher import AnchorMatcher, AnchorMatchResult
    from .frame_manager import FrameManager
    from .drift_monitor import DriftMonitor
    from .draft_manager import DraftManager
    from .gauntlet import GauntletEngine
    from .slab_matcher import SlabMatcher

_event_log = EventLog()
logger = logging.getLogger(__name__)


# ── Edge-constrained retrieval ─────────────────────────────────────
# Per-edge-type budgets for subspace construction. Typed edges act as
# dimensional filters: INVOKES is curator-authored precision (anchor says
# "this slab belongs to me"), SEQUENCE is narrative spine, LINKS is
# lexical co-occurrence (weaker but broader net). CONFLICTS is special-
# cased: whenever a conflict edge is traversed, BOTH endpoints are forced
# into the result regardless of budget — seeing one side of a conflict
# without the other is epistemically broken.
_EDGE_BUDGETS = {
    "INVOKES":   8,
    "SEQUENCE":  10,
    "SUPPORTS":  6,
    "LINKS":     6,
    "PARENT_OF": 4,
    # CONFLICTS handled separately — no budget cap, always both endpoints.
}


def _build_edge_subspace(
    corpus,
    seed_ids: set[str],
    budgets: dict[str, int] = _EDGE_BUDGETS,
) -> tuple[set[str], set[str], dict[str, int]]:
    """Walk typed edges 1 hop from seed_ids to build a candidate slab
    subspace for semantic ranking.

    Returns:
      subspace:       set of slab ids reachable from seeds (bounded by
                      per-type budgets)
      conflict_forced: set of slab ids pulled in via CONFLICTS edges, which
                      bypass the ranker — both endpoints of any conflict
                      must be visible to the model.
      edges_walked:   dict of edge_type -> count actually traversed

    Slabs are the only node type added to subspace (anchors/bundles are
    already in the frame or appear via INVOKES expansion). The `seed_ids`
    set can contain any node type — the walker projects to slab neighbors.
    """
    subspace: set[str] = set()
    conflict_forced: set[str] = set()
    edges_walked: dict[str, int] = {}

    if not seed_ids or not corpus:
        return subspace, conflict_forced, edges_walked

    slab_ids = set(corpus.slabs.keys())
    edges = list(corpus.edges.values())

    # CONFLICTS first — always include both endpoints, no budget.
    for e in edges:
        if e.type.value != "CONFLICTS":
            continue
        if e.from_node in seed_ids or e.to_node in seed_ids:
            for endpoint in (e.from_node, e.to_node):
                if endpoint in slab_ids:
                    conflict_forced.add(endpoint)
            edges_walked["CONFLICTS"] = edges_walked.get("CONFLICTS", 0) + 1

    # Typed walk with per-type budgets. Higher-weight edges first within
    # each type so that if budget is tight the strongest signals win.
    for etype, cap in budgets.items():
        if cap <= 0:
            continue
        typed = [e for e in edges if e.type.value == etype]
        typed.sort(key=lambda e: getattr(e, "weight", 0.0), reverse=True)
        taken = 0
        for e in typed:
            if taken >= cap:
                break
            neighbor = None
            if e.from_node in seed_ids and e.to_node not in seed_ids:
                neighbor = e.to_node
            elif e.to_node in seed_ids and e.from_node not in seed_ids:
                neighbor = e.from_node
            if neighbor is None or neighbor in subspace:
                continue
            # Only slab neighbors contribute to the retrieval subspace.
            # Anchor/bundle neighbors are curator structure, not content
            # the model needs to "read."
            if neighbor in slab_ids:
                subspace.add(neighbor)
                taken += 1
        if taken:
            edges_walked[etype] = taken

    return subspace, conflict_forced, edges_walked


async def classify_message(user_text: str) -> MessageClassification:
    """§7 — Function gate. Classify the dominant function of a user message."""
    prompt = FUNCTION_GATE_PROMPT + json.dumps(user_text)
    try:
        result = await ollama.structured_extract(prompt)
        classification = MessageClassification(**result)
    except Exception as exc:
        logger.warning("classify_message failed, defaulting to NEUTRAL: %r", exc)
        classification = MessageClassification(
            function=MessageFunction.NEUTRAL,
            confidence=0.3,
            explicit=False,
            notes=f"Classification failed ({type(exc).__name__}) — defaulting to neutral with low confidence",
        )

    _event_log.log_gate_event(
        function=classification.function.value,
        confidence=classification.confidence,
        explicit=classification.explicit,
        notes=classification.notes,
    )
    return classification


async def estimate_drift(
    user_text: str,
    recent_context: str = "",
) -> DriftEstimate:
    """§17.7 — Drift signal estimation. Model proposes; code computes window."""
    prompt = DRIFT_ESTIMATE_PROMPT.replace("$CONTEXT", recent_context) + json.dumps(user_text)
    try:
        result = await ollama.structured_extract(prompt)
        return DriftEstimate(**result)
    except Exception as exc:
        logger.warning("estimate_drift failed, returning zero signals: %r", exc)
        return DriftEstimate(
            affect_density=0.0,
            claim_volatility=0.0,
            rigor_drop=0.0,
        )


async def classify_and_drift(
    user_text: str,
    recent_context: str = "",
) -> tuple[MessageClassification, DriftEstimate]:
    """Combined classify + drift in a single LLM call (saves ~60s on slow hardware)."""
    prompt = COMBINED_CLASSIFY_DRIFT_PROMPT.replace("$CONTEXT", recent_context or "(start of session)") + json.dumps(user_text)
    try:
        result = await ollama.structured_extract(prompt)

        # Split the combined response into classification and drift
        classification = MessageClassification(
            function=MessageFunction(result.get("function", "neutral")),
            confidence=result.get("confidence", 0.5),
            explicit=result.get("explicit", False),
            mention_type=result.get("mention_type"),
            notes=result.get("notes"),
        )
        drift = DriftEstimate(
            affect_density=float(result.get("affect_density", 0.0)),
            claim_volatility=float(result.get("claim_volatility", 0.0)),
            rigor_drop=float(result.get("rigor_drop", 0.0)),
            domain_mode=result.get("domain_mode", "external"),
        )
    except Exception as exc:
        logger.warning("classify_and_drift failed, defaulting to NEUTRAL + zero drift: %r", exc)
        classification = MessageClassification(
            function=MessageFunction.NEUTRAL,
            confidence=0.3,
            explicit=False,
            notes=f"Combined classify+drift failed ({type(exc).__name__}) — defaults applied",
        )
        drift = DriftEstimate(
            affect_density=0.0,
            claim_volatility=0.0,
            rigor_drop=0.0,
        )

    _event_log.log_gate_event(
        function=classification.function.value,
        confidence=classification.confidence,
        explicit=classification.explicit,
        notes=classification.notes,
    )
    return classification, drift


def build_runtime_header(
    oli_mode: OLIMode,
    classification: MessageClassification,
    frame_state: Optional[FrameState],
    drift: DriftEstimate,
    dampening: DampeningLevel = DampeningLevel.NONE,
    match_result: Optional["AnchorMatchResult"] = None,
    anchor_hits_context: Optional[list[dict]] = None,
) -> RuntimeHeader:
    """§26.3 — Assemble per-turn runtime control header."""
    gate = GateState(
        message_function=classification.function,
        anchor_resolution_allowed=(
            classification.function == MessageFunction.CONTEXT_COMPRESSION
            or classification.explicit
        ),
        confidence_flag=(
            "low" if classification.confidence < policy.gate.confidence.low
            else "ambiguous" if classification.confidence < policy.gate.confidence.ambiguous
            else "normal"
        ),
    )

    frame_summary = FrameStateSummary()
    if frame_state:
        frame_summary = FrameStateSummary(
            active_nodes=frame_state.active_nodes[:24],
            high_tension_pairs=[
                [c.node_a, c.node_b]
                for c in frame_state.conflicts
                if c.tension > 0.5
            ],
            mismatch_score=frame_state.mismatch_score,
        )

    # OP_01 operator state — every wrapped span carries its feature
    # set, and the unclosed-tail correction hint is threaded separately.
    from ..models.schemas import OperatorState
    op_state = OperatorState()
    if match_result is not None:
        op_state = OperatorState(
            wrapped_spans=[w.model_dump() for w in match_result.wrapped_spans],
            correction_hint=match_result.correction_hint,
        )

    # Include active model info so the model knows what it is
    from . import model_profiles
    active_profile = model_profiles.active()

    return RuntimeHeader(
        active_model=active_profile.name,
        active_model_family=active_profile.family,
        oli_mode=oli_mode,
        gate_state=gate,
        frame_state_summary=frame_summary,
        drift_estimate=drift,
        enforcement_flags=EnforcementFlags(
            claim_admissibility_required=(oli_mode == OLIMode.ON),
            dampening_level=dampening,
            review_mode=(classification.function == MessageFunction.CORPUS_REVIEW),
        ),
        operator_state=op_state,
        anchor_hits=anchor_hits_context or [],
    )


async def process_turn(
    user_text: str,
    chat_messages: list[ChatMessage],
    oli_mode: OLIMode = OLIMode.OFF,
    frame_state: Optional[FrameState] = None,
    *,
    session_id: Optional[str] = None,
    chat_id: Optional[str] = None,
    anchor_matcher: Optional[AnchorMatcher] = None,
    frame_manager: Optional[FrameManager] = None,
    drift_monitor: Optional[DriftMonitor] = None,
    gauntlet_engine: Optional["GauntletEngine"] = None,
    slab_matcher: Optional["SlabMatcher"] = None,
    web_mode: str = "off",  # "off" | "on" | "auto"
    think_level: str = "medium",
) -> AsyncIterator[dict]:
    """Full pipeline for one user turn. Yields streaming response chunks.

    STREAM-FIRST architecture — no LLM call before streaming starts.

    Fast path (~0.5s, embeddings only):
      1. Anchor matching with default NEUTRAL classification
      2. Frame update (code-only graph ops)
      3. Build runtime header with defaults
      4. Pack context + start streaming immediately

    Background (concurrent with streaming):
      5. classify_and_drift runs as background task
      6. Results merged into final metadata chunk

    Deferred (after stream completes):
      7. Gauntlet check (only if classification warrants it)
      8. OLI validation + metadata assembly
    """
    turn = len(chat_messages) + 1
    match_result = None

    # ── Step 0: Resolve any push event pending from the prior turn ───
    # The gauntlet may have fired last turn and logged a push event
    # with no resolution. The user's current message is the evidence
    # that categorises it (accept / reject / implicit / unresolved).
    # Runs first so the feedback loop closes before this turn's own
    # gauntlet check (which could generate a new pending push event).
    if gauntlet_engine and session_id:
        try:
            await gauntlet_engine.detect_and_log_resolution(
                session_id, turn, user_text
            )
        except Exception as exc:
            # Resolution logging is diagnostic — never fail the turn.
            import logging
            logging.getLogger(__name__).warning(
                "Push resolution detection failed: %s", exc
            )

    # ── Fast defaults (no LLM call) ──────────────────────────────
    # NEUTRAL classification passes gate_check for ALL anchors
    # (NEUTRAL is in safe_functions). This is the same result as
    # a real classification for ~95% of messages.
    default_classification = MessageClassification(
        function=MessageFunction.NEUTRAL,
        confidence=0.5,
        explicit=False,
        notes="stream-first default — real classification running in background",
    )
    default_drift = DriftEstimate(
        affect_density=0.0,
        claim_volatility=0.0,
        rigor_drop=0.0,
    )

    # ── Keyword pre-classifier ───────────────────────────────────
    # Upgrades the default function BEFORE the system prompt is built,
    # so function-dependent signals (like review_mode) reach the model
    # on the turn they're needed. The background LLM classifier still
    # runs and may disagree in the final metadata — that's accepted:
    # the pre-classifier only needs to get the GATING right, not the
    # final category label for the UI badge.
    from .gate_preclass import preclassify_function
    _preclass_hit = preclassify_function(user_text)
    if _preclass_hit is not None:
        default_classification = MessageClassification(
            function=_preclass_hit,
            confidence=0.7,  # heuristic — LLM classifier may override in metadata
            explicit=False,
            notes=f"pre-classified via keyword heuristic ({_preclass_hit.value})",
        )

    # ── Step 1: Anchor matching (~0.5s, embedding call only) ─────
    if anchor_matcher:
        match_result = await anchor_matcher.match_all(user_text, default_classification)

    # ── Step 2: Frame update (code-only, instant) ────────────────
    if frame_manager and session_id:
        frame_state = await frame_manager.update_turn(
            session_id, turn, user_text, match_result, default_classification
        )

    # Step 2a: Augment affect_density from wrapped-span features.
    if match_result is not None:
        from .anchor_matcher import compute_wrap_affect_boost
        bump = compute_wrap_affect_boost(match_result)
        if bump > 0:
            default_drift.affect_density = min(1.0, default_drift.affect_density + bump)

    # ── Step 3: Build anchor hit context ─────────────────────────
    anchor_hits_ctx: list[dict] = []
    if match_result and anchor_matcher:
        seen: set[str] = set()
        for m in match_result.auto_activate + match_result.candidates:
            if m.anchor_id in seen:
                continue
            seen.add(m.anchor_id)
            anchor_obj = anchor_matcher.corpus.anchors.get(m.anchor_id)
            if anchor_obj:
                anchor_hits_ctx.append({
                    "anchor_id": m.anchor_id,
                    "canonical_phrase": anchor_obj.canonical_phrase,
                    "notes": anchor_obj.notes or "",
                    "confidence": round(m.confidence, 2),
                    "method": m.match_method,
                    "invokes": anchor_obj.invokes,
                })

    # ── Step 3.5: Edge-constrained slab retrieval ────────────────
    # CONSTITUTIONAL + CANONICAL slabs are always full-text; REFERENCE
    # slabs are retrieved via a graph-constrained pipeline where the
    # typed-edge graph defines the *candidate region* and embeddings
    # *rank within it*. Flow:
    #
    #   1. Collect seeds — nodes the frame/match says are "live now":
    #        frame-active non-base_set nodes (weight >= 0.5),
    #        anchors the matcher just fired on this turn.
    #   2. Walk typed edges from seeds (per-type budgets) → subspace
    #        of candidate REFERENCE slabs. CONFLICTS are force-included
    #        (both endpoints, no budget).
    #   3. Rank within subspace via cosine similarity to the user query
    #        (SlabMatcher.top_k_for with id_filter).
    #   4. Collection-name detection adds all slabs of explicitly named
    #        collections (strong user-intent signal, bypasses subspace).
    #   5. Fallbacks when subspace is thin:
    #        cold start (no seeds) → flat semantic at default threshold
    #        topic shift (subspace yielded <N hits) → flat semantic at
    #        a HIGHER threshold (0.70), to catch only genuinely strong
    #        topical pivots without washing out the edge signal.
    #
    # All signals converge on retrieved_ids; full_text split below.
    MAX_REFERENCE_FULL_TEXT = 25
    SUBSPACE_MIN_RANKED = 3          # below this, topic-shift fallback fires
    FLAT_FALLBACK_THRESHOLD = 0.70   # higher bar for flat cosine when fallback
    FRAME_ACTIVATION_THRESHOLD = 0.5

    retrieved_ids: set[str] = set()
    signal_counts = {
        "seeds": 0,
        "subspace": 0,
        "ranked": 0,
        "collection": 0,
        "frame": 0,
        "flat": 0,
        "conflicts": 0,
        "ppr_lift": 0,       # slabs where PR materially changed combined score
        "edges_walked": {},
    }

    # ── 1. Seed collection ──
    # Seeds are nodes currently in conversational attention. Three sources:
    #   - Frame-active nodes with a non-base_set source (anchor cascade,
    #     prior retrieval, user action) and weight above threshold.
    #   - Anchors the matcher fired on this very turn (from match_result).
    # Anchors and bundles are valid seeds in addition to slabs — they're
    # where typed edges radiate from.
    seed_ids: set[str] = set()
    if frame_state is not None:
        for node_id in frame_state.active_nodes:
            weight = max(
                frame_state.active_slabs.get(node_id, 0.0),
                frame_state.active_anchors.get(node_id, 0.0),
                frame_state.active_bundles.get(node_id, 0.0),
            )
            if weight < FRAME_ACTIVATION_THRESHOLD:
                continue
            sources = frame_state.activation_sources.get(node_id, [])
            if sources and all(s.source_type == "base_set" for s in sources):
                continue
            seed_ids.add(node_id)

    if match_result is not None:
        for m in getattr(match_result, "auto_activate", []) or []:
            if getattr(m, "anchor_id", None):
                seed_ids.add(m.anchor_id)
        for m in getattr(match_result, "candidates", []) or []:
            if getattr(m, "anchor_id", None):
                seed_ids.add(m.anchor_id)

    signal_counts["seeds"] = len(seed_ids)

    # Frame-active slabs themselves (non-base_set, above threshold) go
    # directly into retrieved_ids — the frame already decided these are
    # live, no need to re-rank them.
    if frame_state is not None and frame_manager:
        before = len(retrieved_ids)
        for sid, weight in frame_state.active_slabs.items():
            if weight < FRAME_ACTIVATION_THRESHOLD:
                continue
            sources = frame_state.activation_sources.get(sid, [])
            if sources and all(s.source_type == "base_set" for s in sources):
                continue
            if sid in frame_manager.corpus.slabs:
                retrieved_ids.add(sid)
        signal_counts["frame"] = len(retrieved_ids) - before

    # ── 2. Subspace construction via typed-edge walk ──
    subspace: set[str] = set()
    conflict_forced: set[str] = set()
    if seed_ids and frame_manager:
        subspace, conflict_forced, edges_walked = _build_edge_subspace(
            frame_manager.corpus, seed_ids,
        )
        signal_counts["subspace"] = len(subspace)
        signal_counts["edges_walked"] = edges_walked
        if conflict_forced:
            retrieved_ids.update(conflict_forced)
            signal_counts["conflicts"] = len(conflict_forced)

    # ── 3. Rank within subspace (hybrid: cosine + PPR + global PR) ──
    # The graph chose the candidates; three signals choose which
    # of them survive and in what order:
    #   cosine:    query-dependent text similarity
    #   PPR:       reachability from conversational attention (seeds)
    #   global PR: intrinsic structural importance
    # See slab_matcher.hybrid_rank for the linear-combination details.
    ppr_lift_count = 0
    if slab_matcher is not None and slab_matcher.has_cache() and subspace:
        try:
            hits = await slab_matcher.hybrid_rank(
                user_text,
                seed_ids=seed_ids,
                id_filter=subspace,
            )
            ranked = {sid for sid, _s, _c in hits}
            retrieved_ids.update(ranked)
            signal_counts["ranked"] = len(ranked)
            # Count slabs whose combined score was materially boosted by
            # PR (ppr or global_pr contribution >= 0.1 after weighting).
            # Useful signal: did PR change the ranking vs pure cosine?
            for _sid, _score, comp in hits:
                ppr_bonus = 0.5 * comp.get("ppr", 0.0) + 0.3 * comp.get("global_pr", 0.0)
                if ppr_bonus >= 0.1:
                    ppr_lift_count += 1
        except Exception as exc:
            logger.warning("subspace hybrid rank failed: %r", exc)
    signal_counts["ppr_lift"] = ppr_lift_count

    # ── 4. Collection-name detection (user-intent override) ──
    # If the user explicitly names a collection, pull its slabs whole —
    # this is a lexical signal embeddings can't match ("v5" never appears
    # in canonical_text) and it's a strong enough intent that we bypass
    # the subspace and budget caps.
    if frame_manager:
        try:
            import re as _re
            from ..api import deps as _deps
            low = (user_text or "").lower()
            active_ids = list(_deps.registry.active_ids)
            mentioned = set()
            for cid in active_ids:
                if len(cid) >= 3 and cid.lower() in low:
                    mentioned.add(cid)
            trailing_digits = {}
            for cid in active_ids:
                md = _re.search(r"(\d+)$", cid)
                if md:
                    trailing_digits[cid] = md.group(1)
            for m in _re.finditer(r"\bv(?:ersion)?[_\s]*(\d+)\b", low):
                num = m.group(1)
                for cid, tail in trailing_digits.items():
                    if tail == num:
                        mentioned.add(cid)
            if mentioned:
                before = len(retrieved_ids)
                for cid in mentioned:
                    store = _deps.registry.get_store(cid)
                    if store:
                        retrieved_ids.update(store.slabs.keys())
                signal_counts["collection"] = len(retrieved_ids) - before
        except Exception as exc:
            logger.warning("collection-name detection failed: %r", exc)

    # ── 5. Fallbacks ──
    # Cold start: no seeds means the frame has nothing active yet (turn 1
    # of a fresh chat), so the subspace is empty by construction. Run
    # flat cosine at the default threshold.
    # Topic shift: seeds exist but the subspace rank produced few hits —
    # user may have pivoted away from the current attention. Run flat
    # cosine at a HIGHER threshold so only genuinely strong matches slip
    # through as evidence of the pivot; this avoids diluting the graph
    # signal when the current topic is just narrow.
    needs_cold_start = not seed_ids
    needs_topic_shift = (
        bool(seed_ids)
        and signal_counts["ranked"] < SUBSPACE_MIN_RANKED
        and signal_counts["collection"] == 0  # collection override already served
    )
    if (needs_cold_start or needs_topic_shift) and slab_matcher is not None and slab_matcher.has_cache():
        try:
            fb_threshold = 0.55 if needs_cold_start else FLAT_FALLBACK_THRESHOLD
            # Fallback still hybrids in global PR (gamma) — even in
            # cold-start, structurally-central slabs deserve a bias.
            # PPR (beta) is skipped in cold-start (no seeds, nothing
            # to teleport to); in topic-shift it still fires from the
            # existing seeds as a conservative tiebreaker.
            hits = await slab_matcher.hybrid_rank(
                user_text,
                seed_ids=seed_ids if not needs_cold_start else None,
                threshold=fb_threshold,
            )
            before = len(retrieved_ids)
            retrieved_ids.update(sid for sid, _s, _c in hits)
            signal_counts["flat"] = len(retrieved_ids) - before
        except Exception as exc:
            logger.warning("flat fallback failed: %r", exc)

    # Close the loop: slabs retrieved/augmented for this turn activate in
    # the frame so the graph canvas reflects what the model is reading.
    if retrieved_ids and frame_manager and session_id:
        try:
            newly_active = frame_manager.activate_retrieved_slabs(
                session_id, list(retrieved_ids),
            )
            if newly_active:
                logger.debug(
                    "retrieval activated %d new frame nodes (from %d slabs)",
                    newly_active, len(retrieved_ids),
                )
        except Exception as exc:
            logger.warning("retrieval activation failed: %r", exc)

    full_text_slabs: list = []
    catalog_slabs: list = []
    collection_by_id: dict[str, str] = {}
    reference_count = 0  # REFERENCE slabs that made it to full-text (budget cap)
    if frame_manager:
        from ..models.enums import SlabType as _SlabType
        all_slabs = frame_manager.corpus.base_set_slabs(oli_mode)
        for s in all_slabs:
            if s.type in (_SlabType.CONSTITUTIONAL, _SlabType.CANONICAL):
                full_text_slabs.append(s)
            elif s.id in retrieved_ids and reference_count < MAX_REFERENCE_FULL_TEXT:
                full_text_slabs.append(s)
                reference_count += 1
            else:
                # REFERENCE, not retrieved OR over cap → catalog only
                catalog_slabs.append(s)
        try:
            from ..api import deps as _deps
            collection_by_id = _deps.registry.slab_collection_map()
        except Exception:
            collection_by_id = {}
    print(
        f"[RAG] slabs: full_text={len(full_text_slabs)} (ref={reference_count}) "
        f"catalog={len(catalog_slabs)} signals={signal_counts}",
        flush=True,
    )

    # ── Step 4: Runtime header + system prompt (instant) ──────────
    header = build_runtime_header(
        oli_mode, default_classification, frame_state, default_drift,
        DampeningLevel.NONE,
        match_result=match_result,
        anchor_hits_context=anchor_hits_ctx,
    )
    system_prompt = build_system_prompt(
        oli_mode, header,
        base_set_slabs=full_text_slabs or None,
        catalog_slabs=catalog_slabs or None,
        collection_by_id=collection_by_id or None,
    )

    messages = build_messages(system_prompt, chat_messages, frame_state)
    messages.append({"role": "user", "content": user_text})

    # Estimate context usage (chars → tokens) for status bar display
    from .context_packer import CHARS_PER_TOKEN, TARGET_CONTEXT_TOKENS
    _used_chars = sum(len(m.get("content", "")) + 20 for m in messages)
    context_usage = {
        "used_tokens": _used_chars // CHARS_PER_TOKEN,
        "budget_tokens": TARGET_CONTEXT_TOKENS,
        "packed_messages": len(messages),
    }

    # ── Step 5: Launch background classify+drift ─────────────────
    # Runs concurrently while the user sees streaming tokens.
    recent = "\n".join(
        f"[{m.role}] {m.content[:200]}" for m in chat_messages[-10:]
    )
    bg_classify_task = asyncio.create_task(
        classify_and_drift(user_text, recent)
    )

    # ── Step 6: Stream response IMMEDIATELY ──────────────────────
    use_think = think_level != "off"
    full_response = ""
    is_retry = False

    async for chunk in ollama.chat_stream(messages, think=use_think, think_level=think_level, web_mode=web_mode):
        if chunk.get("done"):
            # ── Step 7: Collect background results ───────────────
            # classify+drift should be done by now (ran during streaming).
            # If not, await with a short timeout — don't block the user.
            try:
                classification, drift = await asyncio.wait_for(
                    bg_classify_task, timeout=2.0
                )
            except (asyncio.TimeoutError, Exception):
                classification = default_classification
                drift = default_drift

            # Augment drift with wrap affect from anchor matching
            if match_result is not None:
                from .anchor_matcher import compute_wrap_affect_boost
                bump = compute_wrap_affect_boost(match_result)
                if bump > 0:
                    drift.affect_density = min(1.0, drift.affect_density + bump)

            # Compute windowed drift severity (§17.7-17.8)
            drift_assessment = None
            dampening = DampeningLevel.NONE
            if drift_monitor and session_id:
                drift_assessment = drift_monitor.record_and_compute(session_id, drift, turn)
                dampening = drift_assessment["dampening"]

            # §17.2 Pushback gauntlet — deferred to post-stream.
            # Only fires when drift is elevated + anchors active.
            # Gauntlet friction is stored for NEXT turn's prompt.
            gauntlet_result = None
            if gauntlet_engine and session_id and frame_state:
                gauntlet_ctx = "\n".join(
                    f"[{m.role}] {m.content[:200]}" for m in chat_messages[-6:]
                )
                gauntlet_result = await gauntlet_engine.run(
                    session_id, turn, user_text, gauntlet_ctx,
                    frame_state, drift, oli_mode,
                )
                if gauntlet_result.fired:
                    if frame_manager:
                        pressure_targets = list(frame_state.active_anchors.keys())
                        if pressure_targets:
                            delta = 0.20 if gauntlet_result.anchor_conflicts else 0.10
                            frame_manager.apply_truth_pressure(
                                session_id, pressure_targets, delta=delta
                            )

            # §26.4 — validate before yielding final metadata
            validation = validate_output(full_response, oli_mode, is_retry=is_retry)

            if validation.status == ValidationStatus.REGENERATE and not is_retry:
                # Hard OLI violation on first attempt — retry with correction
                is_retry = True
                yield {
                    "done": False,
                    "content": "\n\n---\n*[OLI violation detected — regenerating...]*\n\n",
                    "oli_regenerate": True,
                }

                retry_messages = list(messages) + [
                    {"role": "assistant", "content": full_response},
                    {"role": "system", "content": (
                        f"[OLI ENFORCEMENT — REGENERATE]\n"
                        f"{validation.correction_guidance}\n"
                        f"Violations: {'; '.join(validation.reasons)}\n"
                        f"Rewrite your response from scratch, addressing "
                        f"these violations while preserving the useful content.\n"
                        f"[/OLI ENFORCEMENT]"
                    )},
                ]
                full_response = ""
                async for retry_chunk in ollama.chat_stream(
                    retry_messages, think=use_think,
                    think_level=think_level, web_mode="off",
                ):
                    if retry_chunk.get("done"):
                        break
                    retry_content = retry_chunk.get("content", "")
                    full_response += retry_content
                    yield {"done": False, "content": retry_content}

                validation = validate_output(full_response, oli_mode, is_retry=True)

            # Build enforcement flags from the REAL classification (not defaults).
            # This tells the frontend whether review_mode was active for this turn.
            real_enforcement = EnforcementFlags(
                claim_admissibility_required=(oli_mode == OLIMode.ON),
                review_mode=(classification.function == MessageFunction.CORPUS_REVIEW),
            )

            # Build final metadata (with REAL classification/drift, not defaults)
            meta: dict = {
                "done": True,
                "content": "",
                "classification": classification.model_dump(),
                "drift_estimate": drift.model_dump(),
                "enforcement": real_enforcement.model_dump(),
                "full_response": full_response,
                "turn": turn,
                "context_usage": context_usage,
                "oli_validation": {
                    "status": validation.status.value,
                    "flag_count": len(validation.flags),
                    "regenerated": is_retry,
                },
            }
            if drift_assessment:
                meta["drift_window"] = {
                    "composite": drift_assessment["composite"],
                    "severity": drift_assessment["severity"].value,
                    "dampening": drift_assessment["dampening"].value,
                    "window_size": drift_assessment["window_size"],
                }
            if match_result:
                meta["match_result"] = {
                    "auto_activate": [m.model_dump() for m in match_result.auto_activate],
                    "candidates": [m.model_dump() for m in match_result.candidates],
                    "wrapped_spans": [w.model_dump() for w in match_result.wrapped_spans],
                    "correction_hint": match_result.correction_hint,
                }
            meta["retrieval"] = {
                "full_text": len(full_text_slabs),
                "reference_full": reference_count,
                "catalog": len(catalog_slabs),
                "retrieved_count": len(retrieved_ids),
                "signals": signal_counts,
            }
            if frame_state:
                meta["frame_summary"] = {
                    "active_nodes": frame_state.active_nodes[:12],
                    "mismatch_score": frame_state.mismatch_score,
                    "conflict_count": len(frame_state.conflicts),
                }
            if validation.flags:
                meta["oli_flags"] = [
                    {
                        "layer": f["layer"],
                        "name": f["name"],
                        "pattern": f["pattern"],
                        "overridable": f.get("overridable", True),
                        "severity": f.get("severity", 0.5),
                    }
                    for f in validation.flags
                ]
            if validation.status == ValidationStatus.BLOCK:
                meta["oli_blocked"] = True
                meta["oli_block_reasons"] = validation.reasons
            if gauntlet_result and gauntlet_result.fired:
                meta["gauntlet"] = {
                    "verdict": gauntlet_result.verdict,
                    "claim_summary": gauntlet_result.claim_summary,
                    "alternatives": gauntlet_result.alternatives,
                    "counterfactuals": gauntlet_result.counterfactuals,
                    "friction_note": gauntlet_result.friction_note,
                    "push_event_id": gauntlet_result.push_event_id,
                }
            yield meta
        else:
            content = chunk.get("content", "")
            full_response += content
            yield {"done": False, "content": content}
