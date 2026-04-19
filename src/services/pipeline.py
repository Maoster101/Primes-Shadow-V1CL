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

    # ── Step 3.5: Slab retrieval (RAG over REFERENCE slabs) ───────
    # CONSTITUTIONAL + CANONICAL slabs are always full-text; REFERENCE
    # slabs go through a multi-signal retrieval pipeline:
    #
    #   (a) Semantic similarity — cosine on nomic-embed-text against the
    #       user's message (slab_matcher.top_k_for).
    #   (b) Frame activation bridge — slabs currently active in the frame
    #       via non-base_set sources (anchor_match cascade, prior retrieval,
    #       user actions) with weight >= 0.5 promote to full-text, so the
    #       model sees the content the frame says is "live this turn".
    #   (c) Collection-name mentions — if the user names a collection
    #       (e.g. "vindiesel5"), pull all that collection's slabs. Handles
    #       abbreviations/proper nouns that semantic embeddings can't match.
    #   (d) Edge-traversal expansion — for each slab surfaced by (a-c),
    #       walk 1-hop via corpus.edges to add neighbors (SEQUENCE neighbors
    #       for narrative continuity, LINKS neighbors for anchor context).
    #
    # The four signals converge on a single ``retrieved_ids`` set; the
    # slab-split logic below promotes anything in it to full-text.
    # Capped at MAX_REFERENCE_FULL_TEXT to bound token budget.
    MAX_REFERENCE_FULL_TEXT = 25
    EDGE_EXPANSION_CAP = 10

    retrieved_ids: set[str] = set()
    signal_counts = {"semantic": 0, "frame": 0, "collection": 0, "edges": 0}

    # (a) Semantic retrieval
    if slab_matcher is not None and slab_matcher.has_cache():
        try:
            hits = await slab_matcher.top_k_for(user_text)
            sem_ids = {sid for sid, _score in hits}
            retrieved_ids.update(sem_ids)
            signal_counts["semantic"] = len(sem_ids)
        except Exception as exc:
            logger.warning("slab retrieval failed: %r", exc)

    # (b) Frame-activation bridge — any slab the frame says is live this
    # turn (anchor cascade, etc) with non-base_set source and meaningful weight.
    if frame_state is not None:
        before = len(retrieved_ids)
        for sid, weight in frame_state.active_slabs.items():
            if weight < 0.5:
                continue
            sources = frame_state.activation_sources.get(sid, [])
            # Skip base_set-only (those are CONSTITUTIONAL/CANONICAL or
            # ambient REFERENCE seeding that hasn't been specifically engaged).
            if sources and all(s.source_type == "base_set" for s in sources):
                continue
            if frame_manager and sid in frame_manager.corpus.slabs:
                retrieved_ids.add(sid)
        signal_counts["frame"] = len(retrieved_ids) - before

    # (c) Collection-name detection — pull all slabs from collections the
    # user named. Two match modes:
    #   1. Full-name substring: "vindiesel5" matches collection vindiesel5
    #   2. Version-suffix abbreviation: "v5" or "version 5" matches any
    #      collection whose ID ends with that number (vindiesel5, vinN_v5)
    # The second mode is critical because users abbreviate mining passes
    # ("compare v5 with v6") and those short forms don't appear as
    # substrings in full collection IDs.
    if frame_manager:
        try:
            import re as _re
            from ..api import deps as _deps
            low = (user_text or "").lower()
            active_ids = list(_deps.registry.active_ids)
            mentioned = set()
            # Full-name match
            for cid in active_ids:
                if len(cid) >= 3 and cid.lower() in low:
                    mentioned.add(cid)
            # Version-suffix match: "v5", "version 5", "v_5" → find
            # collections whose trailing digits equal the mentioned number.
            # "vindiesel5" → trailing "5"; "vin_diesel_v2" → trailing "2".
            # Pre-extract trailing digits once per collection.
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

    # (d) 1-hop edge traversal — for each slab already in retrieved_ids,
    # add its neighbor slabs via corpus.edges up to a cap. Prioritizes
    # SEQUENCE and LINKS edges (narrative and semantic), deprioritizes
    # SUPPORTS (usually bundle→slab, less useful for context expansion).
    if retrieved_ids and frame_manager:
        seed_ids = set(retrieved_ids)
        expansion: set[str] = set()
        prioritized_types = ("SEQUENCE", "LINKS", "INVOKES", "SUPPORTS", "PARENT_OF")
        for etype in prioritized_types:
            if len(expansion) >= EDGE_EXPANSION_CAP:
                break
            for e in frame_manager.corpus.edges.values():
                if e.type.value != etype:
                    continue
                if len(expansion) >= EDGE_EXPANSION_CAP:
                    break
                neighbor = None
                if e.from_node in seed_ids and e.to_node not in seed_ids:
                    neighbor = e.to_node
                elif e.to_node in seed_ids and e.from_node not in seed_ids:
                    neighbor = e.from_node
                if neighbor and neighbor in frame_manager.corpus.slabs:
                    expansion.add(neighbor)
        retrieved_ids.update(expansion)
        signal_counts["edges"] = len(expansion)

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
