"""Core runtime pipeline — §6.

On each user turn: ingest → classify → match → frame+drift → header → respond.
LLM = sensor, code = actuator throughout.
"""
from __future__ import annotations
import asyncio
import json
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

_event_log = EventLog()


async def classify_message(user_text: str) -> MessageClassification:
    """§7 — Function gate. Classify the dominant function of a user message."""
    prompt = FUNCTION_GATE_PROMPT + json.dumps(user_text)
    try:
        result = await ollama.structured_extract(prompt)
        classification = MessageClassification(**result)
    except Exception:
        classification = MessageClassification(
            function=MessageFunction.NEUTRAL,
            confidence=0.3,
            explicit=False,
            notes="Classification failed — defaulting to neutral with low confidence",
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
    except Exception:
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
    except Exception:
        classification = MessageClassification(
            function=MessageFunction.NEUTRAL,
            confidence=0.3,
            explicit=False,
            notes="Combined classify+drift failed — defaults applied",
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

    return RuntimeHeader(
        oli_mode=oli_mode,
        gate_state=gate,
        frame_state_summary=frame_summary,
        drift_estimate=drift,
        enforcement_flags=EnforcementFlags(
            claim_admissibility_required=(oli_mode == OLIMode.ON),
            dampening_level=dampening,
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

    # ── Step 4: Runtime header + context packing (instant) ───────
    header = build_runtime_header(
        oli_mode, default_classification, frame_state, default_drift,
        DampeningLevel.NONE,
        match_result=match_result,
        anchor_hits_context=anchor_hits_ctx,
    )

    base_slabs = None
    if frame_manager:
        base_slabs = frame_manager.corpus.base_set_slabs(oli_mode)
    system_prompt = build_system_prompt(oli_mode, header, base_set_slabs=base_slabs)

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

            # Build final metadata (with REAL classification/drift, not defaults)
            meta: dict = {
                "done": True,
                "content": "",
                "classification": classification.model_dump(),
                "drift_estimate": drift.model_dump(),
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
