"""Context packing service — §20.4.

Assembles the prompt payload sent to GPT-OSS 20B each turn.
Budget: OLI constitutional prompt + runtime header + chat slice
fill the full 128k context window GPT-OSS 20B supports.

We target the full 128k for marathon sessions — eviction only kicks in
when packed chat history + system prompt exceed that.
"""
from __future__ import annotations
from typing import Optional

from ..models.schemas import (
    ChatMessage, RuntimeHeader, FrameState, FrameStateSummary, GateState,
)
from ..models.enums import OLIMode
from ..models.schemas import Slab
from ..prompts.oli_constitutional import OLI_CONSTITUTIONAL_PROMPT, BMD_SCAFFOLD


# Rough token estimate: ~4 chars per token for English text
CHARS_PER_TOKEN = 4

# Context budget matches the Ollama num_ctx setting.
# Default 32k tokens balances quality with VRAM (KV cache for 32k on
# a 20B model uses ~2-3GB). Override with PS_NUM_CTX env var.
import os
TARGET_CONTEXT_TOKENS = int(os.environ.get("PS_NUM_CTX", "32768"))
TARGET_CONTEXT_CHARS = TARGET_CONTEXT_TOKENS * CHARS_PER_TOKEN


def build_system_prompt(
    oli_mode: OLIMode = OLIMode.OFF,
    runtime_header: Optional[RuntimeHeader] = None,
    base_set_slabs: Optional[list[Slab]] = None,
) -> str:
    """Build the full system message: constitutional prompt + base set + runtime header.

    Phase 5: base_set_slabs injects CONSTITUTIONAL/CANONICAL slab text into
    the system prompt so the model has access to layer rules and domain context
    from turn 0, before any anchor fires.
    """
    parts = []

    if oli_mode == OLIMode.ON:
        parts.append(OLI_CONSTITUTIONAL_PROMPT)
    else:
        # OLI OFF: still inject BMD scaffold + OLI-off rules
        parts.append(BMD_SCAFFOLD)

    parts.append(_build_base_system())

    # §Phase 5 — Base set slab injection
    if base_set_slabs:
        parts.append(_format_base_set_slabs(base_set_slabs))

    if runtime_header:
        parts.append(_format_runtime_header(runtime_header))

    return "\n\n".join(parts)


def build_messages(
    system_prompt: str,
    chat_messages: list[ChatMessage],
    frame_state: Optional[FrameState] = None,
    max_context_chars: int = TARGET_CONTEXT_CHARS,
) -> list[dict]:
    """Assemble the messages list for the Ollama chat API.

    Trims older messages to fit the context budget.
    System prompt always included in full.
    """
    messages = [{"role": "system", "content": system_prompt}]
    system_chars = len(system_prompt)

    # Optional frame state context injection
    frame_context = ""
    if frame_state and frame_state.active_nodes:
        frame_context = _format_frame_context(frame_state)
        system_chars += len(frame_context)

    budget = max_context_chars - system_chars
    if budget < 2000:
        budget = 2000  # Minimum space for at least a few turns

    # Pack messages from most recent backward
    packed: list[dict] = []
    used = 0
    for msg in reversed(chat_messages):
        msg_chars = len(msg.content) + 20  # overhead for role/metadata
        if used + msg_chars > budget:
            break
        packed.append({"role": msg.role, "content": msg.content})
        used += msg_chars

    packed.reverse()

    if frame_context:
        messages.append({"role": "system", "content": frame_context})

    messages.extend(packed)
    return messages


def _build_base_system() -> str:
    return (
        "You are the Mirror — a semantic reasoning engine inside Prime's Shadow. "
        "You generate, interpret, propose activations, estimate salience, and suggest "
        "tentative graph structure. You do NOT own gate decisions, corpus mutation, "
        "or persistence — the application code does.\n\n"
        "OUTPUT FORMATTING (conversational responses):\n"
        "- Use markdown for headings, lists, bold, and tables.\n"
        "- When you produce structured data (JSON objects, arrays, configs, schemas, "
        "plans-as-data), ALWAYS wrap it in a fenced code block with a language tag, "
        "e.g. ```json ... ``` or ```yaml ... ```. Never emit raw JSON as prose.\n"
        "- Code goes in fenced blocks with the appropriate language tag.\n\n"
        "OUTPUT FORMATTING (tool calls / structured extraction):\n"
        "When the application asks you for structured extraction via a schema prompt, "
        "return ONLY raw JSON — no markdown fences, no comments, no explanation — "
        "because the code is parsing you directly. Schema-prompt responses are the "
        "ONE exception to the fence-your-JSON rule above."
    )


def _format_runtime_header(header: RuntimeHeader) -> str:
    """Format §26.3 runtime control header as structured text for the model."""
    lines = [
        "[RUNTIME HEADER]",
        f"oli_mode: {header.oli_mode.value}",
        f"oli_version: {header.oli_version}",
        (f"layer_control: max_layer={header.layer_control['max_layer']}, "
         f"l4_posture={header.layer_control['l4_posture']}"),
        (f"gate: function={header.gate_state.message_function.value}, "
         f"anchor_resolution={header.gate_state.anchor_resolution_allowed}, "
         f"confidence={header.gate_state.confidence_flag}"),
        (f"frame: active_nodes={header.frame_state_summary.active_nodes}, "
         f"tension_pairs={header.frame_state_summary.high_tension_pairs}, "
         f"mismatch={header.frame_state_summary.mismatch_score}"),
        (f"drift: affect={header.drift_estimate.affect_density}, "
         f"volatility={header.drift_estimate.claim_volatility}, "
         f"rigor_drop={header.drift_estimate.rigor_drop}, "
         f"domain={header.drift_estimate.domain_mode.value}, "
         f"composite={header.drift_estimate.window_composite}"),
        (f"enforcement: admissibility={header.enforcement_flags.claim_admissibility_required}, "
         f"degradation={header.enforcement_flags.degradation_flag}, "
         f"pushback={header.enforcement_flags.pushback_required}, "
         f"dampening={header.enforcement_flags.dampening_level.value}"),
    ]

    # OP_01 operator state — feature-based wrapped span inventory.
    # Only rendered when something interesting happened, so neutral
    # turns don't bloat the header. Each wrapped span shows its text,
    # derived primary_reading, and the full feature set so the Mirror
    # can compose interpretations across multiple overlapping signals
    # (GPT-OSS 20B doesn't have RL-learned operator pragmatics; we
    # spell out the contract every turn).
    op = header.operator_state
    if op.wrapped_spans or op.correction_hint:
        lines.append("operator_state (OP_01 — asterisk wrap):")
        for w in op.wrapped_spans:
            text = w.get("text", "")
            reading = w.get("primary_reading", "semantic_depth")
            feats = w.get("features", [])
            hits = w.get("anchor_hits", [])
            hit_ids = [h.get("anchor_id") for h in hits] if hits else []
            if hit_ids:
                lines.append(
                    f"  wrap[{w.get('span_index')}] {text!r}  "
                    f"reading={reading}  features={feats}  hits={hit_ids}"
                )
            else:
                lines.append(
                    f"  wrap[{w.get('span_index')}] {text!r}  "
                    f"reading={reading}  features={feats}"
                )
        if op.correction_hint:
            lines.append(
                f"  correction_hint: {op.correction_hint!r}  "
                f"# user left an UNCLOSED `*...` — self-correction of prior text"
            )
        # Inline interpretation contract — concise version of the rules
        # so the model has them at the point of use. The full spec lives
        # in the constitutional prompt.
        lines.append(
            "  # HOW TO READ wrapped spans:"
        )
        lines.append(
            "  #   explicit_invocation        = activate the matched anchor"
        )
        lines.append(
            "  #   ambiguous_invocation       = ASK which anchor; do not silently pick"
        )
        lines.append(
            "  #   self_correction            = treat as retraction/amendment of prior text"
        )
        lines.append(
            "  #   self_directed_frustration   = user is frustrated AT THEMSELVES; "
            "do NOT apologise or pivot; offer light continuity / gentle reassurance; "
            "continue the thread"
        )
        lines.append(
            "  #   model_directed_exasperation = user is mildly/affectionately "
            "exasperated AT YOU (DEFAULT model-directed register); match register "
            "light and self-aware, do NOT grovel, do NOT course-correct aggressively"
        )
        lines.append(
            "  #   model_directed_complaint    = user is HARSHLY frustrated AT YOU "
            "(harsh_descriptor feature present — rare); take seriously, check prior turn, adjust"
        )
        lines.append(
            "  #   world_directed_frustration  = user is frustrated at something EXTERNAL "
            "(not you, not themselves); commiserate lightly, engage with the target if relevant, "
            "do NOT take personally"
        )
        lines.append(
            "  #   ambient_sigh                = bare sigh interjection (urgh/ugh/oof); "
            "vent into the air, acknowledge lightly or not at all, continue"
        )
        lines.append(
            "  #   performed_imitation        = quoted/meme/imitative stylization; "
            "if you recognise the reference engage with it, otherwise read as strong affect"
        )
        lines.append(
            "  #   stage_direction            = paralinguistic emote, register as affect "
            "and match conversational register, do not dissect"
        )
        lines.append(
            "  #   strong_affect              = exclamation/intensity, register tone"
        )
        lines.append(
            "  #   semantic_depth             = unpack the layered meaning"
        )
        lines.append(
            "  # Pragmatic modifiers (compose with any reading):"
        )
        lines.append(
            "  #   rhetorical_question = interrogative + affect — user is NOT asking "
            "for an answer, they are venting/teasing; do NOT answer literally"
        )
        lines.append(
            "  #   sigh_interjection   = urgh/ugh/argh/hmph/oof present — affective marker"
        )
        lines.append(
            "  #   harsh_descriptor    = escalates model_directed_affect from "
            "exasperation → complaint"
        )
        lines.append(
            "  # Features compose — a span can be stage_direction AND performed_imitation "
            "AND carry a reference simultaneously. Use the full feature set, not just "
            "primary_reading."
        )

    lines.append("[/RUNTIME HEADER]")
    return "\n".join(lines)


def _format_base_set_slabs(slabs: list[Slab]) -> str:
    """§Phase 5 — Format base set slab canonical text for system prompt injection.

    Each slab's canonical_text is injected under a [CORPUS BASE SET] block so
    the model has access to constitutional layer rules and domain context from
    turn 0. Slabs are already dependency-sorted by CorpusStore.base_set_slabs().
    """
    if not slabs:
        return ""
    lines = ["[CORPUS BASE SET]"]
    for slab in slabs:
        lines.append(f"--- {slab.id} ({slab.type.value}) ---")
        if slab.title:
            lines.append(f"# {slab.title}")
        # Inject canonical text (truncate very long slabs to preserve budget)
        text = slab.canonical_text or ""
        if len(text) > 8000:
            text = text[:8000] + "\n[... truncated ...]"
        lines.append(text)
        lines.append("")  # blank line separator
    lines.append("[/CORPUS BASE SET]")
    return "\n".join(lines)


def _format_frame_context(frame: FrameState) -> str:
    """Compact frame state for context injection."""
    lines = ["[ACTIVE FRAME]"]
    for node_id in frame.active_nodes[:24]:  # §21 cap
        sal = frame.salience_now.get(node_id, 0)
        lines.append(f"  {node_id}: salience={sal:.2f}")
    if frame.conflicts:
        lines.append("conflicts:")
        for c in frame.conflicts[:8]:
            lines.append(f"  {c.node_a} <-> {c.node_b}: tension={c.tension:.2f}")
    lines.append(f"mismatch_score: {frame.mismatch_score:.2f}")
    lines.append("[/ACTIVE FRAME]")
    return "\n".join(lines)
