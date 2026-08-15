"""Chat auto-titling — derive a human-readable label from live frame state.

Phase 2B: the sidebar was filling with timestamp-labelled chats ("Chat 22:24:24"),
which is useless for navigation once you have more than a handful. This service
produces short descriptive titles by combining:

  1. Top salience_smoothed nodes in the current frame (the "what's hot" signal)
  2. Human-readable labels via corpus lookup + tentative concept registry
  3. First user message as last-resort context when the frame is still cold

Gate slabs, epistemic-floor slabs, and other system-level nodes are stripped
because they're always-on and reveal nothing about the chat's topic. What
remains is user-facing content: concepts, anchors, domain-specific slabs.

Title format: up to 3 labels joined with " · ", truncated to ~50 chars. If
the frame has no interesting salience yet (early turns, generic chatter),
fall back to the first user message's first sentence.
"""
from __future__ import annotations
from typing import Optional

from ..models.schemas import FrameState, Chat, ChatMessage

# Node id prefixes/patterns that are system-level and shouldn't appear in
# chat titles — these are always active via base_set injection and have no
# topical signal. Tentative concepts and mined/user anchors/slabs pass
# through unfiltered.
_SYSTEM_PREFIXES = (
    "SLAB_OLI_",
    "SLAB_PUSHBACK_",
    "SLAB_CLAIM_",
    "SLAB_EPISTEMIC_",
    "SLAB_SOVEREIGN_",
    "SLAB_REGULATION_",
    "ANCHOR_OLI_",
    "ANCHOR_REGULATION_",
    "ANCHOR_SENSOR_",
    "ANCHOR_COHERENCE_",
    "ANCHOR_HANDWAVE_",
    "ANCHOR_HUMAN_PROMOTION_",
    "ANCHOR_EXTERNAL_",
    "ANCHOR_KEEP_UP_",
    "ANCHOR_ABSENCE_",
    "ANCHOR_GHOST_",
    "ANCHOR_SENTENCE_",
    "ANCHOR_NOT_MY_CIRCUS_",
    "BUNDLE_REGULATION_",
    "BUNDLE_GOVERNANCE_",
    "BUNDLE_OBLIGATION_",
    "BUNDLE_ARCHER_",
    "confidence_gate_",
    "function_gate_",
    "explicitness_gate_",
)

_MAX_TITLE_LEN = 50
_MAX_LABELS = 3


def _is_system_node(node_id: str) -> bool:
    return any(node_id.startswith(p) for p in _SYSTEM_PREFIXES)


def _label_for(node_id: str, corpus, tentative_registry: dict) -> Optional[str]:
    """Resolve a node_id to a short human label, or None if unresolvable.

    Priority:
      1. Corpus anchor canonical_phrase (often already 1-3 words)
      2. Corpus slab title (truncated)
      3. Corpus bundle first intent (truncated)
      4. Tentative registry concept name (from concept_detection)
      5. None (caller skips)
    """
    if corpus and node_id in corpus.anchors:
        return corpus.anchors[node_id].canonical_phrase
    if corpus and node_id in corpus.slabs:
        title = corpus.slabs[node_id].title or ""
        return title[:30] if title else None
    if corpus and node_id in corpus.bundles:
        bundle = corpus.bundles[node_id]
        if bundle.payload and bundle.payload.intent:
            return bundle.payload.intent[0][:30]
        return None
    # Tentative registry: maps concept_name -> {id: node_id, ...}.
    # We need the reverse: node_id -> concept_name.
    for concept_name, info in tentative_registry.items():
        if info.get("id") == node_id:
            return concept_name
    return None


def _titlecase(s: str) -> str:
    """Light Title Case — capitalize words >2 chars, keep short words lower."""
    parts = s.split()
    return " ".join(
        p[:1].upper() + p[1:] if len(p) > 2 else p
        for p in parts
    )


def derive_title(
    frame: Optional[FrameState],
    corpus,
    tentative_registry: dict,
    first_user_message: Optional[str] = None,
) -> str:
    """Generate a chat title from current state. Never returns empty string.

    Selection algorithm:
      1. Rank active nodes by salience_smoothed (higher = more chat-relevant)
      2. Drop system-level nodes via prefix check
      3. Resolve top N to human labels, skipping unresolvable
      4. Join with " · " up to _MAX_TITLE_LEN chars
      5. If fewer than 2 labels survived, fall back to first user message
    """
    if frame and frame.salience_smoothed:
        ranked = sorted(
            frame.salience_smoothed.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )
        labels: list[str] = []
        seen_lower: set[str] = set()
        for node_id, salience in ranked:
            if salience < 0.15:  # below this, not really "active" for our purposes
                break
            if _is_system_node(node_id):
                continue
            label = _label_for(node_id, corpus, tentative_registry)
            if not label:
                continue
            key = label.lower().strip()
            if not key or key in seen_lower:
                continue
            seen_lower.add(key)
            labels.append(label.strip())
            if len(labels) >= _MAX_LABELS:
                break

        if len(labels) >= 2:
            candidate = " · ".join(_titlecase(l) for l in labels)
            if len(candidate) > _MAX_TITLE_LEN:
                # Trim labels one by one from the end
                while labels and len(" · ".join(_titlecase(l) for l in labels)) > _MAX_TITLE_LEN:
                    labels.pop()
                candidate = " · ".join(_titlecase(l) for l in labels)
            if candidate:
                return candidate

    # Fallback: first user message's first sentence
    if first_user_message:
        snippet = first_user_message.strip().split("\n")[0]
        # First sentence up to punctuation
        for sep in (". ", "? ", "! "):
            if sep in snippet:
                snippet = snippet.split(sep, 1)[0] + sep[0]
                break
        if len(snippet) > _MAX_TITLE_LEN:
            snippet = snippet[: _MAX_TITLE_LEN - 1].rstrip() + "…"
        return snippet or "New Chat"

    return "New Chat"


def should_auto_title(chat: Chat, message_count: int) -> bool:
    """Whether auto-titling should run for this chat on this turn.

    Triggers only when:
      - Title is the default "New Chat" or starts with "Chat " (timestamp form)
      - At least 3 messages exist (enough signal for the frame)
      - But within the first ~8 turns (after that the title is probably
        stable and re-deriving from a drifted frame gives worse results)

    User-set titles (anything else) are left alone forever.
    """
    if not chat or not chat.title:
        return True
    default_forms = chat.title == "New Chat" or chat.title.startswith("Chat ")
    return default_forms and 3 <= message_count <= 16
