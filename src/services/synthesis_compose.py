"""Synthesis Step 2 — compose layer.

Turns a ``SynthesisResult`` (the structured selection output of
``synthesis.synthesize``) into a system prompt for the chat pipeline.

Step 1 was data manipulation — graph walk, embedding match, tier
packing. Step 2 is prompt engineering — wording the surfaced
content so the chat model treats each tier with the right
epistemic posture and preserves dialectic where the corpus has it.
Separate concerns, separate tuning surface: confidence thresholds
live in ``synthesis.py``; "stay grounded in this content" wording
lives here.

Architecture: the composed prompt becomes the system prompt for
one chat turn. The existing Mirror layers (OLI, claim gate,
anti-self-sealing) enforce that the model stays grounded in the
surfaced corpus. No separate hallucination guard is added here.

The unsurfaced count signal is included so anti-self-sealing has
structured input to acknowledge the gap rather than papering over
it. Without that signal the model has no way to know what's missing.

Citation behaviour: by default, no inline citations — clean
Karpathy-style prose. ``include_citations=True`` (e.g. when the
user explicitly asks for sources) appends a citation appendix and
instructs the model to use [N] markers.

Step 3 (chat pipeline integration) will detect synthesis intent in
the user's message, call ``synthesize()`` and then this module to
build the system prompt, route through the normal chat pipeline.
"""
from __future__ import annotations

from .synthesis import SynthesisResult, SelectedNode


# ── Section composers ─────────────────────────────────────────────


def _fmt_node(n: SelectedNode, max_text: int = 600) -> str:
    """Format a single SelectedNode as a prompt-friendly bullet.

    Caps text at ``max_text`` chars to keep the composed prompt
    inside the chat ctx budget. Slabs are typically the long ones;
    anchors and bundles are short.
    """
    text = n.text.strip()
    if len(text) > max_text:
        text = text[:max_text] + "..."
    return f"  [{n.node_type}] {text}"


def _compose_header(result: SynthesisResult) -> str:
    return (
        "You are answering the user's query using a curated knowledge "
        "corpus. The content below has been selected via tiered "
        "argumentative retrieval — it represents the corpus's most "
        "relevant material, organised by epistemic role.\n\n"
        f"USER QUERY: {result.query!r}"
    )


def _compose_seeds(seeds: list[SelectedNode]) -> str:
    if not seeds:
        return ""
    lines = [
        "=== DIRECT MATCHES (corpus content the query directly maps to) ===",
        "",
        "These nodes are the embedding-matched starting points for this "
        "synthesis. Treat them as core to the question:",
        "",
    ]
    for n in seeds:
        lines.append(_fmt_node(n))
    return "\n".join(lines)


def _compose_conflicts(
    conflicts: list[SelectedNode],
    partner_lookup: dict[str, str],
) -> str:
    """Compose the CONFLICTS section.

    Each conflict node carries ``conflicts_with`` as an ID — the
    partner anchor that's already in the candidate set (seeds or
    tier-1 supported). We resolve that ID to the partner's text via
    ``partner_lookup`` so the LLM sees the actual phrase / title
    rather than an opaque mined_anchor_xxxxxxxx_v1 ID.
    """
    if not conflicts:
        return ""
    lines = [
        "=== DIALECTIC / DIRECT OPPOSITION ===",
        "",
        "The corpus contains content that DIRECTLY OPPOSES the thesis. "
        "When discussing the topic, surface this opposition explicitly — "
        "the speaker has positioned this as wrong/incompatible with their "
        "own view. Do NOT collapse the opposition into a single thesis.",
        "",
    ]
    for n in conflicts:
        lines.append(_fmt_node(n))
        if n.conflicts_with:
            partner_text = partner_lookup.get(n.conflicts_with, "").strip()
            if partner_text:
                if len(partner_text) > 180:
                    partner_text = partner_text[:180] + "..."
                lines.append(f"      [in opposition to: {partner_text}]")
            else:
                lines.append(f"      [in opposition to: {n.conflicts_with}]")
        # Surface the edge's mining-time justification — gives the LLM
        # the model's original "why this is a conflict" reasoning, not
        # just the bare existence of the edge.
        if n.via_edge_justification:
            lines.append(f"      [why: {n.via_edge_justification.strip()}]")
    return "\n".join(lines)


def _compose_tensions(tensions: list[SelectedNode]) -> str:
    if not tensions:
        return ""
    lines = [
        "=== DIALECTIC / PRODUCTIVE TENSION ===",
        "",
        "The corpus contains content in productive tension with the "
        "thesis — concepts that COUNTERBALANCE rather than reject. Both "
        "sides remain valid simultaneously (e.g., mercy and justice both "
        "have merit; neither cancels the other). When discussing these "
        "pairs, present both poles as valid; do not collapse them into "
        "a single position.",
        "",
    ]
    for n in tensions:
        lines.append(_fmt_node(n))
        # Surface the TENSIONS edge's mining-time justification when
        # present — gives the LLM the model's original "why these
        # counterbalance" reasoning.
        if n.via_edge_justification:
            lines.append(f"      [why: {n.via_edge_justification.strip()}]")
    return "\n".join(lines)


def _compose_supported(supported: list[SelectedNode]) -> str:
    if not supported:
        return ""
    lines = [
        "=== SUPPORTING DETAIL ===",
        "",
        "These nodes are structurally connected to the thesis (via "
        "SUPPORTS / INVOKES / LINKS / PARENT_OF edges) and provide "
        "supporting context:",
        "",
    ]
    for n in supported:
        lines.append(_fmt_node(n))
    return "\n".join(lines)


def _compose_tier2_supports(supports: list[SelectedNode]) -> str:
    if not supports:
        return ""
    lines = [
        "=== ADDITIONAL SUPPORT (lower priority) ===",
        "",
        "These nodes are reachable from the thesis but at lower "
        "priority. Use sparingly — they're context, not foundation.",
        "",
    ]
    for n in supports:
        lines.append(_fmt_node(n))
    return "\n".join(lines)


def _compose_tier2_divergent(divergent: list[SelectedNode]) -> str:
    if not divergent:
        return ""
    lines = [
        "=== ADJACENT CONTEXT (embedding-similar, not graph-connected) ===",
        "",
        "These nodes are similar to the query in meaning but not "
        "formally linked to the thesis in the corpus graph. Treat as "
        "'related but not directly tied' — surface only when genuinely "
        "relevant; acknowledge the looser connection if it matters.",
        "",
    ]
    for n in divergent:
        lines.append(_fmt_node(n))
    return "\n".join(lines)


def _compose_unsurfaced(result: SynthesisResult) -> str:
    """The structured "what we didn't show" signal.

    Without this, the model has no way to know what's missing and
    will paper over gaps. With this, Mirror's anti-self-sealing
    layer has concrete evidence to surface "the corpus has more"
    when the user's question reaches beyond the surfaced content.
    """
    u = result.unsurfaced
    parts: list[str] = []
    if u.tier1_supported_skipped:
        parts.append(
            f"{u.tier1_supported_skipped} additional supporting slabs/anchors"
        )
    if u.conflicts_skipped:
        parts.append(
            f"{u.conflicts_skipped} additional CONFLICTS edges "
            "(not topically aligned with this query)"
        )
    if u.tier2_supports_skipped:
        parts.append(f"{u.tier2_supports_skipped} secondary supporting nodes")
    if u.tier2_divergent_skipped:
        parts.append(f"{u.tier2_divergent_skipped} adjacent-context nodes")
    if not parts:
        return ""
    items = "\n".join(f"  - {p}" for p in parts)
    return (
        "=== NOT SURFACED HERE ===\n\n"
        "The corpus contains additional content not included due to "
        "selection budget:\n"
        f"{items}\n\n"
        "If your answer would benefit from this content, acknowledge "
        "explicitly that 'the corpus has more on this — I'm working "
        "from a curated subset' rather than papering over the gap. "
        "The user can ask you to expand if they need it."
    )


_INSTRUCTIONS_BASE = """=== SYNTHESIS INSTRUCTIONS ===

1. Stay grounded in the content above. If a claim isn't directly supported by the surfaced corpus content, mark it as inference (e.g., "this implies...", "the corpus suggests...") rather than presenting it as known.

2. When dialectic surfaces (CONFLICTS or TENSIONS), preserve it. CONFLICTS pairs are positioned in opposition — surface both sides explicitly. TENSIONS pairs counterbalance without rejecting — present both as valid simultaneously. Do NOT collapse opposing or counterbalancing views into a single thesis.

3. Match the corpus's framing. The user has coined specific terms (visible in the surfaced content); use them as named rather than translating to generic synonyms.

4. If the surfaced content doesn't cover what the user asked, acknowledge the gap rather than fabricating to fill it."""


_INSTRUCTIONS_NO_CITATIONS = """

5. No inline citations. Write clean prose — quote sparingly when verbatim wording matters, but do not annotate every claim with a source ID. The user has not asked for sources."""


_INSTRUCTIONS_WITH_CITATIONS = """

5. The user has asked for sources. Reference each substantive claim with a [N] marker matching the citation list at the end of the prompt. Place citations at the end of sentences, not mid-sentence."""


def _compose_instructions(*, include_citations: bool) -> str:
    if include_citations:
        return _INSTRUCTIONS_BASE + _INSTRUCTIONS_WITH_CITATIONS
    return _INSTRUCTIONS_BASE + _INSTRUCTIONS_NO_CITATIONS


def _compose_citations(result: SynthesisResult) -> str:
    """Citation appendix — human-readable names for every surfaced node.

    Only included when ``include_citations=True``. The model uses
    these to populate inline ``[N]`` markers in its response.
    """
    items: list[tuple[str, SelectedNode]] = []
    for n in result.seeds:
        items.append(("seed", n))
    for n in result.tier1_supported:
        items.append(("supporting" if n.via_edge != "TENSIONS" else "tension", n))
    for n in result.tier1_conflicts:
        items.append(("dialectic", n))
    for n in result.tier2_supports:
        items.append(("secondary", n))
    for n in result.tier2_divergent:
        items.append(("adjacent", n))
    if not items:
        return ""
    lines = ["=== CITATION DATA ===", ""]
    for i, (role, n) in enumerate(items, 1):
        display_name = (n.display_name or n.text.split("\n", 1)[0]).strip()[:120]
        lines.append(f"[{i}] ({role}, {n.node_type}) {display_name}")
    return "\n".join(lines)


# ── Top-level entry ───────────────────────────────────────────────


def compose_synthesis_prompt(
    result: SynthesisResult,
    *,
    include_citations: bool = False,
) -> str:
    """Build a system prompt from a SynthesisResult for chat-pipeline use.

    The prompt instructs the model to stay grounded in the surfaced
    corpus content, preserve dialectic where present, acknowledge
    unsurfaced content via the structured count signal, and (by
    default) write clean prose without inline citations.

    Designed to drop into the chat pipeline as the system prompt for
    a single synthesis turn. The existing Mirror / OLI / claim-gate /
    anti-self-sealing layers handle the honesty enforcement during
    generation; this module just sets up the input.
    """
    # Build a partner-text lookup for the CONFLICTS section. Conflict
    # partners are by construction already in seeds or tier1_supported
    # (that's how the dialectic surfaced — one side was a candidate),
    # so this lookup is sufficient. Falls back to ID display if a
    # partner isn't found.
    partner_lookup: dict[str, str] = {}
    for n in result.seeds:
        partner_lookup[n.id] = n.text
    for n in result.tier1_supported:
        partner_lookup[n.id] = n.text

    sections: list[str] = []
    sections.append(_compose_header(result))

    if result.seeds:
        sections.append(_compose_seeds(result.seeds))

    if result.tier1_conflicts:
        sections.append(_compose_conflicts(result.tier1_conflicts, partner_lookup))

    # Split tier-1 supported into TENSIONS partners (need dialectic
    # framing) vs structural (regular supporting context). The BFS
    # records via_edge per node, so we can sort them at compose time
    # without changing the synthesis selection layer.
    tensioned = [n for n in result.tier1_supported if n.via_edge == "TENSIONS"]
    structural = [n for n in result.tier1_supported if n.via_edge != "TENSIONS"]

    if tensioned:
        sections.append(_compose_tensions(tensioned))
    if structural:
        sections.append(_compose_supported(structural))

    if result.tier2_supports:
        sections.append(_compose_tier2_supports(result.tier2_supports))

    if result.tier2_divergent:
        sections.append(_compose_tier2_divergent(result.tier2_divergent))

    unsurfaced_section = _compose_unsurfaced(result)
    if unsurfaced_section:
        sections.append(unsurfaced_section)

    sections.append(_compose_instructions(include_citations=include_citations))

    if include_citations:
        cite_section = _compose_citations(result)
        if cite_section:
            sections.append(cite_section)

    return "\n\n".join(sections)
