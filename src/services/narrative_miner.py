"""Narrative miner — outline-first mining for cohesive single-author content.

Sibling of OutlineMiner (documents / papers) and ConversationMiner
(chat logs). Where the doc miner asks "what are the SECTIONS, in
order?", the narrative miner asks "what are the BEATS, in order?" — the
movements, turning points and developments a story / essay / pitch /
scene works through. Same outline-first shape:

  Pass 1 — beat outline.  Identify the ordered beats. A piece that
           carries explicit headings (a structured design doc) is
           parsed deterministically; plain prose gets one LLM
           "identify the beats in order" call. Either route yields an
           ordered list of beats, each a CONTIGUOUS span of the source.

  Pass 2 — drill.  Mine each beat span into slabs + anchors. Small,
           parallel calls. Every slab is tagged with its beat.

  Pass 3 — beat summaries.  Backfill a 1-3 sentence summary per beat —
           the pillar ``summary`` a reader sees at top zoom.

There is deliberately NO cross-relevance pass. A narrative is a
SEQUENCE, not a graph of interacting subsystems: beat-to-beat relations
ARE the SEQUENCE spine, and an entity recurring across beats is already
carried by the anchor LINKS edges. An embedding cross-beat pass would
mostly re-derive that entity continuity as noise.

Edges: the SEQUENCE spine (beats in document order) + dialectic
CONFLICTS / TENSIONS (foil pairs) + slab→anchor LINKS. Plus the
pipeline-integrated anchor consolidation. Output shape matches the
other miners so /push-mined and the pillar-overlay machinery work
unchanged — the ``outline`` field is the beat skeleton, and a
narrative-mined collection's pillar overlay is its acts/beats.

The outline-first machinery (heading parse, beat-boundary location,
span drill, summary backfill, tree assembly, SEQUENCE spine) is shared
with OutlineMiner — see outline_miner.py. This module owns only the
narrative-genre prompts and the dialectic-edge pass.
"""

from __future__ import annotations

import asyncio
import logging
import re

from .convo_miner import MiningProposal, EdgeProposal, deduplicate_proposals
from . import ollama, mining_progress
from .outline_miner import (
    parse_heading_outline,
    select_tiers,
    resolve_chapters,
    _outline_via_llm,
    _drill_chapter,
    _summarize_pillar,
    _slab_summary_item,
    build_sequence_edges,
    build_outline_tree,
)

logger = logging.getLogger(__name__)

# Extraction concurrency is model-aware — see ollama.mining_parallelism().
# Read at each gather site so a local VRAM-bound default (2, safe on a 16GB
# card) or a wide hosted fan-out is chosen from the active extract model.


# ─── Pass 1 — beat outline (narrative-genre prompt) ──────────────────────
#
# Consumed by outline_miner._outline_via_llm via its ``system=`` param.
# The JSON key MUST be "sections" — that's what _outline_via_llm parses;
# the prose calls them beats, which is what the model needs to hear.

NARRATIVE_BEAT_SYSTEM = """You are reading a NARRATIVE — a story, scene, \
essay, pitch, or other cohesive single-author piece — and producing its \
BEAT STRUCTURE: the ordered list of beats it works through, start to end.

This is a comprehension task. Read the piece and identify the 4-15 beats \
it moves through, IN ORDER. A beat is a coherent narrative or argumentative \
unit — a setup, an inciting moment, a complication, a reversal, a climax, a \
resolution; or, for an essay, a claim, its development, a counter-position, \
a synthesis. For each beat give:
  - label: a 2-6 word headline NAMING the beat by what happens in it
  - summary: 1-2 sentences on what happens / what is argued in this beat
  - starts_with: a SHORT verbatim quote (6-12 words) copied EXACTLY from
    the text marking where this beat begins. It is used to locate the
    beat boundary — it must appear verbatim in the text.

RULES
1. Beats are CONTIGUOUS and IN ORDER — beat 2 begins where beat 1 ends.
   Together they tile the whole piece.
2. 4-15 beats. Fewer for short pieces.
3. starts_with must be copied character-for-character from the text.
4. Name beats by what HAPPENS — "The Pact Is Broken", not "Beat 4".

Output ONLY valid JSON (the key is "sections"):
{
  "sections": [
    {"label": "...", "summary": "...", "starts_with": "..."}
  ]
}
"""


# ─── Pass 2 — per-beat extraction (narrative-genre prompt) ───────────────
#
# Consumed by outline_miner._drill_chapter via its ``system=`` param.
# Same JSON contract as the doc miner's drill prompt.

NARRATIVE_DRILL_SYSTEM = """You are mining ONE beat of a narrative into \
corpus objects for a knowledge graph. The beat's place in the piece is \
given — use it as context.

Extract:

1. SLABS — the primary unit. Each slab is 2-8 complete sentences capturing
   one coherent moment, claim, image, or development within this beat.
   - title: short and specific (3-7 words)
   - canonical_text: complete sentences. Condense rich prose faithfully;
     reconstitute terse or schematic passages into full sentences. The
     slab is stored as-is and must stand on its own.
   - A long beat yields several slabs; a short one yields one.

2. ANCHORS — 2-8 word phrases naming distinctive entities the beat
   introduces: characters, places, factions, named objects, coined terms,
   technologies. Only things that would be referenced elsewhere — not
   every noun.
   - references_anchors on a slab: anchor phrases that slab involves.
   - When the beat sets one force against another — a protagonist vs an
     antagonist, a thesis vs the view it argues against — extract BOTH
     sides as anchors. The dialectic-edge pass needs both to surface a
     CONFLICTS / TENSIONS pair.

Match density to the beat. Do not manufacture objects to hit a count.

Give each slab a `description`: ONE dense sentence written FOR RETRIEVAL, like a
wiki index entry — what the slab establishes and the questions it answers, in the
third person about the slab, NOT a restatement of canonical_text.

Give each anchor a `description` too: ONE dense sentence stating its IDENTITY — what
the concept/entity IS, third person, self-contained, with no relational context and no
invoking characters. Aliases are lexical alternatives for the SAME referent (Harry /
the boy who lived); a distinct concept that merely co-occurs (a contested title, a
category term) is its own anchor, never folded in as an alias.

Output ONLY valid JSON:
{
  "slabs": [
    {"title": "...", "canonical_text": "...", "description": "one dense retrieval sentence",
     "references_anchors": ["..."], "confidence": 0.0-1.0,
     "justification": "..."}
  ],
  "anchors": [
    {"canonical_phrase": "...", "aliases": ["..."], "description": "one dense identity sentence", "confidence": 0.0-1.0}
  ]
}
"""


# ─── Dialectic edges (CONFLICTS / TENSIONS) ──────────────────────────────


DIALECTIC_PROMPT = """You are scanning a list of corpus proposals for FOIL PAIRS.

A foil pair is two concepts where the text positions one against the other:
  - one is the author's own concept; the other is the opposing force, default behaviour, or characterisation being argued against
  - or both are equally-valid concepts that pull in different directions and must be balanced (e.g. mercy vs justice)

Edge types you may emit:

- **CONFLICTS** — Direct opposition. One concept rejects, contradicts, or refutes the other. The text takes a stance that one side is WRONG / incompatible / mutually exclusive with the other.
  Examples: "individual sovereignty" CONFLICTS with "central authority mandate"; "Private Logic" CONFLICTS with "Internet Average"

- **TENSIONS** — Productive tension. Two concepts pull against each other but BOTH remain valid; they counterbalance rather than reject. Use this for dialectical pairs, ethical counterweights, design tradeoffs where both sides have merit.
  Examples: "mercy" TENSIONS with "justice"; "creative breadth" TENSIONS with "rigorous depth"; "user agency" TENSIONS with "system safety"

CONFLICTS vs TENSIONS distinction:
  - "X is wrong, Y is right" or "X and Y are incompatible" → CONFLICTS
  - "X and Y are both true and we have to balance them" → TENSIONS

Proposals (anchors only — slabs and bundles are not edge endpoints for this pass):
{proposals_text}

Return ONLY valid JSON. Do NOT emit other edge types — this pass is dialectic-only:

{{
  "edges": [
    {{
      "type": "CONFLICTS",
      "from": "exact canonical_phrase of source anchor",
      "to": "exact canonical_phrase of target anchor",
      "justification": "what makes this a foil pair",
      "confidence": 0.0-1.0
    }}
  ]
}}

If you find ZERO foil pairs, return ``{{"edges": []}}``. Don't manufacture pairs that aren't there — non-argumentative content (technical specs, descriptive narrative, declarative documents) often has no foil pairs at all. Returning [] is the correct answer for that content.

Use EXACT canonical_phrase from the proposals list. Maximum 12 dialectic edges (real foil pairs are usually rare even in argumentative content).
"""


async def extract_dialectic_edges(
    proposals: list[MiningProposal],
) -> list[EdgeProposal]:
    """Run a focused LLM pass to identify CONFLICTS / TENSIONS edges
    between anchor proposals.

    Operates on anchors only. Returns [] when there are fewer than 2
    anchors, when the model finds no foil pairs (the right answer for
    non-argumentative content), or when the call fails.
    """
    anchors = [p for p in proposals if p.proposal_type == "anchor"]
    if len(anchors) < 2:
        return []

    lines = []
    for a in anchors:
        just = (a.justification or "").strip()[:120]
        suffix = f"  — {just}" if just else ""
        lines.append(f"  - \"{a.canonical_phrase}\"{suffix}")
    prompt = DIALECTIC_PROMPT.replace("{proposals_text}", "\n".join(lines))

    try:
        resp = await ollama.structured_extract(prompt)
    except Exception as exc:
        logger.warning("Dialectic-edge pass failed: %r", exc)
        return []

    raw_edges = resp.get("edges") if isinstance(resp, dict) else None
    if not isinstance(raw_edges, list):
        return []

    out: list[EdgeProposal] = []
    for r in raw_edges:
        if not isinstance(r, dict):
            continue
        edge = EdgeProposal.from_raw(r, default_conf=0.65)
        # Drop unusable / self-loop edges.
        if edge is None or edge.from_label == edge.to_label:
            continue
        # Strict allowlist — the SEQUENCE / LINKS edges have their own paths.
        if edge.edge_type not in {"CONFLICTS", "TENSIONS"}:
            continue
        out.append(edge)

    if out:
        logger.info(
            "Dialectic-edge pass: emitted %d edges (%d CONFLICTS, %d TENSIONS)",
            len(out),
            sum(1 for e in out if e.edge_type == "CONFLICTS"),
            sum(1 for e in out if e.edge_type == "TENSIONS"),
        )
    return out


# ─── Heading-shape filter ────────────────────────────────────────────────
# Retained from the pre-outline-first miner because draft_manager imports
# it (the commit path uses it to drop heading-shaped anchor proposals).
# Not used by the outline-first mine() below — beat-span drilling doesn't
# surface section headings as anchors the way segment mining did.


def _looks_like_section_heading(phrase: str) -> bool:
    """Heuristic: does this phrase look like a section heading rather than
    a named concept the corpus should anchor?

    Three signals, any one triggers the filter:
      1. Two-or-more comma compound forms with conjunctions.
      2. Long Title-Case-dominant phrases (>8 words, most capitalised).
      3. Title-Case-everywhere phrases of >3 words with no lowercase
         function words.
      4. Short compound-conjunction headings ("Intent and Scope").
    """
    p = phrase.strip()
    if not p:
        return True
    comma_count = p.count(",")
    has_conj = bool(re.search(r"\b(and|or)\b", p, re.IGNORECASE))
    if comma_count >= 2 and has_conj:
        return True
    words = p.split()
    if len(words) > 8:
        uppercase_starts = sum(
            1 for w in words
            if w and w[0].isalpha() and w[0].isupper()
        )
        if uppercase_starts / len(words) >= 0.6:
            return True
    lowercase_function_words = {
        "a", "an", "the", "of", "and", "or", "in", "for", "to",
        "with", "on", "at", "by", "from", "as",
    }
    if len(words) > 3:
        all_titlecased = all(
            (w[0].isupper() if w[0].isalpha() else True)
            for w in words
        )
        has_function_word = any(
            w.lower() in lowercase_function_words for w in words
        )
        if all_titlecased and not has_function_word:
            return True
    if 2 <= len(words) <= 4:
        non_conj = [w for w in words if w.lower() not in {"and", "or"}]
        has_conj = any(w.lower() in {"and", "or"} for w in words)
        if has_conj and non_conj and all(
            (w[0].isupper() if w[0].isalpha() else True) for w in non_conj
        ):
            return True
    return False


# ─── Miner ───────────────────────────────────────────────────────────────


class NarrativeMiner:
    """Mine cohesive narrative / document text outline-first.

    Beat outline → per-beat drill → beat summaries → SEQUENCE + dialectic
    + LINKS edges → consolidation. Output dict shape matches the other
    miners so /push-mined and the pillar-overlay machinery are unchanged.
    """

    def __init__(self, corpus):
        self.corpus = corpus

    async def mine(
        self,
        raw_text: str,
        source_label: str = "narrative",
        min_confidence: float = 0.4,
        max_segment_chars: int = 1800,   # legacy param — outline-first
                                         # mining no longer segments by
                                         # char count; kept for API compat.
    ) -> dict:
        mining_progress.reset("narrative")

        # ── Pass 1 — beat outline ────────────────────────────────────
        mining_progress.set_phase("outlining", total=1)
        sections = parse_heading_outline(raw_text)
        outline_route = "headings"
        if not sections:
            outline_route = "llm_beats"
            sections = await _outline_via_llm(
                raw_text, system=NARRATIVE_BEAT_SYSTEM,
            )
        mining_progress.increment()

        if not sections:
            mining_progress.mark_error("no beats could be derived")
            return {
                "format": "narrative", "segments": 0, "outline": [],
                "proposals": [], "edges": [], "proposal_count": 0,
                "edge_count": 0, "source_label": source_label,
                "error": "Could not derive a beat structure from this text.",
            }

        top_level, beat_level = select_tiers(sections)
        beats, beat_top, beat_paths = resolve_chapters(
            sections, top_level, beat_level,
        )
        logger.info(
            "NarrativeMiner: route=%s, %d sections, %d beats (tiers %d/%d)",
            outline_route, len(sections), len(beats), top_level, beat_level,
        )

        # ── Pass 2 — drill each beat (parallel) ──────────────────────
        mining_progress.set_phase("beat_drill", total=len(beats))
        sem = asyncio.Semaphore(ollama.mining_parallelism())
        order_bases = [i * 1000 for i in range(len(beats))]

        async def _drill(idx: int):
            async with sem:
                r = await _drill_chapter(
                    beats[idx], beat_paths[idx], order_bases[idx],
                    min_confidence, system=NARRATIVE_DRILL_SYSTEM,
                )
                mining_progress.increment()
                return r

        results = await asyncio.gather(*[_drill(i) for i in range(len(beats))])

        all_slabs: list[MiningProposal] = []
        all_anchors: list[MiningProposal] = []
        all_links: list[EdgeProposal] = []
        for r in results:
            all_slabs.extend(r.slabs)
            all_anchors.extend(r.anchors)
            all_links.extend(r.links)

        canonical_anchors = deduplicate_proposals(all_anchors)
        canonical_anchors = [
            a for a in canonical_anchors if a.proposal_type == "anchor"
        ]

        all_proposals = canonical_anchors + all_slabs
        all_proposals = deduplicate_proposals(all_proposals)
        all_proposals = [
            p for p in all_proposals if p.confidence >= min_confidence
        ]
        all_proposals.sort(key=lambda p: (
            p.source_pairs[0] if p.source_pairs else 999999,
            p.proposal_type,
        ))

        slabs_by_path: dict[str, list[MiningProposal]] = {}
        for s in all_slabs:
            slabs_by_path.setdefault(s.source_topic, []).append(s)

        # ── Pass 3 — beat summaries ──────────────────────────────────
        mining_progress.set_phase("beat_summaries", total=len(beats) + 1)

        async def _sum_beat(beat, path: str) -> tuple[str, str]:
            async with sem:
                slabs = slabs_by_path.get(path, [])
                items = [_slab_summary_item(s) for s in slabs]
                summ = await _summarize_pillar(beat.label, items)
                mining_progress.increment()
                return path, summ

        beat_summaries: dict[str, str] = dict(await asyncio.gather(*[
            _sum_beat(b, p) for b, p in zip(beats, beat_paths)
        ]))
        top_summaries: dict[int, str] = {}
        for top in (s for s in sections if s.level == top_level):
            child_paths = [
                p for b, p in zip(beats, beat_paths)
                if beat_top[id(b)] is top
            ]
            items = [
                f"{cp.split(' / ')[-1]}: {beat_summaries.get(cp, '')}"
                for cp in child_paths
            ]
            if items:
                top_summaries[id(top)] = await _summarize_pillar(top.label, items)
        mining_progress.increment()

        # ── Edges — SEQUENCE spine + dialectic + LINKS ───────────────
        # No cross-beat pass: a narrative is a sequence, not a graph.
        mining_progress.set_phase("narrative_edges", total=1)
        seq_edges = build_sequence_edges(all_proposals)
        seq_edges = [e for e in seq_edges if e.confidence >= min_confidence]
        links_edges = [e for e in all_links if e.confidence >= min_confidence]
        try:
            dialectic_edges = await extract_dialectic_edges(all_proposals)
        except Exception as exc:
            logger.warning("Dialectic-edge pass failed: %r", exc)
            dialectic_edges = []
        all_edges = seq_edges + links_edges + dialectic_edges
        mining_progress.increment()

        # ── Consolidation ────────────────────────────────────────────
        # Pipeline-integrated anchor consolidation — leaf anchors never
        # reach /push-mined; demoted ones ride along as inline records.
        from .anchor_consolidation import analyze_proposals, apply_to_proposals
        mining_progress.set_phase("consolidation", total=1)
        cons_summary = None
        inline_map: dict[str, list[dict]] = {}
        try:
            plan, id_to_proposal = analyze_proposals(all_proposals, all_edges)
            all_proposals, all_edges, inline_map, cons_summary = apply_to_proposals(
                all_proposals, all_edges, plan, id_to_proposal,
            )
            logger.info("[CONS] %s", cons_summary)
        except Exception as exc:
            logger.warning(
                "Consolidation failed (passing through unfiltered): %r", exc,
            )
        mining_progress.increment()
        mining_progress.mark_done()

        # ── Beat tree (the pillar skeleton) ──────────────────────────
        # cross_edges is {} — the narrative miner has no cross-beat pass,
        # so the overlay is a pure tree (acts → beats).
        slab_titles_by_path = {
            p: [s.title for s in sl] for p, sl in slabs_by_path.items()
        }
        outline_tree = build_outline_tree(
            sections, top_level, beat_level, beats, beat_top, beat_paths,
            slab_titles_by_path, beat_summaries, top_summaries, {},
        )

        return {
            "format": "narrative",
            "outline_route": outline_route,
            "segments": len(beats),
            "outline": outline_tree,
            "proposals": [
                {
                    "type": p.proposal_type,
                    "canonical_phrase": p.canonical_phrase,
                    "canonical_text": p.canonical_text or "",
                    "description": p.description,
                    "title": p.title,
                    "label": p.label,
                    "aliases": p.aliases,
                    "source_topic": p.source_topic,
                    "confidence": round(p.confidence, 2),
                    "justification": p.justification,
                    "source_pairs": p.source_pairs,
                    # No cross-beat pass — always empty for narrative.
                    "cross_pillars": [],
                }
                for p in all_proposals
            ],
            "edges": [
                {
                    "type": e.edge_type,
                    "from": e.from_label,
                    "to": e.to_label,
                    "confidence": round(e.confidence, 2),
                    "justification": e.justification,
                }
                for e in all_edges
            ],
            "proposal_count": len(all_proposals),
            "edge_count": len(all_edges),
            "source_label": source_label,
            "_inline_anchors_per_slab": inline_map,
            "_consolidation_summary": cons_summary,
        }

    async def mine_file(self, path, **kwargs) -> dict:
        from pathlib import Path
        path = Path(path)
        if not path.exists():
            return {"error": f"File not found: {path}"}
        raw = path.read_text(encoding="utf-8")
        kwargs.setdefault("source_label", path.name)
        return await self.mine(raw, **kwargs)
