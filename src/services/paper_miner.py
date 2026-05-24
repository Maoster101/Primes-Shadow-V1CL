"""Paper miner — recursive hierarchy + semantic-chunk slabs.

Fourth genre miner alongside OutlineMiner (documentation), NarrativeMiner
(stories / essays), and ConversationMiner (chat logs). Where the doc
miner asks for sections-in-order *once* and drills them top-down, the
paper miner builds its hierarchy **recursively** and generates slabs
**bottom-up** at the leaves. Two questions, both content-driven, asked
at two different granularities:

  Stage A — Recursive hierarchy emergence.
    Pass 1     identifies the paper's top-level structure (Abstract /
               Intro / Method / Results / ...), once over the document
               digest.
    Passes 2..N do a batched "density audit" of all current leaves:
               for each one, the model decides whether it's a single
               coherent topic (`subdivide: false`) or splits into
               sub-topics it can name. Halt when no leaf wants
               subdivision — depth is data, not a target.

  Stage B — Semantic-chunk slab generation at each leaf.
    For each leaf the model splits the text into atomic semantic units
    — each unit captures one distinct argumentative MOVE (claim /
    method / finding / interpretation / comparison / limitation /
    principle). The chunker is the slab generator: chunk boundary =
    slab boundary, move-recognition is internal to the split. Because
    the input is a leaf (small, topically coherent), the model can't
    fall back to "summarise the main 4 ideas".

  Stage C — Propagation (no LLM).
    Leaves carry slab IDs in `members`; non-leaves carry sub-pillar
    IDs in `children`; summaries propagate bottom-up — leaves from
    their slabs, non-leaves from their children's summaries.

Cross-pillar Pass 4 runs (papers genuinely cross-cut Method↔Results)
and anchor consolidation runs (papers spawn leaf-anchor noise). The
SEQUENCE spine ties leaves in document order.

Reuses the genre-neutral outline-first machinery (`_outline_via_llm`,
`_locate_marker`, `_digest_for_outline`, `_summarize_pillar`,
`build_sequence_edges`, `detect_cross_pillar`, `build_outline_tree_recursive`)
from outline_miner; this module owns only the paper-specific prompts
and the recursive Stage-A orchestration.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field

import numpy as np

from .convo_miner import MiningProposal, EdgeProposal, deduplicate_proposals
from . import ollama, mining_progress
from .outline_miner import (
    Section,
    parse_heading_outline,
    _outline_via_llm,
    _locate_marker,
    _digest_for_outline,
    _summarize_pillar,
    detect_cross_pillar,
    build_sequence_edges,
    build_outline_tree_recursive,
    _compute_section_paths,
    _build_recursive_hierarchy,
)

logger = logging.getLogger(__name__)

# Match the daemon's OLLAMA_NUM_PARALLEL; default 2 is safe on 16GB GPU
# with gemma3:12b. See outline_miner / narrative_miner for the rationale.
_PASS_PARALLEL = int(os.environ.get("PS_MINING_PARALLEL", "2"))


# ─── Paper-genre prompts ─────────────────────────────────────────────────

PAPER_TOP_OUTLINE_SYSTEM = """You are reading a RESEARCH PAPER and \
identifying its TOP-LEVEL STRUCTURE — the small set of major sections \
it works through.

Research papers follow a canonical arc:
  - Abstract
  - Introduction (motivation, problem, contributions)
  - Related Work / Background
  - Method / Approach (the contribution)
  - Experiments / Evaluation (setup, datasets, baselines)
  - Results (findings)
  - Discussion / Analysis (interpretation, limitations)
  - Conclusion
  - Appendices

Identify the 3-9 top-level sections THIS paper presents, in document \
order. Don't force every canonical name — some papers merge Method+Results, \
others split Approach across several sections. Use the paper's actual \
top-level structure as evidenced by the text.

CRITICAL: figures, tables and equations belong INSIDE sections — they \
are evidence WITHIN a section's argument, never section boundaries. Do \
NOT create "Figure 3", "Table 1", or "Equation 5" as a section. If a \
heading reads "Figure 3 Results", the section is "Results" (it contains \
Figure 3). Name sections by FUNCTION (what they do for the paper), not \
by the figures they happen to contain.

For each top section:
  - label: 2-6 word headline naming the section by its function
    ("Experimental Setup", not "Figure 4 Details").
  - summary: 1-2 sentences on what this section establishes or shows.
  - starts_with: SHORT verbatim quote (6-12 words) copied EXACTLY from
    the text, marking where this section begins. Used to locate the
    boundary — it must appear verbatim.

RULES
1. Top sections are CONTIGUOUS and IN ORDER, tiling the whole paper.
2. 3-9 sections — fewer for short papers.
3. starts_with is character-for-character from the text.

Output ONLY valid JSON (the key is "sections"):
{
  "sections": [
    {"label": "...", "summary": "...", "starts_with": "..."}
  ]
}
"""


PAPER_DENSITY_AUDIT_SYSTEM = """You are auditing the structure of a \
research paper. You will be given several CURRENT LEAF SECTIONS — \
sections as currently identified. For each one, decide whether it is \
ONE coherent sub-topic that should stay as-is, OR whether it covers \
MULTIPLE distinct sub-topics and should be subdivided.

A section should STAY (subdivide=false) when:
  - It is one coherent topic developed throughout.
  - It is short — a few paragraphs.
  - Its content is a single move (an introduction, a conclusion, a
    definition, a brief acknowledgements list).

A section should SUBDIVIDE (subdivide=true) when:
  - It covers 2 or more genuinely distinct sub-topics that each warrant
    their own focus.
  - Examples: a Method section that has Data Setup + Training Procedure
    + Inference Procedure; a Results section that has Quantitative
    Results + Qualitative Examples + Ablations; an Approach section
    that splits into Critique Generation + Revision + Selection.
  - Each sub-section must be coherent on its own as a topic.

For each leaf you decide to subdivide, name 2-5 sub-sections. Each:
  - label: 2-6 word headline naming the sub-topic by its function.
  - starts_with: SHORT verbatim quote (6-12 words) copied EXACTLY from
    THIS leaf's text, marking where the sub-section begins. Used to
    locate the boundary — must appear verbatim in the leaf's text.

RULES
1. Be CONSERVATIVE. Only subdivide when the section genuinely has
   multiple distinct topics. A section developing ONE argument across
   many paragraphs should NOT subdivide.
2. Sub-sections must be topically distinct — "first paragraph",
   "second paragraph" is wrong; "Background Definitions",
   "Concrete Procedure" is right.
3. starts_with markers must come from the corresponding leaf's text.

Output ONLY valid JSON:
{
  "decisions": [
    {"id": "l0", "subdivide": false},
    {"id": "l1", "subdivide": true, "sub_sections": [
      {"label": "...", "starts_with": "..."}
    ]}
  ]
}
"""


PAPER_LEAF_SLAB_SYSTEM = """You are extracting SLABS from ONE leaf \
section of a research paper. This leaf covers a single coherent \
sub-topic — your job is to split it into atomic semantic units, where \
each unit captures one distinct argumentative MOVE.

The MOVES of an academic argument are:
  - a CLAIM or THESIS  ("we propose...", "we hypothesise that...")
  - a METHOD or PROCEDURE  ("we trained X on Y with...", "given an
    input, we...")
  - a FINDING or RESULT  ("we observed that...", "the model achieved...")
  - an INTERPRETATION or IMPLICATION  ("this suggests...", "indicates
    that...")
  - a COMPARISON to prior work  ("unlike RLHF, our approach...",
    "in contrast to Sparrow...")
  - a LIMITATION or CAVEAT  ("we did not test...", "this may not
    generalise...")
  - a PRINCIPLE or DESIGN RATIONALE  ("the constitution must be...",
    "we require X because...")

The RULE for slab boundaries:
  - Two adjacent paragraphs making the SAME move = ONE slab.
  - Two adjacent paragraphs making DIFFERENT moves = TWO slabs.

So a "claim plus its three supporting sentences" is one slab. A
"method paragraph followed by a result paragraph" is two slabs, even
back-to-back. A scaffolding paragraph (pure transition with no
substantive content) produces no slab.

A slab is:
  - title: short and specific (3-7 words)
  - canonical_text: complete sentences capturing the move's content.
    Condense rich prose faithfully; reconstitute terse passages into
    full sentences. The slab stands on its own — it is the canonical
    corpus record of this move.
  - references_anchors: anchor phrases (named entities, coined terms,
    methods) this slab involves.
  - confidence: 0.0-1.0
  - justification: why this is a distinct move

Also extract ANCHORS — 2-8 word phrases naming distinctive entities,
methods, or coined terms the leaf introduces. Only things that would
be referenced elsewhere in the paper. Not every noun. When the leaf
positions one concept against another (a method vs an alternative),
extract BOTH as anchors — the dialectic edges depend on it.

Coverage check: at the end, every substantive move in the leaf should
be represented by some slab. Re-read; if a distinct move is missing,
emit a slab for it.

Output ONLY valid JSON:
{
  "slabs": [
    {"title": "...", "canonical_text": "...",
     "references_anchors": ["..."], "confidence": 0.0-1.0,
     "justification": "..."}
  ],
  "anchors": [
    {"canonical_phrase": "...", "aliases": ["..."], "confidence": 0.0-1.0}
  ]
}
"""


# ─── Stage B — semantic-chunk slab generation at leaves ──────────────────


@dataclass
class LeafResult:
    section_path: str
    order_base: int
    slabs: list[MiningProposal] = field(default_factory=list)
    anchors: list[MiningProposal] = field(default_factory=list)
    links: list[EdgeProposal] = field(default_factory=list)


async def _drill_leaf(
    leaf: Section, section_path: str, order_base: int, min_confidence: float,
) -> LeafResult:
    """Stage B — split a leaf into atomic semantic units (slabs) via the
    move-recognition prompt. The chunker IS the slab generator.
    """
    result = LeafResult(section_path=section_path, order_base=order_base)
    body = leaf.text.strip()
    if len(body) < 40:
        return result

    user = (
        f"LEAF SECTION: {section_path}\n\n"
        f"LEAF TEXT:\n\"\"\"\n{body}\n\"\"\"\n\n"
        f"Split this leaf into atomic semantic units, each a slab."
    )
    try:
        resp = await ollama.structured_extract(
            user, system=PAPER_LEAF_SLAB_SYSTEM, num_ctx=16384,
        )
    except Exception as exc:
        logger.warning("Leaf drill failed for %r: %r", section_path, exc)
        return result

    pos = order_base
    for s in resp.get("slabs", []) or []:
        text = (s.get("canonical_text") or "").strip()
        if not text:
            continue
        conf = float(s.get("confidence") or 0.7)
        if conf < min_confidence:
            continue
        title = (s.get("title") or text[:48]).strip()
        slab = MiningProposal(
            proposal_type="slab",
            canonical_text=text,
            title=title,
            # section_path rides in source_topic — picked up by the
            # rebuild path (build_pillars_from_collection) so pillars
            # reassemble from draft sidecars even across refresh.
            source_topic=section_path,
            confidence=conf,
            source_pairs=[pos],
            justification=(s.get("justification") or "").strip(),
        )
        result.slabs.append(slab)
        for ap in s.get("references_anchors", []) or []:
            ap = (ap or "").strip()
            if ap:
                result.links.append(EdgeProposal(
                    edge_type="LINKS", from_label=title, to_label=ap,
                    confidence=0.8,
                    justification="leaf co-occurrence (paper drill)",
                ))
        pos += 1

    for a in resp.get("anchors", []) or []:
        phrase = (a.get("canonical_phrase") or "").strip()
        if not phrase:
            continue
        conf = float(a.get("confidence") or 0.7)
        if conf < min_confidence:
            continue
        result.anchors.append(MiningProposal(
            proposal_type="anchor",
            canonical_phrase=phrase,
            aliases=[x for x in (a.get("aliases") or []) if x],
            source_topic=section_path,
            confidence=conf,
            source_pairs=[order_base],
        ))
    return result


# ─── Miner class ─────────────────────────────────────────────────────────


class PaperMiner:
    """Mine research papers with recursive hierarchy + semantic-chunk slabs.

    Output dict shape matches the other miners so /push-mined and the
    pillar-overlay machinery (build_pillars_from_collection) work
    unchanged. The ``outline`` field carries the N-tier beat skeleton —
    `build_pillars_recursive` is its build counterpart.
    """

    def __init__(self, corpus):
        self.corpus = corpus

    async def mine(
        self,
        raw_text: str,
        source_label: str = "paper",
        min_confidence: float = 0.4,
        max_segment_chars: int = 0,   # legacy/unused — recursive hierarchy
    ) -> dict:
        mining_progress.reset("paper")

        # ── Stage A — recursive hierarchy ────────────────────────────
        mining_progress.set_phase("outlining", total=1)
        sections, leaf_ids, route = await _build_recursive_hierarchy(
            raw_text,
            top_system=PAPER_TOP_OUTLINE_SYSTEM,
            audit_system=PAPER_DENSITY_AUDIT_SYSTEM,
        )
        mining_progress.increment()

        if not sections:
            mining_progress.mark_error("no outline could be derived")
            return {
                "format": "paper", "segments": 0, "outline": [],
                "proposals": [], "edges": [], "proposal_count": 0,
                "edge_count": 0, "source_label": source_label,
                "error": "Could not derive a top-level paper structure.",
            }

        paths = _compute_section_paths(sections)
        leaves = [s for s in sections if id(s) in leaf_ids]
        leaves.sort(key=lambda s: s.char_start)

        logger.info(
            "PaperMiner: route=%s, %d total sections, %d leaves, "
            "max_depth=%d",
            route, len(sections), len(leaves),
            max((s.level for s in sections), default=1),
        )

        # ── Stage B — drill each leaf in parallel ────────────────────
        mining_progress.set_phase("leaf_drill", total=len(leaves))
        sem = asyncio.Semaphore(_PASS_PARALLEL)
        order_bases = [i * 1000 for i in range(len(leaves))]

        async def _drill(idx: int) -> LeafResult:
            async with sem:
                r = await _drill_leaf(
                    leaves[idx], paths[id(leaves[idx])],
                    order_bases[idx], min_confidence,
                )
                mining_progress.increment()
                return r

        results = await asyncio.gather(*[_drill(i) for i in range(len(leaves))])

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

        # ── Pass 3 — summaries (bottom-up propagation) ───────────────
        # Leaf summaries from slabs; non-leaf summaries from children's
        # summaries, walked level-descending so children are ready
        # when their parent is summarised.
        slabs_by_path: dict[str, list[MiningProposal]] = {}
        for s in all_slabs:
            slabs_by_path.setdefault(s.source_topic, []).append(s)

        n_leaves = len(leaves)
        n_non_leaves = sum(
            1 for s in sections if id(s) not in leaf_ids
        )
        mining_progress.set_phase(
            "pillar_summaries", total=max(1, n_leaves + n_non_leaves),
        )
        summaries: dict[int, str] = {}

        async def _leaf_summ(leaf: Section) -> tuple[int, str]:
            async with sem:
                path = paths[id(leaf)]
                slabs = slabs_by_path.get(path, [])
                items = [
                    f"{s.title}: {(s.canonical_text or '')[:160]}"
                    for s in slabs
                ]
                summ = await _summarize_pillar(leaf.label, items)
                mining_progress.increment()
                return id(leaf), summ

        leaf_pairs = await asyncio.gather(*[_leaf_summ(l) for l in leaves])
        for sid, summ in leaf_pairs:
            summaries[sid] = summ

        # Walk non-leaves bottom-up (deepest first) so each non-leaf
        # has its children's summaries ready before it's synthesised.
        non_leaves_by_depth = sorted(
            (s for s in sections if id(s) not in leaf_ids),
            key=lambda s: -s.level,
        )
        for nl in non_leaves_by_depth:
            child_items = []
            for c in sections:
                if c.parent_idx >= 0 and sections[c.parent_idx] is nl:
                    child_summ = summaries.get(id(c), "")
                    child_items.append(f"{c.label}: {child_summ}")
            if child_items:
                summaries[id(nl)] = await _summarize_pillar(
                    nl.label, child_items,
                )
            mining_progress.increment()

        # ── Pass 4 — cross-pillar across leaves ──────────────────────
        # Papers genuinely cross-cut (Method↔Results, etc.) so this
        # pass earns its keep — unlike on narratives, where it was
        # mostly entity-continuity noise.
        mining_progress.set_phase("cross_pillar", total=1)
        leaf_paths = [paths[id(l)] for l in leaves]
        leaf_summaries = {
            paths[id(l)]: summaries.get(id(l), "") for l in leaves
        }
        leaf_slabs = {
            paths[id(l)]: slabs_by_path.get(paths[id(l)], [])
            for l in leaves
        }
        cross_by_title, cross_edges = await detect_cross_pillar(
            leaf_paths, leaf_summaries, leaf_slabs,
        )
        mining_progress.increment()

        # ── Assemble proposals + edges ───────────────────────────────
        all_proposals = canonical_anchors + all_slabs
        all_proposals = deduplicate_proposals(all_proposals)
        all_proposals = [
            p for p in all_proposals if p.confidence >= min_confidence
        ]
        all_proposals.sort(key=lambda p: (
            p.source_pairs[0] if p.source_pairs else 999999,
            p.proposal_type,
        ))

        mining_progress.set_phase("outline_edges", total=1)
        seq_edges = build_sequence_edges(all_proposals)
        seq_edges = [e for e in seq_edges if e.confidence >= min_confidence]
        links_edges = [e for e in all_links if e.confidence >= min_confidence]
        all_edges = seq_edges + links_edges
        mining_progress.increment()

        # ── Anchor consolidation ─────────────────────────────────────
        # Papers spawn leaf-anchor noise — figure refs, one-off terms.
        # Demoted ones ride inline; orphans drop.
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

        # ── Outline tree (N-tier skeleton, summaries threaded) ──────
        slab_titles_by_path = {
            p: [s.title for s in sl] for p, sl in slabs_by_path.items()
        }
        outline_tree = build_outline_tree_recursive(
            sections, leaf_ids, summaries,
            slab_titles_by_path, cross_edges,
        )

        return {
            "format": "paper",
            "outline_route": route,
            "segments": len(leaves),
            "total_sections": len(sections),
            "max_depth": max((s.level for s in sections), default=1),
            "outline": outline_tree,
            "proposals": [
                {
                    "type": p.proposal_type,
                    "canonical_phrase": p.canonical_phrase,
                    "canonical_text": p.canonical_text or "",
                    "title": p.title,
                    "label": p.label,
                    "aliases": p.aliases,
                    "source_topic": p.source_topic,
                    "confidence": round(p.confidence, 2),
                    "justification": p.justification,
                    "source_pairs": p.source_pairs,
                    "cross_pillars": (
                        cross_by_title.get(p.title, [])
                        if p.proposal_type == "slab" else []
                    ),
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
