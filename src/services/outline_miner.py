"""Outline-first miner — for structured documents (papers, design docs).

Third miner alongside ConversationMiner and NarrativeMiner. Where the
narrative miner is *structure-blind* (segment → mine → flatten, the
document's outline discarded), this miner is *structure-first*:

  Pass 1 — outline.  Produce an ORDERED outline of the document.
           - If the source carries explicit headings (markdown #/##/###),
             parse the heading tree deterministically — no LLM.
           - Otherwise, one LLM "summarize the key sections in order"
             call yields the outline + per-section boundary markers.
           Either route produces the same shape: an ordered list of
           sections, each a CONTIGUOUS span of the source.

  Pass 2 — drill.  Mine each section span independently into slabs +
           anchors. Small, focused, parallel calls. Every slab is
           tagged (via ``source_topic``) with the section it came from.

Because a structured document is linear and its sections are
contiguous, a section is a *span*, not a *cluster* — so there is no
global 311-slab assignment problem. The pillar overlay falls out of
Pass 1 by construction: top headings → top pillars, sub-headings →
sub-pillars, a section's mined slabs → that pillar's members.

Output dict shape matches NarrativeMiner.mine() (so /push-mined accepts
the proposals unchanged) plus an ``outline`` field carrying the pillar
skeleton.

See the design conversation in this PR (WIP §6.6 genre-specific miners).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field

import numpy as np

from .convo_miner import MiningProposal, EdgeProposal, deduplicate_proposals
from .narrative_miner import build_sequence_edges
from . import ollama, mining_progress

logger = logging.getLogger(__name__)

_PASS_PARALLEL = int(os.environ.get("PS_MINING_PARALLEL", "2"))

# Headings that are document scaffolding, not real sections.
_SKIP_HEADINGS = {
    "contents", "table of contents", "toc", "index", "appendices",
}


# ─── Outline parsing ─────────────────────────────────────────────────────


@dataclass
class Section:
    """One outline section — a contiguous span of the source document."""
    level: int                 # heading depth (1 = top, 2 = sub, ...)
    label: str                 # heading text, markup stripped
    char_start: int            # offset of the heading line
    body_start: int            # offset where the section's content begins
    char_end: int = 0          # offset of the next same-or-higher heading
    parent_idx: int = -1       # index into the section list of the parent
    text: str = ""             # the section body (filled after spans resolve)


_MD_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.MULTILINE)


def _strip_heading_markup(s: str) -> str:
    return s.strip().lstrip("#").strip().strip("*_`").strip()


def parse_heading_outline(raw: str) -> list[Section]:
    """Parse markdown headings into an ordered, span-resolved Section list.

    Returns [] when the document has fewer than 2 usable headings — the
    caller then falls back to the LLM-summarize route.
    """
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    matches = list(_MD_HEADING.finditer(raw))
    sections: list[Section] = []
    for m in matches:
        label = _strip_heading_markup(m.group(2))
        if not label or label.lower() in _SKIP_HEADINGS:
            continue
        sections.append(Section(
            level=len(m.group(1)),
            label=label,
            char_start=m.start(),
            body_start=m.end(),
        ))
    if len(sections) < 2:
        return []

    # Resolve each section's end = start of the next heading whose level
    # is <= this section's level (a sibling or an ancestor's next child).
    for i, sec in enumerate(sections):
        end = len(raw)
        for j in range(i + 1, len(sections)):
            if sections[j].level <= sec.level:
                end = sections[j].char_start
                break
        sec.char_end = end
        sec.text = raw[sec.body_start:end].strip()

    # Parent pointers: each section's parent is the nearest preceding
    # section of strictly shallower level.
    for i, sec in enumerate(sections):
        for j in range(i - 1, -1, -1):
            if sections[j].level < sec.level:
                sec.parent_idx = j
                break
    return sections


def select_tiers(sections: list[Section]) -> tuple[int, int]:
    """Pick the (top_level, chapter_level) from the heading levels present.

    top_level    → becomes the top-pillar tier
    chapter_level → becomes the sub-pillar tier AND the mining unit

    Anything deeper than chapter_level is content mined into its chapter.
    A document with a single heading level has top_level == chapter_level
    (a flat list of chapter-pillars, no top tier).
    """
    levels = sorted({s.level for s in sections})
    if not levels:
        return (1, 1)
    if len(levels) == 1:
        return (levels[0], levels[0])
    return (levels[0], levels[1])


# ─── LLM-summarize fallback (headingless documents) ──────────────────────


_OUTLINE_SUMMARIZE_SYSTEM = """You are reading a document with no usable \
heading structure and producing its OUTLINE — the ordered list of key \
sections / arguments it works through, top to bottom.

This is a comprehension+summarization task. Read the document and identify \
the 4-15 major sections it moves through, IN ORDER. For each, give:
  - label: 2-6 word headline for the section
  - summary: 1-2 sentences on what the section covers
  - starts_with: a SHORT verbatim quote (6-12 words) copied EXACTLY from
    the document marking where this section begins. This is used to locate
    the section boundary — it must appear verbatim in the text.

RULES
1. Sections are CONTIGUOUS and IN ORDER — section 2 begins where section 1
   ends. Together they tile the whole document.
2. 4-15 sections. Fewer for short documents.
3. starts_with must be copied character-for-character from the document.

Output ONLY valid JSON:
{
  "sections": [
    {"label": "...", "summary": "...", "starts_with": "..."}
  ]
}
"""


def _digest_for_outline(raw_norm: str, whole_below: int = 25000) -> str:
    """Document digest for the outline call — each paragraph's first
    sentence. Outline detection needs the document's SHAPE (topic shifts
    between sections), not its full text; the lede of each paragraph
    carries that at a fraction of the tokens, keeping the single Pass-1
    call within a local model's timeout. Digest sentences are verbatim,
    so ``starts_with`` markers the model copies still locate in the
    full text. Short documents are sent whole.
    """
    if len(raw_norm) <= whole_below:
        return raw_norm
    paras = re.split(r"\n\s*\n", raw_norm)
    ledes = []
    for p in paras:
        p = p.strip()
        if not p:
            continue
        sents = re.split(r"(?<=[.!?])\s+", p)
        ledes.append(sents[0])
    return "\n\n".join(ledes)


_WORD_RE = re.compile(r"[a-z0-9]+")


def _locate_marker(raw_norm: str, marker: str, cursor: int) -> tuple[int, bool]:
    """Find where a section-boundary marker begins, at or after ``cursor``.

    Returns (position, was_fuzzy). Exact substring search first. On miss
    — the model paraphrased its ``starts_with`` quote instead of copying
    it verbatim — fall back to word-overlap against paragraph starts:
    the paragraph (at/after cursor) sharing the most content words with
    the marker. A paraphrase keeps most content words even when wording
    drifts, so this still places the section. Returns (-1, False) when
    nothing clears the overlap floor.
    """
    if not marker:
        return -1, False
    exact = raw_norm.find(marker, cursor)
    if exact >= 0:
        return exact, False

    marker_words = set(_WORD_RE.findall(marker.lower()))
    if len(marker_words) < 3:
        return -1, False  # too short to fuzzy-match safely
    span = max(len(marker) + 40, 160)
    best_pos, best_score = -1, 0.0
    for m in re.finditer(r"\n\s*\n", raw_norm):
        p_start = m.end()
        if p_start < cursor:
            continue
        cand_words = set(_WORD_RE.findall(raw_norm[p_start:p_start + span].lower()))
        score = len(marker_words & cand_words) / len(marker_words)
        if score > best_score:
            best_score, best_pos = score, p_start
    if best_score >= 0.6:
        return best_pos, True
    return -1, False


async def _outline_via_llm(raw: str) -> list[Section]:
    """Fallback Pass 1 — synthesize an outline for a headingless document.

    One LLM call over a paragraph-lede digest of the document. Outlining
    is retrospective (a section's place is clear only given what
    follows), so the model needs the whole document's arc — but it needs
    the *shape*, which the digest preserves cheaply enough to fit a
    local model's 600s budget.

    Section boundaries are then located by searching each ``starts_with``
    marker MONOTONICALLY — from just after the previous section's
    position — so a generic repeated lede ("This section defines...")
    can't snap a later section back to an earlier offset.
    """
    raw_norm = raw.replace("\r\n", "\n").replace("\r", "\n")
    sample = _digest_for_outline(raw_norm)
    user = f"DOCUMENT:\n\"\"\"\n{sample}\n\"\"\"\n\nProduce the ordered outline."
    try:
        resp = await ollama.structured_extract(
            user, system=_OUTLINE_SUMMARIZE_SYSTEM, num_ctx=32768, timeout=600,
        )
    except Exception as exc:
        logger.warning("Outline LLM-summarize failed: %r", exc)
        return []

    raw_sections = resp.get("sections", []) or []
    out: list[Section] = []
    cursor = 0  # monotonic — each marker is found at or after this
    exact_n = fuzzy_n = misses = 0
    for sec in raw_sections:
        label = (sec.get("label") or "").strip()
        marker = (sec.get("starts_with") or "").strip()
        if not label:
            continue
        pos, was_fuzzy = _locate_marker(raw_norm, marker, cursor)
        if pos < 0:
            misses += 1
            continue
        if was_fuzzy:
            fuzzy_n += 1
        else:
            exact_n += 1
        out.append(Section(level=1, label=label, char_start=pos, body_start=pos))
        cursor = pos + 1
    # First section always begins at 0 (content before the first located
    # marker would otherwise be dropped).
    if out:
        out[0].char_start = 0
        out[0].body_start = 0
    for i, sec in enumerate(out):
        sec.char_end = out[i + 1].char_start if i + 1 < len(out) else len(raw_norm)
        sec.text = raw_norm[sec.body_start:sec.char_end].strip()
    logger.info(
        "Outline LLM-summarize: %d sections proposed, %d located "
        "(%d exact, %d fuzzy), %d markers missed",
        len(raw_sections), len(out), exact_n, fuzzy_n, misses,
    )
    return [s for s in out if s.text]


# ─── Pass 2 — per-chapter extraction ─────────────────────────────────────


_CHAPTER_EXTRACT_SYSTEM = """You are mining ONE section of a structured \
document into corpus objects for a knowledge graph. The section's place in \
the document is given — use it as context.

Extract:

1. SLABS — the primary unit. Each slab is 2-8 complete sentences capturing
   one coherent claim, step, principle, or sub-topic within this section.
   - title: short and specific (3-7 words)
   - canonical_text: complete sentences. Condense dense prose faithfully;
     reconstitute terse/schematic passages into full sentences. The slab
     is stored as-is and must stand on its own.
   - A long section yields several slabs; a short one yields one or two.

2. ANCHORS — 2-8 word phrases naming distinctive entities or coined terms
   the section introduces: named concepts, components, technical terms.
   Only things that would be referenced elsewhere. Not every noun.
   - references_anchors on a slab: anchor phrases that slab involves.

Match density to the section. Do not manufacture objects to hit a count.

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


# ─── Pass 3 — pillar summary backfill ────────────────────────────────────


_PILLAR_SUMMARY_SYSTEM = """You are writing a SUMMARY for one pillar of a \
knowledge corpus — a 1-3 sentence distillation a reader sees at top zoom, \
before drilling in.

You are given the pillar's label and the slabs it contains (each a title +
excerpt of mined content). Distill what this pillar actually covers.

RULES
1. 1-3 sentences. Specific and substantive — a reader should know what
   they would learn by drilling into this pillar.
2. Describe CONTENT, not structure. "Covers flow, turbulence and the
   bubble floor as independently controllable layers" — not "contains
   9 slabs about water".
3. Summarize what the SLABS say, not what the label promises — the slabs
   are the corpus's own version of the content.

Output ONLY valid JSON: {"summary": "..."}
"""


async def _summarize_pillar(label: str, items: list[str]) -> str:
    """One small LLM call — distill a pillar's member content into a summary.

    ``items`` are short strings (slab "title: excerpt", or, for a top
    pillar, child-pillar "label: summary" lines).
    """
    if not items:
        return ""
    user = (
        f"PILLAR: {label}\n\n"
        f"MEMBER CONTENT ({len(items)} items):\n"
        + "\n".join(f"- {it}" for it in items)
        + "\n\nWrite the pillar summary."
    )
    try:
        resp = await ollama.structured_extract(user, system=_PILLAR_SUMMARY_SYSTEM)
    except Exception as exc:
        logger.warning("Pass 3 summary failed for %r: %r", label, exc)
        return ""
    return (resp.get("summary") or "").strip()


# ─── Pass 4 — cross-pillar relevance ─────────────────────────────────────


def _l2_normalize(v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(a))
    return a / n if n > 0 else a


async def detect_cross_pillar(
    chapter_paths: list[str],
    chapter_summaries: dict[str, str],
    slabs_by_path: dict[str, list],
    margin: float = 0.06,
    top_k: int = 2,
) -> tuple[dict[str, list[str]], dict[tuple[str, str], int]]:
    """Pass 4 — find which non-home pillars each slab is also relevant to.

    Embedding-based, no LLM: every slab and every pillar (its Pass-3
    summary) is embedded; a slab is cross-relevant to pillar Y if its
    cosine to Y lands within ``margin`` of its cosine to its home pillar
    — a *relative* test, robust on coherent documents where an absolute
    threshold would flood. Up to ``top_k`` cross pillars per slab.

    Returns:
      cross_by_title : {slab_title -> [cross-relevant pillar path, ...]}
      cross_edges    : {(home_path, cross_path) -> slab count}  — the
                       pillar-level lift: how many slabs carry each link.
    """
    paths = [p for p in chapter_paths if slabs_by_path.get(p)]
    if len(paths) < 2:
        return {}, {}

    pillar_texts = [
        f"{p.split(' / ')[-1]}. {chapter_summaries.get(p, '')}".strip()
        for p in chapter_paths
    ]
    flat = [(p, s) for p in chapter_paths for s in slabs_by_path.get(p, [])]
    if not flat:
        return {}, {}
    slab_texts = [
        f"{s.title}. {(s.canonical_text or '')[:240]}".strip() for _, s in flat
    ]
    try:
        pillar_vecs = await ollama.embed(pillar_texts)
        slab_vecs = await ollama.embed(slab_texts)
    except Exception as exc:
        logger.warning("Pass 4 embedding failed: %r", exc)
        return {}, {}

    pn = [_l2_normalize(v) for v in pillar_vecs]
    sn = [_l2_normalize(v) for v in slab_vecs]
    path_idx = {p: i for i, p in enumerate(chapter_paths)}

    cross_by_title: dict[str, list[str]] = {}
    cross_edges: dict[tuple[str, str], int] = {}
    for (home_path, slab), sv in zip(flat, sn):
        home_i = path_idx.get(home_path)
        sims = [float(np.dot(sv, pv)) for pv in pn]
        home_sim = sims[home_i] if home_i is not None else max(sims)
        cands = sorted(
            ((i, sims[i]) for i in range(len(chapter_paths))
             if i != home_i and sims[i] >= home_sim - margin),
            key=lambda x: -x[1],
        )[:top_k]
        if not cands:
            continue
        cross_paths = [chapter_paths[i] for i, _ in cands]
        cross_by_title[slab.title] = cross_paths
        for cp in cross_paths:
            key = (home_path, cp)
            cross_edges[key] = cross_edges.get(key, 0) + 1

    logger.info(
        "Pass 4: %d slabs cross-linked, %d pillar-level cross-edges",
        len(cross_by_title), len(cross_edges),
    )
    return cross_by_title, cross_edges


@dataclass
class ChapterResult:
    section_path: str
    order_base: int
    slabs: list[MiningProposal] = field(default_factory=list)
    anchors: list[MiningProposal] = field(default_factory=list)
    links: list[EdgeProposal] = field(default_factory=list)


async def _drill_chapter(
    chapter: Section, section_path: str, order_base: int, min_confidence: float,
) -> ChapterResult:
    """Pass 2 — mine one chapter span into slabs + anchors.

    ``order_base`` is the global monotonic position of this chapter; slab
    ``source_pairs`` are stamped order_base, order_base+1, ... so the
    downstream SEQUENCE-edge builder chains slabs in document order.
    """
    result = ChapterResult(section_path=section_path, order_base=order_base)
    body = chapter.text.strip()
    if len(body) < 40:
        return result

    user = (
        f"SECTION: {section_path}\n\n"
        f"SECTION TEXT:\n\"\"\"\n{body}\n\"\"\"\n\n"
        f"Extract slabs and anchors for this section."
    )
    try:
        resp = await ollama.structured_extract(
            user, system=_CHAPTER_EXTRACT_SYSTEM, num_ctx=16384,
        )
    except Exception as exc:
        logger.warning("Pass 2 extraction failed for %r: %r", section_path, exc)
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
            # section_path rides in source_topic — a free-text field the
            # contract already serializes. Downstream pillar building
            # reads it to place the slab under its sub-pillar.
            source_topic=section_path,
            confidence=conf,
            source_pairs=[pos],
            justification=(s.get("justification") or "").strip(),
        )
        result.slabs.append(slab)
        # references_anchors → LINKS edges (slab title → anchor phrase)
        for ap in s.get("references_anchors", []) or []:
            ap = (ap or "").strip()
            if ap:
                result.links.append(EdgeProposal(
                    edge_type="LINKS", from_label=title, to_label=ap,
                    confidence=0.8, justification="section co-occurrence",
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


# ─── Miner ───────────────────────────────────────────────────────────────


class OutlineMiner:
    """Mine a structured document outline-first. Sibling of NarrativeMiner."""

    def __init__(self, corpus):
        self.corpus = corpus

    async def mine(
        self,
        raw_text: str,
        source_label: str = "document",
        min_confidence: float = 0.4,
        max_segment_chars: int = 6000,
    ) -> dict:
        mining_progress.reset("outline")

        # ── Pass 1 — outline ─────────────────────────────────────────
        mining_progress.set_phase("outlining", total=1)
        sections = parse_heading_outline(raw_text)
        outline_route = "headings"
        if not sections:
            outline_route = "llm_summarize"
            sections = await _outline_via_llm(raw_text)
        mining_progress.increment()

        if not sections:
            mining_progress.mark_error("no outline could be derived")
            return {
                "format": "outline", "segments": 0, "outline": [],
                "proposals": [], "edges": [], "proposal_count": 0,
                "edge_count": 0, "source_label": source_label,
                "error": "No headings found and outline synthesis failed. "
                         "Try the Narrative miner for unstructured text.",
            }

        top_level, chapter_level = select_tiers(sections)

        # Each section's top-tier ancestor (None if it has none).
        def _top_of(sec: Section):
            cur = sec
            while cur.parent_idx >= 0:
                cur = sections[cur.parent_idx]
                if cur.level == top_level:
                    return cur
            return None

        # Chapters = the mining unit. Normally the chapter_level sections.
        # A top-level section with NO chapter-level descendant (e.g. an
        # appendix whose subsections are all one level deeper) becomes
        # its own chapter — otherwise its content would never be mined.
        chapters = [s for s in sections if s.level == chapter_level]
        if top_level != chapter_level:
            tops_with_chapters = {
                id(t) for t in (_top_of(ch) for ch in chapters) if t is not None
            }
            for s in sections:
                if s.level == top_level and id(s) not in tops_with_chapters:
                    chapters.append(s)
            chapters.sort(key=lambda s: s.char_start)

        logger.info(
            "OutlineMiner: route=%s, %d sections, %d chapters (tiers %d/%d)",
            outline_route, len(sections), len(chapters), top_level, chapter_level,
        )

        chapter_top = {id(ch): _top_of(ch) for ch in chapters}
        chapter_paths: list[str] = []
        for ch in chapters:
            t = chapter_top[id(ch)]
            chapter_paths.append(f"{t.label} / {ch.label}" if t else ch.label)

        # ── Pass 2 — drill each chapter (parallel) ───────────────────
        mining_progress.set_phase("section_drill", total=len(chapters))
        sem = asyncio.Semaphore(_PASS_PARALLEL)
        # Stamp generous order spacing so slabs across chapters stay
        # globally ordered without per-chapter slab counts colliding.
        order_bases = [i * 1000 for i in range(len(chapters))]

        async def _drill(idx: int) -> ChapterResult:
            async with sem:
                r = await _drill_chapter(
                    chapters[idx], chapter_paths[idx],
                    order_bases[idx], min_confidence,
                )
                mining_progress.increment()
                return r

        results = await asyncio.gather(*[_drill(i) for i in range(len(chapters))])

        # ── Assemble proposals ───────────────────────────────────────
        all_slabs: list[MiningProposal] = []
        all_anchors: list[MiningProposal] = []
        all_links: list[EdgeProposal] = []
        for r in results:
            all_slabs.extend(r.slabs)
            all_anchors.extend(r.anchors)
            all_links.extend(r.links)

        canonical_anchors = deduplicate_proposals(all_anchors)
        canonical_anchors = [a for a in canonical_anchors if a.proposal_type == "anchor"]

        all_proposals = canonical_anchors + all_slabs
        all_proposals = deduplicate_proposals(all_proposals)
        all_proposals = [p for p in all_proposals if p.confidence >= min_confidence]
        all_proposals.sort(key=lambda p: (
            p.source_pairs[0] if p.source_pairs else 999999,
            p.proposal_type,
        ))

        slabs_by_path: dict[str, list[MiningProposal]] = {}
        for s in all_slabs:
            slabs_by_path.setdefault(s.source_topic, []).append(s)

        # ── Pass 3 — backfill pillar summaries ───────────────────────
        # A standalone-shaped step: it only needs {pillar → its slabs},
        # which now exists. Chapter summaries distill the chapter's
        # slabs; a Part summary then distills its chapters' summaries
        # (cheaper than re-reading every slab, and the chapter summaries
        # are already good distillations).
        mining_progress.set_phase("pillar_summaries", total=len(chapters) + 1)

        async def _sum_chapter(ch: Section, path: str) -> tuple[str, str]:
            async with sem:
                slabs = slabs_by_path.get(path, [])
                items = [
                    f"{s.title}: {(s.canonical_text or '')[:160]}" for s in slabs
                ]
                summ = await _summarize_pillar(ch.label, items)
                mining_progress.increment()
                return path, summ

        chapter_summaries: dict[str, str] = dict(await asyncio.gather(*[
            _sum_chapter(ch, p) for ch, p in zip(chapters, chapter_paths)
        ]))
        top_summaries: dict[int, str] = {}
        for top in (s for s in sections if s.level == top_level):
            child_paths = [
                p for ch, p in zip(chapters, chapter_paths)
                if chapter_top[id(ch)] is top
            ]
            items = [
                f"{cp.split(' / ')[-1]}: {chapter_summaries.get(cp, '')}"
                for cp in child_paths
            ]
            if items:
                top_summaries[id(top)] = await _summarize_pillar(top.label, items)
        mining_progress.increment()

        # ── Pass 4 — cross-pillar relevance ──────────────────────────
        # Which non-home pillars is each slab also relevant to? Turns the
        # pillar tree into a graph. Embedding-based, uses Pass 3's pillar
        # summaries as the comparison target. A backfill like Pass 3.
        mining_progress.set_phase("cross_pillar", total=1)
        cross_by_title, cross_edges = await detect_cross_pillar(
            chapter_paths, chapter_summaries, slabs_by_path,
        )
        mining_progress.increment()

        # ── Edges ────────────────────────────────────────────────────
        mining_progress.set_phase("outline_edges", total=1)
        seq_edges = build_sequence_edges(all_proposals)
        seq_edges = [e for e in seq_edges if e.confidence >= min_confidence]
        links_edges = [e for e in all_links if e.confidence >= min_confidence]
        all_edges = seq_edges + links_edges
        mining_progress.increment()
        mining_progress.mark_done()

        # ── Outline tree (the pillar skeleton, summaries threaded) ───
        slab_titles_by_path = {
            p: [s.title for s in sl] for p, sl in slabs_by_path.items()
        }
        outline_tree = self._build_outline_tree(
            sections, top_level, chapter_level, chapters, chapter_top,
            chapter_paths, slab_titles_by_path, chapter_summaries,
            top_summaries, cross_edges,
        )

        return {
            "format": "outline",
            "outline_route": outline_route,
            "segments": len(chapters),
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
                    # Pass 4: non-home pillars this slab is also relevant
                    # to (slabs only; empty for anchors).
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
            "_inline_anchors_per_slab": {},
            "_consolidation_summary": None,
        }

    def _build_outline_tree(
        self, sections, top_level, chapter_level, chapters, chapter_top,
        chapter_paths, slab_titles_by_path, chapter_summaries, top_summaries,
        cross_edges,
    ) -> list[dict]:
        """Build the nested outline the response carries as the pillar skeleton.

        Top headings → top-pillar nodes; chapter headings → sub-pillar
        nodes, each carrying its Pass-3 summary, the titles of the slabs
        mined into it, and its Pass-4 cross_edges to other pillars. An
        orphan top (its own chapter) becomes a leaf top node carrying
        its slabs directly.
        """
        path_of = {id(ch): p for ch, p in zip(chapters, chapter_paths)}
        chapter_set = {id(ch) for ch in chapters}

        # Group Pass-4 cross-edges by their home (from) pillar path.
        cross_by_home: dict[str, list[dict]] = {}
        for (home_path, cross_path), weight in cross_edges.items():
            cross_by_home.setdefault(home_path, []).append({
                "to": cross_path.split(" / ")[-1],
                "to_path": cross_path,
                "weight": weight,
            })

        def chapter_node(ch: Section) -> dict:
            p = path_of[id(ch)]
            return {
                "label": ch.label,
                "level": ch.level,
                "section_path": p,
                "summary": chapter_summaries.get(p, ""),
                "slab_titles": slab_titles_by_path.get(p, []),
                "cross_edges": sorted(
                    cross_by_home.get(p, []), key=lambda x: -x["weight"],
                ),
            }

        if top_level == chapter_level:
            # Single tier — chapters are the top pillars.
            return [chapter_node(ch) for ch in chapters]

        tree = []
        for top in (s for s in sections if s.level == top_level):
            kids = [chapter_node(ch) for ch in chapters
                    if chapter_top[id(ch)] is top]
            node = {
                "label": top.label,
                "level": top.level,
                "summary": top_summaries.get(id(top), ""),
                "children": kids,
            }
            # Orphan top (no chapter children, but mined as its own
            # chapter) — carry its slabs + summary directly.
            if not kids and id(top) in chapter_set:
                p = path_of[id(top)]
                node["section_path"] = p
                node["slab_titles"] = slab_titles_by_path.get(p, [])
                node["summary"] = chapter_summaries.get(p, "") or node["summary"]
            if kids or node.get("slab_titles"):
                tree.append(node)
        return tree

    async def mine_file(self, path, **kwargs) -> dict:
        from pathlib import Path
        path = Path(path)
        if not path.exists():
            return {"error": f"File not found: {path}"}
        raw = path.read_text(encoding="utf-8")
        kwargs.setdefault("source_label", path.name)
        return await self.mine(raw, **kwargs)
