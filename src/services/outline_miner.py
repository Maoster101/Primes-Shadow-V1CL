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
from . import ollama, mining_progress
from ..models.schemas import PillarDefinition, PillarCrossEdge
from ..models.enums import EdgeType

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


def resolve_chapters(
    sections: list[Section], top_level: int, chapter_level: int,
) -> tuple[list[Section], dict, list[str]]:
    """Resolve the mining units ("chapters") from a parsed outline.

    Chapters are the chapter_level sections — plus any top-level section
    with no chapter-level descendant (an appendix whose subsections are
    all one level deeper), which becomes its own chapter so its content
    is still mined. Returns ``(chapters, chapter_top, chapter_paths)``:
    ``chapter_top`` maps ``id(chapter)`` → its top-tier Section (or None),
    ``chapter_paths`` is the "Top / Chapter" (or bare "Chapter") path.

    Genre-neutral — the outline tiering is the same whether the units
    are document sections or narrative beats.
    """
    def _top_of(sec: Section):
        cur = sec
        while cur.parent_idx >= 0:
            cur = sections[cur.parent_idx]
            if cur.level == top_level:
                return cur
        return None

    chapters = [s for s in sections if s.level == chapter_level]
    if top_level != chapter_level:
        tops_with_chapters = {
            id(t) for t in (_top_of(ch) for ch in chapters) if t is not None
        }
        for s in sections:
            if s.level == top_level and id(s) not in tops_with_chapters:
                chapters.append(s)
        chapters.sort(key=lambda s: s.char_start)

    chapter_top = {id(ch): _top_of(ch) for ch in chapters}
    chapter_paths: list[str] = []
    for ch in chapters:
        t = chapter_top[id(ch)]
        chapter_paths.append(f"{t.label} / {ch.label}" if t else ch.label)
    return chapters, chapter_top, chapter_paths


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


async def _outline_via_llm(
    raw: str, system: str = _OUTLINE_SUMMARIZE_SYSTEM,
) -> list[Section]:
    """Fallback Pass 1 — synthesize an outline for a headingless document.

    ``system`` selects the genre framing — the default treats the source
    as a document (sections/arguments); the narrative miner passes a
    beat-flavoured prompt. The marker-location + span-resolution logic
    below is genre-neutral, so only the prompt changes.

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
            user, system=system, num_ctx=32768, timeout=600,
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
    system: str = _CHAPTER_EXTRACT_SYSTEM,
) -> ChapterResult:
    """Pass 2 — mine one chapter span into slabs + anchors.

    ``order_base`` is the global monotonic position of this chapter; slab
    ``source_pairs`` are stamped order_base, order_base+1, ... so the
    downstream SEQUENCE-edge builder chains slabs in document order.

    ``system`` selects the genre extraction prompt — the default mines a
    document section; the narrative miner passes a beat-flavoured prompt.
    The span → proposals plumbing is identical for both.
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
            user, system=system, num_ctx=16384,
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


# ─── SEQUENCE spine ──────────────────────────────────────────────────────


def build_sequence_edges(proposals: list[MiningProposal]) -> list[EdgeProposal]:
    """Emit SEQUENCE edges between consecutive slabs in document order.

    Walks slabs sorted by ``(source_pairs[0], intra_segment_order)`` and
    connects each to the next. Weight is lifted slightly when the two
    slabs share capitalised entities (a continuity signal). This is the
    ordered spine — the doc miner uses it for section flow, the narrative
    miner for the beat-to-beat progression.

    The two-key sort matters: ``source_pairs[0]`` alone collapses every
    slab from one drill unit into a single bucket, so when a unit yields
    several slabs ``intra_segment_order`` is the only thing preserving
    their within-unit order.
    """
    slabs = [p for p in proposals if p.proposal_type == "slab"]
    slabs.sort(key=lambda p: (
        p.source_pairs[0] if p.source_pairs else 0,
        p.intra_segment_order,
    ))
    edges: list[EdgeProposal] = []
    for i in range(len(slabs) - 1):
        a, b = slabs[i], slabs[i + 1]
        a_label = a.title or a.canonical_text[:40]
        b_label = b.title or b.canonical_text[:40]
        # Continuity signal: shared capitalised tokens (cheap proxy).
        a_tokens = set(re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?", a.canonical_text))
        b_tokens = set(re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?", b.canonical_text))
        shared = len(a_tokens & b_tokens)
        weight = min(0.55 + 0.1 * shared, 0.95)
        a_seg = a.source_pairs[0] if a.source_pairs else 0
        b_seg = b.source_pairs[0] if b.source_pairs else 0
        a_pos, b_pos = a.intra_segment_order, b.intra_segment_order
        if a_pos or b_pos:
            ordering_str = f"unit {a_seg}/{a_pos} → unit {b_seg}/{b_pos}"
        else:
            ordering_str = f"unit {a_seg} → unit {b_seg}"
        edges.append(EdgeProposal(
            edge_type="SEQUENCE",
            from_label=a_label,
            to_label=b_label,
            confidence=weight,
            justification=(
                f"Ordering: {ordering_str}"
                + (f" (shared entities: {shared})" if shared else "")
            ),
        ))
    return edges


# ─── Outline tree (the pillar skeleton) ──────────────────────────────────


def build_outline_tree(
    sections, top_level, chapter_level, chapters, chapter_top,
    chapter_paths, slab_titles_by_path, chapter_summaries, top_summaries,
    cross_edges,
) -> list[dict]:
    """Build the nested outline the response carries as the pillar skeleton.

    Top headings → top-pillar nodes; chapter headings → sub-pillar nodes,
    each carrying its summary, the titles of the slabs mined into it, and
    its cross_edges to other pillars. An orphan top (its own chapter)
    becomes a leaf top node carrying its slabs directly.

    Genre-neutral: ``cross_edges`` is ``{}`` for the narrative miner,
    which deliberately skips the cross-relevance pass.
    """
    path_of = {id(ch): p for ch, p in zip(chapters, chapter_paths)}
    chapter_set = {id(ch) for ch in chapters}

    # Group cross-edges by their home (from) pillar path.
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
        chapters, chapter_top, chapter_paths = resolve_chapters(
            sections, top_level, chapter_level,
        )
        logger.info(
            "OutlineMiner: route=%s, %d sections, %d chapters (tiers %d/%d)",
            outline_route, len(sections), len(chapters), top_level, chapter_level,
        )

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
        outline_tree = build_outline_tree(
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

    async def mine_file(self, path, **kwargs) -> dict:
        from pathlib import Path
        path = Path(path)
        if not path.exists():
            return {"error": f"File not found: {path}"}
        raw = path.read_text(encoding="utf-8")
        kwargs.setdefault("source_label", path.name)
        return await self.mine(raw, **kwargs)


# ─── Last mile — outline tree → committed PillarDefinitions ───────────────


_PILLAR_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _pillar_slug(s: str, max_len: int = 40) -> str:
    return _PILLAR_SLUG_RE.sub("_", s.lower()).strip("_")[:max_len] or "pillar"


def _index_slabs_by_title(store) -> dict[str, list[str]]:
    """{normalized title -> [slab_id, ...]} for resolving outline slab_titles.

    A list per title, not a single ID: two slabs can carry the same
    title, and greedy resolution pops from the list so each committed
    slab is claimed by at most one pillar.
    """
    index: dict[str, list[str]] = {}
    for sid, slab in store.slabs.items():
        key = (getattr(slab, "title", "") or "").strip().lower()
        if key:
            index.setdefault(key, []).append(sid)
    return index


def build_pillars_from_outline(
    store, outline_tree: list[dict], origin: str = "", replace: bool = True,
) -> dict:
    """Last mile — turn a mined outline tree into committed PillarDefinitions.

    Run AFTER the outline miner's slabs have been pushed and committed to
    ``store``. The outline tree (the ``outline`` field of
    ``OutlineMiner.mine``) is the Part/chapter skeleton: each chapter node
    carries its Pass-3 summary, the *titles* of the slabs mined into it,
    and its Pass-4 cross-edges. This resolves those titles to the
    committed slab IDs and writes the pillar overlay to ``store.pillars``
    (persisted to pillars.yaml).

    Why title-based resolution: the miner emits *proposals*, /push-mined
    mints draft IDs, and the commit step mints final node IDs — the slab's
    ID is not stable across that boundary, but its title is. So the
    outline tree records titles and we re-join here. Greedy with a
    claimed-ID ``used`` set so duplicate titles disambiguate by order.

    The pillar tier is a pure overlay (§ schema PillarDefinition): it
    references content nodes by ID and never copies their text. A chapter
    that resolves to zero committed slabs is dropped — a pillar with no
    members is navigationally dead weight, and an empty Part (all
    children dropped) is dropped with it.

    Returns a report dict: counts, unresolved titles, validation errors.
    """
    title_index = _index_slabs_by_title(store)
    used: set[str] = set()
    seen_ids: set[str] = set()
    unresolved: list[str] = []

    def _new_id(label: str) -> str:
        slug = _pillar_slug(label)
        pid = f"pillar_{slug}_v1"
        n = 2
        while pid in seen_ids:
            pid = f"pillar_{slug}_{n}_v1"
            n += 1
        seen_ids.add(pid)
        return pid

    def _resolve(titles: list[str]) -> list[str]:
        ids: list[str] = []
        for t in titles or []:
            key = (t or "").strip().lower()
            picked = None
            for sid in title_index.get(key, []):
                if sid not in used:
                    picked = sid
                    break
            if picked:
                used.add(picked)
                ids.append(picked)
            else:
                unresolved.append(t)
        return ids

    pillars: dict[str, PillarDefinition] = {}
    path_to_pillar: dict[str, str] = {}        # section_path -> pillar id
    deferred_cross: dict[str, list[dict]] = {}  # pillar id -> raw cross_edges

    def _make_chapter(node: dict, parent_id: str | None) -> str | None:
        """Build one leaf/chapter pillar. Returns its id, or None if it
        resolved to zero members (caller drops it from any parent).

        A node may carry ``slab_ids`` (exact committed IDs — used by the
        session-rebuild path, which knows them) or ``slab_titles`` (used
        by the fresh-mine path, resolved by title). slab_ids win when
        present: no lossy title round-trip."""
        ids = node.get("slab_ids")
        if ids:
            members = [s for s in ids if s in store.slabs and s not in used]
            used.update(members)
        else:
            members = _resolve(node.get("slab_titles", []))
        if not members:
            return None
        pid = _new_id(node.get("label") or "Section")
        pillars[pid] = PillarDefinition(
            id=pid,
            label=(node.get("label") or "Section").strip(),
            summary=(node.get("summary") or "").strip(),
            members=members,
            parent=parent_id,
            origin=origin,
            pillar_role="chapter" if parent_id else "section",
        )
        path = node.get("section_path")
        if path:
            path_to_pillar[path] = pid
        if node.get("cross_edges"):
            deferred_cross[pid] = node["cross_edges"]
        return pid

    # First pass — build the tier nodes. Top IDs are minted before their
    # children so each child can carry a ``parent`` pointer.
    for top in outline_tree:
        kids = top.get("children")
        if kids:
            top_id = _new_id(top.get("label") or "Part")
            child_ids = [
                cid for cid in
                (_make_chapter(ch, parent_id=top_id) for ch in kids)
                if cid is not None
            ]
            if not child_ids:
                continue  # every child resolved empty — drop the Part
            pillars[top_id] = PillarDefinition(
                id=top_id,
                label=(top.get("label") or "Part").strip(),
                summary=(top.get("summary") or "").strip(),
                children=child_ids,
                origin=origin,
                pillar_role="section",
            )
        else:
            # Single-tier chapter, or an orphan top carrying slabs direct.
            _make_chapter(top, parent_id=None)

    # Second pass — lift cross-edges. The outline's per-chapter
    # ``cross_edges`` weight is a slab COUNT; PillarCrossEdge.weight is a
    # 0-1 strength, so normalize (5+ shared slabs = full weight).
    cross_count = 0
    for pid, raw_edges in deferred_cross.items():
        lifted: list[PillarCrossEdge] = []
        for ce in raw_edges:
            target = path_to_pillar.get(ce.get("to_path"))
            if not target or target == pid:
                continue
            count = int(ce.get("weight") or 1)
            lifted.append(PillarCrossEdge(
                to_pillar=target,
                type=EdgeType.LINKS,
                weight=min(1.0, count / 5.0),
                confidence=0.7,
                rationale=f"{count} slab(s) cross-relevant between pillars",
            ))
        if lifted:
            pillars[pid].cross_edges = lifted
            cross_count += len(lifted)

    if not pillars:
        return {
            "pillars_created": 0,
            "error": "No pillars built — no outline slab titles resolved to "
                     "committed slabs. Push and commit the mined slabs "
                     "before building pillars.",
            "unresolved_titles": unresolved,
        }

    if replace:
        store.pillars = pillars
    else:
        store.pillars.update(pillars)

    errors = store.validate()
    store.save()

    logger.info(
        "build_pillars_from_outline: %d pillars (%d top, %d sub), "
        "%d slabs placed, %d cross-edges, %d unresolved titles",
        len(pillars),
        sum(1 for p in pillars.values() if not p.parent),
        sum(1 for p in pillars.values() if p.parent),
        len(used), cross_count, len(unresolved),
    )
    return {
        "pillars_created": len(pillars),
        "top_pillars": sum(1 for p in pillars.values() if not p.parent),
        "sub_pillars": sum(1 for p in pillars.values() if p.parent),
        "members_resolved": len(used),
        "cross_edges": cross_count,
        "unresolved_titles": unresolved,
        "validation_errors": errors,
    }


def _collect_slab_records(store, raw_paths: list[str]) -> list[dict]:
    """Read draft raw sidecars into outline-rebuild records.

    Keeps a sidecar only if it is a slab proposal, carries a
    ``source_topic`` (i.e. came from the outline miner), and the slab
    actually committed into ``store`` — so discarded / still-pending /
    other-collection drafts are filtered out. The id↔draft_id identity
    (``_convert_to_corpus_object``) is what lets us match committed
    slabs to their sidecars by filename.
    """
    import json as _json
    from pathlib import Path

    recs: list[dict] = []
    for raw_path in raw_paths:
        try:
            d = _json.loads(Path(raw_path).read_text(encoding="utf-8"))
        except Exception:
            continue
        if d.get("type") != "slab":
            continue
        slab_id = Path(raw_path).name[:-len("_raw.json")]
        if slab_id not in store.slabs:
            continue  # discarded, still pending, or committed elsewhere
        topic = (d.get("source_topic") or "").strip()
        if not topic:
            continue  # not outline-mined (no section tag)
        sp = d.get("source_pairs") or []
        recs.append({
            "id": slab_id,
            "topic": topic,
            "cross": [c for c in (d.get("cross_pillars") or []) if c],
            "order": sp[0] if sp else 10 ** 9,
        })
    return recs


async def _pillars_from_slab_records(
    store, recs: list[dict], origin: str, source_tag: str,
    regenerate_summaries: bool = True, replace: bool = True,
) -> dict:
    """Shared core: turn outline-rebuild records into a committed overlay.

    ``recs`` come from ``_collect_slab_records`` (one session's drafts or
    a whole collection's). Groups them back into their sections by
    ``source_topic``, regenerates the Pass-3 summaries (the one field the
    sidecars don't carry), lifts ``cross_pillars`` to section-level
    cross-edges, and delegates to ``build_pillars_from_outline``.
    """
    if not recs:
        return {
            "pillars_created": 0,
            "error": "No committed outline-mined slabs found. Push and "
                     "promote a doc/paper mine into this collection first.",
        }

    # Regroup into sections, ordered by document position (source_pairs).
    sections: dict[str, dict] = {}
    for r in recs:
        sec = sections.setdefault(r["topic"], {"ids": [], "order": r["order"]})
        sec["ids"].append(r["id"])
        sec["order"] = min(sec["order"], r["order"])
    ordered_topics = [
        t for t, _ in sorted(sections.items(), key=lambda kv: kv[1]["order"])
    ]

    # Pass 3 redo — summaries are the one field not in the sidecars.
    summaries: dict[str, str] = {t: "" for t in ordered_topics}
    if regenerate_summaries:
        sem = asyncio.Semaphore(_PASS_PARALLEL)

        async def _resummarize(topic: str):
            async with sem:
                items = []
                for sid in sections[topic]["ids"]:
                    sl = store.slabs.get(sid)
                    if sl:
                        items.append(
                            f"{sl.title}: {(sl.canonical_text or '')[:160]}"
                        )
                summaries[topic] = await _summarize_pillar(
                    topic.split(" / ")[-1], items
                )

        await asyncio.gather(*[_resummarize(t) for t in ordered_topics])

    # Aggregate per-slab cross_pillars up to section-level cross-edges.
    cross_by_home: dict[str, dict[str, int]] = {}
    for r in recs:
        for to_topic in r["cross"]:
            if to_topic and to_topic != r["topic"]:
                tally = cross_by_home.setdefault(r["topic"], {})
                tally[to_topic] = tally.get(to_topic, 0) + 1

    def _edges_for(topic: str) -> list[dict]:
        return sorted(
            ({"to": to.split(" / ")[-1], "to_path": to, "weight": n}
             for to, n in cross_by_home.get(topic, {}).items()),
            key=lambda x: -x["weight"],
        )

    # Assemble an outline tree (slab_ids carried — exact, no title round
    # trip) and hand it to the shared builder. " / " in a topic means the
    # source was two-tier (Part / Chapter); a flat topic is its own top.
    two_tier = any(" / " in t for t in ordered_topics)
    outline_tree: list[dict] = []
    if two_tier:
        parts: dict[str, list[str]] = {}
        part_order: list[str] = []
        for t in ordered_topics:
            part = t.split(" / ")[0]
            if part not in parts:
                parts[part] = []
                part_order.append(part)
            parts[part].append(t)
        for part in part_order:
            outline_tree.append({
                "label": part, "level": 1, "summary": "",
                "children": [{
                    "label": t.split(" / ")[-1], "level": 2,
                    "section_path": t, "summary": summaries.get(t, ""),
                    "slab_ids": sections[t]["ids"],
                    "cross_edges": _edges_for(t),
                } for t in parts[part]],
            })
    else:
        for t in ordered_topics:
            outline_tree.append({
                "label": t, "level": 1, "section_path": t,
                "summary": summaries.get(t, ""),
                "slab_ids": sections[t]["ids"],
                "cross_edges": _edges_for(t),
            })

    report = build_pillars_from_outline(
        store, outline_tree, origin=origin, replace=replace,
    )
    report["sections"] = len(ordered_topics)
    report["source"] = source_tag
    return report


async def build_pillars_from_session(
    store, session_store, session_id: str, origin: str = "",
    regenerate_summaries: bool = True, replace: bool = True,
) -> dict:
    """Rebuild the pillar overlay from ONE mining session's committed drafts.

    The ``outline`` tree /mine-outline returns lives only in browser
    memory; a refresh between mining and committing wipes it. But every
    pushed draft writes a raw sidecar (``{id}_raw.json``) persisting
    ``source_topic`` + ``cross_pillars`` — so the overlay is rebuildable
    from disk with nothing but a session id.
    """
    import glob
    from pathlib import Path

    drafts_dir = Path(session_store.drafts_dir(session_id))
    recs = _collect_slab_records(store, glob.glob(str(drafts_dir / "*_raw.json")))
    return await _pillars_from_slab_records(
        store, recs, origin, "session_sidecars",
        regenerate_summaries=regenerate_summaries, replace=replace,
    )


async def build_pillars_from_collection(
    store, session_store, origin: str = "",
    regenerate_summaries: bool = True, replace: bool = True,
) -> dict:
    """Rebuild the overlay from EVERY session's drafts — needs only the store.

    The fully stateless path: the corpus-view "Rebuild Pillar Overlay"
    button calls this for whatever collection is on screen. It scans all
    session draft sidecars and keeps the ones whose slab committed into
    ``store`` (the ``slab_id in store.slabs`` filter), so a collection
    mined across several sessions still reassembles correctly. ~thousands
    of small JSON reads — fine for a manual one-shot action.
    """
    import glob
    from pathlib import Path

    root = Path(session_store.root)
    recs = _collect_slab_records(
        store, glob.glob(str(root / "*" / "drafts" / "*_raw.json"))
    )
    return await _pillars_from_slab_records(
        store, recs, origin, "collection_sidecars",
        regenerate_summaries=regenerate_summaries, replace=replace,
    )
