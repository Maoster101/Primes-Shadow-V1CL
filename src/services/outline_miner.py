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


# ─── Recursive density audit (genre-neutral via prompt param) ────────────
#
# Stage A of the paper miner and the doc miner's recursive mode. Same
# machinery — the prompts differ to encode genre-specific calibration:
# papers are densely argumentative (PAPER_DENSITY_AUDIT_SYSTEM is more
# willing to subdivide); design docs are mostly flat per-subsystem
# (_DOC_DENSITY_AUDIT_SYSTEM is conservative). Density audit AND leaf
# drill are independent levers — the audit controls hierarchy depth,
# the drill controls slab density. Tuning them separately is what lets
# each genre's miner pick its own calibration.


_DOC_DENSITY_AUDIT_SYSTEM = """You are auditing the structure of a \
DESIGN DOCUMENT or TECHNICAL SPECIFICATION. You will be given several \
CURRENT LEAF SECTIONS — sections as currently identified. For each \
one, decide whether it is ONE coherent topic / subsystem that should \
stay as-is, OR whether it covers MULTIPLE genuinely-distinct \
sub-subsystems and should be subdivided.

DESIGN DOCS ARE MOSTLY FLAT. Design docs and technical specs organise \
content by subsystem / feature / component; each subsystem is usually \
one coherent thing developed in one section. **Be CONSERVATIVE: the \
default answer is `subdivide: false`.** Most leaves should stay.

A section should STAY (subdivide=false) when:
  - It describes ONE subsystem, feature, component, or concept.
  - It is a scope / boundary / intent block, or a list of design
    principles.
  - Its content enumerates properties or details of ONE subject.
  - It contains a list of related items that share one parent concept
    (the listing IS the topic).
  - It is a short section — anything under ~600 chars or ~10 sentences.

A section should SUBDIVIDE (subdivide=true) ONLY when:
  - It clearly groups 2+ NAMED sub-subsystems or sub-features, each
    described as its own designed thing.
  - Example: a section "Sensory Layers" that develops Water + Air +
    Sound + Light as four separately-designed subsystems.
  - The sub-sections you name must each be a coherent unit on their
    own — not "first half" / "second half" of the section.

If you find yourself looking for excuses to subdivide, the answer is
`subdivide: false`. **A flat hierarchy is the correct outcome for most
design-doc sections.** Subdivide only when the substructure jumps out
of the text — usually because the prose itself uses sub-headings or
explicitly names sub-subsystems.

For each leaf you decide to subdivide, name 2-5 sub-sections. Each:
  - label: 2-6 word headline naming the sub-subsystem.
  - starts_with: SHORT verbatim quote (6-12 words) copied EXACTLY from
    THIS leaf's text, marking where the sub-section begins. Used to
    locate the boundary — must appear verbatim in the leaf's text.

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


def _get_section_index(sections: list[Section], target: Section) -> int:
    """Find a section's index by identity (handles same-label collisions)."""
    for i, s in enumerate(sections):
        if s is target:
            return i
    return -1


def _create_sub_sections(
    parent: Section,
    sub_sections_data: list[dict],
    all_sections: list[Section],
) -> list[Section]:
    """Locate the audit's sub-section markers within ``parent.text`` and
    mint new Section objects with ``parent_idx`` set.

    The first sub-section is forced to start at the parent's body_start
    even if its marker resolves later — text before the first marker
    would otherwise be orphaned. Sub-sections tile the parent's text in
    order, last one running to ``parent.char_end``.

    Returns [] when fewer than 2 markers locate — subdivision into a
    single child is meaningless (the parent stays as a leaf).
    """
    parent_idx = _get_section_index(all_sections, parent)
    if parent_idx < 0:
        return []
    parent_text = parent.text

    positions: list[tuple[int, str]] = []
    cursor = 0
    for sd in sub_sections_data:
        marker = (sd.get("starts_with") or "").strip()
        label = (sd.get("label") or "").strip()
        if not label or not marker:
            continue
        pos, _ = _locate_marker(parent_text, marker, cursor)
        if pos < 0:
            continue
        positions.append((pos, label))
        cursor = pos + 1

    if len(positions) < 2:
        return []

    # First sub-section absorbs any prefix text — never orphan content.
    positions[0] = (0, positions[0][1])

    new_subs: list[Section] = []
    for i, (pos, label) in enumerate(positions):
        sub_char_start = parent.body_start + pos
        next_pos = positions[i + 1][0] if i + 1 < len(positions) else None
        if next_pos is not None:
            sub_char_end = parent.body_start + next_pos
            sub_text = parent_text[pos:next_pos]
        else:
            sub_char_end = parent.char_end
            sub_text = parent_text[pos:]
        new_subs.append(Section(
            level=parent.level + 1,
            label=label,
            char_start=sub_char_start,
            body_start=sub_char_start,
            char_end=sub_char_end,
            parent_idx=parent_idx,
            text=sub_text.strip(),
        ))
    return new_subs


async def _audit_density(
    leaves: list[Section], paths: dict[int, str], audit_system: str,
) -> list[dict]:
    """One batched LLM call: for each leaf, decide subdivide-or-not.

    Each leaf is shown with its **digested** text (paragraph ledes when
    long, full text when short) so a single call covers many leaves
    without blowing num_ctx. The audit decides from the leaf's *shape*
    — where topic shifts happen — which the digest preserves. Returned
    ``starts_with`` markers are still verbatim sentence ledges that
    locate in the FULL leaf text downstream.

    ``audit_system`` is the genre-specific calibration — the paper
    miner passes ``PAPER_DENSITY_AUDIT_SYSTEM``; the doc miner passes
    ``_DOC_DENSITY_AUDIT_SYSTEM``. Same plumbing, different bias.

    Returns a list aligned with ``leaves``: each entry is the model's
    decision dict (or a default ``{"subdivide": False}`` on a parse miss).
    """
    if not leaves:
        return []

    parts: list[str] = []
    for i, leaf in enumerate(leaves):
        path = paths.get(id(leaf), leaf.label)
        sample = _digest_for_outline(leaf.text, whole_below=4000)
        parts.append(f'LEAF {i} (id=l{i}, path="{path}"):')
        parts.append(f'"""\n{sample}\n"""')
        parts.append("")
    user = (
        "\n".join(parts)
        + "\nFor each leaf above, decide subdivide-or-not per the rules. "
        "Use the leaf's own text for any starts_with markers."
    )

    try:
        resp = await ollama.structured_extract(
            user, system=audit_system, num_ctx=32768, timeout=600,
        )
    except Exception as exc:
        logger.warning("Density audit LLM call failed: %r", exc)
        return [{"subdivide": False} for _ in leaves]

    raw = resp.get("decisions", []) if isinstance(resp, dict) else []
    by_id: dict[str, dict] = {}
    for d in raw:
        if isinstance(d, dict) and isinstance(d.get("id"), str):
            by_id[d["id"]] = d

    out: list[dict] = []
    for i in range(len(leaves)):
        out.append(by_id.get(f"l{i}", {"subdivide": False}))
    return out


async def _build_recursive_hierarchy(
    raw_text: str, *,
    top_system: str, audit_system: str,
    max_depth: int = 10, min_leaf_chars: int = 200,
) -> tuple[list[Section], set[int], str]:
    """Build the recursive section tree. Returns (sections, leaf_ids, route).

    Loop:
      1. Top pillars from headings (deterministic) or LLM digest with
         ``top_system``.
      2. While a non-empty frontier of unaudited leaves exists:
         - Drop leaves below the structural floor (or at max depth) —
           they go straight to final leaves.
         - Batched density audit on the rest, using ``audit_system``.
         - subdivide=false → final leaf.
         - subdivide=true → create sub-sections (parent_idx + char
           offsets), append to sections, recurse on them next pass.
      3. Halt when frontier is empty.

    Cap and floor are safety; the principled halt is the audit
    returning no-subdivide for every remaining leaf. ``top_system`` and
    ``audit_system`` carry the genre-specific calibration.
    """
    # Pass 1: top-level pillars
    sections = parse_heading_outline(raw_text)
    route = "headings"
    if not sections:
        route = "llm_top"
        sections = await _outline_via_llm(raw_text, system=top_system)
    if not sections:
        return [], set(), route

    to_audit: list[Section] = list(sections)
    final_leaves: set[int] = set()
    pass_num = 1

    while to_audit:
        pass_num += 1
        # Structural floor / depth cap filter — these never go to the LLM.
        next_to_audit: list[Section] = []
        for leaf in to_audit:
            if len(leaf.text) < min_leaf_chars or leaf.level >= max_depth:
                final_leaves.add(id(leaf))
            else:
                next_to_audit.append(leaf)
        if not next_to_audit:
            break

        # Compute current paths once for this audit batch.
        paths = _compute_section_paths(sections)

        mining_progress.set_phase("density_audit", total=1)
        decisions = await _audit_density(next_to_audit, paths, audit_system)
        mining_progress.increment()

        new_frontier: list[Section] = []
        for leaf, decision in zip(next_to_audit, decisions):
            if not decision.get("subdivide"):
                final_leaves.add(id(leaf))
                continue
            sub_data = decision.get("sub_sections") or []
            new_subs = _create_sub_sections(leaf, sub_data, sections)
            if not new_subs:
                final_leaves.add(id(leaf))
                continue
            sections.extend(new_subs)
            new_frontier.extend(new_subs)

        logger.info(
            "Stage A pass %d: audited %d leaves, subdivided %d into %d new leaves",
            pass_num, len(next_to_audit),
            sum(1 for d in decisions if d.get("subdivide")),
            len(new_frontier),
        )
        to_audit = new_frontier

    return sections, final_leaves, route


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
        recursive: bool = False,
    ) -> dict:
        """Mine a structured document outline-first.

        ``recursive=True`` switches to N-tier hierarchy emergence: a
        conservative density audit (``_DOC_DENSITY_AUDIT_SYSTEM``) walks
        the outline, subdividing only sections that group genuinely-distinct
        sub-subsystems. The existing per-section drill prompt is kept,
        so slab density stays as design-doc-calibrated as before — only
        the hierarchy depth changes. The non-recursive default (2-tier)
        is unchanged.
        """
        if recursive:
            return await self._mine_recursive(
                raw_text, source_label=source_label,
                min_confidence=min_confidence,
            )

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

    async def _mine_recursive(
        self,
        raw_text: str,
        source_label: str = "document",
        min_confidence: float = 0.4,
    ) -> dict:
        """N-tier recursive variant of mine() — design-doc-calibrated.

        Same outline-first machinery the paper miner uses, but with a
        CONSERVATIVE density audit (``_DOC_DENSITY_AUDIT_SYSTEM``) and
        the EXISTING doc drill (``_CHAPTER_EXTRACT_SYSTEM``). Recursion
        adds depth; the drill controls density; tuning them separately
        is what lets the doc miner produce shallow trees on design docs
        (where the audit should mostly say "no, this is one subsystem")
        while keeping its known-good slab calibration.

        Reuses the "paper" miner_kind in mining_progress — phase names
        and weights align with the paper miner; only the prompts differ.
        """
        # Reuse paper miner_kind — same phase sequence (outlining /
        # density_audit / leaf_drill / pillar_summaries / cross_pillar
        # / outline_edges / consolidation), only the underlying prompts
        # carry the doc-vs-paper calibration difference.
        mining_progress.reset("paper")

        # ── Stage A — recursive hierarchy with doc prompts ───────────
        mining_progress.set_phase("outlining", total=1)
        sections, leaf_ids, route = await _build_recursive_hierarchy(
            raw_text,
            top_system=_OUTLINE_SUMMARIZE_SYSTEM,
            audit_system=_DOC_DENSITY_AUDIT_SYSTEM,
        )
        mining_progress.increment()

        if not sections:
            mining_progress.mark_error("no outline could be derived")
            return {
                "format": "outline", "segments": 0, "outline": [],
                "proposals": [], "edges": [], "proposal_count": 0,
                "edge_count": 0, "source_label": source_label,
                "error": "Could not derive a top-level structure. Try "
                         "the Narrative miner for unstructured text.",
            }

        paths = _compute_section_paths(sections)
        leaves = [s for s in sections if id(s) in leaf_ids]
        leaves.sort(key=lambda s: s.char_start)
        max_depth_observed = max((s.level for s in sections), default=1)

        logger.info(
            "OutlineMiner (recursive): route=%s, %d total sections, "
            "%d leaves, max_depth=%d",
            route, len(sections), len(leaves), max_depth_observed,
        )

        # ── Stage B — drill each leaf with the existing doc prompt ──
        # _drill_chapter defaults to _CHAPTER_EXTRACT_SYSTEM; that's
        # the slab density the doc miner has always produced.
        mining_progress.set_phase("leaf_drill", total=len(leaves))
        sem = asyncio.Semaphore(_PASS_PARALLEL)
        order_bases = [i * 1000 for i in range(len(leaves))]

        async def _drill(idx: int) -> ChapterResult:
            async with sem:
                r = await _drill_chapter(
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

        # ── Stage C — summaries (bottom-up propagation) ──────────────
        slabs_by_path: dict[str, list[MiningProposal]] = {}
        for s in all_slabs:
            slabs_by_path.setdefault(s.source_topic, []).append(s)

        n_non_leaves = sum(1 for s in sections if id(s) not in leaf_ids)
        mining_progress.set_phase(
            "pillar_summaries",
            total=max(1, len(leaves) + n_non_leaves),
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

        non_leaves_by_depth = sorted(
            (s for s in sections if id(s) not in leaf_ids),
            key=lambda s: -s.level,
        )
        for nl in non_leaves_by_depth:
            child_items: list[str] = []
            for c in sections:
                if c.parent_idx >= 0 and sections[c.parent_idx] is nl:
                    child_items.append(
                        f"{c.label}: {summaries.get(id(c), '')}"
                    )
            if child_items:
                summaries[id(nl)] = await _summarize_pillar(
                    nl.label, child_items,
                )
            mining_progress.increment()

        # ── Pass 4 — cross-pillar at leaves ──────────────────────────
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
            "format": "outline",
            "outline_route": route,
            "segments": len(leaves),
            "total_sections": len(sections),
            "max_depth": max_depth_observed,
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
    a whole collection's). Each carries a ``source_topic`` that is a
    " / "-separated path through the hierarchy ("Part / Chapter / Sub /
    Leaf" for paper-mined corpora; "Section" or "Part / Chapter" for the
    doc/narrative miners).

    The path itself encodes the tree — every prefix of a leaf's path is
    a node in the hierarchy, the deepest segment is the leaf carrying
    slabs. This function synthesises that **prefix tree** as a list of
    ``Section`` objects with ``parent_idx`` chained back, regenerates
    leaf summaries from slabs and non-leaf summaries bottom-up, then
    hands off to ``build_outline_tree_recursive`` / ``build_pillars_recursive``
    — which handle arbitrary depth in one shape, collapsing what used to
    be flat / 2-tier / N-tier branches into one path.
    """
    if not recs:
        return {
            "pillars_created": 0,
            "error": "No committed outline-mined slabs found. Push and "
                     "promote a doc/paper mine into this collection first.",
        }

    # ── Group records by leaf path. Every distinct topic IS a leaf —
    # the outline miners only stamp source_topic at the leaf level.
    leaf_data: dict[str, dict] = {}
    for r in recs:
        d = leaf_data.setdefault(
            r["topic"], {"ids": [], "order": r["order"], "cross": {}},
        )
        d["ids"].append(r["id"])
        d["order"] = min(d["order"], r["order"])
        for to_topic in r["cross"]:
            if to_topic and to_topic != r["topic"]:
                d["cross"][to_topic] = d["cross"].get(to_topic, 0) + 1

    leaf_paths = set(leaf_data.keys())

    # ── Synthesise the prefix tree. Every prefix of every leaf path is
    # a Section node; sort shallow-first so each child's parent_idx is
    # already in ``sections`` when the child is appended.
    all_prefixes: set[str] = set()
    for p in leaf_paths:
        parts = p.split(" / ")
        for i in range(1, len(parts) + 1):
            all_prefixes.add(" / ".join(parts[:i]))

    sections: list[Section] = []
    section_idx: dict[str, int] = {}
    # Sort by (depth, document_order). Leaves carry an order from
    # source_pairs; non-leaves inherit the min order of their leaf
    # descendants so sibling ordering stays in document order.
    def _prefix_order(prefix: str) -> int:
        depth_orders = [
            leaf_data[p]["order"] for p in leaf_paths
            if p == prefix or p.startswith(prefix + " / ")
        ]
        return min(depth_orders) if depth_orders else 10 ** 9

    for prefix in sorted(all_prefixes, key=lambda p: (p.count(" / "), _prefix_order(p), p)):
        parts = prefix.split(" / ")
        depth = len(parts)
        parent_path = " / ".join(parts[:-1]) if depth > 1 else None
        parent_idx = section_idx.get(parent_path, -1) if parent_path else -1
        sec = Section(
            level=depth, label=parts[-1],
            char_start=_prefix_order(prefix),
            body_start=0, char_end=0,
            parent_idx=parent_idx, text="",
        )
        section_idx[prefix] = len(sections)
        sections.append(sec)

    leaf_ids: set[int] = {id(sections[section_idx[p]]) for p in leaf_paths}

    # ── Summaries: leaves from their slabs (parallel), non-leaves from
    # their children's summaries walked bottom-up.
    summaries: dict[int, str] = {}
    if regenerate_summaries:
        sem = asyncio.Semaphore(_PASS_PARALLEL)

        async def _leaf_summ(path: str) -> tuple[str, str]:
            async with sem:
                items = []
                for sid in leaf_data[path]["ids"]:
                    sl = store.slabs.get(sid)
                    if sl:
                        items.append(
                            f"{sl.title}: {(sl.canonical_text or '')[:160]}"
                        )
                summ = await _summarize_pillar(path.split(" / ")[-1], items)
                return path, summ

        leaf_pairs = await asyncio.gather(*[_leaf_summ(p) for p in leaf_paths])
        for path, summ in leaf_pairs:
            summaries[id(sections[section_idx[path]])] = summ

        # Non-leaves bottom-up (deepest first) — each non-leaf's children
        # are ready before it's summarised.
        non_leaves_by_depth = sorted(
            ((i, s) for i, s in enumerate(sections) if id(s) not in leaf_ids),
            key=lambda pair: -pair[1].level,
        )
        for idx, nl in non_leaves_by_depth:
            child_items: list[str] = []
            for c in sections:
                if c.parent_idx == idx:
                    child_items.append(
                        f"{c.label}: {summaries.get(id(c), '')}"
                    )
            if child_items:
                summaries[id(nl)] = await _summarize_pillar(nl.label, child_items)

    # ── Aggregate per-slab cross_pillars to leaf-level cross-edges.
    cross_edges: dict[tuple[str, str], int] = {}
    for path, d in leaf_data.items():
        for to_topic, count in d["cross"].items():
            cross_edges[(path, to_topic)] = count

    # ── Slab data per leaf path, for the build.
    slab_ids_by_path = {p: leaf_data[p]["ids"] for p in leaf_paths}
    slab_titles_by_path: dict[str, list[str]] = {}
    for p in leaf_paths:
        titles: list[str] = []
        for sid in leaf_data[p]["ids"]:
            sl = store.slabs.get(sid)
            if sl:
                titles.append(getattr(sl, "title", "") or "")
        slab_titles_by_path[p] = titles

    # ── Build the outline tree (N-tier capable) and the pillars
    # recursively. These functions handle 1-tier, 2-tier, and N-tier
    # inputs uniformly — that's the whole point of the recursive build.
    outline_tree = build_outline_tree_recursive(
        sections, leaf_ids, summaries,
        slab_titles_by_path, cross_edges,
        slab_ids_by_path=slab_ids_by_path,
    )
    report = build_pillars_recursive(
        store, outline_tree, origin=origin, replace=replace,
    )
    report["sections"] = len(leaf_paths)
    report["total_sections"] = len(sections)
    report["max_depth"] = max((s.level for s in sections), default=1)
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


# ─── N-tier recursive build (for the paper miner) ────────────────────────
#
# Counterparts to build_outline_tree / build_pillars_from_outline. The
# 2-tier versions stay live for the doc and narrative miners; these
# handle arbitrary-depth trees produced by the paper miner's Stage A
# recursive density audit. The section tree carries depth implicitly via
# parent_idx (same shape parse_heading_outline produces for nested
# markdown headings — the data model was always N-tier capable; only
# the build was capped).


def _compute_section_paths(sections: list[Section]) -> dict[int, str]:
    """For each section, compute its " / "-joined ancestor-label path."""
    paths: dict[int, str] = {}
    for sec in sections:
        labels: list[str] = []
        cur = sec
        while True:
            labels.append(cur.label)
            if cur.parent_idx < 0:
                break
            cur = sections[cur.parent_idx]
        paths[id(sec)] = " / ".join(reversed(labels))
    return paths


def build_outline_tree_recursive(
    sections: list[Section],
    leaf_ids: set[int],
    summaries: dict[int, str],
    slab_titles_by_path: dict[str, list[str]],
    cross_edges: dict[tuple[str, str], int],
    slab_ids_by_path: dict[str, list[str]] | None = None,
) -> list[dict]:
    """Build a nested-dict outline tree from an N-tier section forest.

    Recursive counterpart of ``build_outline_tree``. Each section emits
    a node:
      - **Leaf** (id in ``leaf_ids``) — carries section_path, slab_titles,
        cross_edges. This is where slabs were drilled in Stage B.
      - **Non-leaf** — carries ``children`` (recursive). No slabs of its
        own; it groups sub-pillars.

    Top-level sections (``parent_idx < 0``) form the return list. The
    section tree is taken as-is from Stage A's recursive density audit;
    this function is pure plumbing.
    """
    paths = _compute_section_paths(sections)

    cross_by_home: dict[str, list[dict]] = {}
    for (home_path, target_path), weight in cross_edges.items():
        cross_by_home.setdefault(home_path, []).append({
            "to": target_path.split(" / ")[-1],
            "to_path": target_path,
            "weight": weight,
        })

    children_of: dict[int, list[Section]] = {}
    for sec in sections:
        if sec.parent_idx >= 0:
            parent = sections[sec.parent_idx]
            children_of.setdefault(id(parent), []).append(sec)
    # Children in document order — the SEQUENCE spine depends on it.
    for k in children_of:
        children_of[k].sort(key=lambda s: s.char_start)

    def _to_node(sec: Section) -> dict:
        path = paths[id(sec)]
        node = {
            "label": sec.label,
            "level": sec.level,
            "summary": summaries.get(id(sec), ""),
        }
        if id(sec) in leaf_ids:
            node["section_path"] = path
            node["slab_titles"] = slab_titles_by_path.get(path, [])
            # slab_ids takes precedence in build_pillars_recursive (exact,
            # no greedy-by-title round-trip) — emit it when the caller
            # has resolved IDs already, as the rebuild path does.
            if slab_ids_by_path is not None:
                node["slab_ids"] = slab_ids_by_path.get(path, [])
            node["cross_edges"] = sorted(
                cross_by_home.get(path, []), key=lambda x: -x["weight"],
            )
        else:
            node["children"] = [_to_node(c) for c in children_of.get(id(sec), [])]
        return node

    tops = sorted(
        (s for s in sections if s.parent_idx < 0),
        key=lambda s: s.char_start,
    )
    return [_to_node(t) for t in tops]


def build_pillars_recursive(
    store, outline_tree: list[dict], origin: str = "",
    replace: bool = True,
) -> dict:
    """Build PillarDefinitions from an N-tier nested outline tree.

    Recursive counterpart of ``build_pillars_from_outline``. Walks the
    tree once depth-first, minting a PillarDefinition per node:
      - **Leaf** — ``members`` = resolved slab IDs (greedy by title).
      - **Non-leaf** — ``children`` = sub-pillar IDs.
    Every pillar's ``parent`` is set, threading the tree back up.

    A second pass lifts cross-edges (per-leaf in the tree) to
    PillarCrossEdge objects — same machinery as the 2-tier version.

    Pillars whose subtree resolves to zero members are dropped, with
    their parent's child list pruned to match.
    """
    title_index = _index_slabs_by_title(store)
    used: set[str] = set()
    seen_ids: set[str] = set()
    unresolved: list[str] = []
    pillars: dict[str, PillarDefinition] = {}
    path_to_pillar: dict[str, str] = {}
    deferred_cross: dict[str, list[dict]] = {}

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

    def _build(node: dict, parent_id: str | None) -> str | None:
        """Build a pillar for this node + descendants. Returns the
        pillar id, or None if the whole subtree resolved empty."""
        pid = _new_id(node.get("label") or "Section")

        if "children" in node:
            child_ids: list[str] = []
            for c in node["children"]:
                cid = _build(c, parent_id=pid)
                if cid:
                    child_ids.append(cid)
            if not child_ids:
                return None  # empty subtree — drop the container
            pillars[pid] = PillarDefinition(
                id=pid,
                label=(node.get("label") or "Section").strip(),
                summary=(node.get("summary") or "").strip(),
                children=child_ids,
                parent=parent_id,
                origin=origin,
                pillar_role="section",
            )
        else:
            # Leaf: prefer slab_ids (exact, from session rebuild) over
            # slab_titles (greedy by title, from fresh mine). Mirrors
            # build_pillars_from_outline's resolution priority.
            ids = node.get("slab_ids")
            if ids:
                members = [
                    s for s in ids if s in store.slabs and s not in used
                ]
                used.update(members)
            else:
                members = _resolve(node.get("slab_titles", []))
            if not members:
                return None
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

    for top in outline_tree:
        _build(top, parent_id=None)

    # Cross-edges — second pass, resolve to_path to pillar IDs.
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
            "error": "No pillars built — no outline slab titles resolved.",
            "unresolved_titles": unresolved,
        }

    if replace:
        store.pillars = pillars
    else:
        store.pillars.update(pillars)

    errors = store.validate()
    store.save()

    # Compute max depth by walking parent chains (useful for verifying
    # the recursion actually produced depth in the report).
    leaf_count = sum(1 for p in pillars.values() if p.members)
    non_leaf = sum(1 for p in pillars.values() if p.children)
    max_depth = 1
    for p in pillars.values():
        d, cur = 1, p
        while cur.parent:
            cur = pillars.get(cur.parent)
            if not cur:
                break
            d += 1
        max_depth = max(max_depth, d)

    logger.info(
        "build_pillars_recursive: %d pillars (%d leaf, %d non-leaf, "
        "depth %d), %d slabs placed, %d cross-edges, %d unresolved",
        len(pillars), leaf_count, non_leaf, max_depth,
        len(used), cross_count, len(unresolved),
    )
    return {
        "pillars_created": len(pillars),
        "leaf_pillars": leaf_count,
        "non_leaf_pillars": non_leaf,
        "max_depth": max_depth,
        "members_resolved": len(used),
        "cross_edges": cross_count,
        "unresolved_titles": unresolved,
        "validation_errors": errors,
    }
