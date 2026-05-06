"""Narrative / document miner.

Sibling of ConversationMiner for cohesive single-author content — stories,
essays, design docs, condensed pitches — where ordering is load-bearing
and there is no turn structure.

Key differences from ConversationMiner:
  - Segmentation: paragraph-based, not exchange-pair-based
  - Ordering: each proposal carries an `order` field; SEQUENCE edges are
    emitted between consecutive slabs as a first-class extraction product
  - Density: the LLM is encouraged to condense dense prose and reconstitute
    terse/schematic passages to a corpus-appropriate fullness rather than
    hit a fixed proposal count

Shares with ConversationMiner:
  - MiningProposal / EdgeProposal dataclasses
  - deduplicate_proposals()
  - Ollama structured_extract plumbing (via services.ollama)

Output shape is intentionally identical to ConversationMiner.mine() so the
/push-mined endpoint and downstream dreaming pass work unchanged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from .convo_miner import (
    MiningProposal,
    EdgeProposal,
    deduplicate_proposals,
)
from . import ollama

logger = logging.getLogger(__name__)

# ─── Pass concurrency ────────────────────────────────────────────────────
# Match the daemon's OLLAMA_NUM_PARALLEL — running more concurrent in-flight
# requests than Ollama has slots just queues them client-side without
# improving throughput. Default 2 is safe on a 16GB card with gemma3:12b.
# Set higher if you've configured a larger NUM_PARALLEL on the daemon.
import os as _os
_PASS_PARALLEL = int(_os.environ.get("PS_MINING_PARALLEL", "2"))


# ─── Segmentation ────────────────────────────────────────────────────────


@dataclass
class NarrativeSegment:
    """A contiguous passage of narrative text with a stable order index."""
    order: int                  # 0-based position in source
    heading: str = ""           # optional section heading if detected
    text: str = ""              # the passage body
    char_start: int = 0         # start offset in source (for provenance)


_HEADING_RE = re.compile(r"^(#{1,6}\s+.+|[A-Z][A-Z0-9 \-:_.]{3,}$)")


def segment_narrative(raw: str, max_chars: int = 1800, min_chars: int = 120) -> list[NarrativeSegment]:
    """Split narrative text into ordered segments suitable for extraction.

    Strategy:
      1. Split by blank-line paragraph boundaries.
      2. Attach markdown/SHOUT-CASE headings to the following paragraph.
      3. Merge short paragraphs forward until each segment is >= min_chars
         (keeps narrative beats intact — a one-line scene setter shouldn't
         become its own extraction call).
      4. Split paragraphs that exceed max_chars at sentence boundaries.

    Order is preserved — segment.order is monotonic. This is the spine
    the narrative miner uses to emit SEQUENCE edges downstream.
    """
    raw = raw.strip()
    if not raw:
        return []

    # Normalize line endings
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    # Split on blank lines
    paras = re.split(r"\n\s*\n", raw)

    # Pair headings with their content
    pending_heading = ""
    packed: list[tuple[str, str]] = []  # (heading, text)
    for p in paras:
        p = p.strip()
        if not p:
            continue
        if _HEADING_RE.match(p):
            pending_heading = p.lstrip("# ").strip()
            continue
        packed.append((pending_heading, p))
        pending_heading = ""

    # Merge tiny paragraphs forward
    merged: list[tuple[str, str]] = []
    buf_heading = ""
    buf_text = ""
    for heading, text in packed:
        if buf_text and len(buf_text) < min_chars:
            # extend current buffer
            buf_text = (buf_text + "\n\n" + text).strip()
        else:
            if buf_text:
                merged.append((buf_heading, buf_text))
            buf_heading = heading
            buf_text = text
    if buf_text:
        merged.append((buf_heading, buf_text))

    # Split oversized paragraphs at sentence boundaries
    segments: list[NarrativeSegment] = []
    order = 0
    cursor = 0
    for heading, text in merged:
        if len(text) <= max_chars:
            segments.append(NarrativeSegment(order=order, heading=heading, text=text, char_start=cursor))
            order += 1
            cursor += len(text)
            continue
        # Sentence split
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", text)
        buf = ""
        for s in sentences:
            if len(buf) + len(s) + 1 > max_chars and buf:
                segments.append(NarrativeSegment(order=order, heading=heading, text=buf.strip(), char_start=cursor))
                order += 1
                cursor += len(buf)
                buf = s
            else:
                buf = (buf + " " + s).strip() if buf else s
        if buf:
            segments.append(NarrativeSegment(order=order, heading=heading, text=buf.strip(), char_start=cursor))
            order += 1
            cursor += len(buf)

    return segments


# ─── Extraction prompt ────────────────────────────────────────────────────


NARRATIVE_EXTRACTION_PROMPT = """You are a corpus extraction specialist for Prime's Shadow, a neuro-symbolic knowledge system.

The input is a cohesive NARRATIVE or DOCUMENT — a story, essay, scene, design doc, condensed pitch, or similar single-author piece. This is NOT a chat log. There is no user/assistant structure. The text has a natural flow and ordering that you must preserve.

SEGMENT {order} of the source{heading_suffix}:
\"\"\"
{text}
\"\"\"

{context_block}
Your job: extract corpus objects that capture this segment at the RIGHT DENSITY. You have freedom here:
  - If the prose is dense and rich, CONDENSE faithfully — distill into the core beats/claims/entities.
  - If the prose is terse, schematic, or in note-form, RECONSTITUTE — expand into complete sentences the corpus can store. The goal is corpus-appropriate fullness, not verbatim copying.
  - If the segment is thin (small-talk, throwaway), return few or zero proposals.

Object types:

1. **Slabs** — the primary unit. 3-10 complete sentences capturing a SCENE, BEAT, SECTION, CLAIM-STEP, or THEMATIC MOVEMENT. For stories: one beat per slab ideally (setup, complication, reversal, etc). For essays: one claim-plus-support per slab. Write full sentences — the slab is stored as canonical text and must stand on its own.

2. **Anchors** — 2-8 word phrases naming distinctive ENTITIES: characters, places, named concepts, technologies, factions, idioms the piece coins. Include 1-3 aliases if the piece uses alt names. Only create anchors for things that would be referenced elsewhere — not every noun deserves an anchor.

3. **Bundles** — thematic/structural groupings of 3+ anchors that cohere at a higher level: an act, an arc, a thesis with its supporting ideas, a faction-and-its-members, a through-line.

Return ONLY valid JSON:
{{
  "proposals": [
    {{
      "type": "slab",
      "title": "brief beat/section title",
      "canonical_text": "the full 3-10 sentence passage capturing this beat — condensed or reconstituted as appropriate",
      "narrative_role": "SETUP | INCITING | RISING | CLIMAX | RESOLUTION | CLAIM | EVIDENCE | SECTION | OTHER",
      "justification": "why this beat matters to the narrative spine",
      "confidence": 0.0-1.0
    }},
    {{
      "type": "anchor",
      "canonical_phrase": "distinctive entity name",
      "aliases": ["alt1", "alt2"],
      "justification": "what this entity is and why it recurs or matters",
      "confidence": 0.0-1.0
    }},
    {{
      "type": "bundle",
      "label": "arc / theme / act label",
      "members": ["anchor phrase 1", "anchor phrase 2", "anchor phrase 3"],
      "justification": "what ties these together at the higher level",
      "confidence": 0.0-1.0
    }}
  ]
}}

Rules:
- PRESERVE ORDER — output proposals in the order they appear in the segment. The narrative spine depends on it.
- Slabs are preferred for beats and sections; anchors are for reusable named entities.
- DO NOT truncate slab canonical_text mid-thought. Write complete sentences.
- A single segment might yield 1 slab + 2 anchors, or 3 slabs + 1 bundle, or 0 proposals if the segment is filler. Don't force a count.
- Confidence 0.8+ = explicit in the text; 0.5-0.8 = clearly implied; <0.5 = interpretive reach.
- narrative_role is optional but helpful — use "OTHER" if unsure.
"""


async def extract_from_segment(
    segment: NarrativeSegment,
    prior_anchor_phrases: list[str],
    prior_slab_titles: list[str],
) -> list[MiningProposal]:
    """Run the LLM on a single segment and return proposals (in source order)."""
    heading_suffix = f" (heading: \"{segment.heading}\")" if segment.heading else ""
    context_block = ""
    if prior_anchor_phrases:
        context_block = (
            "Anchors extracted from earlier segments (do NOT re-propose if the same entity — we dedupe later but prefer continuity):\n  - "
            + "\n  - ".join(prior_anchor_phrases[-20:])
            + "\n\n"
        )
    if prior_slab_titles:
        context_block += (
            "Earlier beat titles (the spine so far):\n  - "
            + "\n  - ".join(prior_slab_titles[-10:])
            + "\n\n"
        )

    prompt = NARRATIVE_EXTRACTION_PROMPT.format(
        order=segment.order,
        heading_suffix=heading_suffix,
        text=segment.text,
        context_block=context_block,
    )

    try:
        resp = await ollama.structured_extract(prompt)
    except Exception as e:
        # Include exception type + repr — bare str(e) is empty for
        # httpx.ReadTimeout / asyncio.TimeoutError which are the common
        # failure modes on slow models, making silent drops undebuggable.
        logger.warning(
            "Narrative extraction failed on segment %d (%s): %r",
            segment.order, type(e).__name__, e,
        )
        return []

    # Parse JSON response
    try:
        if isinstance(resp, dict):
            data = resp
        else:
            # strip code fences if present
            txt = str(resp).strip()
            if txt.startswith("```"):
                txt = re.sub(r"^```(?:json)?\s*", "", txt)
                txt = re.sub(r"\s*```$", "", txt)
            data = json.loads(txt)
    except Exception as e:
        logger.warning("Narrative parse failed on segment %d: %s", segment.order, e)
        return []

    raw_proposals = data.get("proposals", []) if isinstance(data, dict) else []
    out: list[MiningProposal] = []
    # Slab counter for intra_segment_order — increments only when a slab
    # is actually emitted, so the index is contiguous across emitted
    # slabs even if anchors/bundles are interleaved in the model output.
    # SEQUENCE-edge derivation uses this as a tiebreaker when a segment
    # yields multiple slabs.
    slab_idx = 0
    for rp in raw_proposals:
        if not isinstance(rp, dict):
            continue
        ptype = (rp.get("type") or "").lower().strip()
        if ptype not in ("anchor", "slab", "bundle"):
            continue
        try:
            conf = float(rp.get("confidence", 0.5))
        except Exception:
            conf = 0.5

        source_topic = rp.get("narrative_role", "") or segment.heading or "narrative"

        if ptype == "anchor":
            phrase = (rp.get("canonical_phrase") or "").strip()
            if not phrase:
                continue
            out.append(MiningProposal(
                proposal_type="anchor",
                canonical_phrase=phrase,
                aliases=list(rp.get("aliases") or [])[:5],
                source_topic=source_topic,
                confidence=conf,
                source_pairs=[segment.order],
                justification=rp.get("justification", ""),
            ))
        elif ptype == "slab":
            text = (rp.get("canonical_text") or "").strip()
            title = (rp.get("title") or "").strip()
            if not text:
                continue
            out.append(MiningProposal(
                proposal_type="slab",
                canonical_text=text,
                title=title or f"Beat {segment.order}",
                source_topic=source_topic,
                confidence=conf,
                source_pairs=[segment.order],
                intra_segment_order=slab_idx,
                justification=rp.get("justification", ""),
            ))
            slab_idx += 1
        elif ptype == "bundle":
            label = (rp.get("label") or "").strip()
            members = [str(m).strip() for m in (rp.get("members") or []) if str(m).strip()]
            if not label or len(members) < 2:
                continue
            out.append(MiningProposal(
                proposal_type="bundle",
                label=label,
                aliases=members,  # stash members in aliases for downstream resolution
                source_topic=source_topic,
                confidence=conf,
                source_pairs=[segment.order],
                justification=rp.get("justification", ""),
            ))

    return out


# ─── Sequence edges ───────────────────────────────────────────────────────


def build_sequence_edges(proposals: list[MiningProposal]) -> list[EdgeProposal]:
    """Emit NARRATIVE_PRECEDES (stored as SEQUENCE) edges between consecutive slabs.

    Walks proposals in (segment_order, intra_segment_order) and connects
    each slab to the next slab it sees. Edge weight is higher if the two
    slabs share an anchor/entity (continuity signal), lower otherwise
    (pure ordering).

    Why two-key sort: ``source_pairs[0]`` alone collapses every slab from
    the same segment into a single sort bucket. When the source has long
    un-headed passages (e.g. a vin diesel skit where only the post-credits
    portion has explicit beat headings), one segment can yield 5+ slabs.
    Without ``intra_segment_order`` as the second key, Python's stable
    sort preserves whatever order the LLM emitted those slabs in, which
    is not guaranteed to match the source-text order — and the resulting
    SEQUENCE chain is structurally wrong (the bug that produced
    "Concluding Reflection" as slab #2 of 8 in the v1 vin diesel mine).
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
        # Continuity signal: shared tokens in canonical text (very cheap proxy)
        a_tokens = set(re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?", a.canonical_text))
        b_tokens = set(re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?", b.canonical_text))
        shared = len(a_tokens & b_tokens)
        weight = min(0.55 + 0.1 * shared, 0.95)
        # Justification format: include intra-segment positions when
        # they're non-zero so the chain ordering is fully transparent
        # in proposed_edges.json. "beat 0/2 → beat 0/3" reads as
        # "segment 0 slab #2 → segment 0 slab #3" — diagnoses sort
        # ties at a glance without needing to cross-reference the
        # source data.
        a_seg, b_seg = a.source_pairs[0], b.source_pairs[0]
        a_pos, b_pos = a.intra_segment_order, b.intra_segment_order
        if a_pos or b_pos:
            ordering_str = f"beat {a_seg}/{a_pos} → beat {b_seg}/{b_pos}"
        else:
            ordering_str = f"beat {a_seg} → beat {b_seg}"
        edges.append(EdgeProposal(
            edge_type="SEQUENCE",
            from_label=a_label,
            to_label=b_label,
            confidence=weight,
            justification=(
                f"Narrative ordering: {ordering_str}"
                + (f" (shared entities: {shared})" if shared else "")
            ),
        ))
    return edges


# ─── Co-occurrence edges (slab ↔ anchor) ─────────────────────────────────


def build_cooccurrence_edges(proposals: list[MiningProposal]) -> list[EdgeProposal]:
    """Emit LINKS edges between slabs and anchors that co-occur.

    Without this pass, anchors mined from a narrative float disconnected
    from the slabs that discuss them — the graph has a SEQUENCE spine of
    slabs plus an orphaned cluster of entity anchors. This function stitches
    them together using two signals:

      1. **Phrase-in-text** (weight 0.8): the anchor's ``canonical_phrase``
         appears as a word-bounded match inside a slab's ``canonical_text``.
         Strong evidence the slab elaborates on the anchor concept.
      2. **Shared segment** (weight 0.55): both proposals were extracted
         from the same narrative segment (same ``source_pairs[0]``) without
         textual overlap. Weaker but meaningful — the mining LLM put them
         in the same extraction unit, so they belong to the same beat.

    Edge direction: slab → anchor. This matches the existing convention
    where ``slab.links.anchors`` points FROM the slab TO the anchors it
    mentions. Downstream `/push-mined` resolution treats both endpoints
    uniformly, so directionality is a documentation/UX choice.

    Dedup is by (slab_label_lowercase, anchor_phrase_lowercase) — one edge
    per (slab, anchor) pair regardless of how many signals fired.
    """
    slabs = [p for p in proposals if p.proposal_type == "slab"]
    anchors = [
        p for p in proposals
        if p.proposal_type == "anchor" and p.canonical_phrase
    ]
    if not slabs or not anchors:
        return []

    edges: list[EdgeProposal] = []
    seen_pairs: set[tuple[str, str]] = set()

    for slab in slabs:
        slab_label = slab.title or slab.canonical_text[:40]
        slab_text_lower = slab.canonical_text.lower()
        slab_seg = slab.source_pairs[0] if slab.source_pairs else None

        for anchor in anchors:
            phrase = anchor.canonical_phrase.strip()
            if not phrase:
                continue
            anchor_seg = anchor.source_pairs[0] if anchor.source_pairs else None

            # Phrase-in-text: word-bounded, case-insensitive
            pattern = re.escape(phrase.lower())
            phrase_match = bool(
                re.search(rf"\b{pattern}\b", slab_text_lower)
            )

            # Shared segment (same narrative beat)
            same_segment = (
                slab_seg is not None
                and anchor_seg is not None
                and slab_seg == anchor_seg
            )

            if not (phrase_match or same_segment):
                continue

            key = (slab_label.lower(), phrase.lower())
            if key in seen_pairs:
                continue
            seen_pairs.add(key)

            signals: list[str] = []
            if phrase_match:
                signals.append("phrase_in_text")
            if same_segment:
                signals.append(f"shared_segment_{slab_seg}")
            weight = 0.8 if phrase_match else 0.55

            edges.append(EdgeProposal(
                edge_type="LINKS",
                from_label=slab_label,
                to_label=phrase,
                confidence=weight,
                justification=f"Co-occurrence: {', '.join(signals)}",
            ))

    return edges


# ─── Three-pass extraction architecture ──────────────────────────────────
#
# The narrative miner runs three independent passes over the segments:
#
#   Pass 1 — anchor-only extraction, parallel across all segments.
#            Produces named entities + distinctive phrases. No slabs,
#            no bundles. Tight prompt, fast generation.
#   Pass 2 — bundle synthesis, deterministic. Cluster anchors by
#            embedding similarity + segment co-occurrence in pure
#            Python. One batched LLM call to label the resulting
#            clusters.
#   Pass 3 — slab extraction with anchor context, parallel across all
#            segments. Each segment receives the canonical anchor list
#            scoped to anchors that surface in its text. Slabs come
#            back with explicit references_anchors links, which we
#            convert to LINKS edges in the existing EdgeProposal shape.
#            Slabs may add new_anchors (Pass 1 misses) which are
#            reconciled into the canonical list.
#
# Why three passes instead of one: Pass 1 is fully parallel (no
# inter-segment dependency), Pass 2 needs the full anchor list (so
# runs once after Pass 1), Pass 3 is fully parallel given the anchor
# list. Total latency = 3 rounds × per-round-with-parallelism cost,
# vs N rounds for the legacy single-pass architecture. With
# OLLAMA_NUM_PARALLEL=2 and 16 segments, that's ~3 × 8s = 24s
# instead of 16 × 8s = 128s.

PASS1_ANCHOR_SYSTEM = """You are extracting NAMED ENTITIES and DISTINCTIVE PHRASES from a text segment for a knowledge corpus called Prime's Shadow.

Inputs vary widely: conversation transcripts, scientific papers, news articles, technical documentation, research notes, fiction, etc. The same extraction rules apply across all of them.

What to extract (anchors):
  - Named entities — specific people, places, organizations, technologies, characters, products, datasets, projects, papers, theorems, etc.
  - Distinctive terms — words or phrases the text defines, coins, or returns to repeatedly. Includes jargon, acronyms, technical terms, slang, coined metaphors.
  - Recurring distinctive concepts — phrases the text uses as handles, even if not formally defined.

What NOT to extract:
  - Multi-step processes, methodologies, protocols, algorithms, narrative beats, arcs, or any content that needs more than a phrase to capture. Slab pass.
  - Extended arguments, hypotheses, findings, principles, claims expressed in 3+ sentences. Slab pass.
  - Single common words ("energy", "voice", "system") unless distinctively coined or used in a specific technical sense.
  - Surface mentions that don't have a clear referent (e.g. a passing "they said" without a named subject).

Granularity rules:
  - Prefer ONE longer phrase over two shorter phrases when the longer phrase carries the concept.
  - Include aliases for surface variants the text actually uses (acronyms, abbreviations, name variants, alt spellings).
  - If a concept feels like an explanation rather than a name, skip it — slab pass picks it up.

Density:
  - Match the segment's information density. A compressed paragraph with many distinct named concepts may yield many anchors; a sparse segment may yield one or none.
  - Don't manufacture anchors to hit a target. Don't suppress real ones to stay terse.

Confidence — categories first, numbers as scaffold:
  - HIGH (~0.85+) = explicitly named, defined, or distinctively coined in the text
  - MEDIUM (~0.7)  = strongly implied recurring concept
  - LOW (~0.55)    = tentative — only include if you're confident it's a real anchor

Output ONLY valid JSON:
{
  "anchors": [
    {
      "canonical_phrase": "exact phrase as used (or canonical form when the text varies it)",
      "aliases": ["variant1", "variant2"],
      "justification": "what makes this an anchor",
      "confidence": 0.0-1.0
    }
  ]
}
"""


PASS3_SLAB_SYSTEM = """You are extracting SLABS — multi-sentence canonical descriptions of structured content — from a text segment for a knowledge corpus called Prime's Shadow.

Inputs vary widely: conversation transcripts, scientific papers, news articles, technical documentation, research notes, fiction, etc. Slabs work the same way across all of them.

A slab captures something that takes MORE THAN A NAME to express. The shape varies by genre:

  - Multi-step process — a chain (A → B → C → D), a methodology, an algorithm, a protocol, a recipe
  - Argument or claim — a hypothesis with reasoning, a thesis, a position with support, a finding
  - Result or outcome — an experimental result, an event with consequences, a state-change
  - Beat or arc moment — a narrative beat, a scene, a turning point, an event description
  - Principle or rule — a worldbuilding rule, a design principle, a thematic statement, a definition
  - Relationship or dynamic — a character dynamic, an interaction pattern, a structural relationship
  - Definition or explanation — a multi-sentence definition of a complex concept

The genre of the input doesn't change the rule: slabs are multi-sentence canonical content with internal structure.

Slab format:
  - title: short and specific (typically 3-7 words; use what fits the content)
  - canonical_text: complete sentences capturing the slab's content. Length follows the content — punchy beats may be 2-3 sentences; complex processes may be 8-10. Never truncate mid-thought.
  - references_anchors: anchor canonical_phrases from the supplied list that this slab involves. Match EXACTLY (case-insensitive ok).
  - new_anchors: any concepts the slab references that AREN'T in the supplied list. Use {phrase, aliases} format. Add only when the concept is clearly a named entity or distinctive term Pass 1 missed — not for things that already feel slab-shaped.

Rules:
  - Multi-step processes / methodologies / protocols get ONE slab for the whole chain, not separate slabs per step.
  - Beats, scenes, or events get ONE slab each.
  - A finding plus its supporting argument is usually ONE slab.
  - Don't slab plain entities — those are already anchors.

Density:
  - Match the segment's slab-worthy content. A dense paragraph may yield many slabs; a single-line vignette may yield one. A purely descriptive transition may yield none.

Confidence:
  - HIGH (~0.85+) = explicit in the text
  - MEDIUM (~0.7) = implied or reconstructed from context
  - LOW (~0.55)   = tentative

Output ONLY valid JSON:
{
  "slabs": [
    {
      "title": "...",
      "canonical_text": "...",
      "references_anchors": ["..."],
      "new_anchors": [{"canonical_phrase": "...", "aliases": [...]}],
      "confidence": 0.0-1.0,
      "justification": "..."
    }
  ]
}
"""


BUNDLE_LABEL_SYSTEM = """You are labelling thematic groups of related concepts for a knowledge corpus.

For each group below, produce ONE short label (typically 2-4 words; use what fits) that captures what the members have in common — a theme, a topic, a domain, a function, a relationship, or whatever connective tissue the group exhibits.

Output ONLY valid JSON:
{
  "labels": [
    { "group_id": 0, "label": "...", "justification": "..." },
    { "group_id": 1, "label": "...", "justification": "..." }
  ]
}
"""


# ─── Anchor canonicalization ─────────────────────────────────────────────


_MARKDOWN_EMPHASIS_RE = re.compile(r"[*_`]+")


def _strip_markup(s: str) -> str:
    """Remove markdown emphasis chars from a phrase. Pure cleanup of model
    output — `fusion *juice*` should land as canonical_phrase ``fusion juice``,
    not retain the asterisks.
    """
    return _MARKDOWN_EMPHASIS_RE.sub("", s).strip()


def _normalize_phrase(s: str) -> str:
    """Lowercase + strip non-alphanumerics for fuzzy match keying."""
    s = re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    return re.sub(r"\s+", " ", s)


# ─── Pass 1 — anchor-only extraction (parallel) ──────────────────────────


def _parse_pass1_response(resp, segment_order: int) -> list[MiningProposal]:
    """Convert Pass 1's JSON response into MiningProposal anchors.

    Resilient to the model emitting code fences or comment cruft —
    structured_extract already handles fence-stripping, but we still
    defensively parse non-dict shapes.
    """
    if isinstance(resp, dict):
        raw = resp.get("anchors", [])
    else:
        return []
    out: list[MiningProposal] = []
    for r in raw:
        if not isinstance(r, dict):
            continue
        phrase = _strip_markup((r.get("canonical_phrase") or "").strip())
        if not phrase:
            continue
        aliases = [
            _strip_markup(str(a).strip())
            for a in (r.get("aliases") or [])
        ]
        aliases = [a for a in aliases if a and _normalize_phrase(a) != _normalize_phrase(phrase)][:5]
        try:
            conf = float(r.get("confidence", 0.5))
        except Exception:
            conf = 0.5
        out.append(MiningProposal(
            proposal_type="anchor",
            canonical_phrase=phrase,
            aliases=aliases,
            source_topic="narrative",
            confidence=conf,
            source_pairs=[segment_order],
            justification=(r.get("justification") or "").strip(),
        ))
    return out


async def extract_anchors_pass1(segment: NarrativeSegment) -> list[MiningProposal]:
    """Pass 1: extract anchors only from a single segment.

    Genre-agnostic prompt + system/user split so the daemon's prompt
    cache reuses the (~600 token) system prefix across every parallel
    call in this pass. Fully independent — no inter-segment context.
    """
    user = f"Text segment:\n{segment.text}"
    try:
        resp = await ollama.structured_extract(user, system=PASS1_ANCHOR_SYSTEM)
    except Exception as exc:
        logger.warning(
            "Pass 1 (anchors) failed on segment %d (%s): %r",
            segment.order, type(exc).__name__, exc,
        )
        return []
    return _parse_pass1_response(resp, segment.order)


# ─── Pass 2 — bundle synthesis (Python clustering + batched LLM labels) ──


def _segment_membership_for_anchors(
    segments: list[NarrativeSegment], anchors: list[MiningProposal]
) -> dict[str, set[int]]:
    """For each anchor, find which segment indices its phrase or aliases
    appear in. Returns ``{anchor_canonical_phrase: {segment_order, ...}}``.
    Used by Pass 2's co-occurrence affinity term and by Pass 3's
    per-segment anchor scoping.
    """
    out: dict[str, set[int]] = {}
    # Pre-strip markdown from segment text so anchors that surface as
    # `*foo*` in the source still match.
    seg_haystacks = [
        (s.order, _MARKDOWN_EMPHASIS_RE.sub("", s.text).lower())
        for s in segments
    ]
    for a in anchors:
        forms = {_normalize_phrase(a.canonical_phrase)}
        for al in a.aliases:
            forms.add(_normalize_phrase(al))
        forms.discard("")
        if not forms:
            continue
        seg_set: set[int] = set()
        for seg_order, hay in seg_haystacks:
            # Cheap substring after normalization — true word-bounded
            # matching would be more accurate but this is the same
            # heuristic build_cooccurrence_edges has been using.
            norm_hay = re.sub(r"[^a-z0-9 ]+", " ", hay).strip()
            for f in forms:
                if f and f in norm_hay:
                    seg_set.add(seg_order)
                    break
        out[a.canonical_phrase] = seg_set
    return out


def _cosine(a: list[float], b: list[float]) -> float:
    """Plain float-list cosine — no numpy dependency for Pass 2."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _connected_components(edges: list[tuple[int, int]], n: int) -> list[list[int]]:
    """Union-find to extract connected components from an edge list.
    Returns list of node-index lists, sorted by size descending.
    """
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for a, b in edges:
        union(a, b)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=len, reverse=True)


async def synthesize_bundles(
    anchors: list[MiningProposal],
    segments: list[NarrativeSegment],
    *,
    cosine_weight: float = 0.4,
    cooccur_weight: float = 0.6,
    affinity_threshold: float = 0.35,
    min_bundle_size: int = 3,
) -> list[MiningProposal]:
    """Pass 2: cluster Pass 1 anchors into thematic bundles.

    Affinity = α·cosine(embedding) + β·jaccard(segment_membership). Two
    anchors that semantically resemble each other AND co-occur in the
    same segments form a strong edge; either signal alone is weaker.
    Connected components above the threshold form bundles; clusters of
    size < min_bundle_size are dropped (singletons / pairs aren't
    bundles, just noise).

    Cluster labels come from a single batched LLM call. Embedding fetches
    are batched into one ``ollama.embed()`` call regardless of corpus
    size. So the entire pass is dominated by ONE LLM round-trip plus a
    handful of millisecond-scale numpy-free Python.
    """
    if len(anchors) < min_bundle_size:
        return []

    phrases = [a.canonical_phrase for a in anchors]

    # Embeddings — one batched call for all phrases. Cheap (~50ms for
    # 50 phrases on nomic-embed-text on a warm GPU).
    try:
        embeddings = await ollama.embed(phrases)
    except Exception as exc:
        logger.warning(
            "Pass 2 (bundles) embedding failed: %s. Skipping bundles.",
            exc,
        )
        return []

    # Segment membership for jaccard. Re-uses Pass 1 anchors' source_pairs
    # plus a phrase-in-segment scan for better recall (an anchor may have
    # been extracted from segment 3 but ALSO appear in segments 1 and 5).
    membership = _segment_membership_for_anchors(segments, anchors)

    # Build affinity edge list above threshold. O(N²) — fine for N < 200.
    n = len(anchors)
    edges: list[tuple[int, int]] = []
    for i in range(n):
        ai_phrase = anchors[i].canonical_phrase
        ai_segs = membership.get(ai_phrase, set())
        for j in range(i + 1, n):
            aj_phrase = anchors[j].canonical_phrase
            aj_segs = membership.get(aj_phrase, set())
            cos = _cosine(embeddings[i], embeddings[j])
            cooc = _jaccard(ai_segs, aj_segs)
            aff = cosine_weight * cos + cooccur_weight * cooc
            if aff >= affinity_threshold:
                edges.append((i, j))

    if not edges:
        return []

    components = _connected_components(edges, n)
    clusters = [comp for comp in components if len(comp) >= min_bundle_size]
    if not clusters:
        return []

    # Generate bundle labels via one batched LLM call.
    user_lines = []
    for gid, comp in enumerate(clusters):
        members = [anchors[i].canonical_phrase for i in comp]
        user_lines.append(f"Group {gid}: {members}")
    user = "Groups:\n" + "\n".join(user_lines)

    labels: dict[int, str] = {}
    label_justifications: dict[int, str] = {}
    try:
        resp = await ollama.structured_extract(user, system=BUNDLE_LABEL_SYSTEM)
        if isinstance(resp, dict):
            for entry in resp.get("labels") or []:
                if not isinstance(entry, dict):
                    continue
                gid = entry.get("group_id")
                lbl = (entry.get("label") or "").strip()
                if isinstance(gid, int) and lbl:
                    labels[gid] = _strip_markup(lbl)
                    label_justifications[gid] = (entry.get("justification") or "").strip()
    except Exception as exc:
        logger.warning("Pass 2 (bundles) labelling failed: %s. Using fallback labels.", exc)

    bundles: list[MiningProposal] = []
    for gid, comp in enumerate(clusters):
        members = [anchors[i].canonical_phrase for i in comp]
        # Aggregate confidence: mean of member confidences, slightly
        # discounted because bundle membership is heuristic-derived.
        member_confs = [anchors[i].confidence for i in comp]
        conf = (sum(member_confs) / len(member_confs)) * 0.85 if member_confs else 0.6
        # Use the union of source segments so the bundle is anchored
        # to the spread of segments its members touch.
        source_pairs: set[int] = set()
        for i in comp:
            source_pairs.update(anchors[i].source_pairs or [])
        label = labels.get(gid) or f"Cluster {gid}"
        justification = label_justifications.get(gid) or (
            f"{len(comp)} anchors clustered by embedding+co-occurrence affinity"
        )
        bundles.append(MiningProposal(
            proposal_type="bundle",
            label=label,
            aliases=members,  # convo_miner convention: members live in aliases
            source_topic="narrative",
            confidence=conf,
            source_pairs=sorted(source_pairs),
            justification=justification,
        ))
    return bundles


# ─── Pass 3 — slab extraction with anchor context (parallel) ─────────────


def _format_anchor_list_for_pass3(anchors: list[MiningProposal]) -> str:
    if not anchors:
        return "(none)"
    lines = []
    for a in anchors:
        suffix = f"  (aliases: {', '.join(a.aliases)})" if a.aliases else ""
        lines.append(f"  - {a.canonical_phrase}{suffix}")
    return "\n".join(lines)


def _parse_pass3_response(resp, segment_order: int) -> tuple[list[MiningProposal], list[MiningProposal], list[EdgeProposal]]:
    """Parse Pass 3 JSON into (slabs, new_anchors, links_edges).

    The slab proposal stays in the existing MiningProposal shape (no
    schema change). The references_anchors list is converted into
    LINKS EdgeProposals, replacing the heuristic build_cooccurrence_edges
    output for this segment. new_anchors get materialised as MiningProposal
    anchors and reconciled into the canonical list by mine().
    """
    slabs: list[MiningProposal] = []
    new_anchors: list[MiningProposal] = []
    edges: list[EdgeProposal] = []

    if not isinstance(resp, dict):
        return slabs, new_anchors, edges
    raw_slabs = resp.get("slabs") or []
    if not isinstance(raw_slabs, list):
        return slabs, new_anchors, edges

    # enumerate gives intra_segment_order — when a single segment yields
    # multiple slabs (common for long un-headed passages), this is the
    # only signal that lets SEQUENCE-edge derivation recover narrative
    # order within the segment. Skipped/invalid slabs still increment
    # the index, which is fine: relative order of EMITTED slabs is all
    # the SEQUENCE pass needs.
    for slab_idx, s in enumerate(raw_slabs):
        if not isinstance(s, dict):
            continue
        title = (s.get("title") or "").strip()
        text = (s.get("canonical_text") or "").strip()
        if not text:
            continue
        try:
            conf = float(s.get("confidence", 0.5))
        except Exception:
            conf = 0.5

        slabs.append(MiningProposal(
            proposal_type="slab",
            canonical_text=text,
            title=title or f"Beat {segment_order}",
            source_topic="narrative",
            confidence=conf,
            source_pairs=[segment_order],
            intra_segment_order=slab_idx,
            justification=(s.get("justification") or "").strip(),
        ))

        # references_anchors → LINKS edges, deduped per-slab so a model
        # that lists the same anchor 3× doesn't yield 3 identical edges.
        slab_label = title or text[:40]
        seen_refs: set[str] = set()
        for ref in (s.get("references_anchors") or []):
            ref_str = _strip_markup(str(ref).strip())
            if not ref_str:
                continue
            key = _normalize_phrase(ref_str)
            if key in seen_refs:
                continue
            seen_refs.add(key)
            edges.append(EdgeProposal(
                edge_type="LINKS",
                from_label=slab_label,
                to_label=ref_str,
                confidence=conf,  # inherit slab confidence — the LLM said this slab references this anchor
                justification=f"Slab '{slab_label}' references anchor '{ref_str}' (Pass 3 structural link)",
            ))

        # new_anchors — the safety net for anchors Pass 1 missed.
        for na in (s.get("new_anchors") or []):
            if not isinstance(na, dict):
                continue
            phrase = _strip_markup((na.get("canonical_phrase") or "").strip())
            if not phrase:
                continue
            aliases = [
                _strip_markup(str(a).strip())
                for a in (na.get("aliases") or [])
            ]
            aliases = [a for a in aliases if a and _normalize_phrase(a) != _normalize_phrase(phrase)][:5]
            new_anchors.append(MiningProposal(
                proposal_type="anchor",
                canonical_phrase=phrase,
                aliases=aliases,
                source_topic="narrative",
                # Inherit slab confidence — the LLM's confidence in the
                # slab is the strongest signal we have for the anchor's
                # plausibility, since the anchor was extracted as part
                # of producing this slab.
                confidence=conf,
                source_pairs=[segment_order],
                justification=f"new_anchor from Pass 3 slab '{slab_label}' (Pass 1 miss)",
            ))

    return slabs, new_anchors, edges


async def extract_slabs_pass3(
    segment: NarrativeSegment,
    segment_anchors: list[MiningProposal],
) -> tuple[list[MiningProposal], list[MiningProposal], list[EdgeProposal]]:
    """Pass 3: extract slabs from a segment with the canonical anchor
    list (filtered to this segment) as context.

    Returns three lists: (slabs, new_anchors_safety_net, links_edges).
    """
    user = (
        f"Anchors already extracted from this segment "
        f"(use these in references_anchors when relevant):\n"
        f"{_format_anchor_list_for_pass3(segment_anchors)}\n\n"
        f"Text segment:\n{segment.text}"
    )
    try:
        resp = await ollama.structured_extract(user, system=PASS3_SLAB_SYSTEM)
    except Exception as exc:
        logger.warning(
            "Pass 3 (slabs) failed on segment %d (%s): %r",
            segment.order, type(exc).__name__, exc,
        )
        return [], [], []
    return _parse_pass3_response(resp, segment.order)


# ─── Miner class ─────────────────────────────────────────────────────────


class NarrativeMiner:
    """Mine cohesive narrative/document text for corpus proposals.

    Three-pass architecture (anchors → bundles → slabs) with intra-pass
    parallelism. Output dict shape matches ConversationMiner.mine() so
    /push-mined and the dreaming pass work unchanged.
    """

    def __init__(self, corpus):
        self.corpus = corpus

    async def mine(
        self,
        raw_text: str,
        source_label: str = "narrative",
        min_confidence: float = 0.4,
        max_segment_chars: int = 1800,
    ) -> dict:
        segments = segment_narrative(raw_text, max_chars=max_segment_chars)
        if not segments:
            return {
                "format": "narrative",
                "segments": 0,
                "proposals": [],
                "edges": [],
                "error": "No content found after segmentation",
            }

        sem = asyncio.Semaphore(_PASS_PARALLEL)

        # ── Pass 1 — anchors per segment, parallel ───────────────────
        async def _p1(seg: NarrativeSegment) -> list[MiningProposal]:
            async with sem:
                return await extract_anchors_pass1(seg)

        pass1_results = await asyncio.gather(*[_p1(s) for s in segments])
        all_anchors: list[MiningProposal] = []
        for sub in pass1_results:
            all_anchors.extend(sub)

        # Reconcile: same entity across segments should dedupe to one
        # canonical entry with the merged confidence (deduplicate_proposals
        # keeps the highest-confidence variant).
        canonical_anchors = deduplicate_proposals(all_anchors)
        canonical_anchors = [a for a in canonical_anchors if a.proposal_type == "anchor"]

        # ── Pass 2 — bundle synthesis (Python clustering + 1 LLM call) ─
        bundle_proposals = await synthesize_bundles(canonical_anchors, segments)

        # ── Pass 3 — slabs per segment with anchor context, parallel ───
        # Build the per-segment scoped anchor list once. Each segment's
        # Pass 3 prompt only sees anchors that surface in its text — this
        # mirrors the harness behaviour and keeps the slab prompt focused.
        membership = _segment_membership_for_anchors(segments, canonical_anchors)
        anchors_by_segment: dict[int, list[MiningProposal]] = {s.order: [] for s in segments}
        for anchor in canonical_anchors:
            for seg_idx in membership.get(anchor.canonical_phrase, set()):
                anchors_by_segment.setdefault(seg_idx, []).append(anchor)

        async def _p3(seg: NarrativeSegment):
            async with sem:
                return await extract_slabs_pass3(
                    seg, anchors_by_segment.get(seg.order, []),
                )

        pass3_results = await asyncio.gather(*[_p3(s) for s in segments])
        all_slabs: list[MiningProposal] = []
        all_new_anchors: list[MiningProposal] = []
        all_links_edges: list[EdgeProposal] = []
        for slabs, new_anchors, links in pass3_results:
            all_slabs.extend(slabs)
            all_new_anchors.extend(new_anchors)
            all_links_edges.extend(links)

        # Reconcile new_anchors back into the canonical list.
        if all_new_anchors:
            canonical_anchors = deduplicate_proposals(canonical_anchors + all_new_anchors)
            canonical_anchors = [a for a in canonical_anchors if a.proposal_type == "anchor"]

        # ── Combine + filter + sort ──────────────────────────────────
        all_proposals: list[MiningProposal] = (
            canonical_anchors + bundle_proposals + all_slabs
        )
        all_proposals = deduplicate_proposals(all_proposals)
        all_proposals = [p for p in all_proposals if p.confidence >= min_confidence]
        # Source order first (segment), confidence second, type third for
        # display stability when source_pairs ties.
        all_proposals.sort(key=lambda p: (
            p.source_pairs[0] if p.source_pairs else 999,
            -p.confidence,
            p.proposal_type,
        ))

        # ── SEQUENCE edges from slabs (existing helper, still useful) ─
        seq_edges = build_sequence_edges(all_proposals)
        seq_edges = [e for e in seq_edges if e.confidence >= min_confidence]

        # LINKS edges from Pass 3 (replaces the heuristic
        # build_cooccurrence_edges — Pass 3's structural references are
        # higher signal than phrase-match heuristics).
        links_edges = [e for e in all_links_edges if e.confidence >= min_confidence]

        all_edges = seq_edges + links_edges

        return {
            "format": "narrative",
            "segments": len(segments),
            "chunks_analyzed": len(segments),
            "topic_distribution": {"narrative": len(segments)},
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
        }

    async def mine_file(self, path: str | Path, **kwargs) -> dict:
        path = Path(path)
        if not path.exists():
            return {"error": f"File not found: {path}"}
        raw = path.read_text(encoding="utf-8")
        kwargs.setdefault("source_label", path.name)
        return await self.mine(raw, **kwargs)
