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

import json
import logging
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
                justification=rp.get("justification", ""),
            ))
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

    Walks proposals in (segment_order, list_order) and connects each slab to
    the next slab it sees. Edge weight is higher if the two slabs share an
    anchor/entity (continuity signal), lower otherwise (pure ordering).
    """
    slabs = [p for p in proposals if p.proposal_type == "slab"]
    slabs.sort(key=lambda p: (p.source_pairs[0] if p.source_pairs else 0,))
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
        edges.append(EdgeProposal(
            edge_type="SEQUENCE",
            from_label=a_label,
            to_label=b_label,
            confidence=weight,
            justification=(
                f"Narrative ordering: beat {a.source_pairs[0]} → beat {b.source_pairs[0]}"
                + (f" (shared entities: {shared})" if shared else "")
            ),
        ))
    return edges


# ─── Miner class ─────────────────────────────────────────────────────────


class NarrativeMiner:
    """Mine cohesive narrative/document text for corpus proposals.

    Output dict shape matches ConversationMiner.mine() so /push-mined and the
    dreaming pass work unchanged.
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

        # Existing anchors in target corpus (for prompt continuity)
        existing_phrases = [a.canonical_phrase for a in self.corpus.anchors.values()]

        all_proposals: list[MiningProposal] = []
        seen_slab_titles: list[str] = []
        seen_anchor_phrases: list[str] = list(existing_phrases)

        for seg in segments:
            seg_proposals = await extract_from_segment(
                seg,
                prior_anchor_phrases=seen_anchor_phrases,
                prior_slab_titles=seen_slab_titles,
            )
            for p in seg_proposals:
                all_proposals.append(p)
                if p.proposal_type == "anchor" and p.canonical_phrase:
                    seen_anchor_phrases.append(p.canonical_phrase)
                elif p.proposal_type == "slab" and p.title:
                    seen_slab_titles.append(p.title)

        # Deduplicate (shared helper)
        all_proposals = deduplicate_proposals(all_proposals)
        all_proposals = [p for p in all_proposals if p.confidence >= min_confidence]

        # Keep source order (by segment), then confidence as tiebreaker
        all_proposals.sort(key=lambda p: (p.source_pairs[0] if p.source_pairs else 999, -p.confidence))

        # SEQUENCE edges — narrative miner's signature product
        seq_edges = build_sequence_edges(all_proposals)
        seq_edges = [e for e in seq_edges if e.confidence >= min_confidence]

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
                for e in seq_edges
            ],
            "proposal_count": len(all_proposals),
            "edge_count": len(seq_edges),
            "source_label": source_label,
        }

    async def mine_file(self, path: str | Path, **kwargs) -> dict:
        path = Path(path)
        if not path.exists():
            return {"error": f"File not found: {path}"}
        raw = path.read_text(encoding="utf-8")
        kwargs.setdefault("source_label", path.name)
        return await self.mine(raw, **kwargs)
