"""Pillar extractor — produces PillarDefinition proposals from a slab set.

The pillar overlay (WIP/corpus_graph_structure.md §6) is a curated tier
above the Anchor/Bundle/Slab content graph. Pillars answer "what is this
corpus about?" at top zoom, before any drill. They reference content
nodes via members/children, never store substantive content themselves.

This extractor builds pillars from scratch given a list of slabs. The
strategy is genre-adaptive but the steady state is:

  Stage 0 — genre + structural shape (1 LLM call)
            Asks the model what kind of document this is and whether
            it has explicit sectional structure (headings, numbered
            sections) versus emergent structure that must be inferred.
            For section-titled corpora (pod-like) the downstream work
            is mostly reading along; for paper-like content, real
            inference is required.
  Stage 1 — slab grouping into sub-pillars (1 LLM call)
            Given slab titles + brief excerpts, produce 5-15 named
            groups (the chapter-level tier). Each group gets a label,
            a 1-3 sentence summary, and a member list.
  Stage 2 — top-tier nesting (1 LLM call, conditional)
            If Stage 1 produced >8 sub-pillars, group them into 3-5
            top-tier pillars (the part-level tier). Skipped for
            small corpora where one tier is enough.

Output: a list of ProposedPillar records that map cleanly to
PillarDefinition at commit time. Caller decides whether to write them
as drafts for human review or commit directly.

See the design conversation in this PR for the pod-validation rationale
and the genre-conditional staging.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from . import ollama
from ..models.schemas import Slab

logger = logging.getLogger(__name__)

# Pillar grouping is a SYNTHESIS task with a very wide input (full slab
# title set in one prompt). Like directive Stage 2, a stronger model is
# worth it; allow override via env var with a sensible default.
_GROUP_MODEL = os.environ.get("PS_PILLAR_GROUP_MODEL") or None

# Parallelism for batched Stage 1.5. Matches Ollama's NUM_PARALLEL slot
# count; higher concurrency queues client-side without throughput gain.
_PARALLEL = int(os.environ.get("PS_PILLAR_PARALLEL", "2"))


# ─── Data shapes ─────────────────────────────────────────────────────────


@dataclass
class ProposedPillar:
    """One pillar produced by Stage 1 or 2. Mapped to PillarDefinition at commit."""
    id: str
    label: str
    summary: str
    members: list[str] = field(default_factory=list)              # slab IDs (Stage 1) or pillar IDs (Stage 2)
    children: list[str] = field(default_factory=list)             # populated only at top tier
    parent: Optional[str] = None
    pillar_role: str = "chapter"                                  # "section" at top tier, "chapter" or "topic" below
    origin: str = ""
    rationale: str = ""                                           # 1-2 sentences on why these slabs cohere


@dataclass
class PillarExtractionResult:
    hint: dict
    sub_pillars: list[ProposedPillar]
    top_pillars: list[ProposedPillar]                             # empty if single-tier


# ─── System prompts ──────────────────────────────────────────────────────


STAGE0_SHAPE_SYSTEM = """You are reading slab titles + excerpts to determine \
the STRUCTURAL SHAPE of a corpus, so a downstream pillar pass knows what \
kind of grouping work to do.

Given a sample of slab titles and brief excerpts, output:
  1. document_type: paper | proposal | spec | design_doc | transcript | \
narrative | reference | other
  2. domain: 1-3 word topic descriptor
  3. structural_shape: explicit_sections | semantic_clusters | flat | mixed
     - explicit_sections: titles reveal sectional structure (numbered, named).
       Grouping is mostly reading along.
     - semantic_clusters: titles don't reveal structure; groupings must be
       inferred from content similarity + cross-references.
     - flat: no meaningful sub-structure; the corpus is a single unit.
     - mixed: some explicit structure but with cross-section themes worth
       surfacing as separate pillars.
  4. target_pillar_count: an integer hint, 3-15, for how many sub-pillars
     this corpus probably wants. Long structured docs lean high; short
     focused ones lean low.
  5. shape_notes: 1-2 sentences on what the grouping pass should attend to.

Output ONLY valid JSON:
{
  "document_type": "...",
  "domain": "...",
  "structural_shape": "...",
  "target_pillar_count": <int>,
  "shape_notes": "..."
}
"""


STAGE1_GROUP_SYSTEM = """You are grouping slabs into PILLARS — a curated \
navigational tier that sits above the slabs themselves. Each pillar should \
read like a chapter or section heading of the underlying document.

A pillar has:
  - label: 2-6 word headline (think "Design Philosophy", not "Things about
    design")
  - summary: 1-3 sentences distilling what this pillar covers. Specific and
    substantive — a reader at top zoom should see this and know what they'd
    learn by drilling in.
  - members: slab IDs that belong to this pillar. Each slab should belong
    to EXACTLY ONE pillar (no overlap). Every supplied slab should be
    assigned somewhere.
  - rationale: 1-2 sentences on why these slabs cohere as one unit.

RULES
1. Use the structural_shape hint to calibrate aggressiveness:
   - explicit_sections → mostly read along, group by stated section
   - semantic_clusters → infer groupings from titles + content overlap
   - flat → probably skip; report a single "Whole Document" pillar
   - mixed → both, but flag which pillars are explicit vs inferred
2. Target the corpus's target_pillar_count. Slightly fewer is fine; many
   more is a sign you're over-fragmenting.
3. Every supplied slab MUST be assigned to exactly one pillar. If a slab
   doesn't fit anywhere clean, assign it to the closest pillar with a note
   in that pillar's rationale.
4. Labels must be DISCRIMINATING. "Various Topics", "Other", "Miscellaneous"
   are anti-patterns; if you can't name a pillar specifically, the cluster
   isn't real.
5. Summaries describe CONTENT, not structure. "This pillar contains 12
   slabs" is wrong. "Covers session lifecycle from idle baseline through
   active use to mandatory reset" is right.
6. pillar_role: free-text describing what kind of unit this is in the
   source. Common values: "section", "chapter", "act", "topic", "claim".

Output ONLY valid JSON:
{
  "pillars": [
    {
      "label": "...",
      "summary": "...",
      "members": ["slab_id_1", "slab_id_2", ...],
      "pillar_role": "...",
      "rationale": "..."
    }
  ]
}
"""


STAGE1_5_COVERAGE_SYSTEM = """You are filling COVERAGE GAPS in a pillar \
assignment. A prior pass grouped most slabs into named pillars but dropped \
some — your job is to assign each orphan to its best-fit existing pillar, \
or mark it as genuinely unclusterable if no pillar fits.

INPUT
  pillars: list of {id, label, summary, current_member_count}
  orphans: list of {slab_id, title, excerpt}

YOUR TASK
For each orphan, decide one of:
  - "assigned_to: <pillar_id>" — best-fit pillar from the list
  - "assigned_to: null" — genuinely unclusterable; document has no
    pillar for this content

RULES
1. Prefer assignment over null. Most orphans were dropped accidentally,
   not because they're genuinely unclusterable.
2. Read each orphan's title and excerpt carefully. Pillar labels alone
   are not enough — the summary describes what the pillar covers, use it.
3. If an orphan COULD fit two pillars, pick the more specific one. A
   slab about "Water flow safety" goes under "Water Systems" not under
   "Operational Procedures & Safety", even though safety is mentioned.
4. null should be rare — reserved for truly orthogonal content (e.g.,
   a footnote, an off-topic aside, content the source document itself
   treated as separable from the main argument).
5. You do NOT need to rename or restructure pillars. Just assign.

CRITICAL: Use the slab_id values verbatim. Do not paraphrase or shorten
them. Do not add prefixes. Copy them character-for-character.

Output ONLY valid JSON:
{
  "assignments": [
    {
      "slab_id": "<verbatim id from input>",
      "assigned_to": "<pillar_id>" | null,
      "rationale": "<one short sentence>"
    }
  ]
}

Every orphan in the input must appear in the output exactly once.
"""


STAGE2_NEST_SYSTEM = """You are nesting a flat list of sub-pillars into a \
top tier — the part-level grouping above chapter-level.

A top pillar:
  - has 2-5 sub-pillars as children
  - has a label that captures the part-level theme (not just a list of its
    children)
  - has a summary that reads like a part introduction in a structured
    document — what binds these chapters together?

RULES
1. Aim for 3-5 top pillars. Fewer if the corpus is small.
2. Every sub-pillar should belong to exactly one top pillar.
3. Top pillars should be MEANINGFUL groupings, not arbitrary partitions.
   If you can't name a top pillar specifically, don't create it — better
   to leave the sub-pillars at the top tier.
4. Order matters: top pillars should be returned in the order a reader
   would encounter them in a structured outline.

Output ONLY valid JSON:
{
  "top_pillars": [
    {
      "label": "...",
      "summary": "...",
      "children": ["pillar_slug_1", "pillar_slug_2", ...],
      "pillar_role": "section",
      "rationale": "..."
    }
  ]
}

Use the slugified sub-pillar labels (lowercase, underscores) as children
references — they will be resolved to IDs by the caller.
"""


# ─── Helpers ─────────────────────────────────────────────────────────────


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(s: str, max_len: int = 40) -> str:
    return _SLUG_RE.sub("_", s.lower()).strip("_")[:max_len] or "untitled"


def _format_slab_for_grouping(slab: Slab, excerpt_chars: int = 80) -> str:
    """Format a slab line for grouping prompts.

    Uses ``id=...`` key/value style rather than ``[id]`` bracket form
    because the latter triggered an LLM transcription bug where slab
    IDs came back with duplicated prefixes (the model treated brackets
    as a placeholder slot to "complete"). Key/value reads as data, not
    template.

    Default 80 chars — calibrated for reliability on local 12B models
    at 300-slab corpus sizes. 140 chars produces 1-2 more sub-pillars
    in good runs but flakes out at the 300s httpx timeout on warm-cache
    runs. 220 chars (old default) almost always timed out. Granularity
    is a tunable: longer excerpts via a stronger model (cloud) would
    give richer output. For now, prefer reliable 5-pillar coarse
    groupings + heavy Stage 1.5 coverage lift over flaky 9-pillar runs.
    """
    excerpt = (slab.canonical_text or "").replace("\n", " ")[:excerpt_chars]
    # Avoid repr()/quoting — it adds 20-40 chars per line of overhead
    # that we don't need since title/excerpt are short. Pipe separator
    # is unambiguous enough in practice.
    return f"- id={slab.id} | title: {slab.title} | text: {excerpt}"


# Defense-in-depth ID repair: even with the unambiguous prompt format
# the LLM occasionally munges IDs. The two patterns we've seen are
# duplicated prefixes and stray brackets. Repair before validation so
# borderline-typo IDs survive instead of getting silently dropped.
_BRACKETS_RE = re.compile(r"^\[|\]$")
_DOUBLE_PREFIX_RE = re.compile(r"^(mined_(?:slab|anchor|bundle))_(mined_(?:slab|anchor|bundle))_")


def _repair_member_id(s: str) -> str:
    s = _BRACKETS_RE.sub("", s).strip()
    # Collapse "mined_slab_mined_slab_xxx_v1" → "mined_slab_xxx_v1"
    s = _DOUBLE_PREFIX_RE.sub(r"\1_", s)
    return s


# ─── Stage 0 — structural shape detection ────────────────────────────────


async def detect_shape(slabs: list[Slab], sample_size: int = 30) -> dict:
    """Detect document type + structural shape + target pillar count.

    Samples up to ``sample_size`` slabs evenly. For section-titled
    corpora (pod-like) this call effectively tells downstream Stage 1
    "you're reading an outline; group by it." For paper-like content
    it tells Stage 1 to do real inference.
    """
    if not slabs:
        return {
            "document_type": "empty", "domain": "empty",
            "structural_shape": "flat", "target_pillar_count": 0,
            "shape_notes": "No slabs supplied.",
        }

    sample = slabs
    if len(slabs) > sample_size:
        stride = max(1, len(slabs) // sample_size)
        sample = slabs[::stride][:sample_size]

    lines = [_format_slab_for_grouping(s, excerpt_chars=160) for s in sample]
    user = (
        f"SLAB SAMPLE ({len(sample)} of {len(slabs)} total, title + excerpt):\n\n"
        + "\n".join(lines)
        + "\n\nDetect the corpus shape."
    )
    try:
        resp = await ollama.structured_extract(user, system=STAGE0_SHAPE_SYSTEM)
    except Exception as exc:
        logger.warning("Pillar Stage 0 shape detection failed: %r", exc)
        return {
            "document_type": "unknown", "domain": "unknown",
            "structural_shape": "semantic_clusters",
            "target_pillar_count": min(10, max(3, len(slabs) // 30)),
            "shape_notes": "Shape detection failed; using heuristic defaults.",
        }

    return {
        "document_type": resp.get("document_type", "unknown"),
        "domain": resp.get("domain", "unknown"),
        "structural_shape": resp.get("structural_shape", "semantic_clusters"),
        "target_pillar_count": int(resp.get("target_pillar_count") or
                                   max(3, min(15, len(slabs) // 30))),
        "shape_notes": resp.get("shape_notes", ""),
    }


# ─── Stage 1 — slab grouping into sub-pillars ────────────────────────────


async def group_slabs(
    slabs: list[Slab], shape: dict, origin: str = "",
) -> list[ProposedPillar]:
    """Group slabs into named pillars. One LLM call.

    The whole slab list (titles + excerpts) goes into the prompt. For
    large corpora this can exceed default num_ctx; we bump the context
    window for this call to handle ~500+ slabs comfortably.
    """
    if not slabs:
        return []

    lines = [_format_slab_for_grouping(s) for s in slabs]
    shape_block = (
        f"CORPUS SHAPE:\n"
        f"  document_type: {shape.get('document_type', 'unknown')}\n"
        f"  domain: {shape.get('domain', 'unknown')}\n"
        f"  structural_shape: {shape.get('structural_shape', 'unknown')}\n"
        f"  target_pillar_count: {shape.get('target_pillar_count', 8)}\n"
        f"  shape_notes: {shape.get('shape_notes', '')}\n\n"
    )
    user = (
        shape_block
        + f"SLABS TO GROUP ({len(slabs)} total):\n\n"
        + "\n".join(lines)
        + "\n\nGroup these slabs into pillars per the rules."
    )

    # Stage 1 can run wide — bump num_ctx for big corpora.
    # ~280 chars per slab line × 297 slabs ≈ 25k tokens of input. 32768
    # gives output headroom. If the corpus exceeds ~500 slabs this
    # needs batched grouping (split into chapter-ranges, then merge).
    num_ctx = 32768
    try:
        resp = await ollama.structured_extract(
            user, system=STAGE1_GROUP_SYSTEM,
            num_ctx=num_ctx, model=_GROUP_MODEL,
        )
    except Exception as exc:
        logger.warning("Pillar Stage 1 grouping failed: %r", exc)
        return []

    # Filter LLM output members against the actual slab set — empirically
    # ~1-2% of returned IDs are munged (double prefixes, truncations,
    # transcription typos). Dropping them at this boundary keeps downstream
    # corpus validation happy without surfacing the failures to humans.
    valid_slab_ids = {s.id for s in slabs}

    out: list[ProposedPillar] = []
    seen_ids: set[str] = set()
    dropped_unknown = 0
    for p in resp.get("pillars", []) or []:
        label = (p.get("label") or "").strip()
        summary = (p.get("summary") or "").strip()
        raw_members = [_repair_member_id(m) for m in (p.get("members") or []) if m and m.strip()]
        members = [m for m in raw_members if m in valid_slab_ids]
        dropped_unknown += (len(raw_members) - len(members))
        if not label or not members:
            continue
        slug = _slugify(label)
        pid = f"pillar_{slug}_v1"
        n = 2
        while pid in seen_ids:
            pid = f"pillar_{slug}_{n}_v1"
            n += 1
        seen_ids.add(pid)
        out.append(ProposedPillar(
            id=pid,
            label=label,
            summary=summary,
            members=members,
            pillar_role=(p.get("pillar_role") or "chapter").strip(),
            origin=origin,
            rationale=(p.get("rationale") or "").strip(),
        ))
    if dropped_unknown:
        logger.info(
            "pillar_extractor: dropped %d member IDs not in input slab set "
            "(LLM transcription errors)", dropped_unknown,
        )
    return out


# ─── Stage 1.5 — coverage pass over orphan slabs ─────────────────────────


async def _assign_orphan_batch(
    batch: list[Slab],
    sub_pillars: list[ProposedPillar],
    shape: dict,
    batch_idx: int,
    batch_total: int,
) -> dict:
    """One LLM call for a batch of orphans. Returns the raw response dict
    (or empty dict on failure)."""
    pillar_lines = [
        f"  - id={sp.id} label={sp.label!r} current_members={len(sp.members)} "
        f"summary={sp.summary[:200]!r}"
        for sp in sub_pillars
    ]
    orphan_lines = [_format_slab_for_grouping(s, excerpt_chars=140) for s in batch]
    user = (
        f"PILLARS ({len(sub_pillars)} total):\n"
        + "\n".join(pillar_lines)
        + f"\n\nORPHANS TO ASSIGN — batch {batch_idx + 1}/{batch_total} "
        + f"({len(batch)} of total orphans):\n"
        + "\n".join(orphan_lines)
        + "\n\nAssign each orphan per the rules."
    )
    logger.info("Stage 1.5 batch %d/%d (%d orphans)...",
                batch_idx + 1, batch_total, len(batch))
    try:
        return await ollama.structured_extract(
            user, system=STAGE1_5_COVERAGE_SYSTEM,
            num_ctx=16384, model=_GROUP_MODEL,
        )
    except Exception as exc:
        logger.warning(
            "Stage 1.5 batch %d/%d failed: %r", batch_idx + 1, batch_total, exc,
        )
        return {}


async def assign_orphans(
    slabs: list[Slab],
    sub_pillars: list[ProposedPillar],
    shape: dict,
    batch_size: int = 60,
) -> list[ProposedPillar]:
    """Re-assign orphan slabs to their best-fit pillar.

    After Stage 1 the model often drops 30-70% of slabs unassigned —
    not because they're unclusterable but because the model optimised
    for cluster coherence over coverage. This pass walks the orphans
    explicitly, with the existing pillars as targets, and asks for
    "best fit or null." Empirically lifts coverage from ~30% to ~80-90%.

    Batched: with >300 slabs, a single Stage 1.5 call exceeds the
    300s httpx timeout on local 12B models. Splitting into chunks of
    ~60 orphans keeps each batch's prompt under 25KB and well within
    timeout. Each batch independently sees the full pillar list (small)
    plus its own orphan slice — assignment decisions don't need
    cross-batch context.

    Mutates the supplied sub_pillars in place by appending newly-assigned
    members. Returns the same list for caller convenience.
    """
    if not sub_pillars or not slabs:
        return sub_pillars

    assigned_ids: set[str] = set()
    for sp in sub_pillars:
        assigned_ids.update(sp.members)
    orphan_slabs = [s for s in slabs if s.id not in assigned_ids]
    if not orphan_slabs:
        return sub_pillars

    logger.info(
        "Stage 1.5: %d orphan slabs to re-assign across %d pillars",
        len(orphan_slabs), len(sub_pillars),
    )

    # Split orphans into ordered batches
    batches = [
        orphan_slabs[i:i + batch_size]
        for i in range(0, len(orphan_slabs), batch_size)
    ]
    sem = asyncio.Semaphore(_PARALLEL)

    async def _run(idx: int, batch: list[Slab]) -> dict:
        async with sem:
            return await _assign_orphan_batch(batch, sub_pillars, shape, idx, len(batches))

    batch_responses = await asyncio.gather(*[_run(i, b) for i, b in enumerate(batches)])

    valid_ids = {s.id for s in orphan_slabs}
    pillar_by_id = {sp.id: sp for sp in sub_pillars}
    added = 0
    nullified = 0
    dropped_munged = 0
    for resp in batch_responses:
        for a in resp.get("assignments", []) or []:
            sid = _repair_member_id((a.get("slab_id") or "").strip())
            if sid not in valid_ids:
                dropped_munged += 1
                continue
            target = a.get("assigned_to")
            if target is None or target == "null":
                nullified += 1
                continue
            target = target.strip()
            sp = pillar_by_id.get(target)
            if not sp:
                dropped_munged += 1
                continue
            if sid in sp.members:
                continue
            sp.members.append(sid)
            added += 1

    logger.info(
        "Stage 1.5 done: %d orphans assigned, %d marked null, %d dropped (bad id/target)",
        added, nullified, dropped_munged,
    )
    return sub_pillars


# ─── Stage 2 — top-tier nesting ──────────────────────────────────────────


async def nest_pillars(
    sub_pillars: list[ProposedPillar], shape: dict, origin: str = "",
) -> list[ProposedPillar]:
    """Group sub-pillars into top-tier pillars. Skipped if <8 sub-pillars."""
    if len(sub_pillars) < 8:
        return []

    pillar_lines = [
        f"- [{_slugify(p.label)}] {p.label}: {p.summary}"
        for p in sub_pillars
    ]
    shape_block = (
        f"CORPUS SHAPE:\n"
        f"  document_type: {shape.get('document_type', 'unknown')}\n"
        f"  domain: {shape.get('domain', 'unknown')}\n"
        f"  shape_notes: {shape.get('shape_notes', '')}\n\n"
    )
    user = (
        shape_block
        + f"SUB-PILLARS TO NEST ({len(sub_pillars)} total):\n\n"
        + "\n".join(pillar_lines)
        + "\n\nNest these sub-pillars into 3-5 top-tier pillars per the rules."
    )
    try:
        resp = await ollama.structured_extract(user, system=STAGE2_NEST_SYSTEM)
    except Exception as exc:
        logger.warning("Pillar Stage 2 nesting failed: %r", exc)
        return []

    slug_to_id = {_slugify(p.label): p.id for p in sub_pillars}
    out: list[ProposedPillar] = []
    seen_ids: set[str] = set()
    for tp in resp.get("top_pillars", []) or []:
        label = (tp.get("label") or "").strip()
        summary = (tp.get("summary") or "").strip()
        child_slugs = [s.strip() for s in (tp.get("children") or []) if s and s.strip()]
        if not label or not child_slugs:
            continue
        # Resolve child slugs to pillar IDs; drop any that didn't resolve.
        resolved_children = []
        for slug in child_slugs:
            normalized = _slugify(slug)
            child_id = slug_to_id.get(normalized)
            if child_id:
                resolved_children.append(child_id)
            else:
                logger.info("nest_pillars: child slug %r did not resolve", slug)
        if not resolved_children:
            continue

        slug = _slugify(label)
        tpid = f"pillar_{slug}_v1"
        n = 2
        while tpid in seen_ids:
            tpid = f"pillar_{slug}_{n}_v1"
            n += 1
        seen_ids.add(tpid)
        out.append(ProposedPillar(
            id=tpid,
            label=label,
            summary=summary,
            children=resolved_children,
            pillar_role=(tp.get("pillar_role") or "section").strip(),
            origin=origin,
            rationale=(tp.get("rationale") or "").strip(),
        ))
    # Stamp parent back-pointer on sub-pillars whose ID appears as a child.
    parent_map: dict[str, str] = {}
    for tp in out:
        for cid in tp.children:
            parent_map[cid] = tp.id
    for sp in sub_pillars:
        if sp.id in parent_map:
            sp.parent = parent_map[sp.id]
    return out


# ─── Public entry point ──────────────────────────────────────────────────


async def run(slabs: list[Slab], origin: str = "") -> PillarExtractionResult:
    """End-to-end Stage 0 → 1 → (2) over a slab list.

    ``origin`` is stamped on every proposed pillar for provenance (e.g.
    the source filename or ingestion event id).
    """
    if not slabs:
        return PillarExtractionResult(
            hint={"document_type": "empty", "domain": "empty",
                  "structural_shape": "flat", "target_pillar_count": 0,
                  "shape_notes": ""},
            sub_pillars=[], top_pillars=[],
        )

    logger.info("pillar_extractor: %d slabs, origin=%r", len(slabs), origin)
    shape = await detect_shape(slabs)
    logger.info(
        "pillar_extractor: shape = type=%s domain=%s structural=%s target=%d",
        shape.get("document_type"), shape.get("domain"),
        shape.get("structural_shape"), shape.get("target_pillar_count"),
    )
    sub_pillars = await group_slabs(slabs, shape, origin=origin)
    logger.info("pillar_extractor: Stage 1 produced %d sub-pillars", len(sub_pillars))
    # Stage 1.5 — coverage pass on dropped slabs. Lift coverage from ~30%
    # to ~80-90% by walking orphans explicitly. Skipped when group_slabs
    # produced no pillars (caller will see the empty result anyway).
    if sub_pillars:
        sub_pillars = await assign_orphans(slabs, sub_pillars, shape)
        covered = len({m for sp in sub_pillars for m in sp.members})
        logger.info(
            "pillar_extractor: post-1.5 coverage %d / %d slabs (%.0f%%)",
            covered, len(slabs), 100.0 * covered / max(1, len(slabs)),
        )
    top_pillars = await nest_pillars(sub_pillars, shape, origin=origin)
    logger.info("pillar_extractor: produced %d top-pillars", len(top_pillars))
    return PillarExtractionResult(
        hint=shape, sub_pillars=sub_pillars, top_pillars=top_pillars,
    )


# ─── CLI runner ──────────────────────────────────────────────────────────


def _dump_result(result: PillarExtractionResult) -> str:
    out = []
    out.append("=" * 70)
    out.append("SHAPE")
    out.append(f"  type:    {result.hint.get('document_type')}")
    out.append(f"  domain:  {result.hint.get('domain')}")
    out.append(f"  shape:   {result.hint.get('structural_shape')}")
    out.append(f"  target:  {result.hint.get('target_pillar_count')}")
    out.append(f"  notes:   {result.hint.get('shape_notes')}")
    out.append("=" * 70)
    if result.top_pillars:
        out.append(f"TOP TIER: {len(result.top_pillars)} pillars")
        for tp in result.top_pillars:
            out.append(f"  ▼ {tp.label} ({tp.id})")
            out.append(f"    {tp.summary}")
            out.append(f"    children: {tp.children}")
            out.append("")
    out.append("=" * 70)
    out.append(f"SUB-PILLARS: {len(result.sub_pillars)}")
    for sp in result.sub_pillars:
        parent = f" parent={sp.parent}" if sp.parent else ""
        out.append(f"  ▸ {sp.label} ({sp.id}){parent}")
        out.append(f"    {sp.summary}")
        out.append(f"    members: {len(sp.members)} slabs")
        if sp.rationale:
            out.append(f"    rationale: {sp.rationale}")
        out.append("")
    return "\n".join(out)


async def _cli(collection_id: str, slab_limit: Optional[int] = None) -> None:
    from .corpus import CorpusStore, CORPORA_ROOT
    store = CorpusStore(
        root=CORPORA_ROOT / collection_id, collection_id=collection_id,
    )
    errors = store.load()
    if errors:
        logger.error("Corpus load errors: %s", errors)
        return
    slabs = list(store.slabs.values())
    if slab_limit:
        slabs = slabs[:slab_limit]
    print(f"Loaded {len(slabs)} slabs from collection {collection_id!r}")
    result = await run(slabs, origin=f"cli:{collection_id}")
    print(_dump_result(result))
    json_path = CORPORA_ROOT / collection_id / f"pillar_extract_{slab_limit or 'all'}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "hint": result.hint,
            "sub_pillars": [vars(p) for p in result.sub_pillars],
            "top_pillars": [vars(p) for p in result.top_pillars],
        }, f, indent=2)
    print(f"\nWrote {json_path}")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cid = sys.argv[1] if len(sys.argv) > 1 else "podv3"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    asyncio.run(_cli(cid, limit))
