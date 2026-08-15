"""Directive-packet bundle extractor — replacement for Pass-2 cosine clustering.

The legacy Pass-2 bundle miner produced topical clusters by embedding slabs and
grouping similar ones. The output was schema-misshapen: BundlePayload's slots
(non_assumptions, warnings, heuristics, rules, ...) were left empty because
cosine similarity is the wrong tool for finding directive content. Empirically
this produced 9000+ lines of bundle YAML with only intent + invariants
populated (see podv3).

This extractor produces bundles that match the schema's actual design intent:
coordinated framing constraints (directive packets) lifted from the source
text. Cross-slab by construction. Quality-gated to refuse bundles that are
just topical clusters wearing a directive coat.

Three stages:
  Stage 0 — corpus-level genre hint (1 LLM call)
            Primes Stages 1+2 with what kind of document this is, so the
            descriptive-vs-directive call has a global prior.
  Stage 1 — per-slab directive sentence extraction (N parallel calls)
            NER-shaped: given a slab, point at sentences with directive form.
            Verbatim quote required. Empty output is encouraged.
  Stage 2 — cross-slab bundle assembly (1 LLM call over all extractions)
            Relation-extraction-shaped: which directive sentences belong to
            the same packet? Quality gate enforced in the prompt itself.

Output: list of ProposedBundle. Caller decides whether to write them as
KeyBundle drafts for human review or commit directly.

See WIP/corpus_graph_structure.md §6 and the design conversation that
preceded this file for rationale.
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

# Match the daemon's OLLAMA_NUM_PARALLEL. See narrative_miner for the
# rationale on this default.
_PARALLEL = int(os.environ.get("PS_DIRECTIVE_PARALLEL", "2"))

# Stage 2 is a SYNTHESIS task with a large input — for a 300-slab
# corpus the assembly prompt easily exceeds 15k tokens. Local 12B
# models grind on this (300s timeout territory) and the synthesis
# quality matters more than throughput. Default to a stronger model
# for Stage 2 specifically; fall back to the default if unset.
# Tag ending in "-cloud" routes through the local Ollama daemon's
# cloud proxy.
_STAGE2_MODEL = os.environ.get("PS_DIRECTIVE_STAGE2_MODEL") or None


# ─── Data shapes ─────────────────────────────────────────────────────────


@dataclass
class DirectiveExtraction:
    """One directive sentence pointed at by Stage 1."""
    slab_id: str
    category: str   # intent / invariant / non_assumption / warning / heuristic / rule / marker
    quote: str
    paraphrase: str = ""
    confidence: float = 0.5


@dataclass
class ProposedBundle:
    """Stage 2 output. Mapped to KeyBundle / BundlePayload at commit time.

    ``depends_on`` is the slabs that contributed extractions (provenance).
    ``supports`` is the slabs the bundle frames; usually depends_on plus
    any wider scope the model judged the bundle covers.
    """
    id: str
    label: str
    intent: list[str]
    invariants: list[str] = field(default_factory=list)
    non_assumptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    heuristics: list[str] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)
    canonical_quote_handles: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    supports: list[str] = field(default_factory=list)
    rationale: str = ""


# ─── System prompts ──────────────────────────────────────────────────────


STAGE0_GENRE_HINT_SYSTEM = """You are reading a small sample of a corpus and \
producing a short calibration hint for downstream extraction passes.

Given a list of slab titles and brief excerpts, produce:
  1. document_type: paper | proposal | spec | design_doc | transcript | \
narrative | reference | other
  2. domain: 1-3 word topic descriptor
  3. directive_shape_guidance: 1-2 sentences describing what KIND of \
directive content is likely to appear in this corpus — what rules look \
like, what counts as a warning, what a heuristic might be. Be specific to \
the genre. A research paper's directive content looks different from a \
design document's.

Output ONLY valid JSON:
{
  "document_type": "...",
  "domain": "...",
  "directive_shape_guidance": "..."
}
"""


STAGE1_EXTRACT_SYSTEM = """You are extracting DIRECTIVE CONTENT from one slab \
of a corpus. A directive sentence PRESCRIBES something rather than \
describing it: a rule, a constraint, a stated assumption, a warning, a \
heuristic, an intent.

Categories — assign each extracted sentence to ONE of:

  intent          — statement of purpose, goal, what something is for
                    Markers: "the goal is", "the intent is", "purpose of",
                    "designed to", "exists to", "this section defines"

  invariant       — something stated as required to hold; necessary condition
                    Markers: "must", "is required to", "always", "is essential",
                    "non-negotiable", "fundamental", deontic claim about state

  non_assumption  — something the document explicitly DOES NOT claim or require
                    Markers: "does not assume", "is not claimed", "no claim that",
                    "not assumed", "explicitly excludes", "is not a", negation
                    of an expected position

  warning         — stated failure mode, pitfall, or risk to avoid
                    Markers: "fails when", "breaks down if", "the danger is",
                    "watch out for", "incorrect because", "common pitfall",
                    "the failure mode is"

  heuristic       — soft rule of thumb, preference, or guideline
                    Markers: "rule of thumb", "when in doubt", "as a guide",
                    "prefer X over Y", "it is generally better to", "tends to",
                    "typically"

  rule            — hard prescriptive statement: must / must never / cannot
                    Markers: "must not", "may not", "is forbidden", "never",
                    "cannot", "is not allowed", "is prohibited"

  marker          — a phrase the document uses to FLAG directive content nearby
                    Examples: "non-negotiable", "first-class", "hard limit not
                    recommendation", "load-bearing", "core requirement"

CRITICAL RULES
1. Each quote MUST appear verbatim or near-verbatim in the slab text.
   No invented content. No paraphrased reframing.
2. Descriptive sentences are NOT directives. The test: does the sentence
   describe what something IS (descriptive) or what something MUST BE
   (directive)? Deontic modals (must, should, cannot, never, always in the
   prescriptive sense) are the discriminator.
   - "The pool is 3.0 m long" → descriptive, do not extract
   - "The pool MUST be at minimum 3.0 m long" → invariant
3. If the slab contains no directive content, return empty extractions.
   Many slabs are pure description. DO NOT INVENT directives to fill output.
4. A sentence may match more than one category — return all that fit, but
   only when both genuinely apply. If unsure, pick the strongest single match.
5. Confidence calibration:
   - HIGH (~0.85+) — clearly directive, explicit deontic modal
   - MEDIUM (~0.7) — directive with mild interpretive judgment
   - LOW (~0.5-0.6) — borderline; log as 'marker' if unsure between
     descriptive/directive

Output ONLY valid JSON:
{
  "slab_id": "<echoed from input>",
  "extractions": [
    {
      "category": "<one of the seven categories>",
      "quote": "<verbatim or near-verbatim sentence from slab text>",
      "paraphrase": "<optional 1-line cleaner version>",
      "confidence": 0.0-1.0
    }
  ]
}

If no directives present: {"slab_id": "<id>", "extractions": []}
"""


STAGE2_ASSEMBLE_SYSTEM = """You are assembling DIRECTIVE PACKETS ("bundles") \
from extracted directive sentences across a corpus. Each bundle is a \
COORDINATED set of framing constraints that spans multiple slabs and reads \
as one coherent unit.

A bundle has:
  - a coherent theme expressible in 2-6 words (the label)
  - members across MULTIPLE slabs (≥2; usually 3-8)
  - multiple payload categories populated
  - a rationale you can articulate in 1-3 sentences

QUALITY GATE — a bundle that does not meet ALL of these MUST NOT be proposed:
  1. ≥2 distinct slabs contributed extractions
  2. ≥1 of {warnings, non_assumptions, rules, heuristics} is non-empty.
     This prevents the legacy failure mode of "every slab spawns a near-empty
     bundle of intent + invariants only" — bundles with no constraint-bearing
     categories are not directive packets, they are descriptions.
  3. The theme is nameable in 2-6 words — if you cannot give the bundle
     a label that short, the cluster is incoherent.

ANTI-PATTERNS — refuse to propose bundles that look like:
  - "Things mentioned in chapter 5" (clustering by location, not theme)
  - "Various rules" (clustering by category alone)
  - One-bundle-per-slab (the whole point is cross-slab clustering)
  - Bundles whose extractions come from one slab plus a passing mention
    in another (the second slab must substantively contribute)

THEME RECOGNITION HINTS
  - Marker phrases ("non-negotiable", "first-class", "hard limit", "core
    requirement") often anchor real clusters — sentences near these markers
    in the same slab and similar markers across slabs tend to belong together
  - A theme often has a name the document uses repeatedly — search the
    extractions for repeated noun phrases or named principles
  - Multiple sections that explicitly cross-reference each other suggest
    those sections share a directive theme

Better to return 3 sharp bundles than 8 mushy ones. Coverage is not the goal;
coherence is.

DO NOT propose a bundle if you have to invent payload content not present in
the extractions. Every entry in intent/invariants/rules/etc must trace back
to at least one extracted quote.

Output ONLY valid JSON:
{
  "bundles": [
    {
      "label": "<2-6 word headline>",
      "intent": ["<1-5 statements>"],
      "invariants": ["<up to 8>"],
      "non_assumptions": ["<up to 6>"],
      "warnings": ["<up to 8>"],
      "heuristics": ["<up to 8>"],
      "rules": ["<list>"],
      "canonical_quote_handles": ["<verbatim quotes from extractions>"],
      "depends_on": ["<slab IDs that contributed>"],
      "supports": ["<slab IDs the bundle frames>"],
      "rationale": "<1-3 sentences on why these cohere>"
    }
  ]
}

If no well-formed bundles emerge, return {"bundles": []}.
"""


# ─── Stage 0 — genre hint ────────────────────────────────────────────────


async def genre_hint(slabs: list[Slab], sample_size: int = 20) -> dict:
    """Produce a corpus-level hint that primes Stages 1+2.

    Samples up to ``sample_size`` slabs evenly across the corpus and asks
    the model what kind of document this is and what directive content is
    likely to look like in it. Cheap (one call, small context).
    """
    if not slabs:
        return {
            "document_type": "unknown",
            "domain": "unknown",
            "directive_shape_guidance": "No slabs supplied.",
        }

    sample = slabs
    if len(slabs) > sample_size:
        stride = max(1, len(slabs) // sample_size)
        sample = slabs[::stride][:sample_size]

    lines = []
    for s in sample:
        excerpt = (s.canonical_text or "")[:200].replace("\n", " ")
        lines.append(f"- [{s.id}] {s.title}: {excerpt}")
    user = (
        "SLAB SAMPLE (title + first 200 chars):\n\n"
        + "\n".join(lines)
        + "\n\nProduce the genre hint."
    )
    try:
        resp = await ollama.structured_extract(user, system=STAGE0_GENRE_HINT_SYSTEM)
    except Exception as exc:
        logger.warning("Stage 0 genre hint failed: %r", exc)
        return {
            "document_type": "unknown",
            "domain": "unknown",
            "directive_shape_guidance": (
                "Genre hint extraction failed; proceed with default heuristics."
            ),
        }
    return {
        "document_type": resp.get("document_type", "unknown"),
        "domain": resp.get("domain", "unknown"),
        "directive_shape_guidance": resp.get("directive_shape_guidance", ""),
    }


# ─── Stage 1 — per-slab directive extraction ────────────────────────────


_VALID_CATEGORIES = {
    "intent", "invariant", "non_assumption",
    "warning", "heuristic", "rule", "marker",
}


async def extract_directives(
    slab: Slab, hint: dict,
) -> list[DirectiveExtraction]:
    """Run Stage 1 on one slab. Returns empty list on extraction failure.

    Failure modes treated as "no directives present" (empty list rather than
    raising) so a single slab's bad output doesn't take down the whole pass.
    """
    if not slab.canonical_text or len(slab.canonical_text.strip()) < 20:
        return []

    hint_block = (
        f"CORPUS CONTEXT:\n"
        f"  document_type: {hint.get('document_type', 'unknown')}\n"
        f"  domain: {hint.get('domain', 'unknown')}\n"
        f"  directive shape: {hint.get('directive_shape_guidance', '')}\n\n"
    )
    user = (
        hint_block
        + f"SLAB ID: {slab.id}\n"
        + f"SLAB TITLE: {slab.title}\n"
        + f"SLAB TEXT:\n{slab.canonical_text}\n\n"
        + "Extract directive content per the rules. "
          "Return empty extractions if the slab is purely descriptive."
    )
    try:
        resp = await ollama.structured_extract(user, system=STAGE1_EXTRACT_SYSTEM)
    except Exception as exc:
        logger.warning("Stage 1 extraction failed for %s: %r", slab.id, exc)
        return []

    out: list[DirectiveExtraction] = []
    for ext in resp.get("extractions", []) or []:
        cat = (ext.get("category") or "").strip().lower()
        quote = (ext.get("quote") or "").strip()
        if cat not in _VALID_CATEGORIES or not quote:
            continue
        conf = float(ext.get("confidence") or 0.5)
        out.append(DirectiveExtraction(
            slab_id=slab.id,
            category=cat,
            quote=quote,
            paraphrase=(ext.get("paraphrase") or "").strip(),
            confidence=max(0.0, min(1.0, conf)),
        ))
    return out


async def extract_all_directives(
    slabs: list[Slab], hint: dict,
) -> list[DirectiveExtraction]:
    """Run Stage 1 across all slabs with bounded concurrency.

    Concurrency is bounded by _PARALLEL (matches Ollama's NUM_PARALLEL slot
    count). Higher concurrency just queues client-side without throughput
    gain.
    """
    sem = asyncio.Semaphore(_PARALLEL)

    async def _one(s: Slab) -> list[DirectiveExtraction]:
        async with sem:
            return await extract_directives(s, hint)

    results = await asyncio.gather(*[_one(s) for s in slabs])
    flat: list[DirectiveExtraction] = []
    for r in results:
        flat.extend(r)
    return flat


# ─── Stage 2 — cross-slab bundle assembly ────────────────────────────────


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(s: str, max_len: int = 40) -> str:
    """Bundle-id-safe slug. Strips non-alphanumerics, collapses underscores."""
    return _SLUG_RE.sub("_", s.lower()).strip("_")[:max_len] or "untitled"


async def assemble_bundles(
    slabs: list[Slab],
    extractions: list[DirectiveExtraction],
    hint: dict,
    batch_size: Optional[int] = None,
) -> list[ProposedBundle]:
    """Cluster Stage 1 extractions into directive packets.

    When ``batch_size`` is None (default), all extractions go into a single
    Stage 2 call — best for synthesis quality but limited by num_ctx and
    the 300s httpx timeout. Empirically a 300-slab corpus produces ~400
    extractions, ~17k-token prompts, which local 12-20B models can't
    complete in 5 min.

    When ``batch_size`` is set, extractions are split into ordered chunks
    of that size and Stage 2 runs once per chunk (in parallel under
    _PARALLEL). Bundles are concatenated; cross-batch deduplication is a
    post-pass concern. Order-based batching keeps slabs from the same
    chapter near each other, which means each batch sees locally
    coherent extractions and produces tighter bundles. Cross-chapter
    themes may appear duplicated across batches — that's informative
    signal for the dedup pass.
    """
    if not extractions:
        return []

    if batch_size and len(extractions) > batch_size:
        return await _assemble_bundles_batched(slabs, extractions, hint, batch_size)

    titles = {s.id: s.title for s in slabs}
    ext_lines = []
    for e in extractions:
        title = titles.get(e.slab_id, "")
        ext_lines.append(
            f"- slab={e.slab_id} title={title!r} cat={e.category} "
            f"conf={e.confidence:.2f} quote={e.quote!r}"
        )

    hint_block = (
        f"CORPUS CONTEXT:\n"
        f"  document_type: {hint.get('document_type', 'unknown')}\n"
        f"  domain: {hint.get('domain', 'unknown')}\n"
        f"  directive shape: {hint.get('directive_shape_guidance', '')}\n\n"
    )
    user = (
        hint_block
        + f"EXTRACTIONS ({len(extractions)} total across "
        + f"{len({e.slab_id for e in extractions})} slabs):\n\n"
        + "\n".join(ext_lines)
        + "\n\nAssemble directive packets per the quality gate. "
          "Return empty bundles if no coherent packets emerge."
    )
    # Stage 2 sees ALL Stage 1 extractions in one prompt — a 300-slab
    # corpus easily produces 400+ extractions, which at ~150 chars each
    # blows past the default num_ctx (4096). Bumping to 32768 covers
    # corpora up to ~800 extractions comfortably; if you hit this ceiling
    # the right move is per-chapter batched assembly, not a bigger ctx.
    num_ctx = 32768
    try:
        resp = await ollama.structured_extract(
            user, system=STAGE2_ASSEMBLE_SYSTEM,
            num_ctx=num_ctx, model=_STAGE2_MODEL,
        )
    except Exception as exc:
        logger.warning("Stage 2 assembly failed: %r", exc)
        return []

    out: list[ProposedBundle] = []
    seen_ids: set[str] = set()
    for b in resp.get("bundles", []) or []:
        label = (b.get("label") or "").strip()
        intent = [s.strip() for s in (b.get("intent") or []) if s and s.strip()]
        if not label or not intent:
            continue
        depends = [s.strip() for s in (b.get("depends_on") or []) if s and s.strip()]
        if len({d for d in depends if d in titles}) < 2:
            # Quality gate: cross-slab requirement enforced post-hoc as
            # defense-in-depth; the prompt asks for it but the model
            # occasionally returns single-slab bundles anyway.
            logger.info("Dropping single-slab bundle %r (depends_on=%s)", label, depends)
            continue
        constraint_categories = [
            b.get("warnings"), b.get("non_assumptions"),
            b.get("rules"), b.get("heuristics"),
        ]
        if not any(c for c in constraint_categories if c):
            logger.info(
                "Dropping intent+invariants-only bundle %r — quality gate", label,
            )
            continue

        slug = _slugify(label)
        bundle_id = f"bundle_{slug}_v1"
        # Disambiguate collisions deterministically rather than letting later
        # bundles silently shadow earlier ones.
        n = 2
        while bundle_id in seen_ids:
            bundle_id = f"bundle_{slug}_{n}_v1"
            n += 1
        seen_ids.add(bundle_id)

        out.append(ProposedBundle(
            id=bundle_id,
            label=label,
            intent=intent[:5],
            invariants=[s.strip() for s in (b.get("invariants") or []) if s and s.strip()][:8],
            non_assumptions=[s.strip() for s in (b.get("non_assumptions") or []) if s and s.strip()][:6],
            warnings=[s.strip() for s in (b.get("warnings") or []) if s and s.strip()][:8],
            heuristics=[s.strip() for s in (b.get("heuristics") or []) if s and s.strip()][:8],
            rules=[s.strip() for s in (b.get("rules") or []) if s and s.strip()],
            canonical_quote_handles=[
                s.strip() for s in (b.get("canonical_quote_handles") or []) if s and s.strip()
            ],
            depends_on=depends,
            supports=[s.strip() for s in (b.get("supports") or depends) if s and s.strip()],
            rationale=(b.get("rationale") or "").strip(),
        ))
    return out


async def _assemble_bundles_batched(
    slabs: list[Slab],
    extractions: list[DirectiveExtraction],
    hint: dict,
    batch_size: int,
) -> list[ProposedBundle]:
    """Run Stage 2 against ordered batches in parallel, concat results.

    Batches are ORDER-preserving slices: extractions[0:N], [N:2N], etc.
    Stage 1 emits extractions in slab-iteration order, which for podv3 is
    roughly chapter order, so each batch sees locally coherent material.
    Cross-batch theme overlap is acceptable — the dedup pass downstream
    is a separate concern (see _dedupe_proposed_bundles).
    """
    n = len(extractions)
    batches: list[list[DirectiveExtraction]] = [
        extractions[i:i + batch_size] for i in range(0, n, batch_size)
    ]
    logger.info(
        "Stage 2 batched: %d batches of ~%d extractions each (total %d)",
        len(batches), batch_size, n,
    )

    sem = asyncio.Semaphore(_PARALLEL)

    async def _run_batch(
        idx: int, batch: list[DirectiveExtraction],
    ) -> list[ProposedBundle]:
        async with sem:
            logger.info(
                "Stage 2 batch %d/%d (%d extractions, slabs %d-%d)...",
                idx + 1, len(batches), len(batch),
                batches.index(batch) * batch_size,
                batches.index(batch) * batch_size + len(batch) - 1,
            )
            # Pass batch_size=None to inner call to avoid recursion.
            return await assemble_bundles(slabs, batch, hint, batch_size=None)

    batch_results = await asyncio.gather(*[
        _run_batch(i, b) for i, b in enumerate(batches)
    ])
    flat: list[ProposedBundle] = []
    for r in batch_results:
        flat.extend(r)

    # Disambiguate ID collisions across batches deterministically.
    seen: dict[str, int] = {}
    for b in flat:
        if b.id in seen:
            seen[b.id] += 1
            base = b.id.removesuffix("_v1")
            b.id = f"{base}_b{seen[b.id]}_v1"
        else:
            seen[b.id] = 1

    logger.info("Stage 2 batched: %d total bundles before dedup", len(flat))
    return flat


# ─── Stage 3 — cross-batch dedup / merge ─────────────────────────────────


STAGE3_DEDUPE_SYSTEM = """You are de-duplicating directive packets ("bundles") \
produced by parallel batched extraction over a corpus. Because each batch \
saw only a slice of the corpus, the SAME thematic packet often appears \
multiple times with slightly different labels and slab coverage.

Your job: identify clusters of bundles that describe the SAME directive packet \
and merge each cluster into ONE canonical bundle. Distinct themes stay \
separate.

WHAT MAKES TWO BUNDLES THE SAME PACKET
  - Their intent statements describe the same purpose, possibly in different
    words (e.g., "user agency" vs "user-controlled experience")
  - Their rules/invariants contain ≥50% identical or near-identical strings.
    When this happens, you are looking at TWO COPIES of the same packet —
    merge them. Do not preserve them as "different angles on the same theme."
  - Their non_assumptions / warnings overlap by ≥50% identical strings.
    Same logic — identical payload = same packet.
  - Their canonical_quote_handles draw from the same passages

WHAT KEEPS BUNDLES DISTINCT — STRICT RULES

  ✗ DO NOT merge bundles that operate on different subsystems even if they
    share constraint patterns. "Wind layer agency" and "Sound layer agency"
    and "Scent layer agency" are THREE distinct packets — they share the
    agency-pattern but EACH describes a specific subsystem's contract.
    The right merge for these is into a single "Subsystem Agency Layers"
    bundle ONLY IF their intent statements are explicitly cross-subsystem.
    If each intent names its specific subsystem, KEEP THEM SEPARATE.

  ✗ DO NOT merge "Safety" with "User Agency" — these are sibling packets
    with shared values but different scope. Safety = hard system bounds.
    Agency = user control surfaces.

  ✗ DO NOT merge "MVP Scope" or "Falsifiability Posture" with any
    operational/architectural packet. Scope packets describe what the
    document is NOT; architectural packets describe what it IS.

DECISION TEST: before merging two bundles, ask "if I drilled into this
merged bundle in the UI, would the resulting payload feel like ONE coherent
directive packet I would want to read as one document, or would it feel
like a stapled-together hash of multiple themes?" If the latter, do NOT
merge.

MERGE RULES
  1. Pick the most descriptive label from the cluster. If none is great,
     write a new one (still 2-6 words).
  2. Union all payload fields (intent / invariants / rules / etc.), then
     deduplicate within each field — remove near-duplicate strings keeping
     the most complete version.
  3. Union depends_on and supports lists (deduplicate).
  4. Concatenate rationales with " | " separator if distinct, otherwise
     keep the most informative one.
  5. STRIP any "slab=X:" prefix from canonical_quote_handles — these are
     prompt-following artifacts, the field should contain ONLY the quote.

CRITICAL: Do not invent payload content. Every entry in the merged bundle
must trace back to at least one input bundle. You may rephrase for
deduplication but not add new constraints.

Output ONLY valid JSON:
{
  "merged_bundles": [
    {
      "source_bundle_ids": ["<input bundle id 1>", "<input bundle id 2>", ...],
      "label": "<canonical 2-6 word label>",
      "intent": [...],
      "invariants": [...],
      "non_assumptions": [...],
      "warnings": [...],
      "heuristics": [...],
      "rules": [...],
      "canonical_quote_handles": [...],
      "depends_on": [...],
      "supports": [...],
      "rationale": "..."
    }
  ]
}

If a single input bundle has no duplicates in the set, include it in
``merged_bundles`` with ``source_bundle_ids`` listing only itself — every
input bundle must appear in exactly one output bundle.
"""


# Strip "slab=X:" prefix introduced by Stage 2's prompt-following lapse.
_QUOTE_PREFIX_RE = re.compile(r"^slab=[a-zA-Z0-9_]+(?:\s+title=[^:]*)?:\s*")


def _clean_quote(s: str) -> str:
    """Remove Stage 2 prompt-format leakage from quote strings."""
    return _QUOTE_PREFIX_RE.sub("", s).strip()


def _payload_signature(b: ProposedBundle) -> set[str]:
    """Multiset of normalized constraint strings for similarity comparison.

    We compare on the CONSTRAINT-bearing payload (rules, invariants,
    non_assumptions, warnings) — these are what the schema gates on.
    Intent statements are excluded because they're more variable in
    wording even when bundles describe identical packets.
    """
    parts: list[str] = []
    for field_name in ("rules", "invariants", "non_assumptions", "warnings"):
        for s in getattr(b, field_name, []) or []:
            # Normalize whitespace + lowercase for similarity matching.
            norm = " ".join(s.lower().split())
            if norm:
                parts.append(norm)
    return set(parts)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def _force_merge_obvious_duplicates(
    bundles: list[ProposedBundle], threshold: float = 0.5,
) -> list[ProposedBundle]:
    """Force-merge bundle pairs whose constraint payload Jaccard ≥ threshold.

    Catches the Stage-3-failure-mode where the LLM keeps two bundles
    distinct despite their rules/invariants/non_assumptions being
    near-verbatim copies of each other. The LLM is the primary dedup
    mechanism; this is a tripwire for the cases it misses.

    threshold=0.5 means "≥50% of the union of their constraint strings
    are identical." Tested empirically against the podv5 v1 output where
    Bundle 1 and Bundle 2 had 6/6 identical non_assumptions, 3/3
    identical warnings, and high rule overlap.
    """
    if len(bundles) <= 1:
        return bundles
    sigs = [_payload_signature(b) for b in bundles]
    # Union-find over the bundle list
    parent = list(range(len(bundles)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    for i in range(len(bundles)):
        for j in range(i + 1, len(bundles)):
            if _jaccard(sigs[i], sigs[j]) >= threshold:
                union(i, j)

    # Group by union root
    groups: dict[int, list[int]] = {}
    for i in range(len(bundles)):
        groups.setdefault(find(i), []).append(i)

    out: list[ProposedBundle] = []
    for root, idxs in groups.items():
        if len(idxs) == 1:
            out.append(bundles[idxs[0]])
            continue
        # Pick the bundle with the widest span as the canonical
        canonical_idx = max(idxs, key=lambda i: len({d for d in bundles[i].depends_on}))
        canonical = bundles[canonical_idx]
        # Union depends_on / supports / canonical_quote_handles from siblings
        depends = set(canonical.depends_on)
        supports = set(canonical.supports)
        quotes: list[str] = list(canonical.canonical_quote_handles)
        for i in idxs:
            if i == canonical_idx:
                continue
            depends.update(bundles[i].depends_on)
            supports.update(bundles[i].supports)
            for q in bundles[i].canonical_quote_handles:
                if q not in quotes:
                    quotes.append(q)
        merged = ProposedBundle(
            id=canonical.id,
            label=canonical.label,
            intent=canonical.intent,
            invariants=canonical.invariants,
            non_assumptions=canonical.non_assumptions,
            warnings=canonical.warnings,
            heuristics=canonical.heuristics,
            rules=canonical.rules,
            canonical_quote_handles=quotes,
            depends_on=sorted(depends),
            supports=sorted(supports),
            rationale=(canonical.rationale or "")
                      + " | force-merged with: "
                      + ", ".join(bundles[i].label for i in idxs if i != canonical_idx),
        )
        logger.info(
            "Stage 3 force-merge: %d bundles (Jaccard ≥ 0.5) → canonical %r",
            len(idxs), canonical.label,
        )
        out.append(merged)
    return out


async def dedupe_bundles(
    bundles: list[ProposedBundle], hint: dict,
) -> list[ProposedBundle]:
    """Merge semantically equivalent bundles produced by batched Stage 2.

    Single LLM call — input is bounded by number of bundles (typically
    <20 even on large corpora), not number of extractions.
    """
    if len(bundles) <= 1:
        return bundles

    bundle_lines = []
    for b in bundles:
        bundle_lines.append(json.dumps({
            "id": b.id,
            "label": b.label,
            "intent": b.intent,
            "invariants": b.invariants,
            "non_assumptions": b.non_assumptions,
            "warnings": b.warnings,
            "heuristics": b.heuristics,
            "rules": b.rules,
            "canonical_quote_handles": b.canonical_quote_handles,
            "depends_on": b.depends_on,
            "supports": b.supports,
            "rationale": b.rationale,
        }, ensure_ascii=False))

    hint_block = (
        f"CORPUS CONTEXT:\n"
        f"  document_type: {hint.get('document_type', 'unknown')}\n"
        f"  domain: {hint.get('domain', 'unknown')}\n\n"
    )
    user = (
        hint_block
        + f"INPUT BUNDLES ({len(bundles)} total, one per line as JSON):\n\n"
        + "\n".join(bundle_lines)
        + "\n\nDeduplicate and merge per the rules. Every input bundle id "
        + "must appear in exactly one output bundle's source_bundle_ids."
    )

    try:
        resp = await ollama.structured_extract(
            user, system=STAGE3_DEDUPE_SYSTEM,
            num_ctx=16384, model=_STAGE2_MODEL,
        )
    except Exception as exc:
        logger.warning("Stage 3 dedup failed: %r", exc)
        return bundles  # Fall back to un-deduped set

    merged_raw = resp.get("merged_bundles", []) or []
    out: list[ProposedBundle] = []
    seen_ids: set[str] = set()
    accounted_for: set[str] = set()
    input_ids = {b.id for b in bundles}

    for m in merged_raw:
        label = (m.get("label") or "").strip()
        intent = [s.strip() for s in (m.get("intent") or []) if s and s.strip()]
        if not label or not intent:
            continue
        sources = [s for s in (m.get("source_bundle_ids") or []) if s in input_ids]
        if not sources:
            continue
        accounted_for.update(sources)

        # Union depends_on / supports across source bundles in case the
        # model dropped any during merge.
        depends_union: set[str] = set()
        supports_union: set[str] = set()
        for sid in sources:
            for src in bundles:
                if src.id == sid:
                    depends_union.update(src.depends_on)
                    supports_union.update(src.supports)
                    break
        depends = list(depends_union)
        supports = sorted(supports_union | depends_union)

        slug = _slugify(label)
        bid = f"bundle_{slug}_v1"
        n = 2
        while bid in seen_ids:
            bid = f"bundle_{slug}_{n}_v1"
            n += 1
        seen_ids.add(bid)

        out.append(ProposedBundle(
            id=bid,
            label=label,
            intent=intent[:5],
            invariants=[s.strip() for s in (m.get("invariants") or []) if s and s.strip()][:8],
            non_assumptions=[s.strip() for s in (m.get("non_assumptions") or []) if s and s.strip()][:6],
            warnings=[s.strip() for s in (m.get("warnings") or []) if s and s.strip()][:8],
            heuristics=[s.strip() for s in (m.get("heuristics") or []) if s and s.strip()][:8],
            rules=[s.strip() for s in (m.get("rules") or []) if s and s.strip()],
            canonical_quote_handles=[
                _clean_quote(s)
                for s in (m.get("canonical_quote_handles") or [])
                if s and s.strip()
            ],
            depends_on=depends,
            supports=supports,
            rationale=(m.get("rationale") or "").strip(),
        ))

    # Safety net: any input bundle not accounted for by the model's
    # source_bundle_ids gets carried through unchanged with prefix cleanup.
    # Better to keep an un-merged duplicate than lose a valid bundle.
    for src in bundles:
        if src.id in accounted_for:
            continue
        logger.info("Stage 3: carrying through unmerged bundle %r", src.id)
        carry = ProposedBundle(
            id=src.id,
            label=src.label,
            intent=src.intent,
            invariants=src.invariants,
            non_assumptions=src.non_assumptions,
            warnings=src.warnings,
            heuristics=src.heuristics,
            rules=src.rules,
            canonical_quote_handles=[_clean_quote(s) for s in src.canonical_quote_handles],
            depends_on=src.depends_on,
            supports=src.supports,
            rationale=src.rationale,
        )
        out.append(carry)

    logger.info(
        "Stage 3 dedup (LLM): %d input bundles → %d merged bundles",
        len(bundles), len(out),
    )
    # Post-pass: catch obvious-duplicate pairs the LLM kept distinct.
    # Empirically the LLM sometimes preserves two near-identical bundles
    # if their labels diverge enough — the Jaccard check on constraint
    # payloads doesn't care about labels.
    out = _force_merge_obvious_duplicates(out)
    logger.info(
        "Stage 3 dedup (force-merge): final %d bundles", len(out),
    )
    return out


# ─── Public entry point ──────────────────────────────────────────────────


@dataclass
class ExtractionResult:
    hint: dict
    extractions: list[DirectiveExtraction]
    bundles: list[ProposedBundle]


async def run(
    slabs: list[Slab],
    stage2_batch_size: Optional[int] = None,
    dedupe: bool = True,
) -> ExtractionResult:
    """End-to-end Stage 0 → 1 → 2 over a slab list.

    Returns the full result including extractions, so callers can audit
    Stage 1 output independently of Stage 2 clustering decisions.

    ``stage2_batch_size`` controls Stage 2 batching. None = single call
    (best quality, requires the model to handle the full prompt within
    timeout). Setting it to e.g. 80 splits large corpora into chunks so
    local models stay under the 300s timeout.
    """
    if not slabs:
        return ExtractionResult(
            hint={"document_type": "empty", "domain": "empty",
                  "directive_shape_guidance": ""},
            extractions=[], bundles=[],
        )

    logger.info("directive_extractor: %d slabs", len(slabs))
    hint = await genre_hint(slabs)
    logger.info(
        "directive_extractor: genre hint = type=%s domain=%s",
        hint.get("document_type"), hint.get("domain"),
    )
    extractions = await extract_all_directives(slabs, hint)
    logger.info(
        "directive_extractor: stage 1 produced %d extractions across %d slabs",
        len(extractions), len({e.slab_id for e in extractions}),
    )
    bundles = await assemble_bundles(
        slabs, extractions, hint, batch_size=stage2_batch_size,
    )
    logger.info("directive_extractor: stage 2 produced %d bundles", len(bundles))
    if dedupe and len(bundles) > 1:
        bundles = await dedupe_bundles(bundles, hint)
        logger.info("directive_extractor: post-dedup %d bundles", len(bundles))
    return ExtractionResult(hint=hint, extractions=extractions, bundles=bundles)


# ─── CLI runner (testing) ────────────────────────────────────────────────


def _dump_result(result: ExtractionResult) -> str:
    """Pretty-print Stage 1 + Stage 2 output for human inspection."""
    out = []
    out.append("=" * 70)
    out.append(f"GENRE HINT")
    out.append(f"  type:   {result.hint.get('document_type')}")
    out.append(f"  domain: {result.hint.get('domain')}")
    out.append(f"  shape:  {result.hint.get('directive_shape_guidance')}")
    out.append("=" * 70)
    out.append(f"STAGE 1: {len(result.extractions)} extractions")
    out.append("")
    by_slab: dict[str, list[DirectiveExtraction]] = {}
    for e in result.extractions:
        by_slab.setdefault(e.slab_id, []).append(e)
    for sid, exts in by_slab.items():
        out.append(f"  [{sid}] ({len(exts)} extractions)")
        for e in exts:
            out.append(f"    {e.category:14s} conf={e.confidence:.2f}  {e.quote[:120]}")
    out.append("=" * 70)
    out.append(f"STAGE 2: {len(result.bundles)} bundles")
    out.append("")
    for b in result.bundles:
        out.append(f"  ── {b.label} ({b.id})")
        out.append(f"     spans: {len(set(b.depends_on))} slabs")
        out.append(f"     intent: {b.intent}")
        if b.invariants:
            out.append(f"     invariants: {b.invariants}")
        if b.non_assumptions:
            out.append(f"     non_assumptions: {b.non_assumptions}")
        if b.warnings:
            out.append(f"     warnings: {b.warnings}")
        if b.heuristics:
            out.append(f"     heuristics: {b.heuristics}")
        if b.rules:
            out.append(f"     rules: {b.rules}")
        if b.rationale:
            out.append(f"     rationale: {b.rationale}")
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
    result = await run(slabs)
    print(_dump_result(result))
    # Also dump JSON next to the result for downstream inspection
    json_path = CORPORA_ROOT / collection_id / f"directive_extract_{slab_limit or 'all'}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "hint": result.hint,
            "extractions": [vars(e) for e in result.extractions],
            "bundles": [vars(b) for b in result.bundles],
        }, f, indent=2)
    print(f"\nWrote {json_path}")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cid = sys.argv[1] if len(sys.argv) > 1 else "podv3"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    asyncio.run(_cli(cid, limit))
