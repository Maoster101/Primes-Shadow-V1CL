"""Side-by-side comparison: current narrative miner vs Pass 1 (anchor-only).

Calibration harness for the proposed three-pass mining architecture. Runs
the existing `extract_from_segment` (which extracts anchors + slabs +
bundles in one prompt) AND a new Pass-1-only anchor extraction (with
genre-agnostic prompts + system/user split) against the same fixture
text, segment-by-segment. Prints a side-by-side anchor diff so we can
eyeball whether Pass 1 alone covers the anchors the current pipeline
finds, without losing detail.

Run:
    python scripts/mining_compare_pass1.py

Optionally point at a different fixture via the FIXTURE env var.
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
import time
from pathlib import Path

# Make the project root importable when running directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.services import ollama
from src.services.narrative_miner import (
    NarrativeSegment,
    segment_narrative,
    extract_from_segment,
)


# ── Pass 1 prompt — genre-agnostic anchor-only extraction ─────────────
# System half stays constant across every parallel call in a pass;
# user half holds only the variable segment text. Future implementation
# will pass these via ollama.structured_extract(prompt, system=...) so
# the system prefix gets cached after the first call.

PASS1_SYSTEM = """You are extracting NAMED ENTITIES and DISTINCTIVE PHRASES from a text segment for a knowledge corpus called Prime's Shadow.

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


def _pass1_user_prompt(text: str) -> str:
    return f"Text segment:\n{text}"


# ── Pass 3 prompt — slab extraction with anchor context ───────────────
# Receives the canonical anchor list filtered to those that appear in
# this segment, and extracts narrative beats / processes / arguments /
# principles as multi-sentence slabs. Slabs are allowed to add anchors
# they reference that Pass 1 missed (new_anchors field — the safety
# net for the over-shred / under-shred trade-off).

PASS3_SYSTEM = """You are extracting SLABS — multi-sentence canonical descriptions of structured content — from a text segment for a knowledge corpus called Prime's Shadow.

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


def _anchors_in_segment(segment_text: str, all_anchors: list[dict]) -> list[dict]:
    """Filter anchors to those whose canonical_phrase or any alias
    appears (case-insensitive) in this segment's text. This is the
    "anchor list scoped to this segment" the Pass 3 prompt receives.
    Stripping markdown emphasis from the segment first so we don't
    miss anchors that surface as `*foo*` or `**foo**` in the source.
    """
    haystack = re.sub(r"[*_`]", "", segment_text).lower()
    out = []
    for a in all_anchors:
        forms = _all_forms(a)
        if any(f and f in haystack for f in forms):
            out.append(a)
    return out


def _format_anchor_list(anchors: list[dict]) -> str:
    if not anchors:
        return "(none)"
    lines = []
    for a in anchors:
        phrase = a.get("phrase") or a.get("canonical_phrase") or ""
        aliases = a.get("aliases") or []
        suffix = f"  (aliases: {', '.join(aliases)})" if aliases else ""
        lines.append(f"  - {phrase}{suffix}")
    return "\n".join(lines)


def _pass3_user_prompt(text: str, segment_anchors: list[dict]) -> str:
    return (
        f"Anchors already extracted from this segment "
        f"(use these in references_anchors when relevant):\n"
        f"{_format_anchor_list(segment_anchors)}\n\n"
        f"Text segment:\n{text}"
    )


async def extract_slabs_pass3(
    segment: NarrativeSegment,
    segment_anchors: list[dict],
) -> tuple[list[dict], list[dict], float]:
    """Run Pass 3 (slab extraction with anchor context).

    Returns (slabs, new_anchors, elapsed). new_anchors are pulled out
    of the slab outputs and presented separately so the harness can
    show the safety-net effect at a glance.
    """
    user = _pass3_user_prompt(segment.text, segment_anchors)
    t0 = time.perf_counter()
    try:
        resp = await ollama.structured_extract(user, system=PASS3_SYSTEM)
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print(f"  [Pass 3 segment {segment.order}] FAILED: {type(exc).__name__}: {exc!r}")
        return [], [], elapsed
    elapsed = time.perf_counter() - t0
    raw_slabs = resp.get("slabs", []) if isinstance(resp, dict) else []
    slabs: list[dict] = []
    new_anchors: list[dict] = []
    for s in raw_slabs:
        if not isinstance(s, dict):
            continue
        title = (s.get("title") or "").strip()
        text = (s.get("canonical_text") or "").strip()
        if not text:
            continue
        slabs.append({
            "title": title,
            "canonical_text": text,
            "references_anchors": list(s.get("references_anchors") or [])[:10],
            "confidence": float(s.get("confidence", 0.5)),
            "justification": (s.get("justification") or "").strip(),
        })
        for na in (s.get("new_anchors") or []):
            if not isinstance(na, dict):
                continue
            phrase = (na.get("canonical_phrase") or "").strip()
            if not phrase:
                continue
            new_anchors.append({
                "phrase": phrase,
                "aliases": list(na.get("aliases") or [])[:5],
                "confidence": float(s.get("confidence", 0.7)),
                "justification": f"new_anchor from slab '{title}'",
            })
    return slabs, new_anchors, elapsed


async def extract_anchors_pass1(segment: NarrativeSegment) -> tuple[list[dict], float]:
    """Run the new Pass-1-only extraction on a segment.

    Returns (anchors, elapsed_seconds). Anchors are dicts with the
    same shape as the prompt's JSON schema; we don't bother converting
    to MiningProposal — this harness just compares lists.
    """
    user = _pass1_user_prompt(segment.text)
    t0 = time.perf_counter()
    try:
        resp = await ollama.structured_extract(user, system=PASS1_SYSTEM)
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print(f"  [Pass 1 segment {segment.order}] FAILED: {type(exc).__name__}: {exc!r}")
        return [], elapsed
    elapsed = time.perf_counter() - t0
    raw = resp.get("anchors", []) if isinstance(resp, dict) else []
    out: list[dict] = []
    for r in raw:
        if not isinstance(r, dict):
            continue
        phrase = (r.get("canonical_phrase") or "").strip()
        if not phrase:
            continue
        out.append({
            "phrase": phrase,
            "aliases": list(r.get("aliases") or [])[:5],
            "confidence": float(r.get("confidence", 0.5)),
            "justification": (r.get("justification") or "").strip(),
        })
    return out, elapsed


# ── Comparison helpers ──────────────────────────────────────────────


def _norm(s: str) -> str:
    """Lowercase + collapse whitespace + strip punctuation for fuzzy match."""
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _all_forms(item: dict) -> set[str]:
    """Every surface form an anchor item could match against."""
    forms = {_norm(item.get("phrase") or item.get("canonical_phrase") or "")}
    for a in item.get("aliases") or []:
        forms.add(_norm(a))
    forms.discard("")
    return forms


def overlap_diff(baseline: list[dict], candidate: list[dict]) -> dict:
    """Match candidate anchors against baseline by normalized phrase set.

    Returns three lists:
      - both:    items in both lists (keyed by candidate, with matching baseline)
      - only_baseline: baseline anchors candidate didn't surface
      - only_candidate: candidate anchors baseline didn't have
    """
    base_used: set[int] = set()
    both = []
    only_candidate = []
    for c in candidate:
        c_forms = _all_forms(c)
        match_idx = None
        for i, b in enumerate(baseline):
            if i in base_used:
                continue
            b_forms = _all_forms(b)
            if c_forms & b_forms:
                match_idx = i
                break
        if match_idx is not None:
            base_used.add(match_idx)
            both.append({"baseline": baseline[match_idx], "candidate": c})
        else:
            only_candidate.append(c)
    only_baseline = [b for i, b in enumerate(baseline) if i not in base_used]
    return {"both": both, "only_baseline": only_baseline, "only_candidate": only_candidate}


# ── Pretty printing ────────────────────────────────────────────────


def _fmt_anchor(d: dict) -> str:
    phrase = d.get("phrase") or d.get("canonical_phrase") or ""
    conf = d.get("confidence", 0.0)
    aliases = d.get("aliases") or []
    alias_str = f"  aliases={aliases}" if aliases else ""
    return f'"{phrase}" (conf={conf:.2f}){alias_str}'


def _fmt_slab(s: dict, max_text: int = 220) -> str:
    title = s.get("title", "(untitled)")
    text = s.get("canonical_text", "") or s.get("text", "")
    refs = s.get("references_anchors") or []
    conf = s.get("confidence", 0.0)
    text_short = text if len(text) <= max_text else text[:max_text] + "…"
    ref_line = f"\n        refs: {refs}" if refs else ""
    return f'"{title}" (conf={conf:.2f}){ref_line}\n        text: {text_short}'


def print_segment_compare(
    seg: NarrativeSegment,
    baseline_anchors: list[dict],
    pass1_anchors: list[dict],
    baseline_slabs: list[dict],
    pass3_slabs: list[dict],
    pass3_new_anchors: list[dict],
    baseline_secs: float,
    pass1_secs: float,
    pass3_secs: float,
):
    anchor_diff = overlap_diff(baseline_anchors, pass1_anchors)
    print(f"\n{'='*78}")
    head = f' (heading: "{seg.heading}")' if seg.heading else ""
    print(f"SEGMENT {seg.order}{head}")
    print(f"  text preview: {seg.text[:140]!r}{'…' if len(seg.text) > 140 else ''}")
    print(f"  baseline: {len(baseline_anchors)} anchors + {len(baseline_slabs)} slabs in {baseline_secs:.1f}s")
    print(f"  pass1:    {len(pass1_anchors)} anchors in {pass1_secs:.1f}s")
    print(f"  pass3:    {len(pass3_slabs)} slabs (+ {len(pass3_new_anchors)} new_anchors) in {pass3_secs:.1f}s")

    # ── Anchor diff (baseline vs pass1) ──
    print()
    if anchor_diff["both"]:
        print(f"  ✓ ANCHORS BOTH ({len(anchor_diff['both'])}):")
        for pair in anchor_diff["both"]:
            print(f"      baseline: {_fmt_anchor(pair['baseline'])}")
            print(f"      pass1:    {_fmt_anchor(pair['candidate'])}")
    if anchor_diff["only_baseline"]:
        print(f"\n  ⚠ ANCHORS ONLY BASELINE — Pass 1 missed ({len(anchor_diff['only_baseline'])}):")
        for b in anchor_diff["only_baseline"]:
            print(f"      {_fmt_anchor(b)}")
    if anchor_diff["only_candidate"]:
        print(f"\n  + ANCHORS ONLY PASS 1 ({len(anchor_diff['only_candidate'])}):")
        for c in anchor_diff["only_candidate"]:
            print(f"      {_fmt_anchor(c)}")

    # ── Slab comparison (baseline vs pass3) ──
    if baseline_slabs or pass3_slabs:
        print(f"\n  --- SLABS ---")
    if baseline_slabs:
        print(f"\n  BASELINE SLABS ({len(baseline_slabs)}):")
        for s in baseline_slabs:
            print(f"    - {_fmt_slab(s)}")
    if pass3_slabs:
        print(f"\n  PASS 3 SLABS ({len(pass3_slabs)}):")
        for s in pass3_slabs:
            print(f"    - {_fmt_slab(s)}")
    if pass3_new_anchors:
        print(f"\n  PASS 3 new_anchors (safety-net catches Pass 1 missed):")
        for na in pass3_new_anchors:
            print(f"      {_fmt_anchor(na)}")


# ── Main harness ───────────────────────────────────────────────────


async def main():
    fixture_path = Path(os.environ.get("FIXTURE", "tests/fixtures/vin_diesel_skit.txt"))
    if not fixture_path.exists():
        print(f"Fixture not found: {fixture_path}", file=sys.stderr)
        sys.exit(2)

    raw = fixture_path.read_text(encoding="utf-8")
    segments = segment_narrative(raw)

    print(f"Fixture:  {fixture_path}")
    print(f"Length:   {len(raw)} chars")
    print(f"Segments: {len(segments)}")
    print(f"Model:    {ollama.CHAT_MODEL}")
    print()
    print("Running BASELINE (current narrative miner extract_from_segment) "
          "and PASS 1 (anchor-only, genre-agnostic) on each segment in series.")
    print("Per-segment timings reported below.")

    # Aggregate buckets — baseline extracts anchors+slabs+bundles in one call,
    # so we sort it back into anchor and slab lists for the comparison.
    baseline_total_secs = 0.0
    pass1_total_secs = 0.0
    pass3_total_secs = 0.0
    all_baseline_anchors: list[dict] = []
    all_pass1_anchors: list[dict] = []
    all_baseline_slabs: list[dict] = []
    all_pass3_slabs: list[dict] = []
    all_pass3_new_anchors: list[dict] = []

    for seg in segments:
        # Baseline: full extraction (anchors+slabs+bundles in one call).
        # We pass empty prior context so this matches a "first run" of
        # the current pipeline — same starting condition as Pass 1.
        t0 = time.perf_counter()
        try:
            baseline_props = await extract_from_segment(
                seg, prior_anchor_phrases=[], prior_slab_titles=[],
            )
        except Exception as exc:
            baseline_props = []
            print(f"  [baseline segment {seg.order}] FAILED: "
                  f"{type(exc).__name__}: {exc!r}")
        baseline_secs = time.perf_counter() - t0
        baseline_total_secs += baseline_secs
        baseline_anchors = [
            {"phrase": p.canonical_phrase, "aliases": p.aliases,
             "confidence": p.confidence, "justification": p.justification}
            for p in baseline_props if p.proposal_type == "anchor"
        ]
        baseline_slabs = [
            {"title": p.title, "canonical_text": p.canonical_text,
             "confidence": p.confidence, "justification": p.justification}
            for p in baseline_props if p.proposal_type == "slab"
        ]

        # Pass 1: anchors only with new prompt
        pass1_anchors, pass1_secs = await extract_anchors_pass1(seg)
        pass1_total_secs += pass1_secs

        # Pass 3: slabs with anchor context. Filter the running canonical
        # anchor list (Pass 1 has produced anchors for this + prior
        # segments) to those that surface in this segment's text — that
        # mirrors the production behaviour where slab context is scoped
        # to anchors actually appearing in the segment.
        running_anchors = all_pass1_anchors + pass1_anchors
        seg_anchor_context = _anchors_in_segment(seg.text, running_anchors)
        pass3_slabs, pass3_new_anchors, pass3_secs = await extract_slabs_pass3(
            seg, seg_anchor_context,
        )
        pass3_total_secs += pass3_secs

        all_baseline_anchors.extend(baseline_anchors)
        all_pass1_anchors.extend(pass1_anchors)
        all_baseline_slabs.extend(baseline_slabs)
        all_pass3_slabs.extend(pass3_slabs)
        all_pass3_new_anchors.extend(pass3_new_anchors)
        print_segment_compare(
            seg,
            baseline_anchors=baseline_anchors,
            pass1_anchors=pass1_anchors,
            baseline_slabs=baseline_slabs,
            pass3_slabs=pass3_slabs,
            pass3_new_anchors=pass3_new_anchors,
            baseline_secs=baseline_secs,
            pass1_secs=pass1_secs,
            pass3_secs=pass3_secs,
        )

    # Aggregate diff across all segments (deduped within each side first)
    def _dedupe(items: list[dict]) -> list[dict]:
        seen: dict[frozenset, dict] = {}
        for it in items:
            key = frozenset(_all_forms(it))
            # Empty key items can't be matched, drop them
            if not key:
                continue
            cur = seen.get(key)
            if cur is None or it.get("confidence", 0) > cur.get("confidence", 0):
                seen[key] = it
        return list(seen.values())

    baseline_dedup = _dedupe(all_baseline_anchors)
    pass1_dedup = _dedupe(all_pass1_anchors)
    # Combined Pass 1 + Pass 3 new_anchors — the actual anchor list the
    # full architecture produces. The new_anchors safety net pulls in
    # entities Pass 1 missed.
    pass_combined_dedup = _dedupe(all_pass1_anchors + all_pass3_new_anchors)
    agg_pass1 = overlap_diff(baseline_dedup, pass1_dedup)
    agg_combined = overlap_diff(baseline_dedup, pass_combined_dedup)

    print(f"\n{'='*78}")
    print("AGGREGATE — ANCHORS (deduped across segments)")
    print(f"  baseline anchors total:                    {len(baseline_dedup)}")
    print(f"  pass1 anchors total:                       {len(pass1_dedup)}")
    print(f"  pass3 new_anchors total (pass1 misses):    {len(_dedupe(all_pass3_new_anchors))}")
    print(f"  pass1 + pass3.new_anchors combined total:  {len(pass_combined_dedup)}")
    print()
    print(f"  recall vs baseline:")
    if baseline_dedup:
        r1 = len(agg_pass1['both']) / len(baseline_dedup)
        rc = len(agg_combined['both']) / len(baseline_dedup)
        print(f"    pass1 alone:                           {r1:.0%}")
        print(f"    pass1 + pass3.new_anchors:             {rc:.0%}")
    print(f"  baseline-only (still missed by combined):  {len(agg_combined['only_baseline'])}")
    if agg_combined['only_baseline']:
        for b in agg_combined['only_baseline']:
            print(f"      {_fmt_anchor(b)}")

    print(f"\n{'='*78}")
    print("AGGREGATE — SLABS")
    print(f"  baseline slabs total:                      {len(all_baseline_slabs)}")
    print(f"  pass3 slabs total:                         {len(all_pass3_slabs)}")
    # Anchor reference rate: a slab's "structural connectivity" — does
    # the slab pass actually cite anchors? (Baseline slabs don't, so
    # this is a NEW capability we're adding.)
    refs_total = sum(len(s.get('references_anchors') or []) for s in all_pass3_slabs)
    refs_per_slab = refs_total / max(1, len(all_pass3_slabs))
    print(f"  pass3 anchor references (total):           {refs_total}")
    print(f"  pass3 anchor references per slab (avg):    {refs_per_slab:.1f}")

    print(f"\n{'='*78}")
    print("TIMING")
    print(f"  baseline total: {baseline_total_secs:.1f}s "
          f"({baseline_total_secs/max(1,len(segments)):.1f}s/segment avg)")
    print(f"  pass1 total:    {pass1_total_secs:.1f}s "
          f"({pass1_total_secs/max(1,len(segments)):.1f}s/segment avg)")
    print(f"  pass3 total:    {pass3_total_secs:.1f}s "
          f"({pass3_total_secs/max(1,len(segments)):.1f}s/segment avg)")
    full_serial = pass1_total_secs + pass3_total_secs
    print(f"  pass1+pass3 serial total: {full_serial:.1f}s "
          f"(what this harness measured)")
    print(f"  pass1+pass3 with parallelism (estimated): "
          f"~{(pass1_total_secs/max(1,len(segments)) + pass3_total_secs/max(1,len(segments))) * 1.2:.1f}s "
          f"(per-segment * 1.2 overhead, assuming Semaphore(N) ≥ segments)")
    print()
    print("Note: this harness runs Pass 1 and Pass 3 SERIALLY for a clean")
    print("per-call timing comparison. The production architecture runs")
    print("each pass with asyncio.gather + Semaphore so all N segments")
    print("are dispatched concurrently within a pass — the wallclock win")
    print("comes from N-fold parallelism, not per-call shortening.")


if __name__ == "__main__":
    asyncio.run(main())
