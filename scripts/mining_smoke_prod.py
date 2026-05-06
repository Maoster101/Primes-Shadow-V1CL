"""End-to-end smoke test of the production NarrativeMiner.mine() entrypoint.

Verifies the three-pass refactor preserves the public contract and works
against the fixture inputs. Different from mining_compare_pass1.py: that
script compares per-segment outputs of two extraction strategies. THIS
script invokes the real production mine() the way /push-mined would, and
reports the dict shape downstream consumers will see.

Run:
    python scripts/mining_smoke_prod.py                                 # vin diesel skit
    FIXTURE=tests/fixtures/mirror_paper_excerpt.txt python scripts/mining_smoke_prod.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.services.narrative_miner import NarrativeMiner
from src.services.corpus import CorpusStore
from src.services import ollama


async def main():
    fixture = Path(os.environ.get("FIXTURE", "tests/fixtures/vin_diesel_skit.txt"))
    if not fixture.exists():
        print(f"Fixture not found: {fixture}", file=sys.stderr)
        sys.exit(2)

    raw = fixture.read_text(encoding="utf-8")

    # Empty in-memory corpus — mine() reads existing anchors from
    # corpus.anchors only for prompt-context purposes; an empty store
    # mimics a first-time mining run.
    corpus = CorpusStore()
    miner = NarrativeMiner(corpus)

    print(f"Fixture:  {fixture}")
    print(f"Length:   {len(raw)} chars")
    print(f"Model:    {ollama.CHAT_MODEL}")
    print(f"Extract num_ctx:    {ollama._EXTRACT_NUM_CTX}")
    import os as _os
    print(f"OLLAMA_NUM_PARALLEL (env): {_os.environ.get('OLLAMA_NUM_PARALLEL', '<unset, daemon default>')}")
    print(f"PS_MINING_PARALLEL: {_os.environ.get('PS_MINING_PARALLEL', '2 (default)')}")
    print()

    t0 = time.perf_counter()
    result = await miner.mine(raw, source_label=fixture.name)
    elapsed = time.perf_counter() - t0

    # ── Dict-shape contract check ────────────────────────────────────
    expected_keys = {
        "format", "segments", "chunks_analyzed", "topic_distribution",
        "proposals", "edges", "proposal_count", "edge_count", "source_label",
    }
    missing = expected_keys - set(result.keys())
    extra = set(result.keys()) - expected_keys - {"error"}
    print(f"=== CONTRACT ===")
    print(f"  expected keys present: {'YES' if not missing else 'NO — missing ' + str(missing)}")
    if extra:
        print(f"  extra keys (informational): {extra}")
    print(f"  format = {result.get('format')!r} (expected 'narrative')")
    print()

    # ── Counts ──────────────────────────────────────────────────────
    proposals = result.get("proposals", [])
    edges = result.get("edges", [])
    by_type: dict[str, int] = {}
    for p in proposals:
        by_type[p.get("type", "?")] = by_type.get(p.get("type", "?"), 0) + 1
    edge_by_type: dict[str, int] = {}
    for e in edges:
        edge_by_type[e.get("type", "?")] = edge_by_type.get(e.get("type", "?"), 0) + 1

    print(f"=== COUNTS ===")
    print(f"  segments:   {result.get('segments')}")
    print(f"  proposals:  {len(proposals)} total")
    for t, c in sorted(by_type.items()):
        print(f"    - {t}: {c}")
    print(f"  edges:      {len(edges)} total")
    for t, c in sorted(edge_by_type.items()):
        print(f"    - {t}: {c}")
    print()

    # ── Anchors ──
    anchors = [p for p in proposals if p.get("type") == "anchor"]
    print(f"=== ANCHORS ({len(anchors)}) ===")
    for a in anchors[:30]:
        aliases = a.get("aliases") or []
        suf = f"  aliases={aliases}" if aliases else ""
        print(f"  - \"{a.get('canonical_phrase')}\" (conf={a.get('confidence')}){suf}")
    if len(anchors) > 30:
        print(f"  ... +{len(anchors)-30} more")
    print()

    # ── Bundles ──
    bundles = [p for p in proposals if p.get("type") == "bundle"]
    print(f"=== BUNDLES ({len(bundles)}) ===")
    for b in bundles:
        members = b.get("aliases") or []
        print(f"  - \"{b.get('label')}\" (conf={b.get('confidence')}, {len(members)} members)")
        for m in members[:6]:
            print(f"      · {m}")
        if len(members) > 6:
            print(f"      · ... +{len(members)-6} more")
    print()

    # ── Slabs ──
    slabs = [p for p in proposals if p.get("type") == "slab"]
    print(f"=== SLABS ({len(slabs)}) ===")
    for s in slabs:
        title = s.get("title", "(untitled)")
        text = s.get("canonical_text", "")
        text_short = text if len(text) <= 200 else text[:200] + "…"
        print(f"  - \"{title}\" (conf={s.get('confidence')})")
        print(f"      text: {text_short}")
    print()

    # ── Edges ──
    # ASCII arrow only — Windows cmd/PowerShell default code page (cp1252)
    # can't encode the unicode arrow. Use "->" so the script runs cleanly
    # on any terminal without needing chcp or PYTHONIOENCODING.
    print(f"=== EDGES ({len(edges)}) ===")
    for e in edges[:30]:
        print(f"  - {e.get('type')}: \"{e.get('from')}\" -> \"{e.get('to')}\" (conf={e.get('confidence')})")
    if len(edges) > 30:
        print(f"  ... +{len(edges)-30} more")
    print()

    # ── Timing ──
    print(f"=== TIMING ===")
    print(f"  total wallclock: {elapsed:.1f}s for {result.get('segments')} segments")
    if result.get('segments'):
        print(f"  per-segment avg: {elapsed/result['segments']:.1f}s")
    print()

    # ── Pass 3 connectivity check (LINKS edges ratio) ──
    links = [e for e in edges if e.get("type") == "LINKS"]
    if slabs:
        avg_refs = len(links) / len(slabs)
        print(f"=== STRUCTURAL CONNECTIVITY ===")
        print(f"  LINKS edges:           {len(links)}")
        print(f"  avg LINKS per slab:    {avg_refs:.1f}")
        print(f"  (baseline pre-refactor avg: 0.0 — slabs had no explicit anchor links)")


if __name__ == "__main__":
    asyncio.run(main())
