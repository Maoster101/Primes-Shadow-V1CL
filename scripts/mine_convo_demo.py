"""Convo miner demo — feed a chat-export-style doc through the
non-narrative mining pipeline and dump proposals + edges.

Specifically validates that the new CONFLICTS / TENSIONS edge types
make it through after the convo_miner edge prompt fix.

Run:
    python scripts/mine_convo_demo.py "Sovereign logic OS raw.txt"
"""
from __future__ import annotations

import asyncio
import sys
import time
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.services.convo_miner import ConversationMiner
from src.services.corpus import CorpusStore
from src.services import ollama


async def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/mine_convo_demo.py <path>", file=sys.stderr)
        sys.exit(2)
    src_path = Path(sys.argv[1])
    if not src_path.exists():
        print(f"Not found: {src_path}", file=sys.stderr)
        sys.exit(2)

    raw = src_path.read_text(encoding="utf-8", errors="replace")
    print(f"Source:    {src_path}")
    print(f"Length:    {len(raw)} chars")
    print(f"Model:     {ollama.CHAT_MODEL}")
    print(f"Extract num_ctx: {ollama._EXTRACT_NUM_CTX}")
    print()

    corpus = CorpusStore()  # empty in-memory — we just want extraction output
    miner = ConversationMiner(corpus)

    t0 = time.perf_counter()
    result = await miner.mine(
        raw_text=raw,
        source_label=src_path.name,
        min_confidence=0.4,
    )
    elapsed = time.perf_counter() - t0

    print(f"=== TIMING ===")
    print(f"  format detected: {result.get('format')}")
    print(f"  exchanges:       {result.get('exchanges')}")
    print(f"  pairs:           {result.get('pairs')}")
    print(f"  chunks analyzed: {result.get('chunks_analyzed')}")
    print(f"  wallclock:       {elapsed:.1f}s")
    print()

    proposals = result.get("proposals", [])
    edges = result.get("edges", [])

    by_type = Counter(p.get("type") for p in proposals)
    print(f"=== PROPOSALS ({len(proposals)}) ===")
    for t, c in sorted(by_type.items()):
        print(f"  {t}: {c}")
    print()

    edge_by_type = Counter(e.get("type") for e in edges)
    print(f"=== EDGES ({len(edges)}) ===")
    for t, c in sorted(edge_by_type.items()):
        print(f"  {t}: {c}")
    print()

    # Show all CONFLICTS + TENSIONS edges in detail — these are the
    # NEW capability we're testing
    dialectic = [e for e in edges if e.get("type") in ("CONFLICTS", "TENSIONS")]
    if dialectic:
        print(f"=== DIALECTIC EDGES ({len(dialectic)}) — the new capability ===")
        for e in dialectic:
            print(f"  {e.get('type'):10} {e.get('from')!r}  ->  {e.get('to')!r}")
            j = (e.get('justification') or '').strip()
            if j:
                print(f"             {j[:200]}")
            print(f"             confidence={e.get('confidence')}")
        print()
    else:
        print(f"=== DIALECTIC EDGES (0) ===")
        print(f"  No CONFLICTS or TENSIONS edges produced.")
        print(f"  Either the doc has no contested concepts the model could detect,")
        print(f"  or the prompt change didn't take effect, or the source needs more")
        print(f"  explicit disagreement language for the model to identify.")
        print()

    # Show ALL anchors so we can spot foils ranked low
    print(f"=== ALL ANCHORS ({sum(1 for p in proposals if p.get('type')=='anchor')}) ===")
    anchors = [p for p in proposals if p.get("type") == "anchor"]
    for a in anchors:
        aliases = a.get('aliases') or []
        suf = f"  aliases={aliases}" if aliases else ""
        print(f"  [{a.get('confidence')}] {a.get('canonical_phrase')!r}{suf}")
    print()

    print(f"=== ALL SLABS ({sum(1 for p in proposals if p.get('type')=='slab')}) ===")
    slabs = [p for p in proposals if p.get("type") == "slab"]
    for s in slabs:
        title = s.get("title", "(untitled)")
        text = s.get("canonical_text", "")
        text_short = text if len(text) <= 140 else text[:140] + "..."
        print(f"  [{s.get('confidence')}] {title!r}")
        print(f"        {text_short}")
    print()

    # Show ALL edges (we already saw aggregates earlier)
    print(f"=== ALL EDGES ({len(edges)}) ===")
    for e in edges:
        print(f"  {e.get('type'):10} {e.get('from')!r} -> {e.get('to')!r} (conf={e.get('confidence')})")


if __name__ == "__main__":
    asyncio.run(main())
