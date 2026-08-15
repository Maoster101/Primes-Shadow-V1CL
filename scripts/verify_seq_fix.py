"""Re-mine the vin diesel skit through the narrative miner and dump the
SEQUENCE chain in narrative order. Validates the intra_segment_order
tiebreaker fix from commit 19338fc.

Pre-fix expectation (broken): 5 slabs at "beat 0" in arbitrary order,
with "Concluding Reflection" landing somewhere in the middle.
Post-fix expectation: 5 slabs at "beat 0" in faithful emission order,
chain reads top-to-bottom of the source narrative.

Run:
    python scripts/verify_seq_fix.py tmp_vindiesel_remine.txt
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.services.corpus import CorpusStore
from src.services.narrative_miner import NarrativeMiner


async def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/verify_seq_fix.py <doc>", file=sys.stderr)
        sys.exit(2)
    src = Path(sys.argv[1])
    raw = src.read_text(encoding="utf-8", errors="replace")
    print(f"Source: {src}  ({len(raw)} chars)")

    miner = NarrativeMiner(CorpusStore())
    t0 = time.perf_counter()
    result = await miner.mine(raw, source_label=src.name)
    elapsed = time.perf_counter() - t0

    proposals = result.get("proposals", [])
    edges = result.get("edges", [])
    n_anchor = sum(1 for p in proposals if p.get("type") == "anchor")
    n_slab = sum(1 for p in proposals if p.get("type") == "slab")
    n_bundle = sum(1 for p in proposals if p.get("type") == "bundle")
    print(
        f"Mined in {elapsed:.1f}s | anchors={n_anchor} slabs={n_slab} "
        f"bundles={n_bundle} edges={len(edges)}"
    )
    print()

    # Pull out slabs in narrative order and the SEQUENCE edges
    slabs = [p for p in proposals if p.get("type") == "slab"]
    print("Slabs (in mined order):")
    for i, s in enumerate(slabs):
        sp = s.get("source_pairs", [])
        iso = s.get("intra_segment_order", "?")
        print(f"  [{i}] seg={sp[0] if sp else '?':>2}/{iso} | {s.get('title', '')[:60]}")
    print()

    seq = [e for e in edges if e.get("type") == "SEQUENCE"]
    print(f"SEQUENCE edges ({len(seq)}):")
    for e in seq:
        f = (e.get("from") or e.get("from_label") or "")[:55]
        t = (e.get("to") or e.get("to_label") or "")[:55]
        j = (e.get("justification") or "")[:80]
        print(f"  {f:>55}  ->  {t:<55}")
        print(f"  {' ' * 55}      {j}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
