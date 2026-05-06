"""Synthesis selection-algorithm demo.

Pure CLI harness — no chat, no LLM compose. Loads the configured
corpora, runs synthesize() against a natural-language query, and
pretty-prints the structured SynthesisResult so you can eyeball
which slabs / anchors / bundles got picked into each tier.

Run:
    python scripts/synth_demo.py "tell me about fusion energy"
    python scripts/synth_demo.py "post credits sequence" --collection podv1
    python scripts/synth_demo.py "neuro-symbolic architecture" --budget 14000

Tunables exposed as flags so you can sweep parameters without
re-editing source.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.services.corpus import CorpusRegistry
from src.services.synthesis import (
    synthesize,
    SynthesisResult,
    DEFAULT_HIGH_CONF_THRESHOLD,
    DEFAULT_BUDGET_TOKENS,
    DEFAULT_TIER1_FRACTION,
    DEFAULT_MAX_HOPS,
    DEFAULT_MAX_SEEDS,
    DEFAULT_MIN_SEED_COSINE,
    DEFAULT_MAX_TIER2_DIVERGENT,
)


def _truncate(s: str, n: int = 110) -> str:
    s = s.strip().replace("\n", " ")
    return s if len(s) <= n else s[:n] + "..."


def _print_node(n, prefix: str = "  "):
    extras = []
    if n.hop_distance is not None and n.hop_distance > 0:
        extras.append(f"hop={n.hop_distance}")
    if n.via_edge:
        extras.append(f"via={n.via_edge}")
    if n.cos_sim is not None:
        extras.append(f"cos={n.cos_sim:.2f}")
    if n.conflicts_with:
        extras.append(f"vs={n.conflicts_with}")
    extras.append(f"conf={n.confidence:.2f}")
    extras.append(f"~{n.approx_tokens}t")
    extra_str = " ".join(extras)
    print(f"{prefix}[{n.node_type}] {n.id}  ({extra_str})")
    print(f"{prefix}    {_truncate(n.text)}")


def _print_result(result: SynthesisResult):
    print()
    print("=" * 78)
    print(f"SYNTHESIS RESULT")
    print("=" * 78)
    print(f"  query:         {result.query!r}")
    print(f"  budget:        {result.budget_tokens} tokens")
    print(f"  used:          {result.budget_used_tokens} tokens "
          f"({100 * result.budget_used_tokens // max(1, result.budget_tokens)}%)")
    print(f"  seeds:         {len(result.seeds)}")
    print(f"  tier1 supp:    {len(result.tier1_supported)}")
    print(f"  tier1 conf:    {len(result.tier1_conflicts)}")
    print(f"  tier2 supp:    {len(result.tier2_supports)}")
    print(f"  tier2 div:     {len(result.tier2_divergent)}")
    total = (
        len(result.seeds) + len(result.tier1_supported) + len(result.tier1_conflicts)
        + len(result.tier2_supports) + len(result.tier2_divergent)
    )
    print(f"  TOTAL:         {total} nodes selected")
    print()
    print("=== UNSURFACED (would be told to LLM as 'corpus has more, acknowledge gap') ===")
    u = result.unsurfaced
    print(f"  tier1_supported_skipped:  {u.tier1_supported_skipped}")
    print(f"  conflicts_skipped:        {u.conflicts_skipped}")
    print(f"  tier2_supports_skipped:   {u.tier2_supports_skipped}")
    print(f"  tier2_divergent_skipped:  {u.tier2_divergent_skipped}")
    print(f"  extra_hops_not_walked:    {u.extra_hops_not_walked}  (placeholder)")

    # ── Seeds
    if result.seeds:
        print()
        print(f"=== SEEDS ({len(result.seeds)}) — embedding-matched starting points ===")
        for n in result.seeds:
            _print_node(n)

    # ── Tier 1 conflicts
    if result.tier1_conflicts:
        print()
        print(f"=== TIER 1 — CONFLICTS ({len(result.tier1_conflicts)}) "
              f"— dialectic surfacing ===")
        for n in result.tier1_conflicts:
            _print_node(n)

    # ── Tier 1 supported
    if result.tier1_supported:
        print()
        print(f"=== TIER 1 — SUPPORTED ({len(result.tier1_supported)}) "
              f"— high-conf reachable via SUPPORTS/INVOKES/LINKS/PARENT_OF ===")
        for n in result.tier1_supported:
            _print_node(n)

    # ── Tier 2 supports
    if result.tier2_supports:
        print()
        print(f"=== TIER 2 — SUPPORTS EXPANSION ({len(result.tier2_supports)}) "
              f"— lower-conf reachable, supporting detail ===")
        for n in result.tier2_supports:
            _print_node(n)

    # ── Tier 2 divergent
    if result.tier2_divergent:
        print()
        print(f"=== TIER 2 — DIVERGENT SEMANTIC ({len(result.tier2_divergent)}) "
              f"— embedding-similar but NOT graph-connected ===")
        for n in result.tier2_divergent:
            _print_node(n)

    print()


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Natural-language query")
    parser.add_argument("--collection", default=None,
                        help="Limit synthesis to a specific collection (default: merged across all)")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET_TOKENS,
                        help=f"Token budget (default {DEFAULT_BUDGET_TOKENS})")
    parser.add_argument("--high-conf", type=float, default=DEFAULT_HIGH_CONF_THRESHOLD,
                        help=f"Confidence threshold for tier 1 (default {DEFAULT_HIGH_CONF_THRESHOLD})")
    parser.add_argument("--max-hops", type=int, default=DEFAULT_MAX_HOPS,
                        help=f"Max BFS hops (default {DEFAULT_MAX_HOPS})")
    parser.add_argument("--max-seeds", type=int, default=DEFAULT_MAX_SEEDS,
                        help=f"Top-K seeds (default {DEFAULT_MAX_SEEDS})")
    parser.add_argument("--min-cos", type=float, default=DEFAULT_MIN_SEED_COSINE,
                        help=f"Min cosine for seeds + divergent (default {DEFAULT_MIN_SEED_COSINE})")
    parser.add_argument("--max-divergent", type=int, default=DEFAULT_MAX_TIER2_DIVERGENT,
                        help=f"Cap on tier-2 divergent matches (default {DEFAULT_MAX_TIER2_DIVERGENT})")
    parser.add_argument("--tier1-frac", type=float, default=DEFAULT_TIER1_FRACTION,
                        help=f"Tier 1 budget fraction (default {DEFAULT_TIER1_FRACTION})")
    args = parser.parse_args()

    # Load corpus
    registry = CorpusRegistry()
    errors = registry.load_all()
    for cid, errs in errors.items():
        if errs:
            print(f"[CORPUS] '{cid}' validation errors: {len(errs)}", file=sys.stderr)
    if args.collection:
        store = registry.get_store(args.collection)
        if not store:
            print(f"Collection {args.collection!r} not found. Available: "
                  f"{list(registry.collections.keys())}", file=sys.stderr)
            sys.exit(2)
        corpus = store
    else:
        corpus = registry.merged

    print(f"[CORPUS] {len(corpus.anchors)} anchors, {len(corpus.slabs)} slabs, "
          f"{len(corpus.bundles)} bundles, {len(corpus.edges)} edges")

    result = await synthesize(
        args.query, corpus,
        budget_tokens=args.budget,
        high_conf_threshold=args.high_conf,
        tier1_fraction=args.tier1_frac,
        max_hops=args.max_hops,
        max_seeds=args.max_seeds,
        min_seed_cosine=args.min_cos,
        max_tier2_divergent=args.max_divergent,
    )
    _print_result(result)


if __name__ == "__main__":
    asyncio.run(main())
