"""Combined harness: mine a doc, then run synthesis on contested topics.

End-to-end validation that mining produces CONFLICTS / TENSIONS edges
AND that synthesis surfaces them in the dialectic tier. Operates
entirely in-memory — proposals from mining get directly stuffed into
a CorpusStore that the synthesis pass walks.

Run:
    python scripts/mine_then_synth.py <doc_path> <query>

Example:
    python scripts/mine_then_synth.py "Sovereign logic OS raw.txt" "tokenizer fragility"
    python scripts/mine_then_synth.py "Sovereign logic OS raw.txt" "neural vs symbolic"
"""
from __future__ import annotations

import asyncio
import sys
import time
import uuid
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.services.convo_miner import ConversationMiner
from src.services.corpus import CorpusStore
from src.services.synthesis import synthesize, SynthesisResult
from src.models.schemas import Anchor, Slab, KeyBundle, Edge, AnchorMatchPolicy, BundlePayload, AnchorMeta
from src.models.enums import EdgeType, SlabType


def _load_proposals_into_corpus(corpus: CorpusStore, mined: dict) -> tuple[int, int, int, int]:
    """Convert mined proposals + edges into committed-style corpus entries.

    Returns (n_anchors, n_slabs, n_bundles, n_edges) for reporting.
    Mining produces draft-stage objects; synthesis walks the committed
    corpus (Anchor / Slab / KeyBundle / Edge schemas). We materialise
    each mining proposal as the corresponding committed object so the
    synthesis pass has real data to walk. Label-keyed dedup is fine
    here — if two proposals share a phrase, the first one wins.
    """
    proposals = mined.get("proposals", [])
    edges = mined.get("edges", [])

    label_to_id: dict[str, str] = {}

    def _norm(s: str) -> str:
        return s.lower().strip()

    n_a = n_s = n_b = 0
    for p in proposals:
        ptype = p.get("type", "")
        text = (p.get("canonical_phrase") or p.get("title") or p.get("label") or "").strip()
        if not text:
            continue
        # Stable id from a hash-ish — collision-resistant for our scale
        nid = f"mined_{ptype}_{uuid.uuid4().hex[:8]}_v1"
        if ptype == "anchor":
            phrase = p.get("canonical_phrase", "").strip()
            if not phrase:
                continue
            corpus.anchors[nid] = Anchor(
                id=nid,
                canonical_phrase=phrase,
                aliases=list(p.get("aliases") or []),
                notes=p.get("justification", ""),
                match_policy=AnchorMatchPolicy(),
                meta=AnchorMeta(),
            )
            label_to_id[_norm(phrase)] = nid
            for al in p.get("aliases") or []:
                label_to_id.setdefault(_norm(al), nid)
            n_a += 1
        elif ptype == "slab":
            title = p.get("title", "").strip() or f"Slab {nid}"
            ctext = p.get("canonical_text", "").strip()
            if not ctext:
                continue
            corpus.slabs[nid] = Slab(
                id=nid,
                type=SlabType.REFERENCE,
                title=title,
                canonical_text=ctext,
                meta=AnchorMeta(),
            )
            label_to_id[_norm(title)] = nid
            n_s += 1
        elif ptype == "bundle":
            label = p.get("label", "").strip() or f"Bundle {nid}"
            members = list(p.get("aliases") or [])  # convo miner stashes members in aliases
            corpus.bundles[nid] = KeyBundle(
                id=nid,
                payload=BundlePayload(intent=[label] if label else ["(unlabeled)"]),
                meta=AnchorMeta(),
            )
            label_to_id[_norm(label)] = nid
            n_b += 1

    # Materialise edges with from/to resolved against the label index.
    n_e = 0
    for e in edges:
        etype_raw = (e.get("type") or "LINKS").upper()
        if etype_raw not in {"INVOKES", "SUPPORTS", "CONFLICTS", "TENSIONS",
                              "LINKS", "SEQUENCE", "PARENT_OF"}:
            etype_raw = "LINKS"
        from_label = (e.get("from") or e.get("from_label") or "").strip()
        to_label = (e.get("to") or e.get("to_label") or "").strip()
        from_id = label_to_id.get(_norm(from_label))
        to_id = label_to_id.get(_norm(to_label))
        if not from_id or not to_id or from_id == to_id:
            continue
        eid = f"mined_edge_{uuid.uuid4().hex[:8]}_v1"
        try:
            corpus.edges[eid] = Edge(
                id=eid,
                type=EdgeType(etype_raw),
                from_node=from_id,
                to_node=to_id,
                weight=float(e.get("confidence", 0.5)),
                confidence=float(e.get("confidence", 0.5)),
            )
            n_e += 1
        except Exception:
            continue

    return n_a, n_s, n_b, n_e


def _print_synth(result: SynthesisResult):
    print(f"=== SYNTHESIS RESULT for {result.query!r} ===")
    print(f"  budget used: {result.budget_used_tokens}/{result.budget_tokens} tokens")
    print(f"  seeds: {len(result.seeds)}")
    print(f"  tier1 supported: {len(result.tier1_supported)}")
    print(f"  tier1 conflicts: {len(result.tier1_conflicts)}")
    print(f"  tier2 supports:  {len(result.tier2_supports)}")
    print(f"  tier2 divergent: {len(result.tier2_divergent)}")
    print()

    if result.tier1_conflicts:
        print(f"  --- TIER 1 CONFLICTS ({len(result.tier1_conflicts)}) ---")
        for n in result.tier1_conflicts:
            print(f"    [{n.node_type}] {n.text[:100]} (conflicts_with={n.conflicts_with})")
        print()

    if result.seeds:
        print(f"  --- SEEDS ({len(result.seeds)}) ---")
        for n in result.seeds:
            print(f"    [{n.node_type}] {n.text[:100]} (conf={n.confidence:.2f})")
        print()

    if result.tier1_supported:
        print(f"  --- TIER 1 SUPPORTED (top 8 of {len(result.tier1_supported)}) ---")
        for n in result.tier1_supported[:8]:
            print(f"    [{n.node_type}] hop={n.hop_distance} via={n.via_edge}")
            print(f"      {n.text[:120]}")
        print()


async def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/mine_then_synth.py <doc_path> <query>", file=sys.stderr)
        sys.exit(2)

    doc_path = Path(sys.argv[1])
    query = sys.argv[2]
    if not doc_path.exists():
        print(f"Not found: {doc_path}", file=sys.stderr)
        sys.exit(2)

    raw = doc_path.read_text(encoding="utf-8", errors="replace")
    print(f"Doc:    {doc_path}")
    print(f"Length: {len(raw)} chars")
    print(f"Query:  {query!r}")
    print()

    # ── Mine ───────────────────────────────────────────────────────
    corpus = CorpusStore()
    miner = ConversationMiner(corpus)
    print("[1/3] Mining...")
    t0 = time.perf_counter()
    mined = await miner.mine(raw, source_label=doc_path.name, min_confidence=0.4)
    mine_secs = time.perf_counter() - t0
    edge_types = {}
    for e in mined.get("edges", []):
        edge_types[e.get("type")] = edge_types.get(e.get("type"), 0) + 1
    print(f"      {len(mined.get('proposals', []))} proposals, {len(mined.get('edges', []))} edges in {mine_secs:.1f}s")
    print(f"      edge types: {edge_types}")

    # ── Materialize into corpus ───────────────────────────────────
    print("[2/3] Materialising into corpus...")
    n_a, n_s, n_b, n_e = _load_proposals_into_corpus(corpus, mined)
    print(f"      {n_a} anchors, {n_s} slabs, {n_b} bundles, {n_e} edges (some edges may have been dropped due to unresolved labels)")

    # ── Diagnostic: dump corpus edges for debugging conflict-firing ──
    print(f"[2.5/3] Corpus edge state (for synthesis debugging):")
    edge_by_type: dict[str, int] = {}
    for e in corpus.edges.values():
        et = str(getattr(e.type, "value", e.type))
        edge_by_type[et] = edge_by_type.get(et, 0) + 1
    print(f"      edges by type in corpus: {edge_by_type}")
    # Show every CONFLICTS / TENSIONS edge with its endpoints' labels
    dialectic_edges = [
        e for e in corpus.edges.values()
        if str(getattr(e.type, "value", e.type)) in ("CONFLICTS", "TENSIONS")
    ]
    if dialectic_edges:
        print(f"      DIALECTIC edges in corpus:")
        for e in dialectic_edges:
            from_a = corpus.anchors.get(e.from_node)
            from_s = corpus.slabs.get(e.from_node)
            to_a = corpus.anchors.get(e.to_node)
            to_s = corpus.slabs.get(e.to_node)
            from_label = (from_a.canonical_phrase if from_a else (from_s.title if from_s else e.from_node))
            to_label = (to_a.canonical_phrase if to_a else (to_s.title if to_s else e.to_node))
            etype = str(getattr(e.type, "value", e.type))
            print(f"        {etype} {from_label!r} -> {to_label!r}")
    else:
        print(f"      (no dialectic edges in materialized corpus)")
    print()

    # ── Synthesise ─────────────────────────────────────────────────
    print(f"[3/3] Synthesising '{query}'...")
    t0 = time.perf_counter()
    result = await synthesize(query, corpus)
    synth_secs = time.perf_counter() - t0
    print(f"      done in {synth_secs:.1f}s")
    print()

    _print_synth(result)


if __name__ == "__main__":
    asyncio.run(main())
