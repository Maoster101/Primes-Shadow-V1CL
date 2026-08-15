"""Rebuild deterministic SEQUENCE edges for committed slabs from mining sidecars."""
from __future__ import annotations

import argparse
import glob
import hashlib
from pathlib import Path

from src.models.enums import EdgeType
from src.models.schemas import Edge
from src.services.corpus import CorpusStore
from src.services.outline_miner import _collect_slab_records


def _edge_id(source: str, target: str) -> str:
    digest = hashlib.sha1(f"{source}\0{target}".encode("utf-8")).hexdigest()[:12]
    return f"edge_sequence_{digest}_v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    store = CorpusStore(args.corpus, collection_id=args.corpus.name)
    load_errors = store.load()
    if load_errors:
        raise SystemExit(f"Corpus invalid before rebuild: {load_errors}")
    records = _collect_slab_records(
        store, glob.glob(str(args.session_dir / "*_raw.json"))
    )
    records.sort(key=lambda row: (row["order"], row["id"]))
    ordered_ids = [row["id"] for row in records]
    target_ids = set(ordered_ids)

    removed = [
        edge_id for edge_id, edge in store.edges.items()
        if edge.type == EdgeType.SEQUENCE
        and edge.from_node in target_ids and edge.to_node in target_ids
    ]
    additions = []
    for source, target in zip(ordered_ids, ordered_ids[1:]):
        additions.append(Edge(
            id=_edge_id(source, target),
            type=EdgeType.SEQUENCE,
            **{"from": source, "to": target},
            weight=0.75,
            confidence=0.98,
            justification="Deterministic order of consecutive committed slabs in the source.",
        ))

    if args.apply:
        for edge_id in removed:
            store.edges.pop(edge_id, None)
        for edge in additions:
            store.edges[edge.id] = edge
        errors = store.validate()
        if errors:
            raise SystemExit(f"Corpus invalid after rebuild: {errors}")
        store.save()

    print({
        "mode": "apply" if args.apply else "dry-run",
        "slabs_ordered": len(ordered_ids),
        "sequence_edges_removed": len(removed),
        "sequence_edges_added": len(additions),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())