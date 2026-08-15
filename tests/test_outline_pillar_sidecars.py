"""Regression tests for rebuilding pillars from universal-miner sidecars."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from src.services.outline_miner import (
    Section,
    _collect_slab_records,
    _pillars_from_slab_records,
    _normalize_sidecar_topic,
    _sidecar_source_order,
    build_outline_tree_recursive,
    build_pillars_recursive,
)


def test_list_topic_normalizes_to_outline_path() -> None:
    assert _normalize_sidecar_topic([
        "12", "Safety, Constraints, and Failure Envelope", "12.2 Safety philosophy"
    ]) == "12 / Safety, Constraints, and Failure Envelope / 12.2 Safety philosophy"
    assert _normalize_sidecar_topic(["segment 2", "4.5 / 4.6 / 4.7"]) == (
        "segment 2 / 4.5 + 4.6 + 4.7"
    )
    assert _normalize_sidecar_topic(" Part / Chapter ") == "Part / Chapter"


def test_universal_temp_id_provides_stable_order() -> None:
    assert _sidecar_source_order({"temp_id": "s2_candidate_009"}, 99) == 2_000_009
    assert _sidecar_source_order({"temp_id": "s4_slab_030"}, 99) == 4_000_030
    assert _sidecar_source_order({"source_pairs": [17]}, 99) == 17


def test_collect_accepts_list_topics_and_filters_uncommitted() -> None:
    payloads = {
        "slab_a_raw.json": {
            "type": "slab",
            "temp_id": "s2_candidate_009",
            "source_topic": ["Part I", "4.4 Pool dimensions"],
        },
        "slab_b_raw.json": {
            "type": "slab",
            "temp_id": "s1_candidate_001",
            "source_topic": ["Part I", "Uncommitted"],
        },
    }
    original_read_text = Path.read_text

    def fake_read_text(path: Path, *args, **kwargs) -> str:
        return json.dumps(payloads[path.name])

    Path.read_text = fake_read_text
    try:
        store = SimpleNamespace(slabs={"slab_a": object()})
        records = _collect_slab_records(
            store, ["slab_b_raw.json", "slab_a_raw.json"]
        )
    finally:
        Path.read_text = original_read_text

    assert records == [{
        "id": "slab_a",
        "topic": "Part I / 4.4 Pool dimensions",
        "cross": [],
        "order": 2_000_009,
    }]


def test_content_bearing_parent_keeps_members_and_children() -> None:
    parent = Section(
        level=1, label="Parent", char_start=0, body_start=0,
        char_end=0, parent_idx=-1, text="",
    )
    child = Section(
        level=2, label="Child", char_start=1, body_start=0,
        char_end=0, parent_idx=0, text="",
    )
    sections = [parent, child]
    tree = build_outline_tree_recursive(
        sections=sections,
        leaf_ids={id(parent), id(child)},
        summaries={},
        slab_titles_by_path={},
        slab_ids_by_path={"Parent": ["slab_parent"], "Parent / Child": ["slab_child"]},
        cross_edges={},
    )

    assert tree[0]["slab_ids"] == ["slab_parent"]
    assert tree[0]["children"][0]["slab_ids"] == ["slab_child"]

    store = SimpleNamespace(
        slabs={
            "slab_parent": SimpleNamespace(title="Parent slab"),
            "slab_child": SimpleNamespace(title="Child slab"),
        },
        pillars={},
        validate=lambda: [],
        save=lambda: None,
    )
    report = build_pillars_recursive(store, tree, origin="test")
    parent_pillar = next(p for p in store.pillars.values() if p.label == "Parent")
    child_pillar = next(p for p in store.pillars.values() if p.label == "Child")

    assert report["members_resolved"] == 2
    assert parent_pillar.members == ["slab_parent"]
    assert parent_pillar.children == [child_pillar.id]
    assert child_pillar.parent == parent_pillar.id

def test_coarse_overlay_uses_sequence_for_detail() -> None:
    store = SimpleNamespace(
        slabs={
            "a": SimpleNamespace(title="A", canonical_text="A text"),
            "b": SimpleNamespace(title="B", canonical_text="B text"),
            "c": SimpleNamespace(title="C", canonical_text="C text"),
        },
        pillars={},
        edges={},
        validate=lambda: [],
        save=lambda: None,
    )
    records = [
        {"id": "a", "topic": "Part I / Chapter 1 / Detail", "cross": [], "order": 1},
        {"id": "b", "topic": "Part I / Chapter 2 / Detail", "cross": [], "order": 2},
        {"id": "c", "topic": "Part II / Chapter 3 / Detail", "cross": [], "order": 3},
    ]
    report = asyncio.run(_pillars_from_slab_records(
        store, records, "test", "test", regenerate_summaries=False, max_depth=1,
    ))
    assert report["pillars_created"] == 2
    assert report["members_resolved"] == 3
    assert report["sequence_edges"] == 2
    assert len(store.edges) == 2
    assert sorted(p.label for p in store.pillars.values()) == ["Part I", "Part II"]

def test_cross_pillars_synthesized_when_sidecars_carry_none(monkeypatch) -> None:
    """AI Mine 2 emits no per-slab cross data; the rebuild should synthesise
    leaf-level cross-edges by embedding when summaries are regenerated."""
    import numpy as np
    from src.services import outline_miner as om

    # Two slabs on nearly-identical vectors (should cross-link) + one distant.
    vectors = {
        "Water flow and turbulence": [1.0, 0.0, 0.0],
        "Turbulence control layers": [0.99, 0.01, 0.0],
        "Scent dispersion timing": [0.0, 0.0, 1.0],
        # pillar-summary embeds (label. summary) resolve to their leaf vector
        "A": [1.0, 0.0, 0.0], "B": [0.99, 0.01, 0.0], "C": [0.0, 0.0, 1.0],
    }

    async def fake_embed(texts):
        out = []
        for t in texts:
            key = next((k for k in vectors if k.lower() in t.lower()), None)
            out.append(vectors.get(key, [0.0, 0.0, 0.0]))
        return out

    async def fake_summarize(label, items):
        return label  # summary text = the leaf label, enough to embed on

    monkeypatch.setattr(om.ollama, "embed", fake_embed)
    monkeypatch.setattr(om, "_summarize_pillar", fake_summarize)

    store = SimpleNamespace(
        slabs={
            "a": SimpleNamespace(title="A", canonical_text="x",
                                 description="Water flow and turbulence"),
            "b": SimpleNamespace(title="B", canonical_text="x",
                                 description="Turbulence control layers"),
            "c": SimpleNamespace(title="C", canonical_text="x",
                                 description="Scent dispersion timing"),
        },
        pillars={}, edges={}, validate=lambda: [], save=lambda: None,
    )
    # No cross data on any record (the AI Mine 2 case).
    records = [
        {"id": "a", "topic": "Flow", "cross": [], "order": 1},
        {"id": "b", "topic": "Turbulence", "cross": [], "order": 2},
        {"id": "c", "topic": "Scent", "cross": [], "order": 3},
    ]
    report = asyncio.run(_pillars_from_slab_records(
        store, records, "test", "test", regenerate_summaries=True,
    ))
    # Flow<->Turbulence should cross-link; Scent stays isolated.
    linked = [
        (p.label, ce.to_pillar)
        for p in store.pillars.values() for ce in (p.cross_edges or [])
    ]
    assert linked, "expected synthesised cross-edges, got none"
    assert all("Scent" not in a for a, _b in linked)


if __name__ == "__main__":
    test_list_topic_normalizes_to_outline_path()
    test_universal_temp_id_provides_stable_order()
    test_collect_accepts_list_topics_and_filters_uncommitted()
    test_content_bearing_parent_keeps_members_and_children()
    test_coarse_overlay_uses_sequence_for_detail()
    print("outline pillar sidecar tests passed")