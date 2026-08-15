"""Focused regressions for scoped hybrid slab retrieval."""
from __future__ import annotations

import asyncio

from src.models.schemas import Slab
from src.services import slab_matcher as slab_matcher_module
from src.services.corpus import CorpusStore
from src.services.slab_matcher import SlabMatcher


def _store() -> CorpusStore:
    store = CorpusStore(collection_id="test")
    slabs = [
        Slab(
            id="dimensions",
            title="Pool dimensions and sloped floor",
            canonical_text=(
                "Internal pool dimensions are 3.0 m length and 2.0 m width; "
                "depth slopes from 0.45 m to 1.55 m."
            ),
        ),
        Slab(
            id="salt_assumption",
            title="Explicit design assumptions",
            canonical_text="Baseline water is freshwater; bubbles are used rather than salt.",
        ),
        Slab(
            id="generic_water",
            title="User control of water sound",
            canonical_text="The user can control water sound during a session.",
        ),
        Slab(
            id="outside_scope",
            title="Dimensions from another collection",
            canonical_text="Dimensions are 9.0 m by 9.0 m.",
        ),
    ]
    store.slabs = {slab.id: slab for slab in slabs}
    return store


def test_exact_fact_lexical_recall() -> None:
    store = _store()
    matcher = SlabMatcher(store)
    scope = {"dimensions", "salt_assumption", "generic_water"}

    dimensions = matcher.lexical_top_k_for(
        "dimensions of the pod as per spec in podv9", id_filter=scope,
    )
    assert dimensions[0][0] == "dimensions"
    assert all(sid != "outside_scope" for sid, _score in dimensions)

    salt = matcher.lexical_top_k_for(
        "salt water configuration for the pod based on collection podv9",
        id_filter=scope,
    )
    assert salt[0][0] == "salt_assumption"


def test_fusion_accepts_lexical_only_candidates() -> None:
    matcher = SlabMatcher(_store())  # deliberately cold dense cache
    hits = asyncio.run(matcher.fused_top_k_for("pool dimensions", max_k=3))
    assert hits
    assert hits[0][0] == "dimensions"
    assert hits[0][2]["dense"] == 0.0
    assert hits[0][2]["lexical"] > 0.0


def test_incremental_warm_embeds_only_new_slabs() -> None:
    store = _store()
    matcher = SlabMatcher(store)
    calls: list[list[str]] = []
    original_embed = slab_matcher_module.ollama.embed

    async def fake_embed(texts: list[str]):
        calls.append(list(texts))
        return [[float(i + 1), 1.0] for i, _text in enumerate(texts)]

    slab_matcher_module.ollama.embed = fake_embed
    try:
        asyncio.run(matcher.warm_cache())
        assert len(calls[-1]) == 4
        assert calls[-1][0].startswith("pool dimensions")

        store.slabs["new_slab"] = Slab(
            id="new_slab", title="New fact", canonical_text="A newly promoted fact."
        )
        asyncio.run(matcher.warm_cache(only_new=True))
        assert calls[-1] == ["new fact\na newly promoted fact."]
        assert "new_slab" in matcher._embed_cache
    finally:
        slab_matcher_module.ollama.embed = original_embed


if __name__ == "__main__":
    test_exact_fact_lexical_recall()
    test_fusion_accepts_lexical_only_candidates()
    test_incremental_warm_embeds_only_new_slabs()
    print("slab retrieval tests passed")