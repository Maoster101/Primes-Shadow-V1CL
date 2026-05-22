"""Clean-remine orchestrator: source document → fresh corpus with pillars.

End-to-end pipeline:
  1. narrative_miner.mine_file(source) → proposals + edges + inline-anchor map
  2. Drop "bundle" proposals (we replace cosine-clustering bundles with
     directive_extractor below)
  3. Build Slab + Anchor corpus objects from remaining proposals,
     respecting the inline-anchor demotion plan from anchor consolidation
  4. Resolve edge labels → IDs, write typed Edge objects
  5. directive_extractor.run(slabs) → ProposedBundle list → KeyBundle records
  6. pillar_extractor.run(slabs) → ProposedPillar list → PillarDefinition records
  7. Emit slab→bundle LINKS edges from bundle.depends_on
  8. Save everything to a new collection (default: podv5)

The Anchor/Bundle/Slab content graph is mined and consolidated in step 1-4.
The pillar overlay (Tier-0) is added in step 6 without modifying the
content graph below — pillars reference slabs by id via ``members``.

Run:
    python scripts/remine_to_podv5.py <source_path> [<target_collection>]

Example:
    python scripts/remine_to_podv5.py \\
        "input documents/Aquatic_Sensory_Environment_Canonical_Reference.md" podv5
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.enums import EdgeType, NodeType, SlabLifecycleStatus, SlabType
from src.models.schemas import (
    Anchor, AnchorMeta, BundlePayload, Edge, InlineAnchor, KeyBundle,
    PillarDefinition, ProvenanceRef, Slab, SlabLinks,
)
from src.services import directive_extractor, pillar_extractor
from src.services.corpus import CORPORA_ROOT, CorpusStore
from src.services.narrative_miner import NarrativeMiner

logger = logging.getLogger("remine")


# ─── Helpers ─────────────────────────────────────────────────────────────


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(text: str, max_len: int = 30) -> str:
    return _SLUG_RE.sub("_", text.lower()).strip("_")[:max_len] or "x"


def _stable_id(prefix: str, text: str) -> str:
    """Deterministic id derived from a hash of the text — re-mining
    the same content produces the same ids, which makes diffs sane.
    """
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{h}_v1"


def _label_to_id(label: str, lookup: dict[str, str]) -> str | None:
    """Resolve a from_label/to_label to its corpus id.

    Edges from narrative_miner reference proposals by their CANONICAL
    LABEL (canonical_phrase for anchors, title for slabs, label for
    bundles). We build a label→id index up front, then resolve each
    edge endpoint against it. Case-insensitive — the miner sometimes
    varies surface forms across the proposal and edge passes.
    """
    if not label:
        return None
    direct = lookup.get(label) or lookup.get(label.strip())
    if direct:
        return direct
    return lookup.get(label.lower().strip())


def _convert_proposed_bundle(pb: directive_extractor.ProposedBundle) -> KeyBundle:
    """Map ProposedBundle (LLM output) → KeyBundle (schema record)."""
    return KeyBundle(
        id=pb.id,
        payload=BundlePayload(
            intent=pb.intent or [pb.label],
            invariants=pb.invariants,
            non_assumptions=pb.non_assumptions,
            warnings=pb.warnings,
            heuristics=pb.heuristics,
            rules=pb.rules,
            canonical_quote_handles=pb.canonical_quote_handles,
        ),
        depends_on=pb.depends_on,
        supports=pb.supports,
        meta=AnchorMeta(version="v1"),
    )


def _convert_proposed_pillar(pp: pillar_extractor.ProposedPillar) -> PillarDefinition:
    return PillarDefinition(
        id=pp.id,
        label=pp.label,
        summary=pp.summary,
        members=pp.members,
        children=pp.children,
        parent=pp.parent,
        pillar_role=pp.pillar_role,
        origin=pp.origin,
        lifecycle_status=SlabLifecycleStatus.ACTIVE,
        meta=AnchorMeta(version="v1"),
    )


def _edge_type_or_none(s: str) -> EdgeType | None:
    """Convert a free-text edge type string to the EdgeType enum.

    narrative_miner emits raw type strings (INVOKES, SEQUENCE, LINKS,
    CONFLICTS, TENSIONS, SUPPORTS, PARENT_OF, REGULATES). We accept the
    EdgeType-valued ones and drop unknown types (REGULATES has no
    PillarsShadow EdgeType equivalent today).
    """
    try:
        return EdgeType(s.strip().upper())
    except ValueError:
        return None


# ─── Steps ───────────────────────────────────────────────────────────────


_MINE_CACHE_DIR = _PROJECT_ROOT / ".tmp" / "remine_cache"


async def step1_mine_source(source_path: Path) -> dict:
    """Call narrative_miner against the source. Returns its raw dict.

    Caches the result under .tmp/remine_cache/<src_stem>.json — re-runs
    of the orchestrator skip the ~30-minute Pass 1+3 work. Delete the
    cache file to force a fresh mine.
    """
    import json as _json
    _MINE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _MINE_CACHE_DIR / f"{source_path.stem}_mine.json"

    if cache_path.exists():
        logger.info("Step 1: using cached mine result from %s", cache_path)
        with open(cache_path, encoding="utf-8") as f:
            cached = _json.load(f)
        logger.info(
            "Step 1 cache hit: %d proposals, %d edges, %d segments",
            len(cached.get("proposals", [])),
            len(cached.get("edges", [])), cached.get("segments", 0),
        )
        return cached

    logger.info("Step 1: narrative miner — %s", source_path)
    t0 = time.time()
    miner = NarrativeMiner(corpus=None)
    result = await miner.mine_file(source_path)
    elapsed = time.time() - t0
    logger.info(
        "Step 1 done in %.1fs: %d proposals, %d edges, %d segments",
        elapsed, len(result.get("proposals", [])),
        len(result.get("edges", [])), result.get("segments", 0),
    )
    # Persist for subsequent runs — cache is keyed by source stem so
    # rerunning against the same source path replays cheaply.
    with open(cache_path, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info("Step 1 cached at %s", cache_path)
    return result


def step2_build_content_corpus(
    mine_result: dict, target_collection: str,
) -> tuple[CorpusStore, dict[str, str]]:
    """Convert mine proposals into Slab + Anchor corpus objects.

    Drops bundle proposals (replaced by directive_extractor downstream).
    Returns the populated store + a label→id lookup that step 3 uses
    to resolve edge endpoints.
    """
    logger.info("Step 2: building Slab + Anchor objects")
    root = CORPORA_ROOT / target_collection
    objects = root / "objects"
    objects.mkdir(parents=True, exist_ok=True)
    (root / "state").mkdir(parents=True, exist_ok=True)
    # Seed empty yamls so load() works on the freshly-created collection
    import yaml
    for fname in ["anchors.yaml", "slabs.yaml", "key_bundles.yaml",
                  "edges.yaml", "gates.yaml"]:
        path = objects / fname
        if not path.exists():
            path.write_text("[]\n", encoding="utf-8")

    store = CorpusStore(root=root, collection_id=target_collection)
    # We intentionally do NOT load — this is a fresh collection. Build
    # the in-memory state directly and save at the end.

    label_to_id: dict[str, str] = {}
    inline_map = mine_result.get("_inline_anchors_per_slab") or {}
    proposals = mine_result.get("proposals") or []

    # Pass A: anchors first so slabs can reference them
    for prop in proposals:
        ptype = prop.get("type")
        if ptype != "anchor":
            continue
        phrase = (prop.get("canonical_phrase") or "").strip()
        if not phrase:
            continue
        aid = _stable_id("mined_anchor", phrase.lower())
        if aid in store.anchors:
            continue
        anchor = Anchor(
            id=aid,
            canonical_phrase=phrase,
            aliases=list(prop.get("aliases") or []),
            notes=(prop.get("justification") or "")[:240],
            meta=AnchorMeta(version="v1"),
        )
        store.anchors[aid] = anchor
        label_to_id[phrase] = aid
        label_to_id[phrase.lower()] = aid

    # Pass B: slabs
    for prop in proposals:
        ptype = prop.get("type")
        if ptype != "slab":
            continue
        title = (prop.get("title") or "").strip()
        text = (prop.get("canonical_text") or "").strip()
        if not text:
            continue
        sid = _stable_id("mined_slab", (title + text).lower())
        if sid in store.slabs:
            continue
        # Inline anchors come from the consolidation plan, keyed by
        # the slab's title. Convert each dict into an InlineAnchor.
        raw_inlines = inline_map.get(title) or inline_map.get(sid) or []
        inline_records = [
            InlineAnchor(
                id=ia.get("id") or _stable_id("inline_anchor",
                                              ia.get("canonical_phrase", "")),
                canonical_phrase=ia.get("canonical_phrase", ""),
                aliases=list(ia.get("aliases") or []),
                notes=(ia.get("notes") or "")[:240],
                confidence=float(ia.get("confidence", 0.5)),
            )
            for ia in raw_inlines
            if ia.get("canonical_phrase")
        ]
        slab = Slab(
            id=sid,
            title=title or text[:60],
            canonical_text=text,
            links=SlabLinks(anchors_inline=inline_records),
            type=SlabType.REFERENCE,
            provenance_refs=[ProvenanceRef(
                ref_id=f"WB_{sid}",
                store="workbench",
                type="conversation_segment",
                note=(prop.get("justification") or "")[:240],
            )],
            meta=AnchorMeta(version="v1"),
        )
        store.slabs[sid] = slab
        label_to_id[title] = sid
        label_to_id[title.lower()] = sid

    logger.info(
        "Step 2 done: %d anchors, %d slabs (bundles deferred to directive pass)",
        len(store.anchors), len(store.slabs),
    )
    return store, label_to_id


def step3_emit_mined_edges(
    store: CorpusStore, mine_result: dict, label_to_id: dict[str, str],
) -> None:
    """Convert narrative_miner edges (label-keyed) into typed Edge records."""
    logger.info("Step 3: converting mined edges to typed Edge records")
    dropped_endpoint = 0
    dropped_type = 0
    edges_added = 0
    for raw in (mine_result.get("edges") or []):
        et = _edge_type_or_none(raw.get("type", ""))
        if et is None:
            dropped_type += 1
            continue
        from_id = _label_to_id(raw.get("from", ""), label_to_id)
        to_id = _label_to_id(raw.get("to", ""), label_to_id)
        if not from_id or not to_id:
            dropped_endpoint += 1
            continue
        conf = float(raw.get("confidence") or 0.5)
        from_short = from_id.replace("mined_", "")[:20]
        to_short = to_id.replace("mined_", "")[:20]
        eid = f"edge_{et.value.lower()}_{from_short}_{to_short}_v1"
        if eid in store.edges:
            continue
        store.edges[eid] = Edge(
            id=eid,
            type=et,
            from_node=from_id,  # type: ignore[arg-type]
            to_node=to_id,      # type: ignore[arg-type]
            weight=conf,
            confidence=conf,
            justification=(raw.get("justification") or "")[:240] or None,
        )
        edges_added += 1
    logger.info(
        "Step 3 done: %d edges added (dropped: %d unknown-type, %d unresolved-endpoint)",
        edges_added, dropped_type, dropped_endpoint,
    )


async def step4_directive_pass(store: CorpusStore) -> None:
    """Run directive_extractor against the freshly-built slabs."""
    logger.info("Step 4: directive extraction over %d slabs", len(store.slabs))
    slabs = list(store.slabs.values())
    result = await directive_extractor.run(slabs, stage2_batch_size=80, dedupe=True)
    logger.info(
        "Step 4 done: %d extractions, %d bundles",
        len(result.extractions), len(result.bundles),
    )
    for pb in result.bundles:
        kb = _convert_proposed_bundle(pb)
        if kb.id in store.bundles:
            kb.id = f"{kb.id.removesuffix('_v1')}_dup_v1"
        store.bundles[kb.id] = kb


async def step5_pillar_pass(store: CorpusStore, source_label: str) -> None:
    logger.info("Step 5: pillar extraction over %d slabs", len(store.slabs))
    slabs = list(store.slabs.values())
    result = await pillar_extractor.run(slabs, origin=source_label)
    logger.info(
        "Step 5 done: %d sub-pillars, %d top-pillars",
        len(result.sub_pillars), len(result.top_pillars),
    )
    # Sub-pillars first so children references exist when top-pillars commit
    for sp in result.sub_pillars:
        pd = _convert_proposed_pillar(sp)
        store.pillars[pd.id] = pd
    for tp in result.top_pillars:
        pd = _convert_proposed_pillar(tp)
        store.pillars[pd.id] = pd


def step6_emit_bundle_edges(store: CorpusStore) -> None:
    """LINKS edges from slab → bundle, derived from bundle.depends_on.

    Mirrors the existing slab→bundle wiring convention: when a slab
    contributes evidence to a bundle's directive packet, the slab
    LINKS to the bundle at w=0.8 (packet-layer structural link, see
    draft_manager._generate_commit_edges weight tier).
    """
    logger.info("Step 6: emitting slab → bundle LINKS edges")
    edges_added = 0
    for bundle in store.bundles.values():
        for slab_id in bundle.depends_on:
            if slab_id not in store.slabs:
                continue
            from_short = slab_id.replace("mined_", "")[:20]
            to_short = bundle.id.replace("bundle_", "")[:20]
            eid = f"edge_links_{from_short}_{to_short}_v1"
            if eid in store.edges:
                continue
            store.edges[eid] = Edge(
                id=eid,
                type=EdgeType.LINKS,
                from_node=slab_id,  # type: ignore[arg-type]
                to_node=bundle.id,  # type: ignore[arg-type]
                weight=0.8,
                confidence=0.8,
            )
            # Mirror in slab.links.bundles for inline lookup (matches
            # the slab→bundle inline-vs-edge dual representation).
            slab = store.slabs[slab_id]
            if bundle.id not in slab.links.bundles:
                slab.links.bundles.append(bundle.id)
            edges_added += 1
    logger.info("Step 6 done: %d slab→bundle edges added", edges_added)


def step7_save(store: CorpusStore) -> list[str]:
    logger.info("Step 7: validating + saving %s", store.collection_id)
    errors = store.validate()
    if errors:
        logger.error("Validation produced %d errors:", len(errors))
        for e in errors[:20]:
            logger.error("  %s", e)
        if len(errors) > 20:
            logger.error("  … and %d more", len(errors) - 20)
        return errors
    store.save()
    logger.info("Step 7 done: corpus persisted")
    return []


# ─── Orchestrator ────────────────────────────────────────────────────────


async def remine(source_path: Path, target_collection: str) -> int:
    t_start = time.time()
    mine_result = await step1_mine_source(source_path)
    if mine_result.get("error"):
        logger.error("Mine failed: %s", mine_result["error"])
        return 1
    store, label_to_id = step2_build_content_corpus(mine_result, target_collection)
    step3_emit_mined_edges(store, mine_result, label_to_id)
    await step4_directive_pass(store)
    await step5_pillar_pass(store, source_label=source_path.name)
    step6_emit_bundle_edges(store)
    errors = step7_save(store)
    if errors:
        logger.error("Save aborted on validation errors")
        return 1
    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("REMINE COMPLETE in %.1fs", elapsed)
    logger.info(
        "  %s: %d anchors, %d slabs, %d bundles, %d edges, %d pillars",
        target_collection,
        len(store.anchors), len(store.slabs), len(store.bundles),
        len(store.edges), len(store.pillars),
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if len(sys.argv) < 2:
        print("Usage: remine_to_podv5.py <source_path> [<target_collection>]")
        sys.exit(2)
    source = Path(sys.argv[1]).resolve()
    target = sys.argv[2] if len(sys.argv) > 2 else "podv5"
    if not source.exists():
        print(f"Source not found: {source}")
        sys.exit(2)
    code = asyncio.run(remine(source, target))
    sys.exit(code)
