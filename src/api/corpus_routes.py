"""Corpus management & collection endpoints — extracted from routes.py."""
from __future__ import annotations
import math
import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import deps
from .deps import registry, anchor_matcher, frame_manager, rebind_corpus

logger = logging.getLogger(__name__)

router = APIRouter()


# ── /corpus/full response cache ──────────────────────────────────
#
# Background: /corpus/full re-embeds every anchor + bundle + slab in
# the requested view to compute graph positions via
# compute_positions_and_vectors. On a 1k+ node corpus this is an 8-12s
# Ollama batch call. The frontend polls this endpoint reflexively
# during bulk promote (once before AND after each draft review),
# making it the dominant per-anchor cost — measured at ~8s of every
# ~10s anchor commit during the pod run.
#
# Fix: cheap TTL cache. The corpus barely changes during a bulk
# promote (a few nodes per second at most), and the graph positions
# are derived from canonical_text which is immutable. Caching for 60s
# turns N polls into 1 embed call. Invalidated on corpus mutation
# (collection activation, manual invalidate hook), so freshness
# matters only across the TTL window.
_CORPUS_FULL_CACHE: dict[str, tuple[float, dict]] = {}
_CORPUS_FULL_TTL = 60.0  # seconds


def _corpus_full_cache_get(key: str) -> Optional[dict]:
    """Return a cached /corpus/full response if still fresh, else None."""
    entry = _CORPUS_FULL_CACHE.get(key)
    if not entry:
        return None
    timestamp, payload = entry
    if time.monotonic() - timestamp > _CORPUS_FULL_TTL:
        _CORPUS_FULL_CACHE.pop(key, None)
        return None
    return payload


def _corpus_full_cache_set(key: str, payload: dict) -> None:
    _CORPUS_FULL_CACHE[key] = (time.monotonic(), payload)


def invalidate_corpus_full_cache() -> None:
    """Drop all cached /corpus/full responses.

    Called from collection activation and other mutation paths so the
    next /corpus/full request rebuilds the graph positions against
    fresh corpus state. Cheap — just a dict.clear().
    """
    _CORPUS_FULL_CACHE.clear()


# ── Collection management ─────────────────────────────────────

@router.get("/collections")
async def list_collections():
    """List all corpus collections with metadata and active status."""
    return {"collections": registry.list_collections()}


class CreateCollectionRequest(BaseModel):
    name: str


@router.post("/collections")
async def create_collection(req: CreateCollectionRequest):
    """Create a new empty corpus collection."""
    import re as _re
    name = req.name.strip().lower().replace(" ", "_")
    name = _re.sub(r"[^a-z0-9_-]", "", name)
    if not name:
        raise HTTPException(400, "Invalid collection name")
    if name in registry.collections:
        raise HTTPException(409, f"Collection '{name}' already exists")

    store = registry.create_collection(name)
    rebind_corpus()
    return {
        "id": name,
        "anchors": len(store.anchors),
        "slabs": len(store.slabs),
        "bundles": len(store.bundles),
        "message": f"Collection '{name}' created and activated",
    }


class ToggleCollectionRequest(BaseModel):
    active: bool


@router.patch("/collections/{collection_id}")
async def toggle_collection(collection_id: str, req: ToggleCollectionRequest):
    """Activate or deactivate a collection."""
    if collection_id not in registry.collections:
        raise HTTPException(404, f"Collection '{collection_id}' not found")

    if req.active:
        registry.activate(collection_id)
    else:
        registry.deactivate(collection_id)

    rebind_corpus()
    # Collection activation actually changes the merged corpus view —
    # drop the /corpus/full cache so the next graph render reflects
    # the new active set. Note: rebind_corpus() does NOT invalidate
    # this cache itself; it's called per edge-commit during bulk
    # promote where invalidation would defeat the purpose. Explicit
    # invalidation here gates on real corpus-membership changes.
    invalidate_corpus_full_cache()
    # Re-warm anchor matcher cache with new merged set
    await anchor_matcher.warm_cache()

    merged = registry.merged
    return {
        "id": collection_id,
        "active": req.active,
        "merged_anchors": len(merged.anchors),
        "merged_slabs": len(merged.slabs),
        "merged_bundles": len(merged.bundles),
    }


class PromoteCollectionRequest(BaseModel):
    """Promote a small collection into the main corpus as a single slab."""
    target_collection: str = "default"
    slab_title: str
    slab_id: str = ""
    deactivate_source: bool = True  # deactivate source collection after promote


@router.delete("/collections/{collection_id}")
async def delete_collection(collection_id: str, purge: bool = False):
    """Delete a collection from the registry.

    If `purge=true`, also removes the collection directory from disk.
    The default collection cannot be deleted.
    """
    if collection_id == "default":
        raise HTTPException(400, "Cannot delete the default collection")
    if collection_id not in registry.collections:
        raise HTTPException(404, f"Collection '{collection_id}' not found")

    # Capture counts before deletion for response
    store = registry.get_store(collection_id)
    counts = {
        "anchors": len(store.anchors) if store else 0,
        "slabs": len(store.slabs) if store else 0,
        "bundles": len(store.bundles) if store else 0,
    }

    registry.delete_collection(collection_id)

    if purge:
        import shutil
        from ..services.corpus import CORPORA_ROOT
        collection_dir = CORPORA_ROOT / collection_id
        if collection_dir.exists():
            shutil.rmtree(collection_dir)

    rebind_corpus()
    await anchor_matcher.warm_cache()

    merged = registry.merged
    return {
        "id": collection_id,
        "purged": purge,
        "deleted_counts": counts,
        "merged_anchors": len(merged.anchors),
        "merged_slabs": len(merged.slabs),
        "merged_bundles": len(merged.bundles),
    }


@router.post("/collections/{collection_id}/promote")
async def promote_collection(collection_id: str, req: PromoteCollectionRequest):
    """Slabify: promote a collection into a target collection as a summary slab.

    Takes all anchors/bundles/slabs from the source collection and creates
    a single CANONICAL slab in the target that captures the entire
    neighborhood as a thematic unit.

    The source collection remains on disk (can be re-activated later)
    but is deactivated by default so objects don't double-count in matching.
    """
    source = registry.get_store(collection_id)
    if not source:
        raise HTTPException(404, f"Collection '{collection_id}' not found")
    target = registry.get_store(req.target_collection)
    if not target:
        raise HTTPException(404, f"Target collection '{req.target_collection}' not found")

    # Collect all concept labels from source
    anchor_phrases = [a.canonical_phrase for a in source.anchors.values()]
    anchor_aliases = []
    for a in source.anchors.values():
        anchor_aliases.extend(a.aliases)
    bundle_labels = [b.id for b in source.bundles.values()]
    slab_titles = [s.title or s.id for s in source.slabs.values()]
    edge_count = len(source.edges)

    # Build rich slab text
    summary_parts = [f"Neighborhood: {collection_id}"]
    if slab_titles:
        summary_parts.append(f"Themes: {', '.join(slab_titles)}")
    if anchor_phrases:
        summary_parts.append(f"Key concepts: {', '.join(anchor_phrases)}")
    if bundle_labels:
        summary_parts.append(f"Bundles: {', '.join(bundle_labels)}")
    if edge_count:
        summary_parts.append(f"Relationships: {edge_count} edges")
    summary_parts.append("")
    summary_parts.append(
        f"This slab represents the '{collection_id}' corpus neighborhood, "
        f"promoted as a single thematic unit. It contains {len(source.anchors)} "
        f"anchors, {len(source.slabs)} slabs, {len(source.bundles)} bundles."
    )
    summary_text = "\n".join(summary_parts)

    # Generate slab ID
    import re as _re
    slab_id = req.slab_id or f"SLAB_{_re.sub(r'[^A-Z0-9_]', '_', collection_id.upper())}_v1"

    from ..models.schemas import Slab, AnchorMeta
    from ..models.enums import SlabType, SlabLifecycleStatus
    new_slab = Slab(
        id=slab_id,
        title=req.slab_title,
        canonical_text=summary_text,
        type=SlabType.CANONICAL,
        lifecycle_status=SlabLifecycleStatus.ACTIVE,
        requires_oli_mode=None,
        depends_on=[],
        meta=AnchorMeta(version="v1", source=f"promoted:{collection_id}"),
    )

    target.slabs[slab_id] = new_slab
    target.save()

    # Optionally deactivate source to prevent double-counting
    if req.deactivate_source and collection_id != req.target_collection:
        registry.deactivate(collection_id)

    rebind_corpus()
    return {
        "slab_id": slab_id,
        "title": req.slab_title,
        "source_anchors": len(anchor_phrases),
        "source_bundles": len(bundle_labels),
        "target_collection": req.target_collection,
        "message": f"Promoted '{collection_id}' as slab '{slab_id}' in '{req.target_collection}'",
    }


# ── Anchor consolidation ──────────────────────────────────────


@router.get("/corpora/{collection_id}/consolidation-preview")
async def consolidation_preview(
    collection_id: str,
    max_examples: int = 50,
):
    """Dry-run the anchor consolidation pass on a collection.

    Returns the analysis (counts + sample verdicts) WITHOUT mutating
    the corpus. Use this to inspect what would be demoted before
    committing to the apply step.

    Args:
        collection_id: collection to analyze. Use ``__merged__`` for
            the active merged view.
        max_examples: cap the number of verdicts in each bucket
            (keep / demote / orphan) returned in the response. Use
            ``?max_examples=0`` to see all verdicts (potentially
            hundreds — fine for analysis, heavy in the UI).
    """
    from ..services.anchor_consolidation import analyze

    if collection_id == "__merged__":
        source = deps.corpus
    else:
        source = registry.get_store(collection_id)
        if not source:
            raise HTTPException(404, f"Collection '{collection_id}' not found")

    plan = analyze(source)

    def _serialize(verdict_list):
        verdicts = [
            {
                "anchor_id": v.anchor_id,
                "canonical_phrase": v.canonical_phrase,
                "decision": v.decision,
                "reason": v.reason,
                "own_slab_reach": v.own_slab_reach,
                "bundle_reach": v.bundle_reach,
                "has_semantic_edges": v.has_semantic_edges,
                "parent_slab_id": v.parent_slab_id,
            }
            for v in verdict_list
        ]
        if max_examples > 0:
            verdicts = verdicts[:max_examples]
        return verdicts

    return {
        "collection_id": collection_id,
        "summary": plan.summary(),
        "verdicts": {
            "keep": _serialize(plan.keep),
            "demote": _serialize(plan.demote),
            "orphan": _serialize(plan.orphan),
        },
        "max_examples": max_examples,
    }


class ConsolidateRequest(BaseModel):
    confirm: bool = False  # safety: must be true to actually mutate


@router.post("/corpora/{collection_id}/consolidate")
async def consolidate(collection_id: str, req: ConsolidateRequest):
    """Apply anchor consolidation to a collection. MUTATES the corpus.

    Calls analyze() then apply_plan(). Demotes leaf anchors into
    their parent slab's links.anchors_inline, drops the now-redundant
    LINKS edges, leaves orphan anchors with a flag (no auto-purge in
    v1).

    Requires ``{"confirm": true}`` in the body so the destructive
    action is explicit. After mutation, persists the corpus YAML and
    rebinds dependent services.
    """
    if not req.confirm:
        raise HTTPException(
            400,
            "Pass {\"confirm\": true} to apply. Use the preview endpoint first.",
        )

    from ..services.anchor_consolidation import analyze, apply_plan

    if collection_id == "__merged__":
        raise HTTPException(
            400,
            "Cannot consolidate the merged view directly — pick a specific collection.",
        )
    source = registry.get_store(collection_id)
    if not source:
        raise HTTPException(404, f"Collection '{collection_id}' not found")

    plan = analyze(source)
    result = apply_plan(source, plan)

    # Persist + rebind. apply_plan mutated the source store in place;
    # CorpusStore.save() flushes YAML for all four object lists,
    # rebind_corpus refreshes the merged view + downstream services.
    try:
        source.save()
    except Exception as exc:
        logger.error("consolidate: failed to persist YAML: %r", exc)
        raise HTTPException(500, f"Mutation succeeded but persist failed: {exc}")
    rebind_corpus()
    await anchor_matcher.warm_cache()

    return {
        "collection_id": collection_id,
        "summary": plan.summary(),
        "applied": result,
    }


# ── Corpus status & full browser ──────────────────────────────

@router.get("/corpus/status")
async def corpus_status():
    errors = deps.corpus.validate()
    return {
        "anchors": len(deps.corpus.anchors),
        "slabs": len(deps.corpus.slabs),
        "bundles": len(deps.corpus.bundles),
        "edges": len(deps.corpus.edges),
        "valid": len(errors) == 0,
        "errors": errors,
        "collections": registry.list_collections(),
    }


@router.get("/corpus/full")
async def corpus_full(collection: Optional[str] = None):
    """Return corpus for the Cold Corpus browser tab.

    See ``_CORPUS_FULL_CACHE`` above — responses are cached for
    ``_CORPUS_FULL_TTL`` seconds (default 60s) to avoid re-embedding
    the entire corpus on every UI poll during bulk promote runs.
    Cache invalidates on collection activation / corpus mutation.

    Args:
        collection: Optional collection ID to filter to. If omitted,
                    returns the full merged corpus.

    Computes semantic axis positions (ss11.1) so the corpus graph
    uses the same structured layout as the session graph:
      X = Creative <-> Rigorous (embedding projection)
      Y = structural weight proxy (type-based: slab=0.4, bundle=0.2, anchor=0.1)
      Z = meta depth (slab=0, bundle=1, anchor=2)
    """
    from ..services.embeddings import compute_positions_and_vectors, compute_affinity

    # Cache check — return early if a fresh response is available.
    # Key includes the collection filter so different views don't
    # shadow each other. The expensive work below (embeddings,
    # affinity pairs, layout hints) all derives from canonical_text
    # which is immutable, so a 60s window is safe.
    cache_key = collection or "__all__"
    cached = _corpus_full_cache_get(cache_key)
    if cached is not None:
        return cached

    # Select corpus source: specific collection or full merged view
    if collection and collection != "__all__":
        source = registry.get_store(collection)
        if not source:
            raise HTTPException(404, f"Collection '{collection}' not found")
    else:
        source = deps.corpus

    # Collect texts for batch embedding
    texts_to_embed: list[str] = []
    node_types: list[str] = []  # parallel list for z-axis mapping

    anchors_out = []
    for a in source.anchors.values():
        d = a.model_dump()
        texts_to_embed.append(a.canonical_phrase)
        node_types.append("anchor")
        anchors_out.append(d)

    bundles_out = []
    for b in source.bundles.values():
        d = b.model_dump()
        label = "; ".join(b.payload.intent[:2]) if b.payload.intent else b.id
        texts_to_embed.append(label)
        node_types.append("bundle")
        bundles_out.append(d)

    slabs_out = []
    for s in source.slabs.values():
        d = s.model_dump()
        # Subgraph condensate detection: slabs produced by slabifying a
        # collection start their canonical_text with "Neighborhood: <id>".
        # See the slabify promotion builder around line 1175.
        d["is_subgraph"] = bool(
            s.canonical_text and s.canonical_text.startswith("Neighborhood: ")
        )
        texts_to_embed.append((s.title or s.canonical_text)[:120])
        node_types.append("slab")
        slabs_out.append(d)

    # Batch compute X positions via nomic-embed-text
    # Original type-band heuristic (commented for reference) stacked anchors
    # below bundles below slabs in y/z. It imposed a structural prior that
    # fought the semantic forces: a rigorous-adjacent anchor could not drift
    # toward its parent slab because the y/z bands pinned it in its type row.
    # Flattened to a single plane so affinity springs + repulsion + hub-spoke
    # orbit drive the layout. Type is now a rendering attribute, not a
    # positional one.
    # z_map = {"slab": 0, "bundle": 1, "anchor": 2}
    # y_map = {"slab": 0.4, "bundle": 0.2, "anchor": 0.1}  # structural weight proxy
    z_map = {"slab": 0, "bundle": 0, "anchor": 0}
    y_map = {"slab": 0.4, "bundle": 0.4, "anchor": 0.4}

    try:
        x_positions, node_vectors = await compute_positions_and_vectors(texts_to_embed)
    except Exception:
        import numpy as _np
        x_positions = [0.5] * len(texts_to_embed)
        node_vectors = _np.zeros((len(texts_to_embed), 1))

    # Inject semantic positions into each node dict
    all_nodes = anchors_out + bundles_out + slabs_out
    for i, node in enumerate(all_nodes):
        node["sem_x"] = x_positions[i]
        node["sem_y"] = y_map.get(node_types[i], 0.1)
        node["sem_z"] = z_map.get(node_types[i], 2)

    edges_out = [e.model_dump(by_alias=True) for e in source.edges.values()]

    # -- Hub-and-spoke: pull anchors/bundles toward their linked slabs --
    # Without this, type-based sem_y/sem_z creates flat horizontal layers
    # and nodes float independently. With this, connected children orbit
    # their parent slab, making the graph read as "slab governs these anchors."
    node_type_map = {node["id"]: node_types[i] for i, node in enumerate(all_nodes)}
    node_sem_x = {node["id"]: node["sem_x"] for node in all_nodes}
    # Map each anchor/bundle to its first connected slab
    child_to_slab: dict[str, str] = {}
    for edge in source.edges.values():
        ft = node_type_map.get(edge.from_node)
        tt = node_type_map.get(edge.to_node)
        if ft == "slab" and tt in ("anchor", "bundle"):
            child_to_slab.setdefault(edge.to_node, edge.from_node)
        elif tt == "slab" and ft in ("anchor", "bundle"):
            child_to_slab.setdefault(edge.from_node, edge.to_node)
    # Count children per slab so we can spread them evenly
    slab_child_count: dict[str, int] = {}
    slab_child_idx: dict[str, int] = {}
    for child_id, slab_id in child_to_slab.items():
        slab_child_count[slab_id] = slab_child_count.get(slab_id, 0) + 1
    _slab_counters: dict[str, int] = {}
    for child_id, slab_id in child_to_slab.items():
        idx = _slab_counters.get(slab_id, 0)
        slab_child_idx[child_id] = idx
        _slab_counters[slab_id] = idx + 1
    # Reposition children to orbit their parent slab
    for node in all_nodes:
        nid = node["id"]
        if nid not in child_to_slab:
            continue
        parent_id = child_to_slab[nid]
        parent_x = node_sem_x.get(parent_id, 0.5)
        n_children = max(1, slab_child_count.get(parent_id, 1))
        idx = slab_child_idx.get(nid, 0)
        # Radial angle — spread children evenly around the slab
        angle = (2 * math.pi * idx / n_children) + 0.3  # offset to avoid overlap
        ntype = node_type_map.get(nid, "anchor")
        radius_y = 0.20 if ntype == "anchor" else 0.15
        radius_z = 1.2 if ntype == "anchor" else 0.8
        # Pull x toward parent (70% parent, 30% own semantic position)
        node["sem_x"] = parent_x * 0.7 + node["sem_x"] * 0.3
        # Orbit in y/z around the slab's y/z position
        node["sem_y"] = y_map["slab"] + radius_y * math.cos(angle)
        node["sem_z"] = z_map["slab"] + radius_z * math.sin(angle)

    # -- Embedding-similarity affinity --
    # For children (anchors/bundles) with NO explicit edge to a slab, find the
    # slab with highest cosine similarity >= 0.70 and attach them implicitly.
    # Also compute top-K similarity pairs across all nodes so the frontend can
    # add soft attractive forces (the embedding-space analog of springs).
    implicit_parent_map: dict[str, tuple[str, float]] = {}
    affinity_pairs: list[dict] = []
    try:
        node_ids_list = [n["id"] for n in all_nodes]
        implicit_parent_map, affinity_pairs = compute_affinity(
            node_ids_list, node_types, node_vectors,
            explicit_parent=child_to_slab,
            parent_threshold=0.70,
            pair_threshold=0.65,
            top_k_pairs=3,
        )
    except Exception as e:
        print(f"[CORPUS_FULL] Affinity computation failed: {e}", flush=True)

    # Apply implicit_parent to floaters: orbit their best-match slab just like
    # explicit children, but mark them so the frontend can render a dashed
    # attachment line. Keeps floaters from piling up in the type-based band.
    node_sem_x_after = {n["id"]: n["sem_x"] for n in all_nodes}
    _implicit_slot_counter: dict[str, int] = {}
    _implicit_slot_total: dict[str, int] = {}
    for child_id, (parent_id, sim) in implicit_parent_map.items():
        _implicit_slot_total[parent_id] = _implicit_slot_total.get(parent_id, 0) + 1
    for node in all_nodes:
        nid = node["id"]
        if nid not in implicit_parent_map:
            continue
        parent_id, sim = implicit_parent_map[nid]
        node["implicit_parent"] = parent_id
        node["implicit_parent_sim"] = sim
        # Offset angle from explicit children so implicit orbit doesn't overlap
        idx = _implicit_slot_counter.get(parent_id, 0)
        _implicit_slot_counter[parent_id] = idx + 1
        total = max(1, _implicit_slot_total.get(parent_id, 1))
        angle = (2 * math.pi * idx / total) + 1.7  # offset from explicit orbit
        parent_x = node_sem_x_after.get(parent_id, 0.5)
        ntype = node_type_map.get(nid, "anchor")
        radius_y = 0.18 if ntype == "anchor" else 0.14
        radius_z = 1.4 if ntype == "anchor" else 1.0  # further out than explicit
        node["sem_x"] = parent_x * 0.65 + node["sem_x"] * 0.35
        node["sem_y"] = y_map["slab"] + radius_y * math.cos(angle)
        node["sem_z"] = z_map["slab"] + radius_z * math.sin(angle)

    # -- Narrative layout trigger --
    # Two cases promote a collection to the linear "flow" layout:
    #   (a) No edges at all -> synthesize a SEQUENCE chain from ID order so the
    #       user still sees a readable spine (Phase 1 behaviour, preserved).
    #   (b) SEQUENCE edges dominate the structure (>=2 SEQUENCE edges and at
    #       least as many SEQUENCE as non-SEQUENCE). Phase 3 mining persists
    #       real SEQUENCE edges; when they exist, the collection *is* a
    #       narrative and should render linearly even though edges are present.
    # Also emit `spine_order` -- the topological walk of SEQUENCE edges -- so the
    # frontend can position nodes along the actual story order, not alpha order.
    layout_hint = "cloud"
    spine_order: list[str] = []
    node_id_set = {n["id"] for n in all_nodes}
    seq_edges = [e for e in edges_out if e.get("type") == "SEQUENCE"
                 and e.get("from") in node_id_set and e.get("to") in node_id_set]
    other_edges = [e for e in edges_out if e.get("type") != "SEQUENCE"]
    # Any meaningful SEQUENCE presence (>=2 edges) promotes to the flow layout.
    # Earlier heuristics compared SEQUENCE vs other edges or SEQUENCE vs node
    # count, but both fail in mixed corpora: the narrative spans only slabs,
    # while anchors/bundles and cross-ref SUPPORTS edges inflate the non-spine
    # counts even when SEQUENCE is clearly the backbone. Orphans (anchors,
    # bundles, non-spine slabs) hang perpendicular in the flow layout, so
    # falsely promoting a cloud to flow is cheap; the reverse is visible.
    narrative_dominant = len(seq_edges) >= 2

    if collection and collection != "__all__" and len(all_nodes) > 1:
        if narrative_dominant:
            layout_hint = "flow"
            # Walk SEQUENCE edges: find roots (no incoming SEQUENCE), chain
            # strongest-outgoing until exhausted. Mirrors _buildNarrativeTrail()
            # in the frontend so trace + layout agree on spine order.
            outgoing: dict[str, list] = {}
            incoming: dict[str, list] = {}
            for e in seq_edges:
                outgoing.setdefault(e["from"], []).append(e)
                incoming.setdefault(e["to"], []).append(e)
            roots = [nid for nid in outgoing if nid not in incoming]
            if not roots and outgoing:
                roots = [next(iter(outgoing))]
            visited: set[str] = set()
            for root in roots:
                cur = root
                while cur and cur not in visited:
                    visited.add(cur)
                    spine_order.append(cur)
                    outs = [e for e in outgoing.get(cur, []) if e["to"] not in visited]
                    if not outs:
                        break
                    outs.sort(key=lambda e: -(e.get("weight") or e.get("confidence") or 0))
                    cur = outs[0]["to"]
            # Note: orphans (nodes not in the SEQUENCE walk) are intentionally
            # NOT appended to spine_order. The frontend treats spine_order as
            # "the narrative ribbon"; orphans need to fall into the else-branch
            # of cSemPos so they can orbit their parent via _orphanParent
            # lookup built from SUPPORTS/INVOKES/LINKS edges.
        elif len(edges_out) == 0:
            layout_hint = "flow"
            ordered_ids = (
                [a["id"] for a in anchors_out]
                + [b["id"] for b in bundles_out]
                + [s["id"] for s in slabs_out]
            )
            spine_order = ordered_ids
            for i in range(len(ordered_ids) - 1):
                edges_out.append({
                    "id": f"_seq_{i}",
                    "type": "SEQUENCE",
                    "from": ordered_ids[i],
                    "to": ordered_ids[i + 1],
                    "weight": 0.4,
                    "confidence": 0.4,
                })

    response = {
        "collection": collection or "__all__",
        "layout_hint": layout_hint,
        "spine_order": spine_order,  # ordered node IDs along SEQUENCE spine (flow layout only)
        "anchors": anchors_out,
        "bundles": bundles_out,
        "slabs": slabs_out,
        "edges": edges_out,
        # Similarity-derived attraction graph. affinity_pairs feeds the physics
        # springs; implicit_parent_edges lets the UI render dashed attachment
        # lines for floaters that got auto-assigned to a parent slab.
        "affinity_pairs": affinity_pairs,
        "implicit_parent_edges": [
            {"from": pid, "to": cid, "similarity": sim}
            for cid, (pid, sim) in implicit_parent_map.items()
        ],
        "gates": [g.model_dump() for g in source.gates.values()],
    }
    _corpus_full_cache_set(cache_key, response)
    return response


# ── Corpus node management (dashboard) ────────────────────────

class UpdateLifecycleRequest(BaseModel):
    lifecycle_status: str  # "ACTIVE" | "DORMANT" | "DEPRECATED"


class UpdateNodeRequest(BaseModel):
    canonical_phrase: Optional[str] = None
    canonical_text: Optional[str] = None
    notes: Optional[str] = None
    title: Optional[str] = None
    aliases: Optional[list[str]] = None


def _find_corpus_node(node_id: str):
    """Locate a node + its owning collection store.

    Returns ``(obj, type_name, owning_store)``. The store is the
    on-disk-backed ``CorpusStore`` for the collection that owns this
    node; callers should mutate it and call ``.save()`` on the owning
    store, NOT on ``deps.corpus`` — the latter is the merged virtual
    view whose ``root`` falls back to the legacy ``app/corpus/`` path,
    so saving through it would write to a phantom directory and skip
    the real collection on disk. Returns ``(None, None, None)`` if not
    found.
    """
    for cid in sorted(deps.registry.active_ids):
        store = deps.registry.get_store(cid)
        if store is None:
            continue
        if node_id in store.anchors:
            return store.anchors[node_id], "anchor", store
        if node_id in store.slabs:
            return store.slabs[node_id], "slab", store
        if node_id in store.bundles:
            return store.bundles[node_id], "bundle", store
        if node_id in store.gates:
            return store.gates[node_id], "gate", store
    return None, None, None


@router.get("/corpus/nodes/{node_id}/deps")
async def get_node_deps(node_id: str):
    """Return everything that depends on or references this node."""
    node, ntype, _store = _find_corpus_node(node_id)
    if not node:
        raise HTTPException(404, f"Node '{node_id}' not found in corpus")

    dependents = deps.corpus.get_reverse_deps(node_id)
    # Also find edges that reference this node
    edge_refs = [
        {"edge_id": e.id, "from": e.from_node, "to": e.to_node, "type": e.type.value}
        for e in deps.corpus.edges.values()
        if e.from_node == node_id or e.to_node == node_id
    ]
    return {
        "node_id": node_id,
        "node_type": ntype,
        "dependents": dependents,
        "edges": edge_refs,
        "safe_to_delete": len(dependents) == 0,
    }


@router.patch("/corpus/nodes/{node_id}/lifecycle")
async def update_node_lifecycle(node_id: str, req: UpdateLifecycleRequest):
    """Set lifecycle status on any corpus node (anchor, slab, bundle)."""
    from ..models.enums import SlabLifecycleStatus
    node, ntype, owning_store = _find_corpus_node(node_id)
    if not node:
        raise HTTPException(404, f"Node '{node_id}' not found in corpus")
    try:
        new_status = SlabLifecycleStatus(req.lifecycle_status)
    except ValueError:
        raise HTTPException(400, f"Invalid status '{req.lifecycle_status}'. Must be ACTIVE, DORMANT, or DEPRECATED.")

    # Merged view shares the object by reference, so the in-place mutation
    # propagates without extra dict syncing. Save through the owning store
    # so the right collection's YAML is rewritten; mark the merged cache
    # dirty so the next access recomputes from current member data.
    node.lifecycle_status = new_status
    owning_store.save()
    deps.registry._merged_dirty = True
    return {"node_id": node_id, "type": ntype, "lifecycle_status": new_status.value}


@router.patch("/corpus/nodes/{node_id}")
async def update_node_fields(node_id: str, req: UpdateNodeRequest):
    """Edit fields on a corpus node."""
    node, ntype, owning_store = _find_corpus_node(node_id)
    if not node:
        raise HTTPException(404, f"Node '{node_id}' not found in corpus")

    updated = []
    if req.canonical_phrase is not None and hasattr(node, 'canonical_phrase'):
        node.canonical_phrase = req.canonical_phrase
        updated.append('canonical_phrase')
    if req.canonical_text is not None and hasattr(node, 'canonical_text'):
        node.canonical_text = req.canonical_text
        updated.append('canonical_text')
    if req.notes is not None and hasattr(node, 'notes'):
        node.notes = req.notes
        updated.append('notes')
    if req.title is not None and hasattr(node, 'title'):
        node.title = req.title
        updated.append('title')
    if req.aliases is not None and hasattr(node, 'aliases'):
        node.aliases = req.aliases
        updated.append('aliases')

    if not updated:
        raise HTTPException(400, "No applicable fields to update on this node type.")

    owning_store.save()
    deps.registry._merged_dirty = True
    return {"node_id": node_id, "type": ntype, "updated_fields": updated}


@router.delete("/corpus/nodes/{node_id}")
async def delete_node(node_id: str):
    """Hard-delete a node from the corpus. Checks dependencies first."""
    node, ntype, owning_store = _find_corpus_node(node_id)
    if not node:
        raise HTTPException(404, f"Node '{node_id}' not found in corpus")

    node_deps = deps.corpus.get_reverse_deps(node_id)
    if node_deps:
        raise HTTPException(
            409,
            f"Cannot delete '{node_id}': {len(node_deps)} node(s) depend on it: {node_deps[:5]}. "
            "Deprecate it instead, or remove the dependencies first."
        )

    # Remove from BOTH the owning collection's dict (so the on-disk save
    # actually drops the node) AND the live merged view (so subsequent
    # reads on this process see the deletion before the merged cache is
    # next rebuilt).
    if ntype == "anchor":
        owning_store.anchors.pop(node_id, None)
        deps.corpus.anchors.pop(node_id, None)
    elif ntype == "slab":
        owning_store.slabs.pop(node_id, None)
        deps.corpus.slabs.pop(node_id, None)
    elif ntype == "bundle":
        owning_store.bundles.pop(node_id, None)
        deps.corpus.bundles.pop(node_id, None)
    elif ntype == "gate":
        owning_store.gates.pop(node_id, None)
        deps.corpus.gates.pop(node_id, None)

    # Edges referencing this node may live in any active collection — a
    # mining pass can produce cross-collection edges where the edge file
    # owner differs from either endpoint's owner. Locate each affected
    # edge in its owning store, remove it there + in the merged view,
    # and save every store we touched.
    dead_edges = [eid for eid, e in deps.corpus.edges.items()
                  if e.from_node == node_id or e.to_node == node_id]
    touched_stores: list = [owning_store]
    for eid in dead_edges:
        deps.corpus.edges.pop(eid, None)
        for cid in sorted(deps.registry.active_ids):
            s = deps.registry.get_store(cid)
            if s and eid in s.edges:
                del s.edges[eid]
                if s not in touched_stores:
                    touched_stores.append(s)
                break

    for s in touched_stores:
        s.save()
    deps.registry._merged_dirty = True

    return {"deleted": node_id, "type": ntype, "edges_removed": len(dead_edges)}


# ── Graph shape + communities ─────────────────────────────────
# Feed the frontend enough to decide how to render each collection:
#   - /corpus/shape: cheap classifier (spine vs highway vs sparse vs flat)
#   - /corpus/communities: recursive Leiden tree + labels for cluster/zoom views

@router.get("/corpus/shape")
async def get_corpus_shape(scope: Optional[str] = None):
    """Classify a collection's graph topology for visualization routing.

    ``scope`` is a collection id. When omitted, returns shape reports
    for all active collections plus the merged view. No LLM cost,
    no caching — the classifier is cheap enough to re-run per request.
    """
    from ..services.graph_shape import classify_shape, shape_report_to_dict

    if scope is not None:
        store = registry.get_store(scope)
        if store is None:
            raise HTTPException(404, f"Collection '{scope}' not found")
        return shape_report_to_dict(classify_shape(store))

    # No scope → all active + merged
    out = {}
    for cid in sorted(registry.active_ids):
        store = registry.get_store(cid)
        if store is not None:
            out[cid] = shape_report_to_dict(classify_shape(store))
    # Merged = the unified corpus view
    out["__merged__"] = shape_report_to_dict(classify_shape(deps.corpus))
    return out


@router.get("/corpus/communities")
async def get_corpus_communities(
    scope: Optional[str] = None,
    force: bool = False,
    skip_labels: bool = False,
):
    """Return the community tree + labels for a collection (or merged view).

    ``scope`` behaves like the shape endpoint:
      - collection id → that collection's tree
      - ``"__merged__"`` → cross-collection union graph
      - omitted → defaults to ``"__merged__"``

    ``force=true`` bypasses the fingerprint cache check; useful for
    forcing a recompute after a manual corpus edit that didn't cross
    the 10% node-delta threshold.

    ``skip_labels=true`` returns the tree structure without running
    the LLM labeling pass. Fast — sub-second even on the largest
    collection. Callers that only want visualization without human-
    readable labels can use this.

    Response shape::

        {
          "scope": "podv1",
          "tree": [ {"members": [...], "children": [...], ...}, ... ],
          "labels": {"0": "Physical Infrastructure", "0,1": "Plumbing", ...},
          "fingerprint": {"node_count": 710, ...},
          "computed_at": "...",
          "meta_nodes": [ ... ],      # top-level cluster summary
          "meta_edges": [ ... ]        # inter-cluster edge counts
        }
    """
    from ..services.graph_communities import (
        compute_and_persist,
        CommunityStore,
        build_meta_graph,
    )

    # Resolve scope
    scope = scope or "__merged__"
    if scope == "__merged__":
        corpus = deps.corpus
        node_ids = (
            list(corpus.anchors.keys())
            + list(corpus.slabs.keys())
            + list(corpus.bundles.keys())
        )
        edges = list(corpus.edges.values())
    else:
        store = registry.get_store(scope)
        if store is None:
            raise HTTPException(404, f"Collection '{scope}' not found")
        corpus = store
        node_ids = (
            list(store.anchors.keys())
            + list(store.slabs.keys())
            + list(store.bundles.keys())
        )
        edges = list(store.edges.values())

    if not node_ids:
        return {
            "scope": scope,
            "tree": [],
            "labels": {},
            "fingerprint": {"node_count": 0, "edge_count": 0, "nodes_hash": ""},
            "meta_nodes": [],
            "meta_edges": [],
        }

    # CommunityStore singleton lives on deps.community_store (wired below).
    # Fall back to a fresh instance if not yet wired (e.g. in tests).
    cs = getattr(deps, "community_store", None) or CommunityStore()
    payload = await compute_and_persist(
        corpus=corpus,
        scope=scope,
        node_ids=node_ids,
        edges=edges,
        community_store=cs,
        force=force,
        skip_labels=skip_labels,
    )

    # Enrich with meta-graph for the frontend. Computed here rather
    # than persisted because it's cheap and keeps the on-disk payload
    # smaller (the tree + labels are the expensive-to-compute parts).
    meta_nodes, meta_edges = build_meta_graph(payload["tree"], edges)
    return {
        **payload,
        "meta_nodes": meta_nodes,
        "meta_edges": meta_edges,
    }
