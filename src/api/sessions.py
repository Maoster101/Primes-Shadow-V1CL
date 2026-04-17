"""Session/Frame endpoints and 3D graph builder.

Extracted from routes.py — covers the ``# --- Sessions & Frame ---`` section
and the ``/sessions/{session_id}/graph`` endpoint.
"""
from __future__ import annotations
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import deps
from .deps import frame_manager, session_store, event_log, draft_manager
from ..services.frame_manager import FrameNotFoundError

logger = logging.getLogger(__name__)

router = APIRouter()


# --- Request models ---

class NodeActionRequest(BaseModel):
    node_id: str


class NodeSalienceRequest(BaseModel):
    node_id: str
    delta: float = 0.0  # positive = heat, negative = cool


class CreateBundleRequest(BaseModel):
    node_ids: list[str]
    label: str = ""


class PromoteToAnchorRequest(BaseModel):
    node_id: str


# --- Sessions & Frame ---

@router.post("/sessions/{session_id}/frame/adjust-salience")
async def adjust_salience(session_id: str, req: NodeSalienceRequest):
    """Heat or cool a node — modifies backend salience that the model sees."""
    try:
        result = frame_manager.adjust_node_salience(session_id, req.node_id, req.delta)
    except FrameNotFoundError:
        raise HTTPException(404, "No active frame for this session")
    if result is None:
        return {"error": "Node not in active frame"}

    action = "heated" if req.delta > 0 else "cooled"
    event_log.log_frame_event(
        session_id=session_id, event=f"user_{action}",
        node_id=req.node_id,
        old=round(result["old"], 3), new=round(result["new"], 3),
    )
    return {"status": action, "node_id": req.node_id, "salience": round(result["new"], 3)}


@router.post("/sessions/{session_id}/frame/remove-node")
async def remove_node_from_session(session_id: str, req: NodeActionRequest):
    """Remove a node from the active frame.

    Evicts from frame but does NOT permanently delete from tentative registry.
    Marks as 'dismissed' so the concept detector can bring it back if
    the user organically discusses it again.
    """
    try:
        frame_manager.dismiss_node(session_id, req.node_id)
    except FrameNotFoundError:
        raise HTTPException(404, "No active frame for this session")
    event_log.log_frame_event(session_id=session_id, event="node_removed", node_id=req.node_id)
    return {"status": "removed", "node_id": req.node_id}


@router.post("/sessions/{session_id}/frame/reject-node")
async def reject_node(session_id: str, req: NodeActionRequest):
    """Mark a tentative node as rejected — stays visible but greyed out, will not be proposed for commit."""
    frame_manager.reject_node(session_id, req.node_id)
    event_log.log_frame_event(session_id=session_id, event="node_rejected", node_id=req.node_id)
    return {"status": "rejected", "node_id": req.node_id}


@router.post("/sessions/{session_id}/frame/create-bundle")
async def create_bundle_from_nodes(session_id: str, req: CreateBundleRequest):
    """User-driven: group selected nodes into a tentative bundle."""
    try:
        result = frame_manager.create_tentative_bundle(
            session_id, list(req.node_ids), label=req.label
        )
    except FrameNotFoundError:
        raise HTTPException(404, "No active frame")
    except ValueError as e:
        return {"error": str(e)}

    event_log.log_frame_event(
        session_id=session_id, event="bundle_created_by_user",
        bundle_id=result["bundle_id"], children=req.node_ids,
    )
    return {"status": "created", **result}


@router.post("/sessions/{session_id}/frame/promote-to-anchor")
async def promote_to_anchor(session_id: str, req: PromoteToAnchorRequest):
    """Promote a tentative bundle to a tentative anchor (invocation handle)."""
    try:
        result = frame_manager.promote_to_anchor(session_id, req.node_id)
    except FrameNotFoundError:
        raise HTTPException(404, "No active frame")
    if result is None:
        return {"error": "Node not found in tentative registry"}

    event_log.log_frame_event(
        session_id=session_id, event="bundle_promoted_to_anchor",
        node_id=req.node_id, label=result["label"], children=result["children"],
    )
    return {"status": "promoted", "node_id": req.node_id, "type": "anchor", "invokes": result["children"]}


@router.get("/sessions/{session_id}/frame")
async def get_frame(session_id: str):
    # Prefer the in-memory frame — it holds unsaved hit log entries and
    # salience updates from the current turn. Fall back to disk only when
    # the session hasn't been restored yet (e.g. after a server restart).
    frame = frame_manager.get_frame(session_id)
    if frame is None:
        frame = session_store.load_frame(session_id)
        if frame:
            reg, edges = session_store.load_registry(session_id)
            frame_manager.restore(session_id, frame, reg or None, edges or None)
    if not frame:
        raise HTTPException(404, "No frame state for this session")
    return frame.model_dump(mode="json")


# --- Graph ---

@router.get("/sessions/{session_id}/graph")
async def get_graph_data(session_id: str):
    """Return full graph state for the 3D renderer.

    Combines corpus objects (cold), active frame nodes, tentative nodes,
    edges, and visual encoding fields from S11.
    """
    from ..services.embeddings import compute_x_position, compute_positions_and_vectors, compute_affinity

    # Prefer the in-memory frame — it is the live source of truth during an
    # active session. Only fall back to disk if nothing is loaded (e.g. the
    # user just re-opened a chat and the session hasn't been restored yet).
    # Reading from disk here caused a split-brain bug: active_ids came from a
    # stale snapshot while the tentative registry came from memory, which made
    # concepts from the previous turn look like they had "disappeared" whenever
    # the UI refreshed the graph before the next save commit.
    frame = frame_manager.get_frame(session_id)
    if frame is None:
        frame = session_store.load_frame(session_id)
        if frame:
            reg, edges = session_store.load_registry(session_id)
            frame_manager.restore(session_id, frame, reg or None, edges or None)
    active_ids = set(frame.active_nodes) if frame else set()

    # --- Graph visibility filter ---
    # Base-set-seeded nodes (CONSTITUTIONAL/CANONICAL slabs + their linked
    # anchors/bundles) are loaded into the frame for system-prompt building,
    # but should NOT clutter the graph until the conversation actually
    # engages them. A node becomes graph-visible when:
    #   (a) it has corpus hits (anchor matching fired on it), OR
    #   (b) it has activation sources beyond just "base_set", OR
    #   (c) it's a tentative/concept node (not base-set), OR
    #   (d) it has active conflicts or truth pressure > 0
    # This keeps the graph clean on turn 0 and lets it grow organically
    # as the conversation touches relevant concepts.
    if frame:
        graph_visible = set()
        for nid in active_ids:
            sources = frame.activation_sources.get(nid, [])
            has_base_only = (
                len(sources) > 0
                and all(s.source_type == "base_set" for s in sources)
            )
            if has_base_only:
                # Base-set node — only show if conversation has engaged it
                if frame.corpus_hits.get(nid, 0) > 0:
                    graph_visible.add(nid)
                elif frame.truth_pressure.get(nid, 0) > 0:
                    graph_visible.add(nid)
                # Otherwise: loaded in frame (for system prompt) but hidden on graph
            else:
                # Matched by anchor, concept detection, user action, etc. — always show
                graph_visible.add(nid)
    else:
        graph_visible = active_ids

    nodes = []
    texts_to_embed = []
    node_index = []

    # Collect all visible nodes: graph-visible frame nodes + 1-hop edge neighbors
    visible_ids: set[str] = set(graph_visible)

    # Add cold corpus neighbors within 1 hop of GRAPH-VISIBLE nodes only.
    # Base-set-only nodes that haven't been hit don't expand the graph.
    hop1 = set()
    for node_id in list(graph_visible):
        for edge in deps.corpus.edges.values():
            if edge.from_node == node_id: hop1.add(edge.to_node)
            elif edge.to_node == node_id: hop1.add(edge.from_node)
    # Cap expansion: at most 12 extra nodes from hop neighbors
    if len(hop1 - graph_visible) > 12:
        scored = []
        for nid in hop1 - graph_visible:
            best_w = max(
                (e.weight for e in deps.corpus.edges.values()
                 if (e.from_node == nid or e.to_node == nid)
                 and (e.from_node in graph_visible or e.to_node in graph_visible)),
                default=0
            )
            scored.append((best_w, nid))
        scored.sort(reverse=True)
        hop1 = graph_visible | {nid for _, nid in scored[:12]}
    visible_ids.update(hop1)

    # Build node list with all visual encoding fields
    for anchor in deps.corpus.anchors.values():
        if anchor.id not in visible_ids:
            continue  # Only show corpus nodes within 2 hops of active nodes
        is_active = anchor.id in active_ids
        sal_now = frame.salience_now.get(anchor.id, 0) if frame else 0
        sal_smooth = frame.salience_smoothed.get(anchor.id, 0) if frame else 0
        sw = frame.structural_weight.get(anchor.id, 0) if frame else 0

        texts_to_embed.append(anchor.canonical_phrase)
        node_index.append(len(nodes))
        nodes.append({
            "id": anchor.id,
            "type": "anchor",
            "status": "corpus",
            "label": anchor.canonical_phrase,
            "active": is_active,
            "salience_now": sal_now,
            "salience_smoothed": sal_smooth,
            "structural_weight": sw,
            "depends_on": anchor.depends_on,
            "invokes": anchor.invokes,
            "hit_count": frame.corpus_hits.get(anchor.id, 0) if frame else 0,
            "last_hit_turn": frame.corpus_last_hit.get(anchor.id, 0) if frame else 0,
            "truth_pressure": round(frame.truth_pressure.get(anchor.id, 0), 3) if frame else 0,
        })

    for slab in deps.corpus.slabs.values():
        if slab.id not in visible_ids:
            continue
        is_active = slab.id in active_ids
        sal_now = frame.salience_now.get(slab.id, 0) if frame else 0
        sal_smooth = frame.salience_smoothed.get(slab.id, 0) if frame else 0
        sw = frame.structural_weight.get(slab.id, 0) if frame else 0

        # Subgraph condensate detection: slabs produced by slabifying a
        # collection start their canonical_text with "Neighborhood: <id>".
        # See the slabify promotion builder around line 1175.
        is_subgraph = bool(
            slab.canonical_text
            and slab.canonical_text.startswith("Neighborhood: ")
        )

        texts_to_embed.append(slab.canonical_text[:120])
        node_index.append(len(nodes))
        nodes.append({
            "id": slab.id,
            "type": "slab",
            "status": "corpus",
            "label": slab.canonical_text[:60],
            "active": is_active,
            "is_subgraph": is_subgraph,
            "salience_now": sal_now,
            "salience_smoothed": sal_smooth,
            "structural_weight": sw,
            "depends_on": slab.depends_on,
            "hit_count": frame.corpus_hits.get(slab.id, 0) if frame else 0,
            "last_hit_turn": frame.corpus_last_hit.get(slab.id, 0) if frame else 0,
            "truth_pressure": round(frame.truth_pressure.get(slab.id, 0), 3) if frame else 0,
        })

    for bundle in deps.corpus.bundles.values():
        if bundle.id not in visible_ids:
            continue
        is_active = bundle.id in active_ids
        sal_now = frame.salience_now.get(bundle.id, 0) if frame else 0
        sal_smooth = frame.salience_smoothed.get(bundle.id, 0) if frame else 0
        sw = frame.structural_weight.get(bundle.id, 0) if frame else 0
        label = "; ".join(bundle.payload.intent[:2])

        texts_to_embed.append(label)
        node_index.append(len(nodes))
        nodes.append({
            "id": bundle.id,
            "type": "key_bundle",
            "status": "corpus",
            "label": label[:60],
            "active": is_active,
            "salience_now": sal_now,
            "salience_smoothed": sal_smooth,
            "structural_weight": sw,
            "depends_on": bundle.depends_on,
            "hit_count": frame.corpus_hits.get(bundle.id, 0) if frame else 0,
            "last_hit_turn": frame.corpus_last_hit.get(bundle.id, 0) if frame else 0,
            "truth_pressure": round(frame.truth_pressure.get(bundle.id, 0), 3) if frame else 0,
        })

    # Add tentative nodes from FrameState (not in corpus)
    if frame:
        tentative_registry = frame_manager.get_tentative_registry(session_id)
        for concept_name, info in tentative_registry.items():
            node_id = info["id"]
            if any(n["id"] == node_id for n in nodes):
                continue  # already added
            # Skip absorbed children (they live inside their parent bundle)
            if info.get("parent_id") and node_id not in active_ids:
                continue
            sal_now = frame.salience_now.get(node_id, 0.5)
            sal_smooth = frame.salience_smoothed.get(node_id, 0.5)
            sw = frame.structural_weight.get(node_id, 0)
            texts_to_embed.append(concept_name)
            node_index.append(len(nodes))
            # Determine semantic type from promotion state
            promoted = info.get("promoted")
            if promoted == "slab":          ntype = "slab"
            elif promoted == "anchor":      ntype = "anchor"
            elif promoted == "bundle":      ntype = "key_bundle"
            elif "slab" in node_id:         ntype = "slab"
            elif "bundle" in node_id:       ntype = "key_bundle"
            elif "anchor" in node_id:       ntype = "anchor"
            else:                           ntype = "concept"

            nodes.append({
                "id": node_id,
                "type": ntype,
                "status": "tentative",
                "label": concept_name,
                "description": info.get("description", ""),
                "active": node_id in active_ids,
                "salience_now": sal_now,
                "salience_smoothed": sal_smooth,
                "structural_weight": sw,
                "depends_on": [],
                "turns_seen": info.get("turns_seen", 1),
                "rejected": info.get("rejected", False),
                "parent_id": info.get("parent_id"),
                "children": info.get("children", []),
            })

    # Add library/tentative nodes (PROVISIONAL on disk, not in corpus).
    # Rendered with dashed amber overlay so they're visually distinct;
    # reasoner does NOT see these — they're injected only for display.
    try:
        for t in draft_manager.list_library_tentative():
            tid = t["id"]
            if any(n["id"] == tid for n in nodes):
                continue
            label = t.get("label") or tid
            ttype = t.get("type", "anchor")
            if ttype == "bundle":
                ntype = "key_bundle"
            elif ttype == "slab":
                ntype = "slab"
            else:
                ntype = "anchor"
            texts_to_embed.append(label)
            node_index.append(len(nodes))
            nodes.append({
                "id": tid,
                "type": ntype,
                "status": "tentative",
                "library_tentative": True,  # distinguish from frame-tentative
                "label": label,
                "description": t.get("justification", ""),
                "active": False,
                "salience_now": 0.0,
                "salience_smoothed": 0.0,
                "structural_weight": 0.0,
                "depends_on": [],
                "turns_seen": 0,
                "rejected": False,
                "parent_id": None,
                "children": [],
            })
    except Exception as e:
        logger.warning("Failed to load library/tentative for graph: %s", e)

    # Compute X positions + raw vectors (one embed call, both outputs)
    session_vectors = None
    if texts_to_embed:
        try:
            x_positions, session_vectors = await compute_positions_and_vectors(texts_to_embed)
        except Exception:
            import numpy as _np
            x_positions = [0.5] * len(texts_to_embed)
            session_vectors = _np.zeros((len(texts_to_embed), 1))
        for i, idx in enumerate(node_index):
            nodes[idx]["x"] = x_positions[i]
    else:
        for n in nodes:
            n["x"] = 0.5

    # Y = structural weight (community cluster proxy)
    # Z = meta depth: slab=0 (deep), key_bundle=1 (mid), anchor=2 (surface), tentative=3
    z_map = {"slab": 0, "key_bundle": 1, "anchor": 2, "concept": 3}
    for n in nodes:
        n["y"] = n["structural_weight"] * 0.5
        n["z"] = z_map.get(n["type"], 2)

    # Edges
    edges = []
    for edge in deps.corpus.edges.values():
        edges.append({
            "id": edge.id,
            "type": edge.type.value,
            "from": edge.from_node,
            "to": edge.to_node,
            "confidence": edge.confidence,
            "tension": edge.tension,
        })

    # Add structural edges from depends_on / invokes
    for n in nodes:
        for dep in n.get("depends_on", []):
            edges.append({
                "id": f"dep_{n['id']}_{dep}",
                "type": "LINKS",
                "from": n["id"],
                "to": dep,
                "confidence": 0.8,
                "tension": None,
            })
        for inv in n.get("invokes", []):
            edges.append({
                "id": f"inv_{n['id']}_{inv}",
                "type": "INVOKES",
                "from": n["id"],
                "to": inv,
                "confidence": 0.9,
                "tension": None,
            })

    # Add tentative edges (session-scoped, not in corpus).
    # `tentative: True` is the authoritative commitment flag the renderer
    # keys off — do NOT rely on endpoint-node status to infer tentativeness,
    # because a tentative edge between two corpus endpoints is a legal shape
    # (e.g. user drafts a CONFLICTS between two committed anchors) and the
    # endpoints carry no information about the edge's commitment state.
    tent_edges = frame_manager.get_tentative_edges(session_id)
    for i, te in enumerate(tent_edges):
        edges.append({
            "id": f"tent_{i}_{te['from'][:8]}_{te['to'][:8]}",
            "type": te.get("type", "PARENT_OF"),
            "from": te["from"],
            "to": te["to"],
            "confidence": te.get("strength", 0.7),
            "tension": None,
            "tentative": True,
        })

    # -- Embedding-similarity affinity for session graph --
    # Shares the same attraction logic as the cold corpus: top-K cosine
    # neighbours become soft springs in the frontend physics so semantically
    # related nodes drift together even without explicit edges.
    affinity_pairs: list[dict] = []
    if session_vectors is not None and len(node_index) == session_vectors.shape[0] and len(node_index) > 1:
        try:
            aligned_ids: list[str] = [nodes[i]["id"] for i in node_index]
            aligned_types: list[str] = [nodes[i]["type"] for i in node_index]
            # Normalise type names so the helper recognises slab/anchor/bundle.
            type_alias = {"key_bundle": "bundle"}
            aligned_types = [type_alias.get(t, t) for t in aligned_types]
            _, affinity_pairs = compute_affinity(
                aligned_ids, aligned_types, session_vectors,
                parent_threshold=0.70, pair_threshold=0.65, top_k_pairs=3,
            )
        except Exception as e:
            logger.warning("Affinity computation failed for session graph: %s", e)

    return {
        "nodes": nodes,
        "edges": edges,
        "affinity_pairs": affinity_pairs,
        "frame": {
            "active_nodes": list(active_ids),
            "mismatch_score": frame.mismatch_score if frame else 0,
            "current_turn": frame.last_updated_turn if frame else 0,
            "conflicts": [c.model_dump() for c in frame.conflicts] if frame else [],
        },
    }
