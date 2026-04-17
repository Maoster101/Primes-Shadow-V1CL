"""Draft-related API routes — drafts, proposed edges, library/tentative,
end-review, dream-all, and extract-proposals."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import deps
from .deps import (
    frame_manager, session_store, draft_manager, anchor_matcher,
    drift_monitor, chat_store, registry, event_log,
    rebind_corpus, save_session_state,
)
from ..models.enums import OLIMode, ChatStatus, DraftStatus

logger = logging.getLogger(__name__)

router = APIRouter()


# --- Request models ---

class ReviewDraftRequest(BaseModel):
    action: str  # "discard" | "promote_tentative" | "promote_corpus"
    oli_mode: str = "OFF"


class AmendDraftRequest(BaseModel):
    canonical_phrase: Optional[str] = None
    canonical_text: Optional[str] = None
    justification: Optional[str] = None
    aliases: Optional[list[str]] = None


class ExtractProposalsRequest(BaseModel):
    explicit: bool = True
    user_request: str = ""


class VerifyClaimsRequest(BaseModel):
    context: str = ""
    method: str = "ollama"  # "manual" | "ollama" | "external"


# --- Helpers ---

def _draft_collection_id(session_id: str, draft_id: str) -> str:
    """Read the target collection tag from the raw sidecar.

    Every draft (AI-Mine-pushed or chat-mined) has a `{id}_raw.json` sidecar.
    AI-Mine pushes stamp `_target_collection` there; chat-mined drafts omit
    it and default to "default". This is read at list/display time rather
    than stored as a packet field — keeps schema stable and avoids a
    migration. Promotion logic (draft_manager._convert_to_corpus_object's
    caller) uses the same key.
    """
    try:
        raw_path = session_store.drafts_dir(session_id) / f"{draft_id}_raw.json"
        raw = session_store.read_json(raw_path) or {}
        return raw.get("_target_collection", "default") or "default"
    except Exception:
        return "default"


def _find_draft_session(draft_id: str) -> Optional[str]:
    """Locate which session a draft_id lives in by scanning session dirs.

    Needed because chat-scoped listing aggregates across sessions, but the
    raw sidecar (collection tag, etc.) lives in the session that created
    the draft. Returns None if not found.
    """
    try:
        for sess_dir in session_store.root.iterdir():
            if not sess_dir.is_dir():
                continue
            if (sess_dir / "drafts" / f"{draft_id}.json").exists():
                return sess_dir.name
    except Exception:
        pass
    return None


# --- Endpoints ---

@router.get("/sessions/{session_id}/drafts")
async def list_drafts(
    session_id: str,
    include_resolved: bool = False,
    collection_id: Optional[str] = None,
):
    """List session drafts.

    Default: only pending (DRAFT_UNAUTHORIZED) drafts — the worklist.
    Set ?include_resolved=true to see the full history including
    PROVISIONAL, COMMITTED, and REJECTED packets.

    Pass ?collection_id=X to filter to drafts whose `_target_collection`
    matches — used by the Drafts/Verify tabs when the Active Corpus
    dropdown is pinned to a specific collection.
    """
    drafts = draft_manager.list_drafts(session_id, include_resolved=include_resolved)
    out = []
    for d in drafts:
        dump = d.model_dump(mode="json")
        cid = _draft_collection_id(session_id, d.id)
        dump["collection_id"] = cid
        if collection_id and cid != collection_id:
            continue
        out.append(dump)
    return out


@router.get("/chats/{chat_id}/drafts")
async def list_drafts_by_chat(
    chat_id: str,
    include_resolved: bool = False,
    collection_id: Optional[str] = None,
):
    """List ALL drafts for a chat across every session it ever had.

    Drafts are stored per-session on disk, but a chat can span many SSE
    sessions (restarts, reconnects). UI reviewers think in chat scope,
    not session scope, so this aggregates. Each returned packet also
    carries a `session_id` field indicating where its sidecars live (used
    by downstream promote/verify/amend endpoints which remain session-
    scoped, so callers route actions to the right directory).
    """
    drafts = draft_manager.list_drafts_by_chat(chat_id, include_resolved=include_resolved)
    out = []
    for d in drafts:
        sess = _find_draft_session(d.id)
        if not sess:
            continue  # ghost reference, sidecar missing
        dump = d.model_dump(mode="json")
        cid = _draft_collection_id(sess, d.id)
        dump["collection_id"] = cid
        dump["session_id"] = sess
        if collection_id and cid != collection_id:
            continue
        out.append(dump)
    return out


@router.get("/chats/{chat_id}/proposed_edges")
async def list_proposed_edges_by_chat(chat_id: str, include_resolved: bool = False):
    """List proposed edges across every session this chat ever had.

    Mirrors /chats/{id}/drafts — edges are stored per-session but reviewers
    think in chat scope. Each row carries its owning session_id so the
    review endpoint can route the status update back correctly.

    include_resolved=False filters out COMMITTED/REJECTED (default review view).
    """
    pairs = session_store.list_proposed_edges_by_chat(chat_id)
    out = []
    for sid, edge in pairs:
        if not include_resolved and edge.status in ("COMMITTED", "REJECTED"):
            continue
        d = edge.model_dump(mode="json")
        d["session_id"] = sid
        out.append(d)
    return out


@router.post("/sessions/{session_id}/proposed_edges/{edge_id}/review")
async def review_proposed_edge(session_id: str, edge_id: str, req: dict):
    """Accept or reject a proposed edge.

    Body: {"action": "accept" | "reject"}

    On accept: if both endpoints already exist in corpus, promote directly
    to a real Edge (status -> COMMITTED). Otherwise mark ACCEPTED and wait
    for a later promote hook (when the underlying drafts land in corpus).

    On reject: mark REJECTED; kept for audit but never committed.
    """
    action = (req.get("action") or "").lower()
    if action not in ("accept", "reject"):
        raise HTTPException(400, "action must be 'accept' or 'reject'")

    edges = session_store.list_proposed_edges(session_id)
    target = next((e for e in edges if e.id == edge_id), None)
    if not target:
        raise HTTPException(404, "Proposed edge not found")

    if action == "reject":
        session_store.update_proposed_edge_status(session_id, edge_id, "REJECTED")
        return {"ok": True, "status": "REJECTED"}

    # Accept path — can we commit now?
    from ..models.schemas import Edge as CorpusEdge
    both_live = (
        (target.from_node in deps.corpus.anchors or target.from_node in deps.corpus.slabs
         or target.from_node in deps.corpus.bundles)
        and
        (target.to_node in deps.corpus.anchors or target.to_node in deps.corpus.slabs
         or target.to_node in deps.corpus.bundles)
    )
    if both_live:
        import uuid
        new_edge_id = f"edge_{uuid.uuid4().hex[:8]}"
        real = CorpusEdge(
            id=new_edge_id,
            type=target.type,
            **{"from": target.from_node, "to": target.to_node},
            weight=target.confidence,
            confidence=target.confidence,
        )
        # Resolve the owning collection via the source chat. Writing to the
        # merged virtual store silently misroutes the save to the legacy
        # CORPUS_ROOT (app/corpus) — the per-collection stores never see it,
        # and on reload the edge vanishes. Instead, write to the specific
        # collection's store so it lands in app/corpora/{cid}/objects/edges.yaml
        # and persists. Fall back to merged only if chat/collection lookup fails
        # (best-effort; should log a warning).
        target_store = None
        try:
            if target.source_chat_id:
                _chat = chat_store.get_chat(target.source_chat_id)
                if _chat and _chat.collection_id:
                    target_store = registry.get_store(_chat.collection_id)
        except Exception as exc:
            logger.warning("Edge commit: chat→collection lookup failed: %r", exc)
        if target_store is None:
            # Last resort: pick 'default' so we at least hit a real store.
            target_store = registry.get_store("default")
            logger.warning(
                "Edge commit for %s: no owning collection resolved, writing to 'default'",
                edge_id,
            )
        target_store.edges[new_edge_id] = real
        target_store.save()
        # Invalidate merged view so subsequent reads see the new edge.
        registry._merged_dirty = True
        deps.rebind_corpus()
        session_store.update_proposed_edge_status(
            session_id, edge_id, "COMMITTED", committed_edge_id=new_edge_id,
        )
        return {"ok": True, "status": "COMMITTED", "committed_edge_id": new_edge_id}

    # Endpoints not yet live — mark ACCEPTED, promote later
    session_store.update_proposed_edge_status(session_id, edge_id, "ACCEPTED")
    return {"ok": True, "status": "ACCEPTED", "note": "Awaiting endpoint promotion"}


@router.get("/sessions/{session_id}/drafts/{draft_id}")
async def get_draft(session_id: str, draft_id: str):
    packet = session_store.load_draft_packet(session_id, draft_id)
    if not packet:
        raise HTTPException(404, "Draft not found")
    # Also load raw proposal for context
    raw_path = session_store.drafts_dir(session_id) / f"{draft_id}_raw.json"
    raw = session_store.read_json(raw_path) or {}
    return {
        "draft": packet.model_dump(mode="json"),
        "raw_proposal": raw,
        "collection_id": raw.get("_target_collection", "default") or "default",
    }


@router.get("/sessions/{session_id}/drafts/{draft_id}/preview")
async def commit_preview(session_id: str, draft_id: str):
    """§15.3 — Compute commit preview data for the modal.

    Returns everything the user needs to review before committing.
    """
    # Load draft packet + raw proposal
    packet = session_store.load_draft_packet(session_id, draft_id)
    if not packet:
        raise HTTPException(404, "Draft not found")
    raw_path = session_store.drafts_dir(session_id) / f"{draft_id}_raw.json"
    raw = session_store.read_json(raw_path) or {}

    # 1. Tentative insertion — what's being committed
    insertion = {
        "id": draft_id,
        "type": raw.get("type", "anchor"),
        "label": raw.get("canonical_phrase") or raw.get("canonical_text", "")[:60],
        "status": packet.status.value,
        "justification": packet.justification,
    }

    # 2. Required dependency closure — what gets pulled in
    closure = []
    for dep_id in packet.dependency_closure:
        if dep_id in deps.corpus.anchors:
            closure.append({"id": dep_id, "type": "anchor", "label": deps.corpus.anchors[dep_id].canonical_phrase})
        elif dep_id in deps.corpus.bundles:
            closure.append({"id": dep_id, "type": "bundle", "label": "; ".join(deps.corpus.bundles[dep_id].payload.intent[:2])})
        elif dep_id in deps.corpus.slabs:
            closure.append({"id": dep_id, "type": "slab", "label": deps.corpus.slabs[dep_id].title})

    # 3. Cold committed neighbourhood — what's already near
    neighbourhood = []
    all_ids = deps.corpus.all_ids()
    related_ids = set()
    for edge in deps.corpus.edges.values():
        for dep in packet.dependency_closure + [draft_id]:
            if edge.from_node == dep: related_ids.add(edge.to_node)
            elif edge.to_node == dep: related_ids.add(edge.from_node)
    for nid in related_ids:
        if nid in deps.corpus.anchors:
            neighbourhood.append({"id": nid, "type": "anchor", "label": deps.corpus.anchors[nid].canonical_phrase})
        elif nid in deps.corpus.bundles:
            neighbourhood.append({"id": nid, "type": "bundle", "label": "; ".join(deps.corpus.bundles[nid].payload.intent[:2])})
        elif nid in deps.corpus.slabs:
            neighbourhood.append({"id": nid, "type": "slab", "label": deps.corpus.slabs[nid].title})

    # 4. Conflict / cascade preview — what might break
    cascade_risk = []
    for dep_id in packet.dependency_closure + [draft_id]:
        rev_deps = deps.corpus.get_reverse_deps(dep_id)
        if rev_deps:
            cascade_risk.append({"node_id": dep_id, "dependents": rev_deps})

    # 5. Verification status for FACT claims
    fact_claims = packet.fact_claims or []
    fact_status = [{"claim": c, "verified": False} for c in fact_claims]
    has_unverified = len(fact_claims) > 0

    # 6. Drift disclosure
    drift_info = None
    if session_id in drift_monitor._windows:
        window = drift_monitor._windows[session_id]
        frame = frame_manager.get_frame(session_id)
        if frame:
            drift_result = window.compute(frame.last_updated_turn)
            if drift_result["severity"].value != "low":
                drift_info = {
                    "severity": drift_result["severity"].value,
                    "composite": drift_result["composite"],
                    "dampening": drift_result["dampening"].value,
                }

    # 7. Validation — run corpus validator dry-run
    validation_errors = []
    obj = draft_manager._convert_to_corpus_object(draft_id, raw)
    if obj:
        obj_type, corpus_obj = obj
        # Temporarily add to check validation
        if obj_type == "anchor":
            deps.corpus.anchors[corpus_obj.id] = corpus_obj
        elif obj_type == "slab":
            deps.corpus.slabs[corpus_obj.id] = corpus_obj
        elif obj_type == "bundle":
            deps.corpus.bundles[corpus_obj.id] = corpus_obj
        validation_errors = deps.corpus.validate()
        # Rollback
        if obj_type == "anchor": deps.corpus.anchors.pop(corpus_obj.id, None)
        elif obj_type == "slab": deps.corpus.slabs.pop(corpus_obj.id, None)
        elif obj_type == "bundle": deps.corpus.bundles.pop(corpus_obj.id, None)

    return {
        "insertion": insertion,
        "dependency_closure": closure,
        "neighbourhood": neighbourhood,
        "cascade_risk": cascade_risk,
        "fact_claims": fact_status,
        "has_unverified_facts": has_unverified,
        "drift_disclosure": drift_info,
        "validation_errors": validation_errors,
        "can_commit": not has_unverified and len(validation_errors) == 0,
    }


@router.post("/sessions/{session_id}/drafts/{draft_id}/amend")
async def amend_draft(session_id: str, draft_id: str, req: AmendDraftRequest):
    """§14.2 — Amend a tentative draft. Updates the raw proposal fields
    and resets the packet status to DRAFT_UNAUTHORIZED so it must be
    re-reviewed before any commit/stage action can be taken.
    """
    packet = session_store.load_draft_packet(session_id, draft_id)
    if not packet:
        raise HTTPException(404, "Draft not found")
    raw_path = session_store.drafts_dir(session_id) / f"{draft_id}_raw.json"
    raw = session_store.read_json(raw_path) or {}

    if req.canonical_phrase is not None:
        raw["canonical_phrase"] = req.canonical_phrase
    if req.canonical_text is not None:
        raw["canonical_text"] = req.canonical_text
    if req.justification is not None:
        raw["justification"] = req.justification
        packet.justification = req.justification
    if req.aliases is not None:
        raw["aliases"] = req.aliases

    session_store.write_json(raw_path, raw)

    # Reset to unauthorized — must be re-reviewed
    packet.status = DraftStatus.DRAFT_UNAUTHORIZED
    session_store.save_draft_packet(session_id, packet)

    return {
        "status": "AMENDED",
        "draft_id": draft_id,
        "packet_status": packet.status.value,
        "raw": raw,
    }


# ─── Library: tentative (PROVISIONAL on disk, not in corpus) ────────────
@router.get("/library/tentative")
async def list_library_tentative():
    """List every JSON record in app/library/tentative/.

    These are drafts promoted as PROVISIONAL — preserved on disk but not
    part of the active corpus. The reasoner does not see them.
    """
    items = draft_manager.list_library_tentative()
    return {"items": items, "count": len(items)}


@router.get("/library/tentative/{tid}")
async def get_library_tentative(tid: str):
    items = draft_manager.list_library_tentative()
    for it in items:
        if it["id"] == tid:
            return it
    raise HTTPException(404, "Tentative record not found")


@router.delete("/library/tentative/{tid}")
async def delete_library_tentative(tid: str):
    ok = draft_manager.delete_library_tentative(tid)
    if not ok:
        raise HTTPException(404, "Tentative record not found")
    return {"status": "deleted", "id": tid}


@router.post("/library/tentative/{tid}/promote")
async def promote_library_tentative(tid: str):
    """Commit a tentative record into the active corpus and remove it from the library."""
    result = draft_manager.promote_library_tentative(tid)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@router.post("/sessions/{session_id}/drafts/{draft_id}/review")
async def review_draft(session_id: str, draft_id: str, req: ReviewDraftRequest):
    # Compute current drift severity for the drift gate
    drift_severity = "low"
    window = drift_monitor.get_window(session_id)
    frame = frame_manager.get_frame(session_id)
    if frame and frame.last_updated_turn > 0:
        drift_state = window.compute(frame.last_updated_turn)
        drift_severity = drift_state["severity"].value

    result = await draft_manager.review_draft(
        session_id, draft_id, req.action, req.oli_mode,
        drift_severity=drift_severity,
    )
    if "error" in result:
        raise HTTPException(400, result)
    # If committed to corpus, warm the anchor cache
    if result.get("status") == "COMMITTED" and "anchor" in draft_id:
        await anchor_matcher.warm_cache()
    return result


@router.post("/sessions/{session_id}/drafts/{draft_id}/verify")
async def verify_draft_claims(session_id: str, draft_id: str, req: VerifyClaimsRequest):
    """Verify FACT claims in a draft before promotion.

    Runs claims through the verification router and returns outcomes.
    """
    result = await draft_manager.verify_draft_claims(
        session_id, draft_id, context=req.context, method=req.method,
    )
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@router.post("/sessions/{session_id}/drafts/{draft_id}/dream")
async def dream_draft(session_id: str, draft_id: str):
    """Run the dreaming pass (grounding audit + content rewrite) on a draft.

    This is the middle layer between mining and commit. It checks whether
    the miner's output is grounded in actual source code, and if not,
    rewrites the content fields to match reality.

    The enriched output is written as {draft_id}.enriched.json alongside
    the original packet. It is NOT auto-promoted — the reviewer must
    still explicitly accept it before commit.
    """
    from ..services.dreaming import DreamingPass
    dreamer = DreamingPass(deps.corpus, session_store, chat_store)
    result = await dreamer.dream(session_id, draft_id)
    if result.get("status") == "error":
        raise HTTPException(400, result)
    return result


@router.post("/sessions/{session_id}/dream-all")
async def dream_all(session_id: str):
    """Run the dreaming pass on all pending drafts in a session.

    Skips drafts that already have an .enriched.json file.
    Returns a list of results, one per draft processed.
    """
    from ..services.dreaming import dream_all_pending
    results = await dream_all_pending(
        deps.corpus, session_store, chat_store, session_id,
    )
    return {"session_id": session_id, "results": results}


@router.post("/sessions/{session_id}/extract-proposals")
async def extract_proposals(session_id: str, req: ExtractProposalsRequest):
    """Manual proposal extraction trigger."""
    meta_path = session_store.session_dir(session_id) / "meta.json"
    meta = session_store.read_json(meta_path)
    if not meta:
        raise HTTPException(404, "Session not found")
    chat_id = meta["chat_id"]
    messages = chat_store.get_messages(chat_id)
    recent = [
        {"role": m.role, "content": m.content, "turn": m.turn}
        for m in messages[-12:]
    ]
    turn = len(messages)
    drafts = await draft_manager.extract_proposals(
        session_id, chat_id, recent, turn,
        explicit=req.explicit, user_request=req.user_request,
    )

    # Auto-trigger dreaming on newly mined drafts (background)
    if drafts:
        async def _dream_extracted():
            try:
                from ..services.dreaming import DreamingPass
                dreamer = DreamingPass(deps.corpus, session_store, chat_store)
                for packet in drafts:
                    try:
                        await dreamer.dream(session_id, packet.id)
                    except Exception as e:
                        print(f"[DREAM] Error dreaming {packet.id}: {e}", flush=True)
            except Exception as e:
                print(f"[DREAM] Dreaming pass failed: {e}", flush=True)
        asyncio.create_task(_dream_extracted())

    return [d.model_dump(mode="json") for d in drafts]


@router.get("/sessions/{session_id}/end-review")
async def end_session_review(session_id: str):
    """§14.2 — End-of-session promotion flow.

    Returns the full draft stack for batch review plus the per-turn corpus
    access trajectory (for the conversation thread path animation).

    Everything the user needs to commit, stage, or discard the session's
    tentative work is packaged in a single call.
    """
    # Ensure frame is loaded into memory so we have access to hit log
    frame = frame_manager.get_frame(session_id)
    if not frame:
        frame = session_store.load_frame(session_id)
        if frame:
            reg, edges = session_store.load_registry(session_id)
            frame_manager.restore(session_id, frame, reg or None, edges or None)

    # 0. Run dreaming pass on any un-enriched pending drafts.
    #    This is the natural "enrich everything before review" trigger —
    #    by the time the user sees the review panel, every draft has been
    #    audited and (if needed) rewritten with grounded content.
    from ..services.dreaming import dream_all_pending
    dream_results = await dream_all_pending(
        deps.corpus, session_store, chat_store, session_id,
    )
    dream_summary = {}
    for dr in dream_results:
        dream_summary[dr.get("draft_id", "")] = {
            "status": dr.get("status"),
            "grounding_mode": dr.get("grounding_mode"),
            "verdict": dr.get("audit", {}).get("verdict") if dr.get("audit") else None,
        }

    # 1. Full draft stack — every packet created this session, with raw
    #    proposal data so the panel can render without a second round-trip.
    #    Reload packets after dreaming so enriched justifications appear.
    drafts_out = []
    drafts = draft_manager.list_drafts(session_id, include_resolved=True)
    for packet in drafts:
        raw_path = session_store.drafts_dir(session_id) / f"{packet.id}_raw.json"
        raw = session_store.read_json(raw_path) or {}
        enriched_path = session_store.drafts_dir(session_id) / f"{packet.id}.enriched.json"
        has_enriched = enriched_path.exists()
        # Prefer enriched inline content for label/text if available
        inline = packet.anchor or packet.slab or {}
        label = (
            inline.get("canonical_phrase")
            or inline.get("title")
            or raw.get("canonical_phrase")
            or raw.get("canonical_text", "")[:80]
        )
        drafts_out.append({
            "id": packet.id,
            "type": packet.packet_type or raw.get("type", "anchor"),
            "label": label,
            "canonical_text": inline.get("canonical_text") or raw.get("canonical_text", ""),
            "justification": packet.justification or raw.get("justification", ""),
            "status": packet.status.value,
            "confidence": packet.confidence,
            "source_turns": packet.source_turns,
            "fact_claims": packet.fact_claims,
            "has_unverified_facts": len(packet.fact_claims) > 0,
            "has_enriched": has_enriched,
            "dream": dream_summary.get(packet.id),
        })

    # 2. Corpus hit trajectory — per-turn [turn, [node_ids]] sequence
    trajectory = []
    if frame:
        trajectory = [[entry[0], list(entry[1])] for entry in frame.corpus_hit_log]

    # 3. Session summary stats
    summary = {
        "session_id": session_id,
        "last_turn": frame.last_updated_turn if frame else 0,
        "total_hits": sum(frame.corpus_hits.values()) if frame else 0,
        "unique_nodes_hit": len(frame.corpus_hits) if frame else 0,
        "draft_count": len(drafts_out),
        "draft_by_status": {},
    }
    for d in drafts_out:
        summary["draft_by_status"][d["status"]] = summary["draft_by_status"].get(d["status"], 0) + 1

    # 4. Record session duration for adaptive baseline
    from ..services.drift_monitor import record_session_end
    if frame and frame.last_updated_turn > 0:
        record_session_end(session_id, frame.last_updated_turn)

    return {
        "summary": summary,
        "drafts": drafts_out,
        "trajectory": trajectory,
    }
