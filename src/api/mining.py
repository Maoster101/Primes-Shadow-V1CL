"""Mining endpoints — conversation and narrative mining into corpus proposals."""
from __future__ import annotations
import asyncio
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..models.schemas import DraftPacket, DraftStack, ProposedEdge
from ..models.enums import DraftStatus, EdgeType
from ..services.convo_miner import ConversationMiner
from ..services.narrative_miner import NarrativeMiner

from . import deps
from .deps import (
    frame_manager, session_store, draft_manager,
    chat_store, registry, event_log, rebind_corpus,
)

router = APIRouter()


# --- Request models ---

class MineRequest(BaseModel):
    text: str                          # Raw conversation export content
    source_label: str = "import"       # Provenance label
    min_confidence: float = 0.4        # Minimum proposal confidence
    chunk_size: int = 4                # Exchange pairs per chunk
    target_collection: str = "default" # Which collection to mine into


class MineNarrativeRequest(BaseModel):
    text: str                          # Cohesive narrative / document content
    source_label: str = "narrative"    # Provenance label
    min_confidence: float = 0.4        # Minimum proposal confidence
    max_segment_chars: int = 1800      # Soft cap on segment size
    target_collection: str = "default" # Which collection to mine into


class PushMinedRequest(BaseModel):
    proposals: list[dict]           # Raw mined proposal dicts from /mine response
    edges: list[dict] = []          # Raw mined edge dicts from /mine response (Phase 3)
    target_collection: str = "default"  # Which collection to commit into


class ScratchSessionRequest(BaseModel):
    title: str = "Mining Session"
    collection_id: str = "default"


# --- Endpoints ---

@router.post("/mine")
async def mine_conversation(req: MineRequest):
    """Mine a conversation export for corpus proposals.

    Accepts raw text (Claude JSON, ChatGPT JSON, markdown, or plaintext).
    Returns detected format, topic distribution, and proposed corpus objects.
    """
    if not req.text.strip():
        raise HTTPException(400, "Empty text")
    if len(req.text) > 5_000_000:
        raise HTTPException(413, "Text too large (max 5MB)")

    # Use target collection's store for existing-anchor dedup
    target_store = registry.get_store(req.target_collection) or deps.corpus
    miner = ConversationMiner(target_store)

    result = await miner.mine(
        raw_text=req.text,
        source_label=req.source_label,
        min_confidence=req.min_confidence,
        chunk_size=req.chunk_size,
    )
    result["target_collection"] = req.target_collection
    return result


@router.post("/mine-narrative")
async def mine_narrative(req: MineNarrativeRequest):
    """Mine a cohesive narrative / document for corpus proposals.

    Treats the input as a single-author piece where paragraph ordering is
    load-bearing (stories, essays, design docs, condensed pitches). Emits
    SEQUENCE edges between consecutive beats so the narrative spine is
    preserved in the corpus graph. Output shape matches /mine so the same
    /push-mined flow accepts its proposals.
    """
    if not req.text.strip():
        raise HTTPException(400, "Empty text")
    if len(req.text) > 5_000_000:
        raise HTTPException(413, "Text too large (max 5MB)")

    target_store = registry.get_store(req.target_collection) or deps.corpus
    miner = NarrativeMiner(target_store)

    result = await miner.mine(
        raw_text=req.text,
        source_label=req.source_label,
        min_confidence=req.min_confidence,
        max_segment_chars=req.max_segment_chars,
    )
    result["target_collection"] = req.target_collection
    return result


@router.post("/sessions/{session_id}/push-mined")
async def push_mined_proposals(session_id: str, req: PushMinedRequest):
    """Push mined proposals directly into the session draft stack.

    Converts raw mined proposals (from /mine) into DraftPackets without
    re-running LLM extraction. Each proposal becomes a DRAFT_UNAUTHORIZED
    packet with its raw data preserved for later review/promotion.
    """
    meta_path = session_store.session_dir(session_id) / "meta.json"
    meta = session_store.read_json(meta_path)
    if not meta:
        raise HTTPException(404, "Session not found")

    stack = session_store.load_draft_stack(session_id)
    if not stack:
        stack = DraftStack(session_id=session_id)

    created = []
    created_packets = []
    for prop in req.proposals:
        if not isinstance(prop, dict):
            continue
        prop_type = prop.get("type", "anchor")
        text = prop.get("canonical_phrase") or prop.get("canonical_text", "") or prop.get("title", "")
        if not text:
            continue

        draft_id = f"mined_{prop_type}_{uuid.uuid4().hex[:8]}_v1"

        # Build inline typed payload so the packet is self-contained
        # (same pattern as DraftManager.extract_proposals).
        inline_anchor = None
        inline_slab = None
        if prop_type == "anchor":
            inline_anchor = {
                "id": draft_id,
                "canonical_phrase": prop.get("canonical_phrase", "") or "",
                "aliases": list(prop.get("aliases") or []),
                "invokes": [],
                "notes": prop.get("justification", "") or "",
            }
        elif prop_type == "slab":
            inline_slab = {
                "id": draft_id,
                "title": prop.get("title") or prop.get("canonical_phrase") or "",
                "canonical_text": prop.get("canonical_text", "") or "",
                "links": {"anchors": [], "bundles": []},
                "version": "v1",
            }

        packet = DraftPacket(
            id=draft_id,
            packet_type=prop_type,
            source_chat_id=meta.get("chat_id", session_id),
            source_turns=[0],
            proposed_nodes=[draft_id],
            justification=prop.get("justification", ""),
            confidence=prop.get("confidence", 0.5),
            status=DraftStatus.DRAFT_UNAUTHORIZED,
            fact_claims=[text] if prop.get("claim_tag") == "FACT" else [],
            anchor=inline_anchor,
            slab=inline_slab,
        )

        session_store.save_draft_packet(session_id, packet)
        # Store raw proposal + target collection for commit step
        prop["_target_collection"] = req.target_collection
        raw_path = session_store.drafts_dir(session_id) / f"{draft_id}_raw.json"
        session_store.write_json(raw_path, prop)

        stack.packets.append(draft_id)
        created.append({"id": draft_id, "type": prop_type, "label": text[:80]})
        created_packets.append(packet)

    session_store.save_draft_stack(session_id, stack)

    # Phase 3 — resolve & persist mined edges into the ProposedEdge pipeline.
    # Convo/narrative miners emit edges with keys {type, from, to, confidence,
    # justification}. We reuse the same label index strategy as the chat path:
    # just-pushed drafts (canonical_phrase / title / aliases) take priority,
    # then live corpus anchors/slabs.
    edges_persisted = 0
    if req.edges:
        try:
            label_to_id: dict[str, str] = {}
            def _reg(label: str, nid: str) -> None:
                if not label: return
                label_to_id.setdefault(label.strip().lower(), nid)

            for pkt in created_packets:
                if pkt.anchor:
                    _reg(pkt.anchor.get("canonical_phrase", ""), pkt.id)
                    for al in pkt.anchor.get("aliases") or []:
                        _reg(al, pkt.id)
                if pkt.slab:
                    _reg(pkt.slab.get("title", ""), pkt.id)
                if pkt.bundle:
                    _reg(pkt.bundle.get("id", ""), pkt.id)

            # Corpus endpoints (use target collection's store for consistency with
            # how dedup was run during mining).
            tgt_store = registry.get_store(req.target_collection) or deps.corpus
            for a in tgt_store.anchors.values():
                _reg(a.canonical_phrase, a.id)
                for al in a.aliases or []:
                    _reg(al, a.id)
            for s in tgt_store.slabs.values():
                if getattr(s, "title", ""):
                    _reg(s.title, s.id)

            resolved: list = []
            chat_id_for_edges = meta.get("chat_id", session_id)
            for spec in req.edges:
                if not isinstance(spec, dict):
                    continue
                etype_raw = (spec.get("type") or "LINKS").upper()
                if etype_raw not in {"INVOKES", "SUPPORTS", "CONFLICTS", "TENSIONS", "LINKS", "SEQUENCE", "PARENT_OF"}:
                    etype_raw = "LINKS"
                # Convo miner emits from/to; chat path emits from_label/to_label.
                from_label = (spec.get("from_label") or spec.get("from") or "").strip()
                to_label = (spec.get("to_label") or spec.get("to") or "").strip()
                if not from_label or not to_label:
                    continue
                from_id = label_to_id.get(from_label.lower())
                to_id = label_to_id.get(to_label.lower())
                if not from_id or not to_id or from_id == to_id:
                    if not from_id:
                        print(f"[PUSH-MINED] Edge dropped: from_label={from_label!r} not in label index ({len(label_to_id)} entries)", flush=True)
                    if not to_id:
                        print(f"[PUSH-MINED] Edge dropped: to_label={to_label!r} not in label index", flush=True)
                    continue
                try:
                    confidence = float(spec.get("confidence", 0.5))
                except Exception:
                    confidence = 0.5
                resolved.append(ProposedEdge(
                    id=f"proposed_edge_{uuid.uuid4().hex[:8]}",
                    type=EdgeType(etype_raw),
                    from_node=from_id, to_node=to_id,
                    from_label=from_label, to_label=to_label,
                    confidence=max(0.0, min(1.0, confidence)),
                    justification=spec.get("justification", "") or "",
                    status="PROPOSED",
                    source_chat_id=chat_id_for_edges,
                    source_turn=0,
                ))
            if resolved:
                session_store.append_proposed_edges(session_id, resolved)
                edges_persisted = len(resolved)
        except Exception as e:
            print(f"[PUSH-MINED] Edge resolution failed: {e}", flush=True)

    # Auto-trigger dreaming on pushed drafts (background).
    # Uses dream_all_pending so the concurrent path (PS_DREAMING_PARALLEL)
    # applies — a 5-draft push gets ~3 effective rounds at parallel=2
    # rather than 5 serial calls.
    if created_packets:
        async def _dream_pushed():
            try:
                from ..services.dreaming import dream_all_pending
                await dream_all_pending(
                    deps.corpus, session_store, chat_store, session_id,
                )
            except Exception as e:
                print(f"[DREAM] Dreaming pass failed: {e}", flush=True)
        asyncio.create_task(_dream_pushed())

    # --- Inject mined proposals into frame_manager's tentative registry ---
    # Without this, /sessions/{id}/graph can't see the mined nodes because
    # it reads from the in-memory registry, not from disk DraftPackets.
    # Mining intentionally uses a base-set-free frame (ensure_empty_frame
    # inside the verb) so proposals aren't polluted with pre-loaded slabs.
    if created_packets:
        frame_manager.inject_mined_proposals(
            session_id, meta.get("chat_id", session_id), created_packets
        )

    return {"created": len(created), "drafts": created, "edges_persisted": edges_persisted}


@router.post("/sessions/scratch")
async def create_scratch_session(req: ScratchSessionRequest):
    """Create a lightweight chat + session without running inference.

    Used by the mining UI to get a session_id instantly — no SSE stream,
    no model warm-up, no dummy turn. The chat is created in ACTIVE state
    with an empty message history, and a session is bound to it immediately.
    """
    chat = chat_store.create_chat(req.title, collection_id=req.collection_id)
    session_id = session_store.create_session(chat.id)
    return {
        "chat_id": chat.id,
        "session_id": session_id,
        "collection_id": req.collection_id,
    }
