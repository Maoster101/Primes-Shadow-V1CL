"""Chat CRUD and messaging endpoints — extracted from routes.py."""
from __future__ import annotations
import asyncio
import json
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..models.schemas import ChatMessage, MessageClassification, DriftEstimate
from ..models.enums import ChatStatus, OLIMode
from ..services.pipeline import process_turn

from . import deps
from .deps import (
    chat_store, session_store, frame_manager, anchor_matcher, slab_matcher,
    draft_manager, drift_monitor, gauntlet_engine, event_log, registry,
    save_session_state,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# --- Request models ---

class CreateChatRequest(BaseModel):
    title: str = "New Chat"
    collection_id: Optional[str] = None  # Phase 2A: binds chat-mined drafts to this collection


class SendMessageRequest(BaseModel):
    content: str
    oli_mode: str = "OFF"
    web_mode: str = "off"  # "off" | "on" | "auto"
    think_level: str = "medium"  # "off" | "low" | "medium" | "high"


class UpdateChatStatusRequest(BaseModel):
    status: ChatStatus


# --- Chat CRUD ---

@router.post("/chats")
async def create_chat(req: CreateChatRequest):
    chat = chat_store.create_chat(req.title, collection_id=req.collection_id)
    return chat.model_dump(mode="json")


@router.patch("/chats/{chat_id}/collection")
async def update_chat_collection(chat_id: str, req: CreateChatRequest):
    """Re-bind a chat to a different collection. Accepts CreateChatRequest
    but only collection_id is read — title is ignored here."""
    ok = chat_store.update_collection(chat_id, req.collection_id)
    if not ok:
        raise HTTPException(404, "Chat not found")
    return {"status": "ok", "chat_id": chat_id, "collection_id": req.collection_id}


@router.patch("/chats/{chat_id}/title")
async def update_chat_title(chat_id: str, req: CreateChatRequest):
    """Manual rename. Reuses CreateChatRequest for its `title` field."""
    ok = chat_store.update_title(chat_id, req.title or "")
    if not ok:
        raise HTTPException(400, "Invalid title or chat not found")
    return {"status": "ok", "chat_id": chat_id, "title": req.title}


@router.post("/chats/{chat_id}/auto_title")
async def auto_title_chat(chat_id: str):
    """Regenerate a chat's title from current frame salience.

    Looks up the chat's active session, pulls FrameState, and runs the
    titler service. Always overwrites — user invokes this explicitly, so
    the should_auto_title gate doesn't apply. Useful when a chat's topic
    has drifted and the auto-titled label is stale.
    """
    from ..services.chat_titler import derive_title

    chat = chat_store.get_chat(chat_id)
    if not chat:
        raise HTTPException(404, "Chat not found")
    session_id = session_store.get_active_session(chat_id)
    frame = frame_manager.get_frame(session_id) if session_id else None
    if frame is None and session_id:
        frame = session_store.load_frame(session_id)
    tentative_registry = (
        frame_manager.get_tentative_registry(session_id) if session_id else {}
    )
    # First user message as fallback
    msgs = chat_store.get_messages(chat_id)
    first_user = next((m.content for m in msgs if m.role == "user"), None)
    new_title = derive_title(frame, deps.corpus, tentative_registry, first_user)
    chat_store.update_title(chat_id, new_title)
    return {"status": "ok", "chat_id": chat_id, "title": new_title}


@router.get("/chats")
async def list_chats():
    chats = chat_store.list_chats()
    return [c.model_dump(mode="json") for c in chats]


@router.get("/chats/{chat_id}")
async def get_chat(chat_id: str):
    chat = chat_store.get_chat(chat_id)
    if not chat:
        raise HTTPException(404, "Chat not found")
    messages = chat_store.get_messages(chat_id)
    # Find active session for this chat so frontend can restore graph
    active_session = session_store.get_active_session(chat_id)
    return {
        "chat": chat.model_dump(mode="json"),
        "messages": [m.model_dump(mode="json") for m in messages],
        "session_id": active_session,
    }


@router.patch("/chats/{chat_id}/status")
async def update_chat_status(chat_id: str, req: UpdateChatStatusRequest):
    chat_store.update_status(chat_id, req.status)
    return {"status": "ok"}


# --- Chat messaging ---

@router.post("/chats/{chat_id}/messages")
async def send_message(chat_id: str, req: SendMessageRequest):
    """Send a user message and stream the assistant response.

    Integrates: classification, anchor matching, frame state update,
    drift estimation, and background proposal extraction.
    """
    chat = chat_store.get_chat(chat_id)
    if not chat:
        raise HTTPException(404, "Chat not found")
    if chat.status != ChatStatus.ACTIVE:
        raise HTTPException(400, f"Chat is {chat.status.value}, not ACTIVE")

    history = chat_store.get_messages(chat_id)
    turn = len(history) + 1

    # Store user message
    user_msg = ChatMessage(role="user", content=req.content, turn=turn)
    chat_store.append_message(chat_id, user_msg)

    # Resolve OLI mode early (needed for base set seeding)
    oli_mode = OLIMode(req.oli_mode)

    # Resolve or create session
    session_id = session_store.get_active_session(chat_id)
    if not session_id:
        session_id = session_store.create_session(chat_id)

    # Restore frame state + tentative registry + edges if available.
    # Note: base_set_slabs() now includes REFERENCE type across all active
    # collections, so the model sees the full cold corpus (mined narratives
    # included) without needing per-chat collection binding to unlock it.
    saved_frame = session_store.load_frame(session_id)
    if saved_frame:
        reg, edges = session_store.load_registry(session_id)
        frame_manager.restore(session_id, saved_frame, reg or None, edges or None)
    else:
        frame_manager.get_or_create(chat_id, session_id, oli_mode=oli_mode)

    # Shared state between the generator and the post-stream callback.
    # The generator writes into this; the background task reads from it.
    # `_stream_done` is an asyncio.Event so the post-stream task can wait
    # signal-based instead of polling — cheaper and correctly handles
    # long inferences (OLI-ON + gemma3:12b + heavy retrieval can exceed
    # 5 minutes). The generator sets this in its finally block so it
    # fires on both success and error paths.
    _stream_result = {"metadata": {}, "assistant_text": "", "done": False}
    _stream_done = asyncio.Event()

    async def stream():
        try:
            async for chunk in process_turn(
                req.content, history, oli_mode,
                session_id=session_id,
                chat_id=chat_id,
                anchor_matcher=anchor_matcher,
                frame_manager=frame_manager,
                drift_monitor=drift_monitor,
                gauntlet_engine=gauntlet_engine,
                slab_matcher=slab_matcher,
                web_mode=req.web_mode,
                think_level=req.think_level,
            ):
                if chunk.get("done"):
                    _stream_result["metadata"] = chunk
                    _stream_result["assistant_text"] = chunk.get("full_response", "")
                else:
                    yield f"data: {json.dumps({'content': chunk.get('content', '')})}\n\n"
        except Exception as e:
            error_msg = f"[Inference error: {type(e).__name__}: {e}]"
            try:
                yield f"data: {json.dumps({'content': error_msg})}\n\n"
            except Exception:
                pass  # client may already be gone
            _stream_result["assistant_text"] = error_msg
            _stream_result["stream_error"] = str(e)

        try:
            metadata = _stream_result["metadata"]
            assistant_text = _stream_result["assistant_text"]

            # Store assistant message
            assistant_msg = ChatMessage(
                role="assistant",
                content=assistant_text,
                turn=turn + 1,
                classification=MessageClassification(**metadata.get("classification", {}))
                    if metadata.get("classification") else None,
                drift_estimate=DriftEstimate(**metadata.get("drift_estimate", {}))
                    if metadata.get("drift_estimate") else None,
            )
            chat_store.append_message(chat_id, assistant_msg)

            # Persist frame state
            frame = frame_manager.get_frame(session_id)
            if frame:
                save_session_state(session_id)

            # Build final metadata payload
            final: dict = {
                "done": True,
                "session_id": session_id,
                "classification": metadata.get("classification"),
                "drift_estimate": metadata.get("drift_estimate"),
            }
            if metadata.get("match_result"):
                final["match_result"] = metadata["match_result"]
            if metadata.get("frame_summary"):
                final["frame_summary"] = metadata["frame_summary"]
            if metadata.get("context_usage"):
                final["context_usage"] = metadata["context_usage"]

            try:
                yield f"data: {json.dumps(final)}\n\n"
            except Exception:
                pass  # client disconnected — extraction still runs
        finally:
            # ALWAYS mark done so the post-stream extraction task proceeds.
            # Previously this was only set on the happy path, so any exception
            # in the save/final-yield steps (or a client disconnect mid-OLI-ON
            # turn where latency exceeds the poll window) silently killed
            # draft extraction for the whole turn. Extraction now runs on any
            # turn that got as far as starting the generator — it will see
            # partial/empty metadata and gracefully no-op if there's nothing
            # to extract.
            _stream_result["done"] = True
            _stream_done.set()

    async def _post_stream_drafts():
        """Background task: runs AFTER the SSE stream completes.

        This avoids the generator-cancellation problem where code after
        the last yield gets killed when the client disconnects.
        """
        # Wait for stream completion via asyncio.Event (signal-based, not
        # polling). The generator sets _stream_done in its finally block,
        # so this fires on both success and error paths.
        #
        # 10-minute ceiling: OLI-ON + gemma3:12b + heavy retrieval can
        # legitimately take several minutes. The previous 180s polling
        # window was tripping on genuine completions, not runaway
        # inference. If 600s IS hit, something's actually wrong.
        try:
            await asyncio.wait_for(_stream_done.wait(), timeout=600.0)
        except asyncio.TimeoutError:
            print("[DRAFT] Stream didn't complete in 600s — skipping extraction", flush=True)
            return

        if _stream_result.get("stream_error"):
            print(f"[DRAFT] Stream had error ({_stream_result['stream_error']}) — skipping extraction", flush=True)
            return

        metadata = _stream_result["metadata"]
        actual_turn = metadata.get("turn", turn)
        recent = [
            {"role": m.role, "content": m.content, "turn": m.turn}
            for m in chat_store.get_message_window(chat_id, last_n=12)
        ]

        cls = metadata.get("classification") or {}
        new_drafts: list = []
        # Phase 2A: inherit the chat's bound collection so chat-mined drafts
        # target the right subcorpus instead of defaulting to "default".
        _chat = chat_store.get_chat(chat_id)
        _chat_cid = (_chat.collection_id if _chat else None) or "default"
        if cls.get("explicit"):
            print(f"[DRAFT] Explicit extraction at turn {actual_turn} (-> {_chat_cid})", flush=True)
            new_drafts = await draft_manager.extract_proposals(
                session_id, chat_id, recent, actual_turn,
                explicit=True, user_request=req.content,
                collection_id=_chat_cid,
            )
        elif draft_manager.should_sweep(actual_turn):
            print(f"[DRAFT] Sweep triggered at turn {actual_turn} (-> {_chat_cid})", flush=True)
            new_drafts = await draft_manager.extract_proposals(
                session_id, chat_id, recent, actual_turn,
                collection_id=_chat_cid,
            )
        else:
            print(f"[DRAFT] No sweep at turn {actual_turn}", flush=True)

        # Auto-trigger dreaming pass on newly mined drafts.
        # Runs as another background task so the user sees drafts in the
        # review panel immediately; by the time they open one, the
        # .enriched.json is likely already written.
        if new_drafts:
            async def _dream_new():
                try:
                    from ..services.dreaming import DreamingPass
                    dreamer = DreamingPass(deps.corpus, session_store, chat_store)
                    for packet in new_drafts:
                        try:
                            await dreamer.dream(session_id, packet.id)
                        except Exception as e:
                            print(f"[DREAM] Error dreaming {packet.id}: {e}", flush=True)
                except Exception as e:
                    print(f"[DREAM] Dreaming pass failed: {e}", flush=True)
            asyncio.create_task(_dream_new())

        # Phase 2B: auto-title once the frame has enough salience signal.
        # Runs every eligible turn but only overwrites if the title is still
        # the default placeholder — user-named chats are left alone. Kept
        # inline (not a background task) because it's cheap: just reads the
        # in-memory frame and writes one JSON.
        try:
            from ..services.chat_titler import derive_title, should_auto_title
            _chat_fresh = chat_store.get_chat(chat_id)
            _msgs = chat_store.get_messages(chat_id)
            if _chat_fresh and should_auto_title(_chat_fresh, len(_msgs)):
                _frame = frame_manager.get_frame(session_id)
                _reg = frame_manager.get_tentative_registry(session_id)
                _first_user = next((m.content for m in _msgs if m.role == "user"), None)
                _new_title = derive_title(_frame, deps.corpus, _reg, _first_user)
                if _new_title and _new_title != _chat_fresh.title:
                    chat_store.update_title(chat_id, _new_title)
                    print(f"[TITLE] Auto-titled chat {chat_id}: {_new_title!r}", flush=True)
        except Exception as e:
            print(f"[TITLE] Auto-title failed for {chat_id}: {e}", flush=True)

    # Fire background draft extraction as a free-standing task
    asyncio.create_task(_post_stream_drafts())

    return StreamingResponse(stream(), media_type="text/event-stream")
