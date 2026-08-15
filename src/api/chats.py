"""Chat CRUD and messaging endpoints — extracted from routes.py."""
from __future__ import annotations
import asyncio
import json
import logging
import os
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

# Per-turn mining. Chat turns are RECALL-only by default: the post-stream
# authoring cascade (live-mining reference matcher, proposal + relationship
# extraction, and the dreaming pass those trigger) does NOT run per turn.
# Even though it runs in a background task, it contends with the next turn's
# retrieval for the GPU. Mining is instead triggered explicitly
# (/extract-proposals) or at end of chat. Set PS_MINE_PER_TURN=1 to restore.
_MINE_PER_TURN = os.environ.get("PS_MINE_PER_TURN", "0") == "1"

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
    # Reference prompt content is selected later inside process_turn via the
    # bounded retrieval pipeline; restoring a frame does not preload every
    # REFERENCE slab into the model context.
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

        # Live mining — cross-reference matcher. Fires BEFORE the proposal
        # extractor on this turn so the new turn's content gets matched
        # against drafts that existed at turn-start. Drafts created on
        # this same turn (by extract_proposals below) won't accidentally
        # self-reference; future turns will pick them up. See
        # services/live_mining.update_reference_history for the matcher
        # contract. Failures must never block the post-stream pipeline.
        if _MINE_PER_TURN:
            try:
                from ..services.live_mining import update_reference_history
                await update_reference_history(session_id, actual_turn, req.content)
            except Exception as e:
                print(f"[LIVE-MINE] Reference matcher failed: {e}", flush=True)

        if _MINE_PER_TURN and cls.get("explicit"):
            print(f"[DRAFT] Explicit extraction at turn {actual_turn} (-> {_chat_cid})", flush=True)
            new_drafts = await draft_manager.extract_proposals(
                session_id, chat_id, recent, actual_turn,
                explicit=True, user_request=req.content,
                collection_id=_chat_cid,
            )
        elif _MINE_PER_TURN and draft_manager.should_sweep(actual_turn):
            print(f"[DRAFT] Sweep triggered at turn {actual_turn} (-> {_chat_cid})", flush=True)
            new_drafts = await draft_manager.extract_proposals(
                session_id, chat_id, recent, actual_turn,
                collection_id=_chat_cid,
            )
        else:
            print(f"[DRAFT] No per-turn mining at turn {actual_turn}", flush=True)

        # Phase 4 — second-pass relationship miner. Complementary to
        # extract_proposals: the primary miner is oriented toward concept
        # extraction (new anchors/slabs); this pass is narrower, asking
        # specifically "which EXISTING corpus nodes does this conversation
        # relate to each other?" Fires on every turn that got as far as
        # the post-stream phase (no turn-threshold gate), because existing-
        # to-existing edges are cheap to propose and directly flesh out
        # the graph's relational structure. Dedup against both corpus
        # edges and already-proposed edges is inside extract_relationships,
        # so re-firing per turn is safe — we just won't duplicate.
        if _MINE_PER_TURN:
            try:
                await draft_manager.extract_relationships(
                    session_id, chat_id, recent, actual_turn,
                )
            except Exception as e:
                print(f"[RELMINE] outer failure: {e}", flush=True)

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


@router.post("/chats/{chat_id}/end")
async def end_chat(chat_id: str):
    """Explicit end-of-chat sweep — the ONLY mining trigger.

    Recall turns don't mine (see ``_MINE_PER_TURN``). Ending a chat is where
    authoring happens: derive the *ghost stack* — concepts discussed in the
    conversation that aren't yet corpus nodes — check each for **uniqueness**
    (dedup vs the corpus, inside ``extract_proposals`` → ``_dedup_check``),
    keep the salient survivors, enrich them via the grounding/dreaming pass,
    and surface them for human review. Also records the session duration and
    marks the chat CLOSED.

    Returns the ghost stack so the caller can render the review panel without
    a second round-trip (the full stack + trajectory is also available via
    ``GET /sessions/{session_id}/end-review``).
    """
    chat = chat_store.get_chat(chat_id)
    if not chat:
        raise HTTPException(404, "Chat not found")

    session_id = session_store.get_active_session(chat_id)
    if not session_id:
        raise HTTPException(400, "No session for this chat — nothing to sweep")

    # Ensure the frame is in memory for the session-end record.
    if not frame_manager.get_frame(session_id):
        saved = session_store.load_frame(session_id)
        if saved:
            reg, edges = session_store.load_registry(session_id)
            frame_manager.restore(session_id, saved, reg or None, edges or None)

    messages = chat_store.get_messages(chat_id)
    if not messages:
        return {"chat_id": chat_id, "session_id": session_id, "ghost_stack": [], "count": 0,
                "note": "empty chat — nothing to sweep"}

    # 1. Derive + dedup the ghost stack from the WHOLE conversation.
    #    explicit=False → the conversation-ANALYSIS prompt (find load-bearing
    #    concepts across the transcript). NOT the explicit-concept prompt: that
    #    one expects a user_request naming a specific concept, and a marker
    #    string there leaks in as a bogus proposal ("end-of-chat ghost sweep").
    recent = [{"role": m.role, "content": m.content, "turn": m.turn} for m in messages]
    _chat_cid = chat.collection_id or "default"
    drafts = await draft_manager.extract_proposals(
        session_id, chat_id, recent, len(messages),
        explicit=False,
        collection_id=_chat_cid,
    )

    # 2. Enrich (grounding/dreaming) so the surfaced stack is audited.
    try:
        from ..services.dreaming import dream_all_pending
        await dream_all_pending(deps.corpus, session_store, chat_store, session_id)
    except Exception as e:
        logger.warning("end-chat %s: dreaming pass failed: %s", chat_id, e)

    # 3. Record session end (drift baseline) + mark the chat CLOSED.
    try:
        from ..services.drift_monitor import record_session_end
        _frame = frame_manager.get_frame(session_id)
        if _frame and _frame.last_updated_turn > 0:
            record_session_end(session_id, _frame.last_updated_turn)
    except Exception as e:
        logger.warning("end-chat %s: session-end record failed: %s", chat_id, e)
    try:
        chat_store.update_status(chat_id, ChatStatus.CLOSED)
    except Exception as e:
        logger.warning("end-chat %s: status update failed: %s", chat_id, e)

    ghost_stack = [d.model_dump(mode="json") for d in drafts]
    return {
        "chat_id": chat_id,
        "session_id": session_id,
        "ghost_stack": ghost_stack,
        "count": len(ghost_stack),
    }
