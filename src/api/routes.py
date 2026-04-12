"""API routes — chat, corpus, sessions, drafts, and system endpoints."""
from __future__ import annotations
import asyncio
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
from typing import Optional

from ..models.schemas import ChatMessage, MessageClassification, DriftEstimate
from ..models.enums import ChatStatus, OLIMode
from ..services.chat_store import ChatStore
from ..services.corpus import CorpusStore, CorpusRegistry
from ..services.pipeline import process_turn
from ..services.event_log import EventLog
from ..services.session_store import SessionStore
from ..services.anchor_matcher import AnchorMatcher
from ..services.frame_manager import FrameManager
from ..services.draft_manager import DraftManager
from ..services import ollama

router = APIRouter()

# --- Service instances ---
chat_store = ChatStore()
registry = CorpusRegistry()
corpus = CorpusStore()  # Will be replaced by registry.merged after load
event_log = EventLog()
session_store = SessionStore()
frame_manager = FrameManager(corpus)
anchor_matcher = AnchorMatcher(corpus)
draft_manager = DraftManager(corpus, session_store)

from ..services.drift_monitor import DriftMonitor
drift_monitor = DriftMonitor()

from ..services.gauntlet import GauntletEngine
gauntlet_engine = GauntletEngine(corpus)


def _rebind_corpus():
    """Rebind all services to the registry's merged corpus view.

    Called after collection activation changes so services see the
    updated anchor/slab/bundle set without a full restart.
    """
    global corpus
    corpus = registry.merged
    frame_manager.corpus = corpus
    anchor_matcher.corpus = corpus
    draft_manager.corpus = corpus
    draft_manager._registry = registry
    gauntlet_engine.corpus = corpus


# --- Request models ---

class CreateChatRequest(BaseModel):
    title: str = "New Chat"


class SendMessageRequest(BaseModel):
    content: str
    oli_mode: str = "OFF"
    web_mode: str = "off"  # "off" | "on" | "auto"
    think_level: str = "medium"  # "off" | "low" | "medium" | "high"


class UpdateChatStatusRequest(BaseModel):
    status: ChatStatus


class ReviewDraftRequest(BaseModel):
    action: str  # "discard" | "promote_tentative" | "promote_corpus"
    oli_mode: str = "OFF"


class AmendDraftRequest(BaseModel):
    canonical_phrase: Optional[str] = None
    canonical_text: Optional[str] = None
    justification: Optional[str] = None
    aliases: Optional[list[str]] = None


class CreateBundleRequest(BaseModel):
    node_ids: list[str]
    label: str = ""


class PromoteToAnchorRequest(BaseModel):
    node_id: str


class ExtractProposalsRequest(BaseModel):
    explicit: bool = True
    user_request: str = ""


def _save_session_state(sid: str):
    """Save frame + tentative registry + edges together."""
    frame = frame_manager._frames.get(sid)
    if frame:
        session_store.save_frame(sid, frame)
    registry = frame_manager._tentative_registry.get(sid, {})
    edges = frame_manager._tentative_edges.get(sid, [])
    session_store.save_registry(sid, registry, edges)


# --- Settings ---

class SearchSettingsRequest(BaseModel):
    api_key: str = ""
    provider: str = "duckduckgo"  # "duckduckgo" | "perplexity" | "brave" | "google"


@router.post("/settings/search")
async def update_search_settings(req: SearchSettingsRequest):
    from ..services import web_search
    key = req.api_key.strip() if req.api_key else None
    web_search.configure(api_key=key, provider=req.provider)
    return {"status": "ok", "provider": req.provider, "has_key": key is not None}


@router.get("/settings/search")
async def get_search_settings():
    from ..services import web_search
    return {
        "provider": web_search._api_provider,
        "has_key": web_search._api_key is not None,
    }


# --- Health ---

@router.get("/health")
async def health():
    ollama_health = await ollama.health_check()
    corpus_errors = corpus.validate()
    return {
        "status": "ok" if ollama_health["chat_model"] else "degraded",
        "ollama": ollama_health,
        "corpus_valid": len(corpus_errors) == 0,
        "corpus_errors": corpus_errors[:5],
    }


# --- Model management ---

@router.get("/models")
async def list_models():
    """List all locally available Ollama models + which one is active."""
    import httpx
    try:
        async with httpx.AsyncClient(base_url="http://localhost:11434") as client:
            resp = await client.get("/api/tags")
            resp.raise_for_status()
            models = resp.json().get("models", [])
            # Simplify: return name + size for each
            model_list = [
                {"name": m["name"], "size": m.get("size", 0)}
                for m in models
            ]
            return {"models": model_list, "active": ollama.CHAT_MODEL}
    except Exception as e:
        return {"models": [], "active": ollama.CHAT_MODEL, "error": str(e)}


class SwitchModelRequest(BaseModel):
    model: str


@router.post("/models/switch")
async def switch_model(req: SwitchModelRequest):
    """Hot-swap the active chat model.

    Unloads the current model from VRAM, loads the new one with
    keep_alive=-1 (pinned), and updates the global CHAT_MODEL.
    """
    import httpx
    old_model = ollama.CHAT_MODEL
    new_model = req.model

    try:
        async with httpx.AsyncClient(base_url="http://localhost:11434",
                                      timeout=httpx.Timeout(120.0)) as client:
            # Unload old model
            if old_model != new_model:
                await client.post("/api/generate", json={
                    "model": old_model, "prompt": "", "keep_alive": 0,
                })

            # Load new model with max GPU + pinned
            resp = await client.post("/api/generate", json={
                "model": new_model,
                "prompt": "",
                "stream": False,
                "keep_alive": -1,
                "options": {"num_gpu": 99, "num_ctx": ollama._NUM_CTX},
            })
            resp.raise_for_status()

            # Update the global model reference + auto-detect capabilities
            ollama.CHAT_MODEL = new_model
            from ..services import model_profiles
            profile = await model_profiles.set_active(new_model)

            # Inject model-switch boundary marker into active chat history
            # so the new model knows prior identity-adjacent nodes are historical
            if old_model != new_model:
                try:
                    # Find the most recent active chat
                    all_chats = chat_store.list_chats()
                    active = [c for c in all_chats if c.get("status") == "active"]
                    if active:
                        active_chat_id = active[0]["id"]
                        from ..models.schemas import ChatMessage
                        boundary = ChatMessage(
                            role="system",
                            content=(
                                f"[MODEL SWITCH] {old_model} → {new_model}. "
                                f"Prior messages were generated by {old_model}. "
                                f"Identity-related frame nodes from before this point are historical context, "
                                f"not your identity. You are {new_model} ({profile.family} family)."
                            ),
                        )
                        chat_store.append_message(active_chat_id, boundary)
                except Exception as e:
                    logger.warning("Failed to inject model-switch boundary: %s", e)

            # Build profile info for frontend
            profile_info = {
                "family": profile.family if profile else "",
                "params": profile.parameter_size if profile else "",
                "think": profile.supports_think if profile else False,
                "tools": profile.supports_tools if profile else False,
                "vision": profile.supports_vision if profile else False,
            }

            # Check actual VRAM usage
            ps_resp = await client.get("/api/ps")
            ps_data = ps_resp.json()
            for m in ps_data.get("models", []):
                if m["name"] == new_model:
                    return {
                        "ok": True,
                        "model": new_model,
                        "profile": profile_info,
                        "size_vram": m.get("size_vram", 0),
                        "size": m.get("size", 0),
                        "gpu_pct": round(
                            m.get("size_vram", 0) / max(1, m.get("size", 1)) * 100
                        ),
                    }
            return {"ok": True, "model": new_model, "profile": profile_info, "size_vram": 0, "size": 0, "gpu_pct": 0}
    except Exception as e:
        # Rollback on failure
        ollama.CHAT_MODEL = old_model
        return {"ok": False, "error": str(e), "model": old_model}


# --- File upload ---

# Supported text-extractable file types
_TEXT_EXTENSIONS = {
    ".txt", ".md", ".csv", ".tsv", ".json", ".yaml", ".yml",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css",
    ".c", ".cpp", ".h", ".hpp", ".java", ".go", ".rs", ".rb",
    ".sh", ".bash", ".zsh", ".ps1", ".bat",
    ".toml", ".ini", ".cfg", ".conf", ".env", ".xml",
    ".sql", ".r", ".m", ".swift", ".kt", ".scala", ".lua",
    ".log", ".tex",
}
_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Extract text content from an uploaded file.

    Supports plaintext/code files directly and PDFs via pdfplumber.
    Returns extracted text for the frontend to prepend to the user message.
    """
    import os
    ext = os.path.splitext(file.filename or "")[1].lower()

    # Read file bytes (with size guard)
    data = await file.read()
    if len(data) > _MAX_FILE_SIZE:
        raise HTTPException(413, f"File too large ({len(data)} bytes). Max is {_MAX_FILE_SIZE}.")

    extracted = ""

    if ext == ".pdf":
        try:
            import pdfplumber
            import io
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                pages = []
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        pages.append(f"--- Page {i+1} ---\n{text}")
                extracted = "\n\n".join(pages)
                if not extracted.strip():
                    raise HTTPException(422, "PDF appears to contain no extractable text (scanned/image PDF).")
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(422, f"Failed to extract PDF text: {exc}")

    elif ext in (".docx", ".doc"):
        try:
            import docx
            import io
            doc = docx.Document(io.BytesIO(data))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            # Also extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if cells:
                        paragraphs.append(" | ".join(cells))
            extracted = "\n\n".join(paragraphs)
            if not extracted.strip():
                raise HTTPException(422, "DOCX appears to contain no extractable text.")
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(422, f"Failed to extract DOCX text: {exc}")

    elif ext in _TEXT_EXTENSIONS or ext == "":
        # Try decoding as text
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                extracted = data.decode(encoding)
                break
            except (UnicodeDecodeError, ValueError):
                continue
        else:
            raise HTTPException(422, f"Could not decode file as text.")

    else:
        raise HTTPException(
            415,
            f"Unsupported file type '{ext}'. Supported: PDF, DOCX, and text/code files "
            f"({', '.join(sorted(list(_TEXT_EXTENSIONS)[:12]))}...)"
        )

    # Truncate very long files to avoid blowing up context
    char_limit = 80_000  # ~20k tokens
    truncated = False
    if len(extracted) > char_limit:
        extracted = extracted[:char_limit]
        truncated = True

    return {
        "filename": file.filename,
        "extension": ext,
        "chars": len(extracted),
        "truncated": truncated,
        "content": extracted,
    }


# --- Chat CRUD ---

@router.post("/chats")
async def create_chat(req: CreateChatRequest):
    chat = chat_store.create_chat(req.title)
    return chat.model_dump(mode="json")


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

    # Restore frame state + tentative registry + edges if available
    saved_frame = session_store.load_frame(session_id)
    if saved_frame:
        reg, edges = session_store.load_registry(session_id)
        frame_manager.restore(session_id, saved_frame, reg or None, edges or None)
    else:
        frame_manager.get_or_create(chat_id, session_id, oli_mode=oli_mode)

    # Shared state between the generator and the post-stream callback.
    # The generator writes into this; the background task reads from it.
    _stream_result = {"metadata": {}, "assistant_text": "", "done": False}

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
            yield f"data: {json.dumps({'content': error_msg})}\n\n"
            _stream_result["assistant_text"] = error_msg

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
        frame = frame_manager._frames.get(session_id)
        if frame:
            _save_session_state(session_id)

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

        # Mark done so background task knows stream completed
        _stream_result["done"] = True

        yield f"data: {json.dumps(final)}\n\n"

    async def _post_stream_drafts():
        """Background task: runs AFTER the SSE stream completes.

        This avoids the generator-cancellation problem where code after
        the last yield gets killed when the client disconnects.
        """
        # Wait for stream to finish (poll briefly)
        for _ in range(600):  # up to 60s
            if _stream_result["done"]:
                break
            await asyncio.sleep(0.1)

        if not _stream_result["done"]:
            print("[DRAFT] Stream didn't complete — skipping extraction")
            return

        metadata = _stream_result["metadata"]
        actual_turn = metadata.get("turn", turn)
        recent = [
            {"role": m.role, "content": m.content, "turn": m.turn}
            for m in chat_store.get_message_window(chat_id, last_n=12)
        ]

        cls = metadata.get("classification") or {}
        new_drafts: list = []
        if cls.get("explicit"):
            print(f"[DRAFT] Explicit extraction at turn {actual_turn}", flush=True)
            new_drafts = await draft_manager.extract_proposals(
                session_id, chat_id, recent, actual_turn,
                explicit=True, user_request=req.content,
            )
        elif draft_manager.should_sweep(actual_turn):
            print(f"[DRAFT] Sweep triggered at turn {actual_turn}", flush=True)
            new_drafts = await draft_manager.extract_proposals(
                session_id, chat_id, recent, actual_turn,
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
                    dreamer = DreamingPass(corpus, session_store, chat_store)
                    for packet in new_drafts:
                        try:
                            await dreamer.dream(session_id, packet.id)
                        except Exception as e:
                            print(f"[DREAM] Error dreaming {packet.id}: {e}", flush=True)
                except Exception as e:
                    print(f"[DREAM] Dreaming pass failed: {e}", flush=True)
            asyncio.create_task(_dream_new())

    # Fire background draft extraction as a free-standing task
    asyncio.create_task(_post_stream_drafts())

    return StreamingResponse(stream(), media_type="text/event-stream")


# --- Sessions & Frame ---

class NodeActionRequest(BaseModel):
    node_id: str


class NodeSalienceRequest(BaseModel):
    node_id: str
    delta: float = 0.0  # positive = heat, negative = cool


@router.post("/sessions/{session_id}/frame/remove-node")
@router.post("/sessions/{session_id}/frame/adjust-salience")
async def adjust_salience(session_id: str, req: NodeSalienceRequest):
    """Heat or cool a node — modifies backend salience that the model sees."""
    frame = frame_manager._frames.get(session_id)
    if not frame:
        raise HTTPException(404, "No active frame for this session")

    node_id = req.node_id
    if node_id not in frame.active_nodes:
        return {"error": "Node not in active frame"}

    old = frame.salience_now.get(node_id, 0.0)
    new_sal = max(0.0, min(1.0, old + req.delta))
    frame.salience_now[node_id] = new_sal
    frame.salience_smoothed[node_id] = new_sal  # immediate effect

    _save_session_state(session_id)
    action = "heated" if req.delta > 0 else "cooled"
    event_log.log_frame_event(
        session_id=session_id, event=f"user_{action}",
        node_id=node_id, old=round(old, 3), new=round(new_sal, 3),
    )
    return {"status": action, "node_id": node_id, "salience": round(new_sal, 3)}


@router.post("/sessions/{session_id}/frame/remove-node")
async def remove_node_from_session(session_id: str, req: NodeActionRequest):
    """Remove a node from the active frame.

    Evicts from frame but does NOT permanently delete from tentative registry.
    Marks as 'dismissed' so the concept detector can bring it back if
    the user organically discusses it again.
    """
    frame = frame_manager._frames.get(session_id)
    if not frame:
        raise HTTPException(404, "No active frame for this session")

    node_id = req.node_id
    if node_id in frame.active_nodes:
        frame.active_nodes.remove(node_id)
    frame.salience_now.pop(node_id, None)
    frame.salience_smoothed.pop(node_id, None)
    frame.structural_weight.pop(node_id, None)
    frame.activation_sources.pop(node_id, None)
    frame.active_anchors.pop(node_id, None)
    frame.active_bundles.pop(node_id, None)
    frame.active_slabs.pop(node_id, None)
    frame.active_concepts.pop(node_id, None)

    # Mark as dismissed in tentative registry — NOT deleted
    # Can be re-added if concept detector finds it organically again
    registry = frame_manager._tentative_registry.get(session_id, {})
    for k, v in registry.items():
        if v.get("id") == node_id:
            v["dismissed"] = True
            break

    _save_session_state(session_id)
    event_log.log_frame_event(session_id=session_id, event="node_removed", node_id=node_id)
    return {"status": "removed", "node_id": node_id}


@router.post("/sessions/{session_id}/frame/reject-node")
async def reject_node(session_id: str, req: NodeActionRequest):
    """Mark a tentative node as rejected — stays visible but greyed out, will not be proposed for commit."""
    node_id = req.node_id

    # Mark in tentative registry
    registry = frame_manager._tentative_registry.get(session_id, {})
    for k, v in registry.items():
        if v.get("id") == node_id:
            v["rejected"] = True
            break

    event_log.log_frame_event(session_id=session_id, event="node_rejected", node_id=node_id)
    return {"status": "rejected", "node_id": node_id}


@router.post("/sessions/{session_id}/frame/create-bundle")
async def create_bundle_from_nodes(session_id: str, req: CreateBundleRequest):
    """User-driven: group selected nodes into a tentative bundle."""
    import uuid
    from ..models.schemas import ActivationSource

    frame = frame_manager._frames.get(session_id)
    if not frame:
        raise HTTPException(404, "No active frame")
    if len(req.node_ids) < 2:
        return {"error": "Bundle requires at least 2 nodes"}

    registry = frame_manager._tentative_registry.get(session_id, {})

    # Build label from children
    child_labels = []
    for nid in req.node_ids:
        for name, info in registry.items():
            if info.get("id") == nid:
                child_labels.append(name)
                break
        else:
            if nid in corpus.anchors:
                child_labels.append(corpus.anchors[nid].canonical_phrase)
            else:
                child_labels.append(nid)

    label = req.label or f"Bundle: {', '.join(child_labels[:3])}"
    bundle_id = f"tentative_bundle_{uuid.uuid4().hex[:8]}"

    # Create registry entry
    registry[label] = {
        "id": bundle_id,
        "description": f"User-grouped: {', '.join(child_labels)}",
        "turns_seen": 1,
        "promoted": None,
        "parent_id": None,
        "children": list(req.node_ids),
    }

    # Add to frame
    frame.active_nodes.append(bundle_id)
    frame.active_bundles[bundle_id] = 0.7
    frame.salience_now[bundle_id] = 0.7
    frame.salience_smoothed[bundle_id] = 0.7
    frame.structural_weight[bundle_id] = float(len(req.node_ids))
    frame.activation_sources[bundle_id] = [
        ActivationSource(source_type="user_bundling", source_ref=",".join(req.node_ids))
    ]

    # Create PARENT_OF edges from bundle to children
    if session_id not in frame_manager._tentative_edges:
        frame_manager._tentative_edges[session_id] = []
    for child_id in req.node_ids:
        frame_manager._tentative_edges[session_id].append({
            "from": bundle_id, "to": child_id, "type": "PARENT_OF", "strength": 0.8,
        })
        # Update child registry to point to parent
        for name, info in registry.items():
            if info.get("id") == child_id:
                info["parent_id"] = bundle_id
                break
        # Absorb children — remove from active frame (they live inside the bundle now)
        if child_id in frame.active_nodes:
            frame.active_nodes.remove(child_id)
        frame.salience_now.pop(child_id, None)
        frame.salience_smoothed.pop(child_id, None)
        frame.structural_weight.pop(child_id, None)
        frame.activation_sources.pop(child_id, None)
        frame.active_concepts.pop(child_id, None)
        frame.active_anchors.pop(child_id, None)
        frame.active_bundles.pop(child_id, None)

    # Record in corpus hit trajectory so it appears in session review
    turn = frame.last_updated_turn or 1
    frame.corpus_hits[bundle_id] = frame.corpus_hits.get(bundle_id, 0) + 1
    frame.corpus_last_hit[bundle_id] = turn
    frame.corpus_hit_log.append([turn, [bundle_id] + list(req.node_ids)])

    _save_session_state(session_id)
    event_log.log_frame_event(
        session_id=session_id, event="bundle_created_by_user",
        bundle_id=bundle_id, children=req.node_ids,
    )
    return {"status": "created", "bundle_id": bundle_id, "label": label}


@router.post("/sessions/{session_id}/frame/promote-to-anchor")
async def promote_to_anchor(session_id: str, req: PromoteToAnchorRequest):
    """Promote a tentative bundle to a tentative anchor (invocation handle)."""
    frame = frame_manager._frames.get(session_id)
    if not frame:
        raise HTTPException(404, "No active frame")

    registry = frame_manager._tentative_registry.get(session_id, {})

    # Find the node in registry
    target_name = None
    target_info = None
    for name, info in registry.items():
        if info.get("id") == req.node_id:
            target_name = name
            target_info = info
            break

    if not target_info:
        return {"error": "Node not found in tentative registry"}

    # Mark as promoted to anchor
    target_info["promoted"] = "anchor"

    # The node keeps its ID but the graph API will now report it as type=anchor
    # The anchor.invokes will point to its children (the bundle contents)
    children = target_info.get("children", [])

    event_log.log_frame_event(
        session_id=session_id, event="bundle_promoted_to_anchor",
        node_id=req.node_id, label=target_name, children=children,
    )
    _save_session_state(session_id)
    return {"status": "promoted", "node_id": req.node_id, "type": "anchor", "invokes": children}


@router.get("/sessions/{session_id}/frame")
async def get_frame(session_id: str):
    # Prefer the in-memory frame — it holds unsaved hit log entries and
    # salience updates from the current turn. Fall back to disk only when
    # the session hasn't been restored yet (e.g. after a server restart).
    frame = frame_manager._frames.get(session_id)
    if frame is None:
        frame = session_store.load_frame(session_id)
        if frame:
            reg, edges = session_store.load_registry(session_id)
            frame_manager.restore(session_id, frame, reg or None, edges or None)
    if not frame:
        raise HTTPException(404, "No frame state for this session")
    return frame.model_dump(mode="json")


# --- Drafts ---

@router.get("/sessions/{session_id}/drafts")
async def list_drafts(session_id: str):
    drafts = draft_manager.list_drafts(session_id)
    return [d.model_dump(mode="json") for d in drafts]


@router.get("/sessions/{session_id}/drafts/{draft_id}")
async def get_draft(session_id: str, draft_id: str):
    packet = session_store.load_draft_packet(session_id, draft_id)
    if not packet:
        raise HTTPException(404, "Draft not found")
    # Also load raw proposal for context
    raw_path = session_store._drafts_dir(session_id) / f"{draft_id}_raw.json"
    raw = session_store._read_json(raw_path) or {}
    return {
        "draft": packet.model_dump(mode="json"),
        "raw_proposal": raw,
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
    raw_path = session_store._drafts_dir(session_id) / f"{draft_id}_raw.json"
    raw = session_store._read_json(raw_path) or {}

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
        if dep_id in corpus.anchors:
            closure.append({"id": dep_id, "type": "anchor", "label": corpus.anchors[dep_id].canonical_phrase})
        elif dep_id in corpus.bundles:
            closure.append({"id": dep_id, "type": "bundle", "label": "; ".join(corpus.bundles[dep_id].payload.intent[:2])})
        elif dep_id in corpus.slabs:
            closure.append({"id": dep_id, "type": "slab", "label": corpus.slabs[dep_id].title})

    # 3. Cold committed neighbourhood — what's already near
    neighbourhood = []
    all_ids = corpus.all_ids()
    related_ids = set()
    for edge in corpus.edges.values():
        for dep in packet.dependency_closure + [draft_id]:
            if edge.from_node == dep: related_ids.add(edge.to_node)
            elif edge.to_node == dep: related_ids.add(edge.from_node)
    for nid in related_ids:
        if nid in corpus.anchors:
            neighbourhood.append({"id": nid, "type": "anchor", "label": corpus.anchors[nid].canonical_phrase})
        elif nid in corpus.bundles:
            neighbourhood.append({"id": nid, "type": "bundle", "label": "; ".join(corpus.bundles[nid].payload.intent[:2])})
        elif nid in corpus.slabs:
            neighbourhood.append({"id": nid, "type": "slab", "label": corpus.slabs[nid].title})

    # 4. Conflict / cascade preview — what might break
    cascade_risk = []
    for dep_id in packet.dependency_closure + [draft_id]:
        rev_deps = corpus.get_reverse_deps(dep_id)
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
        frame = frame_manager._frames.get(session_id)
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
            corpus.anchors[corpus_obj.id] = corpus_obj
        elif obj_type == "slab":
            corpus.slabs[corpus_obj.id] = corpus_obj
        elif obj_type == "bundle":
            corpus.bundles[corpus_obj.id] = corpus_obj
        validation_errors = corpus.validate()
        # Rollback
        if obj_type == "anchor": corpus.anchors.pop(corpus_obj.id, None)
        elif obj_type == "slab": corpus.slabs.pop(corpus_obj.id, None)
        elif obj_type == "bundle": corpus.bundles.pop(corpus_obj.id, None)

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
    raw_path = session_store._drafts_dir(session_id) / f"{draft_id}_raw.json"
    raw = session_store._read_json(raw_path) or {}

    if req.canonical_phrase is not None:
        raw["canonical_phrase"] = req.canonical_phrase
    if req.canonical_text is not None:
        raw["canonical_text"] = req.canonical_text
    if req.justification is not None:
        raw["justification"] = req.justification
        packet.justification = req.justification
    if req.aliases is not None:
        raw["aliases"] = req.aliases

    session_store._write_json(raw_path, raw)

    # Reset to unauthorized — must be re-reviewed
    from ..models.enums import DraftStatus
    packet.status = DraftStatus.DRAFT_UNAUTHORIZED
    session_store.save_draft_packet(session_id, packet)

    return {
        "status": "AMENDED",
        "draft_id": draft_id,
        "packet_status": packet.status.value,
        "raw": raw,
    }


@router.post("/sessions/{session_id}/drafts/{draft_id}/review")
async def review_draft(session_id: str, draft_id: str, req: ReviewDraftRequest):
    # Compute current drift severity for the drift gate
    drift_severity = "low"
    window = drift_monitor.get_window(session_id)
    frame = frame_manager._frames.get(session_id)
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


class VerifyClaimsRequest(BaseModel):
    context: str = ""
    method: str = "ollama"  # "manual" | "ollama" | "external"


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
    dreamer = DreamingPass(corpus, session_store, chat_store)
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
        corpus, session_store, chat_store, session_id,
    )
    return {"session_id": session_id, "results": results}


@router.post("/sessions/{session_id}/extract-proposals")
async def extract_proposals(session_id: str, req: ExtractProposalsRequest):
    """Manual proposal extraction trigger."""
    meta_path = session_store._session_dir(session_id) / "meta.json"
    meta = session_store._read_json(meta_path)
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
                dreamer = DreamingPass(corpus, session_store, chat_store)
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
    frame = frame_manager._frames.get(session_id)
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
        corpus, session_store, chat_store, session_id,
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
    drafts = draft_manager.list_drafts(session_id)
    for packet in drafts:
        raw_path = session_store._drafts_dir(session_id) / f"{packet.id}_raw.json"
        raw = session_store._read_json(raw_path) or {}
        enriched_path = session_store._drafts_dir(session_id) / f"{packet.id}.enriched.json"
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


# --- Corpus ---

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
    _rebind_corpus()
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

    _rebind_corpus()
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

    _rebind_corpus()
    return {
        "slab_id": slab_id,
        "title": req.slab_title,
        "source_anchors": len(anchor_phrases),
        "source_bundles": len(bundle_labels),
        "target_collection": req.target_collection,
        "message": f"Promoted '{collection_id}' as slab '{slab_id}' in '{req.target_collection}'",
    }


@router.get("/corpus/status")
async def corpus_status():
    errors = corpus.validate()
    return {
        "anchors": len(corpus.anchors),
        "slabs": len(corpus.slabs),
        "bundles": len(corpus.bundles),
        "edges": len(corpus.edges),
        "valid": len(errors) == 0,
        "errors": errors,
        "collections": registry.list_collections(),
    }


@router.get("/corpus/full")
async def corpus_full(collection: Optional[str] = None):
    """Return corpus for the Cold Corpus browser tab.

    Args:
        collection: Optional collection ID to filter to. If omitted,
                    returns the full merged corpus.

    Computes semantic axis positions (§11.1) so the corpus graph
    uses the same structured layout as the session graph:
      X = Creative ↔ Rigorous (embedding projection)
      Y = structural weight proxy (type-based: slab=0.4, bundle=0.2, anchor=0.1)
      Z = meta depth (slab=0, bundle=1, anchor=2)
    """
    from ..services.embeddings import compute_x_positions

    # Select corpus source: specific collection or full merged view
    if collection and collection != "__all__":
        source = registry.get_store(collection)
        if not source:
            raise HTTPException(404, f"Collection '{collection}' not found")
    else:
        source = corpus

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
    z_map = {"slab": 0, "bundle": 1, "anchor": 2}
    y_map = {"slab": 0.4, "bundle": 0.2, "anchor": 0.1}  # structural weight proxy

    try:
        x_positions = await compute_x_positions(texts_to_embed)
    except Exception:
        x_positions = [0.5] * len(texts_to_embed)

    # Inject semantic positions into each node dict
    all_nodes = anchors_out + bundles_out + slabs_out
    for i, node in enumerate(all_nodes):
        node["sem_x"] = x_positions[i]
        node["sem_y"] = y_map.get(node_types[i], 0.1)
        node["sem_z"] = z_map.get(node_types[i], 2)

    edges_out = [e.model_dump(by_alias=True) for e in source.edges.values()]

    # Auto-generate SEQUENCE edges for collections with no explicit edges.
    # Uses the alphabetical ID order (which preserves mining extraction order)
    # to create a narrative flow chain: node₁ → node₂ → node₃ → ...
    layout_hint = "cloud"  # default force-directed
    if collection and collection != "__all__" and len(edges_out) == 0 and len(all_nodes) > 1:
        layout_hint = "flow"  # narrative/sequential layout
        # Sort anchors by ID (mining order), then bundles, then slabs
        ordered_ids = (
            [a["id"] for a in anchors_out]
            + [b["id"] for b in bundles_out]
            + [s["id"] for s in slabs_out]
        )
        for i in range(len(ordered_ids) - 1):
            edges_out.append({
                "id": f"_seq_{i}",
                "type": "SEQUENCE",
                "from": ordered_ids[i],
                "to": ordered_ids[i + 1],
                "weight": 0.4,
                "confidence": 0.4,
            })

    return {
        "collection": collection or "__all__",
        "layout_hint": layout_hint,
        "anchors": anchors_out,
        "bundles": bundles_out,
        "slabs": slabs_out,
        "edges": edges_out,
        "gates": [g.model_dump() for g in source.gates.values()],
    }


# --- Corpus management (dashboard) ---

class UpdateLifecycleRequest(BaseModel):
    lifecycle_status: str  # "ACTIVE" | "DORMANT" | "DEPRECATED"


class UpdateNodeRequest(BaseModel):
    canonical_phrase: Optional[str] = None
    canonical_text: Optional[str] = None
    notes: Optional[str] = None
    title: Optional[str] = None
    aliases: Optional[list[str]] = None


def _find_corpus_node(node_id: str):
    """Locate a node across all corpus object types. Returns (obj, type_name)."""
    if node_id in corpus.anchors:
        return corpus.anchors[node_id], "anchor"
    if node_id in corpus.slabs:
        return corpus.slabs[node_id], "slab"
    if node_id in corpus.bundles:
        return corpus.bundles[node_id], "bundle"
    if node_id in corpus.gates:
        return corpus.gates[node_id], "gate"
    return None, None


@router.get("/corpus/nodes/{node_id}/deps")
async def get_node_deps(node_id: str):
    """Return everything that depends on or references this node."""
    node, ntype = _find_corpus_node(node_id)
    if not node:
        raise HTTPException(404, f"Node '{node_id}' not found in corpus")

    dependents = corpus.get_reverse_deps(node_id)
    # Also find edges that reference this node
    edge_refs = [
        {"edge_id": e.id, "from": e.from_node, "to": e.to_node, "type": e.type.value}
        for e in corpus.edges.values()
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
    node, ntype = _find_corpus_node(node_id)
    if not node:
        raise HTTPException(404, f"Node '{node_id}' not found in corpus")
    try:
        new_status = SlabLifecycleStatus(req.lifecycle_status)
    except ValueError:
        raise HTTPException(400, f"Invalid status '{req.lifecycle_status}'. Must be ACTIVE, DORMANT, or DEPRECATED.")

    node.lifecycle_status = new_status
    corpus.save()
    return {"node_id": node_id, "type": ntype, "lifecycle_status": new_status.value}


@router.patch("/corpus/nodes/{node_id}")
async def update_node_fields(node_id: str, req: UpdateNodeRequest):
    """Edit fields on a corpus node."""
    node, ntype = _find_corpus_node(node_id)
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

    corpus.save()
    return {"node_id": node_id, "type": ntype, "updated_fields": updated}


@router.delete("/corpus/nodes/{node_id}")
async def delete_node(node_id: str):
    """Hard-delete a node from the corpus. Checks dependencies first."""
    node, ntype = _find_corpus_node(node_id)
    if not node:
        raise HTTPException(404, f"Node '{node_id}' not found in corpus")

    deps = corpus.get_reverse_deps(node_id)
    if deps:
        raise HTTPException(
            409,
            f"Cannot delete '{node_id}': {len(deps)} node(s) depend on it: {deps[:5]}. "
            "Deprecate it instead, or remove the dependencies first."
        )

    # Remove from the appropriate dict
    if ntype == "anchor":
        del corpus.anchors[node_id]
    elif ntype == "slab":
        del corpus.slabs[node_id]
    elif ntype == "bundle":
        del corpus.bundles[node_id]
    elif ntype == "gate":
        del corpus.gates[node_id]

    # Remove edges that reference this node
    dead_edges = [eid for eid, e in corpus.edges.items()
                  if e.from_node == node_id or e.to_node == node_id]
    for eid in dead_edges:
        del corpus.edges[eid]

    corpus.save()
    return {"deleted": node_id, "type": ntype, "edges_removed": len(dead_edges)}


# --- Graph data for 3D renderer ---

@router.get("/sessions/{session_id}/graph")
async def get_graph_data(session_id: str):
    """Return full graph state for the 3D renderer.

    Combines corpus objects (cold), active frame nodes, tentative nodes,
    edges, and visual encoding fields from §11.
    """
    from ..services.embeddings import compute_x_position, compute_x_positions

    # Prefer the in-memory frame — it is the live source of truth during an
    # active session. Only fall back to disk if nothing is loaded (e.g. the
    # user just re-opened a chat and the session hasn't been restored yet).
    # Reading from disk here caused a split-brain bug: active_ids came from a
    # stale snapshot while the tentative registry came from memory, which made
    # concepts from the previous turn look like they had "disappeared" whenever
    # the UI refreshed the graph before the next save commit.
    frame = frame_manager._frames.get(session_id)
    if frame is None:
        frame = session_store.load_frame(session_id)
        if frame:
            reg, edges = session_store.load_registry(session_id)
            frame_manager.restore(session_id, frame, reg or None, edges or None)
    active_ids = set(frame.active_nodes) if frame else set()

    nodes = []
    texts_to_embed = []
    node_index = []

    # Collect all visible nodes: active frame + 1-hop edge neighbors
    visible_ids: set[str] = set(active_ids)

    # Add cold corpus neighbors within 1 hop via explicit edges only.
    # Previous 2-hop traversal via invokes/supports/depends_on was pulling
    # in the entire corpus — too aggressive for a focused session view.
    # The cold corpus tab shows everything; the session graph shows the thread.
    hop1 = set()
    for node_id in list(active_ids):
        for edge in corpus.edges.values():
            if edge.from_node == node_id: hop1.add(edge.to_node)
            elif edge.to_node == node_id: hop1.add(edge.from_node)
    # Cap expansion: at most 12 extra nodes from hop neighbors
    # (prevents dense collections from overwhelming the session view)
    if len(hop1 - active_ids) > 12:
        # Prioritize by edge weight: keep the strongest connections
        scored = []
        for nid in hop1 - active_ids:
            best_w = max(
                (e.weight for e in corpus.edges.values()
                 if (e.from_node == nid or e.to_node == nid)
                 and (e.from_node in active_ids or e.to_node in active_ids)),
                default=0
            )
            scored.append((best_w, nid))
        scored.sort(reverse=True)
        hop1 = active_ids | {nid for _, nid in scored[:12]}
    visible_ids.update(hop1)

    # Build node list with all visual encoding fields
    for anchor in corpus.anchors.values():
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

    for slab in corpus.slabs.values():
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

    for bundle in corpus.bundles.values():
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
        tentative_registry = frame_manager._tentative_registry.get(session_id, {})
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

    # Compute X positions via nomic-embed-text
    if texts_to_embed:
        x_positions = await compute_x_positions(texts_to_embed)
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
    for edge in corpus.edges.values():
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

    # Add tentative edges (session-scoped, not in corpus)
    tent_edges = frame_manager._tentative_edges.get(session_id, [])
    for i, te in enumerate(tent_edges):
        edges.append({
            "id": f"tent_{i}_{te['from'][:8]}_{te['to'][:8]}",
            "type": te.get("type", "PARENT_OF"),
            "from": te["from"],
            "to": te["to"],
            "confidence": te.get("strength", 0.7),
            "tension": None,
        })

    return {
        "nodes": nodes,
        "edges": edges,
        "frame": {
            "active_nodes": list(active_ids),
            "mismatch_score": frame.mismatch_score if frame else 0,
            "current_turn": frame.last_updated_turn if frame else 0,
            "conflicts": [c.model_dump() for c in frame.conflicts] if frame else [],
        },
    }


# --- Events ---

@router.get("/events/{log_name}")
async def get_events(log_name: str, last_n: int = 50):
    valid_logs = [
        "gate_events.jsonl", "push_events.jsonl", "degradation_flags.jsonl",
        "verification_log.jsonl", "drift_events.jsonl",
        "match_events.jsonl", "frame_events.jsonl", "proposal_events.jsonl",
    ]
    if log_name not in valid_logs:
        raise HTTPException(400, f"Invalid log name. Valid: {valid_logs}")
    return event_log.read_recent(log_name, last_n)


# --- Conversation Mining ---

from ..services.convo_miner import ConversationMiner
_miner = ConversationMiner(corpus)


class MineRequest(BaseModel):
    text: str                          # Raw conversation export content
    source_label: str = "import"       # Provenance label
    min_confidence: float = 0.4        # Minimum proposal confidence
    chunk_size: int = 4                # Exchange pairs per chunk
    target_collection: str = "default" # Which collection to mine into


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
    target_store = registry.get_store(req.target_collection) or corpus
    miner = ConversationMiner(target_store)

    result = await miner.mine(
        raw_text=req.text,
        source_label=req.source_label,
        min_confidence=req.min_confidence,
        chunk_size=req.chunk_size,
    )
    result["target_collection"] = req.target_collection
    return result


class PushMinedRequest(BaseModel):
    proposals: list[dict]           # Raw mined proposal dicts from /mine response
    target_collection: str = "default"  # Which collection to commit into


@router.post("/sessions/{session_id}/push-mined")
async def push_mined_proposals(session_id: str, req: PushMinedRequest):
    """Push mined proposals directly into the session draft stack.

    Converts raw mined proposals (from /mine) into DraftPackets without
    re-running LLM extraction. Each proposal becomes a DRAFT_UNAUTHORIZED
    packet with its raw data preserved for later review/promotion.
    """
    import uuid
    from ..models.schemas import DraftPacket, DraftStack
    from ..models.enums import DraftStatus

    meta_path = session_store._session_dir(session_id) / "meta.json"
    meta = session_store._read_json(meta_path)
    if not meta:
        raise HTTPException(404, "Session not found")

    stack = session_store.load_draft_stack(session_id)
    if not stack:
        stack = DraftStack(session_id=session_id)

    created = []
    for prop in req.proposals:
        if not isinstance(prop, dict):
            continue
        prop_type = prop.get("type", "anchor")
        text = prop.get("canonical_phrase") or prop.get("canonical_text", "") or prop.get("title", "")
        if not text:
            continue

        draft_id = f"mined_{prop_type}_{uuid.uuid4().hex[:8]}_v1"

        packet = DraftPacket(
            id=draft_id,
            source_chat_id=meta.get("chat_id", session_id),
            source_turns=[0],
            proposed_nodes=[draft_id],
            justification=prop.get("justification", ""),
            confidence=prop.get("confidence", 0.5),
            status=DraftStatus.DRAFT_UNAUTHORIZED,
            fact_claims=[text] if prop.get("claim_tag") == "FACT" else [],
        )

        session_store.save_draft_packet(session_id, packet)
        # Store raw proposal + target collection for commit step
        prop["_target_collection"] = req.target_collection
        raw_path = session_store._drafts_dir(session_id) / f"{draft_id}_raw.json"
        session_store._write_json(raw_path, prop)

        stack.packets.append(draft_id)
        created.append({"id": draft_id, "type": prop_type, "label": text[:80]})

    session_store.save_draft_stack(session_id, stack)
    return {"created": len(created), "drafts": created}
