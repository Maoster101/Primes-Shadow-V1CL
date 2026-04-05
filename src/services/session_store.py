"""Session persistence service.

A session is a contiguous activity period within a chat.
Stores FrameState snapshots and DraftStacks on disk so they
survive app restarts within a session.
"""
from __future__ import annotations
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..models.schemas import FrameState, DraftStack, DraftPacket

SESSIONS_ROOT = Path("app/workbench/sessions")


class SessionStore:

    def __init__(self, root: Optional[Path] = None):
        self.root = root or SESSIONS_ROOT
        self.root.mkdir(parents=True, exist_ok=True)

    def _session_dir(self, session_id: str) -> Path:
        return self.root / session_id

    def _drafts_dir(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "drafts"

    def create_session(self, chat_id: str) -> str:
        session_id = str(uuid.uuid4())[:8]
        d = self._session_dir(session_id)
        d.mkdir(parents=True, exist_ok=True)
        self._drafts_dir(session_id).mkdir(exist_ok=True)
        meta = {
            "session_id": session_id,
            "chat_id": chat_id,
            "created_at": datetime.utcnow().isoformat(),
            "status": "active",
        }
        self._write_json(d / "meta.json", meta)
        # Initialise empty draft stack
        stack = DraftStack(session_id=session_id)
        self.save_draft_stack(session_id, stack)
        return session_id

    def get_active_session(self, chat_id: str) -> Optional[str]:
        """Find most recent active session for a chat."""
        for d in sorted(self.root.iterdir(), reverse=True):
            meta_path = d / "meta.json"
            if not meta_path.exists():
                continue
            meta = self._read_json(meta_path)
            if meta and meta.get("chat_id") == chat_id and meta.get("status") == "active":
                return meta["session_id"]
        return None

    # --- FrameState ---

    def save_frame(self, session_id: str, frame: FrameState) -> None:
        path = self._session_dir(session_id) / "frame.json"
        self._write_json(path, frame.model_dump(mode="json"))

    def load_frame(self, session_id: str) -> Optional[FrameState]:
        path = self._session_dir(session_id) / "frame.json"
        data = self._read_json(path)
        return FrameState(**data) if data else None

    # --- Tentative registry + edges (persisted for restore) ---

    def save_registry(self, session_id: str, registry: dict, edges: list) -> None:
        path = self._session_dir(session_id) / "tentative_state.json"
        self._write_json(path, {"registry": registry, "edges": edges})

    def load_registry(self, session_id: str) -> tuple:
        """Returns (registry_dict, edges_list) or ({}, [])."""
        path = self._session_dir(session_id) / "tentative_state.json"
        data = self._read_json(path)
        if not data:
            return {}, []
        return data.get("registry", {}), data.get("edges", [])

    # --- DraftStack ---

    def save_draft_stack(self, session_id: str, stack: DraftStack) -> None:
        path = self._session_dir(session_id) / "draft_stack.json"
        self._write_json(path, stack.model_dump(mode="json"))

    def load_draft_stack(self, session_id: str) -> Optional[DraftStack]:
        path = self._session_dir(session_id) / "draft_stack.json"
        data = self._read_json(path)
        return DraftStack(**data) if data else None

    # --- DraftPackets ---

    def save_draft_packet(self, session_id: str, packet: DraftPacket) -> None:
        path = self._drafts_dir(session_id) / f"{packet.id}.json"
        self._write_json(path, packet.model_dump(mode="json"))

    def load_draft_packet(self, session_id: str, draft_id: str) -> Optional[DraftPacket]:
        path = self._drafts_dir(session_id) / f"{draft_id}.json"
        data = self._read_json(path)
        return DraftPacket(**data) if data else None

    def list_draft_packets(self, session_id: str) -> list[DraftPacket]:
        d = self._drafts_dir(session_id)
        if not d.exists():
            return []
        packets = []
        for f in sorted(d.glob("*.json")):
            data = self._read_json(f)
            if data:
                packets.append(DraftPacket(**data))
        return packets

    # --- Helpers ---

    def _write_json(self, path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)
        tmp.replace(path)

    def _read_json(self, path: Path) -> Optional[dict]:
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
