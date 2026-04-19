"""Chat persistence service.

Chats persist locally until user archives or closes them.
No implicit cross-chat loading of assumptions or conclusions (§4.3).
"""
from __future__ import annotations
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..models.schemas import Chat, ChatMessage
from ..models.enums import ChatStatus
from .atomic_io import atomic_write_json

CHATS_ROOT = Path("app/chats")


class ChatStore:
    """File-backed chat persistence. One JSON file per chat."""

    def __init__(self, root: Optional[Path] = None):
        self.root = root or CHATS_ROOT
        self.root.mkdir(parents=True, exist_ok=True)

    def _chat_path(self, chat_id: str) -> Path:
        return self.root / f"{chat_id}.json"

    def _load_raw(self, chat_id: str) -> Optional[dict]:
        path = self._chat_path(chat_id)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_raw(self, chat_id: str, data: dict) -> None:
        """Atomic JSON save — crash-safe via temp-and-rename + fsync.

        Chat transcripts are append-heavy (every user turn appends a
        message pair), so a non-atomic save could lose the whole
        chat if the process dies mid-write. atomic_write_json
        guarantees the file is either pre-update or fully-updated.
        """
        atomic_write_json(self._chat_path(chat_id), data)

    def create_chat(self, title: str = "New Chat", collection_id: Optional[str] = None) -> Chat:
        chat = Chat(id=str(uuid.uuid4())[:8], title=title, collection_id=collection_id)
        self._save_raw(chat.id, {
            "meta": chat.model_dump(mode="json"),
            "messages": [],
        })
        return chat

    def update_collection(self, chat_id: str, collection_id: Optional[str]) -> bool:
        """Re-bind a chat to a different collection (or null to unset)."""
        raw = self._load_raw(chat_id)
        if not raw:
            return False
        raw["meta"]["collection_id"] = collection_id
        self._save_raw(chat_id, raw)
        return True

    def update_title(self, chat_id: str, title: str) -> bool:
        """Rename a chat. Manual or auto. Stored as-is after strip()."""
        title = (title or "").strip()
        if not title:
            return False
        raw = self._load_raw(chat_id)
        if not raw:
            return False
        raw["meta"]["title"] = title
        raw["meta"]["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save_raw(chat_id, raw)
        return True

    def get_chat(self, chat_id: str) -> Optional[Chat]:
        raw = self._load_raw(chat_id)
        if not raw:
            return None
        return Chat(**raw["meta"])

    def list_chats(self) -> list[Chat]:
        chats = []
        for path in sorted(self.root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                chats.append(Chat(**raw["meta"]))
            except (json.JSONDecodeError, KeyError):
                continue
        return chats

    def get_messages(self, chat_id: str) -> list[ChatMessage]:
        raw = self._load_raw(chat_id)
        if not raw:
            return []
        return [ChatMessage(**m) for m in raw.get("messages", [])]

    def append_message(self, chat_id: str, message: ChatMessage) -> None:
        raw = self._load_raw(chat_id)
        if not raw:
            raise ValueError(f"Chat {chat_id} not found")
        raw["messages"].append(message.model_dump(mode="json"))
        raw["meta"]["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save_raw(chat_id, raw)

    def update_status(self, chat_id: str, status: ChatStatus) -> None:
        raw = self._load_raw(chat_id)
        if not raw:
            raise ValueError(f"Chat {chat_id} not found")
        raw["meta"]["status"] = status.value
        raw["meta"]["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save_raw(chat_id, raw)

    def get_message_window(self, chat_id: str, last_n: int = 20) -> list[ChatMessage]:
        """Get recent message window for context packing."""
        messages = self.get_messages(chat_id)
        return messages[-last_n:] if len(messages) > last_n else messages
