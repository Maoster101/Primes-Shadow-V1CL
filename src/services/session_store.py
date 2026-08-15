"""Session persistence service.

A session is a contiguous activity period within a chat.
Stores FrameState snapshots and DraftStacks on disk so they
survive app restarts within a session.
"""
from __future__ import annotations
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..models.schemas import FrameState, DraftStack, DraftPacket, ProposedEdge
from .atomic_io import atomic_write_json

SESSIONS_ROOT = Path("app/workbench/sessions")


class SessionStore:

    def __init__(self, root: Optional[Path] = None):
        self.root = root or SESSIONS_ROOT
        self.root.mkdir(parents=True, exist_ok=True)

    def session_dir(self, session_id: str) -> Path:
        return self.root / session_id

    def drafts_dir(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "drafts"

    def create_session(self, chat_id: str) -> str:
        session_id = str(uuid.uuid4())[:8]
        d = self.session_dir(session_id)
        d.mkdir(parents=True, exist_ok=True)
        self.drafts_dir(session_id).mkdir(exist_ok=True)
        meta = {
            "session_id": session_id,
            "chat_id": chat_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "active",
        }
        self.write_json(d / "meta.json", meta)
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
            meta = self.read_json(meta_path)
            if meta and meta.get("chat_id") == chat_id and meta.get("status") == "active":
                return meta["session_id"]
        return None

    # --- FrameState ---

    def save_frame(self, session_id: str, frame: FrameState) -> None:
        path = self.session_dir(session_id) / "frame.json"
        self.write_json(path, frame.model_dump(mode="json"))

    def load_frame(self, session_id: str) -> Optional[FrameState]:
        path = self.session_dir(session_id) / "frame.json"
        data = self.read_json(path)
        return FrameState(**data) if data else None

    # --- Tentative registry + edges (persisted for restore) ---

    def save_registry(self, session_id: str, registry: dict, edges: list) -> None:
        path = self.session_dir(session_id) / "tentative_state.json"
        self.write_json(path, {"registry": registry, "edges": edges})

    def load_registry(self, session_id: str) -> tuple:
        """Returns (registry_dict, edges_list) or ({}, [])."""
        path = self.session_dir(session_id) / "tentative_state.json"
        data = self.read_json(path)
        if not data:
            return {}, []
        return data.get("registry", {}), data.get("edges", [])

    # --- DraftStack ---

    def save_draft_stack(self, session_id: str, stack: DraftStack) -> None:
        path = self.session_dir(session_id) / "draft_stack.json"
        self.write_json(path, stack.model_dump(mode="json"))

    def load_draft_stack(self, session_id: str) -> Optional[DraftStack]:
        path = self.session_dir(session_id) / "draft_stack.json"
        data = self.read_json(path)
        return DraftStack(**data) if data else None

    # --- DraftPackets ---

    def save_draft_packet(self, session_id: str, packet: DraftPacket) -> None:
        path = self.drafts_dir(session_id) / f"{packet.id}.json"
        self.write_json(path, packet.model_dump(mode="json"))

    def load_draft_packet(self, session_id: str, draft_id: str) -> Optional[DraftPacket]:
        path = self.drafts_dir(session_id) / f"{draft_id}.json"
        data = self.read_json(path)
        return DraftPacket(**data) if data else None

    def list_draft_packets(self, session_id: str) -> list[DraftPacket]:
        d = self.drafts_dir(session_id)
        if not d.exists():
            return []
        packets = []
        for f in sorted(d.glob("*.json")):
            if f.name.endswith("_raw.json"):
                continue  # Skip raw proposal files — not DraftPackets
            # Skip sidecar JSONs that aren't DraftPacket records
            if f.name.endswith(".enriched.json"):
                continue
            data = self.read_json(f)
            if data:
                try:
                    packets.append(DraftPacket(**data))
                except Exception:
                    continue  # Skip malformed drafts rather than crashing review
        return packets

    # Phase 3 — Proposed edges storage (session-scoped, chat-aggregated)

    def _edges_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "proposed_edges.json"

    def list_proposed_edges(self, session_id: str) -> list[ProposedEdge]:
        """Load this session's proposed edges. Empty list if none."""
        data = self.read_json(self._edges_path(session_id))
        if not data or "edges" not in data:
            return []
        out = []
        for e in data["edges"]:
            try:
                out.append(ProposedEdge(**e))
            except Exception:
                continue  # skip malformed rather than crash the review UI
        return out

    def save_proposed_edges(self, session_id: str, edges: list[ProposedEdge]) -> None:
        """Overwrite the full proposed_edges list for a session."""
        self.session_dir(session_id).mkdir(parents=True, exist_ok=True)
        self.write_json(
            self._edges_path(session_id),
            {"session_id": session_id, "edges": [e.model_dump(mode="json") for e in edges]},
        )

    def append_proposed_edges(self, session_id: str, new_edges: list[ProposedEdge]) -> None:
        """Add new proposed edges to the existing list, deduped by (from,to,type)."""
        if not new_edges:
            return
        existing = self.list_proposed_edges(session_id)
        existing_keys = {(e.from_node, e.to_node, e.type) for e in existing}
        for edge in new_edges:
            if (edge.from_node, edge.to_node, edge.type) not in existing_keys:
                existing.append(edge)
                existing_keys.add((edge.from_node, edge.to_node, edge.type))
        self.save_proposed_edges(session_id, existing)

    def update_proposed_edge_status(
        self, session_id: str, edge_id: str, status: str,
        committed_edge_id: Optional[str] = None,
    ) -> bool:
        """Mark an edge PROPOSED/ACCEPTED/COMMITTED/REJECTED. Returns True on match."""
        edges = self.list_proposed_edges(session_id)
        hit = False
        for e in edges:
            if e.id == edge_id:
                e.status = status
                if committed_edge_id is not None:
                    e.committed_edge_id = committed_edge_id
                hit = True
                break
        if hit:
            self.save_proposed_edges(session_id, edges)
        return hit

    def list_proposed_edges_by_chat(self, chat_id: str) -> list[tuple[str, ProposedEdge]]:
        """Aggregate proposed edges across ALL sessions for a chat.

        Returns list of (session_id, edge) tuples so callers can route
        review actions back to the owning session. Mirrors the per-chat
        draft listing pattern established in Phase 2A.
        """
        if not self.root.exists():
            return []
        out: list[tuple[str, ProposedEdge]] = []
        for sess_dir in self.root.iterdir():
            if not sess_dir.is_dir():
                continue
            edges_path = sess_dir / "proposed_edges.json"
            if not edges_path.exists():
                continue
            data = self.read_json(edges_path) or {}
            for e in data.get("edges", []):
                if e.get("source_chat_id") != chat_id:
                    continue
                try:
                    out.append((sess_dir.name, ProposedEdge(**e)))
                except Exception:
                    continue
        return out

    def list_draft_packets_by_chat(self, chat_id: str) -> list[DraftPacket]:
        """Return every DraftPacket across ALL sessions where source_chat_id matches.

        Drafts are stored per-session, but a chat persists across many sessions
        (e.g. after restart). For UI purposes the draft belongs to the chat it
        was mined in, not the SSE session that happened to be live. This scans
        every session dir once and filters. Deduped by packet id (latest write
        wins via sort on path mtime -> later).
        """
        if not self.root.exists():
            return []
        by_id: dict[str, DraftPacket] = {}
        for sess_dir in self.root.iterdir():
            if not sess_dir.is_dir():
                continue
            drafts_dir = sess_dir / "drafts"
            if not drafts_dir.exists():
                continue
            for f in sorted(drafts_dir.glob("*.json"), key=lambda p: p.stat().st_mtime):
                name = f.name
                if name.endswith("_raw.json") or name.endswith(".enriched.json"):
                    continue
                data = self.read_json(f)
                if not data or data.get("source_chat_id") != chat_id:
                    continue
                try:
                    pkt = DraftPacket(**data)
                    by_id[pkt.id] = pkt  # last write wins
                except Exception:
                    continue
        return list(by_id.values())

    # --- Helpers ---

    def write_json(self, path: Path, data: dict) -> None:
        """Atomic JSON save.

        Previously used a fixed ``.tmp`` suffix which could collide
        between concurrent writes of the same logical path. Now
        delegates to the shared atomic helper, which uses a unique
        temp name, fsyncs before rename, and os.replace for
        cross-platform atomicity.
        """
        atomic_write_json(path, data)

    def read_json(self, path: Path) -> Optional[dict]:
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
