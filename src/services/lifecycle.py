"""LifecycleService — coordinator for atomic multi-location state transitions.

Five state locations hold draft/tentative data and must stay consistent:

  1. ``app/library/tentative/*.json``              — disk, shared across sessions
  2. Session draft stack                           — ``{session}/drafts/*.json``
  3. Session tentative registry                    — ``{session}/tentative_state.json``
  4. Session frame active_nodes                    — ``{session}/frame.json``
  5. Session proposed edges                        — ``{session}/proposed_edges.json``

Historically each user action only touched one or two of these, leaving
"ghost" state in the others (registry entries without library files,
ACCEPTED edges whose endpoints never landed, etc.). LifecycleService wraps
each user-facing operation in a single atomic method that updates every
location that should move together, and exposes ``run_startup_sweep`` to
reconcile any dangling state from before this code existed.

Design notes:

- **Coordinator, not redesign.** Existing verbs on ``FrameManager``,
  ``DraftManager``, and ``SessionStore`` stay exactly as they are. This
  service composes them into atomic sequences.
- **Single-process assumption.** FastAPI runs one uvicorn worker here;
  no cross-process locking is needed. Steps are ordered so that any
  mid-sequence crash leaves the system in a state the startup sweep can
  clean up on next boot.
- **Concept-detection artifacts are protected.** ``tentative_concept_*``
  entries in the registry are intentional session-scoped working memory
  (the detector keeps them so dismissed concepts can re-engage). The
  sweep's "orphan registry" pass only touches entries whose ids match
  the draft-flow prefixes (``mined_*``, ``tentative_anchor_*``,
  ``tentative_slab_*``). Concept entries are left alone.
"""
from __future__ import annotations
import json
import logging
import uuid
from pathlib import Path
from typing import Optional

from ..models.schemas import (
    ProposedEdge, Edge as CorpusEdge, FrameState,
)
from .corpus import CorpusStore, CorpusRegistry
from .session_store import SessionStore
from .frame_manager import FrameManager
from .draft_manager import DraftManager
from .chat_store import ChatStore
from .event_log import EventLog

logger = logging.getLogger(__name__)


# IDs starting with these prefixes came through the mining/draft→library flow
# and should be garbage-collected when their library record vanishes.
# ``tentative_concept_*`` (concept detection) and ``tentative_bundle_*``
# (user-created in-session bundles) are intentionally excluded — they have
# different lifecycle rules.
_LIBRARY_FLOW_PREFIXES = ("mined_", "tentative_anchor_", "tentative_slab_")

_LIBRARY_TENTATIVE_DIR = Path("app/library/tentative")


class LifecycleService:
    """Atomic coordinator across the five draft/tentative state locations."""

    def __init__(
        self,
        corpus: CorpusStore,
        registry: CorpusRegistry,
        session_store: SessionStore,
        frame_manager: FrameManager,
        draft_manager: DraftManager,
        chat_store: ChatStore,
        event_log: EventLog,
    ):
        self.corpus = corpus
        self.registry = registry
        self.session_store = session_store
        self.frame_manager = frame_manager
        self.draft_manager = draft_manager
        self.chat_store = chat_store
        self.event_log = event_log

    # ───────────────────────── private helpers ─────────────────────────

    def _session_ids(self) -> list[str]:
        """All session ids known to the disk store (not just live in memory)."""
        root = self.session_store.root
        if not root.exists():
            return []
        return sorted(d.name for d in root.iterdir() if d.is_dir())

    def _library_ids(self) -> set[str]:
        """Every id currently present in the library/tentative directory."""
        if not _LIBRARY_TENTATIVE_DIR.exists():
            return set()
        return {p.stem for p in _LIBRARY_TENTATIVE_DIR.glob("*.json")}

    def _corpus_ids(self) -> set[str]:
        """All anchor/slab/bundle ids in the current merged corpus view."""
        return set(self.corpus.anchors) | set(self.corpus.slabs) | set(self.corpus.bundles)

    def _find_sessions_holding_node(self, node_id: str) -> list[str]:
        """Return session ids whose registry, frame.active_nodes, or proposed
        edges reference ``node_id``. Prefers in-memory frame_manager state
        when the session is live; otherwise reads from disk.
        """
        hits: set[str] = set()

        # In-memory pass (fast path — covers live sessions)
        for sid, frame in self.frame_manager._frames.items():
            if node_id in frame.active_nodes:
                hits.add(sid)
                continue
            reg = self.frame_manager.get_tentative_registry(sid)
            if any(info.get("id") == node_id for info in reg.values()):
                hits.add(sid)

        # Disk pass (covers cold sessions)
        for sid in self._session_ids():
            if sid in hits:
                continue
            reg_data, _ = self.session_store.load_registry(sid)
            if reg_data and any(
                isinstance(info, dict) and info.get("id") == node_id
                for info in reg_data.values()
            ):
                hits.add(sid)
                continue
            # Check proposed edges on disk too — a session might reference the
            # node only through an edge and have no registry entry.
            for edge in self.session_store.list_proposed_edges(sid):
                if edge.from_node == node_id or edge.to_node == node_id:
                    hits.add(sid)
                    break

        return sorted(hits)

    def _evict_node_from_session(self, session_id: str, node_id: str) -> bool:
        """Hard-evict a node: remove from registry entirely and from frame
        active_nodes + all keyed maps. Persists. Unlike ``dismiss_node`` which
        marks ``dismissed=True``, this is a full removal — used when the node
        has been authoritatively moved (promoted, discarded, deleted).

        Returns True if anything was removed.
        """
        touched = False

        # In-memory path if the session's frame is loaded
        frame = self.frame_manager._frames.get(session_id)
        reg = self.frame_manager._tentative_registry.get(session_id)

        if frame is not None:
            if node_id in frame.active_nodes:
                frame.active_nodes.remove(node_id)
                touched = True
            for d in (
                frame.salience_now, frame.salience_smoothed,
                frame.structural_weight, frame.activation_sources,
                frame.active_anchors, frame.active_bundles,
                frame.active_slabs, frame.active_concepts,
                frame.corpus_hits, frame.corpus_last_hit,
                frame.truth_pressure,
            ):
                if node_id in d:
                    d.pop(node_id, None)
                    touched = True

        if reg is not None:
            names_to_pop = [
                name for name, info in reg.items()
                if isinstance(info, dict) and info.get("id") == node_id
            ]
            for name in names_to_pop:
                reg.pop(name, None)
                touched = True

        # Also filter tentative_edges (session-scoped PARENT_OF etc. stored
        # as plain dicts on _tentative_edges, separate from ProposedEdge)
        t_edges = self.frame_manager._tentative_edges.get(session_id)
        if t_edges is not None:
            keep = [
                e for e in t_edges
                if e.get("from") != node_id and e.get("to") != node_id
            ]
            if len(keep) != len(t_edges):
                self.frame_manager._tentative_edges[session_id] = keep
                touched = True

        if touched and frame is not None:
            self.frame_manager.persist(session_id)
            return True

        # Cold path — session not in memory. Load, mutate, save directly.
        if frame is None:
            reg_data, edges = self.session_store.load_registry(session_id)
            cold_frame = self.session_store.load_frame(session_id)
            if not reg_data and cold_frame is None:
                return False

            names_to_pop = []
            if reg_data:
                for name, info in reg_data.items():
                    if isinstance(info, dict) and info.get("id") == node_id:
                        names_to_pop.append(name)
                for name in names_to_pop:
                    reg_data.pop(name)
                    touched = True

            if cold_frame is not None and node_id in cold_frame.active_nodes:
                cold_frame.active_nodes.remove(node_id)
                for d in (
                    cold_frame.salience_now, cold_frame.salience_smoothed,
                    cold_frame.structural_weight, cold_frame.activation_sources,
                    cold_frame.active_anchors, cold_frame.active_bundles,
                    cold_frame.active_slabs, cold_frame.active_concepts,
                    cold_frame.corpus_hits, cold_frame.corpus_last_hit,
                    cold_frame.truth_pressure,
                ):
                    d.pop(node_id, None)
                touched = True

            filtered_edges = [
                e for e in (edges or [])
                if e.get("from") != node_id and e.get("to") != node_id
            ]
            if len(filtered_edges) != len(edges or []):
                touched = True

            if touched:
                self.session_store.save_registry(
                    session_id, reg_data or {}, filtered_edges,
                )
                if cold_frame is not None:
                    self.session_store.save_frame(session_id, cold_frame)

        return touched

    def _auto_accept_edges_for_node(self, session_id: str, node_id: str) -> list[str]:
        """Lift every PROPOSED edge referencing ``node_id`` to ACCEPTED.

        When a user promotes a node, they implicitly approve the edges
        connecting that node to the rest of the mined batch (otherwise
        they'd be rejecting a node + its relationships piecemeal, which
        is tedious UX for mining output that arrives as a coherent unit).

        Only ``PROPOSED`` transitions. ``ACCEPTED`` edges stay ACCEPTED
        (already in the resolve-sweep pool); ``REJECTED`` edges stay
        rejected (user explicitly said no — auto-accept must not undo
        their decision); ``COMMITTED`` is a terminal state.
        """
        accepted: list[str] = []
        for edge in self.session_store.list_proposed_edges(session_id):
            if edge.status != "PROPOSED":
                continue
            if edge.from_node == node_id or edge.to_node == node_id:
                self.session_store.update_proposed_edge_status(
                    session_id, edge.id, "ACCEPTED",
                )
                accepted.append(edge.id)
        return accepted

    def _reject_edges_for_node(self, session_id: str, node_id: str) -> list[str]:
        """Mark every PROPOSED/ACCEPTED edge with ``node_id`` as an endpoint
        REJECTED. Returns the list of edge ids updated.
        """
        rejected: list[str] = []
        for edge in self.session_store.list_proposed_edges(session_id):
            if edge.status not in ("PROPOSED", "ACCEPTED"):
                continue
            if edge.from_node == node_id or edge.to_node == node_id:
                self.session_store.update_proposed_edge_status(
                    session_id, edge.id, "REJECTED",
                )
                rejected.append(edge.id)
        return rejected

    def _resolve_edge_collection(
        self, from_node: str, to_node: str,
        source_chat_id: Optional[str],
    ) -> str:
        """Decide which collection an edge should be committed to.

        Priority order:
          1. Both endpoints in the same non-default collection → that one.
             (Strongest signal: the edge connects content that all lives
             together.)
          2. Exactly one endpoint in a specific collection (the other in
             'default' or absent from everywhere specific) → the specific
             collection. Captures "narrative beat linked to a default anchor".
          3. Endpoints split across two different specific collections →
             the source chat's bound collection, if set.
          4. Fallback: 'default'.

        Reading collection membership from ``registry.active_ids`` only;
        inactive collections aren't considered. IDs are cheap string
        lookups against each store's three object dicts.
        """
        def _find_coll(node_id: str) -> Optional[str]:
            for cid in sorted(self.registry.active_ids):
                store = self.registry.get_store(cid)
                if store is None:
                    continue
                if (node_id in store.anchors
                        or node_id in store.slabs
                        or node_id in store.bundles):
                    return cid
            return None

        from_coll = _find_coll(from_node)
        to_coll = _find_coll(to_node)

        # (1) Same non-default collection
        if from_coll and from_coll == to_coll and from_coll != "default":
            return from_coll
        # (1b) Both in default
        if from_coll == "default" and to_coll == "default":
            return "default"

        # (2) One specific, one default (or missing)
        specific_candidates = [
            c for c in (from_coll, to_coll)
            if c is not None and c != "default"
        ]
        if len(set(specific_candidates)) == 1:
            return specific_candidates[0]

        # (3) Split across specifics → chat binding
        if source_chat_id:
            try:
                chat = self.chat_store.get_chat(source_chat_id)
                if chat and chat.collection_id:
                    return chat.collection_id
            except Exception as exc:
                logger.warning("Edge routing: chat lookup failed: %r", exc)

        # (4) Fallback
        return "default"

    def _try_commit_accepted_edge(
        self, session_id: str, edge: ProposedEdge,
    ) -> Optional[str]:
        """If both endpoints are now in corpus, materialise a real Edge and
        update the proposed edge to COMMITTED. Returns committed_edge_id or
        None if either endpoint is still missing.

        Mirrors the accept-path logic in ``api/drafts.py`` review_proposed_edge
        (collection resolution via source_chat, fallback to 'default').
        """
        corpus_ids = self._corpus_ids()
        if edge.from_node not in corpus_ids or edge.to_node not in corpus_ids:
            return None

        new_edge_id = f"edge_{uuid.uuid4().hex[:8]}"
        real = CorpusEdge(
            id=new_edge_id,
            type=edge.type,
            **{"from": edge.from_node, "to": edge.to_node},
            weight=edge.confidence,
            confidence=edge.confidence,
        )

        # Route by endpoint collection membership first — if both endpoints
        # live in the same specific collection, the edge belongs there.
        # Only fall back to chat binding when endpoints are split across
        # collections (rare) or when the endpoint lookup fails.
        #
        # This is important because mining a conversation INTO a specific
        # collection (e.g. vindiesel6) doesn't rebind the source chat —
        # the chat stays pointed at 'default' for its own drafts. Using
        # chat.collection_id as the edge-routing signal misrouted all
        # SEQUENCE/LINKS edges from narrative mining to 'default'.
        target_collection_id = self._resolve_edge_collection(
            edge.from_node, edge.to_node, edge.source_chat_id,
        )
        target_store = self.registry.get_store(target_collection_id)
        if target_store is None:
            target_store = self.registry.get_store("default")
            logger.warning(
                "Edge commit for %s: collection %r missing, writing to 'default'",
                edge.id, target_collection_id,
            )

        target_store.edges[new_edge_id] = real
        target_store.save()
        self.registry._merged_dirty = True
        # Note: caller is responsible for triggering rebind_corpus() once at
        # the end of a batch — we don't rebind per-edge to avoid thrash.

        self.session_store.update_proposed_edge_status(
            session_id, edge.id, "COMMITTED", committed_edge_id=new_edge_id,
        )
        return new_edge_id

    def _resolve_accepted_edges_for_node(self, node_id: str) -> list[tuple[str, str]]:
        """Across every session, try to commit ACCEPTED edges touching
        ``node_id`` now that the node may have landed in corpus. Returns
        [(session_id, new_edge_id)] pairs for each successful commit.
        """
        committed: list[tuple[str, str]] = []
        for sid in self._session_ids():
            for edge in self.session_store.list_proposed_edges(sid):
                if edge.status != "ACCEPTED":
                    continue
                if edge.from_node != node_id and edge.to_node != node_id:
                    continue
                new_id = self._try_commit_accepted_edge(sid, edge)
                if new_id:
                    committed.append((sid, new_id))
        return committed

    # ────────────────────────── public verbs ────────────────────────────

    async def discard_draft(self, session_id: str, draft_id: str) -> dict:
        """Atomically discard a draft: mark REJECTED, evict from session
        frame/registry, reject referencing edges.
        """
        result = await self.draft_manager.review_draft(
            session_id, draft_id, "discard",
        )
        if "error" in result:
            return result

        evicted = self._evict_node_from_session(session_id, draft_id)
        rejected_edges = self._reject_edges_for_node(session_id, draft_id)

        self.event_log.log_proposal_event(
            event="lifecycle.draft_discarded",
            session_id=session_id,
            draft_id=draft_id,
            evicted_from_frame=evicted,
            edges_rejected=len(rejected_edges),
        )
        result.update({
            "evicted_from_frame": evicted,
            "edges_rejected": rejected_edges,
        })
        return result

    async def promote_draft_tentative(
        self, session_id: str, draft_id: str,
        drift_severity: str = "low",
    ) -> dict:
        """Promote a draft to library/tentative: write library file, mark
        packet PROVISIONAL, and evict session-scope state (library now owns).

        Referencing ``PROPOSED`` edges are auto-accepted (not rejected), so
        when the node eventually promotes to corpus via
        ``promote_library_tentative`` the resolve-sweep can commit edge pairs
        whose other endpoint is also live. Rejecting edges here would
        orphan them permanently — a bad outcome for mining batches where
        the user intends to promote several related nodes together.
        """
        result = await self.draft_manager.review_draft(
            session_id, draft_id, "promote_tentative",
            drift_severity=drift_severity,
        )
        if "error" in result:
            return result

        evicted = self._evict_node_from_session(session_id, draft_id)
        accepted_edges = self._auto_accept_edges_for_node(session_id, draft_id)

        self.event_log.log_proposal_event(
            event="lifecycle.draft_promoted_tentative",
            session_id=session_id,
            draft_id=draft_id,
            evicted_from_frame=evicted,
            edges_auto_accepted=len(accepted_edges),
        )
        result.update({
            "evicted_from_frame": evicted,
            "edges_auto_accepted": accepted_edges,
        })
        return result

    async def promote_draft_corpus(
        self, session_id: str, draft_id: str,
        oli_mode: str = "OFF",
        drift_severity: str = "low",
    ) -> dict:
        """Promote a draft to corpus: write corpus object, mark packet
        COMMITTED, evict session-scope state, auto-accept any PROPOSED edges
        referencing this node, then resolve ACCEPTED edges across all
        sessions whose endpoints may now both be corpus-live.

        The auto-accept step is what lets a bulk-promote of a mining batch
        commit edges alongside nodes without a separate edge-review pass.
        When the first endpoint promotes, edges touching it transition
        PROPOSED → ACCEPTED (pending the other endpoint). When the second
        endpoint promotes, the resolve-sweep finds them both live and
        commits the edge. Edges the user explicitly rejected stay REJECTED.
        """
        result = await self.draft_manager.review_draft(
            session_id, draft_id, "promote_corpus", oli_mode,
            drift_severity=drift_severity,
        )
        if "error" in result:
            return result

        evicted = self._evict_node_from_session(session_id, draft_id)

        # Lift PROPOSED edges touching this draft to ACCEPTED, so the
        # resolve-sweep below can commit any pair already fully corpus-live.
        accepted_edges = self._auto_accept_edges_for_node(session_id, draft_id)

        # The corpus object id typically equals draft_id (see
        # DraftManager._convert_to_corpus_object). Resolve any ACCEPTED edges
        # across all sessions (covers this session's just-lifted ones plus
        # any other session that accepted an edge pointing at this id).
        committed_edges = self._resolve_accepted_edges_for_node(draft_id)
        if committed_edges:
            from ..api import deps as _deps  # local import to avoid cycle
            _deps.rebind_corpus()

        self.event_log.log_proposal_event(
            event="lifecycle.draft_promoted_corpus",
            session_id=session_id,
            draft_id=draft_id,
            evicted_from_frame=evicted,
            edges_auto_accepted=len(accepted_edges),
            edges_committed=len(committed_edges),
        )
        result.update({
            "evicted_from_frame": evicted,
            "edges_auto_accepted": accepted_edges,
            "edges_committed": [eid for _, eid in committed_edges],
        })
        return result

    def delete_library_tentative(self, tid: str) -> dict:
        """Delete a library/tentative record AND purge it from every session
        that still has it in registry, frame, or proposed edges.
        """
        ok = self.draft_manager.delete_library_tentative(tid)
        if not ok:
            return {"error": "Tentative record not found", "id": tid}

        sessions_cleaned: list[str] = []
        edges_rejected_total = 0
        for sid in self._find_sessions_holding_node(tid):
            evicted = self._evict_node_from_session(sid, tid)
            rejected = self._reject_edges_for_node(sid, tid)
            if evicted or rejected:
                sessions_cleaned.append(sid)
                edges_rejected_total += len(rejected)

        self.event_log.log_proposal_event(
            event="lifecycle.library_tentative_deleted",
            id=tid,
            sessions_cleaned=len(sessions_cleaned),
            edges_rejected=edges_rejected_total,
        )
        return {
            "status": "deleted",
            "id": tid,
            "sessions_cleaned": sessions_cleaned,
            "edges_rejected": edges_rejected_total,
        }

    def promote_library_tentative(self, tid: str) -> dict:
        """Promote a library/tentative record to corpus AND purge it from every
        session that held it, AND resolve ACCEPTED edges referencing it.
        """
        result = self.draft_manager.promote_library_tentative(tid)
        if "error" in result:
            return result

        sessions_cleaned: list[str] = []
        for sid in self._find_sessions_holding_node(tid):
            if self._evict_node_from_session(sid, tid):
                sessions_cleaned.append(sid)

        committed_edges = self._resolve_accepted_edges_for_node(tid)
        if committed_edges:
            from ..api import deps as _deps
            _deps.rebind_corpus()

        self.event_log.log_proposal_event(
            event="lifecycle.library_tentative_promoted",
            id=tid,
            sessions_cleaned=len(sessions_cleaned),
            edges_committed=len(committed_edges),
        )
        result.update({
            "sessions_cleaned": sessions_cleaned,
            "edges_committed": [eid for _, eid in committed_edges],
        })
        return result

    def accept_proposed_edge(self, session_id: str, edge_id: str) -> dict:
        """Accept a proposed edge. Commits immediately if both endpoints live
        in corpus; otherwise marks ACCEPTED pending endpoint promotion.
        """
        edges = self.session_store.list_proposed_edges(session_id)
        target = next((e for e in edges if e.id == edge_id), None)
        if target is None:
            return {"error": "Proposed edge not found", "status": 404}

        new_id = self._try_commit_accepted_edge(session_id, target)
        if new_id:
            from ..api import deps as _deps
            _deps.rebind_corpus()
            return {"ok": True, "status": "COMMITTED", "committed_edge_id": new_id}

        self.session_store.update_proposed_edge_status(
            session_id, edge_id, "ACCEPTED",
        )
        return {
            "ok": True,
            "status": "ACCEPTED",
            "note": "Awaiting endpoint promotion",
        }

    def reject_proposed_edge(self, session_id: str, edge_id: str) -> dict:
        """Reject a proposed edge (kept for audit, never committed)."""
        ok = self.session_store.update_proposed_edge_status(
            session_id, edge_id, "REJECTED",
        )
        if not ok:
            return {"error": "Proposed edge not found", "status": 404}
        return {"ok": True, "status": "REJECTED"}

    def dismiss_node(self, session_id: str, node_id: str) -> dict:
        """Dismiss a tentative node (frame-only eviction, registry entry
        kept with ``dismissed=True`` so concept detector can re-engage).
        Thin delegate to ``frame_manager.dismiss_node`` for uniformity.
        """
        from .frame_manager import FrameNotFoundError
        try:
            self.frame_manager.dismiss_node(session_id, node_id)
        except FrameNotFoundError:
            return {"error": "No active frame", "status": 404}
        return {"ok": True, "status": "dismissed", "node_id": node_id}

    def reject_node(self, session_id: str, node_id: str) -> dict:
        """Reject a tentative node (registry entry marked rejected).
        Thin delegate to ``frame_manager.reject_node``.
        """
        self.frame_manager.reject_node(session_id, node_id)
        return {"ok": True, "status": "rejected", "node_id": node_id}

    # ─────────────────────────── startup sweep ──────────────────────────

    def run_startup_sweep(self) -> dict:
        """One-shot reconciliation across all five locations.

        Performs three passes:

        1. **Auto-commit dangling ACCEPTED edges** — for every session, try
           to commit each ACCEPTED edge now that corpus may have gained the
           missing endpoint(s).
        2. **Prune orphan registry entries** — remove library-flow entries
           (prefixes: mined_/tentative_anchor_/tentative_slab_) whose id is
           not in corpus, not in library, and not a live DRAFT_UNAUTHORIZED
           packet.
        3. **Reject fully-orphan edges** — proposed edges whose endpoints
           are nowhere (corpus, library, active drafts).

        Returns a summary of what changed. Called once at server startup.
        """
        corpus_ids = self._corpus_ids()
        library_ids = self._library_ids()

        summary = {
            "sessions_scanned": 0,
            "edges_committed": 0,
            "registry_cleaned": 0,
            "edges_orphaned": 0,
        }

        for sid in self._session_ids():
            summary["sessions_scanned"] += 1

            # Which drafts are still awaiting review in this session?
            live_draft_ids = self._live_draft_ids(sid)

            # Pass 1: auto-commit ACCEPTED edges
            for edge in self.session_store.list_proposed_edges(sid):
                if edge.status != "ACCEPTED":
                    continue
                if (edge.from_node in corpus_ids and
                        edge.to_node in corpus_ids):
                    new_id = self._try_commit_accepted_edge(sid, edge)
                    if new_id:
                        summary["edges_committed"] += 1

            # Pass 2: prune orphan registry entries (library-flow only)
            reg_data, tent_edges = self.session_store.load_registry(sid)
            if reg_data:
                orphan_names = []
                orphan_ids = set()
                for name, info in reg_data.items():
                    if not isinstance(info, dict):
                        continue
                    nid = info.get("id")
                    if not nid or not nid.startswith(_LIBRARY_FLOW_PREFIXES):
                        continue
                    if (nid in corpus_ids or nid in library_ids
                            or nid in live_draft_ids):
                        continue
                    orphan_names.append(name)
                    orphan_ids.add(nid)

                if orphan_names:
                    for name in orphan_names:
                        reg_data.pop(name, None)
                    for nid in orphan_ids:
                        self._evict_node_from_session(sid, nid)
                    # Evict also cleans tent_edges through its cold path if
                    # session isn't in memory; but we already loaded reg+edges
                    # above so re-save explicitly to be sure.
                    filtered = [
                        e for e in (tent_edges or [])
                        if e.get("from") not in orphan_ids
                        and e.get("to") not in orphan_ids
                    ]
                    self.session_store.save_registry(sid, reg_data, filtered)
                    summary["registry_cleaned"] += len(orphan_names)

            # Pass 3: reject fully-orphan edges (re-list after Pass 2
            # since pass 2 may have rejected some already via eviction)
            for edge in self.session_store.list_proposed_edges(sid):
                if edge.status not in ("PROPOSED", "ACCEPTED"):
                    continue
                if self._endpoint_exists(edge.from_node, corpus_ids, library_ids, live_draft_ids):
                    if self._endpoint_exists(edge.to_node, corpus_ids, library_ids, live_draft_ids):
                        continue
                # At least one endpoint is nowhere — orphan the edge
                self.session_store.update_proposed_edge_status(
                    sid, edge.id, "REJECTED",
                )
                summary["edges_orphaned"] += 1

        if summary["edges_committed"] > 0:
            # Rebind once at the end of the sweep so the merged view includes
            # newly-committed edges.
            try:
                from ..api import deps as _deps
                _deps.rebind_corpus()
            except Exception as exc:
                logger.warning("rebind_corpus after sweep failed: %r", exc)

        self.event_log.log_proposal_event(
            event="lifecycle.startup_sweep",
            **summary,
        )
        return summary

    def _live_draft_ids(self, session_id: str) -> set[str]:
        """DRAFT_UNAUTHORIZED packet ids still awaiting review in ``session_id``."""
        result: set[str] = set()
        drafts_dir = self.session_store.drafts_dir(session_id)
        if not drafts_dir.exists():
            return result
        for f in drafts_dir.glob("*.json"):
            name = f.stem
            if name.endswith("_raw") or name.endswith(".enriched"):
                continue
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if data.get("status") == "DRAFT_UNAUTHORIZED":
                    result.add(name)
            except Exception:
                continue
        return result

    @staticmethod
    def _endpoint_exists(
        node_id: str,
        corpus_ids: set[str],
        library_ids: set[str],
        live_draft_ids: set[str],
    ) -> bool:
        """True if ``node_id`` lives in corpus, library/tentative, or the
        session's live draft stack. Concept-detection artifacts
        (``tentative_concept_*``) are NOT considered here — edges endpointing
        at a detected concept should wait for the concept to be formalized.
        """
        return (
            node_id in corpus_ids
            or node_id in library_ids
            or node_id in live_draft_ids
        )
