"""FrameState runtime manager — §9.

Session-scoped. The model proposes salience; code stores, smooths,
detects conflicts, and enforces caps. LLM = sensor, code = actuator.
"""
from __future__ import annotations
import json
from typing import Optional

from ..models.schemas import (
    FrameState, ActivationSource, ConflictPair,
)
from ..models.enums import EdgeType, OLIMode, SlabType
from ..models.enums import MessageFunction
from ..prompts.classification import SALIENCE_PROMPT, CONCEPT_DETECT_PROMPT
from .corpus import CorpusStore
from .anchor_matcher import AnchorMatchResult
from . import ollama
from .event_log import EventLog
from .policy import policy
from .session_store import SessionStore


class FrameNotFoundError(LookupError):
    """Raised when a lifecycle verb is called on a session with no active frame.

    Subclass of LookupError so existing `except KeyError` callers still catch
    it, but narrow enough that route handlers can translate it to an HTTP 404
    without swallowing unrelated lookup failures.
    """
    def __init__(self, session_id: str) -> None:
        super().__init__(f"No active frame for session: {session_id}")
        self.session_id = session_id

# All thresholds now read from frame_policy.yaml via the policy singleton.
# Legacy aliases for any remaining inline references (prefer policy.frame.*):
SALIENCE_ALPHA = policy.frame.salience.alpha
MAX_ACTIVE_NODES = policy.frame.limits.max_active_nodes
TENTATIVE_CONSOLIDATION_THRESHOLD = policy.frame.concepts.consolidation_similarity
CONCEPT_TO_ANCHOR_THRESHOLD = policy.frame.concepts.promotion_turns
BUNDLE_SUGGESTION_THRESHOLD = policy.frame.concepts.bundle_suggestion_turns

_event_log = EventLog()


# §9 Cascade semantics — type-specific propagation kernels.
#
# Applied as a multiplier on `edge.weight * edge.confidence` when propagating
# activation from source to target. Captures the idea that different edge
# types carry activation at different strengths:
#
#   INVOKES    — full cascade. "When source fires, target fires." The anchor
#                → slab/bundle invocation chain lives here; kernel 1.0 matches
#                the hand-curated weight 1.0 * confidence 1.0 behaviour so
#                existing INVOKES edges cascade identically to the old
#                hardcoded 1.0 path.
#   SUPPORTS   — reinforcement, not activation. Source strengthens target's
#                *salience* when both are already active (see Step 4b), but
#                does not auto-activate the target. Kernel 0.5 means at max
#                weight*confidence the reinforcement boost is 50% of source
#                salience (further damped by 0.1 below to stay subtle).
#   LINKS      — co-activation. Weaker than INVOKES; "these belong together"
#                rather than "this fires that". Kernel 0.7.
#   CONFLICTS  — antagonism; handled separately in Step 5 conflict detection,
#                not positive cascade. Kernel 0.0 so accidental co-firing via
#                the generic cascade path is a no-op.
#   PARENT_OF  — structural hierarchy. Parent activation partially activates
#                children. Kernel 0.8.
#   SEQUENCE   — temporal. Predecessor hints the successor but doesn't fire
#                it outright. Kernel 0.6.
CASCADE_KERNEL: dict[EdgeType, float] = {
    EdgeType.INVOKES: 1.0,
    EdgeType.SUPPORTS: 0.5,
    EdgeType.LINKS: 0.7,
    EdgeType.CONFLICTS: 0.0,
    EdgeType.PARENT_OF: 0.8,
    EdgeType.SEQUENCE: 0.6,
}

# Damping factor on SUPPORTS reinforcement — keeps the boost sub-dominant
# relative to model-proposed salience. At source_sal=1.0, weight=1.0,
# confidence=1.0, kernel=0.5, the per-turn boost is 0.05.
SUPPORTS_REINFORCE_DAMP = 0.1


class FrameManager:

    def __init__(self, corpus: CorpusStore, session_store: Optional[SessionStore] = None):
        self.corpus = corpus
        # Persistence dependency. Optional so MCP server and tests that don't
        # need disk can construct a FrameManager without one — lifecycle verbs
        # no-op on persistence in that case.
        self.session_store = session_store
        self._frames: dict[str, FrameState] = {}
        # Session-scoped tentative edges (not in corpus until commit)
        self._tentative_edges: dict[str, list[dict]] = {}  # session_id -> [{from, to, type, ...}]
        # Session-scoped tentative nodes: session_id -> { concept_name: { id, description, turns_seen, embedding } }
        self._tentative_registry: dict[str, dict] = {}
        # Cached adjacency index: (from_node, to_node) -> Edge.
        # Built lazily on first cascade lookup. The corpus is append-mostly at
        # runtime (tentative edges live in a separate session-scoped store),
        # so invalidation is rare and handled explicitly in tests that reload.
        self._edge_index: Optional[dict[tuple[str, str], object]] = None

    def _get_edge(self, from_id: str, to_id: str):
        """O(1) edge lookup by endpoint pair. None if no such edge exists."""
        if self._edge_index is None:
            self._edge_index = {
                (e.from_node, e.to_node): e
                for e in self.corpus.edges.values()
            }
        return self._edge_index.get((from_id, to_id))

    def _cascade_strength(
        self, from_id: str, to_id: str, fallback: float
    ) -> float:
        """Return type-weighted cascade strength for from_id → to_id.

        Looks for a reified Edge record linking the two nodes; if found,
        returns `edge.weight * edge.confidence * CASCADE_KERNEL[edge.type]`.
        If no edge exists (e.g. implicit `slab.links.*` relationships for
        which we never minted Edge objects), returns `fallback` — this
        preserves current behaviour for edge-less pairs while letting
        curator-authored edge metadata drive cascades where it exists.
        """
        edge = self._get_edge(from_id, to_id)
        if edge is None:
            return fallback
        kernel = CASCADE_KERNEL.get(edge.type, 0.5)
        return edge.weight * edge.confidence * kernel

    def get_or_create(self, chat_id: str, session_id: str,
                       oli_mode: OLIMode = OLIMode.OFF) -> FrameState:
        """Get or create a session frame, seeding the base set on first call.

        The base set now includes ALL ACTIVE slabs across all active corpus
        collections (CONSTITUTIONAL + CANONICAL + REFERENCE), so the model
        can see anything stored in cold corpus. The graph visibility filter
        keeps REFERENCE nodes hidden from the canvas until conversation
        engages them.
        """
        if session_id in self._frames:
            return self._frames[session_id]
        frame = FrameState(chat_id=chat_id, session_id=session_id)
        self._seed_base_set(frame, oli_mode)
        self._frames[session_id] = frame
        return frame

    # --- Read accessors ---
    # Read-only entry points into session-scoped state. Callers may still
    # mutate through the returned references (live, not copies) — existing
    # read-heavy sites rely on this and the lifecycle verbs below do their
    # own mutation + persistence, so the accessors stay simple.

    def get_frame(self, session_id: str) -> Optional[FrameState]:
        """Return the live FrameState for a session, or None if no frame exists.

        Does NOT create a frame. Use get_or_create() for the chat flow or
        ensure_empty_frame() for the mining flow.
        """
        return self._frames.get(session_id)

    def get_tentative_registry(self, session_id: str) -> dict:
        """Return the session's tentative-node registry, or an empty dict."""
        return self._tentative_registry.get(session_id, {})

    def get_tentative_edges(self, session_id: str) -> list[dict]:
        """Return the session's tentative edges, or an empty list."""
        return self._tentative_edges.get(session_id, [])

    # --- Private helpers ---

    def _find_tentative_by_id(
        self, session_id: str, node_id: str
    ) -> tuple[Optional[str], Optional[dict]]:
        """Locate a registry entry by its node_id. Returns (name, info) or (None, None).

        The registry is keyed by concept-name; this helper is the standard
        entry point for lookup-by-id, used by the dismiss/reject/promote verbs
        and by create_tentative_bundle when resolving children.
        """
        registry = self._tentative_registry.get(session_id, {})
        for name, info in registry.items():
            if info.get("id") == node_id:
                return name, info
        return None, None

    # --- Persistence ---

    def persist(self, session_id: str) -> None:
        """Persist frame + tentative registry + tentative edges to disk.

        No-op if no session_store is wired (MCP server / tests). Idempotent;
        safe to call from verbs that want write-through semantics and from
        route handlers that want to flush state at arbitrary moments.
        """
        if self.session_store is None:
            return
        frame = self._frames.get(session_id)
        if frame is not None:
            self.session_store.save_frame(session_id, frame)
        registry = self._tentative_registry.get(session_id, {})
        edges = self._tentative_edges.get(session_id, [])
        self.session_store.save_registry(session_id, registry, edges)

    # --- Frame constructors ---
    # get_or_create() (defined above) is the chat-flow constructor — it
    # seeds the base set (CONSTITUTIONAL + CANONICAL slabs) so the model
    # has its foundation loaded. ensure_empty_frame() is the mining-flow
    # constructor — draft mining intentionally wants NO base-set activation
    # so mined proposals don't get polluted with pre-loaded corpus slabs.

    def ensure_empty_frame(self, session_id: str, chat_id: str) -> FrameState:
        """Return existing frame or create an empty one (no base-set seeding).

        Used by the draft-mining path where seeding would pollute the set of
        mined proposals with base-set corpus slabs. Callers that want the
        normal chat flow should use get_or_create() instead.
        """
        if session_id in self._frames:
            return self._frames[session_id]
        frame = FrameState(
            session_id=session_id,
            chat_id=chat_id,
            active_nodes=[],
        )
        self._frames[session_id] = frame
        return frame

    # --- Lifecycle verbs ---
    # Each verb mutates session-scoped state in-memory and persists via
    # self.persist(). Route handlers call the verb and then log the
    # corresponding frame/proposal event (logging is a cross-cutting
    # concern kept at the handler level).

    def activate_retrieved_slabs(
        self, session_id: str, slab_ids: list[str],
        weight: float = 0.8,
    ) -> int:
        """Activate slabs in the frame that were retrieved for this turn's prompt.

        This closes the loop between RAG retrieval (which injects slab text
        into the system prompt) and the working-memory graph (which should
        reflect what the model is actually reading). Without this, retrieved
        slabs would be invisible in the graph even while the model is using
        them — breaking the "graph = observable model state" contract.

        Each retrieved slab:
          * Is added to ``active_slabs`` at ``weight`` (default 0.8 — higher
            than base-set 0.6 to reflect that retrieval is a stronger,
            query-driven signal than passive seeding).
          * Is added to ``active_nodes`` if not already present.
          * Gets an ActivationSource with ``source_type="retrieval"`` — the
            graph visibility filter keys off this (non-base_set sources pass
            the filter, so retrieved slabs immediately render on the canvas).
          * Has its linked anchors/bundles activated at cascade weight 0.4,
            so related graph neighbours surface alongside the slab itself.

        Decay is handled by ``update_turn``'s Step 0: slabs retrieved this
        turn stay at ``weight``; slabs retrieved last turn but not this one
        fade naturally. If a slab is retrieved repeatedly its salience stays
        hot; if it falls out of relevance it fades.

        Returns the count of slabs newly added to ``active_nodes``.
        """
        frame = self._frames.get(session_id)
        if frame is None or not slab_ids:
            return 0

        source = ActivationSource(source_type="retrieval", source_ref="rag")
        newly_added = 0
        for sid in slab_ids:
            # Slab itself
            frame.active_slabs[sid] = max(frame.active_slabs.get(sid, 0.0), weight)
            if sid not in frame.active_nodes:
                frame.active_nodes.append(sid)
                newly_added += 1
            # Retain any existing sources (anchor_match etc.) — retrieval is
            # an additional signal. If already only base_set-sourced, replace
            # with retrieval so the graph filter picks it up.
            existing = frame.activation_sources.get(sid, [])
            if not existing or all(s.source_type == "base_set" for s in existing):
                frame.activation_sources[sid] = [source]
            else:
                # Append retrieval to the list, dedup by source_type
                if not any(s.source_type == "retrieval" for s in existing):
                    frame.activation_sources[sid] = existing + [source]

            # Cascade to linked anchors + bundles at 0.4
            slab = self.corpus.slabs.get(sid)
            if slab and slab.links:
                for anchor_id in slab.links.anchors or []:
                    if anchor_id in self.corpus.anchors:
                        frame.active_anchors[anchor_id] = max(
                            frame.active_anchors.get(anchor_id, 0.0), 0.4,
                        )
                        if anchor_id not in frame.active_nodes:
                            frame.active_nodes.append(anchor_id)
                            newly_added += 1
                for bundle_id in slab.links.bundles or []:
                    if bundle_id in self.corpus.bundles:
                        frame.active_bundles[bundle_id] = max(
                            frame.active_bundles.get(bundle_id, 0.0), 0.4,
                        )
                        if bundle_id not in frame.active_nodes:
                            frame.active_nodes.append(bundle_id)
                            newly_added += 1

        return newly_added

    def adjust_node_salience(
        self, session_id: str, node_id: str, delta: float
    ) -> Optional[dict]:
        """Heat or cool a node's salience. Returns {old, new} or None if node not active.

        Raises FrameNotFoundError if the session has no frame.
        """
        frame = self._frames.get(session_id)
        if frame is None:
            raise FrameNotFoundError(session_id)
        if node_id not in frame.active_nodes:
            return None
        old = frame.salience_now.get(node_id, 0.0)
        new_sal = max(0.0, min(1.0, old + delta))
        frame.salience_now[node_id] = new_sal
        frame.salience_smoothed[node_id] = new_sal  # immediate effect
        self.persist(session_id)
        return {"old": old, "new": new_sal}

    def dismiss_node(self, session_id: str, node_id: str) -> None:
        """Evict a node from the active frame, mark its registry entry dismissed.

        The registry entry is NOT deleted — the concept detector can
        re-activate it if the user discusses the concept organically again
        (see _detect_novel_concepts' redetection path).

        Raises FrameNotFoundError if the session has no frame.
        """
        frame = self._frames.get(session_id)
        if frame is None:
            raise FrameNotFoundError(session_id)
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
        _, info = self._find_tentative_by_id(session_id, node_id)
        if info is not None:
            info["dismissed"] = True
        self.persist(session_id)

    def reject_node(self, session_id: str, node_id: str) -> None:
        """Mark a tentative node as rejected — stays visible but filtered from commit.

        Does not raise if the session has no frame or the registry entry is
        missing; rejection is advisory metadata. (Previously this endpoint
        skipped persistence entirely, losing the flag on server restart —
        persisting here fixes that.)
        """
        _, info = self._find_tentative_by_id(session_id, node_id)
        if info is not None:
            info["rejected"] = True
        self.persist(session_id)

    def create_tentative_bundle(
        self,
        session_id: str,
        child_ids: list[str],
        label: Optional[str] = None,
    ) -> dict:
        """Group selected nodes into a tentative bundle.

        Creates one registry entry for the bundle, one PARENT_OF edge per
        child, sets parent_id on each child's registry entry, and absorbs
        children out of the active frame. Returns {"bundle_id", "label"}.

        Raises FrameNotFoundError if the session has no frame.
        Raises ValueError if fewer than 2 children are supplied.
        """
        import uuid

        frame = self._frames.get(session_id)
        if frame is None:
            raise FrameNotFoundError(session_id)
        if len(child_ids) < 2:
            raise ValueError("Bundle requires at least 2 nodes")

        registry = self._tentative_registry.setdefault(session_id, {})

        # Resolve human labels for children for description/label text.
        child_labels = []
        for nid in child_ids:
            name, info = self._find_tentative_by_id(session_id, nid)
            if name is not None:
                child_labels.append(name)
            elif nid in self.corpus.anchors:
                child_labels.append(self.corpus.anchors[nid].canonical_phrase)
            else:
                child_labels.append(nid)

        resolved_label = label or f"Bundle: {', '.join(child_labels[:3])}"
        bundle_id = f"tentative_bundle_{uuid.uuid4().hex[:8]}"

        registry[resolved_label] = {
            "id": bundle_id,
            "description": f"User-grouped: {', '.join(child_labels)}",
            "turns_seen": 1,
            "promoted": None,
            "parent_id": None,
            "children": list(child_ids),
        }

        # Add bundle to frame
        frame.active_nodes.append(bundle_id)
        frame.active_bundles[bundle_id] = 0.7
        frame.salience_now[bundle_id] = 0.7
        frame.salience_smoothed[bundle_id] = 0.7
        frame.structural_weight[bundle_id] = float(len(child_ids))
        frame.activation_sources[bundle_id] = [
            ActivationSource(source_type="user_bundling", source_ref=",".join(child_ids))
        ]

        # PARENT_OF edges bundle→child + set parent_id on children + absorb from frame
        edges = self._tentative_edges.setdefault(session_id, [])
        for child_id in child_ids:
            edges.append({
                "from": bundle_id,
                "to": child_id,
                "type": "PARENT_OF",
                "strength": 0.8,
            })
            _, child_info = self._find_tentative_by_id(session_id, child_id)
            if child_info is not None:
                child_info["parent_id"] = bundle_id
            if child_id in frame.active_nodes:
                frame.active_nodes.remove(child_id)
            frame.salience_now.pop(child_id, None)
            frame.salience_smoothed.pop(child_id, None)
            frame.structural_weight.pop(child_id, None)
            frame.activation_sources.pop(child_id, None)
            frame.active_concepts.pop(child_id, None)
            frame.active_anchors.pop(child_id, None)
            frame.active_bundles.pop(child_id, None)

        # Record in corpus-hit trajectory so the bundle appears in session review
        turn = frame.last_updated_turn or 1
        frame.corpus_hits[bundle_id] = frame.corpus_hits.get(bundle_id, 0) + 1
        frame.corpus_last_hit[bundle_id] = turn
        frame.corpus_hit_log.append([turn, [bundle_id] + list(child_ids)])

        self.persist(session_id)
        return {"bundle_id": bundle_id, "label": resolved_label}

    def promote_to_anchor(
        self, session_id: str, node_id: str
    ) -> Optional[dict]:
        """Promote a tentative bundle to a tentative anchor.

        Returns {"label", "children"} on success, or None if the node is not
        in the tentative registry.

        Raises FrameNotFoundError if the session has no frame.
        """
        frame = self._frames.get(session_id)
        if frame is None:
            raise FrameNotFoundError(session_id)
        name, info = self._find_tentative_by_id(session_id, node_id)
        if info is None:
            return None
        info["promoted"] = "anchor"
        children = info.get("children", [])
        self.persist(session_id)
        return {"label": name, "children": children}

    def inject_mined_proposals(
        self, session_id: str, chat_id: str, packets: list
    ) -> None:
        """Inject mined DraftPackets as tentative registry entries.

        Ensures a frame exists (base-set-free via ensure_empty_frame), creates
        one registry entry per packet, marks packets active in the frame, and
        persists. This is the mining-flow counterpart to _detect_novel_concepts
        (which is the chat-flow counterpart). Both end up writing to the
        tentative registry with the same entry shape.
        """
        if not packets:
            return
        frame = self.ensure_empty_frame(session_id, chat_id)
        registry = self._tentative_registry.setdefault(session_id, {})

        for pkt in packets:
            ntype = pkt.packet_type or "anchor"
            # Human-readable label extraction mirrors the original mining code.
            if pkt.anchor:
                label = pkt.anchor.get("canonical_phrase", "") or pkt.id
            elif pkt.slab:
                label = (
                    pkt.slab.get("title", "")
                    or pkt.slab.get("canonical_text", "")[:60]
                    or pkt.id
                )
            else:
                label = pkt.id

            registry[label] = {
                "id": pkt.id,
                "description": pkt.justification or "",
                "turns_seen": 1,
                "promoted": ntype if ntype in ("anchor", "slab", "bundle") else None,
                "parent_id": None,
                "children": [],
            }
            if pkt.id not in frame.active_nodes:
                frame.active_nodes.append(pkt.id)

        self.persist(session_id)

    def _seed_base_set(
        self, frame: FrameState, oli_mode: OLIMode,
        bound_collection_store: Optional[CorpusStore] = None,
    ) -> None:
        """Populate a new frame with the corpus base set.

        CONSTITUTIONAL and CANONICAL slabs (filtered by OLI mode and lifecycle)
        are loaded at weight 1.0. Their linked anchors and bundles are also
        activated at weight 0.5 (lower than match-triggered activation).

        If ``bound_collection_store`` is provided, a second seeding pass adds
        ALL ACTIVE slabs from that collection — including REFERENCE type —
        at an ambient weight (0.6). This makes a chat's bound collection
        visible to the model without requiring the user to invoke specific
        concepts each turn. Linked anchors/bundles from those slabs are
        seeded at 0.3 (below match-triggered, above evicted). The existing
        graph-visibility filter in the session endpoint keeps these nodes
        hidden on the canvas until conversation actually engages them, so
        the UI stays clean while the model has the context available.
        """
        # Only CONSTITUTIONAL + CANONICAL slabs are seeded into the frame
        # at session start. REFERENCE slabs live in cold corpus and come
        # into the frame on demand via retrieval (see
        # ``activate_retrieved_slabs``) or anchor-match cascade. This is
        # the "pointer architecture" — the frame reflects live working
        # memory, not the full corpus catalog.
        base_slabs = self.corpus.base_set_slabs(oli_mode, include_reference=False)
        for slab in base_slabs:
            frame.active_slabs[slab.id] = 1.0
            if slab.id not in frame.active_nodes:
                frame.active_nodes.append(slab.id)
            frame.activation_sources[slab.id] = [
                ActivationSource(source_type="base_set", source_ref=f"type={slab.type.value}")
            ]
            # Follow slab links → activate linked anchors and bundles.
            # Cascade strength comes from the reified edge if one exists
            # (weight * confidence * kernel), otherwise falls back to the
            # historic 0.5 seed value — matching current behaviour for
            # slab.links.* pairs that lack an explicit Edge record.
            for anchor_id in (slab.links.anchors if slab.links else []):
                if anchor_id in self.corpus.anchors:
                    strength = self._cascade_strength(slab.id, anchor_id, 0.5)
                    frame.active_anchors.setdefault(anchor_id, strength)
                    if anchor_id not in frame.active_nodes:
                        frame.active_nodes.append(anchor_id)
            for bundle_id in (slab.links.bundles if slab.links else []):
                if bundle_id in self.corpus.bundles:
                    strength = self._cascade_strength(slab.id, bundle_id, 0.5)
                    frame.active_bundles.setdefault(bundle_id, strength)
                    if bundle_id not in frame.active_nodes:
                        frame.active_nodes.append(bundle_id)

        # --- Bound-collection ambient seed ---
        # When a chat is bound to a specific collection (e.g. vindiesel5),
        # that collection's full slab set is loaded as ambient context so
        # the model can answer meta-questions ("tell me the narrative")
        # without requiring the user to trigger specific anchor matches.
        from ..models.enums import SlabLifecycleStatus
        bound_seeded = 0
        if bound_collection_store is not None:
            for slab in bound_collection_store.slabs.values():
                if slab.lifecycle_status != SlabLifecycleStatus.ACTIVE:
                    continue
                # OLI-mode gate — respect the slab's requires_oli_mode
                if slab.requires_oli_mode is not None:
                    if OLIMode(slab.requires_oli_mode) != oli_mode:
                        continue
                # Don't downgrade something already seeded at 1.0
                if slab.id in frame.active_slabs:
                    continue
                frame.active_slabs[slab.id] = 0.6
                if slab.id not in frame.active_nodes:
                    frame.active_nodes.append(slab.id)
                frame.activation_sources[slab.id] = [
                    ActivationSource(
                        source_type="base_set",
                        source_ref=f"bound_collection={bound_collection_store.collection_id}",
                    )
                ]
                for anchor_id in (slab.links.anchors if slab.links else []):
                    if anchor_id in self.corpus.anchors:
                        frame.active_anchors.setdefault(anchor_id, 0.3)
                        if anchor_id not in frame.active_nodes:
                            frame.active_nodes.append(anchor_id)
                for bundle_id in (slab.links.bundles if slab.links else []):
                    if bundle_id in self.corpus.bundles:
                        frame.active_bundles.setdefault(bundle_id, 0.3)
                        if bundle_id not in frame.active_nodes:
                            frame.active_nodes.append(bundle_id)
                bound_seeded += 1

        _event_log.log_frame_event(
            event="base_set_seeded",
            slabs=len(base_slabs),
            bound_collection_slabs=bound_seeded,
            oli_mode=oli_mode.value,
            total_active=len(frame.active_nodes),
        )

    def _check_invariant_activation(
        self, frame: FrameState,
        match_result: Optional[AnchorMatchResult],
        classification: Optional[object],
        turn: int,
    ) -> None:
        """§Phase 5 — Conditional activation rules for INVARIANT slabs.

        INVARIANT slabs don't load in the base set. They activate when:
          1. An anchor that links TO them fires (anchor match → slab.links.anchors)
          2. A slab's linked anchor is already active with weight >= 0.8
          3. Drift-based escalation: mismatch_score > 0.6 activates pushback invariants

        Once activated, an INVARIANT slab stays in the frame (subject to normal decay).
        """
        invariant_slabs = self.corpus.invariant_slabs()
        activated = []

        for slab in invariant_slabs:
            # Skip if already active
            if slab.id in frame.active_slabs:
                continue

            triggered = False
            trigger_reason = ""

            # Rule 1: Anchor match invokes chain — the anchor explicitly INVOKES this slab
            if match_result:
                for match in match_result.auto_activate:
                    anchor = self.corpus.anchors.get(match.anchor_id)
                    if anchor and slab.id in anchor.invokes:
                        triggered = True
                        trigger_reason = f"anchor_invokes:{anchor.id}"
                        break

            # Rule 2: Linked anchor is active with high weight
            if not triggered and slab.links:
                for anchor_id in slab.links.anchors:
                    if frame.active_anchors.get(anchor_id, 0) >= policy.frame.activation.invariant_linked_anchor_min:
                        triggered = True
                        trigger_reason = f"linked_anchor_active:{anchor_id}"
                        break

            # Rule 3: High mismatch triggers pushback-type invariant slabs
            if not triggered and frame.mismatch_score > policy.frame.activation.mismatch_escalation:
                # Only auto-load slabs that have "pushback" or "gauntlet" in their ID
                if "pushback" in slab.id.lower() or "gauntlet" in slab.id.lower():
                    triggered = True
                    trigger_reason = f"mismatch_escalation:{frame.mismatch_score:.2f}"

            if triggered:
                frame.active_slabs[slab.id] = 1.0
                if slab.id not in frame.active_nodes:
                    frame.active_nodes.append(slab.id)
                frame.activation_sources[slab.id] = [
                    ActivationSource(source_type="invariant_rule", source_ref=trigger_reason)
                ]
                # Narrow the bundle cascade for back-link activations.
                # Cross-pollution avoidance: when an INVARIANT slab activates
                # because ONE of its linked anchors fired (Rule 2), only
                # cascade to bundles that are ALSO in that specific anchor's
                # invokes list. Bundles belonging to the slab's OTHER linked
                # anchors do not ride along. Rule 1 (anchor_invokes) and Rule 3
                # (mismatch_escalation) keep the full cascade because the
                # curator's intent is explicit in those cases.
                candidate_bundles = list(slab.links.bundles) if slab.links else []
                if trigger_reason.startswith("linked_anchor_active:"):
                    trigger_anchor_id = trigger_reason.split(":", 1)[1]
                    trigger_anchor = self.corpus.anchors.get(trigger_anchor_id)
                    if trigger_anchor:
                        trigger_invokes = set(trigger_anchor.invokes)
                        candidate_bundles = [
                            b for b in candidate_bundles if b in trigger_invokes
                        ]
                for bundle_id in candidate_bundles:
                    if bundle_id in self.corpus.bundles:
                        # Fallback 0.8 preserves the historic "invariant
                        # activation cascade is more assertive than base-set
                        # seeding" tuning for edge-less pairs.
                        strength = self._cascade_strength(slab.id, bundle_id, 0.8)
                        frame.active_bundles.setdefault(bundle_id, strength)
                        if bundle_id not in frame.active_nodes:
                            frame.active_nodes.append(bundle_id)
                activated.append(slab.id)

        if activated:
            _event_log.log_frame_event(
                event="invariant_activated",
                slabs=activated,
            )

    def apply_truth_pressure(
        self, session_id: str, node_ids: list[str], delta: float = 0.15
    ) -> None:
        """Increment truth pressure on specific nodes.

        Called when:
          - The gauntlet fires and identifies anchor_conflicts
          - A user explicitly challenges or questions a node
          - A verification check contradicts a node's claims

        Pressure is clamped to [0.0, 1.0] and decays at 0.95 per turn.
        """
        frame = self._frames.get(session_id)
        if not frame:
            return
        for nid in node_ids:
            old = frame.truth_pressure.get(nid, 0.0)
            frame.truth_pressure[nid] = min(1.0, old + delta)

    def get_high_pressure_nodes(
        self, session_id: str, threshold: float = 0.5
    ) -> list[tuple[str, float]]:
        """Return nodes with truth pressure above threshold.

        Returns list of (node_id, pressure) tuples sorted by pressure descending.
        """
        frame = self._frames.get(session_id)
        if not frame:
            return []
        high = [
            (nid, p) for nid, p in frame.truth_pressure.items()
            if p >= threshold
        ]
        return sorted(high, key=lambda x: x[1], reverse=True)

    def restore(self, session_id: str, frame: FrameState,
                registry: dict = None, edges: list = None) -> None:
        """Restore a previously persisted FrameState + tentative state.

        If registry/edges are provided (from session_store.load_registry),
        uses them directly. Otherwise falls back to rebuilding from active_nodes.
        """
        self._frames[session_id] = frame

        if registry is not None:
            # Use persisted registry — preserves parent/child, labels, promoted state
            self._tentative_registry[session_id] = registry
        else:
            # Fallback: rebuild from active_nodes (loses hierarchy info)
            if session_id not in self._tentative_registry:
                self._tentative_registry[session_id] = {}
            reg = self._tentative_registry[session_id]
            for node_id in frame.active_nodes:
                if node_id.startswith("tentative_") and node_id not in [v.get("id") for v in reg.values()]:
                    label = node_id.replace("tentative_concept_", "").replace("tentative_anchor_", "").replace("tentative_bundle_", "").replace("tentative_slab_", "").replace("_", " ")
                    reg[label] = {
                        "id": node_id, "description": "", "turns_seen": 1,
                        "promoted": "anchor" if "anchor" in node_id else ("bundle" if "bundle" in node_id else None),
                        "parent_id": None, "children": [],
                    }

        if edges is not None:
            self._tentative_edges[session_id] = edges
        elif session_id not in self._tentative_edges:
            self._tentative_edges[session_id] = []

    async def update_turn(
        self,
        session_id: str,
        turn: int,
        user_text: str,
        match_result: Optional[AnchorMatchResult],
        classification: Optional[object] = None,
    ) -> FrameState:
        """Full per-turn FrameState update cycle.

        1. Process anchor matches → activate nodes
        1b. Novel concept detection → inject tentative nodes
        2. Compute structural weights
        3. Estimate salience (model proposes)
        4. Smooth salience (code decides)
        5. Detect conflicts
        6. Compute mismatch score
        7. Evict if over cap
        """
        frame = self._frames.get(session_id)
        if not frame:
            return FrameState(chat_id="", session_id=session_id)

        # --- Step 0: Apply per-turn decay to existing weights ---
        decay = frame.decay.per_turn
        floor = frame.decay.floor
        for d in (frame.active_anchors, frame.active_bundles, frame.active_slabs, frame.active_concepts):
            for k in list(d.keys()):
                d[k] = max(floor, d[k] * decay)

        # --- Step 0b: Decay truth pressure per turn ---
        # Pressure decays slower than salience (0.95 vs 0.85) because
        # epistemic tension is stickier than topic relevance.
        for nid in list(frame.truth_pressure.keys()):
            frame.truth_pressure[nid] *= 0.95
            if frame.truth_pressure[nid] < 0.05:
                del frame.truth_pressure[nid]

        # --- Step 1: Activate from Tier 1 anchor matches ---
        # Collect per-turn hits for trajectory replay
        turn_hits: list[str] = []
        if match_result:
            for match in match_result.auto_activate:
                anchor = self.corpus.anchors.get(match.anchor_id)
                if not anchor:
                    continue

                # Activate the anchor itself
                if anchor.id not in frame.active_nodes:
                    frame.active_nodes.append(anchor.id)
                frame.active_anchors[anchor.id] = 1.0
                frame.activation_sources[anchor.id] = [
                    ActivationSource(
                        source_type="anchor_match",
                        source_ref=match.anchor_id,
                    )
                ]
                # Track corpus access pattern — anchor hit
                frame.corpus_hits[anchor.id] = frame.corpus_hits.get(anchor.id, 0) + 1
                frame.corpus_last_hit[anchor.id] = turn
                if anchor.id not in turn_hits:
                    turn_hits.append(anchor.id)

                # Follow invokes chains: activate referenced slabs/bundles.
                # Cascade strength = edge.weight * edge.confidence * kernel,
                # with fallback 1.0 preserving the hardcoded behaviour for
                # any anchor.invokes targets that lack a reified Edge.
                # Use max() against any existing activation so a stronger
                # prior source this turn isn't silently clobbered.
                for target_id in anchor.invokes:
                    if target_id in self.corpus.all_ids() and target_id not in frame.active_nodes:
                        frame.active_nodes.append(target_id)
                    strength = self._cascade_strength(anchor.id, target_id, 1.0)
                    # Populate typed dict
                    if target_id in self.corpus.slabs:
                        frame.active_slabs[target_id] = max(
                            frame.active_slabs.get(target_id, 0.0), strength
                        )
                    elif target_id in self.corpus.bundles:
                        frame.active_bundles[target_id] = max(
                            frame.active_bundles.get(target_id, 0.0), strength
                        )
                    frame.activation_sources[target_id] = [
                        ActivationSource(
                            source_type="anchor_invocation",
                            source_ref=anchor.id,
                        )
                    ]
                    # Track corpus access pattern — invoked node hit
                    frame.corpus_hits[target_id] = frame.corpus_hits.get(target_id, 0) + 1
                    frame.corpus_last_hit[target_id] = turn
                    if target_id not in turn_hits:
                        turn_hits.append(target_id)

        # Also track hits from wrapped spans (anchor references that didn't
        # auto-activate — e.g. performed_imitation, semantic_depth, etc.)
        if match_result:
            for ws in getattr(match_result, 'wrapped_spans', []):
                for hit in getattr(ws, 'anchor_hits', []):
                    aid = hit.anchor_id if hasattr(hit, 'anchor_id') else hit.get('anchor_id', '')
                    if aid and aid in self.corpus.anchors and aid not in turn_hits:
                        frame.corpus_hits[aid] = frame.corpus_hits.get(aid, 0) + 1
                        frame.corpus_last_hit[aid] = turn
                        turn_hits.append(aid)

            # Track slab/bundle name references (Pass C corpus_refs)
            for ref in getattr(match_result, 'corpus_refs', []):
                nid = ref.node_id
                if nid not in turn_hits:
                    frame.corpus_hits[nid] = frame.corpus_hits.get(nid, 0) + 1
                    frame.corpus_last_hit[nid] = turn
                    turn_hits.append(nid)
                    # Activate the referenced node
                    if nid not in frame.active_nodes:
                        frame.active_nodes.append(nid)
                    if ref.node_type == "slab":
                        frame.active_slabs[nid] = 1.0
                    elif ref.node_type == "bundle":
                        frame.active_bundles[nid] = 1.0
                    frame.activation_sources[nid] = [
                        ActivationSource(
                            source_type="corpus_ref_match",
                            source_ref=ref.matched_phrase,
                        )
                    ]

        # Append this turn's hit batch for trajectory replay
        if turn_hits:
            frame.corpus_hit_log.append([turn, turn_hits])

        # --- Step 1a: Conditional activation rules for INVARIANT slabs ---
        self._check_invariant_activation(frame, match_result, classification, turn)

        # --- Step 1b: Novel concept detection → tentative nodes ---
        # Concept detection is a sensor — cheap, near-unconditional. The
        # CONCEPT_DETECT_PROMPT itself has its own "return [] if trivial" guard,
        # so we only hard-skip the one case where we actively do NOT want to
        # mine concepts: pure affect_release (venting / emotional processing).
        # Everything else — including short "just mentioned X" turns — goes
        # through. Anchor hits don't suppress concept detection either; a user
        # can match an anchor AND introduce a new concept in the same turn.
        skip_concept_detection = False
        if classification and hasattr(classification, 'function'):
            if classification.function == MessageFunction.AFFECT_RELEASE:
                skip_concept_detection = True
        # Also skip for trivially short messages (greetings, acknowledgements)
        if len(user_text.strip()) < 8:
            skip_concept_detection = True
        if not skip_concept_detection:
            new_tentatives = await self._detect_novel_concepts(session_id, user_text, frame)
            if new_tentatives:
                for t in new_tentatives:
                    _event_log.log_frame_event(
                        session_id=session_id, turn=turn,
                        event="tentative_injected", concept=t,
                    )

        # If still no active nodes after tentative injection, return early
        if not frame.active_nodes:
            frame.last_updated_turn = turn
            return frame

        # --- Step 2: Structural weights ---
        for node_id in frame.active_nodes:
            rev_deps = self.corpus.get_reverse_deps(node_id)
            edge_count = sum(
                1 for e in self.corpus.edges.values()
                if e.from_node == node_id or e.to_node == node_id
            )
            frame.structural_weight[node_id] = float(len(rev_deps) + edge_count)

        # --- Step 3: Salience estimation (model proposes) ---
        # We MERGE rather than replace: on low-content turns (e.g. "re-emit
        # your last message", "go on", "yes") the model often returns an
        # empty or partial dict, and a full replacement would silently zero
        # out every previously-scored node. Missing nodes instead inherit
        # their prior smoothed value lightly decayed — the normal EWA step
        # below still applies, so real drops still propagate, they just
        # don't collapse in one turn.
        node_descriptions = self._build_node_descriptions(frame.active_nodes)
        raw_salience = await self._estimate_salience(user_text, node_descriptions)
        new_now: dict[str, float] = {}
        for node_id in frame.active_nodes:
            if node_id in raw_salience:
                new_now[node_id] = raw_salience[node_id]
            else:
                # Inherit prior smoothed value lightly decayed (not zero)
                prior = frame.salience_smoothed.get(
                    node_id, frame.salience_now.get(node_id, 0.0)
                )
                new_now[node_id] = max(0.0, prior * 0.9)
        frame.salience_now = new_now

        # --- Step 4: Smooth salience (code decides) ---
        for node_id in frame.active_nodes:
            now = frame.salience_now.get(node_id, 0.0)
            prev = frame.salience_smoothed.get(node_id, now)
            frame.salience_smoothed[node_id] = (
                SALIENCE_ALPHA * now + (1 - SALIENCE_ALPHA) * prev
            )

        # --- Step 4b: SUPPORTS reinforcement pass ---
        # SUPPORTS edges don't auto-activate their target (that's INVOKES's
        # job). Instead, when both endpoints are already in the frame, the
        # source's salience reinforces the target's — modelling "X lends
        # credence / relevance to Y". Subtle by design: heavily damped so
        # the model-proposed salience stays dominant, and iterated over a
        # snapshot so two SUPPORTS edges firing in the same turn don't
        # compound into runaway feedback.
        active_set = set(frame.active_nodes)
        smoothed_snapshot = dict(frame.salience_smoothed)
        for edge in self.corpus.edges.values():
            if edge.type != EdgeType.SUPPORTS:
                continue
            if edge.from_node not in active_set or edge.to_node not in active_set:
                continue
            source_sal = smoothed_snapshot.get(edge.from_node, 0.0)
            if source_sal <= 0.0:
                continue
            boost = (
                source_sal
                * edge.weight
                * edge.confidence
                * CASCADE_KERNEL[EdgeType.SUPPORTS]
                * SUPPORTS_REINFORCE_DAMP
            )
            current = frame.salience_smoothed.get(edge.to_node, 0.0)
            frame.salience_smoothed[edge.to_node] = min(1.0, current + boost)

        # --- Step 5: Detect conflicts ---
        frame.conflicts = []
        for edge in self.corpus.edges.values():
            if edge.type != EdgeType.CONFLICTS:
                continue
            if edge.from_node in frame.active_nodes and edge.to_node in frame.active_nodes:
                sal_a = frame.salience_now.get(edge.from_node, 0)
                sal_b = frame.salience_now.get(edge.to_node, 0)
                frame.conflicts.append(ConflictPair(
                    node_a=edge.from_node,
                    node_b=edge.to_node,
                    tension=edge.tension or 0.0,
                    semantic_change=abs(sal_a - sal_b),
                ))

        # --- Step 6: Mismatch score ---
        if frame.active_nodes:
            diffs = [
                abs(frame.salience_now.get(n, 0) - frame.salience_smoothed.get(n, 0))
                for n in frame.active_nodes
            ]
            frame.mismatch_score = sum(diffs) / len(diffs)
        else:
            frame.mismatch_score = 0.0

        # --- Step 7: Evict if over cap ---
        if len(frame.active_nodes) > MAX_ACTIVE_NODES:
            # Sort by smoothed salience ascending, keep top MAX
            ranked = sorted(
                frame.active_nodes,
                key=lambda n: frame.salience_smoothed.get(n, 0),
            )
            to_remove = ranked[:len(frame.active_nodes) - MAX_ACTIVE_NODES]
            for node_id in to_remove:
                frame.active_nodes.remove(node_id)
                frame.salience_now.pop(node_id, None)
                frame.salience_smoothed.pop(node_id, None)
                frame.structural_weight.pop(node_id, None)
                frame.activation_sources.pop(node_id, None)

        frame.last_updated_turn = turn

        _event_log.log_frame_event(
            session_id=session_id,
            turn=turn,
            active_count=len(frame.active_nodes),
            mismatch=frame.mismatch_score,
            conflicts=len(frame.conflicts),
        )

        return frame

    def _build_node_descriptions(self, node_ids: list[str]) -> str:
        """Build a compact description string for salience estimation."""
        lines = []
        for nid in node_ids[:MAX_ACTIVE_NODES]:
            desc = nid
            if nid in self.corpus.anchors:
                desc = f"{nid} ({self.corpus.anchors[nid].canonical_phrase})"
            elif nid in self.corpus.slabs:
                text = self.corpus.slabs[nid].canonical_text[:80]
                desc = f"{nid} ({text})"
            elif nid in self.corpus.bundles:
                intent = "; ".join(self.corpus.bundles[nid].payload.intent[:2])
                desc = f"{nid} ({intent})"
            lines.append(desc)
        return "\n".join(lines)

    async def _estimate_salience(
        self, user_text: str, node_descriptions: str
    ) -> dict[str, float]:
        """Model proposes salience. Code stores. Returns {node_id: float}."""
        prompt = (
            SALIENCE_PROMPT.replace("$NODES", node_descriptions)
            + json.dumps(user_text)
        )
        try:
            result = await ollama.structured_extract(prompt)
            # Expect dict[str, float]
            if isinstance(result, dict):
                return {k: max(0.0, min(1.0, float(v))) for k, v in result.items()}
            # Handle array format fallback
            if isinstance(result, list):
                return {
                    item["node_id"]: max(0.0, min(1.0, float(item["salience"])))
                    for item in result
                    if "node_id" in item and "salience" in item
                }
        except Exception:
            pass
        return {}

    # --- Tentative node tracking ---
    # (registry initialized in __init__ as instance state)

    async def _detect_novel_concepts(
        self, session_id: str, user_text: str, frame: FrameState
    ) -> list[str]:
        """Detect novel concepts not in corpus and inject tentative nodes.

        Returns list of concept names that were injected.
        """
        import numpy as np

        # Extract concepts from user text
        prompt = CONCEPT_DETECT_PROMPT + json.dumps(user_text)
        try:
            result = await ollama.structured_extract(prompt)
            if not isinstance(result, list):
                return []
        except Exception:
            return []

        if not result:
            return []

        if session_id not in self._tentative_registry:
            self._tentative_registry[session_id] = {}
        registry = self._tentative_registry[session_id]

        injected = []

        for item in result[:2]:
            if not isinstance(item, dict):
                continue
            concept = item.get("concept", "").strip()
            description = item.get("description", "").strip()
            if not concept or len(concept) < 2:
                continue

            # Check against existing corpus (cosine similarity)
            from . import embeddings
            is_known = False
            concept_hits = []
            for anchor in self.corpus.anchors.values():
                sim = await embeddings.cosine_similarity(concept, anchor.canonical_phrase)
                if sim > 0.7:
                    is_known = True
                    # Indirect hit — user discussing something close to this anchor
                    frame.corpus_hits[anchor.id] = frame.corpus_hits.get(anchor.id, 0) + 1
                    frame.corpus_last_hit[anchor.id] = frame.last_updated_turn + 1
                    concept_hits.append(anchor.id)
                    # Also hit invoked bundles
                    for inv in anchor.invokes:
                        frame.corpus_hits[inv] = frame.corpus_hits.get(inv, 0) + 1
                        frame.corpus_last_hit[inv] = frame.last_updated_turn + 1
                        concept_hits.append(inv)
                    break
            # Record concept-detection hits in trajectory
            if concept_hits:
                frame.corpus_hit_log.append([frame.last_updated_turn + 1, concept_hits])
            if not is_known:
                for slab in self.corpus.slabs.values():
                    sim = await embeddings.cosine_similarity(concept, slab.canonical_text[:100])
                    if sim > 0.7:
                        is_known = True
                        break
            if is_known:
                continue

            # Check against existing tentative nodes — consolidate if similar
            consolidated = False
            for existing_name, existing in registry.items():
                sim = await embeddings.cosine_similarity(concept, existing_name)
                if sim > TENTATIVE_CONSOLIDATION_THRESHOLD:
                    # Same concept, reinforce it — bump salience
                    existing["turns_seen"] += 1
                    node_id = existing["id"]

                    # Re-add dismissed nodes if they come up organically again
                    if existing.get("dismissed") and node_id not in frame.active_nodes:
                        existing["dismissed"] = False
                        frame.active_nodes.append(node_id)
                        frame.active_concepts[node_id] = 1.0
                        frame.salience_now[node_id] = 0.6
                        frame.salience_smoothed[node_id] = 0.6
                        frame.activation_sources[node_id] = [
                            ActivationSource(source_type="organic_redetection", source_ref=f"turn_{frame.last_updated_turn}")
                        ]
                        _event_log.log_frame_event(
                            session_id=session_id, event="dismissed_node_redetected",
                            concept=existing_name, node_id=node_id,
                        )

                    if node_id in frame.active_nodes:
                        old_sal = frame.salience_now.get(node_id, 0.5)
                        frame.salience_now[node_id] = min(1.0, old_sal + 0.15)

                    # Promote concept → anchor at threshold
                    if (existing["turns_seen"] >= CONCEPT_TO_ANCHOR_THRESHOLD
                            and existing.get("promoted") != "anchor"):
                        existing["promoted"] = "anchor"
                        _event_log.log_proposal_event(
                            session_id=session_id,
                            event="concept_promoted_to_anchor",
                            concept=existing_name,
                            turns_seen=existing["turns_seen"],
                            node_id=node_id,
                        )

                    consolidated = True
                    injected.append(existing_name)
                    break

            if consolidated:
                continue

            # Genuinely novel — create tentative concept node
            slug = concept.lower().replace(' ', '_')[:30]
            node_id = f"tentative_concept_{slug}"

            # Avoid duplicate IDs
            if node_id in frame.active_nodes:
                old_sal = frame.salience_now.get(node_id, 0.5)
                frame.salience_now[node_id] = min(1.0, old_sal + 0.15)
                injected.append(concept)
                continue

            registry[concept] = {
                "id": node_id,
                "description": description,
                "turns_seen": 1,
                "promoted": None,
                "parent_id": None,
                "children": [],
            }

            frame.active_nodes.append(node_id)
            frame.active_concepts[node_id] = 1.0
            frame.activation_sources[node_id] = [
                ActivationSource(
                    source_type="concept_detection",
                    source_ref=f"turn_{frame.last_updated_turn + 1}",
                )
            ]
            frame.salience_now[node_id] = 0.6
            frame.salience_smoothed[node_id] = 0.6
            frame.structural_weight[node_id] = 0.0

            # Hierarchy detection — is this a child of an existing concept?
            parent_name, parent_id = await self._detect_concept_hierarchy(
                session_id, concept, registry
            )
            if parent_name and parent_id:
                registry[concept]["parent_id"] = parent_id
                if parent_name in registry:
                    if node_id not in registry[parent_name].get("children", []):
                        registry[parent_name].setdefault("children", []).append(node_id)
                    # Check if parent has enough children to suggest bundling
                    child_count = len(registry[parent_name].get("children", []))
                    if child_count >= BUNDLE_SUGGESTION_THRESHOLD:
                        _event_log.log_proposal_event(
                            session_id=session_id,
                            event="bundle_suggested",
                            parent=parent_name,
                            parent_id=parent_id,
                            children=registry[parent_name]["children"],
                            child_count=child_count,
                        )
                # Create tentative PARENT_OF edge
                if session_id not in self._tentative_edges:
                    self._tentative_edges[session_id] = []
                self._tentative_edges[session_id].append({
                    "from": parent_id,
                    "to": node_id,
                    "type": "PARENT_OF",
                    "strength": 0.7,
                })
                _event_log.log_frame_event(
                    session_id=session_id,
                    event="hierarchy_detected",
                    child=concept, parent=parent_name,
                )

            injected.append(concept)

        return injected

    async def _detect_concept_hierarchy(
        self, session_id: str, new_concept: str, registry: dict
    ) -> tuple:
        """Check if new_concept is a child of an existing concept.

        Returns (parent_name, parent_id) or (None, None).
        Heuristic: cosine > 0.75 AND the new concept is more specific
        (the existing concept's name appears as a substring, or
        the new concept is longer/more specific).
        """
        from . import embeddings

        best_sim = 0.0
        best_parent = (None, None)

        # Check tentative registry
        for existing_name, existing_info in registry.items():
            if existing_name == new_concept:
                continue
            if existing_info.get("id", "").startswith("tentative_concept_") or existing_info.get("id", "").startswith("tentative_bundle_"):
                sim = await embeddings.cosine_similarity(new_concept, existing_name)
                # Parent should be the more general concept
                is_more_specific = (
                    len(new_concept) > len(existing_name)
                    or existing_name.lower() in new_concept.lower()
                )
                if sim > 0.75 and is_more_specific and sim > best_sim:
                    best_sim = sim
                    best_parent = (existing_name, existing_info["id"])

        # Check corpus anchors
        for anchor in self.corpus.anchors.values():
            sim = await embeddings.cosine_similarity(new_concept, anchor.canonical_phrase)
            is_more_specific = (
                len(new_concept) > len(anchor.canonical_phrase)
                or anchor.canonical_phrase.lower() in new_concept.lower()
            )
            if sim > 0.75 and is_more_specific and sim > best_sim:
                best_sim = sim
                best_parent = (anchor.canonical_phrase, anchor.id)

        return best_parent
