"""Shared service instances and cross-cutting helpers for API sub-routers.

All sub-routers import their service dependencies from here. Module-level
singletons are wired once at import time; ``_rebind_corpus()`` updates
their ``.corpus`` attributes when the active collection set changes.

**Important:** ``corpus`` is reassigned by ``_rebind_corpus()``. Sub-routers
that need the live corpus reference MUST access it as ``deps.corpus`` (via
``from . import deps``), NOT as ``from .deps import corpus`` — the latter
captures the value at import time and won't see reassignments.
"""
from __future__ import annotations
import logging
from typing import Optional

from ..models.schemas import ChatMessage, MessageClassification, DriftEstimate
from ..models.enums import ChatStatus, OLIMode
from ..services.chat_store import ChatStore
from ..services.corpus import CorpusStore, CorpusRegistry
from ..services.pipeline import process_turn
from ..services.event_log import EventLog
from ..services.session_store import SessionStore
from ..services.anchor_matcher import AnchorMatcher
from ..services.frame_manager import FrameManager, FrameNotFoundError
from ..services.draft_manager import DraftManager
from ..services.slab_matcher import SlabMatcher
from ..services import ollama

logger = logging.getLogger(__name__)

# --- Service instances ---
chat_store = ChatStore()
registry = CorpusRegistry()
corpus: CorpusStore = CorpusStore()  # Replaced by registry.merged after load
event_log = EventLog()
session_store = SessionStore()
frame_manager = FrameManager(corpus, session_store=session_store)
anchor_matcher = AnchorMatcher(corpus)
slab_matcher = SlabMatcher(corpus)
draft_manager = DraftManager(corpus, session_store)

# Per-collection community trees + labels, persisted under
# app/state/communities/ so cluster IDs are stable across reboots
# despite Leiden's randomization. See graph_communities.CommunityStore.
from ..services.graph_communities import CommunityStore
community_store = CommunityStore()

from ..services.drift_monitor import DriftMonitor
drift_monitor = DriftMonitor()

from ..services.gauntlet import GauntletEngine
gauntlet_engine = GauntletEngine(corpus)

# LifecycleService coordinates atomic transitions across the five
# draft/tentative state locations (library, drafts, registry, frame,
# proposed edges). Route handlers route user-facing transitions through
# this service instead of calling individual verbs directly.
from ..services.lifecycle import LifecycleService
lifecycle = LifecycleService(
    corpus=corpus,
    registry=registry,
    session_store=session_store,
    frame_manager=frame_manager,
    draft_manager=draft_manager,
    chat_store=chat_store,
    event_log=event_log,
)


# --- Cross-cutting helpers ---

def rebind_corpus() -> None:
    """Rebind all services to the registry's merged corpus view.

    Called after collection activation changes so services see the
    updated anchor/slab/bundle set without a full restart.

    Note on caches: the anchor and slab matchers hold embeddings keyed by
    id. After a rebind, ids that are no longer in the corpus become dead
    entries (harmless — they just never get matched), and newly added ids
    aren't embedded until the next warm_cache() call. For correctness in
    the common case this is fine; for freshness the collection-management
    endpoints should schedule a background warm after activation changes.
    """
    global corpus
    corpus = registry.merged
    frame_manager.corpus = corpus
    anchor_matcher.corpus = corpus
    slab_matcher.corpus = corpus
    draft_manager.corpus = corpus
    draft_manager._registry = registry
    gauntlet_engine.corpus = corpus
    lifecycle.corpus = corpus

    # Community cache: collection activation changes can affect the
    # merged-view cluster tree. Invalidate the in-memory cache so next
    # GET /corpus/communities?scope=__merged__ recomputes against the
    # new active set. Per-collection cached trees are unaffected —
    # deactivating a collection doesn't change its own clustering.
    try:
        community_store.invalidate("__merged__")
    except Exception:
        pass

    # /corpus/full response cache: do NOT invalidate from this path.
    # rebind_corpus() fires per edge commit during bulk promote (via
    # lifecycle._resolve_accepted_edges_for_node), so invalidating
    # here defeats the cache — every promote triggers a fresh 8-12s
    # full-corpus re-embed for graph positions. The cache's 60s TTL
    # is sufficient: positions derive from canonical_text which is
    # immutable, so the only "staleness" is a new anchor not appearing
    # in the graph view for up to 60s. Acceptable for any bulk run.
    # Collection-activation paths invalidate explicitly via
    # invalidate_corpus_full_cache() in corpus_routes.py instead.


def save_session_state(sid: str) -> None:
    """Save frame + tentative registry + edges together.

    Thin delegate to frame_manager.persist(). Kept as a standalone helper
    because the send_message and related flows call it at points where
    state has been mutated by update_turn (not via a lifecycle verb) and
    still need explicit flushing.
    """
    frame_manager.persist(sid)
