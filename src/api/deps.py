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
draft_manager = DraftManager(corpus, session_store)

from ..services.drift_monitor import DriftMonitor
drift_monitor = DriftMonitor()

from ..services.gauntlet import GauntletEngine
gauntlet_engine = GauntletEngine(corpus)


# --- Cross-cutting helpers ---

def rebind_corpus() -> None:
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


def save_session_state(sid: str) -> None:
    """Save frame + tentative registry + edges together.

    Thin delegate to frame_manager.persist(). Kept as a standalone helper
    because the send_message and related flows call it at points where
    state has been mutated by update_turn (not via a lifecycle verb) and
    still need explicit flushing.
    """
    frame_manager.persist(sid)
