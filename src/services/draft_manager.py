"""Draft manager — tentative anchor/slab proposal system.

Model proposes load-bearing concepts; user reviews. Nothing touches
the authoritative corpus without explicit human sign-off.

Conservative limits per session:
  - Max 3 slabs from periodic sweep
  - Max 5 anchors from periodic sweep
  - Bundles only as minimum coherence set for a promoted slab
  - DraftStack cap: 10 total
  - Periodic sweep cadence: every 8 turns
  - Explicit user requests are uncapped
"""
from __future__ import annotations
import json
import logging
import uuid
from pathlib import Path
from typing import Optional
import yaml

logger = logging.getLogger(__name__)

from ..models.schemas import (
    DraftPacket, DraftStack, Anchor, Slab, AnchorMeta, AnchorMatchPolicy,
    ProvenanceRef, Edge,
)
from ..models.enums import DraftStatus, ClaimTag, DriftSeverity, EdgeType
from ..prompts.proposals import (
    PROPOSAL_EXTRACTION_PROMPT,
    PROPOSAL_EXPLICIT_PROMPT,
    RELATIONSHIP_MINING_PROMPT,
)
from .corpus import CorpusStore
from .session_store import SessionStore
from .verification_router import VerificationRouter
from .label_resolver import resolve_label as _shared_resolve_label
from . import ollama
from .event_log import EventLog

DRAFT_STACK_CAP = 10
MAX_PERIODIC_SLABS = 3
MAX_PERIODIC_ANCHORS = 5
SWEEP_CADENCE = 1  # every turn — aggressive extraction for interactive use
DEDUP_THRESHOLD = 0.75  # lower threshold catches more near-duplicates

# Fold "fancy" unicode punctuation the extraction model emits (smart quotes,
# non-breaking / en / em dashes, ellipsis, nbsp, prime marks) down to plain
# ASCII so proposal text is clean for downstream consumers. Only punctuation
# LOOKALIKES are touched — accented letters, symbols, and non-Latin scripts
# are left intact (str.translate only rewrites the mapped code points).
_PUNCT_FOLD = str.maketrans({
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-",
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "′": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
    "…": "...", " ": " ", " ": " ", " ": " ",
})
# Arrows — models love these in edge justifications ("X → Y"), and a bare →
# in a diagnostic print() crashes cp1252 stdout (see app.py stdout fix).
_PUNCT_FOLD.update(str.maketrans({
    "→": "->", "←": "<-", "↔": "<->", "↦": "->",
    "⇒": "=>", "⇐": "<=", "⇔": "<=>",
}))


def _normalize_punct(obj):
    """Recursively ASCII-fold fancy unicode punctuation in every string."""
    if isinstance(obj, str):
        return obj.translate(_PUNCT_FOLD)
    if isinstance(obj, list):
        return [_normalize_punct(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _normalize_punct(v) for k, v in obj.items()}
    return obj


_event_log = EventLog()


def _bundle_label(bundle) -> str:
    """Human-readable label for a bundle, synthesized from its intent list.

    Bundles have no dedicated title field — the UI and we both produce
    a label by joining the first two entries of ``payload.intent``. Used
    as the canonical handle when an LLM miner references a bundle as an
    edge endpoint. Falls back to ``bundle.id`` if intent is missing.
    """
    payload = getattr(bundle, "payload", None)
    intents = getattr(payload, "intent", None) if payload else None
    if not intents:
        return getattr(bundle, "id", "")
    return "; ".join(intents[:2])


# Fuzzy label resolution lives in src/services/label_resolver.py — kept
# here as a thin alias so existing call sites (`_fuzzy_resolve_label`)
# don't break. Behaviour and tuning unchanged.
_fuzzy_resolve_label = _shared_resolve_label


class DraftManager:

    def __init__(self, corpus: CorpusStore, session_store: SessionStore):
        self.corpus = corpus
        self.session_store = session_store
        self.verifier = VerificationRouter()
        self._registry = None  # Set by routes.py after registry init
        # Track periodic proposal counts per session
        self._periodic_counts: dict[str, dict[str, int]] = {}
        # Dedup embedding cache (Action 2): corpus object id -> L2-normalized
        # embedding of its lowercased canonical text. Lazy + self-healing —
        # _dedup_check batch-embeds only newly-present ids and prunes deleted
        # ones, so it stays correct across promotions/deletes without hooking
        # every mutation site. Convention is LOWERCASED (matches the prior
        # cosine_similarity path and dodges nomic-embed's title-case collapse
        # defect). Persists for the process lifetime.
        self._dedup_vec_cache: dict = {}

    def _get_counts(self, session_id: str) -> dict[str, int]:
        if session_id not in self._periodic_counts:
            self._periodic_counts[session_id] = {"slab": 0, "anchor": 0}
        return self._periodic_counts[session_id]

    def should_sweep(self, turn: int) -> bool:
        """Check if periodic sweep should fire this turn."""
        return turn > 0 and turn % SWEEP_CADENCE == 0

    async def _semantic_catalogs(self, query: str, k: int = 20) -> tuple[str, str]:
        """Return (anchor_phrases, slab_titles) for the extraction prompt,
        scoped to the top-K EXISTING nodes most related to ``query``.

        Reuses the retrieval cosine wiring (AnchorMatcher.top_k_anchors +
        SlabMatcher.top_k_for) so the model gets a bounded, RELEVANT
        comparison set for soft dedup and edge-partner discovery (esp.
        TENSIONS/CONFLICTS, which need real corpus comparison) instead of a
        flat dump that truncates unpredictably at corpus scale. Falls back to
        a CAPPED explicit list on cold caches — never the full dump.
        """
        existing_anchors = ""
        existing_slabs = ""
        try:
            from ..api import deps as _deps
            am = getattr(_deps, "anchor_matcher", None)
            sm = getattr(_deps, "slab_matcher", None)
            if am is not None:
                a_hits = await am.top_k_anchors(query, max_k=k)
                existing_anchors = ", ".join(
                    a.canonical_phrase for aid, _s in a_hits
                    if (a := self.corpus.anchors.get(aid))
                )
            if sm is not None and sm.has_cache():
                s_hits = await sm.top_k_for(query, max_k=k, threshold=0.4)
                existing_slabs = ", ".join(
                    s.title for sid, _s in s_hits
                    if (s := self.corpus.slabs.get(sid)) and getattr(s, "title", "")
                )
        except Exception as exc:
            print(f"[DRAFT] semantic catalog failed ({exc}); using capped fallback", flush=True)
        if not existing_anchors:
            existing_anchors = ", ".join(
                a.canonical_phrase for a in list(self.corpus.anchors.values())[:60]
            )
        if not existing_slabs:
            existing_slabs = ", ".join(
                s.title for s in list(self.corpus.slabs.values())[:60]
                if getattr(s, "title", "")
            )
        return existing_anchors, existing_slabs

    async def extract_proposals(
        self,
        session_id: str,
        chat_id: str,
        recent_messages: list[dict],
        current_turn: int,
        explicit: bool = False,
        user_request: str = "",
        collection_id: Optional[str] = None,
    ) -> list[DraftPacket]:
        """Extract proposed anchors/slabs from conversation.

        Args:
            explicit: True if user explicitly asked for extraction.
            user_request: The user's explicit request text (for explicit mode).
        """
        stack = self.session_store.load_draft_stack(session_id)
        if not stack:
            stack = DraftStack(session_id=session_id)

        # Check cap (explicit requests bypass cap)
        if not explicit and len(stack.packets) >= DRAFT_STACK_CAP:
            return []

        # Build conversation context
        conversation = "\n".join(
            f"[turn {m.get('turn', '?')}] [{m['role']}] {m['content'][:300]}"
            for m in recent_messages[-12:]
        )

        # Existing-node catalogs — SEMANTICALLY SCOPED to this conversation
        # (top-K nearest), not a full dump. See _semantic_catalogs. The hard
        # semantic dedup (_dedup_check) still guards every proposal regardless.
        existing_anchors, existing_slabs = await self._semantic_catalogs(conversation)
        existing_bundles = ", ".join(
            _bundle_label(b) for b in self.corpus.bundles.values() if _bundle_label(b)
        )
        # Active collection IDs — shown to the LLM as FORBIDDEN edge
        # endpoints. Without this, small models (and occasionally large
        # ones) happily emit edges with from_label="vindiesel5" even
        # though "vindiesel5" is a container, not a node, and won't
        # resolve to anything in the label index.
        try:
            active_collections = ", ".join(sorted(self._registry.active_ids)) if self._registry else ""
        except Exception:
            active_collections = ""

        def _fill_catalogs(p: str) -> str:
            return (
                p.replace("$EXISTING_ANCHORS", existing_anchors or "(none)")
                 .replace("$EXISTING_SLABS", existing_slabs or "(none)")
                 .replace("$EXISTING_BUNDLES", existing_bundles or "(none)")
                 .replace("$ACTIVE_COLLECTIONS", active_collections or "(none)")
                 .replace("$CONVERSATION", conversation)
            )

        # Choose prompt
        if explicit:
            prompt = _fill_catalogs(PROPOSAL_EXPLICIT_PROMPT) + json.dumps(user_request)
        else:
            prompt = _fill_catalogs(PROPOSAL_EXTRACTION_PROMPT)

        # Model proposes. Phase 3: output is now a wrapper object
        #   {"proposals": [...], "edges": [...]}
        # but we keep backward compat with the legacy bare-array / bare-object
        # shapes in case the model regresses.
        try:
            raw = _normalize_punct(await ollama.structured_extract(prompt))
            print(f"[DRAFT] Extraction result (turn {current_turn}): {type(raw).__name__} = {str(raw)[:300]}", flush=True)
            proposed_edge_specs: list = []
            if isinstance(raw, dict):
                if "proposals" in raw and isinstance(raw["proposals"], list):
                    # New wrapper format
                    proposals = raw["proposals"]
                    proposed_edge_specs = raw.get("edges") or []
                else:
                    # Legacy: single-object output (explicit path often does this)
                    proposals = [raw]
            elif isinstance(raw, list):
                proposals = raw
            else:
                print(f"[DRAFT] Unexpected result type: {type(raw).__name__}", flush=True)
                return []
        except Exception as e:
            print(f"[DRAFT] Extraction FAILED (turn {current_turn}): {e}", flush=True)
            return []

        if not proposals and not proposed_edge_specs:
            return []

        # Process each proposal
        counts = self._get_counts(session_id)
        created: list[DraftPacket] = []

        for prop in proposals:
            if not isinstance(prop, dict):
                continue

            prop_type = prop.get("type", "anchor")

            # Enforce periodic limits (explicit bypasses)
            if not explicit:
                if prop_type == "slab" and counts["slab"] >= MAX_PERIODIC_SLABS:
                    continue
                if prop_type == "anchor" and counts["anchor"] >= MAX_PERIODIC_ANCHORS:
                    continue
                if len(stack.packets) >= DRAFT_STACK_CAP:
                    break

            # Dedup: check semantic similarity against existing corpus
            text = prop.get("canonical_phrase") or prop.get("canonical_text", "")
            if not text:
                continue

            # Defense-in-depth heading filter for anchor proposals — same
            # rule the document miners apply at extraction parse time. The
            # per-turn sweep prompt rarely produces section-heading-shaped
            # anchors (chat content has no headings) but the filter is
            # cheap and catches edge cases like "Topic A and Topic B" type
            # compound headings the speaker might dictate aloud.
            if prop_type == "anchor":
                from .narrative_miner import _looks_like_section_heading
                if _looks_like_section_heading(text):
                    print(f"[DRAFT]   skip (heading-shape): {text[:60]}", flush=True)
                    continue

            dup = await self._dedup_check(text)
            if dup:
                print(f"[DRAFT]   skip (dup of {dup}): {text[:60]}", flush=True)
                continue
            print(f"[DRAFT]   accept {prop_type}: {text[:60]}", flush=True)

            # Create draft packet
            draft_id = f"tentative_{prop_type}_{uuid.uuid4().hex[:8]}_v1"
            claim_tag = prop.get("claim_tag", "UNKNOWN")

            # source_turns normalization: chat-based mining emits
            # "source_turns" (turn indices in a chat), collection-based
            # mining emits "source_pairs" (node indices in a corpus
            # collection). Packet has one field, so we collapse both
            # into source_turns here; the raw sidecar preserves the
            # distinction for downstream enrichment passes that need
            # to know whether to load a chat or a collection.
            st = prop.get("source_turns") or prop.get("source_pairs")
            if not st:
                st = [current_turn]

            # Build inline typed payload from raw. This mirrors
            # _convert_to_corpus_object() but runs eagerly so the packet
            # is self-contained for review UIs and the dashboard —
            # previously these fields stayed null and the raw file was
            # the only source of truth, which leaked an undocumented
            # sidecar into every consumer that read packets.
            inline_anchor: Optional[dict] = None
            inline_slab: Optional[dict] = None
            inline_bundle: Optional[dict] = None
            if prop_type == "anchor":
                inline_anchor = {
                    "id": draft_id,
                    "canonical_phrase": prop.get("canonical_phrase", "") or "",
                    "aliases": list(prop.get("aliases") or []),
                    "invokes": [],
                    "notes": prop.get("justification", "") or "",
                }
            elif prop_type == "slab":
                # Slabs prefer `title` (explicit) over `canonical_phrase`
                # (which the miner sometimes leaves empty for slab types).
                inline_slab = {
                    "id": draft_id,
                    "title": (
                        prop.get("title")
                        or prop.get("canonical_phrase")
                        or ""
                    ),
                    "canonical_text": prop.get("canonical_text", "") or "",
                    "links": {"anchors": [], "bundles": []},
                    "version": "v1",
                }
            elif prop_type == "bundle":
                # Bundles aren't normally proposed standalone by the miner,
                # but handle the shape defensively in case a future prompt
                # starts emitting them. The payload.intent list is the one
                # required field on BundlePayload per schemas.py §4.5.
                payload = prop.get("payload") or {}
                if "intent" not in payload or not payload["intent"]:
                    justification = prop.get("justification", "").strip()
                    payload["intent"] = [justification] if justification else [draft_id]
                inline_bundle = {
                    "id": draft_id,
                    "payload": payload,
                    "version": "v1",
                }

            packet = DraftPacket(
                id=draft_id,
                packet_type=prop_type,
                source_chat_id=chat_id,
                source_turns=st,
                proposed_nodes=[draft_id],
                justification=prop.get("justification", ""),
                confidence=0.75 if explicit else 0.5,
                status=DraftStatus.DRAFT_UNAUTHORIZED,
                fact_claims=[text] if claim_tag == "FACT" else [],
                anchor=inline_anchor,
                slab=inline_slab,
                bundle=inline_bundle,
            )

            # Store the raw proposal data alongside the draft packet for later conversion
            self.session_store.save_draft_packet(session_id, packet)
            # Also store raw proposal fields for conversion at review time.
            # Phase 2A: stamp _target_collection from the chat's binding so
            # this draft lands in the right subcorpus on promote, instead of
            # silently falling back to "default".
            raw_path = self.session_store.drafts_dir(session_id) / f"{draft_id}_raw.json"
            prop_to_save = dict(prop)
            if collection_id:
                prop_to_save["_target_collection"] = collection_id
            self.session_store.write_json(raw_path, prop_to_save)

            stack.packets.append(draft_id)
            if not explicit:
                counts[prop_type] = counts.get(prop_type, 0) + 1
            created.append(packet)

        self.session_store.save_draft_stack(session_id, stack)

        # Phase 3 — resolve proposed edges (labels -> node_ids) and persist.
        # Factored into _resolve_and_persist_edges so the relationship-
        # mining pass (Phase 4) can reuse the same logic.
        if proposed_edge_specs:
            try:
                n = self._resolve_and_persist_edges(
                    proposed_edge_specs,
                    session_id=session_id,
                    chat_id=chat_id,
                    current_turn=current_turn,
                    new_proposals=created,
                    tag="DRAFT",
                )
                if n:
                    print(f"[DRAFT] Persisted {n} proposed edge(s)", flush=True)
            except Exception as e:
                print(f"[DRAFT] Edge resolution FAILED: {e}", flush=True)

        _event_log.log_proposal_event(
            session_id=session_id,
            turn=current_turn,
            explicit=explicit,
            count=len(created),
            types=[p.id.split("_")[1] for p in created],
        )

        return created

    # ── Edge resolution helpers (shared by extract_proposals + extract_relationships) ──

    def _build_label_index(
        self,
        new_proposals: Optional[list["DraftPacket"]] = None,
    ) -> dict[str, str]:
        """Build a lowercase-label → node_id map for LLM-emitted edge labels.

        Priority order (first-write-wins in the dict):
          1. Newly-mined proposals this turn (if any) — most specific.
          2. Corpus anchors (canonical_phrase + aliases).
          3. Corpus slabs (title).
          4. Corpus bundles (synthesized intent label via _bundle_label).

        Used by:
          - extract_proposals edge resolver (where new_proposals is the
            just-mined packets — they can be edge endpoints by label).
          - extract_relationships edge resolver (where new_proposals is
            None — only existing nodes can be endpoints).
        """
        label_to_id: dict[str, str] = {}

        def _register(label: str, nid: str) -> None:
            if not label or not nid:
                return
            key = label.strip().lower()
            label_to_id.setdefault(key, nid)

        # 1. New proposals
        if new_proposals:
            for pkt in new_proposals:
                if pkt.anchor:
                    _register(pkt.anchor.get("canonical_phrase", ""), pkt.id)
                    for al in pkt.anchor.get("aliases") or []:
                        _register(al, pkt.id)
                if pkt.slab:
                    _register(pkt.slab.get("title", ""), pkt.id)
                if pkt.bundle:
                    _register(pkt.bundle.get("id", ""), pkt.id)

        # 2. Corpus anchors
        for a in self.corpus.anchors.values():
            _register(a.canonical_phrase, a.id)
            for al in a.aliases or []:
                _register(al, a.id)

        # 3. Corpus slabs
        for s in self.corpus.slabs.values():
            if getattr(s, "title", ""):
                _register(s.title, s.id)

        # 4. Corpus bundles
        for b in self.corpus.bundles.values():
            _register(_bundle_label(b), b.id)

        return label_to_id

    def _resolve_and_persist_edges(
        self,
        specs: list,
        session_id: str,
        chat_id: str,
        current_turn: int,
        new_proposals: Optional[list["DraftPacket"]] = None,
        tag: str = "DRAFT",
    ) -> int:
        """Resolve edge specs (LLM output) to ProposedEdge records and persist.

        Dedupes against:
          - Committed corpus.edges (same (from, to, type) triple).
          - Previously-proposed edges in this session (same triple).

        Returns the number of new edges actually persisted.
        """
        from ..models.schemas import ProposedEdge

        label_to_id = self._build_label_index(new_proposals=new_proposals)

        # Signature set for dedup: (from_id, to_id, edge_type_value).
        # Direction-preserving — CONFLICTS and SEQUENCE are directional in
        # their semantics, and for LINKS an asymmetric directional duplicate
        # is still a duplicate for our purposes.
        existing_sigs: set[tuple[str, str, str]] = set()
        for e in self.corpus.edges.values():
            existing_sigs.add((e.from_node, e.to_node, e.type.value))
        try:
            for pe in self.session_store.list_proposed_edges(session_id):
                existing_sigs.add((pe.from_node, pe.to_node, pe.type.value))
        except Exception:
            pass  # no proposed-edges file yet; continue with empty set

        resolved: list[ProposedEdge] = []
        for spec in specs:
            if not isinstance(spec, dict):
                continue
            etype_raw = (spec.get("type") or "LINKS").upper()
            # Whitelist must match EdgeType exactly — TENSIONS was missing, so
            # every dialectic edge the model emitted got silently downgraded to
            # LINKS here (not a model-reasoning issue, a coercion bug).
            if etype_raw not in {
                "INVOKES", "SUPPORTS", "CONFLICTS", "TENSIONS",
                "LINKS", "SEQUENCE", "PARENT_OF",
            }:
                etype_raw = "LINKS"
            from_label = (spec.get("from_label") or "").strip()
            to_label = (spec.get("to_label") or "").strip()
            if not from_label or not to_label:
                continue
            from_id, from_strategy = _fuzzy_resolve_label(from_label, label_to_id)
            to_id, to_strategy = _fuzzy_resolve_label(to_label, label_to_id)
            if not from_id or not to_id:
                print(
                    f"[{tag}]   skip edge (unresolved): "
                    f"{from_label!r} -> {to_label!r}",
                    flush=True,
                )
                continue
            # Surface fuzzy matches so we can spot false positives early.
            if from_strategy != "exact" or to_strategy != "exact":
                print(
                    f"[{tag}]   fuzzy edge resolve: "
                    f"{from_label!r} ({from_strategy}→{from_id}) -> "
                    f"{to_label!r} ({to_strategy}→{to_id})",
                    flush=True,
                )
            if from_id == to_id:
                continue  # no self-loops
            sig = (from_id, to_id, etype_raw)
            if sig in existing_sigs:
                continue  # already in corpus or already proposed
            existing_sigs.add(sig)  # dedupe within this batch too
            try:
                confidence = float(spec.get("confidence", 0.5))
            except Exception:
                confidence = 0.5
            resolved.append(ProposedEdge(
                id=f"proposed_edge_{uuid.uuid4().hex[:8]}",
                type=EdgeType(etype_raw),
                from_node=from_id,
                to_node=to_id,
                from_label=from_label,
                to_label=to_label,
                confidence=max(0.0, min(1.0, confidence)),
                justification=spec.get("justification", "") or "",
                status="PROPOSED",
                source_chat_id=chat_id,
                source_turn=current_turn,
            ))

        if resolved:
            self.session_store.append_proposed_edges(session_id, resolved)
        return len(resolved)

    async def extract_relationships(
        self,
        session_id: str,
        chat_id: str,
        recent_messages: list[dict],
        current_turn: int,
    ) -> int:
        """Second-pass miner: detect edges between EXISTING corpus nodes only.

        Runs after ``extract_proposals`` on every sweep. The primary miner
        is oriented toward concept extraction (find new anchors/slabs);
        this pass is narrower and complementary — it asks the LLM:
        "given this conversation and the list of existing nodes, which
        pairs of existing nodes are related, and how?"

        Neither a tentative nor a library draft is produced. Only
        ProposedEdges that reference two corpus-resident nodes.

        Returns the number of edges persisted.
        """
        if not recent_messages:
            return 0

        conversation = "\n".join(
            f"[turn {m.get('turn', '?')}] [{m['role']}] {m['content'][:300]}"
            for m in recent_messages[-12:]
        )

        # Relationship mining is pure existing-to-existing edge detection, so
        # the endpoint catalog IS the label space — scoping it to the nodes
        # semantically relevant to this conversation matters even more here
        # than for proposals. Same top-K helper (reuses the cosine wiring).
        existing_anchors, existing_slabs = await self._semantic_catalogs(conversation)
        existing_bundles = ", ".join(
            _bundle_label(b) for b in self.corpus.bundles.values() if _bundle_label(b)
        )

        # If there are no existing anchors AND no slabs (brand-new corpus),
        # there's nothing to relate. Skip the LLM call.
        if not existing_anchors and not existing_slabs:
            return 0

        try:
            active_collections = ", ".join(sorted(self._registry.active_ids)) if self._registry else ""
        except Exception:
            active_collections = ""

        prompt = (
            RELATIONSHIP_MINING_PROMPT
            .replace("$EXISTING_ANCHORS", existing_anchors or "(none)")
            .replace("$EXISTING_SLABS", existing_slabs or "(none)")
            .replace("$EXISTING_BUNDLES", existing_bundles or "(none)")
            .replace("$ACTIVE_COLLECTIONS", active_collections or "(none)")
            .replace("$CONVERSATION", conversation)
        )

        try:
            raw = _normalize_punct(await ollama.structured_extract(prompt))
        except Exception as e:
            print(f"[RELMINE] Extraction FAILED (turn {current_turn}): {e}", flush=True)
            return 0

        specs: list = []
        if isinstance(raw, dict):
            specs = raw.get("edges") or []
        elif isinstance(raw, list):
            # Model regressed to bare array — treat as edge list directly.
            specs = raw
        if not specs:
            print(f"[RELMINE] No relationships detected at turn {current_turn}", flush=True)
            return 0

        try:
            n = self._resolve_and_persist_edges(
                specs,
                session_id=session_id,
                chat_id=chat_id,
                current_turn=current_turn,
                new_proposals=None,  # existing-only miner
                tag="RELMINE",
            )
        except Exception as e:
            print(f"[RELMINE] Edge resolution FAILED: {e}", flush=True)
            return 0

        if n:
            print(f"[RELMINE] Persisted {n} existing-to-existing edge(s) at turn {current_turn}", flush=True)
        else:
            print(f"[RELMINE] {len(specs)} candidate edge(s) but all dedup'd or unresolved", flush=True)
        return n

    async def verify_draft_claims(
        self,
        session_id: str,
        draft_id: str,
        context: str = "",
        method: str = "ollama",
    ) -> dict:
        """Verify all FACT claims in a draft packet.

        Returns verification results. Does NOT auto-promote — the user
        must still call review_draft with promote_corpus after seeing results.
        """
        packet = self.session_store.load_draft_packet(session_id, draft_id)
        if not packet:
            return {"error": "Draft not found"}
        if not packet.fact_claims:
            return {"draft_id": draft_id, "claims": [], "message": "No FACT claims to verify"}

        batch = await self.verifier.verify_batch(
            packet.fact_claims, context=context, method=method,
        )

        return {
            "draft_id": draft_id,
            "verification": batch.summary(),
            "results": [
                {
                    "claim": r.claim[:200],
                    "outcome": r.outcome.value,
                    "confidence": round(r.confidence, 2),
                    "evidence": r.evidence[:200],
                    "notes": r.notes[:200],
                }
                for r in batch.results
            ],
            "can_promote": batch.all_confirmed or (
                not batch.has_contradictions and batch.unresolvable_count == 0
            ),
        }

    async def review_draft(
        self,
        session_id: str,
        draft_id: str,
        action: str,
        oli_mode: str = "OFF",
        drift_severity: str = "low",
        defer_persist: bool = False,
    ) -> dict:
        """Review a draft: discard / promote_tentative / promote_corpus.

        §26.6: corpus commits require OLI ON. FACT verification mandatory.
        Drift gate: blocks corpus promotion when drift severity is HIGH.

        ``defer_persist`` (promote_corpus only) is the bulk-commit fast
        path: the corpus object is added to the store in memory but the
        whole-corpus ``validate()`` + ``save()`` are SKIPPED — the batch
        caller runs them once for the whole selection. Per-draft those
        two are an accidental O(M^2) (each save rewrites the growing
        YAML); hoisting them out of the loop makes a 300-draft promote
        O(M) work + O(1) disk. Single-draft callers leave it False and
        keep the per-draft validate/rollback safety net.
        """
        packet = self.session_store.load_draft_packet(session_id, draft_id)
        if not packet:
            return {"error": "Draft not found"}

        raw_path = self.session_store.drafts_dir(session_id) / f"{draft_id}_raw.json"
        raw = self.session_store.read_json(raw_path) or {}

        if action == "discard":
            packet.status = DraftStatus.REJECTED
            self.session_store.save_draft_packet(session_id, packet)
            self._mark_resolved(session_id, draft_id)
            return {"status": "REJECTED", "draft_id": draft_id}

        if action == "promote_tentative":
            # Drift gate (soft): warn but allow tentative promotion during high drift
            warning = None
            if drift_severity == DriftSeverity.HIGH.value:
                warning = (
                    "Session drift is HIGH. Tentative promotion allowed but "
                    "this node may reflect volatile epistemic state."
                )

            packet.status = DraftStatus.PROVISIONAL
            self.session_store.save_draft_packet(session_id, packet)
            # Write to library/tentative/
            from pathlib import Path
            tent_dir = Path("app/library/tentative")
            tent_dir.mkdir(parents=True, exist_ok=True)
            self.session_store.write_json(tent_dir / f"{draft_id}.json", {
                "draft": packet.model_dump(mode="json"),
                "raw_proposal": raw,
            })
            self._mark_resolved(session_id, draft_id)
            result = {"status": "PROVISIONAL", "location": "library/tentative", "draft_id": draft_id}
            if warning:
                result["warning"] = warning
            return result

        if action == "promote_corpus":
            # §26.6: Corpus commits prefer OLI ON for in-session promotion.
            # End-of-session review is an explicit curation act — user has full
            # context and is deliberately choosing to commit. Warn but allow.
            oli_warning = None
            if oli_mode != "ON":
                oli_warning = "Note: OLI is OFF. Commit proceeding — you're in review mode."

            # Drift gate (hard): block corpus promotion during HIGH drift
            if drift_severity == DriftSeverity.HIGH.value:
                return {
                    "error": "Corpus promotion blocked: session drift is HIGH. "
                    "Wait for drift to subside or end the session and review in a calmer state.",
                    "drift_severity": drift_severity,
                }

            # FACT claims: attempt auto-verification if not yet verified
            if packet.fact_claims:
                batch = await self.verifier.verify_batch(packet.fact_claims)
                if batch.has_contradictions:
                    return {
                        "error": "FACT claims contain contradictions — cannot promote to corpus",
                        "claims": packet.fact_claims,
                        "verification": batch.summary(),
                    }
                if not batch.all_confirmed:
                    return {
                        "error": "FACT claims not fully verified — review verification results",
                        "claims": packet.fact_claims,
                        "verification": batch.summary(),
                    }

            # Convert to corpus object. Pass the already-loaded packet so
            # any post-wrap enrichment (dreaming-phase rewrites, reviewer
            # edits, schema repairs) reaches the corpus. Without the packet
            # argument the function falls back to raw-only behavior for
            # backward compatibility.
            obj = self._convert_to_corpus_object(draft_id, raw, packet=packet)
            if not obj:
                return {"error": "Could not convert proposal to corpus object"}

            # Determine target collection for commit
            target_cid = raw.get("_target_collection", "default")
            target_store = self.corpus  # default: merged view

            # Try to resolve to a specific collection store
            if hasattr(self, '_registry') and self._registry:
                specific = self._registry.get_store(target_cid)
                if specific:
                    target_store = specific

            # Add to target store, validate, save or rollback
            obj_type, corpus_obj = obj
            generated_bundle = None  # kept for back-compat with edge generator + result dict

            if obj_type == "anchor":
                target_store.anchors[corpus_obj.id] = corpus_obj
                # Also add to merged view so validation sees it
                if target_store is not self.corpus:
                    self.corpus.anchors[corpus_obj.id] = corpus_obj
            elif obj_type == "bundle":
                # Pass-2 thematic bundle commit. Lives in target_store.bundles
                # (KeyBundle objects). Member-anchor resolution is deferred to
                # edge-commit time: the label resolver consumes the bundle's
                # _member_phrases from the raw sidecar and emits bundle→anchor
                # INVOKES edges as the referenced anchors commit (now or later).
                target_store.bundles[corpus_obj.id] = corpus_obj
                if target_store is not self.corpus:
                    self.corpus.bundles[corpus_obj.id] = corpus_obj
            elif obj_type == "slab":
                # Populate slab.intent + slab.invariants from the slab's
                # own content BEFORE adding to store, so validation sees
                # the complete object. Replaces the previous 1:1
                # "coherence bundle" auto-generation — same metadata,
                # but now lives ON the slab instead of as a sibling
                # corpus node. See _generate_minimum_bundle docstring.
                intent, invariants = self._generate_minimum_bundle(
                    corpus_obj.id, raw, packet=packet
                )
                if intent:
                    corpus_obj.intent = intent
                if invariants:
                    corpus_obj.invariants = invariants
                target_store.slabs[corpus_obj.id] = corpus_obj
                if target_store is not self.corpus:
                    self.corpus.slabs[corpus_obj.id] = corpus_obj

            # --- Auto-generate edges from inline structural fields ---
            # These are placeholder-weight edges that make the dual
            # representation (inline fields ↔ edges.yaml) consistent at
            # commit time. Weights use tier defaults:
            #   INVOKES  1.0  — curator-declared (anchor.invokes is hand-set)
            #   LINKS    0.8  — packet-layer structural link
            #   SUPPORTS 0.7  — auto-generated coherence bundle
            # The dreaming pass can refine these later based on actual
            # traversal patterns and co-activation frequency.
            generated_edges: list[Edge] = []
            generated_edges = self._generate_commit_edges(
                obj_type, corpus_obj, generated_bundle
            )
            for edge in generated_edges:
                target_store.edges[edge.id] = edge
                if target_store is not self.corpus:
                    self.corpus.edges[edge.id] = edge

            # Per-draft validate + save — skipped in bulk mode, where the
            # batch caller validates and saves each touched store once.
            if not defer_persist:
                errors = target_store.validate()
                if errors:
                    # Rollback nodes + edges. (No coherence bundle to roll
                    # back since slab metadata now lives on the slab itself.)
                    if obj_type == "anchor":
                        target_store.anchors.pop(corpus_obj.id, None)
                        self.corpus.anchors.pop(corpus_obj.id, None)
                    elif obj_type == "slab":
                        target_store.slabs.pop(corpus_obj.id, None)
                        self.corpus.slabs.pop(corpus_obj.id, None)
                    for edge in generated_edges:
                        target_store.edges.pop(edge.id, None)
                        self.corpus.edges.pop(edge.id, None)
                    return {"error": "Corpus validation failed", "errors": errors}

                target_store.save()

            packet.status = DraftStatus.COMMITTED
            self.session_store.save_draft_packet(session_id, packet)
            self._mark_resolved(session_id, draft_id)

            result = {
                "status": "COMMITTED", "draft_id": draft_id,
                "corpus_id": corpus_obj.id,
                # The collection this object landed in — the batch caller
                # reads it to know which stores to validate + save once.
                "target_collection": target_cid,
            }
            if generated_edges:
                result["generated_edges"] = [e.id for e in generated_edges]
            if oli_warning:
                result["warning"] = oli_warning
            return result

        return {"error": f"Unknown action: {action}"}

    def _convert_to_corpus_object(
        self,
        draft_id: str,
        raw: dict,
        packet: Optional[DraftPacket] = None,
    ) -> Optional[tuple]:
        """Convert a reviewed draft into a typed corpus object.

        Fallback ladder (three-deep): for each field we prefer

            1. The packet's inline dict (packet.anchor / packet.slab / packet.bundle)
               — this is where any post-wrap enrichment lives, including the
               dreaming-phase rewrites that land as ``{id}.enriched.json`` and
               get promoted over ``{id}.json``.
            2. The packet's top-level fields (packet.justification, etc.) —
               used for legacy compatibility where the pre-enrichment contract
               stored notes/reasons here.
            3. The raw sidecar (``{id}_raw.json``) — the miner's original
               output, used only when neither of the above is populated.

        Passing ``packet=None`` preserves the legacy raw-only behavior so
        any caller that still doesn't have a packet in scope keeps working.

        Why this matters: before this patch the commit path read exclusively
        from ``raw``, which meant any edit made at the packet layer (a
        reviewer's manual tweak, a dreaming-phase rewrite, a schema-level
        repair) was silently discarded at commit time. The corpus always
        reflected the miner's first-pass output, never the reviewed version.
        That was the mirror image of the (now-fixed) extract_proposals wrap
        bug — together the two functions formed a hermetic seal around the
        raw layer.
        """
        # --- Decide packet_type ---
        if packet is not None and packet.packet_type:
            prop_type = packet.packet_type
        else:
            prop_type = raw.get("type", "anchor")

        # --- Extract inline dicts (may be None if packet missing/incomplete) ---
        p_anchor = packet.anchor if packet else None
        p_slab = packet.slab if packet else None
        p_justification = packet.justification if packet else ""

        def _pick(field: str, inline: Optional[dict], raw_keys: list[str], default):
            """Prefer inline[field], then first non-empty raw key, then default."""
            if inline and inline.get(field) not in (None, ""):
                return inline[field]
            for k in raw_keys:
                v = raw.get(k)
                if v not in (None, ""):
                    return v
            return default

        if prop_type == "anchor":
            canonical_phrase = _pick(
                "canonical_phrase", p_anchor, ["canonical_phrase"], draft_id
            )
            aliases = _pick("aliases", p_anchor, ["aliases"], []) or []
            # Notes: inline anchor.notes is the enriched home; packet.justification
            # is legacy / rewrite-disclaimer; raw.justification is the miner's
            # original reason. All three are checked in that order.
            notes = ""
            if p_anchor and p_anchor.get("notes"):
                notes = p_anchor["notes"]
            elif p_justification:
                notes = p_justification
            else:
                notes = raw.get("justification", "")
            invokes = _pick("invokes", p_anchor, [], []) or []
            depends_on = _pick("depends_on", p_anchor, [], []) or []
            assumptions = _pick("assumptions", p_anchor, [], []) or []

            anchor = Anchor(
                id=draft_id,
                canonical_phrase=canonical_phrase,
                aliases=list(aliases),
                invokes=list(invokes),
                notes=notes,
                match_policy=AnchorMatchPolicy(),
                meta=AnchorMeta(version="v1"),
                depends_on=list(depends_on),
                assumptions=list(assumptions),
            )
            return ("anchor", anchor)

        if prop_type == "slab":
            from ..models.schemas import SlabLinks

            # Title: the enriched slab uses a human-readable title; the raw
            # miner output stored it in canonical_phrase. Check both.
            title = ""
            if p_slab and p_slab.get("title"):
                title = p_slab["title"]
            else:
                title = raw.get("canonical_phrase") or raw.get("title", "")

            canonical_text = _pick(
                "canonical_text", p_slab, ["canonical_text"], ""
            )

            # SlabLinks: the one field with no raw fallback at all. The raw
            # shape doesn't carry cross-draft links; they're a packet-layer
            # phenomenon introduced by the dreaming rewrite. Default empty
            # only if the packet also doesn't carry them.
            links_raw = (p_slab or {}).get("links") or {}
            slab_links = SlabLinks(
                anchors=list(links_raw.get("anchors", []) or []),
                bundles=list(links_raw.get("bundles", []) or []),
            )

            depends_on = _pick("depends_on", p_slab, [], []) or []
            assumptions = _pick("assumptions", p_slab, [], []) or []

            # Provenance note: prefer packet.justification (which on enriched
            # packets carries the rewrite disclaimer pointing at the dream log),
            # else raw.justification (miner original).
            prov_note = p_justification or raw.get("justification", "")

            slab = Slab(
                id=draft_id,
                title=title,
                canonical_text=canonical_text,
                links=slab_links,
                version="v1",
                meta=AnchorMeta(version="v1"),
                depends_on=list(depends_on),
                assumptions=list(assumptions),
                provenance_refs=[
                    ProvenanceRef(
                        ref_id=f"WB_{draft_id}",
                        store="workbench",
                        type="conversation_segment",
                        note=prov_note,
                    )
                ],
            )
            return ("slab", slab)

        if prop_type == "bundle":
            # Pass-2 thematic bundle from convo/narrative miner clustering.
            # Distinct from the (now-removed) auto-generated 1:1 coherence
            # bundles — these are real cross-slab thematic clusters with
            # a curated label + member-anchor list.
            #
            # Shape contract (set by mining.py push-mined handler):
            #   packet.bundle = {
            #     "id": draft_id,
            #     "payload": {intent, invariants, non_assumptions, ...},
            #     "version": "v1",
            #     "_member_phrases": [list of member anchor canonical_phrases],
            #   }
            # Member phrases stay as phrases here; they get resolved to
            # anchor IDs at edge-commit time via the label resolver, which
            # produces INVOKES edges from bundle → anchors.
            from ..models.schemas import BundlePayload, KeyBundle

            p_bundle = packet.bundle if packet else None
            p_payload = (p_bundle or {}).get("payload") or {}

            # Intent is required (min_length=1) by BundlePayload. Fall-through
            # ladder: inline payload intent → packet.justification → raw label/justification → draft_id.
            intent = list(p_payload.get("intent") or [])
            if not intent:
                if p_justification:
                    intent = [p_justification]
                elif raw.get("justification"):
                    intent = [raw["justification"]]
                elif raw.get("label"):
                    intent = [raw["label"]]
                else:
                    intent = [draft_id]
            # Schema cap is 5 entries; truncate defensively.
            intent = intent[:5]

            payload = BundlePayload(
                intent=intent,
                invariants=list(p_payload.get("invariants") or []),
                non_assumptions=list(p_payload.get("non_assumptions") or []),
                warnings=list(p_payload.get("warnings") or []),
                heuristics=list(p_payload.get("heuristics") or []),
                markers=list(p_payload.get("markers") or []),
                rules=list(p_payload.get("rules") or []),
                activation_clause=list(p_payload.get("activation_clause") or []),
                canonical_quote_handles=list(p_payload.get("canonical_quote_handles") or []),
                closing_clause=p_payload.get("closing_clause"),
            )
            bundle = KeyBundle(
                id=draft_id,
                payload=payload,
                version="v1",
                meta=AnchorMeta(version="v1"),
            )
            return ("bundle", bundle)

        return None

    def _generate_minimum_bundle(
        self,
        slab_id: str,
        raw: dict,
        packet: Optional[DraftPacket] = None,
    ) -> tuple[list[str], list[str]]:
        """Extract slab-level intent + invariants metadata from raw content.

        Returns (intent, invariants) tuples to be assigned directly to
        ``slab.intent`` and ``slab.invariants`` at promote time. Slab
        owns this metadata as fields rather than as a sibling 1:1
        "coherence bundle" corpus node.

        Background: this function used to construct a KeyBundle whose
        payload was *literally derived from this single slab's text*
        (see git blame for the prior shape). The bundle then existed
        as a separate corpus node 1:1 with the slab — same content,
        same lifecycle, no cross-slab role. Pure data duplication
        flagged in podv3 audit (300 slabs → 296 coherence bundles, all
        single-slab). Now the metadata lives on the Slab itself; the
        ``bundle`` corpus type is reserved for *actual* thematic
        cross-slab clusters from the Pass-2 mining path.

        Packet-aware: same three-deep fallback as
        ``_convert_to_corpus_object`` — prefers the enriched inline
        slab dict (post-dream rewrite) over raw sidecar.

        Returns ([], []) when the slab is too short to bother
        extracting metadata, mirroring the pre-refactor behaviour
        of returning None.
        """
        # Three-deep fallback: inline slab dict → packet.justification → raw
        p_slab = packet.slab if packet else None
        canonical_text = ""
        if p_slab and p_slab.get("canonical_text"):
            canonical_text = p_slab["canonical_text"]
        else:
            canonical_text = raw.get("canonical_text", "")

        justification = ""
        if packet and packet.justification:
            justification = packet.justification
        else:
            justification = raw.get("justification", "")

        # Only extract if slab has real content
        if len(canonical_text) < 50:
            return [], []

        intent = [justification] if justification else []

        # Extract key claims from canonical text as invariants —
        # sentence-split by period, drop trivially-short fragments.
        # Cap at 5 to keep slab.invariants bounded.
        invariants = [
            s.strip()
            for s in canonical_text.split('.')
            if len(s.strip()) > 15
        ][:5]

        _event_log.log_proposal_event(
            event="slab_metadata_extracted_at_commit",
            slab_id=slab_id,
            intent_count=len(intent[:3]),
            invariants_count=len(invariants),
        )

        return intent[:3], invariants

    # ------------------------------------------------------------------
    # Edge generation at commit time
    # ------------------------------------------------------------------

    def _generate_commit_edges(
        self,
        obj_type: str,
        corpus_obj,
        generated_bundle=None,
    ) -> list[Edge]:
        """Derive explicit Edge objects from the inline structural fields.

        The corpus has a dual representation for relationships:

          1. **Inline fields** on the nodes themselves — ``anchor.invokes``,
             ``slab.links.anchors``, ``slab.links.bundles``, ``bundle.supports``.
             These drive the activation cascade in frame_manager (Rules 1-3).

          2. **Explicit Edge records** in ``edges.yaml`` — typed objects with
             weight/confidence/tension. These drive structural-weight scoring
             (frame_manager Step 2), conflict detection (Step 5), and graph
             visualization in the frontend.

        Before this method existed, the commit path only wrote (1), leaving
        (2) empty for auto-committed nodes. This function bridges the gap
        by emitting edges for every inline structural reference.

        **Weight tiers** (placeholder defaults — dreaming pass refines later):

          - ``INVOKES  w=1.0`` — curator-declared intent (anchor.invokes is
            hand-set or dreaming-grounded; highest confidence).
          - ``LINKS    w=0.8`` — packet-layer structural link (slab↔anchor
            wiring from the dreaming pass; high but not curator-declared).
          - ``SUPPORTS w=0.7`` — auto-generated coherence bundle (§4.1.3
            minimum bundle; lower confidence because the bundle content
            is mechanically derived, not reviewed).

        Dedup: if an edge between the same (from, to, type) triple already
        exists in the corpus, the existing edge is kept and no duplicate is
        emitted. This makes re-commit and manual curation safe.
        """
        edges: list[Edge] = []

        # Quick lookup for existing (from, to, type) triples to avoid dupes
        existing_triples: set[tuple[str, str, str]] = set()
        for e in self.corpus.edges.values():
            existing_triples.add((e.from_node, e.to_node, e.type))

        def _maybe_add(
            edge_type: EdgeType,
            from_id: str,
            to_id: str,
            weight: float,
        ) -> None:
            """Add an edge if the (from, to, type) triple is new."""
            triple = (from_id, to_id, edge_type)
            if triple in existing_triples:
                return
            # Deterministic id: type_fromshort_toshort_v1
            from_short = from_id.replace("tentative_", "").replace("mined_", "")[:20]
            to_short = to_id.replace("tentative_", "").replace("mined_", "")[:20]
            edge_id = f"edge_{edge_type.value.lower()}_{from_short}_{to_short}_v1"
            edge = Edge(
                id=edge_id,
                type=edge_type,
                from_node=from_id,
                to_node=to_id,
                weight=weight,
                confidence=weight,  # mirror weight as initial confidence
            )
            edges.append(edge)
            existing_triples.add(triple)  # prevent self-duplication within batch

        if obj_type == "anchor":
            # anchor.invokes → INVOKES edges (curator-declared, w=1.0)
            for target_id in getattr(corpus_obj, "invokes", []) or []:
                _maybe_add(EdgeType.INVOKES, corpus_obj.id, target_id, 1.0)

        elif obj_type == "slab":
            links = getattr(corpus_obj, "links", None)
            if links:
                # slab.links.anchors → LINKS edges (packet-layer, w=0.8)
                for anchor_id in links.anchors or []:
                    _maybe_add(EdgeType.LINKS, corpus_obj.id, anchor_id, 0.8)
                # slab.links.bundles → LINKS edges (packet-layer, w=0.8)
                for bundle_id in links.bundles or []:
                    _maybe_add(EdgeType.LINKS, corpus_obj.id, bundle_id, 0.8)

        # Coherence-bundle SUPPORTS edges removed — slab metadata now
        # lives on the slab itself (slab.intent + slab.invariants), no
        # sibling bundle to point at. ``generated_bundle`` is always
        # None on this path; argument kept for back-compat with any
        # caller still passing it.

        if edges:
            _event_log.log_proposal_event(
                event="edges_generated_at_commit",
                source_id=corpus_obj.id,
                edge_count=len(edges),
                edge_ids=[e.id for e in edges],
            )

        return edges

    async def _dedup_check(self, text: str) -> Optional[str]:
        """Check if text is too similar to an existing corpus object.

        Uses the self-healing dedup embedding cache (Action 2): the corpus
        side is embedded ONCE and reused across turns; only newly-present
        objects are batch-embedded and deleted ones are pruned, while the
        proposal text is embedded once. The prior implementation called
        cosine_similarity per anchor AND per slab, and each call re-embedded
        *both* texts — i.e. O(proposals x corpus) uncached Ollama round-trips
        per turn. Behavior is preserved: lowercased embeddings, DEDUP_THRESHOLD,
        and anchors-before-slabs return precedence are all unchanged.
        """
        import numpy as np

        text_lower = text.lower().strip()

        # Fast string-match pass (catches exact and near-exact dupes)
        for anchor in self.corpus.anchors.values():
            if text_lower == anchor.canonical_phrase.lower().strip():
                return anchor.id
            # Also check if proposal text contains an existing anchor ID
            if anchor.id.lower() in text_lower:
                return anchor.id
            # Check aliases
            for alias in anchor.aliases:
                if text_lower == alias.lower().strip():
                    return anchor.id

        # ── Embedding similarity pass (cached) ──
        # Build the id -> lowercased canonical text for every comparison
        # target (all anchors + all slabs), matching the prior slices.
        targets: dict[str, str] = {}
        for anchor in self.corpus.anchors.values():
            targets[anchor.id] = anchor.canonical_phrase.lower()
        for slab in self.corpus.slabs.values():
            targets[slab.id] = slab.canonical_text[:200].lower()

        # Prune cache entries for objects no longer in the corpus.
        for stale_id in self._dedup_vec_cache.keys() - targets.keys():
            self._dedup_vec_cache.pop(stale_id, None)

        # Batch-embed any targets not yet cached (new / just-promoted).
        missing_ids = [oid for oid in targets if oid not in self._dedup_vec_cache]
        if missing_ids:
            try:
                missing_vecs = await ollama.embed([targets[oid] for oid in missing_ids])
            except Exception:
                logger.warning(
                    "dedup: corpus embedding failed for %d objects; skipping dedup",
                    len(missing_ids),
                )
                return None
            for oid, vec in zip(missing_ids, missing_vecs):
                v = np.array(vec, dtype=float)
                n = float(np.linalg.norm(v))
                self._dedup_vec_cache[oid] = v / n if n > 1e-8 else v

        # Embed the proposal text once, normalized — cosine becomes a dot.
        try:
            prop_vec = np.array((await ollama.embed([text_lower]))[0], dtype=float)
        except Exception:
            logger.warning("dedup: proposal embedding failed; skipping dedup")
            return None
        prop_norm = float(np.linalg.norm(prop_vec))
        if prop_norm < 1e-8:
            return None
        prop_vec = prop_vec / prop_norm

        # Anchors first (original precedence — any anchor hit returns).
        for anchor in self.corpus.anchors.values():
            vec = self._dedup_vec_cache.get(anchor.id)
            if vec is None:
                continue
            if float(np.dot(prop_vec, vec)) > DEDUP_THRESHOLD:
                return anchor.id

        for slab in self.corpus.slabs.values():
            vec = self._dedup_vec_cache.get(slab.id)
            if vec is None:
                continue
            if float(np.dot(prop_vec, vec)) > DEDUP_THRESHOLD:
                return slab.id

        return None

    def get_draft_stack(self, session_id: str) -> Optional[DraftStack]:
        return self.session_store.load_draft_stack(session_id)

    def list_drafts(self, session_id: str, include_resolved: bool = False) -> list[DraftPacket]:
        """List drafts for a session.

        By default returns only pending (DRAFT_UNAUTHORIZED) drafts — that's
        the set a reviewer actually needs to act on. Pass include_resolved=True
        to include the full history (PROVISIONAL, COMMITTED, REJECTED).

        Packet files stay on disk regardless — they're the audit trail. This
        is purely a view filter.
        """
        packets = self.session_store.list_draft_packets(session_id)
        if include_resolved:
            return packets
        return [p for p in packets if p.status == DraftStatus.DRAFT_UNAUTHORIZED]

    def list_drafts_by_chat(self, chat_id: str, include_resolved: bool = False) -> list[DraftPacket]:
        """List drafts for a chat across ALL its sessions.

        A chat can span many SSE sessions (restarts, reconnects), but UI
        reviewers think in chat scope, not session scope. This aggregates.
        Default filter still DRAFT_UNAUTHORIZED; include_resolved returns all.
        """
        packets = self.session_store.list_draft_packets_by_chat(chat_id)
        if include_resolved:
            return packets
        return [p for p in packets if p.status == DraftStatus.DRAFT_UNAUTHORIZED]

    def _mark_resolved(self, session_id: str, draft_id: str) -> None:
        """Remove a draft ID from the session's stack after promote/discard.

        Packet file remains on disk (status updated) so the audit trail is
        preserved. The stack is the live worklist — resolved drafts leave it.
        Idempotent: no-op if the ID isn't in the stack.
        """
        stack = self.session_store.load_draft_stack(session_id)
        if stack and draft_id in stack.packets:
            stack.packets.remove(draft_id)
            self.session_store.save_draft_stack(session_id, stack)

    # ─── Library tentative (on-disk PROVISIONAL store) ──────────────────
    def list_library_tentative(self) -> list[dict]:
        """Return summary records for every JSON file in app/library/tentative/.

        These are drafts the user pressed "Promote Tentative" on. They are
        NOT in the corpus — they live in a sidecar directory as JSON blobs
        with the original packet + raw_proposal preserved. The reasoner
        never sees them; the graph may render them with a dashed overlay.
        """
        tent_dir = Path("app/library/tentative")
        if not tent_dir.exists():
            return []
        out: list[dict] = []
        for f in sorted(tent_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("Failed to parse tentative %s: %s", f.name, e)
                continue
            draft = data.get("draft", {}) or {}
            raw = data.get("raw_proposal", {}) or {}
            label = (
                raw.get("canonical_phrase")
                or raw.get("title")
                or raw.get("label")
                or (raw.get("canonical_text") or "")[:80]
                or f.stem
            )
            out.append({
                "id": f.stem,
                "label": label,
                "type": raw.get("type") or draft.get("packet_type") or "anchor",
                "status": draft.get("status", "PROVISIONAL"),
                "confidence": draft.get("confidence", 0.0),
                "justification": draft.get("justification") or raw.get("justification", ""),
                "source_turns": draft.get("source_turns", []),
                # Target collection — where this will land if promoted. Pinned
                # at the time the draft was originally pushed (AI Mine picker
                # or chat.collection_id). Surfaced so the Tentative tab can
                # show it without the user having to switch corpora.
                "target_collection": raw.get("_target_collection", "default"),
                "draft": draft,
                "raw": raw,
                "path": str(f),
                "mtime": f.stat().st_mtime,
            })
        return out

    def delete_library_tentative(self, tid: str) -> bool:
        path = Path("app/library/tentative") / f"{tid}.json"
        if not path.exists():
            return False
        path.unlink()
        return True

    def promote_library_tentative(self, tid: str) -> dict:
        """Commit a library-tentative node into the active corpus.

        Mirrors the promote_corpus branch of review_draft() but without the
        session-draft-stack bookkeeping (the packet is no longer tracked by
        a live session). The tentative JSON file is deleted on success.
        """
        path = Path("app/library/tentative") / f"{tid}.json"
        if not path.exists():
            return {"error": "Tentative record not found"}

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            return {"error": f"Failed to parse tentative file: {e}"}

        raw = data.get("raw_proposal", {}) or {}
        draft_dict = data.get("draft", {}) or {}
        try:
            packet = DraftPacket.model_validate(draft_dict) if draft_dict else None
        except Exception as e:
            logger.warning("Tentative %s packet reconstruction failed: %s — using raw only", tid, e)
            packet = None

        obj = self._convert_to_corpus_object(tid, raw, packet=packet)
        if not obj:
            return {"error": "Could not convert to corpus object"}

        target_cid = raw.get("_target_collection", "default")
        target_store = self.corpus
        if hasattr(self, "_registry") and self._registry:
            specific = self._registry.get_store(target_cid)
            if specific:
                target_store = specific

        obj_type, corpus_obj = obj
        generated_bundle = None  # back-compat with _generate_commit_edges signature

        if obj_type == "anchor":
            target_store.anchors[corpus_obj.id] = corpus_obj
            if target_store is not self.corpus:
                self.corpus.anchors[corpus_obj.id] = corpus_obj
        elif obj_type == "slab":
            # Same pattern as the main promote path: populate slab
            # metadata fields directly, no sibling coherence bundle.
            intent, invariants = self._generate_minimum_bundle(corpus_obj.id, raw, packet=packet)
            if intent:
                corpus_obj.intent = intent
            if invariants:
                corpus_obj.invariants = invariants
            target_store.slabs[corpus_obj.id] = corpus_obj
            if target_store is not self.corpus:
                self.corpus.slabs[corpus_obj.id] = corpus_obj

        generated_edges = self._generate_commit_edges(obj_type, corpus_obj, generated_bundle)
        for edge in generated_edges:
            target_store.edges[edge.id] = edge
            if target_store is not self.corpus:
                self.corpus.edges[edge.id] = edge

        errors = target_store.validate()
        if errors:
            if obj_type == "anchor":
                target_store.anchors.pop(corpus_obj.id, None)
                self.corpus.anchors.pop(corpus_obj.id, None)
            elif obj_type == "slab":
                target_store.slabs.pop(corpus_obj.id, None)
                self.corpus.slabs.pop(corpus_obj.id, None)
            for edge in generated_edges:
                target_store.edges.pop(edge.id, None)
                self.corpus.edges.pop(edge.id, None)
            return {"error": "Corpus validation failed", "errors": errors}

        target_store.save()
        # Success — remove from tentative library
        try:
            path.unlink()
        except Exception as e:
            logger.warning("Committed tentative %s but failed to delete file: %s", tid, e)

        return {
            "status": "COMMITTED",
            "id": tid,
            "corpus_id": corpus_obj.id,
            "generated_bundle": generated_bundle.id if generated_bundle else None,
            "generated_edges": [e.id for e in generated_edges],
        }
