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
from ..prompts.proposals import PROPOSAL_EXTRACTION_PROMPT, PROPOSAL_EXPLICIT_PROMPT
from .corpus import CorpusStore
from .session_store import SessionStore
from .verification_router import VerificationRouter
from . import ollama
from .event_log import EventLog

DRAFT_STACK_CAP = 10
MAX_PERIODIC_SLABS = 3
MAX_PERIODIC_ANCHORS = 5
SWEEP_CADENCE = 1  # every turn — aggressive extraction for interactive use
DEDUP_THRESHOLD = 0.75  # lower threshold catches more near-duplicates

_event_log = EventLog()


class DraftManager:

    def __init__(self, corpus: CorpusStore, session_store: SessionStore):
        self.corpus = corpus
        self.session_store = session_store
        self.verifier = VerificationRouter()
        self._registry = None  # Set by routes.py after registry init
        # Track periodic proposal counts per session
        self._periodic_counts: dict[str, dict[str, int]] = {}

    def _get_counts(self, session_id: str) -> dict[str, int]:
        if session_id not in self._periodic_counts:
            self._periodic_counts[session_id] = {"slab": 0, "anchor": 0}
        return self._periodic_counts[session_id]

    def should_sweep(self, turn: int) -> bool:
        """Check if periodic sweep should fire this turn."""
        return turn > 0 and turn % SWEEP_CADENCE == 0

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

        # Build existing anchor list for dedup in prompt
        existing = ", ".join(
            a.canonical_phrase for a in self.corpus.anchors.values()
        )

        # Choose prompt
        if explicit:
            prompt = (
                PROPOSAL_EXPLICIT_PROMPT
                .replace("$CONVERSATION", conversation)
                + json.dumps(user_request)
            )
        else:
            prompt = (
                PROPOSAL_EXTRACTION_PROMPT
                .replace("$EXISTING_ANCHORS", existing or "(none)")
                .replace("$CONVERSATION", conversation)
            )

        # Model proposes. Phase 3: output is now a wrapper object
        #   {"proposals": [...], "edges": [...]}
        # but we keep backward compat with the legacy bare-array / bare-object
        # shapes in case the model regresses.
        try:
            raw = await ollama.structured_extract(prompt)
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
            raw_path = self.session_store._drafts_dir(session_id) / f"{draft_id}_raw.json"
            prop_to_save = dict(prop)
            if collection_id:
                prop_to_save["_target_collection"] = collection_id
            self.session_store._write_json(raw_path, prop_to_save)

            stack.packets.append(draft_id)
            if not explicit:
                counts[prop_type] = counts.get(prop_type, 0) + 1
            created.append(packet)

        self.session_store.save_draft_stack(session_id, stack)

        # Phase 3 — resolve proposed edges (labels -> node_ids) and persist.
        # Label index priority order:
        #   1. Just-mined proposals in this turn (canonical_phrase / title / aliases)
        #   2. Live corpus anchors (canonical_phrase / aliases)
        #   3. Live corpus slabs (title)
        # Endpoints that can't be resolved from any of these are skipped with
        # a debug log — we never invent a node_id for an unresolvable label.
        if proposed_edge_specs:
            try:
                from ..models.schemas import ProposedEdge

                label_to_id: dict[str, str] = {}

                def _register(label: str, nid: str) -> None:
                    if not label:
                        return
                    key = label.strip().lower()
                    # first write wins — earlier sources are more specific
                    label_to_id.setdefault(key, nid)

                # 1. Just-mined proposals
                for pkt in created:
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

                # 3. Corpus slabs (title is the human-facing label)
                for s in self.corpus.slabs.values():
                    if getattr(s, "title", ""):
                        _register(s.title, s.id)

                resolved_edges: list = []
                for spec in proposed_edge_specs:
                    if not isinstance(spec, dict):
                        continue
                    etype_raw = (spec.get("type") or "LINKS").upper()
                    if etype_raw not in {"INVOKES", "SUPPORTS", "CONFLICTS", "LINKS", "SEQUENCE", "PARENT_OF"}:
                        etype_raw = "LINKS"
                    from_label = (spec.get("from_label") or "").strip()
                    to_label = (spec.get("to_label") or "").strip()
                    if not from_label or not to_label:
                        continue
                    from_id = label_to_id.get(from_label.lower())
                    to_id = label_to_id.get(to_label.lower())
                    if not from_id or not to_id:
                        print(f"[DRAFT]   skip edge (unresolved): {from_label!r} -> {to_label!r}", flush=True)
                        continue
                    if from_id == to_id:
                        continue  # no self-loops
                    try:
                        confidence = float(spec.get("confidence", 0.5))
                    except Exception:
                        confidence = 0.5
                    resolved_edges.append(ProposedEdge(
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

                if resolved_edges:
                    self.session_store.append_proposed_edges(session_id, resolved_edges)
                    print(f"[DRAFT] Persisted {len(resolved_edges)} proposed edge(s)", flush=True)
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
    ) -> dict:
        """Review a draft: discard / promote_tentative / promote_corpus.

        §26.6: corpus commits require OLI ON. FACT verification mandatory.
        Drift gate: blocks corpus promotion when drift severity is HIGH.
        """
        packet = self.session_store.load_draft_packet(session_id, draft_id)
        if not packet:
            return {"error": "Draft not found"}

        raw_path = self.session_store._drafts_dir(session_id) / f"{draft_id}_raw.json"
        raw = self.session_store._read_json(raw_path) or {}

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
            self.session_store._write_json(tent_dir / f"{draft_id}.json", {
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
            generated_bundle = None

            if obj_type == "anchor":
                target_store.anchors[corpus_obj.id] = corpus_obj
                # Also add to merged view so validation sees it
                if target_store is not self.corpus:
                    self.corpus.anchors[corpus_obj.id] = corpus_obj
            elif obj_type == "slab":
                target_store.slabs[corpus_obj.id] = corpus_obj
                if target_store is not self.corpus:
                    self.corpus.slabs[corpus_obj.id] = corpus_obj
                # §15.1 + §4.1.3: Auto-generate minimum coherence bundle for slab.
                # Also packet-aware: if the slab was enriched, the bundle is
                # generated from the enriched canonical_text, not the raw.
                generated_bundle = self._generate_minimum_bundle(
                    corpus_obj.id, raw, packet=packet
                )
                if generated_bundle:
                    target_store.bundles[generated_bundle.id] = generated_bundle
                    if target_store is not self.corpus:
                        self.corpus.bundles[generated_bundle.id] = generated_bundle

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

            errors = target_store.validate()
            if errors:
                # Rollback everything: nodes, bundle, AND edges
                if obj_type == "anchor":
                    target_store.anchors.pop(corpus_obj.id, None)
                    self.corpus.anchors.pop(corpus_obj.id, None)
                elif obj_type == "slab":
                    target_store.slabs.pop(corpus_obj.id, None)
                    self.corpus.slabs.pop(corpus_obj.id, None)
                if generated_bundle:
                    target_store.bundles.pop(generated_bundle.id, None)
                    self.corpus.bundles.pop(generated_bundle.id, None)
                for edge in generated_edges:
                    target_store.edges.pop(edge.id, None)
                    self.corpus.edges.pop(edge.id, None)
                return {"error": "Corpus validation failed", "errors": errors}

            target_store.save()
            packet.status = DraftStatus.COMMITTED
            self.session_store.save_draft_packet(session_id, packet)
            self._mark_resolved(session_id, draft_id)

            result = {"status": "COMMITTED", "draft_id": draft_id, "corpus_id": corpus_obj.id}
            if generated_bundle:
                result["generated_bundle"] = generated_bundle.id
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

        return None

    def _generate_minimum_bundle(
        self,
        slab_id: str,
        raw: dict,
        packet: Optional[DraftPacket] = None,
    ) -> Optional[KeyBundle]:
        """§4.1.3 + §15.1: Generate minimum coherence bundle for a committed slab.

        Bundles are never proposed standalone. They are the minimum structural
        context a slab needs to be coherent in the corpus: intent, key invariants,
        and what must NOT be assumed.

        Only generated if the slab has enough substance to warrant one.

        Packet-aware: when ``packet`` is provided, prefers the enriched
        inline slab dict over the raw sidecar — so a dreaming-phase rewrite
        that substitutes the confabulated miner text for a grounded version
        will also produce a bundle built from the grounded text, not the
        discarded original. Same fallback ladder as _convert_to_corpus_object.
        """
        from ..models.schemas import KeyBundle, BundlePayload, AnchorMeta

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

        # Only generate if slab has real content
        if len(canonical_text) < 50:
            return None

        bundle_id = slab_id.replace("tentative_slab_", "bundle_").replace("_v1", "_coherence_v1")
        if not bundle_id.endswith("_v1"):
            bundle_id += "_v1"

        # Build minimal payload from available context
        intent = [justification] if justification else [f"Coherence context for {slab_id}"]
        invariants = []
        non_assumptions = []

        # Extract key claims from canonical text as invariants
        sentences = [s.strip() for s in canonical_text.split('.') if len(s.strip()) > 15]
        for s in sentences[:3]:
            invariants.append(s)

        bundle = KeyBundle(
            id=bundle_id,
            payload=BundlePayload(
                intent=intent[:3],
                invariants=invariants[:5],
                non_assumptions=non_assumptions,
            ),
            version="v1",
            meta=AnchorMeta(version="v1"),
            depends_on=[slab_id],
            supports=[slab_id],
        )

        _event_log.log_proposal_event(
            event="bundle_generated_at_commit",
            slab_id=slab_id,
            bundle_id=bundle_id,
        )

        return bundle

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

        # Auto-generated coherence bundle → SUPPORTS edges (w=0.7)
        if generated_bundle:
            for slab_id in generated_bundle.supports or []:
                _maybe_add(EdgeType.SUPPORTS, generated_bundle.id, slab_id, 0.7)

        if edges:
            _event_log.log_proposal_event(
                event="edges_generated_at_commit",
                source_id=corpus_obj.id,
                edge_count=len(edges),
                edge_ids=[e.id for e in edges],
            )

        return edges

    async def _dedup_check(self, text: str) -> Optional[str]:
        """Check if text is too similar to an existing corpus object."""
        from . import embeddings
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

        # Embedding similarity pass
        for anchor in self.corpus.anchors.values():
            sim = await embeddings.cosine_similarity(text, anchor.canonical_phrase)
            if sim > DEDUP_THRESHOLD:
                return anchor.id

        for slab in self.corpus.slabs.values():
            sim = await embeddings.cosine_similarity(text, slab.canonical_text[:200])
            if sim > DEDUP_THRESHOLD:
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
        generated_bundle = None

        if obj_type == "anchor":
            target_store.anchors[corpus_obj.id] = corpus_obj
            if target_store is not self.corpus:
                self.corpus.anchors[corpus_obj.id] = corpus_obj
        elif obj_type == "slab":
            target_store.slabs[corpus_obj.id] = corpus_obj
            if target_store is not self.corpus:
                self.corpus.slabs[corpus_obj.id] = corpus_obj
            generated_bundle = self._generate_minimum_bundle(corpus_obj.id, raw, packet=packet)
            if generated_bundle:
                target_store.bundles[generated_bundle.id] = generated_bundle
                if target_store is not self.corpus:
                    self.corpus.bundles[generated_bundle.id] = generated_bundle

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
            if generated_bundle:
                target_store.bundles.pop(generated_bundle.id, None)
                self.corpus.bundles.pop(generated_bundle.id, None)
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
