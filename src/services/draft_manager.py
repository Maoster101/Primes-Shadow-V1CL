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
import uuid
from typing import Optional
import yaml

from ..models.schemas import (
    DraftPacket, DraftStack, Anchor, Slab, AnchorMeta, AnchorMatchPolicy,
    ProvenanceRef,
)
from ..models.enums import DraftStatus, ClaimTag, DriftSeverity
from ..prompts.proposals import PROPOSAL_EXTRACTION_PROMPT, PROPOSAL_EXPLICIT_PROMPT
from .corpus import CorpusStore
from .session_store import SessionStore
from .verification_router import VerificationRouter
from . import ollama
from .event_log import EventLog

DRAFT_STACK_CAP = 10
MAX_PERIODIC_SLABS = 3
MAX_PERIODIC_ANCHORS = 5
SWEEP_CADENCE = 8  # every N turns
DEDUP_THRESHOLD = 0.9

_event_log = EventLog()


class DraftManager:

    def __init__(self, corpus: CorpusStore, session_store: SessionStore):
        self.corpus = corpus
        self.session_store = session_store
        self.verifier = VerificationRouter()
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

        # Model proposes
        try:
            raw = await ollama.structured_extract(prompt)
            if isinstance(raw, dict):
                proposals = [raw]
            elif isinstance(raw, list):
                proposals = raw
            else:
                return []
        except Exception:
            return []

        if not proposals:
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
                continue

            # Create draft packet
            draft_id = f"tentative_{prop_type}_{uuid.uuid4().hex[:8]}_v1"
            claim_tag = prop.get("claim_tag", "UNKNOWN")

            packet = DraftPacket(
                id=draft_id,
                source_chat_id=chat_id,
                source_turns=prop.get("source_turns", [current_turn]),
                proposed_nodes=[draft_id],
                justification=prop.get("justification", ""),
                confidence=0.75 if explicit else 0.5,
                status=DraftStatus.DRAFT_UNAUTHORIZED,
                fact_claims=[text] if claim_tag == "FACT" else [],
            )

            # Store the raw proposal data alongside the draft packet for later conversion
            self.session_store.save_draft_packet(session_id, packet)
            # Also store raw proposal fields for conversion at review time
            raw_path = self.session_store._drafts_dir(session_id) / f"{draft_id}_raw.json"
            self.session_store._write_json(raw_path, prop)

            stack.packets.append(draft_id)
            if not explicit:
                counts[prop_type] = counts.get(prop_type, 0) + 1
            created.append(packet)

        self.session_store.save_draft_stack(session_id, stack)

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
            result = {"status": "PROVISIONAL", "location": "library/tentative", "draft_id": draft_id}
            if warning:
                result["warning"] = warning
            return result

        if action == "promote_corpus":
            # §26.6: Corpus commits require OLI ON
            if oli_mode != "ON":
                return {"error": "Corpus commits require OLI ON. Toggle OLI before promoting."}

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

            # Convert to corpus object
            obj = self._convert_to_corpus_object(draft_id, raw)
            if not obj:
                return {"error": "Could not convert proposal to corpus object"}

            # Add to corpus, validate, save or rollback
            obj_type, corpus_obj = obj
            generated_bundle = None

            if obj_type == "anchor":
                self.corpus.anchors[corpus_obj.id] = corpus_obj
            elif obj_type == "slab":
                self.corpus.slabs[corpus_obj.id] = corpus_obj
                # §15.1 + §4.1.3: Auto-generate minimum coherence bundle for slab
                generated_bundle = self._generate_minimum_bundle(corpus_obj.id, raw)
                if generated_bundle:
                    self.corpus.bundles[generated_bundle.id] = generated_bundle

            errors = self.corpus.validate()
            if errors:
                # Rollback everything
                if obj_type == "anchor":
                    self.corpus.anchors.pop(corpus_obj.id, None)
                elif obj_type == "slab":
                    self.corpus.slabs.pop(corpus_obj.id, None)
                if generated_bundle:
                    self.corpus.bundles.pop(generated_bundle.id, None)
                return {"error": "Corpus validation failed", "errors": errors}

            self.corpus.save()
            packet.status = DraftStatus.COMMITTED
            self.session_store.save_draft_packet(session_id, packet)

            result = {"status": "COMMITTED", "draft_id": draft_id, "corpus_id": corpus_obj.id}
            if generated_bundle:
                result["generated_bundle"] = generated_bundle.id
            return result

        return {"error": f"Unknown action: {action}"}

    def _convert_to_corpus_object(self, draft_id: str, raw: dict) -> Optional[tuple]:
        """Convert raw proposal to a typed corpus object."""
        prop_type = raw.get("type", "anchor")

        if prop_type == "anchor":
            anchor = Anchor(
                id=draft_id,
                canonical_phrase=raw.get("canonical_phrase", draft_id),
                aliases=raw.get("aliases", []),
                invokes=[],
                notes=raw.get("justification", ""),
                match_policy=AnchorMatchPolicy(),
                meta=AnchorMeta(version="v1"),
                depends_on=[],
                assumptions=[],
            )
            return ("anchor", anchor)

        if prop_type == "slab":
            from ..models.schemas import SlabLinks
            slab = Slab(
                id=draft_id,
                title=raw.get("canonical_phrase", raw.get("title", "")),
                canonical_text=raw.get("canonical_text", ""),
                links=SlabLinks(),
                version="v1",
                meta=AnchorMeta(version="v1"),
                depends_on=[],
                assumptions=[],
                provenance_refs=[ProvenanceRef(
                    ref_id=f"WB_{draft_id}",
                    store="workbench",
                    type="conversation_segment",
                    note=raw.get("justification", ""),
                )],
            )
            return ("slab", slab)

        return None

    def _generate_minimum_bundle(self, slab_id: str, raw: dict) -> Optional[KeyBundle]:
        """§4.1.3 + §15.1: Generate minimum coherence bundle for a committed slab.

        Bundles are never proposed standalone. They are the minimum structural
        context a slab needs to be coherent in the corpus: intent, key invariants,
        and what must NOT be assumed.

        Only generated if the slab has enough substance to warrant one.
        """
        from ..models.schemas import KeyBundle, BundlePayload, AnchorMeta

        canonical_text = raw.get("canonical_text", "")
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

    async def _dedup_check(self, text: str) -> Optional[str]:
        """Check if text is too similar to an existing corpus object."""
        from . import embeddings

        # Check against anchors
        for anchor in self.corpus.anchors.values():
            sim = await embeddings.cosine_similarity(text, anchor.canonical_phrase)
            if sim > DEDUP_THRESHOLD:
                return anchor.id

        # Check against slabs
        for slab in self.corpus.slabs.values():
            sim = await embeddings.cosine_similarity(text, slab.canonical_text[:200])
            if sim > DEDUP_THRESHOLD:
                return slab.id

        return None

    def get_draft_stack(self, session_id: str) -> Optional[DraftStack]:
        return self.session_store.load_draft_stack(session_id)

    def list_drafts(self, session_id: str) -> list[DraftPacket]:
        return self.session_store.list_draft_packets(session_id)
