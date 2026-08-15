"""Mining endpoints — conversation and narrative mining into corpus proposals."""
from __future__ import annotations
import asyncio
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..models.schemas import DraftPacket, DraftStack, ProposedEdge
from ..models.enums import DraftStatus, EdgeType
from ..services.convo_miner import ConversationMiner
from ..services.narrative_miner import NarrativeMiner
from ..services.outline_miner import OutlineMiner
from ..services.paper_miner import PaperMiner

from . import deps
from .deps import (
    frame_manager, session_store, draft_manager,
    chat_store, registry, event_log, rebind_corpus,
)

router = APIRouter()


# --- Request models ---

class MineRequest(BaseModel):
    text: str                          # Raw conversation export content
    source_label: str = "import"       # Provenance label
    min_confidence: float = 0.4        # Minimum proposal confidence
    chunk_size: int = 4                # Exchange pairs per chunk
    target_collection: str = "default" # Which collection to mine into


class MineNarrativeRequest(BaseModel):
    text: str                          # Cohesive narrative / document content
    source_label: str = "narrative"    # Provenance label
    min_confidence: float = 0.4        # Minimum proposal confidence
    max_segment_chars: int = 1800      # Soft cap on segment size
    target_collection: str = "default" # Which collection to mine into


class MineOutlineRequest(BaseModel):
    text: str                          # Structured document (paper, design doc)
    source_label: str = "document"     # Provenance label
    min_confidence: float = 0.4        # Minimum proposal confidence
    max_segment_chars: int = 6000      # Soft cap on per-chapter span size
    target_collection: str = "default" # Which collection to mine into
    recursive: bool = False            # N-tier hierarchy via conservative density audit


class MinePaperRequest(BaseModel):
    text: str                          # Research paper (extracted PDF, etc.)
    source_label: str = "paper"        # Provenance label
    min_confidence: float = 0.4        # Minimum proposal confidence
    target_collection: str = "default" # Which collection to mine into


class BuildOutlinePillarsRequest(BaseModel):
    target_collection: str = "default"  # Collection whose slabs were committed
    outline: list[dict] = []            # `outline` field from /mine-outline
    origin: str = ""                    # Source document label for provenance
    # Rebuild mode: when set, the outline is reconstructed server-side
    # from this session's committed draft sidecars (source_topic +
    # cross_pillars), so the build survives a page refresh and doesn't
    # need the browser to still hold the mine result. Preferred path.
    session_id: str = ""
    max_depth: int = 0                 # 0 = full tree; 1 = coarse source-level overlay
    # Content-clustering overlay: ignore model source_paths and derive the
    # coarse overlay by embedding + k-means-clustering the committed slabs.
    # Robust when a weak extraction model fragments source_path into ~1
    # pillar/slab. target_k=0 auto-picks ~sqrt(N).
    cluster: bool = False
    target_k: int = 0


class PushMinedRequest(BaseModel):
    proposals: list[dict]           # Raw mined proposal dicts from /mine response
    edges: list[dict] = []          # Raw mined edge dicts from /mine response (Phase 3)
    target_collection: str = "default"  # Which collection to commit into
    # Pipeline-integrated consolidation handshake. Both miners emit
    # ``_inline_anchors_per_slab`` (slab-title → list of inline anchor
    # records) when they consolidate at mine time. Forwarded by the
    # frontend untouched. Empty dict = no consolidation ran (or it ran
    # and there were no demotions); slabs get empty anchors_inline.
    inline_anchors_per_slab: dict[str, list[dict]] = {}


class ScratchSessionRequest(BaseModel):
    title: str = "Mining Session"
    collection_id: str = "default"


# --- Endpoints ---

@router.post("/mine")
async def mine_conversation(req: MineRequest):
    """Mine a conversation export for corpus proposals.

    Accepts raw text (Claude JSON, ChatGPT JSON, markdown, or plaintext).
    Returns detected format, topic distribution, and proposed corpus objects.
    """
    if not req.text.strip():
        raise HTTPException(400, "Empty text")
    if len(req.text) > 5_000_000:
        raise HTTPException(413, "Text too large (max 5MB)")

    # Use target collection's store for existing-anchor dedup
    target_store = registry.get_store(req.target_collection) or deps.corpus
    miner = ConversationMiner(target_store)

    result = await miner.mine(
        raw_text=req.text,
        source_label=req.source_label,
        min_confidence=req.min_confidence,
        chunk_size=req.chunk_size,
    )
    result["target_collection"] = req.target_collection
    return result


@router.post("/mine-narrative")
async def mine_narrative(req: MineNarrativeRequest):
    """Mine a cohesive narrative / document for corpus proposals.

    Treats the input as a single-author piece where paragraph ordering is
    load-bearing (stories, essays, design docs, condensed pitches). Emits
    SEQUENCE edges between consecutive beats so the narrative spine is
    preserved in the corpus graph. Output shape matches /mine so the same
    /push-mined flow accepts its proposals.
    """
    if not req.text.strip():
        raise HTTPException(400, "Empty text")
    if len(req.text) > 5_000_000:
        raise HTTPException(413, "Text too large (max 5MB)")

    target_store = registry.get_store(req.target_collection) or deps.corpus
    miner = NarrativeMiner(target_store)

    result = await miner.mine(
        raw_text=req.text,
        source_label=req.source_label,
        min_confidence=req.min_confidence,
        max_segment_chars=req.max_segment_chars,
    )
    result["target_collection"] = req.target_collection
    return result


@router.post("/mine-outline")
async def mine_outline(req: MineOutlineRequest):
    """Mine a structured document (paper, design doc) outline-first.

    Parses the document's heading tree (or, for headingless prose,
    synthesizes an ordered outline via one LLM call), then drills each
    section span into slabs + anchors. The response carries an
    ``outline`` field — the ordered Part/chapter skeleton — which is the
    pillar overlay by construction. Proposals match the /mine shape so
    the same /push-mined flow accepts them.
    """
    if not req.text.strip():
        raise HTTPException(400, "Empty text")
    if len(req.text) > 5_000_000:
        raise HTTPException(413, "Text too large (max 5MB)")

    target_store = registry.get_store(req.target_collection) or deps.corpus
    miner = OutlineMiner(target_store)

    result = await miner.mine(
        raw_text=req.text,
        source_label=req.source_label,
        min_confidence=req.min_confidence,
        max_segment_chars=req.max_segment_chars,
        recursive=req.recursive,
    )
    result["target_collection"] = req.target_collection
    return result


@router.post("/mine-paper")
async def mine_paper(req: MinePaperRequest):
    """Mine a research paper — recursive hierarchy + semantic-chunk slabs.

    Pass 1 identifies top-level paper sections; Passes 2..N do a batched
    density audit recursively (sub-pillars emerge only where content
    warrants), halting when no leaf wants further subdivision. Stage B
    drills each LEAF into atomic semantic units via move-recognition.
    The ``outline`` field is the N-tier pillar skeleton; the build
    counterpart is ``build_pillars_recursive`` (auto-dispatched by
    /build-outline-pillars when N-tier is detected).
    """
    if not req.text.strip():
        raise HTTPException(400, "Empty text")
    if len(req.text) > 5_000_000:
        raise HTTPException(413, "Text too large (max 5MB)")

    target_store = registry.get_store(req.target_collection) or deps.corpus
    miner = PaperMiner(target_store)

    result = await miner.mine(
        raw_text=req.text,
        source_label=req.source_label,
        min_confidence=req.min_confidence,
    )
    result["target_collection"] = req.target_collection
    return result


@router.post("/build-outline-pillars")
async def build_outline_pillars(req: BuildOutlinePillarsRequest):
    """Write the pillar overlay from a mined outline tree — the last mile.

    Call AFTER the outline miner's slabs have been pushed and committed
    into the target collection. The outline tree (the ``outline`` field
    of /mine-outline) is the Part/chapter skeleton; this resolves each
    chapter's slab titles to the committed slab IDs and writes
    PillarDefinitions — with Pass-3 summaries and Pass-4 cross-edges — to
    the collection's pillars.yaml. That overlay is what the corpus view
    renders at top zoom.

    Separate from /push-mined because pillars reference *committed* slab
    IDs: the overlay can only be built once the content nodes exist.
    """
    store = registry.get_store(req.target_collection) or deps.corpus
    from ..services.outline_miner import (
        build_pillars_from_outline, build_pillars_from_session,
        build_pillars_from_collection, build_pillars_recursive,
        build_pillars_by_clustering,
    )

    # Content-clustering overlay short-circuits the path-based builders — it
    # derives pillars from slab embeddings, so it needs neither an outline
    # tree nor session sidecars, only the committed slabs in the store.
    if req.cluster:
        report = await build_pillars_by_clustering(
            store, origin=req.origin or req.target_collection,
            target_k=max(0, req.target_k),
        )
        report["target_collection"] = req.target_collection
        return report

    def _is_n_tier(tree: list[dict]) -> bool:
        """True if any node has children that themselves have children —
        i.e. the tree is more than 2 tiers deep, which means it came
        from the paper miner. The doc/narrative miners produce at most
        2 tiers (a top with chapter children); a paper-mined tree can
        have arbitrary depth and needs the recursive builder."""
        for top in tree:
            for child in top.get("children", []):
                if "children" in child:
                    return True
        return False

    origin = req.origin or req.target_collection
    if req.session_id:
        # Rebuild from one mining session's committed draft sidecars.
        report = await build_pillars_from_session(
            store, session_store, req.session_id, origin=origin,
            max_depth=max(0, req.max_depth),
        )
    elif req.outline:
        # Fresh mine result handed straight from the browser. Dispatch
        # to the N-tier recursive builder for paper-mined outlines;
        # 2-tier doc/narrative outlines stay on the original path.
        if _is_n_tier(req.outline):
            report = build_pillars_recursive(store, req.outline, origin=origin)
        else:
            report = build_pillars_from_outline(store, req.outline, origin=origin)
    else:
        # Fully stateless: scan every session's drafts for slabs that
        # committed into this collection. The corpus-view button's path.
        report = await build_pillars_from_collection(
            store, session_store, origin=origin,
            max_depth=max(0, req.max_depth),
        )
    report["target_collection"] = req.target_collection
    return report


class DedupCorpusRequest(BaseModel):
    target_collection: str = "default"
    rebuild_pillars: bool = True   # rebuild the overlay after (dedup voids it)


@router.post("/corpus/dedup")
async def dedup_corpus(req: DedupCorpusRequest):
    """Collapse exact-duplicate slabs/anchors in a collection to one each.

    Exact-content dedup — slabs by canonical_text, anchors by
    canonical_phrase. The cleanup for an accidental double-push, where
    the same mine result was committed twice and every node landed 2x.
    Edges are remapped onto the survivors and de-duplicated; the pillar
    overlay is rebuilt afterward, since dedup removes slabs it referenced.
    """
    store = registry.get_store(req.target_collection) or deps.corpus
    report = store.deduplicate()
    if report.get("remapped", 0) == 0:
        report["status"] = "already clean — nothing to dedup"
        report["target_collection"] = req.target_collection
        return report

    pillar_report = None
    if req.rebuild_pillars:
        from ..services.outline_miner import build_pillars_from_collection
        pillar_report = await build_pillars_from_collection(
            store, session_store, origin=req.target_collection,
        )

    if not pillar_report or pillar_report.get("pillars_created", 0) == 0:
        # No overlay rebuilt — a stale one would now dangle on removed
        # slabs, so drop it. Then persist the deduped corpus ourselves
        # (build_pillars_from_collection saves only when it builds).
        store.pillars = {}
        report["validation_errors"] = store.validate()
        store.save()
    else:
        report["validation_errors"] = pillar_report.get("validation_errors", [])
    report["pillars"] = pillar_report
    report["target_collection"] = req.target_collection
    return report


@router.get("/mining-progress")
async def get_mining_progress():
    """Snapshot of the current mine's progress.

    Singleton: only tracks one mine at a time. Frontend polls this
    while a mine is in flight to render a progress bar with phase +
    counts + ETA. See services/mining_progress.py for the state
    shape and per-phase weighting.
    """
    from ..services import mining_progress
    return mining_progress.get()


@router.get("/dreaming-progress")
async def get_dreaming_progress():
    """Snapshot of the current dreaming batch's progress.

    Singleton, mirrors /mining-progress. The Dream page polls this while a
    dream_all_pending run is in flight to render a live bar that increments
    per draft (completed/total + elapsed + the draft in flight).
    """
    from ..services import dreaming_progress
    return dreaming_progress.get()


@router.post("/sessions/{session_id}/push-mined")
async def push_mined_proposals(session_id: str, req: PushMinedRequest):
    """Push mined proposals directly into the session draft stack.

    Converts raw mined proposals (from /mine) into DraftPackets without
    re-running LLM extraction. Each proposal becomes a DRAFT_UNAUTHORIZED
    packet with its raw data preserved for later review/promotion.
    """
    meta_path = session_store.session_dir(session_id) / "meta.json"
    meta = session_store.read_json(meta_path)
    if not meta:
        raise HTTPException(404, "Session not found")

    stack = session_store.load_draft_stack(session_id)
    if not stack:
        stack = DraftStack(session_id=session_id)

    created = []
    created_packets = []
    # AI Mine 2 stamps server-recoverable source order. Keep selected slab
    # draft IDs so the backend can build an exact sequence spine after any
    # user filtering, rather than trusting model-generated SEQUENCE edges.
    ordered_slab_drafts: list[tuple[int, str, str]] = []
    for prop in req.proposals:
        if not isinstance(prop, dict):
            continue
        prop_type = prop.get("type", "anchor")
        # Bundle proposals carry their handle in `label`, not the
        # other text fields. Adding it here was a pre-existing bug:
        # without it, bundles slip through the empty-text guard
        # below and never become DraftPackets — so Pass-2 thematic
        # bundles silently fail to reach the workbench, and the
        # corpus only ever holds the auto-generated coherence
        # bundles (now removed). See draft_manager.extract_proposals
        # for the canonical bundle-handling shape we mirror here.
        text = (
            prop.get("canonical_phrase")
            or prop.get("canonical_text", "")
            or prop.get("title", "")
            or prop.get("label", "")
        )
        if not text:
            continue

        draft_id = f"mined_{prop_type}_{uuid.uuid4().hex[:8]}_v1"

        # Build inline typed payload so the packet is self-contained
        # (same pattern as DraftManager.extract_proposals).
        inline_anchor = None
        inline_slab = None
        inline_bundle = None
        if prop_type == "anchor":
            inline_anchor = {
                "id": draft_id,
                "canonical_phrase": prop.get("canonical_phrase", "") or "",
                "aliases": list(prop.get("aliases") or []),
                "invokes": [],
                "notes": prop.get("justification", "") or "",
            }
        elif prop_type == "slab":
            slab_title = prop.get("title") or prop.get("canonical_phrase") or ""
            # Pipeline-integrated consolidation: any anchor proposal
            # demoted at mine time that resolved to THIS slab as its
            # parent gets attached as an inline record. The slab title
            # is the join key — both sides agreed on the slab's title
            # at mine time. Falls back to empty when no consolidation
            # happened upstream.
            inline_anchors_for_slab = req.inline_anchors_per_slab.get(
                slab_title.strip(), []
            ) if slab_title else []
            inline_slab = {
                "id": draft_id,
                "title": slab_title,
                "canonical_text": prop.get("canonical_text", "") or "",
                # Retrieval-oriented summary (universal miner). Carried on
                # the packet so promote can persist it to the Slab; empty
                # for legacy miners that don't emit it.
                "description": prop.get("description", "") or "",
                "links": {
                    "anchors": [],
                    "bundles": [],
                    "anchors_inline": inline_anchors_for_slab,
                },
                "version": "v1",
            }
        elif prop_type == "bundle":
            # Bundles are produced by Pass 2 (narrative miner clustering)
            # and by convo_miner's main extraction pass. Their handle is
            # `label`; their members live in `aliases` as a list of
            # member anchor canonical_phrases. The DraftPacket-level
            # representation maps these to a BundlePayload-shaped dict
            # so promote-time conversion lands cleanly as a KeyBundle.
            label = (prop.get("label") or "").strip()
            members = list(prop.get("aliases") or [])
            justification = (prop.get("justification") or "").strip()
            # BundlePayload requires a non-empty intent list. Fall through
            # ladder: justification → label → draft_id.
            intent = (
                [justification] if justification
                else [label] if label
                else [draft_id]
            )
            inline_bundle = {
                "id": draft_id,
                "payload": {
                    "intent": intent[:3],
                    "invariants": [],
                    "non_assumptions": [],
                    "warnings": [],
                    "heuristics": [],
                    "markers": [],
                    "rules": [],
                    "activation_clause": [],
                    "canonical_quote_handles": [],
                },
                "version": "v1",
                # Member anchor phrases stashed for the promote step;
                # they get resolved to anchor IDs and emitted as
                # INVOKES edges in the bundle→slab causality refactor.
                # For now, just preserve the membership info.
                "_member_phrases": members,
            }

        packet = DraftPacket(
            id=draft_id,
            packet_type=prop_type,
            source_chat_id=meta.get("chat_id", session_id),
            source_turns=[0],
            proposed_nodes=[draft_id],
            justification=prop.get("justification", ""),
            confidence=prop.get("confidence", 0.5),
            status=DraftStatus.DRAFT_UNAUTHORIZED,
            fact_claims=[text] if prop.get("claim_tag") == "FACT" else [],
            anchor=inline_anchor,
            slab=inline_slab,
            bundle=inline_bundle,
        )

        session_store.save_draft_packet(session_id, packet)
        # Store raw proposal + target collection for commit step
        prop["_target_collection"] = req.target_collection
        raw_path = session_store.drafts_dir(session_id) / f"{draft_id}_raw.json"
        session_store.write_json(raw_path, prop)

        stack.packets.append(draft_id)
        created.append({"id": draft_id, "type": prop_type, "label": text[:80]})
        created_packets.append(packet)
        if prop_type == "slab" and prop.get("_source_order") is not None:
            try:
                ordered_slab_drafts.append((
                    int(prop["_source_order"]), draft_id,
                    str(prop.get("title") or prop.get("canonical_phrase") or draft_id),
                ))
            except (TypeError, ValueError):
                pass

    session_store.save_draft_stack(session_id, stack)

    # Phase 3 — resolve & persist mined edges into the ProposedEdge pipeline.
    # Convo/narrative miners emit edges with keys {type, from, to, confidence,
    # justification}. We reuse the same label index strategy as the chat path:
    # just-pushed drafts (canonical_phrase / title / aliases) take priority,
    # then live corpus anchors/slabs.
    edges_persisted = 0
    if req.edges:
        try:
            from ..services.label_resolver import (
                build_index, resolve_label,
            )

            # Build (label, id) pairs for all known endpoints, then index
            # them. Pairs are emitted in priority order — just-pushed
            # drafts first (so newly-mined endpoints win on collision),
            # then existing corpus anchors / slabs / bundles.
            entries: list[tuple[str, str]] = []
            for pkt in created_packets:
                if pkt.anchor:
                    entries.append((pkt.anchor.get("canonical_phrase", "") or "", pkt.id))
                    for al in pkt.anchor.get("aliases") or []:
                        entries.append((al, pkt.id))
                if pkt.slab:
                    entries.append((pkt.slab.get("title", "") or "", pkt.id))
                if pkt.bundle:
                    entries.append((pkt.bundle.get("id", "") or "", pkt.id))

            # Existing corpus endpoints (use target collection's store
            # for consistency with how dedup ran during mining).
            tgt_store = registry.get_store(req.target_collection) or deps.corpus
            for a in tgt_store.anchors.values():
                entries.append((a.canonical_phrase, a.id))
                for al in a.aliases or []:
                    entries.append((al, a.id))
            for s in tgt_store.slabs.values():
                if getattr(s, "title", ""):
                    entries.append((s.title, s.id))

            label_to_id = build_index(entries)

            # Track fuzzy-match attribution for diagnostics — surface
            # which strategy resolved each edge so we can spot whether
            # the corpus is drifting toward "exact almost never works".
            fuzzy_hits = {"exact": 0, "substring": 0, "ratio": 0, "miss": 0}

            resolved: list = []
            chat_id_for_edges = meta.get("chat_id", session_id)
            for spec in req.edges:
                if not isinstance(spec, dict):
                    continue
                etype_raw = (spec.get("type") or "LINKS").upper()
                if etype_raw == "SEQUENCE" and spec.get("deterministic"):
                    continue  # rebuilt below from the actually selected slabs
                if etype_raw not in {"INVOKES", "SUPPORTS", "CONFLICTS", "TENSIONS", "LINKS", "SEQUENCE", "PARENT_OF"}:
                    etype_raw = "LINKS"
                # Convo miner emits from/to; chat path emits from_label/to_label.
                from_label = (spec.get("from_label") or spec.get("from") or "").strip()
                to_label = (spec.get("to_label") or spec.get("to") or "").strip()
                if not from_label or not to_label:
                    continue
                from_id, from_strategy = resolve_label(from_label, label_to_id)
                to_id, to_strategy = resolve_label(to_label, label_to_id)
                fuzzy_hits[from_strategy] = fuzzy_hits.get(from_strategy, 0) + 1
                fuzzy_hits[to_strategy] = fuzzy_hits.get(to_strategy, 0) + 1
                if not from_id or not to_id or from_id == to_id:
                    if not from_id:
                        print(f"[PUSH-MINED] Edge dropped: from_label={from_label!r} not in label index ({len(label_to_id)} entries)", flush=True)
                    if not to_id:
                        print(f"[PUSH-MINED] Edge dropped: to_label={to_label!r} not in label index", flush=True)
                    continue
                # Log when fuzzy strategies fired — useful for spotting
                # systematic label-emission issues over time.
                if from_strategy != "exact" and from_strategy != "miss":
                    print(f"[PUSH-MINED] Fuzzy resolved from={from_label!r} via {from_strategy}", flush=True)
                if to_strategy != "exact" and to_strategy != "miss":
                    print(f"[PUSH-MINED] Fuzzy resolved to={to_label!r} via {to_strategy}", flush=True)
                try:
                    confidence = float(spec.get("confidence", 0.5))
                except Exception:
                    confidence = 0.5
                resolved.append(ProposedEdge(
                    id=f"proposed_edge_{uuid.uuid4().hex[:8]}",
                    type=EdgeType(etype_raw),
                    from_node=from_id, to_node=to_id,
                    from_label=from_label, to_label=to_label,
                    confidence=max(0.0, min(1.0, confidence)),
                    justification=spec.get("justification", "") or "",
                    status="PROPOSED",
                    source_chat_id=chat_id_for_edges,
                    source_turn=0,
                ))
            if resolved:
                session_store.append_proposed_edges(session_id, resolved)
                edges_persisted = len(resolved)
            # Aggregate diagnostic — useful for spotting systematic
            # label-emission drift if the model starts producing labels
            # that exact-match never works on. Prints once per push.
            print(
                f"[PUSH-MINED] Edge label resolution: {fuzzy_hits} "
                f"({edges_persisted}/{len(req.edges)} edges persisted)",
                flush=True,
            )
        except Exception as e:
            print(f"[PUSH-MINED] Edge resolution failed: {e}", flush=True)

    # Deterministic document-order edges are structural facts, so they enter
    # ACCEPTED state immediately. The lifecycle commits each one once both
    # selected endpoint drafts have been promoted into the corpus.
    if len(ordered_slab_drafts) > 1:
        ordered_slab_drafts.sort(key=lambda row: row[0])
        sequence_edges = []
        chat_id_for_edges = meta.get("chat_id", session_id)
        for (_a_order, a_id, a_label), (_b_order, b_id, b_label) in zip(
            ordered_slab_drafts, ordered_slab_drafts[1:]
        ):
            sequence_edges.append(ProposedEdge(
                id=f"proposed_edge_{uuid.uuid4().hex[:8]}",
                type=EdgeType.SEQUENCE,
                from_node=a_id, to_node=b_id,
                from_label=a_label, to_label=b_label,
                confidence=0.98,
                justification="Deterministic order of consecutive selected slabs in the source.",
                status="ACCEPTED",
                source_chat_id=chat_id_for_edges,
                source_turn=0,
            ))
        session_store.append_proposed_edges(session_id, sequence_edges)
        edges_persisted += len(sequence_edges)
    # Dreaming is OPT-IN, not auto-triggered on push. Auto-dreaming every
    # pushed draft was fine for small chat pushes (~5 drafts) but a bulk
    # AI-Mine push of 100+ slabs fans out to hundreds of background LLM
    # calls at PS_DREAMING_PARALLEL=2, monopolizing the local Ollama slots
    # and starving chat queries / pillar rebuilds (ReadTimeouts). The Dream
    # page is now the deterministic surface: the reviewer triggers
    # dream_all_pending there when they choose, with a live progress bar.
    # (Chat-driven pushes still dream via their own end-of-turn path.)

    # --- Inject mined proposals into frame_manager's tentative registry ---
    # Without this, /sessions/{id}/graph can't see the mined nodes because
    # it reads from the in-memory registry, not from disk DraftPackets.
    # Mining intentionally uses a base-set-free frame (ensure_empty_frame
    # inside the verb) so proposals aren't polluted with pre-loaded slabs.
    if created_packets:
        frame_manager.inject_mined_proposals(
            session_id, meta.get("chat_id", session_id), created_packets
        )

    return {"created": len(created), "drafts": created, "edges_persisted": edges_persisted}


@router.post("/sessions/scratch")
async def create_scratch_session(req: ScratchSessionRequest):
    """Create a lightweight chat + session without running inference.

    Used by the mining UI to get a session_id instantly — no SSE stream,
    no model warm-up, no dummy turn. The chat is created in ACTIVE state
    with an empty message history, and a session is bound to it immediately.
    """
    chat = chat_store.create_chat(req.title, collection_id=req.collection_id)
    session_id = session_store.create_session(chat.id)
    return {
        "chat_id": chat.id,
        "session_id": session_id,
        "collection_id": req.collection_id,
    }
