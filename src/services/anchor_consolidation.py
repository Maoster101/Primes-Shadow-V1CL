"""Anchor consolidation — prune leaf anchors, keep concepts.

Background:
  Mining produces an anchor for every named concept it spots in the
  source. On a substantial doc this can be 800+ anchors. Many of those
  appear in exactly one slab and have no semantic edges beyond a
  single LINKS edge to that slab — they're effectively highlighted
  phrases inside slab text, duplicated as standalone corpus nodes
  for no good reason. These "leaves" don't earn the runtime
  affordances of a corpus-node (chat-time matching, frame activation,
  edge endpoints, embedding seeds) and they dilute synthesis-time
  embedding discrimination.

Criterion:
  Keep an anchor as a corpus-level node if EITHER:
    (a) Its own LINKS edges touch ≥2 distinct slabs (own reach), OR
    (b) It's a member of a bundle whose own slab-reach is ≥2 (bundle
        reach — concept is cluster-significant even if its raw text
        only surfaces in one slab).
  Otherwise demote: write the anchor's text record into its parent
  slab's ``links.anchors_inline`` list, drop it from corpus.anchors,
  drop the now-redundant LINKS edge.

Edge cases:
  - 0 LINKS edges, 0 bundle membership: orphan. The anchor is
    unreachable from any slab. v1 leaves these in the corpus but
    flags them; deletion is a separate decision (could be a future
    "purge orphans" step).
  - Anchor referenced by an INVOKES / SUPPORTS / CONFLICTS / TENSIONS
    edge from a different node type: KEEP regardless. These are
    semantic-edge endpoints; demoting would orphan the edge.

Reversibility:
  Inline anchors keep their original ``id``. Re-running consolidation
  is idempotent (already-demoted anchors don't appear in
  ``corpus.anchors``, so the analysis re-classifies whatever's left).
  Promotion in reverse is a separate operation — when a second doc
  introduces a slab that references a previously-demoted concept,
  that's a future "rehydrate" pass. Not in scope here.

This module is pure analysis + transform — no I/O, no API. The
endpoint layer wires it to /corpora/{id}/consolidate.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .corpus import CorpusStore
from ..models.schemas import InlineAnchor


# Edge types that COUNT toward keeping an anchor as a corpus node.
# LINKS is the slab→anchor reach signal we count for the cross-slab
# rule. The semantic-edge types (INVOKES, SUPPORTS, CONFLICTS,
# TENSIONS) keep an anchor regardless of slab-reach because demoting
# would break the edge — those edges express concept-level semantics
# that an inline record can't substitute for.
_REACH_EDGE_TYPES = {"LINKS"}
_KEEP_REGARDLESS_EDGE_TYPES = {"INVOKES", "SUPPORTS", "CONFLICTS", "TENSIONS"}


@dataclass
class AnchorVerdict:
    """Per-anchor classification produced by the analysis pass."""
    anchor_id: str
    canonical_phrase: str
    decision: str  # "KEEP" | "DEMOTE" | "ORPHAN"
    reason: str
    own_slab_reach: int = 0          # distinct slabs touched by anchor's LINKS
    bundle_reach: int = 0            # max slab-reach across bundles this anchor is in
    has_semantic_edges: bool = False # any INVOKES/SUPPORTS/CONFLICTS/TENSIONS
    parent_slab_id: Optional[str] = None  # set on DEMOTE — slab the anchor inlines into


@dataclass
class ConsolidationPlan:
    """The plan produced by analyze() — feeds dry-run preview AND apply()."""
    keep: list[AnchorVerdict] = field(default_factory=list)
    demote: list[AnchorVerdict] = field(default_factory=list)
    orphan: list[AnchorVerdict] = field(default_factory=list)
    total_anchors: int = 0
    edges_to_drop: list[str] = field(default_factory=list)  # LINKS-edge IDs that go with demoted anchors

    def summary(self) -> dict:
        """Compact summary suitable for API responses + UI display."""
        return {
            "total_anchors": self.total_anchors,
            "keep_count": len(self.keep),
            "demote_count": len(self.demote),
            "orphan_count": len(self.orphan),
            "edges_to_drop": len(self.edges_to_drop),
            "kept_pct": round(100 * len(self.keep) / max(1, self.total_anchors), 1),
        }


# ── Analysis ──────────────────────────────────────────────────────


def analyze(corpus: CorpusStore) -> ConsolidationPlan:
    """Classify every anchor in the corpus as KEEP / DEMOTE / ORPHAN.

    Pure function — does NOT mutate the corpus. The returned plan
    can be displayed in a dry-run preview before any apply.
    """
    plan = ConsolidationPlan()
    plan.total_anchors = len(corpus.anchors)
    if not corpus.anchors:
        return plan

    # Pre-build the indexes we need to look up reach efficiently.
    #   anchor_id → set of slab IDs reached via LINKS
    #   anchor_id → True if any non-LINKS semantic edge touches it
    #   anchor_id → list of LINKS edge ids (to be dropped on demote)
    anchor_slab_reach: dict[str, set[str]] = defaultdict(set)
    anchor_has_semantic: dict[str, bool] = defaultdict(bool)
    anchor_links_edges: dict[str, list[str]] = defaultdict(list)
    slab_ids = set(corpus.slabs.keys())
    anchor_ids = set(corpus.anchors.keys())

    for edge in corpus.edges.values():
        etype = str(getattr(edge.type, "value", edge.type))
        # Direction-agnostic: we care about whether anchor X is touched
        # by an edge whose other endpoint is a slab.
        for endpoint in (edge.from_node, edge.to_node):
            if endpoint not in anchor_ids:
                continue
            other = edge.to_node if endpoint == edge.from_node else edge.from_node
            if etype in _REACH_EDGE_TYPES and other in slab_ids:
                anchor_slab_reach[endpoint].add(other)
                anchor_links_edges[endpoint].append(edge.id)
            if etype in _KEEP_REGARDLESS_EDGE_TYPES:
                anchor_has_semantic[endpoint] = True

    # Bundle membership — `bundle.payload.intent` carries the bundle's
    # canonical labels but anchors-as-members live in different fields
    # depending on the upstream miner. Convo miner: members in
    # `bundle.aliases` (mining proposal). Coherence bundles (one per
    # slab): `bundle.depends_on` lists the anchor IDs.
    # We treat both as membership signals for reach computation.
    bundle_members: dict[str, set[str]] = defaultdict(set)  # bundle_id → anchor_ids
    bundle_slabs: dict[str, set[str]] = defaultdict(set)    # bundle_id → slabs touched
    for bid, bundle in corpus.bundles.items():
        # depends_on can hold mixed node types — filter to anchors.
        for ref in (bundle.depends_on or []):
            if ref in anchor_ids:
                bundle_members[bid].add(ref)
        # `supports` holds slab IDs — direct slab reach for the bundle.
        for ref in (bundle.supports or []):
            if ref in slab_ids:
                bundle_slabs[bid].add(ref)

    # Compute per-anchor max bundle-reach.
    anchor_bundle_reach: dict[str, int] = defaultdict(int)
    for bid, members in bundle_members.items():
        bslab_count = len(bundle_slabs.get(bid, set()))
        for member in members:
            if bslab_count > anchor_bundle_reach[member]:
                anchor_bundle_reach[member] = bslab_count

    # Classify each anchor.
    for aid, anchor in corpus.anchors.items():
        own_reach = len(anchor_slab_reach.get(aid, set()))
        bundle_reach = anchor_bundle_reach.get(aid, 0)
        has_semantic = anchor_has_semantic.get(aid, False)

        verdict_args = {
            "anchor_id": aid,
            "canonical_phrase": anchor.canonical_phrase,
            "own_slab_reach": own_reach,
            "bundle_reach": bundle_reach,
            "has_semantic_edges": has_semantic,
        }

        if has_semantic:
            plan.keep.append(AnchorVerdict(
                **verdict_args, decision="KEEP",
                reason=f"semantic edge endpoint (INVOKES/SUPPORTS/CONFLICTS/TENSIONS)",
            ))
            continue
        if own_reach >= 2:
            plan.keep.append(AnchorVerdict(
                **verdict_args, decision="KEEP",
                reason=f"own LINKS reach ≥2 ({own_reach} slabs)",
            ))
            continue
        if bundle_reach >= 2:
            plan.keep.append(AnchorVerdict(
                **verdict_args, decision="KEEP",
                reason=f"member of bundle with ≥2 slab reach ({bundle_reach} slabs)",
            ))
            continue

        # Below the bar — demote or orphan.
        if own_reach == 1:
            # Single-slab leaf — demote into that slab.
            parent = next(iter(anchor_slab_reach[aid]))
            plan.demote.append(AnchorVerdict(
                **verdict_args, decision="DEMOTE",
                reason="single-slab leaf (own reach=1, no bundle reach, no semantic edges)",
                parent_slab_id=parent,
            ))
            # Drop ALL LINKS edges touching this anchor — they're
            # redundant once the anchor lives inline in the slab.
            plan.edges_to_drop.extend(anchor_links_edges.get(aid, []))
        else:
            # 0 LINKS, 0 bundle reach, 0 semantic edges — unreachable.
            plan.orphan.append(AnchorVerdict(
                **verdict_args, decision="ORPHAN",
                reason="no slab reach, no bundle reach, no semantic edges",
            ))

    return plan


# ── Apply ─────────────────────────────────────────────────────────


def apply_plan(corpus: CorpusStore, plan: ConsolidationPlan) -> dict:
    """Apply a plan produced by analyze() to the corpus IN PLACE.

    Mutations:
      - For each DEMOTE verdict, append an InlineAnchor record to the
        parent slab's links.anchors_inline list.
      - Remove the demoted anchor from corpus.anchors.
      - Remove all LINKS edges in plan.edges_to_drop from corpus.edges.
      - ORPHAN anchors are LEFT in corpus.anchors with a flag — v1
        does not auto-purge orphans.

    Returns counts for diagnostic reporting.
    """
    demoted = 0
    edges_dropped = 0

    for verdict in plan.demote:
        anchor = corpus.anchors.get(verdict.anchor_id)
        slab = corpus.slabs.get(verdict.parent_slab_id) if verdict.parent_slab_id else None
        if not anchor or not slab:
            # Stale verdict — corpus changed between analyze and apply.
            # Skip rather than crash; caller can re-run analyze.
            continue
        slab.links.anchors_inline.append(InlineAnchor(
            id=verdict.anchor_id,
            canonical_phrase=anchor.canonical_phrase,
            aliases=list(anchor.aliases),
            notes=anchor.notes or "",
            confidence=anchor.match_policy.min_confidence_exact
            if anchor.match_policy else 0.5,
        ))
        corpus.anchors.pop(verdict.anchor_id, None)
        demoted += 1

    for edge_id in plan.edges_to_drop:
        if corpus.edges.pop(edge_id, None) is not None:
            edges_dropped += 1

    return {
        "demoted": demoted,
        "edges_dropped": edges_dropped,
        "remaining_anchors": len(corpus.anchors),
        "remaining_edges": len(corpus.edges),
        "orphans_kept": len(plan.orphan),
    }


# ── Draft-time consolidation (Option B pipeline integration) ────────
#
# The corpus-level analyze + apply_plan above operates on a CorpusStore
# (committed nodes). The pipeline-integrated path runs consolidation
# at MINE TIME, before drafts get pushed to the workbench — so the
# user never sees leaf-anchor noise in the drafts list and the dream
# pass never burns LLM calls on anchors destined for inline demotion.
#
# These helpers operate on the MiningProposal / EdgeProposal lists
# that both convo_miner and narrative_miner produce. They materialize
# a temporary in-memory CorpusStore (using the same label-resolution
# pattern as /push-mined), run the existing analyze() unchanged, and
# return the surviving proposal set + the inline-anchor mapping that
# /push-mined uses to populate slab.links.anchors_inline at draft
# write time.


def _materialize_proposals_to_temp_corpus(proposals, edges):
    """Build an in-memory CorpusStore from proposal/edge lists.

    Uses fuzzy label resolution (label_resolver) to map edge endpoints
    to the temp ids we assign here, mirroring /push-mined's behaviour.
    Bundle membership: narrative_miner stores member anchor phrases in
    ``MiningProposal.aliases``; we resolve those into ``bundle.depends_on``
    so analyze() can count bundle reach correctly.

    Bundle slab-reach for the analyze step: bundles don't have explicit
    "bundle supports slab" edges in mining output. We approximate via
    member-anchor reach — a bundle's slab reach is the union of slabs
    its member anchors LINK to. This is populated into bundle.supports
    so analyze() picks it up via its existing reach logic.

    Returns:
        (temp_corpus, id_to_proposal_index)
        where id_to_proposal_index maps each temp node id back to the
        proposal's position in the input list.
    """
    import uuid
    from .corpus import CorpusStore
    from .label_resolver import build_index, resolve_label
    from ..models.schemas import (
        Anchor, Slab, KeyBundle, Edge, AnchorMatchPolicy, AnchorMeta,
        BundlePayload,
    )
    from ..models.enums import EdgeType, SlabType

    temp = CorpusStore()
    label_entries: list[tuple[str, str]] = []
    id_to_proposal: dict[str, int] = {}
    bundle_member_labels: dict[str, list[str]] = {}  # tmp_bundle_id → member phrases

    for i, p in enumerate(proposals):
        ptype = p.proposal_type
        nid = f"_cons_{ptype}_{i}_{uuid.uuid4().hex[:6]}"
        if ptype == "anchor":
            phrase = (p.canonical_phrase or "").strip()
            if not phrase:
                continue
            temp.anchors[nid] = Anchor(
                id=nid,
                canonical_phrase=phrase,
                aliases=list(p.aliases or []),
                notes=p.justification or "",
                match_policy=AnchorMatchPolicy(),
                meta=AnchorMeta(),
            )
            label_entries.append((phrase, nid))
            for al in p.aliases or []:
                label_entries.append((al, nid))
            id_to_proposal[nid] = i
        elif ptype == "slab":
            text = (p.canonical_text or "").strip()
            if not text:
                continue
            title = (p.title or "").strip() or f"Slab {nid}"
            temp.slabs[nid] = Slab(
                id=nid,
                type=SlabType.REFERENCE,
                title=title,
                canonical_text=text,
                meta=AnchorMeta(),
            )
            label_entries.append((title, nid))
            id_to_proposal[nid] = i
        elif ptype == "bundle":
            label = (p.label or "").strip() or f"Bundle {nid}"
            temp.bundles[nid] = KeyBundle(
                id=nid,
                payload=BundlePayload(intent=[label]),
                meta=AnchorMeta(),
                depends_on=[],
                supports=[],
            )
            label_entries.append((label, nid))
            # narrative_miner stuffs member anchor phrases into aliases.
            # Stash them; we resolve to anchor IDs below once all anchors
            # are materialized + label_to_id is built.
            bundle_member_labels[nid] = list(p.aliases or [])
            id_to_proposal[nid] = i

    label_to_id = build_index(label_entries)

    # Resolve bundle membership (depends_on = list of member anchor IDs)
    for bid, members in bundle_member_labels.items():
        bundle = temp.bundles.get(bid)
        if not bundle:
            continue
        for label in members:
            rid, _ = resolve_label(label, label_to_id)
            if rid and rid in temp.anchors:
                bundle.depends_on.append(rid)

    # Materialize edges with the same fuzzy label resolution that
    # /push-mined uses. Skip unresolvable / self-loop edges.
    for e in edges:
        etype_raw = (e.edge_type or "LINKS").upper()
        if etype_raw not in {
            "INVOKES", "SUPPORTS", "CONFLICTS", "TENSIONS",
            "LINKS", "SEQUENCE", "PARENT_OF",
        }:
            etype_raw = "LINKS"
        from_id, _ = resolve_label((e.from_label or "").strip(), label_to_id)
        to_id, _ = resolve_label((e.to_label or "").strip(), label_to_id)
        if not from_id or not to_id or from_id == to_id:
            continue
        eid = f"_cons_edge_{uuid.uuid4().hex[:8]}"
        try:
            temp.edges[eid] = Edge(
                id=eid,
                type=EdgeType(etype_raw),
                from_node=from_id,
                to_node=to_id,
                weight=float(e.confidence),
                confidence=float(e.confidence),
            )
        except Exception:
            continue

    # Compute bundle.supports = union of slabs reachable from member
    # anchors via LINKS edges. analyze() reads this for bundle reach.
    slab_ids = set(temp.slabs.keys())
    for bundle in temp.bundles.values():
        reach: set[str] = set()
        for member_id in bundle.depends_on:
            for edge in temp.edges.values():
                etype = str(getattr(edge.type, "value", edge.type))
                if etype != "LINKS":
                    continue
                if edge.from_node == member_id and edge.to_node in slab_ids:
                    reach.add(edge.to_node)
                elif edge.to_node == member_id and edge.from_node in slab_ids:
                    reach.add(edge.from_node)
        bundle.supports = list(reach)

    return temp, id_to_proposal


def analyze_proposals(proposals, edges) -> tuple[ConsolidationPlan, dict[str, int]]:
    """Run consolidation analysis on draft proposal/edge lists.

    Returns the plan plus a mapping from temp node IDs back to the
    proposal's position in the input list, so callers can translate
    verdicts into actions on the proposal set.
    """
    temp, id_to_proposal = _materialize_proposals_to_temp_corpus(proposals, edges)
    plan = analyze(temp)
    return plan, id_to_proposal


def apply_to_proposals(proposals, edges, plan, id_to_proposal):
    """Apply a consolidation plan to draft proposal/edge lists.

    Pipeline-integrated counterpart to apply_plan() (which mutates a
    committed CorpusStore). This one filters the in-memory proposal +
    edge lists used by /push-mined and produces the inline-anchor
    mapping needed for slab DraftPackets.

    Returns:
        (surviving_proposals, surviving_edges, inline_map, summary)
        where:
          surviving_proposals: KEEP anchors + all slabs + all bundles
                               (DEMOTE and ORPHAN anchors filtered out)
          surviving_edges:     edges except those touching DEMOTE/ORPHAN
                               anchors (which are now graph-redundant)
          inline_map:          {slab_title: [{id, canonical_phrase,
                                aliases, notes, confidence}, ...]}
                               consumed by /push-mined to populate
                               slab.links.anchors_inline at draft
                               write time. Slab title is the join key
                               because draft-time slab IDs aren't
                               assigned until /push-mined runs.
          summary:             counts for the response body and logs.
    """
    import uuid

    # Map verdict anchor_ids back to proposal indices
    demote_indices: set[int] = set()
    orphan_indices: set[int] = set()
    inline_map: dict[str, list[dict]] = {}

    proposal_by_temp_id = {tid: proposals[idx] for tid, idx in id_to_proposal.items()}

    for verdict in plan.demote:
        anchor_idx = id_to_proposal.get(verdict.anchor_id)
        if anchor_idx is None:
            continue
        demote_indices.add(anchor_idx)
        anchor_p = proposals[anchor_idx]

        # Resolve the parent slab proposal
        parent_p = proposal_by_temp_id.get(verdict.parent_slab_id)
        if not parent_p or parent_p.proposal_type != "slab":
            # Verdict's parent_slab_id didn't resolve to a slab proposal —
            # fall through; the anchor still gets dropped, just no inline
            # record. Shouldn't happen for well-formed verdicts.
            continue

        slab_title = (parent_p.title or "").strip()
        if not slab_title:
            continue
        inline_map.setdefault(slab_title, []).append({
            "id": f"mined_anchor_{uuid.uuid4().hex[:8]}_v1",
            "canonical_phrase": anchor_p.canonical_phrase,
            "aliases": list(anchor_p.aliases or []),
            "notes": anchor_p.justification or "",
            "confidence": float(anchor_p.confidence),
        })

    for verdict in plan.orphan:
        anchor_idx = id_to_proposal.get(verdict.anchor_id)
        if anchor_idx is not None:
            orphan_indices.add(anchor_idx)

    # Filter proposals
    drop_indices = demote_indices | orphan_indices
    surviving_proposals = [p for i, p in enumerate(proposals) if i not in drop_indices]

    # Filter edges referencing dropped anchors. Edges identify endpoints
    # by label, not id, so we drop by canonical_phrase match.
    dropped_phrases: set[str] = set()
    for i in drop_indices:
        p = proposals[i]
        if p.proposal_type == "anchor" and p.canonical_phrase:
            dropped_phrases.add(p.canonical_phrase.strip().lower())
            for alias in (p.aliases or []):
                if alias:
                    dropped_phrases.add(alias.strip().lower())

    surviving_edges = []
    edges_dropped = 0
    for e in edges:
        from_l = (e.from_label or "").strip().lower()
        to_l = (e.to_label or "").strip().lower()
        if from_l in dropped_phrases or to_l in dropped_phrases:
            edges_dropped += 1
            continue
        surviving_edges.append(e)

    summary = {
        "total_anchors": plan.total_anchors,
        "kept": len(plan.keep),
        "demoted": len(plan.demote),
        "orphans_dropped": len(plan.orphan),
        "edges_dropped": edges_dropped,
        "anchors_inlined_into_slabs": sum(len(v) for v in inline_map.values()),
        "slabs_with_inline_anchors": len(inline_map),
    }

    return surviving_proposals, surviving_edges, inline_map, summary
