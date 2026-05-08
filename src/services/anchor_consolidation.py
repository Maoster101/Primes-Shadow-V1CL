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
