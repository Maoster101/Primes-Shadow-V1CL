"""Synthesis — argumentative retrieval over the corpus graph.

The inverse of mining: mining turns text → corpus nodes; synthesis
turns a query → coherent narrative drawn from the corpus.

The selection algorithm is *tiered* by epistemic role rather than
fused into a single relevance score:

  Tier 1 (the floor — always included if budget permits):
    - Supported: HIGH-confidence nodes reachable from seeds via
      structural edges (SUPPORTS, INVOKES, LINKS, PARENT_OF).
      These are the thesis content.
    - Conflicts: any CONFLICTS endpoints touching the seeds or the
      tier-1 supported set, regardless of confidence. Surface
      dialectic on purpose.

  Tier 2 (expansion — fills budget after tier 1):
    - Supports expansion: lower-confidence reachable nodes (within
      MAX_HOPS hops). Adds nuance and supporting detail.
    - Divergent matches: HIGH-confidence nodes that are
      embedding-similar to the query but NOT graph-connected.
      Cross-cutting context the corpus hasn't yet explicitly linked.

  Unsurfaced count: when budget can't fit everything, record exactly
  how much was skipped per tier. The compose layer feeds this into
  the prompt as structured signal — Mirror's anti-self-sealing
  constraint then surfaces "I'm not showing the whole picture" in
  the response.

Mirror does the heavy lifting for honesty: by routing synthesis
through the existing OLI / claim-gate / anti-self-sealing chat
pipeline, we don't need a separate hallucination guard. The
synthesis-selected content becomes the authoritative knowledge base
for that turn; Mirror enforces that the model doesn't argue past it.

This module is the SELECTION layer only — it returns a structured
SynthesisResult. The COMPOSE layer (turning the result into a
response prompt) lives separately, as does chat-pipeline
integration. Step 1 of three.
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .corpus import CorpusStore
from . import ollama

logger = logging.getLogger(__name__)


# ── Tuning defaults ─────────────────────────────────────────────────
# All thresholds are exposed as kwargs to synthesize() — these are
# starting points calibrated for the current ~400-anchor / 380-slab
# merged corpus. Likely need re-tuning at 10× scale.

DEFAULT_HIGH_CONF_THRESHOLD = 0.80
"""Confidence floor for tier-1 inclusion. Mining produces conf 0.6-0.95
on slabs and 0.7-0.95 on anchors; 0.80 is a moderate "this content
is reliable enough to anchor a thesis" cutoff."""

DEFAULT_BUDGET_TOKENS = 10_000
"""Total token budget for selected content. ~8k for tier 1 + tier 2
content + ~2k reserve for structural metadata + unsurfaced signals.
Karpathy-quality output needs more breadth than typical RAG."""

DEFAULT_TIER1_FRACTION = 0.40
"""Fraction of budget reserved for tier-1 (supported + conflicts).
The thesis floor — never less than this."""

DEFAULT_MAX_HOPS = 2
"""Maximum BFS distance from seeds for tier-1/tier-2 reachability.
Hop 1 = directly connected, hop 2 = one intermediate. At 3+ hops
relevance decays sharply on this corpus."""

DEFAULT_MAX_SEEDS = 8
"""Top-K seeds picked from embedding-match against query."""

DEFAULT_MIN_SEED_COSINE = 0.55
"""Minimum cosine similarity between query embedding and a candidate
seed for the seed to count. Below this, the candidate isn't relevant."""

DEFAULT_MAX_TIER2_DIVERGENT = 6
"""Cap on the number of "embedding-similar but not graph-connected"
items added to tier 2. Prevents synthesis from drowning in
embedding-only matches when the graph is sparse."""

# Edge type sets — keep this in one place so future edge-type changes
# (e.g. wiring TENSIONS through, currently dead code) only touch here.
STRUCTURAL_EDGE_TYPES = ("SUPPORTS", "INVOKES", "LINKS", "PARENT_OF")
DIALECTIC_EDGE_TYPES = ("CONFLICTS",)


# ── Result types ────────────────────────────────────────────────────


@dataclass
class SelectedNode:
    """One slab / anchor / bundle picked for the synthesis context."""
    id: str
    node_type: str           # "anchor" | "slab" | "bundle"
    confidence: float
    text: str                # canonical_phrase / canonical_text / label
    hop_distance: Optional[int] = None  # graph distance from nearest seed
    cos_sim: Optional[float] = None     # cosine vs query (for divergent)
    via_edge: Optional[str] = None      # edge_type that brought it in
    conflicts_with: Optional[str] = None  # set on dialectic-tier nodes
    approx_tokens: int = 0


@dataclass
class UnsurfacedCounts:
    """How much corpus content was relevant but not included.

    Fed into the compose-layer prompt as a structured signal so the
    LLM (under Mirror's anti-self-sealing constraint) can acknowledge
    "the corpus has more than what I'm showing here". Without this,
    the model has no way to know what it's missing.
    """
    tier1_supported_skipped: int = 0
    tier2_supports_skipped: int = 0
    tier2_divergent_skipped: int = 0
    conflicts_skipped: int = 0
    extra_hops_not_walked: int = 0


@dataclass
class SynthesisResult:
    """Structured output of the selection algorithm.

    Compose layer (separate module) consumes this to build the
    response prompt. Demo harness pretty-prints it for human review.
    """
    query: str
    seeds: list[SelectedNode] = field(default_factory=list)
    tier1_supported: list[SelectedNode] = field(default_factory=list)
    tier1_conflicts: list[SelectedNode] = field(default_factory=list)
    tier2_supports: list[SelectedNode] = field(default_factory=list)
    tier2_divergent: list[SelectedNode] = field(default_factory=list)
    unsurfaced: UnsurfacedCounts = field(default_factory=UnsurfacedCounts)
    budget_tokens: int = 0
    budget_used_tokens: int = 0


# ── Helpers ─────────────────────────────────────────────────────────


def _approx_tokens(text: str) -> int:
    """Cheap chars-to-tokens approximation (~4 chars / token).

    Good enough for budget enforcement — the actual tokeniser
    differs per model and we just need to avoid blowing past
    num_ctx. Round up so we never UNDER-count.
    """
    return max(1, (len(text) + 3) // 4)


_COMMITTED_CONF = 0.85
"""Default confidence for any committed corpus member.

Confidence isn't stored on Anchor / Slab / KeyBundle directly —
these objects are committed corpus entries that have already passed
curator review. By the time content lands in corpus.anchors / .slabs
/ .bundles, the "is this reliable" question has been answered by the
human-in-the-loop commit step. So we treat all committed nodes as
uniformly high-confidence.

The implication: when synthesize() runs against a fully-committed
corpus, the high_conf_threshold filter is effectively a no-op
(everyone passes). Tiering reduces to hop-distance + edge-type +
embedding-similarity, which is what we want for v1. The threshold
becomes meaningful when we extend synthesis to include draft-stage
content (where MiningProposal carries real mining/dreaming confidence
values 0.55-0.95).

Earlier versions used anchor.match_policy.min_confidence_exact as
the proxy — that's the MATCHER firing threshold (default 0.70), not
anchor reliability. Mixing those two notions of "confidence" caused
1-hop anchor neighbours to fall into tier 2 incorrectly.
"""


def _node_text(corpus: CorpusStore, node_id: str) -> tuple[str, str, float]:
    """Return (node_type, text, confidence) for any corpus node id."""
    if node_id in corpus.anchors:
        a = corpus.anchors[node_id]
        return "anchor", a.canonical_phrase, _COMMITTED_CONF
    if node_id in corpus.slabs:
        s = corpus.slabs[node_id]
        return "slab", s.canonical_text, _COMMITTED_CONF
    if node_id in corpus.bundles:
        b = corpus.bundles[node_id]
        # KeyBundle has no .label / .canonical_text. Its content lives
        # in payload.intent (what the bundle is "about" — list of
        # intent statements). Join those for synthesis text.
        payload = getattr(b, "payload", None)
        intent_lines = list(getattr(payload, "intent", []) or []) if payload else []
        text = f"[Bundle {node_id}] " + " | ".join(intent_lines) if intent_lines else f"[Bundle {node_id}]"
        return "bundle", text, _COMMITTED_CONF
    return "", "", 0.0


def _selected_node(corpus: CorpusStore, node_id: str, **extras) -> Optional[SelectedNode]:
    """Build a SelectedNode from a corpus id + extras. None if id unknown."""
    ntype, text, conf = _node_text(corpus, node_id)
    if not ntype:
        return None
    return SelectedNode(
        id=node_id,
        node_type=ntype,
        confidence=conf,
        text=text,
        approx_tokens=_approx_tokens(text),
        **extras,
    )


# ── Seed identification ─────────────────────────────────────────────


def _l2_normalize(v: list[float]) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(arr))
    return arr / n if n > 1e-9 else arr


async def _identify_seeds(
    query: str,
    corpus: CorpusStore,
    *,
    max_seeds: int,
    min_cosine: float,
) -> tuple[list[str], np.ndarray]:
    """Embed the query, return top-K node IDs (anchors + slabs) by
    cosine similarity to the query embedding.

    Returns (seed_ids, query_embedding). Query embedding is returned
    so callers can reuse it for the divergent-matches step without
    re-embedding.
    """
    query_vec = await ollama.embed_single(query)
    query_arr = _l2_normalize(query_vec)

    # Build a target list — anchors keyed by canonical_phrase, slabs
    # by canonical_text. We embed both pools and rank together.
    target_texts: list[tuple[str, str]] = []  # (node_id, text)
    for a in corpus.anchors.values():
        if a.canonical_phrase:
            target_texts.append((a.id, a.canonical_phrase))
    for s in corpus.slabs.values():
        if s.canonical_text:
            target_texts.append((s.id, s.canonical_text[:600]))  # cap for embed cost

    if not target_texts:
        return [], query_arr

    # Batch embed in one call
    texts = [t[1] for t in target_texts]
    vecs = await ollama.embed(texts)

    sims: list[tuple[str, float]] = []
    for (nid, _), v in zip(target_texts, vecs):
        nv = _l2_normalize(v)
        sim = float(np.dot(query_arr, nv))
        if sim >= min_cosine:
            sims.append((nid, sim))
    sims.sort(key=lambda x: -x[1])
    return [nid for nid, _ in sims[:max_seeds]], query_arr


# ── Typed graph walk (BFS, edge-type filtered) ──────────────────────


def _build_typed_adjacency(
    corpus: CorpusStore, edge_types: tuple[str, ...]
) -> dict[str, list[tuple[str, str]]]:
    """Adjacency list keyed by node_id → [(neighbor_id, edge_type), ...].

    Only includes edges whose .type matches edge_types. Treats edges
    as undirected for the BFS — synthesis cares about reachability,
    not direction. (Direction matters for SEQUENCE narrative spine
    but synthesis isn't about narrative ordering.)
    """
    adj: dict[str, list[tuple[str, str]]] = defaultdict(list)
    type_set = set(edge_types)
    for e in corpus.edges.values():
        etype = getattr(e.type, "value", e.type) if hasattr(e, "type") else ""
        etype_str = str(etype)
        if etype_str not in type_set:
            continue
        from_id = e.from_node
        to_id = e.to_node
        if not from_id or not to_id:
            continue
        adj[from_id].append((to_id, etype_str))
        adj[to_id].append((from_id, etype_str))
    return adj


def _bfs_typed(
    corpus: CorpusStore,
    seeds: list[str],
    edge_types: tuple[str, ...],
    max_hops: int,
) -> dict[str, tuple[int, str]]:
    """BFS from seeds along edges of given types up to max_hops.

    Returns ``{node_id: (hop_distance, via_edge_type)}`` for every
    node reachable. Seeds themselves are at hop 0 with no via edge.
    """
    adj = _build_typed_adjacency(corpus, edge_types)
    visited: dict[str, tuple[int, str]] = {sid: (0, "") for sid in seeds if sid in corpus.anchors or sid in corpus.slabs or sid in corpus.bundles}
    if max_hops < 1 or not visited:
        return visited
    q: deque[str] = deque(visited.keys())
    while q:
        cur = q.popleft()
        cur_hops, _ = visited[cur]
        if cur_hops >= max_hops:
            continue
        for neighbor, etype in adj.get(cur, ()):
            if neighbor in visited:
                continue
            visited[neighbor] = (cur_hops + 1, etype)
            q.append(neighbor)
    return visited


# ── Conflict surfacing ──────────────────────────────────────────────


def _find_conflicts(
    corpus: CorpusStore, candidate_ids: set[str]
) -> list[tuple[str, str]]:
    """Find every CONFLICTS edge that touches a candidate.

    Returns ``[(conflict_endpoint_id, anchor_in_set_id), ...]`` — i.e.
    the ID of the node OUTSIDE candidate_ids that's in a CONFLICTS
    relationship with one of the candidates. If both endpoints are
    inside candidate_ids, it's an intra-set tension worth surfacing
    too — emit both as endpoints.
    """
    out: list[tuple[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for e in corpus.edges.values():
        etype = str(getattr(e.type, "value", e.type) if hasattr(e, "type") else "")
        if etype not in DIALECTIC_EDGE_TYPES:
            continue
        f, t = e.from_node, e.to_node
        if not f or not t:
            continue
        # If either endpoint is in our candidate set, the OTHER
        # endpoint is a conflict surface.
        if f in candidate_ids and t not in candidate_ids:
            key = (t, f)
            if key not in seen_pairs:
                out.append((t, f))
                seen_pairs.add(key)
        elif t in candidate_ids and f not in candidate_ids:
            key = (f, t)
            if key not in seen_pairs:
                out.append((f, t))
                seen_pairs.add(key)
        elif f in candidate_ids and t in candidate_ids:
            # Both inside — surface both ends so the prompt sees the
            # tension explicitly. Order the pair so we emit it once.
            pair_key = (min(f, t), max(f, t))
            if pair_key not in seen_pairs:
                out.append((f, t))
                out.append((t, f))
                seen_pairs.add(pair_key)
    return out


# ── Divergent semantic matches ──────────────────────────────────────


async def _divergent_matches(
    corpus: CorpusStore,
    query_arr: np.ndarray,
    excluded: set[str],
    *,
    high_conf: float,
    min_cosine: float,
    max_count: int,
) -> list[SelectedNode]:
    """Find HIGH-confidence anchors / slabs that are embedding-similar
    to the query but NOT in ``excluded`` (already-selected ids).

    These are "adjacent but not yet linked" — the corpus knows about
    them but hasn't drawn a structural edge to the seed. They surface
    cross-cutting context the user might otherwise miss.

    Cap at ``max_count`` to prevent embedding-only matches from
    drowning the structural selection.
    """
    candidates: list[tuple[str, str]] = []
    for a in corpus.anchors.values():
        if a.id in excluded:
            continue
        if a.canonical_phrase:
            candidates.append((a.id, a.canonical_phrase))
    for s in corpus.slabs.values():
        if s.id in excluded:
            continue
        if s.canonical_text:
            candidates.append((s.id, s.canonical_text[:600]))

    if not candidates:
        return []

    texts = [t[1] for t in candidates]
    vecs = await ollama.embed(texts)

    scored: list[tuple[str, float]] = []
    for (nid, _), v in zip(candidates, vecs):
        nv = _l2_normalize(v)
        sim = float(np.dot(query_arr, nv))
        if sim >= min_cosine:
            scored.append((nid, sim))
    scored.sort(key=lambda x: -x[1])

    out: list[SelectedNode] = []
    for nid, sim in scored:
        ntype, _, conf = _node_text(corpus, nid)
        if conf < high_conf:
            continue
        n = _selected_node(corpus, nid, cos_sim=sim)
        if n is None:
            continue
        out.append(n)
        if len(out) >= max_count:
            break
    return out


# ── Main entry point ────────────────────────────────────────────────


async def synthesize(
    query: str,
    corpus: CorpusStore,
    *,
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
    high_conf_threshold: float = DEFAULT_HIGH_CONF_THRESHOLD,
    tier1_fraction: float = DEFAULT_TIER1_FRACTION,
    max_hops: int = DEFAULT_MAX_HOPS,
    max_seeds: int = DEFAULT_MAX_SEEDS,
    min_seed_cosine: float = DEFAULT_MIN_SEED_COSINE,
    max_tier2_divergent: int = DEFAULT_MAX_TIER2_DIVERGENT,
) -> SynthesisResult:
    """Run the three-tier synthesis selection algorithm.

    See module docstring for the full design. This is the SELECTION
    layer only — turning a SynthesisResult into a response prompt
    lives in a separate compose layer (forthcoming).
    """
    result = SynthesisResult(query=query, budget_tokens=budget_tokens)

    # ── 1. Seeds ────────────────────────────────────────────────────
    seed_ids, query_arr = await _identify_seeds(
        query, corpus,
        max_seeds=max_seeds,
        min_cosine=min_seed_cosine,
    )
    if not seed_ids:
        logger.info("[SYNTHESIS] No seeds matched query %r at min_cosine=%s",
                    query[:60], min_seed_cosine)
        return result

    for sid in seed_ids:
        n = _selected_node(corpus, sid, hop_distance=0)
        if n is not None:
            result.seeds.append(n)

    # ── 2. BFS along structural edges (tier-1 candidates) ──────────
    structural_reach = _bfs_typed(
        corpus, seed_ids, STRUCTURAL_EDGE_TYPES, max_hops,
    )
    # tier-1 supported = high-conf, hop_distance >= 1 (seeds already in result.seeds)
    candidates_supported_t1: list[SelectedNode] = []
    candidates_supported_t2: list[SelectedNode] = []
    for nid, (hop, edge) in structural_reach.items():
        if hop == 0:
            continue  # seeds — already separate
        n = _selected_node(corpus, nid, hop_distance=hop, via_edge=edge)
        if n is None:
            continue
        if n.confidence >= high_conf_threshold:
            candidates_supported_t1.append(n)
        else:
            candidates_supported_t2.append(n)

    # Sort tier-1 supported by (hop ascending, confidence descending)
    candidates_supported_t1.sort(
        key=lambda n: (n.hop_distance or 99, -n.confidence)
    )
    # Sort tier-2 supports the same way — closer + higher conf wins
    candidates_supported_t2.sort(
        key=lambda n: (n.hop_distance or 99, -n.confidence)
    )

    # ── 3. Conflicts (tier-1) ──────────────────────────────────────
    in_set = set(seed_ids) | {n.id for n in candidates_supported_t1}
    conflict_pairs = _find_conflicts(corpus, in_set)
    candidates_conflict: list[SelectedNode] = []
    seen_conflict_ids: set[str] = set()
    for endpoint_id, anchor_id in conflict_pairs:
        if endpoint_id in seen_conflict_ids:
            continue
        seen_conflict_ids.add(endpoint_id)
        n = _selected_node(corpus, endpoint_id,
                           via_edge="CONFLICTS",
                           conflicts_with=anchor_id)
        if n is not None:
            candidates_conflict.append(n)

    # ── 4. Divergent semantic (tier-2) ─────────────────────────────
    excluded = (
        set(seed_ids)
        | {n.id for n in candidates_supported_t1}
        | {n.id for n in candidates_supported_t2}
        | {n.id for n in candidates_conflict}
    )
    candidates_divergent = await _divergent_matches(
        corpus, query_arr, excluded,
        high_conf=high_conf_threshold,
        min_cosine=min_seed_cosine,
        max_count=max_tier2_divergent,
    )

    # ── 5. Pack into budget ────────────────────────────────────────
    # Track running token spend. Tier 1 has a reserved fraction to
    # ensure it always lands. Conflicts go into tier 1's budget too —
    # dialectic surfacing is part of the floor.
    used = 0
    seed_tokens = sum(n.approx_tokens for n in result.seeds)
    used += seed_tokens
    tier1_floor = int(budget_tokens * tier1_fraction)
    # How many tokens we can spend on tier-2 if tier-1 fills its share:
    tier1_cap = max(seed_tokens + 1, tier1_floor)
    tier2_cap = budget_tokens  # tier 2 can fill the rest if tier 1 underuses

    def _pack(items: list[SelectedNode], cap: int, current_used: int):
        """Pack as many items as fit, return (packed_items, new_used,
        skipped_count)."""
        packed: list[SelectedNode] = []
        skipped = 0
        u = current_used
        for n in items:
            if u + n.approx_tokens > cap:
                skipped += 1
                continue
            packed.append(n)
            u += n.approx_tokens
        return packed, u, skipped

    # Conflicts first (dialectic floor — most distinctive design choice)
    result.tier1_conflicts, used, conf_skipped = _pack(
        candidates_conflict, tier1_cap, used,
    )
    result.unsurfaced.conflicts_skipped = conf_skipped

    # Then tier-1 supported, sharing the tier-1 budget
    result.tier1_supported, used, t1_skipped = _pack(
        candidates_supported_t1, tier1_cap, used,
    )
    result.unsurfaced.tier1_supported_skipped = t1_skipped

    # Tier 2 supports — uses remaining budget
    result.tier2_supports, used, t2sup_skipped = _pack(
        candidates_supported_t2, tier2_cap, used,
    )
    result.unsurfaced.tier2_supports_skipped = t2sup_skipped

    # Tier 2 divergent — also uses remaining budget
    result.tier2_divergent, used, t2div_skipped = _pack(
        candidates_divergent, tier2_cap, used,
    )
    result.unsurfaced.tier2_divergent_skipped = t2div_skipped

    # Track hops not walked (rough signal that more graph exists)
    # Currently a placeholder — for a precise count we'd need to walk
    # one extra hop and see how many new nodes appear. Skip for v1.
    result.unsurfaced.extra_hops_not_walked = 0
    result.budget_used_tokens = used
    return result
