"""PageRank + Personalized PageRank over the corpus graph.

Pure math module — no retrieval policy, no caching, no I/O. Callers
(currently SlabMatcher) are responsible for deciding when to recompute
and how to use the scores.

Why this lives alongside SlabMatcher:

* Cosine similarity answers "how textually relevant is this slab to the
  current query?" — a query-dependent, text-surface signal.
* Global PageRank answers "how structurally important is this node in
  the curated graph?" — a query-independent, link-structure signal.
* Personalized PageRank (PPR) answers "given what the conversation is
  already attending to, how structurally close is this node?" — a
  per-turn, seed-conditional signal.

All three are combined downstream in SlabMatcher.hybrid_rank. This
module just produces the PR vectors.

Implementation is a plain power iteration over a row-stochastic
transition matrix built from corpus.edges. At current corpus scales
(<1000 nodes) this is sub-millisecond; a SciPy-sparse version would
be needed above ~10k nodes but is unnecessary now.
"""
from __future__ import annotations

import logging
from typing import Iterable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Tuning ───────────────────────────────────────────────────────
# Damping factor — standard Brin/Page 0.85. Higher → more weight on
# link structure; lower → more weight on teleport distribution.
DEFAULT_DAMPING = 0.85

# Iteration caps. Power iteration for PageRank on small graphs
# converges in ~20-30 iterations; we leave headroom.
DEFAULT_MAX_ITER_GLOBAL = 50
DEFAULT_MAX_ITER_PPR = 30

# L1 convergence tolerance — when ||r_new - r||_1 drops below this, stop.
DEFAULT_TOL = 1e-6


def _build_transition(corpus) -> tuple[np.ndarray, list[str]]:
    """Row-stochastic transition matrix from corpus.edges.

    Node ordering: anchors, then slabs, then bundles. All three node
    types participate so PR can flow through the whole curated graph
    (anchor INVOKES slab, bundle PARENT_OF slab, slab SEQUENCE slab).

    **Edges are symmetrized** for PR purposes even though the authored
    graph is directed. Rationale: an authored edge ``A --INVOKES--> B``
    means "A is about B" — a *relational* claim. For a random walker
    exploring structural closeness, the relationship is bidirectional:
    a seed on B should still be able to reach A. Without symmetry,
    PPR from a leaf slab can't walk back to the anchor that invoked
    it, which is exactly the retrieval signal we want.

    The reverse direction gets the same weight as the forward direction
    (no asymmetric decay), which matches the intuition that the
    authored direction is a convention, not a strength claim.

    Edge weights from ``Edge.weight`` (pydantic field, range [0, 1]).
    Multiple edges between the same node pair sum their weights.
    Dangling nodes (no outgoing edges) teleport uniformly — standard
    PageRank treatment to keep the matrix well-conditioned. With
    symmetrization, dangling nodes are rare (a truly isolated node
    with no edges at all).

    Returns:
        P: (n, n) row-stochastic float32 matrix
        ids: list of node ids in the order corresponding to matrix rows
    """
    ids: list[str] = (
        list(corpus.anchors.keys())
        + list(corpus.slabs.keys())
        + list(corpus.bundles.keys())
    )
    idx = {nid: i for i, nid in enumerate(ids)}
    n = len(ids)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32), []

    A = np.zeros((n, n), dtype=np.float32)
    for e in corpus.edges.values():
        fi = idx.get(e.from_node)
        ti = idx.get(e.to_node)
        if fi is None or ti is None:
            continue  # edge references a node not in the active corpus
        w = float(getattr(e, "weight", 1.0) or 1.0)
        # Symmetric: relationship is bidirectional for random-walk purposes
        A[fi, ti] += w
        A[ti, fi] += w

    # Row-normalize. Dangling rows (all zero) get a uniform distribution
    # so the random walker doesn't get stuck there — equivalent to
    # adding a virtual "follow a random link" fallback.
    row_sums = A.sum(axis=1)
    dangling = row_sums == 0
    if dangling.any():
        A[dangling] = 1.0 / n
        # Rows we just filled are already normalized; mask them out of
        # the division below.
    nonzero_mask = ~dangling
    if nonzero_mask.any():
        A[nonzero_mask] = A[nonzero_mask] / row_sums[nonzero_mask, None]

    return A, ids


def _power_iterate(
    P: np.ndarray,
    teleport: np.ndarray,
    damping: float,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    """Standard damped-random-walk power iteration.

    r_{t+1} = damping · (r_t @ P) + (1 - damping) · teleport

    At convergence (L1 distance < tol) we have r = r @ (damping·P) +
    (1-damping)·teleport — the PageRank fixed point.
    """
    r = teleport.copy()
    for i in range(max_iter):
        r_new = damping * (r @ P) + (1.0 - damping) * teleport
        if float(np.abs(r_new - r).sum()) < tol:
            logger.debug("PR converged in %d iterations", i + 1)
            return r_new
        r = r_new
    logger.debug("PR hit max_iter=%d without converging below tol", max_iter)
    return r


def compute_pagerank(
    corpus,
    damping: float = DEFAULT_DAMPING,
    max_iter: int = DEFAULT_MAX_ITER_GLOBAL,
    tol: float = DEFAULT_TOL,
) -> dict[str, float]:
    """Global PageRank — uniform teleport distribution.

    Scores sum to 1.0 across all nodes. Interpretation: probability
    that a random walker with occasional uniform teleport lands on
    node X in the steady state. Higher = more structurally central.

    Returns a ``{node_id: score}`` map covering every node in the
    corpus (anchors, slabs, bundles). Returns empty dict if the
    graph has no nodes.
    """
    P, ids = _build_transition(corpus)
    n = len(ids)
    if n == 0:
        return {}
    teleport = np.full(n, 1.0 / n, dtype=np.float32)
    r = _power_iterate(P, teleport, damping, max_iter, tol)
    return dict(zip(ids, (float(x) for x in r)))


def compute_ppr(
    corpus,
    seeds: Iterable[str],
    damping: float = DEFAULT_DAMPING,
    max_iter: int = DEFAULT_MAX_ITER_PPR,
    tol: float = DEFAULT_TOL,
) -> dict[str, float]:
    """Personalized PageRank — teleport distribution concentrated on seeds.

    Given seed node ids (typically the frame's active non-base_set
    nodes), returns a reachability distribution where seeds themselves
    have highest mass and it decays along edges by distance and
    branching.

    Unlike the bounded K-hop typed walk in ``_build_edge_subspace``, PPR
    is continuous multi-hop within whatever ``corpus`` it is handed: a node
    several hops away with many paths back to seeds can outrank a node one
    hop away on a single dead-end edge. Callers scope it by passing a
    subgraph corpus (the seed-neighbourhood walk domain) instead of the
    full merged store — the transition matrix is then O(slice²).

    Returns empty dict if there are no seeds in the corpus or the
    corpus is empty.
    """
    P, ids = _build_transition(corpus)
    n = len(ids)
    if n == 0:
        return {}
    idx = {nid: i for i, nid in enumerate(ids)}
    seed_idxs = [idx[s] for s in seeds if s in idx]
    if not seed_idxs:
        return {nid: 0.0 for nid in ids}

    teleport = np.zeros(n, dtype=np.float32)
    for si in seed_idxs:
        teleport[si] = 1.0
    teleport /= teleport.sum()  # normalize over the seed set

    r = _power_iterate(P, teleport, damping, max_iter, tol)
    return dict(zip(ids, (float(x) for x in r)))


def normalize_max(scores: dict[str, float]) -> dict[str, float]:
    """Rescale so the max score is 1.0 (min stays >= 0).

    Use when combining PR with cosine similarity — PR values are tiny
    (~1/N) because they sum to 1, so they get drowned out unless
    rescaled. Dividing by max puts them on the same order of magnitude
    as cosine similarities, which makes the linear combination's
    weights ``alpha``, ``beta``, ``gamma`` interpretable as relative
    importance rather than arbitrary unit-conversion factors.
    """
    if not scores:
        return {}
    peak = max(scores.values())
    if peak <= 0:
        return {k: 0.0 for k in scores}
    return {k: v / peak for k, v in scores.items()}
