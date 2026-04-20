"""Semantic retrieval for REFERENCE slabs — the "pointer" side of the
graph-of-graphs-above-the-corpus architecture.

The model's system prompt carries CONSTITUTIONAL and CANONICAL slabs in full
(rule text and foundational curator content, non-negotiable). REFERENCE
slabs — mined narratives, ambient domain content — are far more numerous
and don't need to all be present every turn. Instead, the model sees:

  * A compact CATALOG of every REFERENCE slab (id + title + short summary),
    so it knows what exists and can reason about what it would ask for.
  * Full text of only the handful of REFERENCE slabs that are semantically
    close to the current user message, above a similarity threshold and
    capped at max_k.

SlabMatcher handles the retrieval side: it embeds every REFERENCE slab's
``canonical_text`` on warm-up, and on each turn computes cosine similarity
against the user's message embedding to rank slabs. Mirrors the shape of
AnchorMatcher._embed_cache but simpler (one embedding per slab, no alias
list).
"""
from __future__ import annotations
import logging
from typing import Optional

import numpy as np

from ..models.enums import SlabType, SlabLifecycleStatus
from .corpus import CorpusStore
from . import ollama
from . import graph_rank

logger = logging.getLogger(__name__)


class SlabMatcher:
    """Retrieve the most relevant REFERENCE slabs for a given query text.

    Cache is built on startup via ``warm_cache`` and can be invalidated /
    rebuilt after corpus changes. Uses nomic-embed-text via ollama.embed for
    vectorization, same as AnchorMatcher — shares the embedding model so no
    additional VRAM cost.
    """

    def __init__(self, corpus: CorpusStore):
        self.corpus = corpus
        # slab_id -> L2-normalized embedding vector (768-dim).
        # Normalizing at cache time means query-time similarity is a single
        # dot product against a pre-stacked matrix, no per-entry norm work.
        self._embed_cache: dict[str, np.ndarray] = {}
        # Parallel stacked matrix for fast batch similarity — rebuilt whenever
        # the cache changes. Shape: (N_slabs, 768). Order matches
        # ``self._ordered_ids``.
        self._matrix: Optional[np.ndarray] = None
        self._ordered_ids: list[str] = []
        # Global PageRank over the full graph (anchors + slabs + bundles),
        # max-normalized to [0, 1] so the numbers are comparable to cosine
        # similarities when combined in hybrid_rank. Keyed by node id
        # (not just slab id) because PR flows through all node types —
        # an anchor that INVOKES a slab donates rank to the slab even
        # though we only retrieve slabs. Recomputed on warm_cache.
        self._global_pr: dict[str, float] = {}

    async def warm_cache(self) -> None:
        """Embed every ACTIVE REFERENCE slab in the current corpus.

        Safe to call repeatedly — each call rebuilds from the current corpus
        view. Intended use: once at server startup, and again after
        collection activation changes (via rebind_corpus).
        """
        candidates = [
            s for s in self.corpus.slabs.values()
            if s.type == SlabType.REFERENCE
            and s.lifecycle_status == SlabLifecycleStatus.ACTIVE
        ]
        if not candidates:
            self._embed_cache.clear()
            self._matrix = None
            self._ordered_ids = []
            logger.info("SlabMatcher: no REFERENCE slabs to embed")
            return

        # Lowercase matches the convention used by embeddings.compute_positions_and_vectors
        # (nomic-embed-text is title-case sensitive without this).
        texts = [(s.canonical_text or "").lower() for s in candidates]
        vectors = await ollama.embed(texts)

        self._embed_cache = {}
        ordered_ids: list[str] = []
        matrix_rows: list[np.ndarray] = []
        for slab, vec in zip(candidates, vectors):
            v = np.asarray(vec, dtype=np.float32)
            norm = float(np.linalg.norm(v))
            if norm > 0:
                v = v / norm
            self._embed_cache[slab.id] = v
            ordered_ids.append(slab.id)
            matrix_rows.append(v)

        self._ordered_ids = ordered_ids
        self._matrix = np.stack(matrix_rows, axis=0) if matrix_rows else None

        # Warm the global PageRank cache alongside embeddings. PR is cheap
        # at this scale (sub-ms for <1k nodes via power iteration), so it
        # rides along with the embedding warm-up instead of getting its
        # own lifecycle. Max-normalized so values are comparable to cosine
        # similarities in the hybrid ranker.
        try:
            raw_pr = graph_rank.compute_pagerank(self.corpus)
            self._global_pr = graph_rank.normalize_max(raw_pr)
        except Exception as exc:
            logger.warning("SlabMatcher: global PageRank failed: %r", exc)
            self._global_pr = {}

        logger.info(
            "SlabMatcher: warmed %d REFERENCE slab embeddings, %d PR nodes",
            len(self._embed_cache), len(self._global_pr),
        )

    async def top_k_for(
        self,
        query_text: str,
        threshold: float = 0.55,
        max_k: int = 15,
        id_filter: Optional[set[str]] = None,
    ) -> list[tuple[str, float]]:
        """Return ``(slab_id, cosine_score)`` pairs for REFERENCE slabs
        semantically close to ``query_text``, sorted descending.

        Filtering:
          * score >= ``threshold`` (default 0.55 — matches AnchorMatcher's
            "candidate" tier for nomic-embed-text).
          * Cap at ``max_k`` entries to bound the token budget.
          * If ``id_filter`` is provided, only slabs whose id is in the
            set are considered — this is the "subspace rank" case used by
            edge-constrained retrieval. The rest of the corpus is excluded
            even if it would have scored higher. Empty filter returns an
            empty list (caller should fall back explicitly).

        Returns an empty list if the cache is cold, the query is empty, or
        no slab scores above threshold.
        """
        if self._matrix is None or not query_text or not query_text.strip():
            return []
        if id_filter is not None and not id_filter:
            # Explicit empty subspace — caller asked for nothing.
            return []

        vecs = await ollama.embed([query_text.strip().lower()])
        if not vecs:
            return []
        q = np.asarray(vecs[0], dtype=np.float32)
        norm = float(np.linalg.norm(q))
        if norm == 0:
            return []
        q = q / norm

        # Vectorized cosine similarity: one matrix-vector product.
        scores = self._matrix @ q  # shape: (N,)

        # argsort descending, filter by threshold, subspace filter, cap at max_k
        idx_sorted = np.argsort(-scores)
        results: list[tuple[str, float]] = []
        for i in idx_sorted:
            score = float(scores[i])
            if score < threshold:
                break  # sorted, no later entry will pass
            sid = self._ordered_ids[i]
            if id_filter is not None and sid not in id_filter:
                continue
            results.append((sid, score))
            if len(results) >= max_k:
                break
        return results

    async def hybrid_rank(
        self,
        query_text: str,
        seed_ids: Optional[set[str]] = None,
        id_filter: Optional[set[str]] = None,
        threshold: float = 0.55,
        max_k: int = 15,
        alpha: float = 1.0,   # cosine weight  — "is this textually relevant?"
        beta: float = 0.5,    # PPR weight     — "is this structurally close to seeds?"
        gamma: float = 0.3,   # global PR weight — "is this intrinsically important?"
    ) -> list[tuple[str, float, dict]]:
        """Combined cosine + Personalized PageRank + global PageRank ranking.

        Three orthogonal signals, linearly combined:

          score(slab) = alpha · cosine(query, slab)
                      + beta  · ppr_from_seeds(slab)
                      + gamma · global_pagerank(slab)

        The PR contributions are max-normalized to [0, 1] inside this
        method so all three signals sit on comparable scales — alpha,
        beta, gamma then read as relative importance weights, not
        arbitrary unit conversions.

        Gating: cosine threshold is applied BEFORE PR bonuses. Rationale:
        PR is a re-ranking signal among semantically-relevant content,
        not a license to retrieve irrelevant content just because it's
        structurally central. Without this gate, a highly-central slab
        unrelated to the query would always surface.

        Args:
            query_text: user's current message
            seed_ids: nodes to teleport to for PPR. Typically the frame's
                     active non-base_set nodes. If None or empty, PPR is
                     skipped (beta effectively 0 for this call).
            id_filter: restrict candidates to this slab id set (subspace).
                      None = full corpus. Empty set = no candidates
                      (caller should fall back explicitly).
            threshold: cosine-similarity floor. Default matches top_k_for.
            max_k: cap on returned entries.
            alpha, beta, gamma: signal weights.

        Returns:
            List of ``(slab_id, combined_score, components)`` sorted by
            combined_score descending. ``components`` is a dict with keys
            "cosine", "ppr", "global_pr" for observability / debugging.
        """
        if self._matrix is None or not query_text or not query_text.strip():
            return []
        if id_filter is not None and not id_filter:
            return []

        vecs = await ollama.embed([query_text.strip().lower()])
        if not vecs:
            return []
        q = np.asarray(vecs[0], dtype=np.float32)
        norm = float(np.linalg.norm(q))
        if norm == 0:
            return []
        q = q / norm

        # Stage 1: cosine threshold gate.
        cosine_scores = self._matrix @ q  # shape: (N,)

        # Stage 2: PPR over the full graph, teleporting to seeds.
        # Empty seeds → skip PPR entirely rather than uniform, because
        # uniform PPR is identical to global PR and we'd double-count it.
        ppr_norm: dict[str, float] = {}
        if seed_ids and beta > 0:
            try:
                raw_ppr = graph_rank.compute_ppr(self.corpus, seed_ids)
                ppr_norm = graph_rank.normalize_max(raw_ppr)
            except Exception as exc:
                logger.warning("hybrid_rank: PPR failed: %r", exc)

        # Stage 3: combine signals for survivors.
        gpr = self._global_pr if gamma > 0 else {}

        scored: list[tuple[str, float, dict]] = []
        for i, sid in enumerate(self._ordered_ids):
            cos = float(cosine_scores[i])
            if cos < threshold:
                continue
            if id_filter is not None and sid not in id_filter:
                continue
            ppr_v = ppr_norm.get(sid, 0.0)
            gpr_v = gpr.get(sid, 0.0)
            combined = alpha * cos + beta * ppr_v + gamma * gpr_v
            scored.append((sid, combined, {
                "cosine": round(cos, 4),
                "ppr": round(ppr_v, 4),
                "global_pr": round(gpr_v, 4),
            }))

        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:max_k]

    def has_cache(self) -> bool:
        """True if warm_cache has populated the embedding matrix."""
        return self._matrix is not None and len(self._embed_cache) > 0
