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
        logger.info(
            "SlabMatcher: warmed %d REFERENCE slab embeddings",
            len(self._embed_cache),
        )

    async def top_k_for(
        self,
        query_text: str,
        threshold: float = 0.55,
        max_k: int = 15,
    ) -> list[tuple[str, float]]:
        """Return ``(slab_id, cosine_score)`` pairs for REFERENCE slabs
        semantically close to ``query_text``, sorted descending.

        Filtering:
          * score >= ``threshold`` (default 0.55 — matches AnchorMatcher's
            "candidate" tier for nomic-embed-text).
          * Cap at ``max_k`` entries to bound the token budget.

        Returns an empty list if the cache is cold, the query is empty, or
        no slab scores above threshold.
        """
        if self._matrix is None or not query_text or not query_text.strip():
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

        # argsort descending, filter by threshold, cap at max_k
        idx_sorted = np.argsort(-scores)
        results: list[tuple[str, float]] = []
        for i in idx_sorted:
            score = float(scores[i])
            if score < threshold:
                break  # sorted, no later entry will pass
            results.append((self._ordered_ids[i], score))
            if len(results) >= max_k:
                break
        return results

    def has_cache(self) -> bool:
        """True if warm_cache has populated the embedding matrix."""
        return self._matrix is not None and len(self._embed_cache) > 0
