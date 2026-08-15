"""Hybrid retrieval for REFERENCE slabs.

CONSTITUTIONAL and CANONICAL slabs remain foundational prompt content. The
larger REFERENCE layer is selected per turn: collection names restrict the
candidate domain, lexical BM25-style matching and dense cosine matching are
fused with reciprocal-rank fusion, and graph rank then expands/reranks the
bounded candidate set. Only the highest-ranked references receive full-text
prompt budget; the rest are represented by collection-level awareness counts.

``SlabMatcher`` owns the dense embedding cache, live lexical scan, fused
candidate ranking, and PageRank caches. Dense embeddings include each slab's
title and canonical text. ``warm_cache(only_new=True)`` incrementally embeds
new immutable slabs, prunes removed entries, and refreshes graph rank.
"""
from __future__ import annotations
from collections import Counter
import logging
import math
import re
from typing import Optional

import numpy as np

from ..models.enums import SlabType, SlabLifecycleStatus
from .corpus import CorpusStore
from . import ollama
from . import graph_rank

logger = logging.getLogger(__name__)


_SEARCH_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "based", "by", "for",
    "from", "how", "i", "in", "is", "it", "me", "of", "on", "per",
    "collection", "config", "configuration", "please", "pod", "spec",
    "specification", "tell", "teh", "the", "to", "was",
    "what", "with",
}

_SEARCH_EXPANSIONS = {
    "dimension": {"length", "width", "height", "depth", "dimensions"},
    "dimensions": {"length", "width", "height", "depth", "dimension"},
    "size": {"length", "width", "height", "depth", "dimensions"},
    "salt": {"saltwater", "saline", "salinity", "freshwater"},
    "saltwater": {"salt", "saline", "salinity", "freshwater"},
    "saline": {"saltwater", "salinity", "freshwater"},
    "volume": {"litres", "liters", "cubic", "capacity"},
}


def _search_tokens(text: str) -> list[str]:
    """Normalize prose, measurements, and common unit spellings for search."""
    normalized = (text or "").lower()
    normalized = normalized.replace("m³", " cubic metres ")
    normalized = normalized.replace("m^3", " cubic metres ")
    normalized = normalized.replace("salt-water", "saltwater")
    normalized = re.sub(r"\b(?:pod_?v?|version|v)[_\s-]*\d+\b", " ", normalized)
    normalized = re.sub(r"\bmeters?\b", "metres", normalized)
    normalized = re.sub(r"\bliters?\b", "litres", normalized)
    return re.findall(r"[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?", normalized)


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

    async def warm_cache(self, only_new: bool = False) -> None:
        """Embed every ACTIVE REFERENCE slab in the current corpus.

        Safe to call repeatedly. A full warm rebuilds from the current corpus;
        ``only_new=True`` retains existing immutable slab vectors, prunes removed
        ids, embeds new slabs, and always refreshes graph rank.
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
        candidate_ids = {s.id for s in candidates}
        if only_new:
            # Drop inactive/removed entries, retain immutable slabs already
            # embedded, and pay only for content promoted since the last warm.
            self._embed_cache = {
                sid: vec for sid, vec in self._embed_cache.items()
                if sid in candidate_ids
            }
            pending = [s for s in candidates if s.id not in self._embed_cache]
        else:
            self._embed_cache = {}
            pending = candidates

        if pending:
            texts = [
                "\n".join(
                    part for part in (
                        s.title or "",
                        getattr(s, "description", "") or "",
                        s.canonical_text or "",
                    ) if part
                ).lower()
                for s in pending
            ]
            vectors = await ollama.embed(texts)
            for slab, vec in zip(pending, vectors):
                v = np.asarray(vec, dtype=np.float32)
                norm = float(np.linalg.norm(v))
                if norm > 0:
                    v = v / norm
                self._embed_cache[slab.id] = v

        ordered_ids: list[str] = []
        matrix_rows: list[np.ndarray] = []
        for slab in candidates:
            v = self._embed_cache.get(slab.id)
            if v is None:
                continue
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
            "SlabMatcher: warmed %d REFERENCE slab embeddings (%d new), %d PR nodes",
            len(self._embed_cache), len(pending), len(self._global_pr),
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

    def lexical_top_k_for(
        self,
        query_text: str,
        max_k: int = 20,
        id_filter: Optional[set[str]] = None,
    ) -> list[tuple[str, float]]:
        """Return BM25-style title/body matches from the live corpus view.

        This path deliberately does not depend on the embedding cache. Besides
        improving exact-fact recall (measurements, configuration terms), it is
        a safe freshness fallback during the brief interval after promotion.
        """
        raw_terms = [t for t in _search_tokens(query_text) if t not in _SEARCH_STOPWORDS]
        if not raw_terms or (id_filter is not None and not id_filter):
            return []

        weighted_terms: dict[str, float] = {term: 1.0 for term in raw_terms}
        for term in raw_terms:
            for expanded in _SEARCH_EXPANSIONS.get(term, set()):
                weighted_terms[expanded] = max(weighted_terms.get(expanded, 0.0), 0.55)

        documents: list[tuple[str, Counter, Counter, int]] = []
        for slab in self.corpus.slabs.values():
            if slab.type != SlabType.REFERENCE or slab.lifecycle_status != SlabLifecycleStatus.ACTIVE:
                continue
            if id_filter is not None and slab.id not in id_filter:
                continue
            body = _search_tokens(
                "\n".join(
                    part for part in (
                        getattr(slab, "description", "") or "",
                        slab.canonical_text or "",
                    ) if part
                )
            )
            title = _search_tokens(slab.title or "")
            documents.append((slab.id, Counter(body), Counter(title), max(1, len(body))))

        if not documents:
            return []

        n_docs = len(documents)
        avg_len = sum(length for _sid, _body, _title, length in documents) / n_docs
        doc_freq = {
            term: sum(1 for _sid, body, title, _len in documents if term in body or term in title)
            for term in weighted_terms
        }
        k1, b = 1.2, 0.75
        scored: list[tuple[str, float]] = []
        for sid, body, title, length in documents:
            score = 0.0
            for term, query_weight in weighted_terms.items():
                tf = body.get(term, 0) + 2.0 * title.get(term, 0)
                if not tf:
                    continue
                df = doc_freq[term]
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                saturation = (tf * (k1 + 1.0)) / (
                    tf + k1 * (1.0 - b + b * length / max(avg_len, 1.0))
                )
                exact_weight = 1.75 if re.fullmatch(r"\d+(?:\.\d+)?", term) else 1.0
                score += query_weight * exact_weight * idf * saturation
            if score > 0:
                scored.append((sid, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:max_k]

    async def fused_top_k_for(
        self,
        query_text: str,
        max_k: int = 15,
        id_filter: Optional[set[str]] = None,
    ) -> list[tuple[str, float, dict]]:
        """Fuse dense and lexical ranks without requiring both to agree."""
        dense = await self.top_k_for(
            query_text, threshold=0.55, max_k=max(max_k, 20), id_filter=id_filter,
        ) if self.has_cache() else []
        lexical = self.lexical_top_k_for(
            query_text, max_k=max(max_k, 20), id_filter=id_filter,
        )
        dense_scores = dict(dense)
        lexical_scores = dict(lexical)
        fused: dict[str, float] = {}
        rrf_k = 60.0
        for rank, (sid, _score) in enumerate(dense, 1):
            fused[sid] = fused.get(sid, 0.0) + 1.0 / (rrf_k + rank)
        for rank, (sid, _score) in enumerate(lexical, 1):
            fused[sid] = fused.get(sid, 0.0) + 1.0 / (rrf_k + rank)

        ranked = sorted(fused, key=lambda sid: fused[sid], reverse=True)[:max_k]
        return [
            (sid, fused[sid], {
                "dense": round(dense_scores.get(sid, 0.0), 4),
                "lexical": round(lexical_scores.get(sid, 0.0), 4),
            })
            for sid in ranked
        ]

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
        ppr_corpus=None,      # scope PPR to this corpus (default: full merged)
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

        # Stage 2: PPR over ppr_corpus (the seed-neighbourhood subgraph when
        # given, else the full graph), teleporting to seeds.
        # Empty seeds → skip PPR entirely rather than uniform, because
        # uniform PPR is identical to global PR and we'd double-count it.
        ppr_norm: dict[str, float] = {}
        if seed_ids and beta > 0:
            try:
                # Scope PPR to ppr_corpus when given (a collection subgraph)
                # so the transition matrix is O(collection²) not O(merged²).
                # Seeds outside the scoped graph are harmlessly ignored.
                raw_ppr = graph_rank.compute_ppr(ppr_corpus or self.corpus, seed_ids)
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
