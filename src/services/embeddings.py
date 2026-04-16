"""Embedding service for semantic axis placement (§11.1).

Uses nomic-embed-text via Ollama for 768-dim embeddings.
X-axis: Creative ←→ Rigorous — defined by two pole embeddings.
Projection is a normalized dot product onto the axis vector.
"""
from __future__ import annotations
import numpy as np
from typing import Optional
from . import ollama

# Pole definitions for the X-axis
_CREATIVE_POLE = "creative writing, artistic expression, imagination, and open exploration"
_RIGOROUS_POLE = "rigorous scientific analysis, empirical methodology, formal logic, and verification"

# Cached pole embeddings and axis vector
_axis_cache: Optional[dict] = None


async def _ensure_axis() -> dict:
    """Compute and cache the creative←→rigorous axis vector."""
    global _axis_cache
    if _axis_cache is not None:
        return _axis_cache

    poles = await ollama.embed([_CREATIVE_POLE, _RIGOROUS_POLE])
    creative = np.array(poles[0])
    rigorous = np.array(poles[1])
    axis = rigorous - creative
    axis_norm = axis / np.linalg.norm(axis)

    _axis_cache = {
        "creative": creative,
        "rigorous": rigorous,
        "axis": axis,
        "axis_norm": axis_norm,
        "axis_len": float(np.linalg.norm(axis)),
    }
    return _axis_cache


async def compute_x_position(text: str) -> float:
    """Project text onto X-axis. Returns 0.0 (creative) to 1.0 (rigorous)."""
    cache = await _ensure_axis()
    vec = np.array(await ollama.embed_single(text))
    proj = float(np.dot(vec - cache["creative"], cache["axis_norm"]) / cache["axis_len"])
    return max(0.0, min(1.0, proj))


async def compute_x_positions(texts: list[str]) -> list[float]:
    """Batch X-axis placement for multiple texts."""
    if not texts:
        return []
    cache = await _ensure_axis()
    vecs = await ollama.embed(texts)
    positions = []
    for vec in vecs:
        v = np.array(vec)
        proj = float(np.dot(v - cache["creative"], cache["axis_norm"]) / cache["axis_len"])
        positions.append(max(0.0, min(1.0, proj)))
    return positions


async def compute_positions_and_vectors(texts: list[str]) -> tuple[list[float], np.ndarray]:
    """Batch: return (x_projections, raw_vectors) — single embedding call.

    Callers that need both the 1D creative↔rigorous projection AND the full
    vectors (e.g. for cosine-similarity affinity) should use this instead of
    compute_x_positions + a second embed call.
    """
    if not texts:
        return [], np.zeros((0, 0))
    cache = await _ensure_axis()
    raw = await ollama.embed(texts)
    V = np.array(raw)
    positions: list[float] = []
    for v in V:
        proj = float(np.dot(v - cache["creative"], cache["axis_norm"]) / cache["axis_len"])
        positions.append(max(0.0, min(1.0, proj)))
    return positions, V


def compute_affinity(
    node_ids: list[str],
    node_types: list[str],
    vectors: np.ndarray,
    *,
    explicit_parent: dict[str, str] | None = None,
    parent_threshold: float = 0.70,
    pair_threshold: float = 0.65,
    top_k_pairs: int = 3,
    parent_types: tuple[str, ...] = ("slab",),
    child_types: tuple[str, ...] = ("anchor", "bundle"),
) -> tuple[dict[str, tuple[str, float]], list[dict]]:
    """Compute embedding-based affinity for a set of nodes.

    Returns:
        implicit_parent_map: {child_id: (parent_id, similarity)} for children
            with no explicit parent and a best-match parent ≥ parent_threshold.
        affinity_pairs: top-K undirected pairs per node above pair_threshold,
            each as {from, to, similarity}. Symmetric pairs deduped.

    Args:
        node_ids, node_types, vectors: parallel arrays; vectors is (N, D).
        explicit_parent: optional set-like dict of children who already have
            an explicit edge to a parent; they skip implicit-parent assignment
            but still participate in pair affinity.
        parent_threshold: minimum cosine similarity to emit an implicit_parent.
        pair_threshold: minimum similarity for any affinity pair.
        top_k_pairs: per-node neighbour cap (sparser graph than full N²).
        parent_types / child_types: restrict implicit_parent search to these
            type combinations (e.g. anchors orbit slabs, not other anchors).
    """
    import numpy as np
    N = len(node_ids)
    if N == 0 or vectors.shape[0] != N:
        return {}, []
    explicit_parent = explicit_parent or {}

    # Normalise and build similarity matrix in one shot.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid /0 for degenerate vectors
    V = vectors / norms
    S = V @ V.T  # (N, N) cosine similarity
    np.fill_diagonal(S, -1.0)  # exclude self-match

    # Implicit parent assignment: for each child, find best parent.
    implicit_parent_map: dict[str, tuple[str, float]] = {}
    parent_idx = [i for i, t in enumerate(node_types) if t in parent_types]
    for i, (nid, nt) in enumerate(zip(node_ids, node_types)):
        if nt not in child_types:
            continue
        if nid in explicit_parent:
            continue
        if not parent_idx:
            continue
        best_j = max(parent_idx, key=lambda j: S[i, j])
        sim = float(S[i, best_j])
        if sim >= parent_threshold:
            implicit_parent_map[nid] = (node_ids[best_j], round(sim, 3))

    # Top-K affinity pairs per node, deduped across (i, j) / (j, i).
    seen: set[tuple[str, str]] = set()
    pairs: list[dict] = []
    for i in range(N):
        # argsort descending; take slice beyond the self-mask
        order = np.argsort(-S[i])
        taken = 0
        for j in order:
            if taken >= top_k_pairs:
                break
            sim = float(S[i, j])
            if sim < pair_threshold:
                break  # sorted desc — remaining are all below
            a, b = node_ids[i], node_ids[j]
            key = (a, b) if a < b else (b, a)
            if key in seen:
                taken += 1
                continue
            seen.add(key)
            pairs.append({"from": a, "to": b, "similarity": round(sim, 3)})
            taken += 1
    return implicit_parent_map, pairs


async def cosine_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity between two texts. Used for anchor matching.

    NOTE: inputs are lowercased before embedding. nomic-embed-text:latest
    (via Ollama) has a reproducible defect where Title-Case 3-word phrases
    collapse to a shared canonical vector, producing false cosine=1.0
    matches between unrelated anchors (e.g. "Terra Preta Australis" vs
    "Iterative Visual Prototyping"). Lowercasing bypasses the pathological
    tokenization path. Anchor-name dedup and similarity semantics are
    case-insensitive anyway so this is a safe normalization.
    """
    vecs = await ollama.embed([text_a.lower(), text_b.lower()])
    a, b = np.array(vecs[0]), np.array(vecs[1])
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
