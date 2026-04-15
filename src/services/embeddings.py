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
