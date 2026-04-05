"""Parallel sqlite-vec anchor index for v3 graph-native.

Additive module — does NOT touch the existing Ollama-based anchor
matcher or the creative/rigorous axis in embeddings.py. Both stacks
run side-by-side until the v3 base-set / cascade pipeline proves the
sentence-transformers path end-to-end, at which point the Ollama
stack can be retired in a follow-up.

Stack:
  - sqlite-vec virtual table (vec0), 384-dim float32 blob per row
  - sentence-transformers `all-MiniLM-L6-v2`, normalized so L2² ≡
    cosine distance ranking
  - one row per (anchor_id, phrase) where phrase ∈ {canonical_phrase}
    ∪ aliases — so fuzzy KNN lights up on synonym variants

DB file lives at `app/corpus/state/vector_index.db` and is regenerable
from `corpus/objects/*.yaml` + the model, so it is gitignored.
"""
from __future__ import annotations

import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import sqlite_vec
from sentence_transformers import SentenceTransformer

from .corpus import CorpusStore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
DEFAULT_DB_PATH = Path("app/corpus/state/vector_index.db")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class KnnHit:
    anchor_id: str
    phrase: str
    distance: float  # L2² on normalized vectors ≡ 2 - 2·cos_sim


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class VectorStore:
    """Synchronous wrapper around sqlite-vec + sentence-transformers.

    The model is lazy-loaded on first embed call so that import-time
    cost stays near zero. Connection and model are single-instance per
    VectorStore; callers that want a fresh index should construct a
    new instance.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)

        self._model: Optional[SentenceTransformer] = None
        self._init_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        """Create the vec0 virtual table and the metadata sidecar.

        vec0 tables only hold the embedding blob plus rowid; anchor id
        and phrase live in a plain sidecar table joined on rowid.
        """
        self.conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS anchor_vecs USING vec0(
                embedding float[{EMBEDDING_DIM}]
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS anchor_vec_meta (
                rowid     INTEGER PRIMARY KEY,
                anchor_id TEXT NOT NULL,
                phrase    TEXT NOT NULL,
                kind      TEXT NOT NULL
            )
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_anchor_vec_meta_anchor "
            "ON anchor_vec_meta(anchor_id)"
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Model + embedding
    # ------------------------------------------------------------------

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(EMBEDDING_MODEL)
        return self._model

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Return (N, 384) L2-normalized float32 matrix."""
        model = self._get_model()
        vecs = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)

    @staticmethod
    def _to_blob(vec: np.ndarray) -> bytes:
        return struct.pack(f"{EMBEDDING_DIM}f", *vec.tolist())

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Wipe the index. Used by seed_from_corpus for full rebuild."""
        self.conn.execute("DELETE FROM anchor_vecs")
        self.conn.execute("DELETE FROM anchor_vec_meta")
        self.conn.commit()

    def seed_from_corpus(self, corpus: CorpusStore) -> int:
        """Full rebuild from corpus anchors. Returns row count inserted."""
        self.clear()

        rows: list[tuple[str, str, str]] = []  # (anchor_id, phrase, kind)
        for anchor in corpus.anchors.values():
            rows.append((anchor.id, anchor.canonical_phrase, "canonical"))
            for alias in anchor.aliases:
                rows.append((anchor.id, alias, "alias"))

        if not rows:
            return 0

        phrases = [r[1] for r in rows]
        embeddings = self._embed(phrases)

        cur = self.conn.cursor()
        for (anchor_id, phrase, kind), vec in zip(rows, embeddings):
            cur.execute(
                "INSERT INTO anchor_vecs(embedding) VALUES (?)",
                (self._to_blob(vec),),
            )
            rowid = cur.lastrowid
            cur.execute(
                "INSERT INTO anchor_vec_meta(rowid, anchor_id, phrase, kind) "
                "VALUES (?, ?, ?, ?)",
                (rowid, anchor_id, phrase, kind),
            )
        self.conn.commit()
        return len(rows)

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def knn(self, query_text: str, k: int = 5) -> list[KnnHit]:
        """Return top-k nearest (anchor_id, phrase) pairs for query_text.

        Distance is L2² on normalized vectors; lower is better. For
        normalized embeddings L2² == 2 - 2·cos_sim, so cos_sim can be
        recovered as 1 - distance/2 if needed.
        """
        query_vec = self._embed([query_text])[0]
        query_blob = self._to_blob(query_vec)

        cur = self.conn.execute(
            """
            SELECT m.anchor_id, m.phrase, v.distance
            FROM anchor_vecs v
            JOIN anchor_vec_meta m ON m.rowid = v.rowid
            WHERE v.embedding MATCH ?
              AND k = ?
            ORDER BY v.distance
            """,
            (query_blob, k),
        )
        return [
            KnnHit(anchor_id=row[0], phrase=row[1], distance=float(row[2]))
            for row in cur.fetchall()
        ]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def row_count(self) -> int:
        cur = self.conn.execute("SELECT COUNT(*) FROM anchor_vec_meta")
        return int(cur.fetchone()[0])

    def close(self) -> None:
        self.conn.close()
