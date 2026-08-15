# Dormant Architecture: Things We Built or Considered, Shelved, and Why

Single source of truth for design rationale on subsystems we built but
removed, OR considered but did not build. Keeps the "why we don't have
this yet" knowledge findable instead of scattered across commit
messages and chat history.

Each entry describes:
- **What** the thing is/was
- **Why** we built or considered it
- **Why** we stalled (or never started)
- **Revival triggers** — specific signals that would make this worth
  picking back up
- **Recovery** — git refs or pointers to find the code if it once existed

---

## sqlite-vec + sentence-transformers anchor index

**Status:** Built March-April 2026, removed in commit `6c031f3` (April 2026).
Lived in `src/services/vector_store.py` for ~6 weeks of dormancy with
zero callers before being pruned.

**What it was**

A parallel embedding/KNN stack independent of Ollama:

- `sqlite-vec` virtual table (vec0) for KNN over a 384-dim float32 blob
  per row, persisted to disk
- `sentence-transformers all-MiniLM-L6-v2` for embedding,
  L2-normalized so squared L2 distance ≡ cosine
- One row per `(anchor_id, phrase)` where phrase ∈
  `{canonical_phrase} ∪ aliases`, so KNN naturally fires on synonym
  variants without explicit aliasing logic

API surface (~210 lines): `VectorStore` class with `seed_from_corpus`,
`knn`, `clear`, `close`, `row_count`. DB at
`app/corpus/state/vector_index.db`, regenerable from `corpus/objects/*.yaml`.

**Why it was built**

1. **Disk-persistent embedding index.** Live anchor/slab matchers
   re-embed on every server restart (~2-5s warm cost). At larger scale
   that becomes painful.
2. **Independence from Ollama.** The live path crashes the entire
   Prime's Shadow startup if Ollama is down (`httpx.ConnectError`
   propagates up). An in-process embedding stack would survive that.
3. **KNN at scale.** Live retrieval uses flat numpy cosine
   (matrix-vector product, O(n) per query). At ~1k nodes that's
   sub-millisecond; at 100k it would be measurable.
4. **Quality alternative.** MiniLM is well-tuned for short-phrase
   semantic match — anchor canonical phrases and aliases are
   short-phrase content.

**Why it stalled**

The "retire Ollama anchor matching, migrate to sqlite-vec" plan
described in the original docstring never had a clear forcing
function. The Ollama-based path scaled fine through every corpus
growth event:

- 60 anchors → fine
- 250 anchors → fine
- 429 anchors (current) → fine
- Warm-cache time stayed ~2-5s, never user-visible

Adding sqlite-vec + sentence-transformers as runtime dependencies is
non-trivial — sqlite-vec needs C extensions, sentence-transformers
pulls in torch. The existing path is a lighter dependency footprint.

**Revival triggers**

Concrete signals that should reopen the question:

1. **Corpus size > ~10k slabs/anchors.** Flat numpy KNN starts to
   become hot. If users notice query latency growing visibly during a
   chat turn, this is the symptom.
2. **Want to deploy without Ollama** (or with Ollama optional).
   Currently Prime's Shadow won't start without Ollama reachable.
   Some deployment paths might want graceful degradation — embeddings
   from local sentence-transformers, generation deferred until Ollama
   shows up.
3. **Persistent embedding cache becomes valuable.** If startup warm-
   cache time crosses a meaningful threshold (>30s?) on any user's
   machine, on-disk persistence is worth the complexity.
4. **Need for richer query types.** sqlite-vec supports more
   sophisticated queries than flat numpy (filtered KNN, range queries,
   composite indices). If retrieval grows beyond simple cosine
   similarity, this is the platform.

**Recovery**

```
git show 6c031f3^:src/services/vector_store.py > src/services/vector_store.py
```

That brings back the full ~210-line module. Then re-add `sqlite-vec`
and `sentence-transformers` to `requirements.txt`. The DB file at
`app/corpus/state/vector_index.db` was also removed in `6c031f3`
but can be regenerated from corpus on first run via
`VectorStore.seed_from_corpus()`.

The deps:
```
sqlite-vec>=0.1.0
sentence-transformers>=3.0.0
```

---

## (Future entries here)

When something gets deferred for a clear reason, add a section above
this line. Goal: every "why don't we have X yet" question should be
answerable by reading this file rather than re-deriving the rationale
from scratch.
