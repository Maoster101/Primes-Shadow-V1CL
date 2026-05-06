"""Fuzzy label-to-id resolution for mined edges.

Mining produces edge proposals with from_label / to_label as strings
(e.g. "Standard English", "Proprietary Symbolic Syntax"). To persist
these as Edge records the labels need to resolve to actual node IDs
in the corpus. Strict lowercase matching misses ~10% of edges in
practice — the model emits labels with case variations, trailing
punctuation, pluralisation, or slight phrasing drift from the actual
canonical_phrase / title.

This module provides ``resolve_label`` with a three-tier strategy:

  1. Exact match (lowercase, stripped) — same as strict resolver
  2. Substring containment with length-ratio safeguard
  3. difflib SequenceMatcher ratio ≥ 0.85

Callers get back ``(node_id, strategy)`` so logging can attribute
which path won. Strategy values: "exact" | "substring" | "ratio" |
"miss".

This was originally inlined in ``draft_manager.py`` for in-chat
edge resolution. Lifted here so the AI-Mine flow (``api/mining.py``
``/push-mined``) and harness scripts (``scripts/mine_then_synth.py``)
can share the same logic — three copies that drift was the previous
state.

Tuning thresholds are module-level constants. Override via the
respective env vars for experimentation:

  PS_FUZZY_MIN_LEN          (default 5)
  PS_FUZZY_SUBSTR_MIN_RATIO (default 0.5)
  PS_FUZZY_RATIO_THRESHOLD  (default 0.85)
"""
from __future__ import annotations

import os as _os
from difflib import SequenceMatcher
from typing import Iterable, Optional


# Minimum length (in characters) for fuzzy strategies to engage.
# Below this we only do exact match — substring/ratio matching on
# short strings is too noisy (common words like "family", "mirror"
# would match many unrelated catalog entries).
FUZZY_MIN_LEN = int(_os.environ.get("PS_FUZZY_MIN_LEN", "5"))

# Substring match requires the shorter string to be at least this
# fraction of the longer (length ratio guard) so we don't match
# arbitrary subwords inside long titles.
FUZZY_SUBSTR_MIN_RATIO = float(_os.environ.get("PS_FUZZY_SUBSTR_MIN_RATIO", "0.5"))

# difflib.SequenceMatcher ratio threshold. 0.85 is fairly tight —
# catches plural/missing-punctuation cases but rejects semantically
# different titles that happen to share a word or two.
FUZZY_RATIO_THRESHOLD = float(_os.environ.get("PS_FUZZY_RATIO_THRESHOLD", "0.85"))


def normalize_key(label: str) -> str:
    """Normalise a label for use as an index key.

    Lowercase + strip whitespace. We deliberately do NOT strip
    punctuation here — punctuation differences are the SequenceMatcher
    layer's job. Stripping punct before indexing would mean the index
    can't distinguish "OLI" (acronym) from "OLI." (sentence-ended).
    """
    return label.strip().lower()


def build_index(entries: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Build a label-to-id index from (label, node_id) pairs.

    Multiple labels can map to the same id (canonical_phrase + aliases).
    First registration wins on collision (so canonical_phrase takes
    precedence over aliases when both normalise to the same key).
    """
    out: dict[str, str] = {}
    for label, nid in entries:
        if not label:
            continue
        key = normalize_key(label)
        if not key:
            continue
        out.setdefault(key, nid)
    return out


def resolve_label(
    label: str,
    index: dict[str, str],
) -> tuple[Optional[str], str]:
    """Resolve ``label`` to a node_id using exact → substring → ratio.

    Returns ``(node_id, strategy)`` — strategy is one of ``"exact"``,
    ``"substring"``, ``"ratio"``, or ``"miss"`` so callers can log
    which path resolved (or attribute drops to a strategy gap).

    Strategy priority:
      1. Exact lowercase match (same as strict resolver).
      2. Substring containment, both directions, with length-ratio
         guard: the shorter string must be at least
         FUZZY_SUBSTR_MIN_RATIO of the longer AND >= FUZZY_MIN_LEN
         chars — otherwise short common words like "family" or
         "mirror" would match every label containing them.
      3. difflib.SequenceMatcher ratio >= FUZZY_RATIO_THRESHOLD —
         catches pluralisation differences, missing trailing
         punctuation, small typos. Requires query_len >= FUZZY_MIN_LEN.
    """
    if not label:
        return None, "miss"
    key = normalize_key(label)
    if not key:
        return None, "miss"
    if key in index:
        return index[key], "exact"
    if len(key) < FUZZY_MIN_LEN:
        # Too short for safe fuzzy; don't risk false positives.
        return None, "miss"

    # Substring + ratio: scan all catalog entries once, pick best match.
    best_ratio = 0.0
    best_id: Optional[str] = None
    substring_hit: Optional[tuple[str, str]] = None  # (cat_key, node_id)

    for cat_key, nid in index.items():
        if len(cat_key) < FUZZY_MIN_LEN:
            continue
        # Substring containment, both directions
        if key in cat_key or cat_key in key:
            shorter = min(len(key), len(cat_key))
            longer = max(len(key), len(cat_key))
            if longer > 0 and shorter / longer >= FUZZY_SUBSTR_MIN_RATIO:
                if substring_hit is None or len(cat_key) > len(substring_hit[0]):
                    # Prefer the LONGER catalog entry when multiple
                    # substring-match — more specific wins over more general.
                    substring_hit = (cat_key, nid)
        # Ratio score
        r = SequenceMatcher(None, key, cat_key).ratio()
        if r > best_ratio:
            best_ratio = r
            best_id = nid

    if substring_hit:
        return substring_hit[1], "substring"
    if best_ratio >= FUZZY_RATIO_THRESHOLD and best_id is not None:
        return best_id, "ratio"
    return None, "miss"
