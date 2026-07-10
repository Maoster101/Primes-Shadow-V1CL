"""Unit tests for the shared mining-proposal factories.

These are the first regression tests over the doc-miner glue code. The
factories (`MiningProposal.anchor_from_raw / slab_from_raw / bundle_from_raw`
and `EdgeProposal.from_raw`) are pure `dict -> object` functions — no LLM, no
documents, no I/O — so they are cheap to pin down exactly. Every doc miner
(convo, narrative, outline, paper) now routes its raw-LLM-dict parsing through
them, so a green run here covers the field mapping / confidence filtering /
None-drop behavior for all four.

Tests cover:
  1-3.  anchor_from_raw: happy path, empty phrase -> None, alias scrubbing
  4-6.  slab_from_raw: happy path, title fallback to text[:48], empty -> None
  7-8.  bundle_from_raw: happy path (members -> aliases), empty label -> None
  9.    min_confidence filtering drops low-confidence items
  10.   default_conf applies when confidence is absent / zero
  11-13. EdgeProposal.from_raw: happy path, key aliases, missing endpoint -> None
  14.   from_raw uppercases + defaults the edge type

Run standalone:
    python tests/test_mining_factories.py

Also pytest-compatible:
    pytest tests/test_mining_factories.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.services.convo_miner import MiningProposal, EdgeProposal


# ── anchor_from_raw ───────────────────────────────────────────────

def test_anchor_happy_path():
    p = MiningProposal.anchor_from_raw(
        {"canonical_phrase": "  Constitutional AI  ",
         "aliases": ["CAI", "self-critique"],
         "confidence": 0.9, "justification": "  central concept  "},
        source_topic="alignment", source_pairs=[3, 4],
    )
    assert p is not None
    assert p.proposal_type == "anchor"
    assert p.canonical_phrase == "Constitutional AI"   # stripped
    assert p.aliases == ["CAI", "self-critique"]
    assert p.source_topic == "alignment"
    assert p.source_pairs == [3, 4]
    assert p.confidence == 0.9
    assert p.justification == "central concept"          # stripped


def test_anchor_empty_phrase_is_none():
    assert MiningProposal.anchor_from_raw({"canonical_phrase": "   "}) is None
    assert MiningProposal.anchor_from_raw({}) is None


def test_anchor_alias_scrubbing():
    # Falsy aliases (None, "") are dropped; missing key -> [].
    p = MiningProposal.anchor_from_raw(
        {"canonical_phrase": "X", "aliases": ["a", "", None, "b"]})
    assert p is not None and p.aliases == ["a", "b"]
    p2 = MiningProposal.anchor_from_raw({"canonical_phrase": "Y"})
    assert p2 is not None and p2.aliases == []


# ── slab_from_raw ─────────────────────────────────────────────────

def test_slab_happy_path():
    p = MiningProposal.slab_from_raw(
        {"canonical_text": "  A dense claim.  ", "title": "The Claim",
         "confidence": 0.8},
        source_topic="ch1", source_pairs=[7], default_conf=0.7,
    )
    assert p is not None
    assert p.proposal_type == "slab"
    assert p.canonical_text == "A dense claim."
    assert p.title == "The Claim"
    assert p.source_topic == "ch1"
    assert p.source_pairs == [7]
    assert p.confidence == 0.8


def test_slab_title_falls_back_to_text_prefix():
    long_text = "x" * 100
    p = MiningProposal.slab_from_raw({"canonical_text": long_text})
    assert p is not None
    assert p.title == long_text[:48]          # 48-char fallback
    assert len(p.title) == 48


def test_slab_empty_text_is_none():
    assert MiningProposal.slab_from_raw({"canonical_text": ""}) is None
    assert MiningProposal.slab_from_raw({"title": "orphan title"}) is None


# ── bundle_from_raw ───────────────────────────────────────────────

def test_bundle_happy_path():
    p = MiningProposal.bundle_from_raw(
        {"label": "  Safety Mechanisms  ",
         "members": ["gate", "", "monitor", None],
         "confidence": 0.75},
        source_topic="safety",
    )
    assert p is not None
    assert p.proposal_type == "bundle"
    assert p.label == "Safety Mechanisms"
    assert p.aliases == ["gate", "monitor"]   # members -> aliases, scrubbed
    assert p.source_topic == "safety"


def test_bundle_empty_label_is_none():
    assert MiningProposal.bundle_from_raw({"label": "  "}) is None
    assert MiningProposal.bundle_from_raw({"members": ["a"]}) is None


# ── confidence handling ───────────────────────────────────────────

def test_min_confidence_filtering():
    raw = {"canonical_phrase": "Weak", "confidence": 0.4}
    assert MiningProposal.anchor_from_raw(raw, min_confidence=0.7) is None
    kept = MiningProposal.anchor_from_raw(raw, min_confidence=0.3)
    assert kept is not None and kept.confidence == 0.4


def test_default_conf_when_absent_or_zero():
    # Missing confidence -> default_conf.
    p = MiningProposal.anchor_from_raw({"canonical_phrase": "A"}, default_conf=0.7)
    assert p is not None and p.confidence == 0.7
    # Explicit 0.0 is treated as "no signal" -> default_conf (matches drill miners).
    p2 = MiningProposal.slab_from_raw(
        {"canonical_text": "t", "confidence": 0.0}, default_conf=0.5)
    assert p2 is not None and p2.confidence == 0.5


# ── EdgeProposal.from_raw ─────────────────────────────────────────

def test_edge_happy_path():
    e = EdgeProposal.from_raw(
        {"type": "conflicts", "from": "  A  ", "to": "  B  ",
         "confidence": 0.9, "justification": "  they clash  "})
    assert e is not None
    assert e.edge_type == "CONFLICTS"        # uppercased
    assert e.from_label == "A"
    assert e.to_label == "B"
    assert e.confidence == 0.9
    assert e.justification == "they clash"


def test_edge_accepts_key_aliases():
    # from_label / to_label / edge_type are accepted as alternate keys.
    e = EdgeProposal.from_raw(
        {"edge_type": "supports", "from_label": "X", "to_label": "Y"})
    assert e is not None
    assert e.edge_type == "SUPPORTS"
    assert e.from_label == "X" and e.to_label == "Y"


def test_edge_missing_endpoint_is_none():
    assert EdgeProposal.from_raw({"type": "LINKS", "from": "A"}) is None
    assert EdgeProposal.from_raw({"type": "LINKS", "to": "B"}) is None
    assert EdgeProposal.from_raw({"from": "A", "to": ""}) is None


def test_edge_default_type_and_conf():
    e = EdgeProposal.from_raw({"from": "A", "to": "B"})
    assert e is not None
    assert e.edge_type == "LINKS"            # default_type
    assert e.confidence == 0.7               # default_conf


# ── standalone runner (mirrors tests/test_gate_eval.py) ───────────

ALL_TESTS = [
    test_anchor_happy_path,
    test_anchor_empty_phrase_is_none,
    test_anchor_alias_scrubbing,
    test_slab_happy_path,
    test_slab_title_falls_back_to_text_prefix,
    test_slab_empty_text_is_none,
    test_bundle_happy_path,
    test_bundle_empty_label_is_none,
    test_min_confidence_filtering,
    test_default_conf_when_absent_or_zero,
    test_edge_happy_path,
    test_edge_accepts_key_aliases,
    test_edge_missing_endpoint_is_none,
    test_edge_default_type_and_conf,
]


def main() -> int:
    print(f"\n{'='*60}\nMining-factory Unit Tests\n{'='*60}\n")
    passed, failed = 0, []
    for fn in ALL_TESTS:
        try:
            fn()
            print(f"  ok:   {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {fn.__name__} -- {e}")
            failed.append(fn.__name__)
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{len(ALL_TESTS)} passed, {len(failed)} failed")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    print(f"{'='*60}\n")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
