from src.services.universal_miner import (
    candidate_source_order,
    consolidate_exact,
    deterministic_sequence_edges,
    evidence_source_order,
    detect_source_kind,
    normalize_source,
    normalize_semantic_path,
    segment_source,
)


def test_detects_chat_and_paper_profiles():
    assert detect_source_kind("User: keep evidence\nAssistant: understood") == "chat"
    assert detect_source_kind("Abstract\nA result.\n\nMethods\nA method.\n\nResults\nA finding.") == "paper"


def test_explicit_profile_wins():
    assert detect_source_kind("User: this is actually fiction", "narrative") == "narrative"


def test_chat_normalization_preserves_role_and_turn_locators():
    text = normalize_source("User: A decision\nAssistant: acknowledged", "chat")
    assert "[turn 0] [user]" in text
    assert "[assistant]" in text


def test_segmentation_respects_hard_bound():
    segments = segment_source("A" * 4_500, 2_000)
    assert len(segments) == 3
    assert all(len(segment) <= 2_000 for segment in segments)


def test_exact_consolidation_merges_evidence_and_aliases():
    candidates = [
        {"type": "anchor", "canonical_phrase": "Evidence Contract", "aliases": ["grounding contract"],
         "source_refs": [{"locator": "turn 1"}], "confidence": 0.7},
        {"type": "anchor", "canonical_phrase": " evidence   contract ", "aliases": ["evidence rule"],
         "source_refs": [{"locator": "turn 5"}], "confidence": 0.9},
    ]
    merged = consolidate_exact(candidates)
    assert len(merged) == 1
    assert len(merged[0]["source_refs"]) == 2
    assert merged[0]["confidence"] == 0.9

def test_semantic_paths_reject_transport_locators():
    assert normalize_semantic_path(
        ["podv9", "segment-3", "6. Air Handling"], "Fallback", "podv9"
    ) == ["6. Air Handling"]
    assert normalize_semantic_path(["segment 2", "4.4"], "Pool geometry", "podv9") == [
        "Pool geometry"
    ]
    assert normalize_semantic_path(
        ["Part II — System Architecture", "5. Water Systems"], "Fallback", "podv9"
    ) == ["Part II — System Architecture", "5. Water Systems"]

def test_deterministic_sequence_uses_source_order():
    candidates = [
        {"temp_id": "s2_candidate_002", "type": "slab", "title": "Third"},
        {"temp_id": "s1_candidate_002", "type": "slab", "title": "Second"},
        {"temp_id": "s1_candidate_001", "type": "slab", "title": "First"},
        {"temp_id": "s1_anchor_003", "type": "anchor", "canonical_phrase": "Handle"},
    ]
    edges = deterministic_sequence_edges(candidates)
    assert [(edge["from"], edge["to"]) for edge in edges] == [
        ("First", "Second"), ("Second", "Third")
    ]
    assert all(edge["deterministic"] and edge["type"] == "SEQUENCE" for edge in edges)
    assert candidate_source_order({"_source_order": 7, "temp_id": "s9_x_9"}) == 7

def test_evidence_order_uses_quote_position_not_model_candidate_number():
    segment = "Earlier source paragraph.\n\nLater source paragraph."
    later = {"source_refs": [{"quote": "Later source paragraph."}]}
    earlier = {"source_refs": [{"quote": "Earlier source paragraph."}]}
    assert evidence_source_order(earlier, segment, 0, 99) < evidence_source_order(
        later, segment, 0, 0
    )
