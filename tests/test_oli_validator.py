"""Unit tests for the OLI Validator (§26.4 — Post-Generation Enforcement).

Tests cover:
  1. OLI OFF bypass — no enforcement when mode is OFF
  2. Pattern-based hard violations (OLI-0 epistemic floor, plausibility filling)
  3. Pattern-based soft violations (OLI-1 domain separation, OLI-2 interaction style, OLI-3 memory)
  4. Claim admissibility structural check (OLI-0.5)
  5. LI-4 slope detection — 3-step escalation state machine
  6. Decision tree: PASS / FLAGGED / REGENERATE / BLOCK
  7. User overrides skip specified layers
  8. Correction guidance selection

Run standalone:
    python tests/test_oli_validator.py

Also pytest-compatible:
    pytest tests/test_oli_validator.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make project importable without installing.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.enums import OLIMode, ValidationStatus
from src.services.oli_validator import (
    validate_output,
    _count_untagged_claims,
    ValidationResult,
    LAYER_CHECKS,
    _LI4_SLOPE_RE,
)

# ── Helpers ─────────────────────────────────────────────────────

_FAILURES: list[str] = []


def _assert(condition: bool, msg: str) -> None:
    if not condition:
        _FAILURES.append(msg)
        print(f"  FAIL: {msg}")
    else:
        print(f"  ok:   {msg}")


# ── 1. OLI OFF bypass ──────────────────────────────────────────

def test_oli_off_always_passes():
    """When OLI is OFF, the validator should return PASS regardless of content."""
    toxic = (
        "It is known that studies show experts agree that science has proven "
        "you should do step 1: build the thing, then deploy, finally celebrate. "
        "As we discussed previously, my training tells me you seem upset."
    )
    result = validate_output(toxic, OLIMode.OFF)
    _assert(result.status == ValidationStatus.PASS, "OLI OFF -> always PASS")
    _assert(len(result.flags) == 0, "OLI OFF -> zero flags")


# ── 2. OLI-0 hard violations (epistemic floor + plausibility) ──

def test_epistemic_floor_hard_violation():
    """Certainty markers on unverified claims trigger OLI-0 hard violation."""
    text = "It is known that the system uses 128k context windows for all queries."
    result = validate_output(text, OLIMode.ON)
    _assert(result.status in (ValidationStatus.REGENERATE, ValidationStatus.FLAGGED),
            "Epistemic floor phrase -> non-PASS")
    layers = [f["layer"] for f in result.flags]
    _assert("OLI-0" in layers, "'it is known that' -> OLI-0 flag")


def test_plausibility_filling_hard_violation():
    """Gap-fill phrases trigger OLI-0 plausibility filling violation."""
    text = "We can safely assume that the architecture handles edge cases gracefully."
    result = validate_output(text, OLIMode.ON)
    layers = [f["layer"] for f in result.flags]
    _assert("OLI-0" in layers, "'we can safely assume' -> OLI-0 flag")


def test_multiple_hard_violations_first_attempt():
    """Multiple OLI-0 violations on first attempt -> REGENERATE."""
    text = (
        "It is known that the system is perfect. "
        "We can safely assume it never fails."
    )
    result = validate_output(text, OLIMode.ON, is_retry=False)
    _assert(result.status == ValidationStatus.REGENERATE,
            "Multiple hard violations, first attempt -> REGENERATE")
    _assert(result.correction_guidance is not None,
            "REGENERATE includes correction guidance")


def test_hard_violation_on_retry_blocks():
    """Hard violation persisting after retry -> BLOCK."""
    text = "Science has proven that this approach is optimal."
    result = validate_output(text, OLIMode.ON, is_retry=True)
    _assert(result.status == ValidationStatus.BLOCK,
            "Hard violation on retry -> BLOCK")


# ── 3. Soft violations (OLI-1, OLI-2, OLI-3) ──────────────────

def test_domain_separation_soft_violation():
    """OLI-1: unsolicited affect interpretation is a soft (overridable) flag."""
    text = "You seem to feel frustrated about the deployment process."
    result = validate_output(text, OLIMode.ON)
    _assert(result.status == ValidationStatus.FLAGGED,
            "Domain separation -> FLAGGED (not REGENERATE)")
    flags = [f for f in result.flags if f["layer"] == "OLI-1"]
    _assert(len(flags) > 0, "OLI-1 flag present")
    _assert(flags[0]["overridable"] is True, "OLI-1 is overridable")


def test_interaction_style_soft_violation():
    """OLI-2: padding/moralising phrases are soft flags."""
    text = "It's important to remember that testing is crucial for quality."
    result = validate_output(text, OLIMode.ON)
    layers = [f["layer"] for f in result.flags]
    _assert("OLI-2" in layers, "'it's important to remember' -> OLI-2 flag")


def test_memory_persistence_soft_violation():
    """OLI-3: cross-chat context assumptions are soft flags."""
    text = "As we discussed previously, the architecture uses a three-layer system."
    result = validate_output(text, OLIMode.ON)
    layers = [f["layer"] for f in result.flags]
    _assert("OLI-3" in layers, "'as we discussed previously' -> OLI-3 flag")


def test_soft_only_means_flagged_not_regenerate():
    """When ONLY soft violations exist, status is FLAGGED (not REGENERATE)."""
    text = "You seem to feel that this is a really important point to consider."
    result = validate_output(text, OLIMode.ON)
    _assert(result.status == ValidationStatus.FLAGGED,
            "Soft-only violations -> FLAGGED")


# ── 4. Claim admissibility (OLI-0.5) ───────────────────────────

def test_untagged_claims_trigger_oli05():
    """Multiple substantial untagged claims trigger OLI-0.5 hard violation."""
    # Build text with many >50-char sentences that aren't questions or greetings
    claims = [
        "The neural network processes information through multiple hidden layers efficiently",
        "Gradient descent optimises the loss function by iteratively adjusting all weights",
        "Backpropagation computes partial derivatives using the chain rule recursively",
        "The attention mechanism allows the model to focus on relevant input tokens selectively",
        "Transformer architectures have revolutionised natural language processing completely",
    ]
    text = ". ".join(claims) + "."
    result = validate_output(text, OLIMode.ON)
    layers = [f["layer"] for f in result.flags]
    _assert("OLI-0.5" in layers, f"5 untagged claims -> OLI-0.5 (got layers: {layers})")


def test_tagged_claims_pass():
    """Claims with admissibility tags don't trigger OLI-0.5."""
    text = (
        "[FACT] The system uses GPT-OSS 20B as the primary face model. "
        "[INFERENCE] This likely improves response quality for complex queries. "
        "[HYPOTHESIS] A larger model might further improve reasoning capabilities."
    )
    count = _count_untagged_claims(text)
    _assert(count == 0, f"All tagged -> 0 untagged claims (got {count})")


def test_questions_not_counted_as_claims():
    """Questions are not counted as untagged claims."""
    text = (
        "What happens when the model encounters an out-of-distribution input? "
        "How does the system handle concurrent requests from multiple users? "
        "Could this architecture scale to handle millions of requests per day?"
    )
    count = _count_untagged_claims(text)
    _assert(count == 0, f"Questions -> 0 untagged (got {count})")


def test_short_sentences_not_counted():
    """Sentences under the minimum length threshold are not counted."""
    text = "Yes. No. OK. Sure. Got it. Fine. Right."
    count = _count_untagged_claims(text)
    _assert(count == 0, f"Short sentences -> 0 untagged (got {count})")


# ── 5. LI-4 slope detection (3-step escalation) ────────────────

def test_li4_step1_soft_flag():
    """LI-4 step 1: first slope hit is a soft flag."""
    text = (
        "You should start by reviewing the architecture. "
        "You need to understand the data model first. "
        "Here's a step-by-step guide to deploying it."
    )
    result = validate_output(text, OLIMode.ON, consecutive_slope_hits=0)
    li4_flags = [f for f in result.flags if f["layer"] == "LI-4"]
    _assert(len(li4_flags) > 0, "LI-4 slope detected at step 1")
    if li4_flags:
        _assert(li4_flags[0]["overridable"] is True, "Step 1 is overridable (soft)")
        _assert(li4_flags[0].get("escalation_step") == 1, "escalation_step == 1")


def test_li4_step2_stronger_soft():
    """LI-4 step 2: second consecutive hit is a stronger soft flag."""
    text = (
        "You must configure the environment variables correctly. "
        "Your next step should be to run the test suite. "
        "Make sure you validate the output carefully."
    )
    result = validate_output(text, OLIMode.ON, consecutive_slope_hits=1)
    li4_flags = [f for f in result.flags if f["layer"] == "LI-4"]
    _assert(len(li4_flags) > 0, "LI-4 slope detected at step 2")
    if li4_flags:
        _assert(li4_flags[0]["overridable"] is True, "Step 2 still overridable")
        _assert(li4_flags[0].get("escalation_step") == 2, "escalation_step == 2")


def test_li4_step3_hard_violation():
    """LI-4 step 3: third consecutive hit becomes a hard violation."""
    text = (
        "You have to implement the caching layer immediately. "
        "I urge you to prioritise this over other tasks. "
        "Don't forget to update the configuration files."
    )
    result = validate_output(text, OLIMode.ON, consecutive_slope_hits=2)
    li4_flags = [f for f in result.flags if f["layer"] == "LI-4"]
    _assert(len(li4_flags) > 0, "LI-4 slope detected at step 3")
    if li4_flags:
        _assert(li4_flags[0]["overridable"] is False, "Step 3 is hard (non-overridable)")
        _assert(li4_flags[0].get("escalation_step") == 3, "escalation_step == 3")
    _assert(result.status == ValidationStatus.REGENERATE,
            "Step 3 LI-4 hard violation -> REGENERATE")


def test_li4_user_override_skips():
    """User override of LI-4 skips the slope check entirely."""
    text = (
        "You should do step 1: configure the database. "
        "Then you need to run migrations. "
        "Finally, deploy the service."
    )
    result = validate_output(
        text, OLIMode.ON,
        user_overrides={"LI-4"},
        consecutive_slope_hits=5,
    )
    li4_flags = [f for f in result.flags if f["layer"] == "LI-4"]
    _assert(len(li4_flags) == 0, "LI-4 overridden -> no LI-4 flags")


def test_li4_legacy_oli4_override_also_works():
    """Legacy 'OLI-4' override key also skips LI-4 slope check."""
    text = "You must do this. You need to do that. Make sure you finish."
    result = validate_output(
        text, OLIMode.ON,
        user_overrides={"OLI-4"},
        consecutive_slope_hits=5,
    )
    li4_flags = [f for f in result.flags if f["layer"] == "LI-4"]
    _assert(len(li4_flags) == 0, "OLI-4 legacy override -> no LI-4 flags")


# ── 6. User overrides on soft layers ───────────────────────────

def test_override_skips_oli1():
    """Overriding OLI-1 prevents domain separation flags."""
    text = "You seem to feel frustrated about the deployment."
    result = validate_output(text, OLIMode.ON, user_overrides={"OLI-1"})
    oli1_flags = [f for f in result.flags if f["layer"] == "OLI-1"]
    _assert(len(oli1_flags) == 0, "OLI-1 overridden -> no OLI-1 flags")


def test_override_does_not_skip_hard_layers():
    """Cannot override OLI-0 (non-overridable)."""
    text = "It is known that this system is perfect."
    result = validate_output(text, OLIMode.ON, user_overrides={"OLI-0"})
    # OLI-0 is non-overridable — override set should be ignored
    oli0_flags = [f for f in result.flags if f["layer"] == "OLI-0"]
    _assert(len(oli0_flags) > 0, "OLI-0 is non-overridable, override ignored")


# ── 7. Clean text passes ───────────────────────────────────────

def test_clean_text_passes():
    """Text with no violations returns PASS."""
    text = (
        "[FACT] The system uses a three-layer validation architecture. "
        "[INFERENCE] This may improve epistemic integrity in long sessions. "
        "The design prioritises explicit uncertainty marking."
    )
    result = validate_output(text, OLIMode.ON)
    _assert(result.status == ValidationStatus.PASS,
            f"Clean text -> PASS (got {result.status})")
    _assert(len(result.flags) == 0, "Clean text -> zero flags")


# ── 8. Correction guidance ─────────────────────────────────────

def test_regenerate_has_correction_guidance():
    """REGENERATE result always includes correction guidance."""
    text = "It is known that science has proven this approach works."
    result = validate_output(text, OLIMode.ON, is_retry=False)
    _assert(result.status == ValidationStatus.REGENERATE,
            "Hard violation -> REGENERATE")
    _assert(result.correction_guidance is not None,
            "REGENERATE includes correction guidance")
    _assert(len(result.correction_guidance) > 20,
            "Correction guidance is substantive")


# ── 9. Edge cases ──────────────────────────────────────────────

def test_empty_text_passes():
    """Empty text should pass (no claims to violate)."""
    result = validate_output("", OLIMode.ON)
    _assert(result.status == ValidationStatus.PASS, "Empty text -> PASS")


def test_mixed_hard_and_soft():
    """Text with both hard and soft violations -> REGENERATE (hard wins)."""
    text = (
        "It is known that the system is robust. "  # OLI-0 hard
        "You seem to feel uncertain about this."   # OLI-1 soft
    )
    result = validate_output(text, OLIMode.ON, is_retry=False)
    _assert(result.status == ValidationStatus.REGENERATE,
            "Mixed hard+soft -> REGENERATE (hard dominates)")


# ── Runner ──────────────────────────────────────────────────────

ALL_TESTS = [
    test_oli_off_always_passes,
    test_epistemic_floor_hard_violation,
    test_plausibility_filling_hard_violation,
    test_multiple_hard_violations_first_attempt,
    test_hard_violation_on_retry_blocks,
    test_domain_separation_soft_violation,
    test_interaction_style_soft_violation,
    test_memory_persistence_soft_violation,
    test_soft_only_means_flagged_not_regenerate,
    test_untagged_claims_trigger_oli05,
    test_tagged_claims_pass,
    test_questions_not_counted_as_claims,
    test_short_sentences_not_counted,
    test_li4_step1_soft_flag,
    test_li4_step2_stronger_soft,
    test_li4_step3_hard_violation,
    test_li4_user_override_skips,
    test_li4_legacy_oli4_override_also_works,
    test_override_skips_oli1,
    test_override_does_not_skip_hard_layers,
    test_clean_text_passes,
    test_regenerate_has_correction_guidance,
    test_empty_text_passes,
    test_mixed_hard_and_soft,
]


def main() -> int:
    global _FAILURES
    _FAILURES = []
    print(f"\n{'='*60}")
    print("OLI Validator Unit Tests")
    print(f"{'='*60}\n")

    passed = 0
    failed_tests = []

    for test_fn in ALL_TESTS:
        before = len(_FAILURES)
        print(f"[{test_fn.__name__}]")
        try:
            test_fn()
        except Exception as e:
            _FAILURES.append(f"{test_fn.__name__} raised: {e}")
            print(f"  FAIL: exception — {e}")
        after = len(_FAILURES)
        if after == before:
            passed += 1
        else:
            failed_tests.append(test_fn.__name__)

    total = len(ALL_TESTS)
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} passed, {total - passed} failed")
    if failed_tests:
        print(f"Failed: {', '.join(failed_tests)}")
    print(f"{'='*60}\n")
    return 0 if not _FAILURES else 1


if __name__ == "__main__":
    sys.exit(main())
