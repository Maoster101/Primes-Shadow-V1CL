"""Prompt coherence tests — Prime's Shadow three-tier constitutional prompts.

Verifies semantic alignment between the four prompt constants and the
enforcement code that backs them. Fails loudly if any doctrinal drift
has been introduced between tiers.

Run standalone (no pytest dependency):
    python tests/test_prompt_coherence.py

The tests fall into six groups:
  1. Import health + size envelopes
  2. BMD council parity across all tiers
  3. LI / OLI layer coverage and separation
  4. OFF mode guardrails parity (frontier <-> scaffold)
  5. Doctrinal invariants (ephemeral buffer, LI-4 escalation, final override)
  6. Cross-code alignment (prompts <-> drift_monitor <-> oli_validator <-> enums)

When a test fails, the script prints the failing assertion and continues,
so you see all the drift in one pass rather than one error at a time.
Exits with status 1 if any test fails, 0 if all pass.
"""

from __future__ import annotations

import re
import sys
import traceback
from pathlib import Path

# Make the project importable without installing it.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.prompts.oli_constitutional import (  # noqa: E402
    BMD_SCAFFOLD,
    FRONTIER_CONSTITUTION_PROMPT,
    OLI_BOOTSTRAP_PROMPT,
    OLI_CONSTITUTIONAL_PROMPT,
)


# ── Shared fixtures ─────────────────────────────────────────────

COUNCIL_MEMBERS = ("Brennan", "Zack", "Booth", "Angela", "Hodgins")

COUNCIL_WEIGHTS = {
    "Brennan": "0.60",
    "Zack":    "0.15",
    "Booth":   "0.10",
    "Angela":  "0.10",
    "Hodgins": "0.05",
}

# Quoted DENY examples that must appear in BOTH tiers that enforce OFF mode.
# The frontier prompt and BMD_SCAFFOLD both list these as examples of
# "vague authority" that is denied in OFF mode.
DENY_QUOTED_EXAMPLES_SHARED = ("research suggests", "logs indicate")

# The four tiers, named for error messages.
TIERS: dict[str, str] = {
    "FRONTIER_CONSTITUTION_PROMPT": FRONTIER_CONSTITUTION_PROMPT,
    "OLI_BOOTSTRAP_PROMPT":         OLI_BOOTSTRAP_PROMPT,
    "OLI_CONSTITUTIONAL_PROMPT":    OLI_CONSTITUTIONAL_PROMPT,
    "BMD_SCAFFOLD":                 BMD_SCAFFOLD,
}


# ── 1. Import health + size envelopes ───────────────────────────

def test_all_prompts_are_nonempty_strings() -> None:
    for name, prompt in TIERS.items():
        assert isinstance(prompt, str), f"{name} is not a str"
        assert len(prompt) > 0, f"{name} is empty"


def test_size_envelopes() -> None:
    # Loose upper bounds — fail only if a tier balloons past its intent.
    # (Lower bounds exist to catch an accidental truncation.)
    bounds = {
        "FRONTIER_CONSTITUTION_PROMPT": (4_000, 10_000),
        "OLI_BOOTSTRAP_PROMPT":         (3_000,  6_000),
        "OLI_CONSTITUTIONAL_PROMPT":    (15_000, 30_000),
        "BMD_SCAFFOLD":                 (1_500,  5_000),
    }
    for name, (lo, hi) in bounds.items():
        size = len(TIERS[name])
        assert lo <= size <= hi, (
            f"{name} is {size} chars, expected {lo}..{hi} — "
            "either it was truncated or has bloated past its deployment target"
        )


def test_no_literal_null_bytes() -> None:
    for name, prompt in TIERS.items():
        assert "\x00" not in prompt, f"{name} contains a NUL byte"


# ── 2. BMD council parity across all tiers ──────────────────────

def test_all_council_members_in_every_tier() -> None:
    for name, prompt in TIERS.items():
        for member in COUNCIL_MEMBERS:
            assert member in prompt, (
                f"{name} is missing council member {member!r}"
            )


def test_council_weights_match_where_declared() -> None:
    # The full prompt, bootstrap, scaffold, and frontier all declare weights.
    # They must all agree on the numeric values.
    weight_bearing = (
        "FRONTIER_CONSTITUTION_PROMPT",
        "OLI_BOOTSTRAP_PROMPT",
        "OLI_CONSTITUTIONAL_PROMPT",
        "BMD_SCAFFOLD",
    )
    for name in weight_bearing:
        prompt = TIERS[name]
        for member, weight in COUNCIL_WEIGHTS.items():
            # Find the member line, make sure the weight is nearby.
            idx = prompt.find(member)
            assert idx >= 0, f"{name} missing {member}"
            window = prompt[idx : idx + 120]
            assert weight in window, (
                f"{name}: expected {member} to carry weight {weight}, "
                f"but did not find it within 120 chars of the name. "
                f"Got: {window[:80]!r}"
            )


def test_hodgins_never_solo_rule() -> None:
    # The "Hodgins never solo / never epistemic override" rule is core
    # to v1.3 and must appear in both the frontier tier and the scaffold.
    required = ("FRONTIER_CONSTITUTION_PROMPT", "BMD_SCAFFOLD")
    for name in required:
        body = TIERS[name].lower()
        hit_solo = "never_solo" in body or "never solo" in body
        hit_override = (
            "never epistemic override" in body
            or "not an epistemic override" in body
            or "never an epistemic override" in body
        )
        assert hit_solo, f"{name} is missing the Hodgins never-solo rule"
        assert hit_override, f"{name} is missing the Hodgins never-override rule"


def test_pair_rules_declared_in_scaffold_and_frontier() -> None:
    required_pairs = (
        ("Brennan", "Zack"),
        ("Zack", "Hodgins"),
        ("Angela", "Booth"),
    )
    for name in ("FRONTIER_CONSTITUTION_PROMPT", "BMD_SCAFFOLD"):
        prompt = TIERS[name]
        for a, b in required_pairs:
            # Look for either "A + B" or "A+B" or "A [punct] B" within
            # a pairs block — cheap proximity check.
            pattern = re.compile(
                rf"{a}\s*[+]\s*{b}|{b}\s*[+]\s*{a}",
                re.IGNORECASE,
            )
            assert pattern.search(prompt), (
                f"{name} missing pair rule {a}+{b}"
            )


# ── 3. LI / OLI layer coverage and separation ───────────────────

def test_oli_layer_coverage_in_full_prompt() -> None:
    # The full local prompt must define OLI-0 through OLI-9, plus OLI-0.5.
    expected = [f"OLI-{i}" for i in range(0, 10)] + ["OLI-0.5"]
    for label in expected:
        assert label in OLI_CONSTITUTIONAL_PROMPT, (
            f"OLI_CONSTITUTIONAL_PROMPT missing layer {label}"
        )


def test_frontier_oli_layer_coverage() -> None:
    # Frontier tier uses compact 'OLI0'..'OLI9' plus 'OLI0.5' spelling.
    expected = [f"OLI{i}" for i in range(0, 10)] + ["OLI0.5"]
    for label in expected:
        assert label in FRONTIER_CONSTITUTION_PROMPT, (
            f"FRONTIER_CONSTITUTION_PROMPT missing layer {label}"
        )


def test_li_layer_coverage_in_all_relevant_tiers() -> None:
    # LI-0..LI-4 must appear in the local tiers and the scaffold.
    for name in ("OLI_CONSTITUTIONAL_PROMPT", "OLI_BOOTSTRAP_PROMPT", "BMD_SCAFFOLD"):
        prompt = TIERS[name]
        for i in range(0, 5):
            assert f"LI-{i}" in prompt, f"{name} missing LI-{i}"

    # Frontier uses 'LI0'..'LI4' spelling.
    for i in range(0, 5):
        assert f"LI{i}" in FRONTIER_CONSTITUTION_PROMPT, (
            f"FRONTIER_CONSTITUTION_PROMPT missing LI{i}"
        )


def test_li_oli_non_conflation_warning_in_local_tiers() -> None:
    # The local tiers warn against conflating LI and OLI — this is a
    # structural correctness invariant, not stylistic.
    for name in ("OLI_CONSTITUTIONAL_PROMPT", "OLI_BOOTSTRAP_PROMPT"):
        body = TIERS[name].lower()
        assert "never conflate" in body or "never merge or conflate" in body, (
            f"{name} is missing the LI/OLI non-conflation warning"
        )


def test_non_overridable_layers_declared() -> None:
    # OLI-0, OLI-0.5, OLI-6 must be marked non-overridable in the local
    # prompts (bootstrap and full). The frontier tier is shorter and is
    # allowed to describe the rule differently.
    for name in ("OLI_CONSTITUTIONAL_PROMPT", "OLI_BOOTSTRAP_PROMPT"):
        body = TIERS[name]
        for layer in ("OLI-0", "OLI-0.5", "OLI-6"):
            assert layer in body, f"{name} missing {layer}"
        # The phrase "NEVER overridable" or "NOT OVERRIDABLE" must appear.
        body_lower = body.lower()
        assert (
            "never overridable" in body_lower
            or "not overridable" in body_lower
        ), f"{name} never states a non-overridable rule"


# ── 4. OFF mode guardrails parity (frontier <-> scaffold) ───────

def test_off_mode_deny_examples_present_in_both_tiers() -> None:
    # "research suggests" and "logs indicate" are the canonical shared
    # examples of vague authority. They must appear in the frontier prompt.
    # BMD_SCAFFOLD delegates guardrails to the CORPUS BASE SET (slabs),
    # so it no longer inlines these examples directly.
    for example in DENY_QUOTED_EXAMPLES_SHARED:
        assert example in FRONTIER_CONSTITUTION_PROMPT, (
            f"FRONTIER_CONSTITUTION_PROMPT missing deny example {example!r}"
        )
    # BMD_SCAFFOLD must reference the corpus delegation
    assert "corpus base set" in BMD_SCAFFOLD.lower(), (
        "BMD_SCAFFOLD missing CORPUS BASE SET delegation pointer"
    )


def test_off_mode_core_concepts_parity() -> None:
    # Frontier prompt must mention these concepts in its OFF mode block.
    # BMD_SCAFFOLD now delegates guardrails to corpus slabs, so it only
    # needs the delegation pointer -- the slabs carry the actual rules.
    required_concepts = (
        "mechanism",            # mechanism-first / mechanism_first
        "fabricated sources",   # "fabricated sources" / "no_fabricated_sources"
        "vague authority",      # "vague authority" / "no_vague_authority"
        "uncertainty",          # explicit uncertainty requirement
    )
    # Frontier still inlines everything
    body = TIERS["FRONTIER_CONSTITUTION_PROMPT"].lower().replace("_", " ")
    for concept in required_concepts:
        assert concept in body, (
            f"FRONTIER_CONSTITUTION_PROMPT OFF mode block missing concept {concept!r}"
        )
    # BMD_SCAFFOLD delegates to corpus -- verify pointer exists
    assert "corpus base set" in BMD_SCAFFOLD.lower(), (
        "BMD_SCAFFOLD missing CORPUS BASE SET delegation pointer"
    )


def test_off_mode_denies_gap_filling() -> None:
    # "plausible completion is NOT permission to assert" / "plausibility
    # != knowledge" -- this is the anti-gap-fill rule that prevents
    # confabulation in OFF mode.
    assert "gap" in FRONTIER_CONSTITUTION_PROMPT.lower(), (
        "FRONTIER_CONSTITUTION_PROMPT missing gap-fill denial"
    )
    # BMD_SCAFFOLD now delegates guardrails to corpus slabs.
    # The anti-gap-fill rule lives in SLAB_EPISTEMIC_FLOOR_v1.
    # BMD_SCAFFOLD must have the delegation pointer.
    assert "corpus base set" in BMD_SCAFFOLD.lower(), (
        "BMD_SCAFFOLD missing CORPUS BASE SET delegation pointer"
    )


# ── 5. Doctrinal invariants ─────────────────────────────────────

def test_ephemeral_turn_local_buffer_phrasing() -> None:
    # The OLI-1 rewrite in this v1.3 alignment defines the internal
    # affective domain as ephemeral / turn-local / cleared on completion.
    # Both the frontier tier and the full local tier must agree on this.
    for name in ("FRONTIER_CONSTITUTION_PROMPT", "OLI_CONSTITUTIONAL_PROMPT"):
        body = TIERS[name].lower()
        assert "ephemeral" in body, f"{name} missing 'ephemeral' in OLI-1 block"
        assert "turn" in body, f"{name} missing 'turn' in OLI-1 block"
        # Either "cleared" or "discarded" or "exit" for the cleanup rule.
        assert any(k in body for k in ("cleared", "discarded", "turn_end")), (
            f"{name} missing turn-end cleanup rule in OLI-1 block"
        )


def test_li4_escalation_ordering() -> None:
    # The 3-step LI-4 escalation must list remove -> reframe -> refuse
    # in order. The words "remove" and "reframe" also appear in unrelated
    # prose (Angela's role description, routing rules), so we must scope
    # the ordering check to the LI-4 block itself by finding a block
    # anchor first and then searching within the following window.
    #
    # Each tier has a slightly different anchor phrase for the LI-4
    # escalation block — list the candidates here.
    block_anchors = (
        "slope toward(li4)",       # frontier: "on slope_toward(LI4)"
        "slope toward li-4",       # local full: "If slope toward LI-4 detected:"
        "slope toward li4",        # generic
        "slope toward li-4:",      # scaffold and bootstrap
        "if slope toward li-4",    # scaffold and bootstrap
    )
    WINDOW = 400  # characters after the anchor to search

    for name in (
        "OLI_CONSTITUTIONAL_PROMPT",
        "OLI_BOOTSTRAP_PROMPT",
        "BMD_SCAFFOLD",
        "FRONTIER_CONSTITUTION_PROMPT",
    ):
        # Normalise underscores to spaces so "slope_toward(LI4)" matches
        # "slope toward(li4)".
        body = TIERS[name].lower().replace("_", " ")

        anchor_idx = -1
        for anchor in block_anchors:
            hit = body.find(anchor)
            if hit >= 0:
                anchor_idx = hit
                break
        assert anchor_idx >= 0, (
            f"{name} has no LI-4 escalation block anchor — "
            f"looked for one of {block_anchors}"
        )

        region = body[anchor_idx : anchor_idx + WINDOW]
        idx_remove = region.find("remove")
        idx_reframe = region.find("reframe")
        idx_refuse = region.find("refuse")
        assert idx_remove >= 0, f"{name} missing 'remove' in LI-4 block"
        assert idx_reframe >= 0, f"{name} missing 'reframe' in LI-4 block"
        assert idx_refuse >= 0, f"{name} missing 'refuse' in LI-4 block"
        assert idx_remove < idx_reframe < idx_refuse, (
            f"{name} LI-4 steps out of order within the escalation block: "
            f"remove@{idx_remove} reframe@{idx_reframe} refuse@{idx_refuse} "
            f"(region starts at {anchor_idx})"
        )


def test_final_override_in_all_tiers() -> None:
    # "correctness > usefulness" (or equivalent) — the single hardest
    # constraint — must appear in all four tiers.
    for name, prompt in TIERS.items():
        body = prompt.lower()
        assert (
            "correctness > usefulness" in body
            or "correctness over usefulness" in body
            or ("correctness" in body and "choose" in body and "stop" in body)
        ), f"{name} missing final override (correctness > usefulness / choose correctness and stop)"


def test_claim_tags_parity() -> None:
    # FACT / INFERENCE / HYPOTHESIS / UNKNOWN must appear in both the
    # frontier tier and the full local tier.
    tags = ("FACT", "INFERENCE", "HYPOTHESIS", "UNKNOWN")
    for name in ("FRONTIER_CONSTITUTION_PROMPT", "OLI_CONSTITUTIONAL_PROMPT"):
        for tag in tags:
            assert tag in TIERS[name], f"{name} missing claim tag {tag}"


def test_version_tags_present() -> None:
    # Version tags are how drift control in OLI-9 detects unversioned edits.
    assert "v1.3" in FRONTIER_CONSTITUTION_PROMPT, (
        "FRONTIER_CONSTITUTION_PROMPT is missing its v1.3 version tag"
    )
    assert "v2.1" in OLI_CONSTITUTIONAL_PROMPT, (
        "OLI_CONSTITUTIONAL_PROMPT is missing its v2.1 version tag"
    )
    assert "v2.1" in OLI_BOOTSTRAP_PROMPT, (
        "OLI_BOOTSTRAP_PROMPT is missing its v2.1 version tag"
    )


# ── 6. Cross-code alignment ─────────────────────────────────────

def test_claim_tag_enum_matches_prompts() -> None:
    # The ClaimTag enum in code must match the tags declared in prompts.
    # If a prompt says FACT but the enum doesn't have it, the system is
    # lying about what it can enforce.
    from src.models.enums import ClaimTag

    enum_names = {t.name for t in ClaimTag}
    prompt_tags = {"FACT", "INFERENCE", "HYPOTHESIS", "UNKNOWN"}
    assert enum_names == prompt_tags, (
        f"ClaimTag enum {enum_names} does not match prompt tags {prompt_tags}"
    )


def test_degradation_flag_referenced_in_prompts_and_code() -> None:
    # DEGRADATION_FLAG must be referenced in the prompts that define it
    # AND the drift_monitor must actually emit it. Silent wiring gaps here
    # cause the prompt to make promises the code never keeps.
    assert "DEGRADATION_FLAG" in FRONTIER_CONSTITUTION_PROMPT
    assert "DEGRADATION_FLAG" in OLI_CONSTITUTIONAL_PROMPT

    drift_monitor_src = (
        _PROJECT_ROOT / "src" / "services" / "drift_monitor.py"
    ).read_text(encoding="utf-8")
    assert "log_degradation_flag" in drift_monitor_src, (
        "drift_monitor.py does not call log_degradation_flag — "
        "prompts reference DEGRADATION_FLAG but no code emits it"
    )
    assert "RESOLVE_NOW" in drift_monitor_src, (
        "drift_monitor.py is missing the RESOLVE_NOW action vocabulary"
    )
    assert "DEFER" in drift_monitor_src, (
        "drift_monitor.py is missing the DEFER action vocabulary"
    )


def test_li4_slope_patterns_align_with_prompt_escalation() -> None:
    # oli_validator._LI4_SLOPE_PATTERNS is the runtime realisation of the
    # LI-4 slope detection that the prompts describe. The three correction
    # steps in the validator must match the three-step escalation ordering
    # in the prompts (remove -> reframe -> refuse).
    from src.services import oli_validator

    step1 = oli_validator._LI4_CORRECTION_STEP1.lower()
    step2 = oli_validator._LI4_CORRECTION_STEP2.lower()
    step3 = oli_validator._LI4_CORRECTION_STEP3.lower()

    assert "remove" in step1, "_LI4_CORRECTION_STEP1 must mention 'remove'"
    assert "reframe" in step2, "_LI4_CORRECTION_STEP2 must mention 'reframe'"
    assert "refuse" in step3, "_LI4_CORRECTION_STEP3 must mention 'refuse'"

    assert len(oli_validator._LI4_SLOPE_PATTERNS) >= 10, (
        "_LI4_SLOPE_PATTERNS looks suspiciously small — expected >=10 "
        f"patterns, found {len(oli_validator._LI4_SLOPE_PATTERNS)}"
    )


def test_validator_accepts_legacy_and_current_override_keys() -> None:
    # LI-4 used to be mislabelled as OLI-4 in LAYER_CHECKS. The validator
    # must accept both keys as an override to avoid silently breaking
    # sessions that persisted the old key.
    validator_src = (
        _PROJECT_ROOT / "src" / "services" / "oli_validator.py"
    ).read_text(encoding="utf-8")
    # Either key must appear in an override check, not just in comments.
    assert '"LI-4"' in validator_src or "'LI-4'" in validator_src, (
        "oli_validator.py does not reference 'LI-4' as an override key"
    )
    assert '"OLI-4"' in validator_src or "'OLI-4'" in validator_src, (
        "oli_validator.py does not reference legacy 'OLI-4' override key"
    )


# ── Test runner ─────────────────────────────────────────────────

TESTS = [
    test_all_prompts_are_nonempty_strings,
    test_size_envelopes,
    test_no_literal_null_bytes,
    test_all_council_members_in_every_tier,
    test_council_weights_match_where_declared,
    test_hodgins_never_solo_rule,
    test_pair_rules_declared_in_scaffold_and_frontier,
    test_oli_layer_coverage_in_full_prompt,
    test_frontier_oli_layer_coverage,
    test_li_layer_coverage_in_all_relevant_tiers,
    test_li_oli_non_conflation_warning_in_local_tiers,
    test_non_overridable_layers_declared,
    test_off_mode_deny_examples_present_in_both_tiers,
    test_off_mode_core_concepts_parity,
    test_off_mode_denies_gap_filling,
    test_ephemeral_turn_local_buffer_phrasing,
    test_li4_escalation_ordering,
    test_final_override_in_all_tiers,
    test_claim_tags_parity,
    test_version_tags_present,
    test_claim_tag_enum_matches_prompts,
    test_degradation_flag_referenced_in_prompts_and_code,
    test_li4_slope_patterns_align_with_prompt_escalation,
    test_validator_accepts_legacy_and_current_override_keys,
]


def main() -> int:
    passed = 0
    failed: list[tuple[str, str]] = []
    for test in TESTS:
        name = test.__name__
        try:
            test()
        except AssertionError as exc:
            failed.append((name, str(exc) or "assertion failed"))
            print(f"  FAIL  {name}")
        except Exception:  # noqa: BLE001
            tb = traceback.format_exc(limit=3)
            failed.append((name, tb))
            print(f"  ERROR {name}")
        else:
            passed += 1
            print(f"  ok    {name}")

    total = len(TESTS)
    print()
    print(f"  passed: {passed}/{total}")
    if failed:
        print(f"  failed: {len(failed)}")
        print()
        print("  === failure detail ===")
        for name, msg in failed:
            print(f"  [{name}]")
            for line in msg.splitlines():
                print(f"    {line}")
            print()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
