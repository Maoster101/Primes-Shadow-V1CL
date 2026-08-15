"""Unit tests for Corpus Validation (§24.1 — 6 structural integrity checks).

Tests cover:
  1. Clean corpus passes all 6 checks
  2. Check 1: Duplicate IDs across object types
  3. Check 2: Anchor invokes targets exist
  4. Check 3: Bundle supports targets exist
  5. Check 4: Version suffix matches ID
  6. Check 5: depends_on targets exist
  7. Check 6: Supersedes target exists
  8. Empty corpus passes
  9. Multiple errors accumulate (fail-closed, reports all)
 10. Base set query respects slab type + lifecycle + OLI mode gating

Run standalone:
    python tests/test_corpus_validation.py

Also pytest-compatible:
    pytest tests/test_corpus_validation.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.schemas import (
    Anchor, Slab, SlabLinks, KeyBundle, BundlePayload, Edge, Gate, GateRule,
    AnchorMeta,
)
from src.models.enums import (
    EdgeType, OLIMode, SlabType, SlabLifecycleStatus, GateStage, GateOutcome,
)
from src.services.corpus import CorpusStore

# ── Helpers ─────────────────────────────────────────────────────

_FAILURES: list[str] = []


def _assert(condition: bool, msg: str) -> None:
    if not condition:
        _FAILURES.append(msg)
        print(f"  FAIL: {msg}")
    else:
        print(f"  ok:   {msg}")


def _make_anchor(id: str, invokes: list[str] | None = None, **kw) -> Anchor:
    return Anchor(
        id=id,
        canonical_phrase=id.replace("_", " "),
        invokes=invokes or [],
        **kw,
    )


def _make_slab(id: str, links_anchors: list[str] | None = None,
               links_bundles: list[str] | None = None, **kw) -> Slab:
    return Slab(
        id=id,
        canonical_text=f"Content for {id}",
        links=SlabLinks(
            anchors=links_anchors or [],
            bundles=links_bundles or [],
        ),
        **kw,
    )


def _make_bundle(id: str, supports: list[str] | None = None, **kw) -> KeyBundle:
    return KeyBundle(
        id=id,
        payload=BundlePayload(intent=["test intent"]),
        supports=supports or [],
        **kw,
    )


def _make_gate(id: str, **kw) -> Gate:
    return Gate(
        id=id,
        stage=GateStage.FUNCTION,
        **kw,
    )


def _fresh_store() -> CorpusStore:
    """Create an in-memory CorpusStore (no disk I/O)."""
    store = CorpusStore(collection_id="test")
    return store


# ── 1. Clean corpus passes ──────────────────────────────────────

def test_clean_corpus_passes():
    """A well-formed corpus with valid references passes all 6 checks."""
    store = _fresh_store()
    store.anchors = {
        "anchor_alpha_v1": _make_anchor("anchor_alpha_v1", invokes=["bundle_beta_v1"]),
    }
    store.bundles = {
        "bundle_beta_v1": _make_bundle("bundle_beta_v1", supports=["anchor_alpha_v1"]),
    }
    store.slabs = {
        "slab_gamma_v1": _make_slab("slab_gamma_v1",
                                     links_anchors=["anchor_alpha_v1"],
                                     links_bundles=["bundle_beta_v1"]),
    }
    errors = store.validate()
    _assert(errors == [], f"Clean corpus -> 0 errors (got {errors})")


# ── 2. Check 1: Duplicate IDs across types ─────────────────────

def test_duplicate_id_across_types():
    """Same ID in both anchors and slabs triggers Check 1."""
    store = _fresh_store()
    store.anchors = {"shared_id_v1": _make_anchor("shared_id_v1")}
    store.slabs = {"shared_id_v1": _make_slab("shared_id_v1")}
    errors = store.validate()
    dup_errors = [e for e in errors if "Duplicate ID across object types" in e]
    _assert(len(dup_errors) > 0, "Duplicate ID across anchor/slab detected")


def test_duplicate_id_anchor_bundle():
    """Same ID in anchors and bundles triggers Check 1."""
    store = _fresh_store()
    store.anchors = {"dup_v1": _make_anchor("dup_v1")}
    store.bundles = {"dup_v1": _make_bundle("dup_v1")}
    errors = store.validate()
    dup_errors = [e for e in errors if "Duplicate ID" in e]
    _assert(len(dup_errors) > 0, "Duplicate ID across anchor/bundle detected")


# ── 3. Check 2: Invokes targets exist ──────────────────────────

def test_anchor_invokes_missing_target():
    """Anchor invoking a nonexistent target triggers Check 2."""
    store = _fresh_store()
    store.anchors = {
        "anchor_a_v1": _make_anchor("anchor_a_v1", invokes=["nonexistent_target_v1"]),
    }
    errors = store.validate()
    invoke_errors = [e for e in errors if "invokes missing target" in e]
    _assert(len(invoke_errors) > 0, "Missing invokes target detected")
    _assert("nonexistent_target_v1" in invoke_errors[0], "Error names the missing target")


def test_anchor_invokes_valid_slab():
    """Anchor invoking an existing slab passes Check 2."""
    store = _fresh_store()
    store.anchors = {
        "anchor_a_v1": _make_anchor("anchor_a_v1", invokes=["slab_b_v1"]),
    }
    store.slabs = {
        "slab_b_v1": _make_slab("slab_b_v1"),
    }
    errors = store.validate()
    invoke_errors = [e for e in errors if "invokes missing target" in e]
    _assert(len(invoke_errors) == 0, "Anchor invoking existing slab -> no error")


# ── 4. Check 3: Supports targets exist ─────────────────────────

def test_bundle_supports_missing_target():
    """Bundle supporting a nonexistent target triggers Check 3."""
    store = _fresh_store()
    store.bundles = {
        "bundle_a_v1": _make_bundle("bundle_a_v1", supports=["ghost_v1"]),
    }
    errors = store.validate()
    support_errors = [e for e in errors if "supports missing target" in e]
    _assert(len(support_errors) > 0, "Missing supports target detected")


# ── 5. Check 4: Version suffix match ───────────────────────────

def test_version_suffix_mismatch():
    """ID that doesn't end with _v{version} triggers Check 4."""
    store = _fresh_store()
    # Anchor has version=v1 in meta but ID ends with _v2
    bad_anchor = _make_anchor("anchor_wrong_v2")
    bad_anchor.meta = AnchorMeta(version="v1")  # version is v1 but ID says v2
    store.anchors = {"anchor_wrong_v2": bad_anchor}
    errors = store.validate()
    suffix_errors = [e for e in errors if "does not end with expected suffix" in e]
    _assert(len(suffix_errors) > 0, "Version suffix mismatch detected")


def test_version_suffix_correct():
    """ID properly ending with _v{version} passes Check 4."""
    store = _fresh_store()
    anchor = _make_anchor("anchor_good_v1")
    anchor.meta = AnchorMeta(version="v1")
    store.anchors = {"anchor_good_v1": anchor}
    errors = store.validate()
    suffix_errors = [e for e in errors if "does not end with expected suffix" in e]
    _assert(len(suffix_errors) == 0, "Correct version suffix -> no error")


# ── 6. Check 5: depends_on targets exist ───────────────────────

def test_depends_on_missing_target():
    """depends_on referencing a nonexistent node triggers Check 5."""
    store = _fresh_store()
    slab = _make_slab("slab_dep_v1", depends_on=["nonexistent_dep_v1"])
    store.slabs = {"slab_dep_v1": slab}
    errors = store.validate()
    dep_errors = [e for e in errors if "depends_on missing target" in e]
    _assert(len(dep_errors) > 0, "Missing depends_on target detected")


def test_depends_on_valid_target():
    """depends_on referencing an existing node passes Check 5."""
    store = _fresh_store()
    anchor = _make_anchor("anchor_base_v1")
    slab = _make_slab("slab_dep_v1", depends_on=["anchor_base_v1"])
    store.anchors = {"anchor_base_v1": anchor}
    store.slabs = {"slab_dep_v1": slab}
    errors = store.validate()
    dep_errors = [e for e in errors if "depends_on missing target" in e]
    _assert(len(dep_errors) == 0, "Valid depends_on -> no error")


# ── 7. Check 6: Supersedes exists ──────────────────────────────

def test_supersedes_missing_target():
    """meta.supersedes referencing a nonexistent node triggers Check 6."""
    store = _fresh_store()
    anchor = _make_anchor("anchor_new_v1")
    anchor.meta = AnchorMeta(version="v1", supersedes="anchor_old_v1")
    store.anchors = {"anchor_new_v1": anchor}
    errors = store.validate()
    sup_errors = [e for e in errors if "supersedes missing target" in e]
    _assert(len(sup_errors) > 0, "Missing supersedes target detected")


def test_supersedes_valid_target():
    """meta.supersedes referencing an existing node passes Check 6."""
    store = _fresh_store()
    old = _make_anchor("anchor_old_v1")
    new = _make_anchor("anchor_new_v1")
    new.meta = AnchorMeta(version="v1", supersedes="anchor_old_v1")
    store.anchors = {"anchor_old_v1": old, "anchor_new_v1": new}
    errors = store.validate()
    sup_errors = [e for e in errors if "supersedes missing target" in e]
    _assert(len(sup_errors) == 0, "Valid supersedes -> no error")


def test_supersedes_none_is_fine():
    """meta.supersedes=None is not an error."""
    store = _fresh_store()
    anchor = _make_anchor("anchor_a_v1")
    anchor.meta = AnchorMeta(version="v1", supersedes=None)
    store.anchors = {"anchor_a_v1": anchor}
    errors = store.validate()
    sup_errors = [e for e in errors if "supersedes" in e]
    _assert(len(sup_errors) == 0, "supersedes=None -> no error")


# ── 8. Empty corpus ────────────────────────────────────────────

def test_empty_corpus_passes():
    """An empty corpus with no objects passes all checks."""
    store = _fresh_store()
    errors = store.validate()
    _assert(errors == [], "Empty corpus -> 0 errors")


# ── 9. Multiple errors accumulate ──────────────────────────────

def test_multiple_errors_all_reported():
    """Validator reports ALL errors in a single pass (fail-closed)."""
    store = _fresh_store()
    # Check 2: bad invokes
    store.anchors = {
        "anchor_a_v1": _make_anchor("anchor_a_v1", invokes=["ghost1_v1"]),
    }
    # Check 3: bad supports
    store.bundles = {
        "bundle_b_v1": _make_bundle("bundle_b_v1", supports=["ghost2_v1"]),
    }
    # Check 5: bad depends_on
    slab = _make_slab("slab_c_v1", depends_on=["ghost3_v1"])
    store.slabs = {"slab_c_v1": slab}

    errors = store.validate()
    _assert(len(errors) >= 3, f"Multiple checks fire -> 3+ errors (got {len(errors)})")
    _assert(any("invokes missing" in e for e in errors), "Invokes error present")
    _assert(any("supports missing" in e for e in errors), "Supports error present")
    _assert(any("depends_on missing" in e for e in errors), "depends_on error present")


# ── 10. Base set query ──────────────────────────────────────────

def test_base_set_filters_by_type_and_lifecycle():
    """base_set_slabs filtering rules.

    Default (include_reference=True): CONSTITUTIONAL + CANONICAL + REFERENCE
    are all in the base set. INVARIANT and DEPRECATED are always excluded.

    With include_reference=False: legacy filter — only CONSTITUTIONAL +
    CANONICAL pass.
    """
    store = _fresh_store()
    store.slabs = {
        "slab_const_v1": _make_slab("slab_const_v1", type=SlabType.CONSTITUTIONAL),
        "slab_canon_v1": _make_slab("slab_canon_v1", type=SlabType.CANONICAL),
        "slab_invar_v1": _make_slab("slab_invar_v1", type=SlabType.INVARIANT),
        "slab_ref_v1": _make_slab("slab_ref_v1", type=SlabType.REFERENCE),
        "slab_dead_v1": _make_slab("slab_dead_v1", type=SlabType.CANONICAL,
                                    lifecycle_status=SlabLifecycleStatus.DEPRECATED),
    }
    # Default behaviour: REFERENCE included
    result = store.base_set_slabs(OLIMode.OFF)
    ids = {s.id for s in result}
    _assert("slab_const_v1" in ids, "CONSTITUTIONAL in base set")
    _assert("slab_canon_v1" in ids, "CANONICAL in base set")
    _assert("slab_invar_v1" not in ids, "INVARIANT excluded from base set")
    _assert("slab_ref_v1" in ids, "REFERENCE in base set (default)")
    _assert("slab_dead_v1" not in ids, "DEPRECATED excluded from base set")

    # Legacy / opt-out behaviour: REFERENCE excluded
    legacy = store.base_set_slabs(OLIMode.OFF, include_reference=False)
    legacy_ids = {s.id for s in legacy}
    _assert("slab_const_v1" in legacy_ids, "CONSTITUTIONAL in legacy base set")
    _assert("slab_canon_v1" in legacy_ids, "CANONICAL in legacy base set")
    _assert("slab_ref_v1" not in legacy_ids,
            "REFERENCE excluded when include_reference=False")


def test_base_set_oli_mode_gating():
    """requires_oli_mode gates slabs into/out of the base set."""
    store = _fresh_store()
    store.slabs = {
        "slab_always_v1": _make_slab("slab_always_v1",
                                      type=SlabType.CONSTITUTIONAL,
                                      requires_oli_mode=None),
        "slab_oli_only_v1": _make_slab("slab_oli_only_v1",
                                        type=SlabType.CONSTITUTIONAL,
                                        requires_oli_mode=OLIMode.ON),
    }
    off_result = store.base_set_slabs(OLIMode.OFF)
    on_result = store.base_set_slabs(OLIMode.ON)

    off_ids = {s.id for s in off_result}
    on_ids = {s.id for s in on_result}

    _assert("slab_always_v1" in off_ids, "None-gated slab in OFF base set")
    _assert("slab_always_v1" in on_ids, "None-gated slab in ON base set")
    _assert("slab_oli_only_v1" not in off_ids, "OLI.ON-gated slab excluded from OFF base set")
    _assert("slab_oli_only_v1" in on_ids, "OLI.ON-gated slab included in ON base set")


def test_invariant_slabs_query():
    """invariant_slabs() returns only ACTIVE INVARIANT slabs."""
    store = _fresh_store()
    store.slabs = {
        "slab_inv_v1": _make_slab("slab_inv_v1", type=SlabType.INVARIANT),
        "slab_inv_dead_v1": _make_slab("slab_inv_dead_v1",
                                        type=SlabType.INVARIANT,
                                        lifecycle_status=SlabLifecycleStatus.DORMANT),
        "slab_canon_v1": _make_slab("slab_canon_v1", type=SlabType.CANONICAL),
    }
    result = store.invariant_slabs()
    ids = {s.id for s in result}
    _assert("slab_inv_v1" in ids, "ACTIVE INVARIANT returned")
    _assert("slab_inv_dead_v1" not in ids, "DORMANT INVARIANT excluded")
    _assert("slab_canon_v1" not in ids, "CANONICAL excluded from invariant query")


# ── 11. Reverse deps ───────────────────────────────────────────

def test_reverse_deps():
    """get_reverse_deps finds anchors that invoke, bundles that support a node."""
    store = _fresh_store()
    store.anchors = {
        "anchor_a_v1": _make_anchor("anchor_a_v1", invokes=["slab_target_v1"]),
        "anchor_b_v1": _make_anchor("anchor_b_v1", invokes=[]),
    }
    store.bundles = {
        "bundle_c_v1": _make_bundle("bundle_c_v1", supports=["slab_target_v1"]),
    }
    store.slabs = {
        "slab_target_v1": _make_slab("slab_target_v1"),
    }
    deps = store.get_reverse_deps("slab_target_v1")
    _assert("anchor_a_v1" in deps, "Anchor that invokes target is a reverse dep")
    _assert("bundle_c_v1" in deps, "Bundle that supports target is a reverse dep")
    _assert("anchor_b_v1" not in deps, "Anchor that doesn't reference target excluded")


# ── Runner ──────────────────────────────────────────────────────

ALL_TESTS = [
    test_clean_corpus_passes,
    test_duplicate_id_across_types,
    test_duplicate_id_anchor_bundle,
    test_anchor_invokes_missing_target,
    test_anchor_invokes_valid_slab,
    test_bundle_supports_missing_target,
    test_version_suffix_mismatch,
    test_version_suffix_correct,
    test_depends_on_missing_target,
    test_depends_on_valid_target,
    test_supersedes_missing_target,
    test_supersedes_valid_target,
    test_supersedes_none_is_fine,
    test_empty_corpus_passes,
    test_multiple_errors_all_reported,
    test_base_set_filters_by_type_and_lifecycle,
    test_base_set_oli_mode_gating,
    test_invariant_slabs_query,
    test_reverse_deps,
]


def main() -> int:
    global _FAILURES
    _FAILURES = []
    print(f"\n{'='*60}")
    print("Corpus Validation Unit Tests")
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
