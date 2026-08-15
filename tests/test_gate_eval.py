"""Unit tests for the Declarative Gate Evaluator (ss8.4 -- simpleeval rule engine).

Tests cover:
  1. Single gate: rule matching by priority order
  2. Single gate: default outcome when no rules match
  3. Single gate: invalid expression handling (graceful skip)
  4. Single gate: attribute error handling (graceful skip)
  5-8. FUNCTION gate parity (not_required, explicit, allowed_function, default_deny)
  9. Pipeline: all-ALLOW passes through all stages
  10. Pipeline: DENY short-circuits at first stage
  11. Pipeline: ESCALATE short-circuits
  12. Pipeline: DEFER continues to next stage
  13. simpleeval `in` operator support check (prerequisite for safe_functions rule)

Run standalone:
    python tests/test_gate_eval.py

Also pytest-compatible:
    pytest tests/test_gate_eval.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.schemas import Gate, GateRule
from src.models.enums import GateOutcome, GateStage
from src.services.gate_eval import GateEvaluator, GateResult, PipelineResult

# -- Helpers ---------------------------------------------------------------

_FAILURES: list[str] = []


def _assert(condition: bool, msg: str) -> None:
    if not condition:
        _FAILURES.append(msg)
        print(f"  FAIL: {msg}")
    else:
        print(f"  ok:   {msg}")


def _gate(id: str, rules: list[GateRule], default: GateOutcome = GateOutcome.DENY,
          stage: GateStage = GateStage.FUNCTION) -> Gate:
    """Shorthand for creating a Gate."""
    return Gate(id=id, stage=stage, rules=rules, default_outcome=default)


def _rule(id: str, condition: str, outcome: GateOutcome, priority: int = 100) -> GateRule:
    """Shorthand for creating a GateRule."""
    return GateRule(id=id, condition=condition, outcome=outcome, priority=priority)


# Lightweight stand-ins for Anchor/MessageClassification.
# The real Pydantic models have many fields -- we only need attribute access
# for the expressions that the gate rules evaluate.

@dataclass
class _MatchPolicy:
    gate_required: bool = True
    allowed_functions: list[str] | None = None

    def __post_init__(self):
        if self.allowed_functions is None:
            self.allowed_functions = ["context_compression"]


@dataclass
class _FakeAnchor:
    match_policy: _MatchPolicy | None = None

    def __post_init__(self):
        if self.match_policy is None:
            self.match_policy = _MatchPolicy()


@dataclass
class _FakeClassification:
    function: str = "neutral"
    explicit: bool = False
    confidence: float = 0.5
    mention_type: str | None = None


_evaluator = GateEvaluator()


# -- 1. Rule matching by priority order ------------------------------------

def test_rule_match_by_priority():
    """Rules evaluate in priority order; first match wins."""
    gate = _gate("g1", [
        _rule("r_low", "x == 1", GateOutcome.DENY, priority=30),
        _rule("r_mid", "x == 2", GateOutcome.ALLOW, priority=20),
        _rule("r_high", "x == 3", GateOutcome.ESCALATE, priority=10),
    ])
    # x == 2 matches at priority 20, x == 3 doesn't match
    result = _evaluator.evaluate(gate, {"x": 2})
    _assert(result.outcome == GateOutcome.ALLOW, "Priority 20 rule fires")
    _assert(result.matched_rule == "r_mid", f"Matched rule is r_mid (got {result.matched_rule})")

    # x == 3 matches at priority 10 (highest prio = lowest number)
    result2 = _evaluator.evaluate(gate, {"x": 3})
    _assert(result2.outcome == GateOutcome.ESCALATE, "Priority 10 fires first")
    _assert(result2.matched_rule == "r_high", f"Matched rule is r_high (got {result2.matched_rule})")


# -- 2. Default outcome when no rules match --------------------------------

def test_default_outcome_when_no_rules_match():
    """When no rule condition is true, gate returns its default_outcome."""
    gate = _gate("g2", [
        _rule("r1", "x == 99", GateOutcome.ALLOW, priority=10),
    ], default=GateOutcome.ESCALATE)
    result = _evaluator.evaluate(gate, {"x": 1})
    _assert(result.outcome == GateOutcome.ESCALATE, "No match -> default ESCALATE")
    _assert(result.matched_rule is None, "matched_rule is None on default")
    _assert("[default]" in result.trace[-1], "Trace shows default fired")


# -- 3. Invalid expression handling ----------------------------------------

def test_invalid_expression_skipped():
    """Malformed condition is skipped; next valid rule evaluated."""
    gate = _gate("g3", [
        _rule("bad", "foo ??? bar", GateOutcome.DENY, priority=10),
        _rule("good", "x == 1", GateOutcome.ALLOW, priority=20),
    ])
    result = _evaluator.evaluate(gate, {"x": 1})
    _assert(result.outcome == GateOutcome.ALLOW, "Bad rule skipped, good rule fires")
    _assert(result.matched_rule == "good", "Matched the valid rule")
    bad_trace = [t for t in result.trace if "INVALID" in t or "ERROR" in t]
    _assert(len(bad_trace) > 0, "Trace records the invalid expression")


# -- 4. Attribute error handling -------------------------------------------

def test_attribute_error_skipped():
    """Rule referencing nonexistent attribute is skipped gracefully."""
    gate = _gate("g4", [
        _rule("bad_attr", "obj.nonexistent_field == True", GateOutcome.DENY, priority=10),
        _rule("fallback", "True", GateOutcome.ALLOW, priority=20),
    ])
    result = _evaluator.evaluate(gate, {"obj": _FakeClassification()})
    _assert(result.outcome == GateOutcome.ALLOW, "Attribute error skipped, fallback fires")
    # simpleeval may raise InvalidExpression or catch as ERROR -- either trace entry is fine
    err_trace = [t for t in result.trace if "ERROR" in t or "INVALID" in t or "no match" in t]
    _assert(len(err_trace) > 0, "Trace records the bad rule (error or skip)")


# -- 5-8. FUNCTION gate parity tests --------------------------------------

def _build_function_gate() -> Gate:
    """Build a gate matching GATE_FUNCTION_v1 from gates.yaml."""
    return _gate("GATE_FUNCTION_v1", [
        _rule("not_required", "anchor.match_policy.gate_required == False",
              GateOutcome.ALLOW, priority=10),
        _rule("explicit", "classification.explicit == True",
              GateOutcome.ALLOW, priority=20),
        _rule("allowed_fn", "classification.function in anchor.match_policy.allowed_functions",
              GateOutcome.ALLOW, priority=30),
    ], default=GateOutcome.DENY)


def test_function_gate_not_required():
    """gate_required=False -> ALLOW regardless of function."""
    gate = _build_function_gate()
    ctx = {
        "anchor": _FakeAnchor(match_policy=_MatchPolicy(gate_required=False)),
        "classification": _FakeClassification(function="affect_release"),
    }
    result = _evaluator.evaluate(gate, ctx)
    _assert(result.outcome == GateOutcome.ALLOW, "gate_required=False -> ALLOW")
    _assert(result.matched_rule == "not_required", "Matched not_required rule")


def test_function_gate_explicit_user():
    """explicit=True -> ALLOW even when function is blocked."""
    gate = _build_function_gate()
    ctx = {
        "anchor": _FakeAnchor(),
        "classification": _FakeClassification(function="affect_release", explicit=True),
    }
    result = _evaluator.evaluate(gate, ctx)
    _assert(result.outcome == GateOutcome.ALLOW, "explicit=True -> ALLOW")


def test_function_gate_allowed_function():
    """Function in allowed_functions -> ALLOW."""
    gate = _build_function_gate()
    ctx = {
        "anchor": _FakeAnchor(match_policy=_MatchPolicy(
            gate_required=True, allowed_functions=["context_compression"]
        )),
        "classification": _FakeClassification(function="context_compression", explicit=False),
    }
    result = _evaluator.evaluate(gate, ctx)
    _assert(result.outcome == GateOutcome.ALLOW, "Function in allowed_functions -> ALLOW")


def test_function_gate_default_deny():
    """Function not allowed, not explicit, gate required -> DENY."""
    gate = _build_function_gate()
    ctx = {
        "anchor": _FakeAnchor(match_policy=_MatchPolicy(
            gate_required=True, allowed_functions=["context_compression"]
        )),
        "classification": _FakeClassification(function="affect_release", explicit=False),
    }
    result = _evaluator.evaluate(gate, ctx)
    _assert(result.outcome == GateOutcome.DENY, "Blocked function -> DENY")


# -- 9. Pipeline: all-ALLOW ------------------------------------------------

def test_pipeline_all_allow():
    """Three gates all returning ALLOW -> pipeline ALLOW."""
    gates = [
        _gate("g1", [_rule("r1", "True", GateOutcome.ALLOW, 10)]),
        _gate("g2", [_rule("r2", "True", GateOutcome.ALLOW, 10)],
              stage=GateStage.EXPLICITNESS),
        _gate("g3", [_rule("r3", "True", GateOutcome.ALLOW, 10)],
              stage=GateStage.CONFIDENCE),
    ]
    result = _evaluator.run_pipeline(gates, {"x": 1})
    _assert(result.final_outcome == GateOutcome.ALLOW, "All ALLOW -> pipeline ALLOW")
    _assert(result.allowed is True, "allowed property is True")
    _assert(len(result.stage_results) == 3, f"All 3 stages ran (got {len(result.stage_results)})")


# -- 10. Pipeline: DENY short-circuits ------------------------------------

def test_pipeline_deny_short_circuits():
    """First gate DENY -> pipeline stops, doesn't evaluate remaining gates."""
    gates = [
        _gate("g1", [_rule("r1", "True", GateOutcome.DENY, 10)]),
        _gate("g2", [_rule("r2", "True", GateOutcome.ALLOW, 10)],
              stage=GateStage.EXPLICITNESS),
    ]
    result = _evaluator.run_pipeline(gates, {"x": 1})
    _assert(result.final_outcome == GateOutcome.DENY, "DENY at stage 1")
    _assert(len(result.stage_results) == 1, f"Short-circuited (got {len(result.stage_results)} stages)")


# -- 11. Pipeline: ESCALATE short-circuits ---------------------------------

def test_pipeline_escalate_short_circuits():
    """Second gate ESCALATE -> pipeline stops with ESCALATE."""
    gates = [
        _gate("g1", [_rule("r1", "True", GateOutcome.ALLOW, 10)]),
        _gate("g2", [], default=GateOutcome.ESCALATE,
              stage=GateStage.EXPLICITNESS),
        _gate("g3", [_rule("r3", "True", GateOutcome.ALLOW, 10)],
              stage=GateStage.CONFIDENCE),
    ]
    result = _evaluator.run_pipeline(gates, {"x": 1})
    _assert(result.final_outcome == GateOutcome.ESCALATE, "ESCALATE at stage 2")
    _assert(len(result.stage_results) == 2, f"Stopped at stage 2 (got {len(result.stage_results)})")


# -- 12. Pipeline: DEFER continues -----------------------------------------

def test_pipeline_defer_continues():
    """DEFER at one stage continues to next stage (doesn't short-circuit)."""
    gates = [
        _gate("g1", [_rule("r1", "True", GateOutcome.DEFER, 10)]),
        _gate("g2", [_rule("r2", "True", GateOutcome.ALLOW, 10)],
              stage=GateStage.EXPLICITNESS),
    ]
    result = _evaluator.run_pipeline(gates, {"x": 1})
    _assert(result.final_outcome == GateOutcome.ALLOW, "DEFER -> continues -> ALLOW")
    _assert(len(result.stage_results) == 2, f"Both stages ran (got {len(result.stage_results)})")


# -- 13. simpleeval `in` operator support ----------------------------------

def test_simpleeval_in_operator():
    """Test simpleeval list literal support and validate or-chain fallback.

    simpleeval's default configuration does NOT allow list literals [...]
    (raises 'Sorry, List is not available'). The safe_functions rule in
    gates.yaml must therefore use an or-chain instead of `in [...]`.
    This test validates that the or-chain works correctly.
    """
    # Verify list literals are NOT supported (documents the constraint)
    gate_list = _gate("g_list", [
        _rule("in_list", "x in ['a', 'b', 'c']", GateOutcome.ALLOW, 10),
    ], default=GateOutcome.DENY)
    r_list = _evaluator.evaluate(gate_list, {"x": "b"})
    _assert(r_list.outcome == GateOutcome.DENY,
            "List literal not supported -> falls to default DENY (expected)")

    # Validate or-chain alternative (this is what gates.yaml will use)
    gate_or = _gate("g_or", [
        _rule("or_chain",
              "x == 'a' or x == 'b' or x == 'c'",
              GateOutcome.ALLOW, 10),
    ], default=GateOutcome.DENY)

    r1 = _evaluator.evaluate(gate_or, {"x": "b"})
    _assert(r1.outcome == GateOutcome.ALLOW, "or-chain: 'b' matches -> ALLOW")

    r2 = _evaluator.evaluate(gate_or, {"x": "z"})
    _assert(r2.outcome == GateOutcome.DENY, "or-chain: 'z' no match -> DENY")

    # Also test with attribute access (closer to real gate usage)
    gate_attr = _gate("g_attr", [
        _rule("attr_or", "obj.fn == 'neutral' or obj.fn == 'rigor_work'",
              GateOutcome.ALLOW, 10),
    ], default=GateOutcome.DENY)

    @dataclass
    class _Obj:
        fn: str = "neutral"

    r3 = _evaluator.evaluate(gate_attr, {"obj": _Obj()})
    _assert(r3.outcome == GateOutcome.ALLOW, "attr or-chain: 'neutral' -> ALLOW")

    r4 = _evaluator.evaluate(gate_attr, {"obj": _Obj(fn="affect_release")})
    _assert(r4.outcome == GateOutcome.DENY, "attr or-chain: 'affect_release' -> DENY")


# -- Runner ----------------------------------------------------------------

ALL_TESTS = [
    test_rule_match_by_priority,
    test_default_outcome_when_no_rules_match,
    test_invalid_expression_skipped,
    test_attribute_error_skipped,
    test_function_gate_not_required,
    test_function_gate_explicit_user,
    test_function_gate_allowed_function,
    test_function_gate_default_deny,
    test_pipeline_all_allow,
    test_pipeline_deny_short_circuits,
    test_pipeline_escalate_short_circuits,
    test_pipeline_defer_continues,
    test_simpleeval_in_operator,
]


def main() -> int:
    global _FAILURES
    _FAILURES = []
    print(f"\n{'='*60}")
    print("Gate Evaluator Unit Tests")
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
            print(f"  FAIL: exception -- {e}")
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
