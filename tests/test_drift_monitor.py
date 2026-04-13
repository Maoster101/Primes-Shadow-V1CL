"""Unit tests for the Drift Monitor (§17.6-17.10 — User Drift Monitoring).

Tests cover:
  1. Empty window returns safe defaults
  2. Single estimate -> correct composite calculation
  3. Linear recency weighting — newer estimates weigh more
  4. Window cap — oldest estimates evicted when window_size exceeded
  5. Domain-based threshold switching (EXTERNAL vs INTERNAL)
  6. Severity bucketing: LOW / MEDIUM / HIGH
  7. Dampening response: NONE / MEDIUM / HIGH
  8. Session duration signal contribution
  9. DriftMonitor per-session isolation
 10. Composite clamped to [0, 1]
 11. Reason summarisation (top signals)

Run standalone:
    python tests/test_drift_monitor.py

Also pytest-compatible:
    pytest tests/test_drift_monitor.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.schemas import DriftEstimate
from src.models.enums import DriftSeverity, DomainMode, DampeningLevel
from src.services.drift_monitor import (
    DriftWindow,
    DriftMonitor,
    _summarise_drift_reason,
)
from src.services.policy import policy

# ── Helpers ─────────────────────────────────────────────────────

_FAILURES: list[str] = []


def _assert(condition: bool, msg: str) -> None:
    if not condition:
        _FAILURES.append(msg)
        print(f"  FAIL: {msg}")
    else:
        print(f"  ok:   {msg}")


def _est(affect: float = 0.0, volatility: float = 0.0,
         rigor: float = 0.0, domain: DomainMode = DomainMode.EXTERNAL) -> DriftEstimate:
    """Shorthand for creating a DriftEstimate."""
    return DriftEstimate(
        affect_density=affect,
        claim_volatility=volatility,
        rigor_drop=rigor,
        domain_mode=domain,
    )


# ── 1. Empty window defaults ───────────────────────────────────

def test_empty_window_returns_defaults():
    """An empty DriftWindow returns composite=0, severity=LOW, dampening=NONE."""
    w = DriftWindow()
    result = w.compute(current_turn=1)
    _assert(result["composite"] == 0.0, "Empty -> composite 0.0")
    _assert(result["severity"] == DriftSeverity.LOW, "Empty -> LOW severity")
    _assert(result["dampening"] == DampeningLevel.NONE, "Empty -> NONE dampening")
    _assert(result["window_size"] == 0, "Empty -> window_size 0")


# ── 2. Single estimate ─────────────────────────────────────────

def test_single_estimate_composite():
    """Single estimate: recency weight = 1/1 = 1.0, so composite is direct weighted sum."""
    w = DriftWindow()
    w.push(_est(affect=0.5, volatility=0.4, rigor=0.3), turn=1)
    result = w.compute(current_turn=1)

    # With n=1: weight = 1/1 = 1.0, so signals pass through at full strength.
    # Composite = affect*0.30 + volatility*0.25 + rigor*0.25 + duration*0.20
    # Duration signal depends on session length vs adaptive baseline.
    cw = policy.drift.composite_weights
    expected_signals = 0.5 * cw.affect_density + 0.4 * cw.claim_volatility + 0.3 * cw.rigor_drop
    _assert(result["composite"] >= expected_signals * 0.9,
            f"Single estimate composite ~= {expected_signals:.3f} (got {result['composite']})")
    _assert(result["window_size"] == 1, "window_size == 1")


# ── 3. Recency weighting — newer estimates weigh more ──────────

def test_recency_weighting():
    """Later estimates have more influence than earlier ones."""
    w = DriftWindow()

    # Push a calm estimate, then a hot estimate
    w.push(_est(affect=0.0, volatility=0.0, rigor=0.0), turn=1)
    w.push(_est(affect=1.0, volatility=1.0, rigor=1.0), turn=2)
    result_hot_recent = w.compute(current_turn=2)

    # Now test the reverse: hot first, then calm
    w2 = DriftWindow()
    w2.push(_est(affect=1.0, volatility=1.0, rigor=1.0), turn=1)
    w2.push(_est(affect=0.0, volatility=0.0, rigor=0.0), turn=2)
    result_calm_recent = w2.compute(current_turn=2)

    _assert(
        result_hot_recent["composite"] > result_calm_recent["composite"],
        f"Hot-recent ({result_hot_recent['composite']:.4f}) > "
        f"calm-recent ({result_calm_recent['composite']:.4f})"
    )


# ── 4. Window cap ──────────────────────────────────────────────

def test_window_cap_evicts_oldest():
    """Window evicts oldest estimates when exceeding window_size."""
    w = DriftWindow()
    ws = policy.drift.window_size  # default 50

    # Push ws+10 estimates
    for i in range(ws + 10):
        w.push(_est(affect=0.1), turn=i + 1)

    result = w.compute(current_turn=ws + 10)
    _assert(result["window_size"] == ws, f"Window capped at {ws} (got {result['window_size']})")


# ── 5. Domain-based thresholds ──────────────────────────────────

def test_internal_domain_has_higher_thresholds():
    """INTERNAL domain has higher thresholds (reads hotter = harder to trigger)."""
    dt = policy.drift.thresholds
    _assert(dt.internal.low > dt.external.low,
            f"Internal low ({dt.internal.low}) > external low ({dt.external.low})")
    _assert(dt.internal.medium > dt.external.medium,
            f"Internal medium ({dt.internal.medium}) > external medium ({dt.external.medium})")


def test_same_signals_external_vs_internal_severity():
    """Same moderate signals: EXTERNAL may be MEDIUM while INTERNAL stays LOW."""
    # Use signals that land between external.low and internal.low thresholds
    dt = policy.drift.thresholds
    # Target composite between external.low (0.39) and internal.low (0.54)
    target = (dt.external.low + dt.internal.low) / 2

    # We need: affect*0.30 + vol*0.25 + rigor*0.25 + duration*0.20 ~= target
    # Set all three model signals equal, ignore duration for simplicity
    # signal * (0.30 + 0.25 + 0.25) = target -> signal = target / 0.80
    signal_val = min(1.0, target / 0.80)

    w_ext = DriftWindow()
    w_ext.push(_est(affect=signal_val, volatility=signal_val,
                     rigor=signal_val, domain=DomainMode.EXTERNAL), turn=1)
    r_ext = w_ext.compute(current_turn=1)

    w_int = DriftWindow()
    w_int.push(_est(affect=signal_val, volatility=signal_val,
                     rigor=signal_val, domain=DomainMode.INTERNAL), turn=1)
    r_int = w_int.compute(current_turn=1)

    # Composites should be similar (same signals), but severity may differ
    # due to different thresholds
    _assert(
        r_ext["severity"].value >= r_int["severity"].value
        or r_ext["composite"] <= dt.external.low,
        f"External severity ({r_ext['severity']}) >= Internal severity ({r_int['severity']}) "
        f"for same signals, or both below threshold"
    )


# ── 6. Severity bucketing ──────────────────────────────────────

def test_low_severity_on_zero_signals():
    """Zero signals -> LOW severity."""
    w = DriftWindow()
    w.push(_est(affect=0.0, volatility=0.0, rigor=0.0), turn=1)
    result = w.compute(current_turn=1)
    _assert(result["severity"] == DriftSeverity.LOW, "Zero signals -> LOW")


def test_high_severity_on_max_signals():
    """Maximum signals -> HIGH severity."""
    w = DriftWindow()
    # Push several max estimates to ensure composite exceeds all thresholds
    for i in range(5):
        w.push(_est(affect=1.0, volatility=1.0, rigor=1.0), turn=i + 1)
    result = w.compute(current_turn=5)
    _assert(result["severity"] == DriftSeverity.HIGH,
            f"Max signals -> HIGH (got {result['severity']}, composite={result['composite']})")


# ── 7. Dampening response ──────────────────────────────────────

def test_low_severity_no_dampening():
    """LOW severity -> NONE dampening."""
    w = DriftWindow()
    w.push(_est(affect=0.0), turn=1)
    result = w.compute(current_turn=1)
    _assert(result["dampening"] == DampeningLevel.NONE, "LOW -> NONE dampening")


def test_high_severity_high_dampening():
    """HIGH severity -> HIGH dampening."""
    w = DriftWindow()
    for i in range(5):
        w.push(_est(affect=1.0, volatility=1.0, rigor=1.0), turn=i + 1)
    result = w.compute(current_turn=5)
    _assert(result["dampening"] == DampeningLevel.HIGH, "HIGH severity -> HIGH dampening")


# ── 8. Session duration signal ──────────────────────────────────

def test_duration_signal_increases_with_length():
    """Longer sessions produce higher duration signal."""
    w_short = DriftWindow()
    w_short.push(_est(affect=0.5, volatility=0.5, rigor=0.5), turn=1)
    r_short = w_short.compute(current_turn=2)  # 2 turns into session

    w_long = DriftWindow()
    w_long.push(_est(affect=0.5, volatility=0.5, rigor=0.5), turn=1)
    r_long = w_long.compute(current_turn=100)  # 100 turns into session

    _assert(
        r_long["signals"]["duration"] >= r_short["signals"]["duration"],
        f"Longer session duration signal ({r_long['signals']['duration']}) "
        f">= shorter ({r_short['signals']['duration']})"
    )


# ── 9. DriftMonitor per-session isolation ───────────────────────

def test_drift_monitor_session_isolation():
    """Different session_ids maintain separate windows."""
    monitor = DriftMonitor()
    monitor.record_and_compute("session_a", _est(affect=1.0, volatility=1.0, rigor=1.0), turn=1)
    monitor.record_and_compute("session_b", _est(affect=0.0, volatility=0.0, rigor=0.0), turn=1)

    window_a = monitor.get_window("session_a")
    window_b = monitor.get_window("session_b")

    result_a = window_a.compute(current_turn=1)
    result_b = window_b.compute(current_turn=1)

    _assert(
        result_a["composite"] > result_b["composite"],
        f"Session A ({result_a['composite']}) > Session B ({result_b['composite']})"
    )


def test_drift_monitor_creates_window_on_demand():
    """get_window creates a new window for unknown session IDs."""
    monitor = DriftMonitor()
    w = monitor.get_window("brand_new_session")
    result = w.compute(current_turn=1)
    _assert(result["composite"] == 0.0, "New session window starts empty")


# ── 10. Composite clamped ──────────────────────────────────────

def test_composite_clamped_0_1():
    """Composite score is always in [0, 1] regardless of input."""
    w = DriftWindow()
    # Push extreme values
    for i in range(10):
        w.push(_est(affect=1.0, volatility=1.0, rigor=1.0), turn=i + 1)
    result = w.compute(current_turn=1000)  # very long session
    _assert(0.0 <= result["composite"] <= 1.0,
            f"Composite clamped: {result['composite']}")


# ── 11. Reason summarisation ───────────────────────────────────

def test_summarise_drift_reason():
    """Reason summary picks top signals by magnitude."""
    result = {
        "composite": 0.72,
        "signals": {
            "affect": 0.85,
            "volatility": 0.60,
            "rigor_drop": 0.20,
            "duration": 0.10,
        }
    }
    reason = _summarise_drift_reason(result)
    _assert("composite=0.72" in reason, "Reason includes composite")
    _assert("affect=0.85" in reason, "Reason includes top signal (affect)")
    _assert("volatility=0.60" in reason, "Reason includes second signal (volatility)")


def test_summarise_empty_signals():
    """Empty signals produce a fallback reason string."""
    reason = _summarise_drift_reason({"composite": 0.5, "signals": {}})
    _assert(len(reason) > 0, "Non-empty reason even with no signals")


# ── 12. Composite weight sanity ─────────────────────────────────

def test_composite_weights_sum_to_one():
    """The four composite weights should sum to 1.0 (or very close)."""
    cw = policy.drift.composite_weights
    total = cw.affect_density + cw.claim_volatility + cw.rigor_drop + cw.session_duration
    _assert(abs(total - 1.0) < 0.01,
            f"Composite weights sum to {total:.3f} (expected ~1.0)")


# ── 13. Multi-turn accumulation ─────────────────────────────────

def test_gradual_buildup_increases_composite():
    """Progressive hot turns increase composite over time."""
    w = DriftWindow()
    composites = []
    for i in range(1, 6):
        w.push(_est(affect=0.7, volatility=0.6, rigor=0.5), turn=i)
        result = w.compute(current_turn=i)
        composites.append(result["composite"])

    # Later turns should have equal or higher composite (more data, more weight on hot signals)
    _assert(composites[-1] >= composites[0],
            f"Composite grows: first={composites[0]:.4f}, last={composites[-1]:.4f}")


def test_cooldown_after_hot_period():
    """Calm turns after hot turns should reduce composite (recency favours calm)."""
    w = DriftWindow()
    # 3 hot turns
    for i in range(1, 4):
        w.push(_est(affect=0.9, volatility=0.8, rigor=0.7), turn=i)
    hot_result = w.compute(current_turn=3)

    # 3 calm turns
    for i in range(4, 7):
        w.push(_est(affect=0.0, volatility=0.0, rigor=0.0), turn=i)
    cool_result = w.compute(current_turn=6)

    _assert(
        cool_result["composite"] < hot_result["composite"],
        f"Cooldown: hot={hot_result['composite']:.4f} > cool={cool_result['composite']:.4f}"
    )


# ── Runner ──────────────────────────────────────────────────────

ALL_TESTS = [
    test_empty_window_returns_defaults,
    test_single_estimate_composite,
    test_recency_weighting,
    test_window_cap_evicts_oldest,
    test_internal_domain_has_higher_thresholds,
    test_same_signals_external_vs_internal_severity,
    test_low_severity_on_zero_signals,
    test_high_severity_on_max_signals,
    test_low_severity_no_dampening,
    test_high_severity_high_dampening,
    test_duration_signal_increases_with_length,
    test_drift_monitor_session_isolation,
    test_drift_monitor_creates_window_on_demand,
    test_composite_clamped_0_1,
    test_summarise_drift_reason,
    test_summarise_empty_signals,
    test_composite_weights_sum_to_one,
    test_gradual_buildup_increases_composite,
    test_cooldown_after_hot_period,
]


def main() -> int:
    global _FAILURES
    _FAILURES = []
    print(f"\n{'='*60}")
    print("Drift Monitor Unit Tests")
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
