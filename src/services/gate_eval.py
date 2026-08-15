"""Declarative gate evaluator — §8.4 simpleeval-based rule engine.

Evaluates Gate rules against a context dict built from the current
MessageClassification, Anchor, and session state. Runs parallel to
the imperative gate_check() in anchor_matcher.py; does NOT replace it.
The pipeline port (Phase 4.4) will wire this into the hot path.

Usage:
    from src.services.gate_eval import GateEvaluator
    evaluator = GateEvaluator()
    result = evaluator.evaluate(gate, context)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from simpleeval import simple_eval, InvalidExpression

from ..models.schemas import Gate, GateRule
from ..models.enums import GateOutcome

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    """Outcome of evaluating a single Gate against a context."""
    gate_id: str
    outcome: GateOutcome
    matched_rule: Optional[str] = None   # id of the rule that fired, or None if default
    trace: list[str] = field(default_factory=list)  # human-readable eval log


@dataclass
class PipelineResult:
    """Outcome of running the full 3-stage gate pipeline."""
    final_outcome: GateOutcome
    stage_results: list[GateResult] = field(default_factory=list)

    @property
    def allowed(self) -> bool:
        return self.final_outcome == GateOutcome.ALLOW


def _flatten_context(context: dict[str, Any]) -> dict[str, Any]:
    """Build a flat namespace for simpleeval from nested context objects.

    Supports both raw dicts and Pydantic models. Dotted attribute access
    in expressions (e.g. ``classification.confidence``) works because
    simpleeval resolves ``a.b`` as ``names['a'].b`` — so we just need
    the top-level keys present and the objects to support attribute access.

    For safety, we also inject Python builtins that expressions might need
    (True, False, None) and prevent access to dunder attributes.
    """
    flat: dict[str, Any] = {
        "True": True,
        "False": False,
        "None": None,
    }
    for key, value in context.items():
        flat[key] = value
    return flat


class GateEvaluator:
    """Evaluate Gate rules using simpleeval expressions.

    Thread-safe (no mutable state). Instantiate once, call evaluate() or
    run_pipeline() per request.
    """

    def evaluate(self, gate: Gate, context: dict[str, Any]) -> GateResult:
        """Evaluate a single Gate's rules against the context.

        Rules are sorted by priority (ascending); first match wins.
        If no rule matches, the gate's default_outcome applies.
        """
        names = _flatten_context(context)
        sorted_rules = sorted(gate.rules, key=lambda r: r.priority)
        trace: list[str] = []

        for rule in sorted_rules:
            try:
                result = simple_eval(rule.condition, names=names)
            except InvalidExpression as exc:
                trace.append(f"[{rule.id}] INVALID expression: {exc}")
                logger.warning("Gate %s rule %s has invalid expression: %s",
                               gate.id, rule.id, exc)
                continue
            except Exception as exc:
                # Catch-all for attribute errors, name errors, etc.
                trace.append(f"[{rule.id}] ERROR: {type(exc).__name__}: {exc}")
                logger.warning("Gate %s rule %s eval error: %s",
                               gate.id, rule.id, exc)
                continue

            if result:
                trace.append(f"[{rule.id}] MATCH → {rule.outcome.value}")
                return GateResult(
                    gate_id=gate.id,
                    outcome=rule.outcome,
                    matched_rule=rule.id,
                    trace=trace,
                )
            else:
                trace.append(f"[{rule.id}] no match")

        # No rule matched — apply default
        trace.append(f"[default] → {gate.default_outcome.value}")
        return GateResult(
            gate_id=gate.id,
            outcome=gate.default_outcome,
            matched_rule=None,
            trace=trace,
        )

    def run_pipeline(
        self,
        gates: list[Gate],
        context: dict[str, Any],
    ) -> PipelineResult:
        """Run an ordered sequence of gates (the 3-stage pipeline).

        Gates are evaluated in list order. The pipeline short-circuits on
        DENY or ESCALATE — only ALLOW and DEFER proceed to the next stage.

        Returns a PipelineResult with the final outcome and per-stage trace.
        """
        stage_results: list[GateResult] = []

        for gate in gates:
            result = self.evaluate(gate, context)
            stage_results.append(result)

            if result.outcome == GateOutcome.DENY:
                return PipelineResult(
                    final_outcome=GateOutcome.DENY,
                    stage_results=stage_results,
                )
            elif result.outcome == GateOutcome.ESCALATE:
                return PipelineResult(
                    final_outcome=GateOutcome.ESCALATE,
                    stage_results=stage_results,
                )
            # ALLOW or DEFER → continue to next stage

        # All stages passed (ALLOW or DEFER) — final outcome is ALLOW
        return PipelineResult(
            final_outcome=GateOutcome.ALLOW,
            stage_results=stage_results,
        )
