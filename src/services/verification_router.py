"""External verification router — §26.6 FACT claim verification.

Routes FACT-tagged claims to verification backends. Currently supports:
  - manual:    flag for human review (default, always available)
  - ollama:    cross-check via a second LLM prompt (local, no API key)
  - external:  stub for future Claude/Perplexity API integration

The router does NOT decide truth — it produces a VerificationOutcome
(CONFIRMED / CONTRADICTED / UNRESOLVABLE) that the draft manager uses
to gate corpus promotion. LLM = sensor, code = actuator.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional

from ..models.enums import VerificationOutcome
from .event_log import EventLog
from . import ollama

logger = logging.getLogger(__name__)
_event_log = EventLog()


@dataclass
class VerificationResult:
    """Outcome of verifying a single claim."""
    claim: str
    outcome: VerificationOutcome
    method: str                     # "manual" | "ollama" | "external"
    confidence: float = 0.0         # 0.0-1.0, how confident the verifier is
    evidence: str = ""              # supporting text / source
    notes: str = ""                 # any caveats


@dataclass
class BatchVerificationResult:
    """Outcome of verifying all claims in a draft."""
    results: list[VerificationResult] = field(default_factory=list)
    all_confirmed: bool = False
    has_contradictions: bool = False
    unresolvable_count: int = 0

    def summary(self) -> dict:
        return {
            "total": len(self.results),
            "confirmed": sum(1 for r in self.results if r.outcome == VerificationOutcome.CONFIRMED),
            "contradicted": sum(1 for r in self.results if r.outcome == VerificationOutcome.CONTRADICTED),
            "unresolvable": sum(1 for r in self.results if r.outcome == VerificationOutcome.UNRESOLVABLE),
            "all_confirmed": self.all_confirmed,
            "has_contradictions": self.has_contradictions,
        }


# ── Verification prompt for local LLM cross-check ─────────────

VERIFY_PROMPT = """You are a fact-checking assistant. Evaluate the following claim for factual accuracy.

Claim: "{claim}"

Context (from the conversation where this claim was made):
{context}

Evaluate this claim and respond with ONLY valid JSON:
{{
  "outcome": "CONFIRMED" | "CONTRADICTED" | "UNRESOLVABLE",
  "confidence": 0.0-1.0,
  "evidence": "Brief explanation of why you reached this conclusion",
  "notes": "Any caveats or limitations of this assessment"
}}

Rules:
- CONFIRMED: The claim is factually accurate based on your training data
- CONTRADICTED: The claim is factually incorrect or misleading
- UNRESOLVABLE: Cannot determine accuracy (too subjective, too niche, or requires real-time data)
- Be conservative: if unsure, use UNRESOLVABLE
- Do NOT confirm claims about real-time events, prices, or rapidly changing data
"""


class VerificationRouter:
    """Routes FACT claims to appropriate verification backends."""

    def __init__(self):
        self._method = "ollama"  # default to local LLM cross-check

    def set_method(self, method: str) -> None:
        """Set verification method: 'manual', 'ollama', or 'external'."""
        if method not in ("manual", "ollama", "external"):
            raise ValueError(f"Unknown method: {method}. Use 'manual', 'ollama', or 'external'.")
        self._method = method

    async def verify_claim(
        self,
        claim: str,
        context: str = "",
        method: Optional[str] = None,
    ) -> VerificationResult:
        """Verify a single claim."""
        method = method or self._method

        if method == "manual":
            return self._manual_flag(claim)
        elif method == "ollama":
            return await self._ollama_verify(claim, context)
        elif method == "external":
            return self._external_stub(claim, context)
        else:
            return self._manual_flag(claim)

    async def verify_batch(
        self,
        claims: list[str],
        context: str = "",
        method: Optional[str] = None,
    ) -> BatchVerificationResult:
        """Verify a batch of claims (e.g., all FACT claims in a draft)."""
        results = []
        for claim in claims:
            result = await self.verify_claim(claim, context, method)
            results.append(result)

            # Log each verification
            _event_log.log_verification(
                claim=claim[:200],
                outcome=result.outcome.value,
                method=result.method,
                confidence=result.confidence,
            )

        batch = BatchVerificationResult(results=results)
        batch.all_confirmed = all(
            r.outcome == VerificationOutcome.CONFIRMED for r in results
        )
        batch.has_contradictions = any(
            r.outcome == VerificationOutcome.CONTRADICTED for r in results
        )
        batch.unresolvable_count = sum(
            1 for r in results if r.outcome == VerificationOutcome.UNRESOLVABLE
        )

        return batch

    # ── Backend implementations ────────────────────────────────

    def _manual_flag(self, claim: str) -> VerificationResult:
        """Flag for human review — no automated check."""
        return VerificationResult(
            claim=claim,
            outcome=VerificationOutcome.UNRESOLVABLE,
            method="manual",
            confidence=0.0,
            evidence="",
            notes="Flagged for manual human review. No automated verification performed.",
        )

    async def _ollama_verify(self, claim: str, context: str) -> VerificationResult:
        """Cross-check via local LLM (second opinion from the same model)."""
        prompt = VERIFY_PROMPT.format(
            claim=claim,
            context=context[:2000] if context else "(no context provided)",
        )

        try:
            result = await ollama.structured_extract(prompt)

            outcome_str = result.get("outcome", "UNRESOLVABLE").upper()
            try:
                outcome = VerificationOutcome(outcome_str)
            except ValueError:
                outcome = VerificationOutcome.UNRESOLVABLE

            return VerificationResult(
                claim=claim,
                outcome=outcome,
                method="ollama",
                confidence=float(result.get("confidence", 0.5)),
                evidence=str(result.get("evidence", "")),
                notes=str(result.get("notes", "")),
            )

        except Exception as e:
            logger.warning("Ollama verification failed: %s", e)
            return VerificationResult(
                claim=claim,
                outcome=VerificationOutcome.UNRESOLVABLE,
                method="ollama",
                confidence=0.0,
                notes=f"Verification failed: {e}",
            )

    def _external_stub(self, claim: str, context: str) -> VerificationResult:
        """Stub for external API verification (Claude, Perplexity, etc.).

        TODO: Implement when API key is available. The interface is:
          1. Send claim + context to external API
          2. Parse structured response
          3. Return VerificationResult

        For now, falls back to manual flag.
        """
        return VerificationResult(
            claim=claim,
            outcome=VerificationOutcome.UNRESOLVABLE,
            method="external",
            confidence=0.0,
            notes="External verification not yet configured. Flagged for manual review.",
        )
