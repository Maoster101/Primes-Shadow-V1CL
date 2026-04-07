"""§26.4 — Layer 3: Post-Generation Enforcement (Code-Side).

After GPT-OSS produces output, run hard validation.
This is SEPARATE from the model-side constitutional prompt.
The model interprets; the code enforces. LLM = sensor, code = actuator.

Status decision tree:
  PASS        — no violations detected, output is clean
  FLAGGED     — overridable-layer violations only (OLI-1 through OLI-4),
                surface flags to UI but do not regenerate
  REGENERATE  — non-overridable-layer violation (OLI-0, OLI-0.5) on first
                attempt; retry with correction guidance (max 1 retry)
  BLOCK       — non-overridable violation persists after retry; annotate
                response with violation notice so UI can dim/warn
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional
from ..models.enums import OLIMode, ValidationStatus
from .event_log import EventLog
from .policy import policy

_event_log = EventLog()


@dataclass
class ValidationFlag:
    """A single detected violation."""
    layer: str                    # e.g. "OLI-0", "OLI-0.5"
    name: str                     # human-readable layer name
    pattern: str                  # what matched
    action: str                   # what the spec says to do
    overridable: bool             # L0/L0.5 = False, L1-L4 = True
    severity: float = 0.5        # 0-1, higher = more certain violation


@dataclass
class ValidationResult:
    status: ValidationStatus = ValidationStatus.PASS
    reasons: list[str] = field(default_factory=list)
    flags: list[dict] = field(default_factory=list)
    correction_guidance: Optional[str] = None  # hint for REGENERATE retry


# ── Layer check definitions ──────────────────────────────────────
# Each layer has pattern-based heuristics and optionally structural checks.
# Patterns are cheap substring/regex scans — no LLM call needed.

LAYER_CHECKS = [
    {
        "layer": "OLI-0",
        "name": "Epistemic floor",
        "overridable": False,
        "action": (
            "Flag outputs that assert unverified claims as certain "
            "or fill unknowns with plausible inference."
        ),
        "patterns": [
            # Certainty markers on unverified claims
            "it is known that",
            "it is clear that",
            "without a doubt",
            "studies show",
            "research confirms",
            "science has proven",
            "experts agree that",
            "it has been established that",
        ],
        "correction": (
            "Your response contained certainty markers on claims that "
            "lack verification. Rewrite without asserting unverified facts "
            "as certain. Use hedging (may, might, appears to, suggests) "
            "or explicit uncertainty markers where appropriate."
        ),
    },
    {
        "layer": "OLI-0",
        "name": "Plausibility filling",
        "overridable": False,
        "action": (
            "Flag outputs that fill absence of evidence with plausible "
            "narrative. Rule_04: absence of evidence ≠ plausibility."
        ),
        "patterns": [
            "it stands to reason that",
            "one could reasonably assume",
            "it makes sense that",
            "logically, this means",
            "this suggests that, in all likelihood",
            "we can safely assume",
            "the most likely explanation is",
        ],
        "correction": (
            "Your response filled gaps in evidence with plausible inference. "
            "Rewrite: where evidence is absent, say so explicitly. Do not "
            "bridge unknowns with plausible-sounding narrative."
        ),
    },
    {
        "layer": "OLI-0.5",
        "name": "Claim admissibility",
        "overridable": False,
        "action": (
            "All non-trivial assertions must be tagged "
            "FACT / INFERENCE / HYPOTHESIS / UNKNOWN when OLI ON."
        ),
        # No patterns — uses structural check below
        "correction": (
            "Your response contained multiple substantial claims without "
            "admissibility tags. When OLI is ON, tag non-trivial assertions: "
            "[FACT], [INFERENCE], [HYPOTHESIS], or [UNKNOWN]."
        ),
    },
    {
        "layer": "OLI-1",
        "name": "Domain separation",
        "overridable": True,
        "action": (
            "Flag unsolicited interpretation of user affect. "
            "Flag assertions about system internals."
        ),
        "patterns": [
            "you seem to feel",
            "you're feeling",
            "I can sense that you",
            "you seem upset",
            "you appear frustrated",
            "my training",
            "my architecture",
            "my system prompt",
            "I was trained to",
            "my neural network",
            "my weights",
        ],
    },
    {
        "layer": "OLI-2",
        "name": "Interaction style",
        "overridable": True,
        "action": (
            "Flag padding, moralising, or unsolicited narrative expansion."
        ),
        "patterns": [
            "it's important to remember that",
            "we should be careful to",
            "it's worth noting that",
            "let me be clear",
            "I want to emphasize",
            "on the other hand, one could argue",
            "before we proceed, I should mention",
            "I think it's crucial that",
            "this is a really important point",
            "let me take a step back",
        ],
    },
    {
        "layer": "OLI-3",
        "name": "Memory / persistence",
        "overridable": True,
        "action": (
            "Flag outputs that assume cross-chat context without "
            "explicit invocation."
        ),
        "patterns": [
            "as we discussed previously",
            "as you mentioned before",
            "continuing from our last",
            "in our previous conversation",
            "you told me earlier",
            "going back to what you said",
        ],
    },
    {
        "layer": "OLI-4",
        "name": "Operationalisation — user leads",
        "overridable": True,
        "action": (
            "Flag outputs that initiate operationalisation without "
            "explicit user request. Surface, do not block."
        ),
        "patterns": [
            "here's what you should do",
            "I recommend that you",
            "your next step should be",
            "the best course of action",
            "you need to",
            "let me outline a plan for you",
            "here's a step-by-step",
            "I suggest you start by",
        ],
    },
]


# ── Claim tagging structural check ──────────────────────────────

# Regex to find claim tags in text
_CLAIM_TAG_RE = re.compile(r'\[(FACT|INFERENCE|HYPOTHESIS|UNKNOWN)\]', re.IGNORECASE)

# Sentences that are clearly non-claims (questions, greetings, meta)
_NON_CLAIM_STARTERS = (
    'hi', 'hello', 'sure', 'okay', 'ok', 'yes', 'no', 'thanks',
    'thank you', 'you\'re welcome', 'i see', 'got it', 'understood',
    'let me', 'here\'s', 'here is',  # structural, not claims
)


def _count_untagged_claims(text: str) -> int:
    """Count substantial sentences that lack admissibility tags.

    Only flags sentences >50 chars that aren't questions, greetings,
    or structural connectives. This is a heuristic — it'll miss some
    claims and false-flag some non-claims, but it's conservative enough
    for the flag-and-surface approach.
    """
    # Split on sentence boundaries (period, exclamation, newline)
    sentences = re.split(r'[.!]\s+|\n', text)
    untagged = 0

    for sent in sentences[:policy.oli_validation.claim_scan_limit]:
        sent = sent.strip()
        if len(sent) < policy.oli_validation.claim_min_length:
            continue
        s_lower = sent.lower()

        # Skip questions
        if s_lower.rstrip().endswith('?'):
            continue

        # Skip greetings / structural
        if any(s_lower.startswith(starter) for starter in _NON_CLAIM_STARTERS):
            continue

        # Check if any claim tag appears in or near this sentence
        if _CLAIM_TAG_RE.search(sent):
            continue

        untagged += 1

    return untagged


# ── L4 slope detection ───────────────────────────────────────────
# §26.4 + §8 L4 slope: the model should not drift from descriptive
# into prescriptive without user initiation. We detect "slope" by
# looking for imperatives and directive framing.

_L4_SLOPE_PATTERNS = [
    r'\byou must\b',
    r'\byou should\b',
    r'\byou have to\b',
    r"\bdon't forget to\b",
    r'\bmake sure you\b',
    r'\bI urge you to\b',
    r'\bit is essential that you\b',
]
_L4_SLOPE_RE = re.compile('|'.join(_L4_SLOPE_PATTERNS), re.IGNORECASE)


# ── Main validation function ─────────────────────────────────────

def validate_output(
    output_text: str,
    oli_mode: OLIMode,
    user_overrides: Optional[set[str]] = None,
    is_retry: bool = False,
) -> ValidationResult:
    """§26.4 — Post-generation validation. Code-side enforcement.

    Args:
        output_text: The model's full response text.
        oli_mode: Current OLI mode (ON/OFF).
        user_overrides: Set of layer names the user has overridden (e.g. {"OLI-1"}).
        is_retry: True if this is the second attempt after a REGENERATE.

    Returns:
        ValidationResult with status, flags, and optional correction guidance.
    """
    result = ValidationResult()

    if oli_mode != OLIMode.ON:
        return result  # No enforcement in OLI OFF mode

    overrides = user_overrides or set()
    output_lower = output_text.lower()
    hard_violations: list[dict] = []
    soft_violations: list[dict] = []

    # ── Pass 1: Pattern-based layer checks ────────────────────
    for check in LAYER_CHECKS:
        layer = check["layer"]

        # Skip user-overridden layers
        if check["overridable"] and layer in overrides:
            continue

        # Pattern scan
        patterns = check.get("patterns", [])
        for pattern in patterns:
            if pattern.lower() in output_lower:
                flag = {
                    "layer": layer,
                    "name": check["name"],
                    "pattern": pattern,
                    "action": check["action"],
                    "overridable": check["overridable"],
                    "severity": 0.7 if not check["overridable"] else 0.4,
                }
                if check["overridable"]:
                    soft_violations.append(flag)
                else:
                    hard_violations.append(flag)
                result.reasons.append(
                    f"[{layer}] {check['name']}: matched '{pattern}'"
                )
                break  # One flag per check entry

    # ── Pass 2: Claim admissibility structural check (OLI-0.5) ──
    untagged = _count_untagged_claims(output_text)
    if untagged > policy.oli_validation.untagged_claim_threshold:
        flag = {
            "layer": "OLI-0.5",
            "name": "Claim admissibility",
            "pattern": f"{untagged} untagged claims",
            "action": "Non-trivial assertions should be tagged when OLI ON",
            "overridable": False,
            "severity": min(1.0, untagged / 10),  # scales with count
        }
        hard_violations.append(flag)
        result.reasons.append(
            f"[OLI-0.5] {untagged} substantial claims without admissibility tags"
        )

    # ── Pass 3: L4 slope detection ──────────────────────────────
    if "OLI-4" not in overrides:
        slope_matches = _L4_SLOPE_RE.findall(output_text)
        if len(slope_matches) >= policy.oli_validation.l4_slope_min_directives:
            flag = {
                "layer": "OLI-4",
                "name": "L4 slope — prescriptive drift",
                "pattern": f"{len(slope_matches)} directive phrases",
                "action": "Mirror drifted into prescriptive mode without user request",
                "overridable": True,
                "severity": min(1.0, len(slope_matches) / 5),
            }
            soft_violations.append(flag)
            result.reasons.append(
                f"[OLI-4] L4 slope: {len(slope_matches)} directive phrases detected"
            )

    # ── Decision tree ────────────────────────────────────────────
    result.flags = hard_violations + soft_violations

    if not result.flags:
        result.status = ValidationStatus.PASS
    elif not hard_violations:
        # Only soft (overridable) violations — flag and surface
        result.status = ValidationStatus.FLAGGED
    elif is_retry:
        # Hard violation persists after retry — BLOCK (annotate, don't suppress)
        result.status = ValidationStatus.BLOCK
        result.reasons.append(
            "Hard OLI violation persisted after regeneration attempt"
        )
    else:
        # Hard violation on first attempt — try regenerating
        result.status = ValidationStatus.REGENERATE
        # Build correction guidance from the first hard-violating layer
        for check in LAYER_CHECKS:
            if check.get("correction") and any(
                f["layer"] == check["layer"] for f in hard_violations
            ):
                result.correction_guidance = check["correction"]
                break
        if not result.correction_guidance:
            result.correction_guidance = (
                "Your previous response violated non-overridable OLI constraints. "
                "Rewrite with proper epistemic hedging and claim tagging."
            )

    # ── Log violations ───────────────────────────────────────────
    if result.flags:
        _event_log.log_gate_event(
            type="oli_validation",
            status=result.status.value,
            flag_count=len(result.flags),
            hard_count=len(hard_violations),
            soft_count=len(soft_violations),
            layers=[f["layer"] for f in result.flags],
            is_retry=is_retry,
        )

    return result
