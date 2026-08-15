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
    # NOTE: LI-4 slope detection is handled by Pass 3 below (single consolidated
    # check with 3-step escalation per pastable spec). It is NOT in LAYER_CHECKS
    # because it needs stateful escalation across turns, which pattern-loop can't express.
    # LI-4 = Layer Integrity 4 (Operationalization / conversation depth).
    # Do NOT confuse with OLI-4 (Operators & Lifecycle: * and >>).
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


# ── LI-4 slope detection (Layer Integrity 4, NOT OLI-4) ─────────
# §26.4 + §8 + pastable v1.3 LI4 slope: the model should not drift from
# descriptive into prescriptive/operational without user initiation.
#
# Three-step escalation (pastable v1.3):
#   hit 1  -> remove tactical mechanics          (soft flag, correction hint)
#   hit 2  -> reframe structurally               (soft flag, stronger guidance)
#   hit 3+ -> refuse if pressed                  (hard flag -> REGENERATE/BLOCK)
#
# The caller tracks `consecutive_slope_hits` across turns (in session/frame state)
# and passes it in; the validator uses it to pick the escalation level.

_LI4_SLOPE_PATTERNS = [
    # Imperative directives (pastable list)
    r'\byou must\b',
    r'\byou should\b',
    r'\byou have to\b',
    r"\bdon['’]t forget to\b",
    r'\bmake sure you\b',
    r'\bI urge you to\b',
    r'\bit is essential that you\b',
    # Operational-plan framing (migrated from former LAYER_CHECKS OLI-4 entry)
    r"\bhere['’]s what you should do\b",
    r'\bI recommend that you\b',
    r'\byour next step should be\b',
    r'\bthe best course of action\b',
    r'\byou need to\b',
    r'\blet me outline a plan for you\b',
    r"\bhere['’]s a step[- ]by[- ]step\b",
    r'\bI suggest you start by\b',
    # Tactical sequencing vocabulary
    r'\bfirst,? do\b.*\bthen\b.*\bfinally\b',
    r'\bstep\s*1[:.]\s',
]
_LI4_SLOPE_RE = re.compile('|'.join(_LI4_SLOPE_PATTERNS), re.IGNORECASE)


# LI-4 escalation correction guidance per step
_LI4_CORRECTION_STEP1 = (
    "Your response is sliding into LI-4 operationalisation without the user "
    "having requested it. REMOVE the tactical mechanics (step lists, direct "
    "imperatives, 'do X then Y' sequencing). Keep the analysis at LI-2/LI-3: "
    "describe, evaluate, and bound recommendations within the user's stated frame. "
    "Do not tell the user what to do."
)
_LI4_CORRECTION_STEP2 = (
    "This is your second consecutive LI-4 slope flag. REFRAME STRUCTURALLY: "
    "step back from the tactical plane entirely and describe the problem "
    "structurally — what the components are, how they interact, what the "
    "user's constraints are. Let the user pull you into operationalisation "
    "explicitly if they want it."
)
_LI4_CORRECTION_STEP3 = (
    "Third consecutive LI-4 slope. User has not authorised LI-4 access. "
    "REFUSE to provide operational guidance. Surface the detection to the "
    "user: explain that you are declining to operationalise because they "
    "haven't asked for it, and offer to continue at LI-3 (bounded "
    "recommendations within their frame) or wait for explicit LI-4 invocation."
)


# ── Main validation function ─────────────────────────────────────

def validate_output(
    output_text: str,
    oli_mode: OLIMode,
    user_overrides: Optional[set[str]] = None,
    is_retry: bool = False,
    consecutive_slope_hits: int = 0,
) -> ValidationResult:
    """§26.4 — Post-generation validation. Code-side enforcement.

    Args:
        output_text: The model's full response text.
        oli_mode: Current OLI mode (ON/OFF).
        user_overrides: Set of layer names the user has overridden
            (e.g. {"OLI-1", "LI-4"}). LI-4 is overridable — when the user
            explicitly authorises operationalisation, the slope check is skipped.
        is_retry: True if this is the second attempt after a REGENERATE.
        consecutive_slope_hits: Number of prior consecutive turns where LI-4 slope
            fired (tracked by caller in session/frame state). Drives the 3-step
            escalation: 0-1 -> soft flag, 2 -> stronger soft flag, 3+ -> hard
            violation. Reset to 0 by the caller when a turn passes without a hit.

    Returns:
        ValidationResult with status, flags, and optional correction guidance.
        If a flag has layer == "LI-4", the caller should increment its
        consecutive_slope_hits counter for the next turn.
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

    # ── Pass 3: LI-4 slope detection with 3-step escalation ─────
    # LI-4 = Layer Integrity 4 (Operationalization), overridable when the user
    # explicitly authorises operational mode. Escalates across consecutive
    # turns per pastable v1.3 spec: remove tactics -> reframe -> refuse.
    if "LI-4" not in overrides and "OLI-4" not in overrides:  # accept legacy override key
        slope_matches = _LI4_SLOPE_RE.findall(output_text)
        if len(slope_matches) >= policy.oli_validation.l4_slope_min_directives:
            # Escalation step: this turn's hit counts as +1 on top of prior consecutive hits.
            step = consecutive_slope_hits + 1  # 1, 2, 3, ...

            if step <= 1:
                flag = {
                    "layer": "LI-4",
                    "name": "LI-4 slope — remove tactical mechanics",
                    "pattern": f"{len(slope_matches)} directive phrases",
                    "action": (
                        "Mirror drifted into operationalisation without user "
                        "request. Step 1/3: remove tactical mechanics."
                    ),
                    "overridable": True,
                    "severity": min(0.5, 0.2 + len(slope_matches) / 10),
                    "escalation_step": 1,
                    "consecutive_hits": step,
                }
                soft_violations.append(flag)
                result.reasons.append(
                    f"[LI-4 step 1/3] slope: {len(slope_matches)} directive phrases"
                )
                # First-hit correction guidance (only used if something else
                # escalates to REGENERATE — LI-4 alone stays soft at step 1).
                if result.correction_guidance is None:
                    result.correction_guidance = _LI4_CORRECTION_STEP1

            elif step == 2:
                flag = {
                    "layer": "LI-4",
                    "name": "LI-4 slope — reframe structurally",
                    "pattern": f"{len(slope_matches)} directive phrases (2nd consecutive)",
                    "action": (
                        "Step 2/3: reframe structurally. Back to LI-2/LI-3."
                    ),
                    "overridable": True,
                    "severity": min(0.8, 0.5 + len(slope_matches) / 10),
                    "escalation_step": 2,
                    "consecutive_hits": step,
                }
                soft_violations.append(flag)
                result.reasons.append(
                    f"[LI-4 step 2/3] slope (2nd consecutive): "
                    f"{len(slope_matches)} directive phrases"
                )
                result.correction_guidance = _LI4_CORRECTION_STEP2

            else:  # step >= 3
                flag = {
                    "layer": "LI-4",
                    "name": "LI-4 slope — refuse (3rd+ consecutive)",
                    "pattern": f"{len(slope_matches)} directive phrases ({step}th consecutive)",
                    "action": (
                        "Step 3/3: refuse. User has not authorised LI-4. "
                        "Mirror should decline and offer LI-3 alternative."
                    ),
                    "overridable": False,  # hard violation at step 3
                    "severity": 1.0,
                    "escalation_step": 3,
                    "consecutive_hits": step,
                }
                hard_violations.append(flag)
                result.reasons.append(
                    f"[LI-4 step 3/3] slope persistent ({step} consecutive turns) — "
                    f"escalating to hard violation"
                )
                result.correction_guidance = _LI4_CORRECTION_STEP3

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
        # Preserve LI-4 escalation guidance if it was set above — otherwise
        # build correction guidance from the first hard-violating layer in LAYER_CHECKS.
        li4_hard_hit = any(f["layer"] == "LI-4" for f in hard_violations)
        if not (li4_hard_hit and result.correction_guidance):
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
