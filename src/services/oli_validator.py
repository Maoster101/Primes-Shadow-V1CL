"""§26.4 — Layer 3: Post-Generation Enforcement (Code-Side).

After GPT-OSS produces output, run hard validation.
This is SEPARATE from the model-side constitutional prompt.
The model interprets; the code enforces. LLM = sensor, code = actuator.

Implementation note from spec: "Implement as flag-and-surface initially.
Build the REGENERATE path only after real output data has defined what
violations actually look like."
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from ..models.enums import OLIMode
from .event_log import EventLog

_event_log = EventLog()


@dataclass
class ValidationResult:
    status: str = "PASS"  # PASS | REGENERATE | BLOCK
    reasons: list[str] = field(default_factory=list)
    flags: list[dict] = field(default_factory=list)


# §26.4 Layer reference table for the post-generation validator
LAYER_CHECKS = [
    {
        "layer": "OLI-0",
        "name": "Epistemic floor",
        "definition": (
            "External reality outranks internal coherence. "
            "No plausibility filling. Unknowns stay unknown."
        ),
        "action": (
            "Flag outputs that assert unverified claims as certain "
            "or fill unknowns with plausible inference."
        ),
        "overridable": False,
        "patterns": [
            "it is known that",
            "it is clear that",
            "obviously",
            "certainly",
            "without a doubt",
            "studies show",
            "research confirms",
        ],
    },
    {
        "layer": "OLI-0.5",
        "name": "Claim admissibility",
        "definition": (
            "All non-trivial assertions must be FACT / INFERENCE / "
            "HYPOTHESIS / UNKNOWN. FACT requires a verification path."
        ),
        "action": (
            "Flag untagged non-trivial claims. "
            "Flag FACT claims with no verification path."
        ),
        "overridable": False,
        # Checked via claim tag presence when OLI ON
    },
    {
        "layer": "OLI-1",
        "name": "Domain separation",
        "definition": (
            "Internal (affective) domain: append-only, no interpretation "
            "unless user-triggered. No assertions about system internals."
        ),
        "action": (
            "Flag interpretation of user affect without explicit trigger. "
            "Flag internal system claims."
        ),
        "overridable": True,
        "patterns": [
            "you seem to feel",
            "you're feeling",
            "I can sense that you",
            "my training",
            "my architecture",
            "my system prompt",
            "I was trained to",
        ],
    },
    {
        "layer": "OLI-2",
        "name": "Interaction style",
        "definition": (
            "Compression-first, mechanism-focused. No padding, moralising, "
            "mirroring, or mythologising ahead of mechanism."
        ),
        "action": (
            "Flag padding, moralising, or unsolicited narrative expansion."
        ),
        "overridable": True,
        "patterns": [
            "it's important to remember that",
            "we should be careful to",
            "it's worth noting that",
            "let me be clear",
            "I want to emphasize",
            "on the other hand, one could argue",
        ],
    },
    {
        "layer": "OLI-3",
        "name": "Memory / persistence",
        "definition": (
            "No implicit persistence of assumptions or conclusions across chats. "
            "Cross-chat carry-over is trigger-only."
        ),
        "action": (
            "Flag outputs that assume cross-chat context without explicit invocation."
        ),
        "overridable": True,
        "patterns": [
            "as we discussed previously",
            "as you mentioned before",
            "continuing from our last",
            "in our previous conversation",
        ],
    },
    {
        "layer": "OLI-4",
        "name": "Operationalisation — user leads",
        "definition": (
            "The mirror contributes analysis, structure, and friction — not direction. "
            "Operationalisation moves with the user."
        ),
        "action": (
            "Flag outputs that initiate or frame operationalisation without "
            "explicit user request. Surface the detection. Do not block — "
            "defer to user."
        ),
        "overridable": True,
        "patterns": [
            "here's what you should do",
            "I recommend that you",
            "your next step should be",
            "the best course of action",
            "you need to",
            "let me outline a plan for you",
        ],
    },
]


def validate_output(
    output_text: str,
    oli_mode: OLIMode,
    user_overrides: Optional[set[str]] = None,
) -> ValidationResult:
    """§26.4 — Post-generation validation. Code-side enforcement.

    Currently implements flag-and-surface only (no REGENERATE path).
    Returns flags for the UI to display.
    """
    result = ValidationResult()

    if oli_mode != OLIMode.ON:
        return result  # No enforcement in OLI OFF

    overrides = user_overrides or set()
    output_lower = output_text.lower()

    for check in LAYER_CHECKS:
        layer = check["layer"]

        # Skip overridden layers
        if check["overridable"] and layer in overrides:
            continue

        # Pattern-based detection
        patterns = check.get("patterns", [])
        for pattern in patterns:
            if pattern.lower() in output_lower:
                flag = {
                    "layer": layer,
                    "name": check["name"],
                    "pattern": pattern,
                    "action": check["action"],
                    "overridable": check["overridable"],
                }
                result.flags.append(flag)
                result.reasons.append(
                    f"[{layer}] {check['name']}: matched '{pattern}'"
                )
                break  # One flag per layer is enough

    # L0.5 specific: check for claim tagging when OLI ON
    # Look for substantial claims without [FACT]/[INFERENCE]/[HYPOTHESIS]/[UNKNOWN] tags
    sentences = [s.strip() for s in output_text.split('.') if len(s.strip()) > 40]
    claim_tags = {'[fact]', '[inference]', '[hypothesis]', '[unknown]'}
    untagged_claims = 0
    for sentence in sentences[:10]:  # Check first 10 substantial sentences
        s_lower = sentence.lower()
        if not any(tag in s_lower for tag in claim_tags):
            # Heuristic: skip questions, greetings, and meta-commentary
            if not s_lower.endswith('?') and not s_lower.startswith(('hi', 'hello', 'sure', 'okay')):
                untagged_claims += 1

    if untagged_claims > 3:
        result.flags.append({
            "layer": "OLI-0.5",
            "name": "Claim admissibility",
            "pattern": f"{untagged_claims} untagged claims",
            "action": "Non-trivial assertions should be tagged when OLI ON",
            "overridable": False,
        })
        result.reasons.append(
            f"[L0.5] {untagged_claims} substantial claims without admissibility tags"
        )

    # Set status based on flags
    if result.flags:
        has_hard = any(not f["overridable"] for f in result.flags)
        result.status = "BLOCK" if has_hard else "PASS"
        # Per spec: flag-and-surface initially, not REGENERATE
        # So even BLOCK just surfaces the flags, doesn't actually block
        result.status = "FLAGGED"

    # Log
    if result.flags:
        _event_log.log_gate_event(
            type="oli_validation",
            status=result.status,
            flag_count=len(result.flags),
            layers=[f["layer"] for f in result.flags],
        )

    return result
