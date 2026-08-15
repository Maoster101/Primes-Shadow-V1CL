"""Core enumerations from the design spec."""
from enum import Enum


class MessageFunction(str, Enum):
    CONTEXT_COMPRESSION = "context_compression"
    OBJECT_OF_WORK = "object_of_work"
    AFFECT_RELEASE = "affect_release"
    RIGOR_WORK = "rigor_work"
    META_SCHEMA = "meta_schema"
    CORPUS_REVIEW = "corpus_review"
    NEUTRAL = "neutral"


class ChatStatus(str, Enum):
    ACTIVE = "ACTIVE"
    PARKED = "PARKED"
    CLOSED = "CLOSED"
    ARCHIVED = "ARCHIVED"


class DraftStatus(str, Enum):
    DRAFT_UNAUTHORIZED = "DRAFT_UNAUTHORIZED"
    PROVISIONAL = "PROVISIONAL"
    COMMITTED = "COMMITTED"
    STALE_REVIEW_REQUIRED = "STALE_REVIEW_REQUIRED"
    REJECTED = "REJECTED"


class NodeType(str, Enum):
    SLAB = "slab"
    ANCHOR = "anchor"
    KEY_BUNDLE = "key_bundle"
    CONCEPT = "concept"
    # Persisted navigational overlay (Tier-0). References content nodes
    # via members + children. Does not store substantive content itself;
    # its summary is a curated distillation produced at construction.
    PILLAR_DEFINITION = "pillar_definition"


class NodeStatus(str, Enum):
    CORPUS = "corpus"
    TENTATIVE = "tentative"


class EdgeType(str, Enum):
    INVOKES = "INVOKES"
    SUPPORTS = "SUPPORTS"
    CONFLICTS = "CONFLICTS"   # Direct opposition: A rejects / contradicts B
    TENSIONS = "TENSIONS"     # Productive tension: A and B counterbalance, both valid (e.g. mercy ↔ justice)
    LINKS = "LINKS"
    PARENT_OF = "PARENT_OF"
    SEQUENCE = "SEQUENCE"  # Narrative ordering — A precedes B in the story spine


class DialecticSubtype(str, Enum):
    """Refines CONFLICTS / TENSIONS edges with the conversational
    mechanism that produced them.

    Edges from document mining are typically COUNTERBALANCE (the
    document holds both views as valid simultaneously) or
    OPPOSITION (the document positions one as wrong). Edges from
    live conversation mining can also surface temporal patterns:
    DRIFT (position evolved across turns), RETRACTION (position
    explicitly withdrawn), LIVE_CORRECTION (mid-conversation
    rephrasing without retraction).

    Optional field — pre-live-mining edges have no subtype and the
    field stays None.
    """
    # Document / static patterns
    OPPOSITION = "OPPOSITION"          # Speaker positions one as wrong
    COUNTERBALANCE = "COUNTERBALANCE"  # Both valid, must balance (mercy ↔ justice)
    # Temporal / conversational patterns
    DRIFT = "DRIFT"                    # Position evolved across turns
    RETRACTION = "RETRACTION"          # Position explicitly withdrawn
    LIVE_CORRECTION = "LIVE_CORRECTION"  # Mid-conversation rephrasing


class EpistemicStatus(str, Enum):
    """Trajectory state of a tentative draft across a chat session.

    Used by live mining to track how a proposal evolves over the
    conversation. The end-of-conversation canonicalization pass
    classifies each draft into one of these states and decides
    promote / demote / drop accordingly.

    Lifecycle:
      NEWLY_RAISED → CONFIRMED (referenced back, validated) → promote
                  → DRIFTED (position evolved into another draft) → record DRIFT edge
                  → RETRACTED (explicitly withdrawn) → record RETRACTION edge, demote
                  → UNTOUCHED (never referenced again) → drop or preserve isolated
                  → CANONICALIZED (final state — committed to corpus or archived)
    """
    NEWLY_RAISED = "NEWLY_RAISED"
    CONFIRMED = "CONFIRMED"
    DRIFTED = "DRIFTED"
    RETRACTED = "RETRACTED"
    UNTOUCHED = "UNTOUCHED"
    CANONICALIZED = "CANONICALIZED"


class OLIMode(str, Enum):
    ON = "ON"
    OFF = "OFF"


class L4Posture(str, Enum):
    USER_LEADS = "user_leads"


class DriftSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DomainMode(str, Enum):
    INTERNAL = "internal"
    EXTERNAL = "external"


class ClaimTag(str, Enum):
    FACT = "FACT"
    INFERENCE = "INFERENCE"
    HYPOTHESIS = "HYPOTHESIS"
    UNKNOWN = "UNKNOWN"


class VerificationOutcome(str, Enum):
    CONFIRMED = "CONFIRMED"
    CONTRADICTED = "CONTRADICTED"
    UNRESOLVABLE = "UNRESOLVABLE"


class RevisionAction(str, Enum):
    AMEND_MINOR = "AMEND_MINOR"
    SUPERSEDE_SEMANTIC = "SUPERSEDE_SEMANTIC"
    DEPRECATE = "DEPRECATE"
    SPLIT = "SPLIT"
    MERGE = "MERGE"


class DampeningLevel(str, Enum):
    NONE = "none"
    MEDIUM = "medium"
    HIGH = "high"


class MatchTier(int, Enum):
    EXACT_OR_PARTIAL = 1
    AMBIGUOUS_FUZZY = 2
    WEAK_SEMANTIC = 3


class SlabType(str, Enum):
    CONSTITUTIONAL = "CONSTITUTIONAL"
    INVARIANT = "INVARIANT"
    REFERENCE = "REFERENCE"
    CANONICAL = "CANONICAL"


class SlabLifecycleStatus(str, Enum):
    ACTIVE = "ACTIVE"
    DORMANT = "DORMANT"
    DEPRECATED = "DEPRECATED"


class MentionType(str, Enum):
    REFERENCE = "reference"
    QUESTION = "question"
    DELIBERATE_INVOCATION = "deliberate_invocation"


class GateStage(str, Enum):
    FUNCTION = "FUNCTION"
    EXPLICITNESS = "EXPLICITNESS"
    CONFIDENCE = "CONFIDENCE"


class ValidationStatus(str, Enum):
    PASS = "PASS"                # Clean — no violations detected
    FLAGGED = "FLAGGED"          # Soft violations (overridable layers) — surface to UI
    REGENERATE = "REGENERATE"    # Hard violation — retry with correction guidance (max 1)
    BLOCK = "BLOCK"              # Critical violation after retry — annotate response


class GateOutcome(str, Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    ESCALATE = "ESCALATE"  # surface ambiguity to user instead of silently defaulting
    DEFER = "DEFER"        # explicit pass-through to next stage without a decision
