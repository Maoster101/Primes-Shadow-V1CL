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


class NodeStatus(str, Enum):
    CORPUS = "corpus"
    TENTATIVE = "tentative"


class EdgeType(str, Enum):
    INVOKES = "INVOKES"
    SUPPORTS = "SUPPORTS"
    CONFLICTS = "CONFLICTS"
    LINKS = "LINKS"
    PARENT_OF = "PARENT_OF"
    SEQUENCE = "SEQUENCE"  # Narrative ordering — A precedes B in the story spine


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
