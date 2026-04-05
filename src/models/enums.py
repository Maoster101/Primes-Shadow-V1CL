"""Core enumerations from the design spec."""
from enum import Enum


class MessageFunction(str, Enum):
    CONTEXT_COMPRESSION = "context_compression"
    OBJECT_OF_WORK = "object_of_work"
    AFFECT_RELEASE = "affect_release"
    RIGOR_WORK = "rigor_work"
    META_SCHEMA = "meta_schema"
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
