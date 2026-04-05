"""Pydantic models for all spec-defined schemas."""
from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from .enums import (
    MessageFunction, ChatStatus, DraftStatus, NodeType, EdgeType,
    OLIMode, L4Posture, DriftSeverity, DomainMode, DampeningLevel,
    ClaimTag, VerificationOutcome, MatchTier,
    SlabType, SlabLifecycleStatus, MentionType,
)


# §7.1 — MessageClassification
class MessageClassification(BaseModel):
    function: MessageFunction
    explicit: bool = False
    confidence: float = Field(ge=0.0, le=1.0)
    mention_type: Optional[MentionType] = None
    notes: Optional[str] = None


# §4.3 — Chat
class Chat(BaseModel):
    id: str
    title: str
    status: ChatStatus = ChatStatus.ACTIVE
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    linked_drafts: list[str] = Field(default_factory=list)
    linked_commits: list[str] = Field(default_factory=list)
    last_frame_snapshot_ref: Optional[str] = None


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str
    turn: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    classification: Optional[MessageClassification] = None
    drift_estimate: Optional[DriftEstimate] = None


# §17.9 — Drift estimate (model-proposed per turn)
class DriftEstimate(BaseModel):
    affect_density: float = Field(ge=0.0, le=1.0, default=0.0)
    claim_volatility: float = Field(ge=0.0, le=1.0, default=0.0)
    rigor_drop: float = Field(ge=0.0, le=1.0, default=0.0)
    domain_mode: DomainMode = DomainMode.EXTERNAL
    window_composite: float = Field(ge=0.0, le=1.0, default=0.0)


# §8.3 — Anchor
class AnchorMeta(BaseModel):
    version: str = "v1"
    supersedes: Optional[str] = None
    semantic_change: bool = False
    created_at: Optional[str] = None


class AnchorMatchPolicy(BaseModel):
    gate_required: bool = True
    allowed_functions: list[MessageFunction] = Field(
        default_factory=lambda: [MessageFunction.CONTEXT_COMPRESSION]
    )
    format_gate: Optional[str] = None
    min_confidence_exact: float = 0.8
    min_confidence_fuzzy: float = 0.6


class Anchor(BaseModel):
    id: str
    canonical_phrase: str
    aliases: list[str] = Field(default_factory=list)
    invokes: list[str] = Field(default_factory=list)
    notes: str = ""
    match_policy: AnchorMatchPolicy = Field(default_factory=AnchorMatchPolicy)
    meta: AnchorMeta = Field(default_factory=AnchorMeta)
    depends_on: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


# §4.1.1 — Slab (Design + Schemas doc)
class SlabLinks(BaseModel):
    anchors: list[str] = Field(default_factory=list)
    bundles: list[str] = Field(default_factory=list)


class Slab(BaseModel):
    id: str
    title: str = ""
    canonical_text: str
    links: SlabLinks = Field(default_factory=SlabLinks)
    version: str = "v1"
    type: SlabType = SlabType.CANONICAL
    lifecycle_status: SlabLifecycleStatus = SlabLifecycleStatus.ACTIVE
    requires_oli_mode: Optional[OLIMode] = None
    meta: AnchorMeta = Field(default_factory=AnchorMeta)
    depends_on: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    provenance_refs: list[ProvenanceRef] = Field(default_factory=list)


# §13.3 — ProvenanceRef
class ProvenanceRef(BaseModel):
    ref_id: str
    store: str  # "workbench" | "library"
    type: str  # conversation_segment | synthesis_notes | etc.
    note: Optional[str] = None


# §4.5 — Bundle payload
class BundlePayload(BaseModel):
    intent: list[str] = Field(min_length=1, max_length=5)
    invariants: list[str] = Field(default_factory=list, max_length=8)
    non_assumptions: list[str] = Field(default_factory=list, max_length=6)
    warnings: list[str] = Field(default_factory=list, max_length=8)
    heuristics: list[str] = Field(default_factory=list, max_length=8)
    markers: list[str] = Field(default_factory=list)
    rules: list[str] = Field(default_factory=list)
    activation_clause: list[str] = Field(default_factory=list)
    canonical_quote_handles: list[str] = Field(default_factory=list)
    closing_clause: Optional[str] = None


class KeyBundle(BaseModel):
    id: str
    payload: BundlePayload
    version: str = "v1"
    meta: AnchorMeta = Field(default_factory=AnchorMeta)
    depends_on: list[str] = Field(default_factory=list)
    supports: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


# §10.3 — Edge (Design + Schemas doc)
class EdgeConditions(BaseModel):
    allowed_functions: Optional[list[MessageFunction]] = None
    note: Optional[str] = None


class Edge(BaseModel):
    id: str
    type: EdgeType
    from_node: str = Field(alias="from")
    to_node: str = Field(alias="to")
    weight: float = Field(ge=0.0, le=1.0, default=0.5)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)  # alias for weight, backward compat
    tension: Optional[float] = None
    conditions: Optional[EdgeConditions] = None

    model_config = {"populate_by_name": True}


# §9 — FrameState
class ActivationSource(BaseModel):
    source_type: str
    source_ref: str


class ConflictPair(BaseModel):
    node_a: str
    node_b: str
    tension: float
    semantic_change: float


class FrameDecay(BaseModel):
    per_turn: float = 0.85
    floor: float = 0.10


class FrameState(BaseModel):
    chat_id: str
    session_id: str
    # Separated by type per Design + Schemas doc
    active_anchors: dict[str, float] = Field(default_factory=dict)  # {anchor_id: weight}
    active_bundles: dict[str, float] = Field(default_factory=dict)  # {bundle_id: weight}
    active_slabs: dict[str, float] = Field(default_factory=dict)    # {slab_id: weight}
    active_concepts: dict[str, float] = Field(default_factory=dict) # {concept_id: weight}
    # Unified view for backward compat and rendering
    active_nodes: list[str] = Field(default_factory=list)
    salience_now: dict[str, float] = Field(default_factory=dict)
    salience_smoothed: dict[str, float] = Field(default_factory=dict)
    structural_weight: dict[str, float] = Field(default_factory=dict)
    activation_sources: dict[str, list[ActivationSource]] = Field(default_factory=dict)
    conflicts: list[ConflictPair] = Field(default_factory=list)
    mismatch_score: float = 0.0
    decay: FrameDecay = Field(default_factory=FrameDecay)
    last_updated_turn: int = 0
    # Corpus access patterns — session-scoped instrumentation
    corpus_hits: dict[str, int] = Field(default_factory=dict)      # {node_id: total_hit_count}
    corpus_last_hit: dict[str, int] = Field(default_factory=dict)  # {node_id: turn_of_last_hit}
    # Per-turn hit events for session-end trajectory replay: [[turn, [node_ids]], ...]
    corpus_hit_log: list[list] = Field(default_factory=list)


# §14.3 — DraftPacket / Packet (Design + Schemas doc)
class DraftPacket(BaseModel):
    id: str
    packet_type: str = "anchor"  # "anchor" | "bundle" | "slab" | "anchor+bundle"
    source_chat_id: str
    source_turns: list[int] = Field(default_factory=list)
    proposed_nodes: list[str] = Field(default_factory=list)
    proposed_edges: list[str] = Field(default_factory=list)
    justification: str = ""
    confidence: float = 0.0
    dependency_closure: list[str] = Field(default_factory=list)
    status: DraftStatus = DraftStatus.DRAFT_UNAUTHORIZED
    fact_claims: list[str] = Field(default_factory=list)
    # Inline objects per Design + Schemas doc
    anchor: Optional[dict] = None   # Anchor dict when packet_type includes "anchor"
    bundle: Optional[dict] = None   # Bundle dict when packet_type includes "bundle"
    slab: Optional[dict] = None     # Slab dict when packet_type is "slab"
    edges: list[dict] = Field(default_factory=list)  # minimal links created by this packet
    stale_reason: Optional[str] = None


# §14.1 — DraftStack
class DraftStack(BaseModel):
    session_id: str
    packets: list[str] = Field(default_factory=list)
    frozen: bool = False


# Proposal extraction raw output (model-proposed, pre-validation)
class ProposalRaw(BaseModel):
    type: str = "anchor"  # "anchor" | "slab"
    canonical_phrase: Optional[str] = None
    canonical_text: Optional[str] = None
    aliases: list[str] = Field(default_factory=list)
    justification: str = ""
    source_turns: list[int] = Field(default_factory=list)
    claim_tag: ClaimTag = ClaimTag.UNKNOWN
    notes: Optional[str] = None


# §26.3 — Runtime control header
class GateState(BaseModel):
    message_function: MessageFunction = MessageFunction.NEUTRAL
    anchor_resolution_allowed: bool = False
    confidence_flag: str = "normal"  # normal | low | ambiguous


class FrameStateSummary(BaseModel):
    active_nodes: list[str] = Field(default_factory=list)
    high_tension_pairs: list[list[str]] = Field(default_factory=list)
    mismatch_score: float = 0.0


class EnforcementFlags(BaseModel):
    claim_admissibility_required: bool = False
    degradation_flag: Optional[str] = None
    pushback_required: bool = False
    dampening_level: DampeningLevel = DampeningLevel.NONE


class OperatorState(BaseModel):
    """OP_01 state — asterisk wrap outcomes from the current turn.

    Every closed `*...*` span becomes one entry in `wrapped_spans` with
    its extracted feature set and a derived `primary_reading` label.
    The Mirror composes interpretations from the feature set — the
    label is a quick hint, not authoritative. `correction_hint` carries
    the tail of an UNCLOSED wrap (user self-correcting mid-turn).
    """
    wrapped_spans: list[dict] = Field(default_factory=list)
    # each: {span_index, text, anchor_hits, features, primary_reading}
    correction_hint: Optional[str] = None


class RuntimeHeader(BaseModel):
    oli_mode: OLIMode = OLIMode.OFF
    oli_version: str = "v2.1"
    layer_control: dict = Field(default_factory=lambda: {
        "max_layer": "L3",
        "l4_posture": L4Posture.USER_LEADS.value,
    })
    gate_state: GateState = Field(default_factory=GateState)
    frame_state_summary: FrameStateSummary = Field(default_factory=FrameStateSummary)
    drift_estimate: DriftEstimate = Field(default_factory=DriftEstimate)
    enforcement_flags: EnforcementFlags = Field(default_factory=EnforcementFlags)
    operator_state: OperatorState = Field(default_factory=OperatorState)


# Fix forward reference
ChatMessage.model_rebuild()
Slab.model_rebuild()
