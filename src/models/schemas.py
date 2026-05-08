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
    GateStage, GateOutcome,
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
    # Phase 2A: every chat binds to a collection at creation. Drafts mined
    # during this chat's turns stamp their raw sidecar's _target_collection
    # with this value, so the worklist filters correctly in Drafts/Verify.
    # Optional for backward compat — pre-2A chats default to "default" at
    # read time. All new chats get this set by the creation endpoint.
    collection_id: Optional[str] = None


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
    min_confidence_exact: float = 0.70   # §8 — exact/partial is reliable, lower bar
    min_confidence_fuzzy: float = 0.90   # §8 — fuzzy is noisy, demand high confidence


class Anchor(BaseModel):
    id: str
    canonical_phrase: str
    aliases: list[str] = Field(default_factory=list)
    invokes: list[str] = Field(default_factory=list)
    notes: str = ""
    lifecycle_status: SlabLifecycleStatus = SlabLifecycleStatus.ACTIVE
    match_policy: AnchorMatchPolicy = Field(default_factory=AnchorMatchPolicy)
    meta: AnchorMeta = Field(default_factory=AnchorMeta)
    depends_on: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


# §4.1.1 — Slab (Design + Schemas doc)
class InlineAnchor(BaseModel):
    """An anchor record stored by-value inside a slab.

    Produced by the anchor consolidation pass: anchors that fail the
    cross-slab-reach criterion (touched by <2 slabs AND not in any
    multi-slab bundle) get demoted from ``corpus.anchors`` into the
    parent slab's ``links.anchors_inline`` list.

    The anchor's text content is preserved verbatim (canonical_phrase,
    aliases, notes) so synthesis-time text search and quote handling
    still resolve. The ``id`` is preserved for traceability — if the
    anchor is later promoted back to the corpus (e.g. another doc
    introduces a second slab that touches the concept), the same id
    can be re-used. The Pydantic default of empty list keeps existing
    corpora backwards-compatible: pre-consolidation slabs have no
    inline anchors and the field is just ``[]``.
    """
    id: str
    canonical_phrase: str
    aliases: list[str] = Field(default_factory=list)
    notes: str = ""
    confidence: float = 0.5


class SlabLinks(BaseModel):
    anchors: list[str] = Field(default_factory=list)
    bundles: list[str] = Field(default_factory=list)
    # Anchors demoted from corpus-level — see InlineAnchor docstring.
    # Backwards-compatible: pre-consolidation slabs default to empty.
    anchors_inline: list[InlineAnchor] = Field(default_factory=list)


class Slab(BaseModel):
    id: str
    title: str = ""
    canonical_text: str
    links: SlabLinks = Field(default_factory=SlabLinks)
    version: str = "v1"
    type: SlabType = SlabType.REFERENCE
    lifecycle_status: SlabLifecycleStatus = SlabLifecycleStatus.ACTIVE
    requires_oli_mode: Optional[OLIMode] = None
    meta: AnchorMeta = Field(default_factory=AnchorMeta)
    depends_on: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    provenance_refs: list[ProvenanceRef] = Field(default_factory=list)
    # Slab-level metadata previously held in a 1:1 "coherence bundle"
    # corpus node. Inlined here because the bundle was structurally
    # redundant (same content, derived only from this slab, no
    # cross-slab role). Backwards-compatible: pre-migration slabs
    # default to empty lists.
    intent: list[str] = Field(default_factory=list)
    invariants: list[str] = Field(default_factory=list)


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
    lifecycle_status: SlabLifecycleStatus = SlabLifecycleStatus.ACTIVE
    meta: AnchorMeta = Field(default_factory=AnchorMeta)
    depends_on: list[str] = Field(default_factory=list)
    supports: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


# §8.4 — Gate + GateRule (v3 declarative gate engine)
# Gates reify the imperative gate logic currently living in
# anchor_matcher.gate_check() and the classifier's confidence checks
# into data that can be edited without touching code. A Gate owns an
# ordered list of GateRule entries; the evaluator iterates rules in
# order, returning the outcome of the first rule whose `condition`
# expression evaluates truthy against a context dict built from the
# current MessageClassification, Anchor, and session state. If no rule
# matches, the gate's default_outcome applies.
class GateRule(BaseModel):
    id: str
    condition: str  # simpleeval expression; e.g. "classification.explicit == True"
    outcome: GateOutcome
    priority: int = 100  # lower runs first; stable sort within a Gate
    notes: str = ""


class Gate(BaseModel):
    id: str
    stage: GateStage
    description: str = ""
    rules: list[GateRule] = Field(default_factory=list)
    default_outcome: GateOutcome = GateOutcome.DENY
    version: str = "v1"
    meta: AnchorMeta = Field(default_factory=AnchorMeta)
    depends_on: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


# §10.3 — Edge (Design + Schemas doc)
class EdgeConditions(BaseModel):
    allowed_functions: Optional[list[MessageFunction]] = None
    note: Optional[str] = None


class Edge(BaseModel):
    """Reified typed edge between two corpus nodes.

    `weight` and `confidence` are semantically distinct and MUST NOT be
    treated as aliases (they were identical in the initial hand-curated
    seed, which caused a lot of confusion):

    - **weight**: how strongly activation should cascade along this edge
      when fired. Scales the target's inherited activation. Think
      "signal gain". Curator-set for hand-authored edges, model-proposed
      for mined edges.
    - **confidence**: our epistemic belief that this edge exists and is
      correctly typed. 1.0 for hand-curated/CONSTITUTIONAL edges (we
      know it's real), lower for LLM-mined edges awaiting corroboration.
      Scales cascade as a multiplier — low-confidence edges propagate
      less until reinforced.

    Effective cascade strength = weight * confidence * kernel_for_type
    (see frame_manager.CASCADE_KERNEL).
    """
    id: str
    type: EdgeType
    from_node: str = Field(alias="from")
    to_node: str = Field(alias="to")
    weight: float = Field(ge=0.0, le=1.0, default=0.5)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    tension: Optional[float] = None
    conditions: Optional[EdgeConditions] = None
    # Free-text "why this edge exists" reasoning produced at mining
    # time. Optional + None default for backward-compat with existing
    # corpus YAML that doesn't have this field. Synthesis surfaces it
    # alongside CONFLICTS / TENSIONS edges so the LLM has the model's
    # original justification for the dialectic pair, not just the
    # bare existence of the edge. Hand-curated edges typically leave
    # this empty.
    justification: Optional[str] = None

    model_config = {"populate_by_name": True}


# Phase 3 — Proposed edges (mining output, pre-review).
# Distinct from Edge: endpoints may reference tentative nodes that don't
# yet exist in corpus.edges. Status transitions:
#   PROPOSED  -> model mined it, awaiting human review
#   ACCEPTED  -> human approved, but endpoints may still be tentative;
#                a promote hook upgrades to COMMITTED once both exist
#                in the live corpus
#   COMMITTED -> written to corpus.edges; mirrors a real Edge object
#   REJECTED  -> human discarded; kept for audit trail, never committed
class ProposedEdge(BaseModel):
    id: str
    type: EdgeType
    from_node: str  # node_id (resolved from LLM label)
    to_node: str    # node_id (resolved from LLM label)
    from_label: str  # original LLM label (audit trail when IDs change)
    to_label: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    justification: str = ""
    status: str = "PROPOSED"  # PROPOSED | ACCEPTED | COMMITTED | REJECTED
    source_chat_id: Optional[str] = None
    source_turn: Optional[int] = None
    committed_edge_id: Optional[str] = None  # set when promoted to corpus.edges
    created_at: datetime = Field(default_factory=datetime.utcnow)


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
    # §17.3 — Truth pressure: per-node epistemic tension accumulator.
    # Incremented when a node's claims are questioned, challenged, or
    # when a pushback gauntlet fires against content related to the node.
    # High truth_pressure on a node triggers targeted gauntlet checks
    # even when the global mismatch_score is low.
    truth_pressure: dict[str, float] = Field(default_factory=dict)  # {node_id: pressure 0.0-1.0}
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
    review_mode: bool = False  # Corpus review — editorial assessment permitted


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
    active_model: str = ""               # Ollama model tag (e.g. "gemma3:12b", "gpt-oss:20b")
    active_model_family: str = ""        # Model family (e.g. "gemma3", "gptoss")
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
    anchor_hits: list[dict] = Field(default_factory=list)
    # each: {anchor_id, canonical_phrase, notes, confidence, method, invokes}


# Fix forward reference
ChatMessage.model_rebuild()
Slab.model_rebuild()
