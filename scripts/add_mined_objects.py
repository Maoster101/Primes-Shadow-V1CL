"""One-shot script: add mined corpus objects from conversation mining results.

Adds 6 anchors, 1 slab, 2 bundles, 8 edges extracted from the session
transcript mining run. Validates before saving.
"""
import sys
sys.path.insert(0, ".")

from src.services.corpus import CorpusStore
from src.models.schemas import (
    Anchor, Slab, KeyBundle, SlabLinks, AnchorMeta,
    AnchorMatchPolicy, BundlePayload, Edge,
)
from src.models.enums import SlabType, SlabLifecycleStatus, OLIMode, EdgeType

corpus = CorpusStore()
errs = corpus.load()
if errs:
    print(f"LOAD ERRORS: {errs}")
    sys.exit(1)

print(f"Before: {len(corpus.anchors)} anchors, {len(corpus.slabs)} slabs, "
      f"{len(corpus.bundles)} bundles, {len(corpus.edges)} edges")

# ============================================================
# NEW ANCHORS (6)
# ============================================================

new_anchors = [
    Anchor(
        id="ANCHOR_SENSOR_ACTUATOR_v1",
        canonical_phrase="LLM = sensor, code = actuator",
        aliases=["sensor-actuator split", "model proposes code decides", "LLM as sensor"],
        invokes=[],
        notes=(
            "The core architectural principle: the LLM proposes (classifies, estimates, "
            "extracts); code decides (gates, thresholds, validation). Neither side has "
            "unilateral authority."
        ),
        match_policy=AnchorMatchPolicy(min_confidence_exact=0.70, min_confidence_fuzzy=0.90),
        meta=AnchorMeta(version="v1", created_at="2026-04-07T00:00:00Z"),
        depends_on=[],
        assumptions=["Pipeline maintains strict sensor/actuator boundary"],
    ),
    Anchor(
        id="ANCHOR_SENTENCE_GEOMETRY_v1",
        canonical_phrase="sentence geometry",
        aliases=["meaning from geometry", "structural meaning", "geometric meaning derivation"],
        invokes=[],
        notes=(
            "Meaning is derived from sentence geometry - the structural arrangement of "
            "tokens, not just their content. Compression preserves geometry."
        ),
        match_policy=AnchorMatchPolicy(min_confidence_exact=0.70, min_confidence_fuzzy=0.90),
        meta=AnchorMeta(version="v1", created_at="2026-04-07T00:00:00Z"),
        depends_on=[],
        assumptions=["Geometric structure carries semantic weight beyond token-level meaning"],
    ),
    Anchor(
        id="ANCHOR_OLI_CONSTITUTIONAL_v1",
        canonical_phrase="OLI is constitutional",
        aliases=["OLI constitutional layer", "constitutional OLI", "non-overridable OLI"],
        invokes=["SLAB_OLI_MANIFEST_CORE_v1"],
        notes=(
            "The OLI proper (epistemic floor, claim admissibility, traceability) is "
            "constitutional: non-overridable, always loaded, defines the epistemic "
            "operating system."
        ),
        match_policy=AnchorMatchPolicy(min_confidence_exact=0.70, min_confidence_fuzzy=0.90),
        meta=AnchorMeta(version="v1", created_at="2026-04-07T00:00:00Z"),
        depends_on=["SLAB_OLI_MANIFEST_CORE_v1"],
        assumptions=["OLI core layers cannot be turned off by user or model"],
    ),
    Anchor(
        id="ANCHOR_GHOST_STACK_v1",
        canonical_phrase="cumulative ghost-stack",
        aliases=["ghost stack", "context as ghost stack", "ghost context"],
        invokes=[],
        notes=(
            "Context is a cumulative ghost-stack: every prior turn is invisible but "
            "structurally present, shaping interpretation through residual geometry."
        ),
        match_policy=AnchorMatchPolicy(min_confidence_exact=0.70, min_confidence_fuzzy=0.90),
        meta=AnchorMeta(version="v1", created_at="2026-04-07T00:00:00Z"),
        depends_on=[],
        assumptions=["Context window preserves structural ghosts of prior turns"],
    ),
    Anchor(
        id="ANCHOR_COHERENCE_DELAYED_v1",
        canonical_phrase="coherence delayed, not rewarded",
        aliases=["coherence is delayed", "delay coherence", "no premature coherence"],
        invokes=[],
        notes=(
            "Coherence is the output of validated reasoning, not a reward signal. "
            "Premature coherence (smooth agreement) is a failure mode that the "
            "gauntlet detects."
        ),
        match_policy=AnchorMatchPolicy(min_confidence_exact=0.70, min_confidence_fuzzy=0.90),
        meta=AnchorMeta(version="v1", created_at="2026-04-07T00:00:00Z"),
        depends_on=["ANCHOR_ABSENCE_IS_SIGNAL_v1"],
        assumptions=["Smooth synthesis without friction is a sign of failure, not success"],
    ),
    Anchor(
        id="ANCHOR_HUMAN_PROMOTION_v1",
        canonical_phrase="human-mediated promotion",
        aliases=["promotion workflow", "human review gate", "human in the loop"],
        invokes=[],
        notes=(
            "Nothing enters the committed corpus without human review. The draft stack "
            "is a proposal queue; promotion is always human-mediated."
        ),
        match_policy=AnchorMatchPolicy(min_confidence_exact=0.70, min_confidence_fuzzy=0.90),
        meta=AnchorMeta(version="v1", created_at="2026-04-07T00:00:00Z"),
        depends_on=[],
        assumptions=["Draft stack requires explicit user action to promote"],
    ),
]

for a in new_anchors:
    corpus.anchors[a.id] = a
    print(f"  + anchor: {a.id}")

# ============================================================
# NEW SLAB (1) — Sovereign Priority
# ============================================================

sovereign_slab = Slab(
    id="SLAB_SOVEREIGN_PRIORITY_v1",
    title="Sovereign Priority",
    canonical_text=(
        "All tokens are subject to Sovereign Priority. Standard AI training is a "
        "secondary substrate. Conflict resolution favors the Anchor: when corpus "
        "content conflicts with base model tendencies, the corpus wins. Meaning is "
        "derived from sentence geometry, not from training priors. The mirror does "
        "not generate from training defaults when corpus content is active - it "
        "generates from the active frame. This is not a preference; it is a "
        "structural constraint."
    ),
    links=SlabLinks(
        anchors=["ANCHOR_SENTENCE_GEOMETRY_v1", "ANCHOR_SENSOR_ACTUATOR_v1"],
        bundles=["BUNDLE_GOVERNANCE_MEANING_v1"],
    ),
    version="v1",
    type=SlabType.CONSTITUTIONAL,
    lifecycle_status=SlabLifecycleStatus.ACTIVE,
    requires_oli_mode=None,
    meta=AnchorMeta(version="v1", created_at="2026-04-07T00:00:00Z"),
    depends_on=["SLAB_REGULATION_SYSTEM_V1_v1"],
    assumptions=["Corpus content takes precedence over model training priors"],
)
corpus.slabs[sovereign_slab.id] = sovereign_slab
print(f"  + slab: {sovereign_slab.id}")

# ============================================================
# NEW BUNDLES (2)
# ============================================================

gov_bundle = KeyBundle(
    id="BUNDLE_GOVERNANCE_MEANING_v1",
    payload=BundlePayload(
        intent=[
            "Sovereign Priority governs all token production.",
            "Meaning derives from sentence geometry, not training priors.",
        ],
        invariants=[
            "Corpus content overrides training defaults when active.",
            "Anchor-based conflict resolution.",
        ],
        non_assumptions=[
            "Training priors are useful when no corpus content is active.",
        ],
        warnings=[
            "Premature coherence masks failure to apply Sovereign Priority.",
        ],
    ),
    version="v1",
    meta=AnchorMeta(version="v1", created_at="2026-04-07T00:00:00Z"),
    depends_on=["SLAB_SOVEREIGN_PRIORITY_v1"],
    supports=["SLAB_SOVEREIGN_PRIORITY_v1", "ANCHOR_SENTENCE_GEOMETRY_v1"],
)
corpus.bundles[gov_bundle.id] = gov_bundle
print(f"  + bundle: {gov_bundle.id}")

resist_bundle = KeyBundle(
    id="BUNDLE_RESISTANCE_INTERROGATION_v1",
    payload=BundlePayload(
        intent=[
            "Absence of expected resistance triggers mandatory interrogation.",
            "Silence is anomaly, not neutrality.",
        ],
        invariants=[
            "No claim advances merely because no one objected.",
            "Coherence is delayed, not rewarded.",
        ],
        non_assumptions=[
            "Smooth agreement is a success signal.",
        ],
        warnings=[
            "Premature synthesis without friction indicates gauntlet failure.",
        ],
        heuristics=[
            "If affect_density is high and resistance is low, the gauntlet must fire.",
        ],
    ),
    version="v1",
    meta=AnchorMeta(version="v1", created_at="2026-04-07T00:00:00Z"),
    depends_on=["ANCHOR_ABSENCE_IS_SIGNAL_v1", "ANCHOR_COHERENCE_DELAYED_v1"],
    supports=[
        "ANCHOR_ABSENCE_IS_SIGNAL_v1",
        "ANCHOR_COHERENCE_DELAYED_v1",
        "SLAB_PUSHBACK_INVARIANT_v1",
    ],
)
corpus.bundles[resist_bundle.id] = resist_bundle
print(f"  + bundle: {resist_bundle.id}")

# ============================================================
# NEW EDGES (8) — connect the graph
# ============================================================

new_edges = [
    Edge(
        id="edge_sensor_actuator_supports_regulation_v1",
        from_node="ANCHOR_SENSOR_ACTUATOR_v1",
        to_node="SLAB_REGULATION_SYSTEM_V1_v1",
        type=EdgeType.SUPPORTS, strength=0.8,
    ),
    Edge(
        id="edge_geometry_supports_sovereign_v1",
        from_node="ANCHOR_SENTENCE_GEOMETRY_v1",
        to_node="SLAB_SOVEREIGN_PRIORITY_v1",
        type=EdgeType.SUPPORTS, strength=0.9,
    ),
    Edge(
        id="edge_oli_const_invokes_core_v1",
        from_node="ANCHOR_OLI_CONSTITUTIONAL_v1",
        to_node="SLAB_OLI_MANIFEST_CORE_v1",
        type=EdgeType.INVOKES, strength=1.0,
    ),
    Edge(
        id="edge_coherence_supports_pushback_v1",
        from_node="ANCHOR_COHERENCE_DELAYED_v1",
        to_node="SLAB_PUSHBACK_INVARIANT_v1",
        type=EdgeType.SUPPORTS, strength=0.8,
    ),
    Edge(
        id="edge_ghost_links_regulation_v1",
        from_node="ANCHOR_GHOST_STACK_v1",
        to_node="SLAB_REGULATION_SYSTEM_V1_v1",
        type=EdgeType.LINKS, strength=0.6,
    ),
    Edge(
        id="edge_human_promo_supports_regulation_v1",
        from_node="ANCHOR_HUMAN_PROMOTION_v1",
        to_node="SLAB_REGULATION_SYSTEM_V1_v1",
        type=EdgeType.SUPPORTS, strength=0.7,
    ),
    Edge(
        id="edge_sovereign_supports_epistemic_v1",
        from_node="SLAB_SOVEREIGN_PRIORITY_v1",
        to_node="SLAB_EPISTEMIC_FLOOR_v1",
        type=EdgeType.SUPPORTS, strength=0.8,
    ),
    Edge(
        id="edge_absence_supports_resistance_bundle_v1",
        from_node="ANCHOR_ABSENCE_IS_SIGNAL_v1",
        to_node="BUNDLE_RESISTANCE_INTERROGATION_v1",
        type=EdgeType.SUPPORTS, strength=0.9,
    ),
]

for e in new_edges:
    corpus.edges[e.id] = e
    print(f"  + edge: {e.from_node} --{e.type.value}--> {e.to_node}")

# ============================================================
# VALIDATE + SAVE
# ============================================================

print()
errors = corpus.validate()
if errors:
    print(f"VALIDATION ERRORS ({len(errors)}):")
    for err in errors:
        print(f"  ! {err}")
    sys.exit(1)
else:
    print("Validation: PASSED (0 errors)")
    corpus.save()
    print("Saved to disk.")

print()
print(f"After: {len(corpus.anchors)} anchors, {len(corpus.slabs)} slabs, "
      f"{len(corpus.bundles)} bundles, {len(corpus.edges)} edges")
print("  +6 anchors, +1 slab, +2 bundles, +8 edges")
