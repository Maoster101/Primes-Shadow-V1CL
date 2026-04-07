"""Frame policy loader — reads frame_policy.yaml into a typed config.

All tunable thresholds live in app/corpus/state/frame_policy.yaml.
This module loads them once at import time and provides dot-access
via nested dataclasses. Services import `policy` and read values like:

    from .policy import policy
    policy.frame.decay.per_turn       # 0.85
    policy.drift.window_size          # 50
    policy.gate.confidence.min_exact  # 0.70

If the YAML file is missing or a key is absent, defaults are used.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml


# ── Nested dataclasses ───────────────────────────────────────────

@dataclass
class GateConfidence:
    low: float = 0.50
    ambiguous: float = 0.70
    min_exact: float = 0.70
    min_fuzzy: float = 0.90

@dataclass
class GateConfig:
    confidence: GateConfidence = field(default_factory=GateConfidence)

@dataclass
class FrameDecayConfig:
    per_turn: float = 0.85
    floor: float = 0.10

@dataclass
class SalienceConfig:
    alpha: float = 0.30
    low_content_inherit: float = 0.90

@dataclass
class FrameLimits:
    max_active_nodes: int = 24
    max_tentative_nodes: int = 8
    cold_corpus_hops_visible: int = 2
    collapse_beyond_hops: int = 3

@dataclass
class ActivationConfig:
    base_set_slab_weight: float = 1.0
    base_set_linked_weight: float = 0.5
    anchor_match_weight: float = 1.0
    invariant_linked_anchor_min: float = 0.8
    invariant_slab_weight: float = 1.0
    invariant_bundle_weight: float = 0.8
    mismatch_escalation: float = 0.6

@dataclass
class ConceptConfig:
    min_text_length: int = 8
    known_similarity: float = 0.70
    consolidation_similarity: float = 0.85
    promotion_turns: int = 3
    bundle_suggestion_turns: int = 3
    re_detect_salience: float = 0.60
    re_hit_boost: float = 0.15
    hierarchy_similarity: float = 0.75

@dataclass
class FrameConfig:
    decay: FrameDecayConfig = field(default_factory=FrameDecayConfig)
    salience: SalienceConfig = field(default_factory=SalienceConfig)
    limits: FrameLimits = field(default_factory=FrameLimits)
    activation: ActivationConfig = field(default_factory=ActivationConfig)
    concepts: ConceptConfig = field(default_factory=ConceptConfig)

@dataclass
class ConflictPreview:
    max_hops: int = 2
    min_threshold: float = 0.35
    decay_per_hop: float = 0.50
    block_crossing_committed_anchor_boundary: bool = True

@dataclass
class ConflictConfig:
    preview: ConflictPreview = field(default_factory=ConflictPreview)
    tension_surface_threshold: float = 0.50

@dataclass
class DriftWeights:
    affect_density: float = 0.30
    claim_volatility: float = 0.25
    rigor_drop: float = 0.25
    session_duration: float = 0.20

@dataclass
class DriftDomainThresholds:
    low: float = 0.39
    medium: float = 0.69

@dataclass
class DriftThresholds:
    external: DriftDomainThresholds = field(default_factory=lambda: DriftDomainThresholds(0.39, 0.69))
    internal: DriftDomainThresholds = field(default_factory=lambda: DriftDomainThresholds(0.54, 0.79))

@dataclass
class DriftConfig:
    window_size: int = 50
    session_duration_baseline: int = 5
    session_duration_history_size: int = 5   # how many past sessions to average
    composite_weights: DriftWeights = field(default_factory=DriftWeights)
    thresholds: DriftThresholds = field(default_factory=DriftThresholds)

@dataclass
class AffectBoost:
    orthographic_feature: float = 0.12
    emote_vocab: float = 0.08
    max_total: float = 0.35

@dataclass
class MatchingConfig:
    confidence_equivalence_delta: float = 0.05
    semantic_match_penalty: float = 0.80
    affect_boost: AffectBoost = field(default_factory=AffectBoost)

@dataclass
class DraftConfig:
    stack_cap: int = 10
    sweep_cadence: int = 8
    max_periodic_slabs: int = 3
    max_periodic_anchors: int = 5
    dedup_threshold: float = 0.90
    explicit_confidence: float = 0.75
    implicit_confidence: float = 0.50
    min_slab_content: int = 50

@dataclass
class ContextConfig:
    chars_per_token: int = 4
    target_tokens: int = 32768
    budget_floor: int = 2000
    message_overhead: int = 20
    slab_truncation: int = 8000
    frame_context_cap: int = 24

@dataclass
class OLIValidationConfig:
    untagged_claim_threshold: int = 3
    claim_min_length: int = 50
    claim_scan_limit: int = 15
    l4_slope_min_directives: int = 2

@dataclass
class PushbackConfig:
    calibration_window_max_hours: float = 2.0
    gauntlet_cooldown_turns: int = 5
    absence_sensitivity: float = 0.40

@dataclass
class FramePolicy:
    gate: GateConfig = field(default_factory=GateConfig)
    frame: FrameConfig = field(default_factory=FrameConfig)
    conflict: ConflictConfig = field(default_factory=ConflictConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    drafts: DraftConfig = field(default_factory=DraftConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    oli_validation: OLIValidationConfig = field(default_factory=OLIValidationConfig)
    pushback: PushbackConfig = field(default_factory=PushbackConfig)


# ── YAML loader ──────────────────────────────────────────────────

def _deep_merge(target: dict, source: dict) -> dict:
    """Recursively merge source into target, preferring source values."""
    for k, v in source.items():
        if k in target and isinstance(target[k], dict) and isinstance(v, dict):
            _deep_merge(target[k], v)
        else:
            target[k] = v
    return target


def _apply_dict(obj: Any, data: dict) -> None:
    """Apply a flat/nested dict onto a dataclass instance."""
    for k, v in data.items():
        if not hasattr(obj, k):
            continue
        current = getattr(obj, k)
        if isinstance(v, dict) and hasattr(current, '__dataclass_fields__'):
            _apply_dict(current, v)
        else:
            setattr(obj, k, type(current)(v) if not isinstance(v, type(current)) else v)


def load_policy(yaml_path: Path | None = None) -> FramePolicy:
    """Load frame_policy.yaml into a FramePolicy dataclass.

    Missing keys use defaults. Missing file returns all-defaults.
    """
    fp = FramePolicy()
    if yaml_path is None:
        yaml_path = Path(__file__).resolve().parent.parent.parent / "app" / "corpus" / "state" / "frame_policy.yaml"
    if not yaml_path.exists():
        return fp
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        _apply_dict(fp, data)
    except Exception as e:
        print(f"[POLICY] Warning: failed to load frame_policy.yaml: {e}")
    return fp


# ── Module-level singleton ───────────────────────────────────────
# Import and use: `from .policy import policy`

policy = load_policy()
