"""Corpus persistence service.

Owns all reads and writes to the authoritative corpus store.
Implements the 6 validator checks from §24.1. Fail-closed.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Optional
import yaml

from ..models.schemas import Anchor, Slab, KeyBundle, Edge, Gate
from ..models.enums import OLIMode, SlabType, SlabLifecycleStatus

CORPUS_ROOT = Path("app/corpus")
OBJECTS = CORPUS_ROOT / "objects"


def _load_yaml(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, list) else []


def _save_yaml(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


class CorpusStore:
    """In-memory corpus backed by YAML files. Validates on load and before commit."""

    def __init__(self, root: Optional[Path] = None):
        self.root = root or CORPUS_ROOT
        self.objects = self.root / "objects"
        self.anchors: dict[str, Anchor] = {}
        self.slabs: dict[str, Slab] = {}
        self.bundles: dict[str, KeyBundle] = {}
        self.edges: dict[str, Edge] = {}
        self.gates: dict[str, Gate] = {}

    def load(self) -> list[str]:
        """Load corpus from disk. Returns list of validation errors (empty = OK)."""
        raw_anchors = _load_yaml(self.objects / "anchors.yaml")
        raw_slabs = _load_yaml(self.objects / "slabs.yaml")
        raw_bundles = _load_yaml(self.objects / "key_bundles.yaml")
        raw_edges = _load_yaml(self.objects / "edges.yaml")
        raw_gates = _load_yaml(self.objects / "gates.yaml")

        self.anchors = {a["id"]: Anchor(**a) for a in raw_anchors}
        self.slabs = {s["id"]: Slab(**s) for s in raw_slabs}
        self.bundles = {b["id"]: KeyBundle(**b) for b in raw_bundles}
        self.edges = {}
        for e in raw_edges:
            # Handle the 'from'/'to' alias
            edge = Edge.model_validate(e)
            self.edges[edge.id] = edge
        self.gates = {g["id"]: Gate(**g) for g in raw_gates}

        return self.validate()

    def save(self) -> None:
        """Persist current state to YAML."""
        _save_yaml(
            self.objects / "anchors.yaml",
            [a.model_dump(mode="json") for a in self.anchors.values()],
        )
        _save_yaml(
            self.objects / "slabs.yaml",
            [s.model_dump(mode="json") for s in self.slabs.values()],
        )
        _save_yaml(
            self.objects / "key_bundles.yaml",
            [b.model_dump(mode="json") for b in self.bundles.values()],
        )
        _save_yaml(
            self.objects / "edges.yaml",
            [e.model_dump(mode="json", by_alias=True) for e in self.edges.values()],
        )
        _save_yaml(
            self.objects / "gates.yaml",
            [g.model_dump(mode="json") for g in self.gates.values()],
        )

    def all_ids(self) -> set[str]:
        return set(self.anchors) | set(self.slabs) | set(self.bundles) | set(self.gates)

    def validate(self) -> list[str]:
        """Run the 6 checks from §24.1. Returns list of errors."""
        errors: list[str] = []
        all_ids = self.all_ids()

        # Check 1: Unique IDs across all object types
        id_sets = {
            "anchor": set(self.anchors),
            "slab":   set(self.slabs),
            "bundle": set(self.bundles),
            "gate":   set(self.gates),
        }
        type_names = list(id_sets)
        for i, a in enumerate(type_names):
            for b in type_names[i + 1:]:
                for dup in id_sets[a] & id_sets[b]:
                    errors.append(f"Duplicate ID across object types ({a}/{b}): {dup}")

        # Check 2: Invokes targets exist
        for anchor in self.anchors.values():
            for target in anchor.invokes:
                if target not in all_ids:
                    errors.append(f"Anchor {anchor.id} invokes missing target: {target}")

        # Check 3: Supports targets exist
        for bundle in self.bundles.values():
            for target in bundle.supports:
                if target not in all_ids:
                    errors.append(f"Bundle {bundle.id} supports missing target: {target}")

        # Check 4: Version suffix match — id ends _vN, meta.version = vN
        for obj_id, obj in [
            *self.anchors.items(), *self.slabs.items(),
            *self.bundles.items(), *self.gates.items(),
        ]:
            version = obj.meta.version if hasattr(obj, "meta") else getattr(obj, "version", None)
            if version:
                expected_suffix = f"_{version}"
                if not obj_id.endswith(expected_suffix):
                    errors.append(
                        f"ID {obj_id} does not end with expected suffix {expected_suffix}"
                    )

        # Check 5: depends_on targets exist
        for obj_id, obj in [
            *self.anchors.items(), *self.slabs.items(),
            *self.bundles.items(), *self.gates.items(),
        ]:
            for dep in getattr(obj, "depends_on", []):
                if dep not in all_ids:
                    errors.append(f"{obj_id} depends_on missing target: {dep}")

        # Check 6: Supersedes exists
        for obj_id, obj in [
            *self.anchors.items(), *self.slabs.items(),
            *self.bundles.items(), *self.gates.items(),
        ]:
            meta = getattr(obj, "meta", None)
            if meta and meta.supersedes and meta.supersedes not in all_ids:
                errors.append(f"{obj_id} supersedes missing target: {meta.supersedes}")

        return errors

    # ── Phase 5: Base set query ──────────────────────────────────

    def base_set_slabs(self, oli_mode: OLIMode = OLIMode.OFF) -> list[Slab]:
        """Return slabs that belong in the session base set for the given OLI mode.

        Selection rules:
          1. Only ACTIVE slabs (DORMANT/DEPRECATED excluded)
          2. CONSTITUTIONAL slabs are always in the base set (they define layer rules)
          3. CANONICAL slabs are always in the base set (domain context)
          4. INVARIANT slabs are NOT in the base set — they activate on trigger only
          5. REFERENCE slabs are NOT in the base set — loaded on demand
          6. requires_oli_mode gating:
             - None  → always included
             - ON    → included only when oli_mode == ON
             - OFF   → included only when oli_mode == OFF

        Results are dependency-sorted: if slab A depends on slab B, B comes first.
        """
        candidates = []
        for slab in self.slabs.values():
            # Filter 1: lifecycle
            if slab.lifecycle_status != SlabLifecycleStatus.ACTIVE:
                continue
            # Filter 2: type — only CONSTITUTIONAL and CANONICAL in base set
            if slab.type not in (SlabType.CONSTITUTIONAL, SlabType.CANONICAL):
                continue
            # Filter 3: OLI mode gating
            if slab.requires_oli_mode is not None:
                if OLIMode(slab.requires_oli_mode) != oli_mode:
                    continue
            candidates.append(slab)

        # Dependency sort (topological): slabs with deps come after their deps
        return self._topo_sort_slabs(candidates)

    def invariant_slabs(self) -> list[Slab]:
        """Return all ACTIVE INVARIANT slabs (for conditional activation rules)."""
        return [
            s for s in self.slabs.values()
            if s.lifecycle_status == SlabLifecycleStatus.ACTIVE
            and s.type == SlabType.INVARIANT
        ]

    def _topo_sort_slabs(self, slabs: list[Slab]) -> list[Slab]:
        """Topological sort: dependencies before dependents."""
        id_set = {s.id for s in slabs}
        by_id = {s.id: s for s in slabs}
        visited = set()
        order = []

        def visit(sid):
            if sid in visited or sid not in id_set:
                return
            visited.add(sid)
            slab = by_id[sid]
            for dep in slab.depends_on:
                if dep in id_set:
                    visit(dep)
            order.append(slab)

        for s in slabs:
            visit(s.id)
        return order

    def get_reverse_deps(self, node_id: str) -> list[str]:
        """§4.4 CascadeIndex — compute who depends on this node."""
        dependents = []
        for obj_id, obj in [
            *self.anchors.items(), *self.slabs.items(),
            *self.bundles.items(), *self.gates.items(),
        ]:
            if node_id in getattr(obj, "depends_on", []):
                dependents.append(obj_id)
            if node_id in getattr(obj, "invokes", []):
                dependents.append(obj_id)
            if node_id in getattr(obj, "supports", []):
                dependents.append(obj_id)
        return dependents
