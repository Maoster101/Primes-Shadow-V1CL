"""Corpus persistence service — multi-collection aware.

Owns all reads and writes to the authoritative corpus store.
Implements the 6 validator checks from §24.1. Fail-closed.

Collections live under app/corpora/{name}/objects/.
The legacy app/corpus/ path is treated as the "default" collection.
A CorpusRegistry manages multiple CorpusStore instances and provides
a merged view for the active set.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Optional
import yaml

from ..models.schemas import Anchor, Slab, KeyBundle, Edge, Gate
from ..models.enums import OLIMode, SlabType, SlabLifecycleStatus

CORPUS_ROOT = Path("app/corpus")         # Legacy single-corpus path
CORPORA_ROOT = Path("app/corpora")       # Multi-collection root


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
    """In-memory corpus backed by YAML files. Validates on load and before commit.

    Each instance represents ONE collection (identified by collection_id).
    """

    def __init__(self, root: Optional[Path] = None, collection_id: str = "default"):
        self.collection_id = collection_id
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

        # Check 4: Version suffix match
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

    def base_set_slabs(
        self, oli_mode: OLIMode = OLIMode.OFF,
        include_reference: bool = True,
    ) -> list[Slab]:
        """Return slabs that belong in the session base set for the given OLI mode.

        By default, ALL ACTIVE slabs (CONSTITUTIONAL + CANONICAL + REFERENCE)
        are returned, so the model can see everything stored in cold corpus —
        including mined narrative content. INVARIANT slabs are still excluded
        (they activate conditionally via anchor match, not as base set).

        The three types are treated identically at the frame level but rendered
        differently in the system prompt (see context_packer._format_base_set_slabs):
        CONSTITUTIONAL/CANONICAL get the full slab text; REFERENCE gets tighter
        truncation to preserve context budget for chat history. The graph
        visibility filter keeps REFERENCE nodes hidden from the canvas until
        conversation engages them, so "everything loaded" doesn't mean "canvas
        cluttered from turn 0".

        Pass ``include_reference=False`` to get the old CONSTITUTIONAL+CANONICAL
        behavior (useful for tools that want only foundational content).
        """
        allowed_types: tuple[SlabType, ...] = (
            SlabType.CONSTITUTIONAL, SlabType.CANONICAL,
        )
        if include_reference:
            allowed_types = allowed_types + (SlabType.REFERENCE,)

        candidates = []
        for slab in self.slabs.values():
            if slab.lifecycle_status != SlabLifecycleStatus.ACTIVE:
                continue
            if slab.type not in allowed_types:
                continue
            if slab.requires_oli_mode is not None:
                if OLIMode(slab.requires_oli_mode) != oli_mode:
                    continue
            candidates.append(slab)
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


# ── Multi-Collection Registry ────────────────────────────────

class CorpusRegistry:
    """Manages multiple corpus collections and provides a merged view.

    The registry loads collections from app/corpora/{name}/ directories.
    It also supports the legacy app/corpus/ path as the "default" collection.

    Active collections can be toggled per-chat. The merged view combines
    all active collections into a single virtual corpus for matching and
    pipeline use.
    """

    def __init__(self):
        self.collections: dict[str, CorpusStore] = {}
        self.active_ids: set[str] = set()
        self._merged: Optional[CorpusStore] = None
        self._merged_dirty = True

    def discover(self) -> list[str]:
        """Discover available collections from disk.

        Scans app/corpora/ for subdirectories with objects/ folders.
        Also checks legacy app/corpus/ path.
        """
        found = []

        # Legacy path → "default"
        if CORPUS_ROOT.exists() and (CORPUS_ROOT / "objects").exists():
            found.append("default")

        # Multi-collection path
        if CORPORA_ROOT.exists():
            for child in sorted(CORPORA_ROOT.iterdir()):
                if child.is_dir() and (child / "objects").exists():
                    name = child.name
                    if name not in found:
                        found.append(name)

        return found

    def load_collection(self, collection_id: str) -> list[str]:
        """Load a single collection. Returns validation errors."""
        # Resolve path: prefer corpora/{id}/, fall back to legacy corpus/
        corpora_path = CORPORA_ROOT / collection_id
        if corpora_path.exists() and (corpora_path / "objects").exists():
            root = corpora_path
        elif collection_id == "default" and CORPUS_ROOT.exists():
            root = CORPUS_ROOT
        else:
            return [f"Collection '{collection_id}' not found"]

        store = CorpusStore(root=root, collection_id=collection_id)
        errors = store.load()
        self.collections[collection_id] = store
        self._merged_dirty = True
        return errors

    def load_all(self) -> dict[str, list[str]]:
        """Discover and load all collections. Returns {id: errors} map."""
        results = {}
        for cid in self.discover():
            results[cid] = self.load_collection(cid)
        # Activate all by default
        self.active_ids = set(self.collections.keys())
        # Self-heal: a slabify (collection → condensate slab) deactivates the
        # source collection in memory, but that state is not persisted. After
        # a server restart, the raw subnodes would leak back into the merged
        # corpus and the matcher would hit them alongside the condensate slab,
        # double-firing the session frame. Re-apply the deactivation by
        # scanning every slab's canonical_text for the "Neighborhood: <id>"
        # prefix that promote_collection bakes in (see routes.py ~line 1175).
        self._deactivate_absorbed_collections()
        return results

    def _deactivate_absorbed_collections(self) -> None:
        """Auto-deactivate any collection whose content has been slabified.

        The slabify promotion (POST /collections/{id}/promote) builds a
        condensate slab whose canonical_text starts with
        ``"Neighborhood: <collection_id>"``. That string is the authoritative
        marker that ``<collection_id>`` has been absorbed: the raw anchors
        should stay on disk for future re-expansion, but they must NOT be in
        the merged corpus used by the matcher, or both the slab and the
        underlying subnodes will fire on the same message.

        This is called from load_all() so the deactivation survives restart
        without adding any new persistence state — the slab itself IS the
        marker. Re-activation via registry.activate() still works for
        transient inspection between restarts.
        """
        prefix = "Neighborhood: "
        absorbed: set[str] = set()
        for owner_cid, store in self.collections.items():
            for slab in store.slabs.values():
                text = (slab.canonical_text or "").strip()
                if not text.startswith(prefix):
                    continue
                # "Neighborhood: foo\nThemes: ..." → "foo"
                rest = text[len(prefix):]
                absorbed_cid = rest.split("\n", 1)[0].strip()
                if not absorbed_cid:
                    continue
                # Don't deactivate a collection that absorbed itself (no-op
                # slabify where source == target), and skip unknown ids.
                if absorbed_cid == owner_cid:
                    continue
                if absorbed_cid in self.collections:
                    absorbed.add(absorbed_cid)
        if absorbed:
            self.active_ids -= absorbed
            self._merged_dirty = True
            print(
                f"[CORPUS] Auto-deactivated absorbed collections: "
                f"{sorted(absorbed)} (each has a condensate slab in another "
                f"active collection)"
            )

    def activate(self, collection_id: str) -> bool:
        """Add a collection to the active set."""
        if collection_id not in self.collections:
            return False
        self.active_ids.add(collection_id)
        self._merged_dirty = True
        return True

    def deactivate(self, collection_id: str) -> bool:
        """Remove a collection from the active set."""
        self.active_ids.discard(collection_id)
        self._merged_dirty = True
        return True

    def create_collection(self, collection_id: str) -> CorpusStore:
        """Create a new empty collection on disk."""
        root = CORPORA_ROOT / collection_id
        objects = root / "objects"
        objects.mkdir(parents=True, exist_ok=True)

        # Seed empty YAML files so load() doesn't fail
        for fname in ["anchors.yaml", "slabs.yaml", "key_bundles.yaml", "edges.yaml", "gates.yaml"]:
            fpath = objects / fname
            if not fpath.exists():
                _save_yaml(fpath, [])

        # Also create state dir
        (root / "state").mkdir(parents=True, exist_ok=True)

        store = CorpusStore(root=root, collection_id=collection_id)
        store.load()
        self.collections[collection_id] = store
        self.active_ids.add(collection_id)
        self._merged_dirty = True
        return store

    def delete_collection(self, collection_id: str) -> bool:
        """Remove a collection from the registry (does NOT delete files)."""
        if collection_id == "default":
            return False  # Can't delete default
        self.collections.pop(collection_id, None)
        self.active_ids.discard(collection_id)
        self._merged_dirty = True
        return True

    @property
    def merged(self) -> CorpusStore:
        """Return a merged view of all active collections.

        The merged store combines anchors/slabs/bundles/edges/gates from
        all active collections. If IDs collide, the first-loaded collection
        wins (default takes priority).

        This is the view that pipeline, matcher, and context packer use.
        """
        if not self._merged_dirty and self._merged is not None:
            return self._merged

        merged = CorpusStore(collection_id="__merged__")
        # Don't set a real root — this is a virtual store

        for cid in sorted(self.active_ids):
            store = self.collections.get(cid)
            if not store:
                continue
            # Merge, first-write-wins (no overwrite on collision)
            for aid, a in store.anchors.items():
                if aid not in merged.anchors:
                    merged.anchors[aid] = a
            for sid, s in store.slabs.items():
                if sid not in merged.slabs:
                    merged.slabs[sid] = s
            for bid, b in store.bundles.items():
                if bid not in merged.bundles:
                    merged.bundles[bid] = b
            for eid, e in store.edges.items():
                if eid not in merged.edges:
                    merged.edges[eid] = e
            for gid, g in store.gates.items():
                if gid not in merged.gates:
                    merged.gates[gid] = g

        self._merged = merged
        self._merged_dirty = False
        return merged

    def get_store(self, collection_id: str) -> Optional[CorpusStore]:
        """Get a specific collection's store."""
        return self.collections.get(collection_id)

    def slab_collection_map(self) -> dict[str, str]:
        """Return ``slab_id -> collection_id`` for every slab in every
        active collection.

        Used when the system prompt needs provenance annotations (e.g.
        the CORPUS CATALOG includes which collection each entry belongs
        to, so the model can disambiguate slabs with similar titles
        across mining passes). First-loaded collection wins on id
        collision, matching ``merged`` semantics.
        """
        out: dict[str, str] = {}
        for cid in sorted(self.active_ids):
            store = self.collections.get(cid)
            if not store:
                continue
            for sid in store.slabs:
                out.setdefault(sid, cid)
        return out

    def list_collections(self) -> list[dict]:
        """List all collections with metadata."""
        result = []
        for cid, store in sorted(self.collections.items()):
            result.append({
                "id": cid,
                "active": cid in self.active_ids,
                "anchors": len(store.anchors),
                "slabs": len(store.slabs),
                "bundles": len(store.bundles),
                "edges": len(store.edges),
                "gates": len(store.gates),
                "path": str(store.root),
            })
        return result
