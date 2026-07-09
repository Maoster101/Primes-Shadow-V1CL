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

from ..models.schemas import Anchor, Slab, KeyBundle, Edge, Gate, PillarDefinition
from ..models.enums import OLIMode, SlabType, SlabLifecycleStatus
from .atomic_io import atomic_write_yaml

CORPUS_ROOT = Path("app/corpus")         # Legacy single-corpus path
CORPORA_ROOT = Path("app/corpora")       # Multi-collection root
ACTIVE_STATE_PATH = Path("app/state/registry/active_collections.json")
# JSON file: {"active": ["collection_id", ...]}. Persists user
# activate/deactivate toggles across server restarts. Without this,
# every restart resets the active set to "all collections active",
# wiping the user's curated active set. The file is rewritten on
# every activate/deactivate; missing or malformed → falls back to
# default-all-active behaviour.


def _load_yaml(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, list) else []


def _save_yaml(path: Path, data: list[dict]) -> None:
    """Atomic YAML save — crash-safe via temp-and-rename + fsync.

    See ``atomic_io.atomic_write_text`` for the full idiom. Historical
    note: this used to be a plain ``open("w")`` truncate, which left
    half-written YAMLs behind on crash. The save() method below writes
    five files sequentially, so a mid-save crash could split the corpus
    across file boundaries (new anchors, stale edges). The per-file
    atomicity shrinks that window from "tens of ms × 5" to "the tiny
    gap between 5 separate os.replace calls" — recoverable on next mine
    pass rather than data loss.
    """
    atomic_write_yaml(path, data)


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
        # Tier-0 navigational overlay. Pre-existing corpora have no
        # pillars.yaml; load() treats a missing file as an empty dict
        # so legacy collections remain readable without migration.
        self.pillars: dict[str, PillarDefinition] = {}

    def load(self) -> list[str]:
        """Load corpus from disk. Returns list of validation errors (empty = OK)."""
        raw_anchors = _load_yaml(self.objects / "anchors.yaml")
        raw_slabs = _load_yaml(self.objects / "slabs.yaml")
        raw_bundles = _load_yaml(self.objects / "key_bundles.yaml")
        raw_edges = _load_yaml(self.objects / "edges.yaml")
        raw_gates = _load_yaml(self.objects / "gates.yaml")
        # pillars.yaml is optional — pre-overlay corpora simply have
        # no file. _load_yaml returns [] for missing paths, which
        # collapses to an empty pillars dict.
        raw_pillars = _load_yaml(self.objects / "pillars.yaml")

        self.anchors = {a["id"]: Anchor(**a) for a in raw_anchors}
        self.slabs = {s["id"]: Slab(**s) for s in raw_slabs}
        self.bundles = {b["id"]: KeyBundle(**b) for b in raw_bundles}
        self.edges = {}
        for e in raw_edges:
            edge = Edge.model_validate(e)
            self.edges[edge.id] = edge
        self.gates = {g["id"]: Gate(**g) for g in raw_gates}
        self.pillars = {p["id"]: PillarDefinition(**p) for p in raw_pillars}

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
        # Only write pillars.yaml when the overlay is populated.
        # Empty-file persistence is fine but creates noise on disk
        # for pre-overlay corpora; the load path tolerates absence.
        if self.pillars:
            _save_yaml(
                self.objects / "pillars.yaml",
                [p.model_dump(mode="json") for p in self.pillars.values()],
            )

    def all_ids(self) -> set[str]:
        return (
            set(self.anchors) | set(self.slabs) | set(self.bundles)
            | set(self.gates) | set(self.pillars)
        )

    def subgraph(self, node_ids) -> "CorpusStore":
        """A store containing only ``node_ids`` and the edges among them.

        Used to make the seed-reachable slice the authoritative walk domain:
        PPR / synthesis run over this induced subgraph instead of the full
        corpus, so the O(n²) transition matrix is O(slice²). Shares node
        objects (not copies) — read-only, per-turn, throwaway.
        """
        ids = set(node_ids)
        sub = CorpusStore(collection_id="__walk__")
        for aid in ids & self.anchors.keys():
            sub.anchors[aid] = self.anchors[aid]
        for sid in ids & self.slabs.keys():
            sub.slabs[sid] = self.slabs[sid]
        for bid in ids & self.bundles.keys():
            sub.bundles[bid] = self.bundles[bid]
        for eid, e in self.edges.items():
            if e.from_node in ids and e.to_node in ids:
                sub.edges[eid] = e
        return sub

    def deduplicate(self) -> dict:
        """Collapse exact-duplicate slabs and anchors to a single survivor.

        Identity is exact content — slabs by ``canonical_text``, anchors
        by ``canonical_phrase``. The survivor of each group is the one
        with the lexicographically-smallest id (deterministic, stable).
        Every reference to a removed id — edge endpoints, anchor.invokes,
        depends_on lists, slab.links — is remapped to the survivor; an
        edge that becomes a self-loop or a duplicate (same type+from+to)
        after remapping is dropped.

        Pillars are deliberately NOT remapped — their members would point
        at removed slabs, so the caller must rebuild the overlay after.

        The caller owns validate() + save(). Returns a report.
        """
        remap: dict[str, str] = {}

        def _group(items: dict, key_fn) -> None:
            seen: dict[str, str] = {}
            for nid in sorted(items):
                k = key_fn(items[nid])
                if not k:
                    continue
                if k in seen:
                    remap[nid] = seen[k]  # nid is redundant -> seen[k] survives
                else:
                    seen[k] = nid

        _group(self.slabs, lambda s: (s.canonical_text or "").strip())
        _group(self.anchors, lambda a: (a.canonical_phrase or "").strip())

        slabs_removed = sum(1 for r in remap if r in self.slabs)
        anchors_removed = sum(1 for r in remap if r in self.anchors)
        if not remap:
            return {"remapped": 0, "slabs_removed": 0, "anchors_removed": 0,
                    "edges_dropped": 0}

        for rid in remap:
            self.slabs.pop(rid, None)
            self.anchors.pop(rid, None)

        def _r(x: str) -> str:
            return remap.get(x, x)

        def _uniq(seq) -> list:
            out: list = []
            seen: set = set()
            for x in seq:
                if x not in seen:
                    seen.add(x)
                    out.append(x)
            return out

        # Remap edge endpoints; drop self-loops and edges that collapse
        # onto an already-kept (type, from, to).
        new_edges: dict = {}
        sigs: set = set()
        edges_dropped = 0
        for eid, e in self.edges.items():
            fn, tn = _r(e.from_node), _r(e.to_node)
            if fn == tn:
                edges_dropped += 1
                continue
            sig = (e.type, fn, tn)
            if sig in sigs:
                edges_dropped += 1
                continue
            sigs.add(sig)
            e.from_node, e.to_node = fn, tn
            new_edges[eid] = e
        self.edges = new_edges

        # Remap reference lists carried on the surviving nodes.
        for a in self.anchors.values():
            a.invokes = _uniq(_r(x) for x in a.invokes)
            a.depends_on = _uniq(_r(x) for x in a.depends_on)
        for s in self.slabs.values():
            s.depends_on = _uniq(_r(x) for x in s.depends_on)
            if s.links:
                s.links.anchors = _uniq(_r(x) for x in s.links.anchors)
                s.links.bundles = _uniq(_r(x) for x in s.links.bundles)
        for b in self.bundles.values():
            b.supports = _uniq(_r(x) for x in b.supports)
            b.depends_on = _uniq(_r(x) for x in b.depends_on)

        return {
            "remapped": len(remap),
            "slabs_removed": slabs_removed,
            "anchors_removed": anchors_removed,
            "edges_dropped": edges_dropped,
            "edges_remaining": len(self.edges),
            "slabs_remaining": len(self.slabs),
            "anchors_remaining": len(self.anchors),
        }

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
            "pillar": set(self.pillars),
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
            *self.pillars.items(),
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
            *self.pillars.items(),
        ]:
            for dep in getattr(obj, "depends_on", []):
                if dep not in all_ids:
                    errors.append(f"{obj_id} depends_on missing target: {dep}")

        # Check 6: Supersedes exists
        for obj_id, obj in [
            *self.anchors.items(), *self.slabs.items(),
            *self.bundles.items(), *self.gates.items(),
            *self.pillars.items(),
        ]:
            meta = getattr(obj, "meta", None)
            if meta and meta.supersedes and meta.supersedes not in all_ids:
                errors.append(f"{obj_id} supersedes missing target: {meta.supersedes}")

        # Check 7: Pillar references resolve.
        # members must point at content nodes (slabs/bundles/anchors)
        # OR at other pillars (for tier-internal grouping). children
        # must point at PillarDefinition IDs. parent (if set) must
        # exist as a pillar. cross_edges.to_pillar must be a pillar.
        # Lifted edges may reference now-pruned underlying edges, so
        # underlying_edge_ids are NOT validated against current edges.
        for pid, pillar in self.pillars.items():
            for mid in pillar.members:
                if mid not in all_ids:
                    errors.append(f"Pillar {pid} member missing target: {mid}")
            for cid in pillar.children:
                if cid not in self.pillars:
                    errors.append(
                        f"Pillar {pid} child {cid} is not a PillarDefinition"
                    )
            if pillar.parent and pillar.parent not in self.pillars:
                errors.append(
                    f"Pillar {pid} parent {pillar.parent} is not a PillarDefinition"
                )
            for xe in pillar.cross_edges:
                if xe.to_pillar not in self.pillars:
                    errors.append(
                        f"Pillar {pid} cross_edge target {xe.to_pillar} is not a PillarDefinition"
                    )

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
        # Default: activate all. Then apply persisted user preferences
        # if they exist — only collections actually discovered count
        # (so a state file mentioning a deleted collection won't error).
        self.active_ids = set(self.collections.keys())
        persisted = self._load_active_state()
        if persisted is not None:
            self.active_ids = persisted & set(self.collections.keys())
        # Self-heal: a slabify (collection → condensate slab) deactivates the
        # source collection in memory, but that state is not persisted. After
        # a server restart, the raw subnodes would leak back into the merged
        # corpus and the matcher would hit them alongside the condensate slab,
        # double-firing the session frame. Re-apply the deactivation by
        # scanning every slab's canonical_text for the "Neighborhood: <id>"
        # prefix that promote_collection bakes in (see routes.py ~line 1175).
        self._deactivate_absorbed_collections()
        return results

    def _load_active_state(self) -> Optional[set[str]]:
        """Read the persisted active-collection set from disk.

        Returns None when the state file doesn't exist or fails to
        parse — caller falls back to default-all-active in that case.
        """
        if not ACTIVE_STATE_PATH.exists():
            return None
        try:
            import json
            data = json.loads(ACTIVE_STATE_PATH.read_text(encoding="utf-8"))
            active = data.get("active")
            if not isinstance(active, list):
                return None
            return {str(x) for x in active}
        except Exception as exc:
            print(
                f"[CORPUS] Failed to load active_collections state: {exc!r} "
                f"— falling back to default-all-active",
                flush=True,
            )
            return None

    def _save_active_state(self) -> None:
        """Persist the active-collection set to disk.

        Called after every activate/deactivate so the user's curated
        active set survives server restarts. Best-effort: failure to
        persist is logged but non-fatal — active_ids in memory is
        still authoritative for the running process.
        """
        try:
            import json
            ACTIVE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            ACTIVE_STATE_PATH.write_text(
                json.dumps({"active": sorted(self.active_ids)}, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            print(
                f"[CORPUS] Failed to save active_collections state: {exc!r}",
                flush=True,
            )

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
        """Add a collection to the active set. Persists across restart."""
        if collection_id not in self.collections:
            return False
        self.active_ids.add(collection_id)
        self._merged_dirty = True
        self._save_active_state()
        return True

    def deactivate(self, collection_id: str) -> bool:
        """Remove a collection from the active set. Persists across restart."""
        self.active_ids.discard(collection_id)
        self._merged_dirty = True
        self._save_active_state()
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
        self._save_active_state()
        return store

    def delete_collection(self, collection_id: str) -> bool:
        """Remove a collection from the registry (does NOT delete files)."""
        if collection_id == "default":
            return False  # Can't delete default
        self.collections.pop(collection_id, None)
        self.active_ids.discard(collection_id)
        self._merged_dirty = True
        self._save_active_state()
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
            for pid, p in store.pillars.items():
                if pid not in merged.pillars:
                    merged.pillars[pid] = p

        self._merged = merged
        self._merged_dirty = False
        return merged

    def get_store(self, collection_id: str) -> Optional[CorpusStore]:
        """Get a specific collection's store."""
        return self.collections.get(collection_id)

    def merged_subset(self, collection_ids) -> CorpusStore:
        """Merged view of ONLY the given collections (first-write-wins).

        Same merge semantics as ``merged`` but restricted to a subset —
        used to *bound the graph walk* to the collection(s) a query names.
        Built on demand (subsets are small and query-specific), not cached.
        """
        subset = CorpusStore(collection_id="__subset__")
        for cid in sorted(set(collection_ids)):
            store = self.collections.get(cid)
            if not store:
                continue
            for aid, a in store.anchors.items():
                subset.anchors.setdefault(aid, a)
            for sid, s in store.slabs.items():
                subset.slabs.setdefault(sid, s)
            for bid, b in store.bundles.items():
                subset.bundles.setdefault(bid, b)
            for eid, e in store.edges.items():
                subset.edges.setdefault(eid, e)
            for gid, g in store.gates.items():
                subset.gates.setdefault(gid, g)
            for pid, p in store.pillars.items():
                subset.pillars.setdefault(pid, p)
        return subset

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
                "pillars": len(store.pillars),
                "path": str(store.root),
            })
        return result
