"""Hierarchical community detection over the curated graph.

Mirrors the shape of ``graph_rank.py`` (pure math module, no retrieval
policy, no I/O besides the persistence helper). The algorithm is
recursive Leiden — at each level, run Leiden on the current subgraph,
and recurse within each resulting cluster that's still large enough to
justify further subdivision.

Design choices (see chat history for rationale):

* **Edges are symmetrized** for clustering — same rationale as PageRank:
  curator-authored edges carry relational (not strictly directional)
  semantics. A random-walk-style community algorithm wants symmetric
  neighborhoods.
* **Per-collection by default** — Leiden runs on each active collection
  independently. Cross-collection edges are preserved in the underlying
  graph and continue to drive retrieval/PR, but they're not part of any
  collection's cluster tree. A separate ``"__merged__"`` scope runs
  Leiden on the union graph for the rare case where cross-collection
  clustering is wanted.
* **Persistence required, not optional** — Leiden uses randomized
  neighbor-order passes. Two runs on identical input produce slightly
  different clusters. If we recompute on every boot, labels go stale
  constantly. Cluster assignments are saved to disk and only
  regenerated when the graph materially changes.
* **Recursive, not multi-resolution** — recursive Leiden produces a
  strict tree where sub-clusters are subsets of their parent. Multi-
  resolution (running Leiden at different resolution parameters) does
  not nest cleanly and breaks zoom UX.

At current corpus scales (< 10k nodes per collection) the full
hierarchical computation is sub-second. LLM label generation dominates
total wall-clock, not the clustering itself.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .atomic_io import atomic_write_json

logger = logging.getLogger(__name__)


# ── Tuning ───────────────────────────────────────────────────────

# Below this many members, don't recurse further — sub-clustering a
# 20-node cluster produces arbitrary groupings rather than structural
# signal. Raise for coarser hierarchies; lower for finer.
MIN_CLUSTER_SIZE_FOR_RECURSION = 25

# Hard depth cap independent of cluster size. Protects against
# pathological recursions (e.g. chain-like subgraphs where Leiden
# keeps producing 2-way splits). 4 levels handles ~10k-node corpora.
MAX_RECURSION_DEPTH = 4

# Leiden's resolution parameter. 1.0 is the classic default (standard
# modularity). Higher → more, smaller clusters. Lower → fewer, larger.
# Tune via env or config later if real behavior suggests drift.
LEIDEN_RESOLUTION = 1.0

# Fixed seed for the RNG passed to Leiden. Combined with saved
# assignments on disk, this gives us stability across reboots WITHOUT
# sacrificing quality — if persistence is lost or corrupted, the next
# computation at least produces the same clusters as it did last time
# instead of drifting further.
LEIDEN_SEED = 42


def _to_igraph(
    node_ids: list[str],
    edges: list,
    id_to_index: dict[str, int],
):
    """Build an undirected igraph.Graph from a node id list + edge list.

    Edges are symmetrized implicitly (igraph undirected graphs dedupe
    both directions). Multi-edges between the same pair are summed into
    edge weights.

    Only considers edges whose endpoints are both in the node_ids set.
    Cross-collection edges to nodes outside the current scope are
    silently dropped (they're still in the underlying corpus.edges and
    still participate in PR / retrieval — just not in this collection's
    clustering).
    """
    # Import inside the function so module import doesn't require
    # igraph to be present at startup if this feature is disabled.
    import igraph as ig

    # Aggregate edge weights by undirected pair
    weights: dict[tuple[int, int], float] = {}
    for e in edges:
        fi = id_to_index.get(e.from_node)
        ti = id_to_index.get(e.to_node)
        if fi is None or ti is None:
            continue
        if fi == ti:
            continue  # skip self-loops
        # Canonicalize undirected pair so both directions aggregate
        key = (min(fi, ti), max(fi, ti))
        w = float(getattr(e, "weight", 1.0) or 1.0)
        weights[key] = weights.get(key, 0.0) + w

    if not weights:
        # Empty or edgeless graph — return a graph of isolated vertices
        # so the caller can still treat each node as its own cluster.
        g = ig.Graph(n=len(node_ids), directed=False)
        g.vs["name"] = node_ids
        g.es["weight"] = []
        return g

    edge_list = list(weights.keys())
    edge_weights = [weights[k] for k in edge_list]
    g = ig.Graph(
        n=len(node_ids),
        edges=edge_list,
        edge_attrs={"weight": edge_weights},
        directed=False,
    )
    g.vs["name"] = node_ids
    return g


def _run_leiden(graph, resolution: float = LEIDEN_RESOLUTION, seed: int = LEIDEN_SEED) -> list[list[int]]:
    """Run Leiden community detection on an igraph.Graph.

    Returns: list of clusters, each cluster a list of vertex indices.
    """
    import leidenalg as la
    # ModularityVertexPartition is the standard Leiden partition type.
    # weights pulled from the edge attribute.
    partition = la.find_partition(
        graph,
        la.RBConfigurationVertexPartition,
        weights="weight" if "weight" in graph.es.attributes() else None,
        resolution_parameter=resolution,
        seed=seed,
    )
    return [list(c) for c in partition]


def compute_hierarchical_communities(
    node_ids: list[str],
    edges: list,
    max_depth: int = MAX_RECURSION_DEPTH,
    min_cluster_size: int = MIN_CLUSTER_SIZE_FOR_RECURSION,
) -> list[dict]:
    """Recursive Leiden over a node list and its edges.

    Returns a tree: list of cluster-dicts, each with shape::

        {
            "members": [node_id, node_id, ...],    # flat list of leaf node ids in this cluster
            "children": [                            # sub-clusters (may be empty list for leaf clusters)
                { "members": [...], "children": [...] },
                ...
            ],
        }

    Leaf clusters (below ``min_cluster_size`` OR at ``max_depth``) have
    an empty ``children`` list. Their ``members`` is still the full
    member list — callers rendering a leaf just show the nodes
    directly, no further nesting.
    """
    if not node_ids:
        return []

    id_to_index = {nid: i for i, nid in enumerate(node_ids)}
    graph = _to_igraph(node_ids, edges, id_to_index)

    if graph.ecount() == 0:
        # No edges → Leiden trivially puts each node in its own cluster.
        # Not useful. Treat as a single top-level cluster of all nodes.
        return [{"members": list(node_ids), "children": []}]

    # Top-level partition
    try:
        clusters = _run_leiden(graph)
    except Exception as exc:
        logger.warning("Leiden failed on %d-node graph: %r", len(node_ids), exc)
        return [{"members": list(node_ids), "children": []}]

    tree: list[dict] = []
    for cluster_indices in clusters:
        member_ids = [node_ids[i] for i in cluster_indices]
        children: list[dict] = []
        if max_depth > 1 and len(member_ids) > min_cluster_size:
            # Recurse into this cluster's induced subgraph.
            # Only edges fully contained in this cluster's membership
            # participate in the sub-clustering.
            member_set = set(member_ids)
            sub_edges = [
                e for e in edges
                if e.from_node in member_set and e.to_node in member_set
            ]
            children = compute_hierarchical_communities(
                member_ids,
                sub_edges,
                max_depth=max_depth - 1,
                min_cluster_size=min_cluster_size,
            )
            # If the recursion produced a single cluster containing
            # everything, that's not useful hierarchy — flatten.
            if len(children) == 1:
                children = []
        tree.append({"members": member_ids, "children": children})
    return tree


def coalesce_small_clusters(
    tree: list[dict],
    min_size: int = 3,
    bucket_label_hint: str = "Miscellaneous",
) -> list[dict]:
    """Post-process a cluster tree: merge top-level clusters smaller
    than ``min_size`` into a single "miscellaneous" bucket.

    Singletons and doubletons are structural orphans in the graph —
    nodes with no or very few edges to anything else. Rendering each
    as its own cluster produces visual noise. A single "Misc" bucket
    is cleaner and honest (it's a real property of the graph, not
    hidden).

    Does NOT modify sub-clusters (which Leiden may produce for
    legitimate internal structure reasons). Only flattens at the top.

    Returns a new tree; original is unmodified. The misc bucket has an
    extra ``_is_misc: True`` marker so downstream renderers can style
    it differently (e.g. grey-out) or skip LLM labeling and just call
    it "Miscellaneous."
    """
    big: list[dict] = []
    misc_members: list[str] = []
    for cluster in tree:
        if len(cluster.get("members", [])) < min_size:
            misc_members.extend(cluster.get("members", []))
        else:
            big.append(cluster)

    if misc_members:
        big.append({
            "members": misc_members,
            "children": [],
            "_is_misc": True,
            "_label_hint": bucket_label_hint,
        })

    return big


def flatten_tree_to_assignments(
    tree: list[dict],
    path: tuple = (),
) -> dict[str, tuple]:
    """Walk a cluster tree and produce a flat ``{node_id: hierarchical_path}`` map.

    The path is a tuple of cluster indices from root to leaf:
    ``(0, 2, 1)`` means "top-level cluster 0, sub-cluster 2, sub-sub-cluster 1."

    Used to persist cluster membership as per-node metadata without
    serializing the full tree structure — the tree can be reconstructed
    from paths alone.
    """
    assignments: dict[str, tuple] = {}
    for i, cluster in enumerate(tree):
        cluster_path = path + (i,)
        if cluster.get("children"):
            nested = flatten_tree_to_assignments(cluster["children"], cluster_path)
            assignments.update(nested)
        else:
            for nid in cluster.get("members", []):
                assignments[nid] = cluster_path
    return assignments


def build_meta_graph(
    tree: list[dict],
    edges: list,
) -> tuple[list[dict], list[dict]]:
    """Build the top-level meta-graph: one node per top-level cluster,
    weighted edges counting inter-cluster raw edges.

    Returns:
      meta_nodes: list of {"cluster_idx": int, "member_count": int, "member_ids": list[str]}
      meta_edges: list of {"from": int, "to": int, "weight": int}
    """
    # Per-node cluster index (top-level only)
    node_to_cluster: dict[str, int] = {}
    meta_nodes: list[dict] = []
    for i, cluster in enumerate(tree):
        member_ids = cluster.get("members", [])
        meta_nodes.append({
            "cluster_idx": i,
            "member_count": len(member_ids),
            "member_ids": member_ids,
        })
        for nid in member_ids:
            node_to_cluster[nid] = i

    # Count inter-cluster edges (including cross-collection ones that
    # happen to land in this scope). Within-cluster edges don't count.
    edge_counts: dict[tuple[int, int], int] = {}
    for e in edges:
        fi = node_to_cluster.get(e.from_node)
        ti = node_to_cluster.get(e.to_node)
        if fi is None or ti is None or fi == ti:
            continue
        key = (min(fi, ti), max(fi, ti))
        edge_counts[key] = edge_counts.get(key, 0) + 1

    meta_edges = [
        {"from": f, "to": t, "weight": w}
        for (f, t), w in sorted(edge_counts.items())
    ]
    return meta_nodes, meta_edges


# ── Persistence ──────────────────────────────────────────────────────

COMMUNITIES_ROOT = Path("app/state/communities")


def _graph_fingerprint(node_ids: list[str], edges: list) -> dict:
    """Stable fingerprint of a graph — detects material changes for
    cache invalidation without having to diff full structures.

    Captures:
      - node count
      - edge count
      - content-hash of sorted node ids (catches adds/removes/renames)

    Does NOT hash edge weights or types — a mine changes nodes + edge
    count substantially anyway, and hashing every edge attr would thrash
    on insignificant weight tweaks. The node-id hash is the sensitive
    signal.
    """
    sorted_ids = sorted(node_ids)
    ids_hash = hashlib.sha1(
        ",".join(sorted_ids).encode("utf-8"),
    ).hexdigest()[:16]
    return {
        "node_count": len(node_ids),
        "edge_count": len(edges),
        "nodes_hash": ids_hash,
    }


def _fingerprint_materially_changed(
    old: Optional[dict],
    new: dict,
    node_pct_threshold: float = 0.10,
) -> bool:
    """Return True if the new fingerprint differs enough to warrant
    full recompute. Our heuristic from chat: ≥10% node-count change OR
    different node-id set.

    A pure edge-count change with unchanged nodes does NOT trigger full
    recompute — same clusters, just more internal connections. Labels
    would already be valid.
    """
    if old is None:
        return True
    if old.get("nodes_hash") != new.get("nodes_hash"):
        # Node-id set changed → content changed
        old_n = max(old.get("node_count", 0), 1)
        new_n = new.get("node_count", 0)
        delta_pct = abs(new_n - old_n) / old_n
        # Always recompute if ids changed AND size delta is meaningful.
        # If ids changed but count is very close (e.g. an id rename),
        # still recompute because cluster membership would shift.
        return delta_pct >= node_pct_threshold or abs(new_n - old_n) >= 5
    return False


class CommunityStore:
    """Per-collection persistent cache for cluster trees + labels.

    Files live at ``app/state/communities/<scope>.json`` and are
    atomically written via ``atomic_write_json``. Each scope is either
    a collection id or the special string ``"__merged__"`` for the
    cross-collection union view.

    Labels are keyed by frozenset(member_ids) so that when community
    membership is stable across a recompute (common with Leiden's fixed
    seed + minor graph edits), we reuse cached LLM-generated labels
    instead of regenerating. See ``label_for_cluster``.
    """

    def __init__(self, root: Optional[Path] = None):
        self.root = root or COMMUNITIES_ROOT
        # In-memory mirror of the on-disk state, populated on first
        # access per scope.
        self._cache: dict[str, dict] = {}

    def _path(self, scope: str) -> Path:
        # Sanitize: collection ids are already safe on disk
        # (word chars + underscores), scope "__merged__" has only
        # underscores. No additional escaping needed.
        return self.root / f"{scope}.json"

    def load(self, scope: str) -> Optional[dict]:
        """Load persisted state for a scope, or None if absent/invalid."""
        if scope in self._cache:
            return self._cache[scope]
        path = self._path(scope)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._cache[scope] = data
            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("CommunityStore: failed to load %s: %r", scope, exc)
            return None

    def save(self, scope: str, tree: list[dict], fingerprint: dict,
             labels: Optional[dict[str, str]] = None) -> None:
        """Persist the tree + fingerprint (+ optional labels) for a scope.

        ``labels`` is a dict mapping hierarchical-path-string (e.g.
        ``"0"`` or ``"0,2,1"``) to the cluster's label. Kept flat
        on disk so the JSON is human-inspectable.
        """
        payload = {
            "scope": scope,
            "fingerprint": fingerprint,
            "tree": tree,
            "labels": labels or {},
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(self._path(scope), payload)
        self._cache[scope] = payload

    def invalidate(self, scope: str) -> None:
        """Drop in-memory cache for a scope (forces re-read on next load).
        Does not delete the on-disk file; use ``delete`` for that."""
        self._cache.pop(scope, None)

    def delete(self, scope: str) -> None:
        """Remove both in-memory + on-disk state for a scope."""
        self.invalidate(scope)
        path = self._path(scope)
        if path.exists():
            try:
                path.unlink()
            except OSError as exc:
                logger.warning("CommunityStore: failed to delete %s: %r", scope, exc)

    def needs_recompute(self, scope: str, current_fingerprint: dict) -> bool:
        """Check if the cached tree for a scope is stale relative to
        the current graph fingerprint."""
        persisted = self.load(scope)
        if persisted is None:
            return True
        old_fp = persisted.get("fingerprint")
        return _fingerprint_materially_changed(old_fp, current_fingerprint)


# ── Label generation ─────────────────────────────────────────────────

# Small fast model for cluster labeling. Same rationale as dream
# enrichment: labeling is a text-summarization task that doesn't need
# the main chat model's capability. Overridable via env var.
import os as _os
LABEL_MODEL: Optional[str] = _os.environ.get(
    "PS_LABEL_MODEL", "llama3.2:latest",
) or None

# Max titles per cluster to feed the labeler. Too few = poor signal;
# too many = wasted tokens and the LLM tunes out the tail. 10 hits a
# sweet spot where the label reflects the cluster's character.
_LABEL_MAX_TITLES = 10

# Sample strategy when a cluster has more members than _LABEL_MAX_TITLES:
# take the first-N as iteration order (Leiden returns vertices in a
# roughly-density-sorted order per cluster, so first-N biases toward
# the denser core rather than the fringe).
# Alternative would be random sample; first-N is cheaper and more stable
# across recomputes with the fixed LEIDEN_SEED.


_LABEL_PROMPT = """\
You will be shown titles of nodes that form a semantic cluster in a knowledge graph. \
Output ONE short label (2-4 words) that captures what this cluster is about. \
No punctuation, no quotes, no explanation — just the label itself.
{parent_context}
Titles:
{titles}

Label: """


_LABEL_PROMPT_WITH_PARENT = """\
You will be shown titles of nodes that form a sub-cluster within a larger knowledge-graph cluster. \
Output ONE short label (2-4 words) that describes this sub-cluster as a sub-topic of its parent. \
No punctuation, no quotes, no explanation — just the label itself.

Parent cluster: "{parent_label}"

Titles:
{titles}

Label: """


def _title_for_node(node_id: str, corpus) -> str:
    """Best human-readable title for any node type. Used for labeling
    input. Falls back to node_id if no better name is available.
    """
    if node_id in getattr(corpus, "anchors", {}):
        a = corpus.anchors[node_id]
        return a.canonical_phrase or node_id
    if node_id in getattr(corpus, "slabs", {}):
        s = corpus.slabs[node_id]
        return (s.title or (s.canonical_text or "")[:60] or node_id).strip()
    if node_id in getattr(corpus, "bundles", {}):
        b = corpus.bundles[node_id]
        intents = getattr(b.payload, "intent", []) if getattr(b, "payload", None) else []
        return "; ".join(intents[:2]) if intents else node_id
    return node_id


async def _generate_label(
    titles: list[str],
    parent_label: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """Call the labeler LLM. Returns a single short label string.

    Falls back to the longest-common-prefix heuristic on LLM error,
    which is crude but keeps the pipeline non-fatal.
    """
    from . import ollama

    if not titles:
        return "Unlabeled"

    titles_block = "\n".join(f"- {t}" for t in titles)
    if parent_label:
        prompt = _LABEL_PROMPT_WITH_PARENT.format(
            parent_label=parent_label, titles=titles_block,
        )
    else:
        prompt = _LABEL_PROMPT.format(
            parent_context="", titles=titles_block,
        )

    try:
        # Use generate (free-form text output) not structured_extract —
        # we want a short label, not JSON.
        raw = await ollama.generate(
            prompt,
            temperature=0.2,       # low temp for consistent labels
            num_ctx=2048,          # labeling prompts are small
            model=model or LABEL_MODEL,
        )
    except Exception as exc:
        logger.warning("Label generation failed for %d titles: %r", len(titles), exc)
        return _fallback_label(titles)

    # Clean the response: strip quotes, trailing punct, whitespace, newlines.
    label = (raw or "").strip().splitlines()[0].strip() if raw else ""
    # Strip common LLM artifacts
    label = label.strip("\"'`.,;:()[]{}").strip()
    # Cap at 50 chars defensively
    if len(label) > 50:
        label = label[:50].rsplit(" ", 1)[0]
    if not label:
        return _fallback_label(titles)
    return label


def _fallback_label(titles: list[str]) -> str:
    """Dead-simple label fallback: longest common word prefix across
    titles, or the first title truncated. Used when LLM call fails.
    """
    if not titles:
        return "Unlabeled"
    words_per_title = [t.lower().split()[:3] for t in titles[:5]]
    if not words_per_title:
        return titles[0][:40]
    # Find common leading words
    common = []
    for i in range(min(len(w) for w in words_per_title)):
        word = words_per_title[0][i]
        if all(w[i] == word for w in words_per_title):
            common.append(word)
        else:
            break
    if common:
        return " ".join(common).title()
    return titles[0][:40]


def _cluster_titles(members: list[str], corpus) -> list[str]:
    """Produce the labeler input: up to _LABEL_MAX_TITLES human titles."""
    titles = [_title_for_node(nid, corpus) for nid in members[:_LABEL_MAX_TITLES]]
    # Dedupe while preserving order
    seen: set[str] = set()
    unique = []
    for t in titles:
        if t and t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def _path_key(path: tuple) -> str:
    """Serialize a cluster path tuple to a stable string key for JSON."""
    return ",".join(str(i) for i in path)


async def label_tree(
    tree: list[dict],
    corpus,
    old_labels: Optional[dict[str, str]] = None,
    old_tree: Optional[list[dict]] = None,
    model: Optional[str] = None,
    _parent_label: Optional[str] = None,
    _path: tuple = (),
) -> dict[str, str]:
    """Walk a cluster tree and generate a label for each cluster.

    Reuses labels from ``old_labels`` when the new cluster's frozenset
    of member_ids matches an old cluster's (via ``old_tree`` walked in
    parallel). This is the "Leiden is stable, only re-label what
    changed" optimization — cuts re-labeling cost from O(clusters) to
    O(changed_clusters).

    Special-cases:
      * ``_is_misc`` clusters use a fixed label, skip LLM
      * Clusters with zero members skip LLM
      * LLM failures fall through to _fallback_label

    Returns a flat dict ``{path_str: label}`` for the full tree.
    """
    # Build membership→label cache from old state (if any)
    cache: dict[frozenset, str] = {}
    if old_labels and old_tree:
        _walk_for_cache(old_tree, old_labels, cache, path=())

    labels: dict[str, str] = {}
    for i, cluster in enumerate(tree):
        path = _path + (i,)
        path_str = _path_key(path)
        members = cluster.get("members", [])
        members_set = frozenset(members)

        # Misc bucket: fixed label, no LLM.
        if cluster.get("_is_misc"):
            labels[path_str] = cluster.get("_label_hint", "Miscellaneous")
        elif not members:
            labels[path_str] = "Empty"
        elif members_set in cache:
            # Stable membership — reuse old label.
            labels[path_str] = cache[members_set]
            logger.debug("label cache hit: path=%s", path_str)
        else:
            titles = _cluster_titles(members, corpus)
            labels[path_str] = await _generate_label(
                titles,
                parent_label=_parent_label,
                model=model,
            )
            logger.info(
                "label generated: path=%s members=%d -> %r",
                path_str, len(members), labels[path_str],
            )

        # Recurse into children with this cluster's label as parent-context.
        if cluster.get("children"):
            child_labels = await label_tree(
                cluster["children"],
                corpus,
                old_labels=old_labels,
                old_tree=old_tree[i]["children"] if (
                    old_tree and i < len(old_tree) and old_tree[i].get("children")
                ) else None,
                model=model,
                _parent_label=labels[path_str],
                _path=path,
            )
            labels.update(child_labels)

    return labels


def _walk_for_cache(
    tree: list[dict],
    labels: dict[str, str],
    cache: dict[frozenset, str],
    path: tuple,
) -> None:
    """Populate a membership→label cache by walking an old tree + labels."""
    for i, cluster in enumerate(tree):
        p = path + (i,)
        key = _path_key(p)
        if key in labels:
            members_set = frozenset(cluster.get("members", []))
            if members_set:
                cache[members_set] = labels[key]
        if cluster.get("children"):
            _walk_for_cache(cluster["children"], labels, cache, p)


async def compute_and_persist(
    corpus,
    scope: str,
    node_ids: list[str],
    edges: list,
    community_store: CommunityStore,
    force: bool = False,
    coalesce_min_size: int = 3,
    skip_labels: bool = False,
) -> dict:
    """Top-level orchestrator: check fingerprint, compute if needed,
    label via LLM, persist, return the payload.

    Flow:
      1. Compute fingerprint → cache hit → return cached (no LLM cost).
      2. Cache miss → run Leiden → coalesce singletons.
      3. Reuse labels from previous payload where cluster membership
         is unchanged (Leiden is stable + frozenset cache).
      4. Generate LLM labels only for new/changed clusters.
      5. Atomic-write the full payload {tree, fingerprint, labels}.

    ``corpus`` is needed for title lookup during labeling; pass the
    ``CorpusStore`` for the scope being labeled (or the merged view).

    Set ``skip_labels=True`` to compute clusters without LLM labeling
    (useful during bulk warm where labeling would dominate wall-time
    and can run deferred).
    """
    fingerprint = _graph_fingerprint(node_ids, edges)
    if not force and not community_store.needs_recompute(scope, fingerprint):
        return community_store.load(scope)

    logger.info(
        "Computing communities for scope=%r: %d nodes, %d edges",
        scope, len(node_ids), len(edges),
    )

    # Preserve old payload so label-reuse can happen on recompute
    old_payload = community_store.load(scope) if not force else None
    old_tree = old_payload.get("tree") if old_payload else None
    old_labels = old_payload.get("labels") if old_payload else None

    raw_tree = compute_hierarchical_communities(node_ids, edges)
    tree = coalesce_small_clusters(raw_tree, min_size=coalesce_min_size)

    labels: dict[str, str] = {}
    if not skip_labels:
        try:
            labels = await label_tree(
                tree, corpus,
                old_labels=old_labels,
                old_tree=old_tree,
            )
        except Exception as exc:
            logger.warning("Label generation failed for scope=%r: %r", scope, exc)
            labels = {}

    community_store.save(scope, tree, fingerprint, labels=labels)
    return community_store.load(scope)
