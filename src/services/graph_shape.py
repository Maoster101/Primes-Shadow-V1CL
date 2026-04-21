"""Graph-shape classification for per-collection visualization routing.

Different corpora have structurally different topologies, and they want
different visualizations:

* **Spine** — narrative content. Heavy SEQUENCE edges, sparse LINKS.
  Renders as a timeline, scrollable beat-by-beat. Clustering is the
  wrong lens because a chain has no density variation for Leiden to
  exploit.
* **Highway** — technical content. Dense LINKS/SUPPORTS, moderate
  SEQUENCE. Renders as a cluster/meta-graph at top level with
  drill-down. Individual-node rendering is overwhelming above ~200
  nodes regardless of density.
* **Sparse** — collections with very few edges (bundles that didn't get
  mined into a connected graph, or tiny scratch corpora). Neither
  spine nor highway applies; flat list rendering is cleanest.
* **Flat** — small collections that are well-connected enough to
  render all nodes at once without clustering. The default for small
  mixed-topology corpora.

The classifier is deliberately coarse — four categories rather than a
continuous score — because the consumer (canvas renderer) makes a
discrete choice about which view to use. Exposing finer gradations
would just ask more complexity of the renderer without delivering
better UX.

Thresholds tuned against the current corpus mix (April 2026); revisit
if the mix shifts significantly.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

# Classification labels. Keeping them as a Literal instead of Enum so
# they serialize cleanly to JSON without extra handling.
Shape = Literal["spine", "highway", "sparse", "flat"]


# ── Thresholds ───────────────────────────────────────────────────

# Below this edges/node, the collection doesn't have enough structural
# connectivity to justify any graph visualization.
SPARSE_EDGES_PER_NODE = 0.5

# Above this SEQUENCE-edge ratio AND below this edges/node, content is
# predominantly sequential — render as timeline.
SPINE_SEQ_RATIO = 0.40
SPINE_MAX_EDGES_PER_NODE = 1.5

# Below this node count, even a highway-shaped collection renders fine
# as flat (all nodes visible at once). Above this, we need clustering
# to avoid visual overwhelm.
FLAT_MAX_NODES = 60

# Any collection with more LINKS/SUPPORTS than SEQUENCE and non-trivial
# connectivity is a highway.
HIGHWAY_MIN_EDGES_PER_NODE = 1.2
HIGHWAY_MAX_SEQ_RATIO = 0.35


@dataclass
class ShapeReport:
    """Full classification + the metrics that drove it.

    Renderers read ``shape``; diagnostics read the rest.
    """
    shape: Shape
    node_count: int
    edge_count: int
    edges_per_node: float
    seq_ratio: float
    edge_type_counts: dict
    # Free-form, human-readable reason for why this shape was chosen.
    # Useful in the dashboard for "why is this collection rendered
    # this way?" affordance.
    reason: str


def classify_shape(corpus) -> ShapeReport:
    """Inspect a corpus store and return a shape classification.

    ``corpus`` is a ``CorpusStore`` or any object with ``.anchors``,
    ``.slabs``, ``.bundles``, ``.edges`` dict-like attributes. Does not
    traverse the graph — purely summarizes edge-type counts and basic
    density. Cheap enough to call on every render.
    """
    n_nodes = (
        len(getattr(corpus, "anchors", {}))
        + len(getattr(corpus, "slabs", {}))
        + len(getattr(corpus, "bundles", {}))
    )
    edges = list(getattr(corpus, "edges", {}).values())
    n_edges = len(edges)

    type_counts: dict[str, int] = {}
    for e in edges:
        t = e.type.value if hasattr(e.type, "value") else str(e.type)
        type_counts[t] = type_counts.get(t, 0) + 1

    eps_node = n_edges / max(n_nodes, 1)
    seq_ratio = type_counts.get("SEQUENCE", 0) / max(n_edges, 1)

    # Decision order matters — earlier rules win.
    if n_nodes == 0:
        shape: Shape = "sparse"
        reason = "empty corpus"
    elif eps_node < SPARSE_EDGES_PER_NODE:
        shape = "sparse"
        reason = f"edges/node={eps_node:.2f} below sparse threshold {SPARSE_EDGES_PER_NODE}"
    elif seq_ratio >= SPINE_SEQ_RATIO and eps_node <= SPINE_MAX_EDGES_PER_NODE:
        shape = "spine"
        reason = (
            f"SEQUENCE-dominant: {seq_ratio:.0%} of edges are SEQUENCE "
            f"and edges/node={eps_node:.2f} <= {SPINE_MAX_EDGES_PER_NODE}"
        )
    elif n_nodes <= FLAT_MAX_NODES and seq_ratio < SPINE_SEQ_RATIO:
        shape = "flat"
        reason = (
            f"small ({n_nodes} <= {FLAT_MAX_NODES} nodes) and not "
            f"sequence-dominant; render all nodes inline"
        )
    elif eps_node >= HIGHWAY_MIN_EDGES_PER_NODE or seq_ratio <= HIGHWAY_MAX_SEQ_RATIO:
        shape = "highway"
        reason = (
            f"cross-reference-dominant: edges/node={eps_node:.2f}, "
            f"SEQUENCE ratio={seq_ratio:.0%}; cluster view with drill-down"
        )
    else:
        # Fallback: moderate connectivity, moderate SEQUENCE, small graph.
        shape = "flat"
        reason = (
            f"borderline metrics (edges/node={eps_node:.2f}, "
            f"SEQUENCE ratio={seq_ratio:.0%}); default to flat"
        )

    return ShapeReport(
        shape=shape,
        node_count=n_nodes,
        edge_count=n_edges,
        edges_per_node=round(eps_node, 3),
        seq_ratio=round(seq_ratio, 3),
        edge_type_counts=type_counts,
        reason=reason,
    )


def shape_report_to_dict(report: ShapeReport) -> dict:
    """Serializable form for API responses."""
    return asdict(report)
