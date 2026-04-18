"""Relocate mined edges that were misrouted to the `default` collection.

A bug in LifecycleService._try_commit_accepted_edge used chat.collection_id
as the routing signal, which is usually "default" regardless of where the
mining pipeline was targeting. Result: all SEQUENCE/LINKS edges from
narrative mining landed in default's edges.yaml while the endpoint nodes
lived in the target collection. This script moves them home.

For each edge in default.edges:
  - If BOTH endpoints are in a single non-default collection → move to that
    collection and remove from default.
  - Otherwise → leave alone (it legitimately spans collections or touches
    default content).

Usage:
  python scripts/migrate_misrouted_edges.py         # dry run
  python scripts/migrate_misrouted_edges.py --apply # actually write
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.services.corpus import CorpusRegistry

APPLY = "--apply" in sys.argv

reg = CorpusRegistry()
reg.load_all()

default_store = reg.get_store("default")
if default_store is None:
    print("No default collection — nothing to migrate.")
    sys.exit(0)

# Build a node -> collection map excluding default
home_by_node: dict[str, str] = {}
for cid, store in reg.collections.items():
    if cid == "default":
        continue
    for nid in list(store.anchors) + list(store.slabs) + list(store.bundles):
        home_by_node.setdefault(nid, cid)

# Scan default's edges
moves: dict[str, list] = {}  # target_collection -> list[(edge_id, edge)]
stays = 0
for eid, edge in list(default_store.edges.items()):
    from_home = home_by_node.get(edge.from_node)
    to_home = home_by_node.get(edge.to_node)
    if from_home and to_home and from_home == to_home:
        moves.setdefault(from_home, []).append((eid, edge))
    else:
        stays += 1

total_moves = sum(len(v) for v in moves.values())
print(f"default has {len(default_store.edges)} edges")
print(f"  staying (endpoints not both in same non-default collection): {stays}")
print(f"  migrating to owning collections: {total_moves}")
for target, items in sorted(moves.items()):
    by_type: dict[str, int] = {}
    for eid, edge in items:
        by_type[edge.type.value] = by_type.get(edge.type.value, 0) + 1
    type_summary = ", ".join(f"{t}={n}" for t, n in sorted(by_type.items()))
    print(f"    -> {target}: {len(items)} edges ({type_summary})")

if not APPLY:
    print("\n[DRY RUN] Re-run with --apply to migrate.")
    sys.exit(0)

# Apply
moved_total = 0
for target, items in moves.items():
    target_store = reg.get_store(target)
    if target_store is None:
        print(f"  ! target collection {target} missing; skipping {len(items)} edges")
        continue
    for eid, edge in items:
        target_store.edges[eid] = edge
        del default_store.edges[eid]
        moved_total += 1
    target_store.save()

default_store.save()
print(f"\n[APPLIED] Moved {moved_total} edges out of default.")
