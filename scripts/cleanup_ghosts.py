"""One-shot library-flow ghost cleanup.

Purges session tentative_registry entries whose IDs came through the
draft→library flow (prefixes: mined_, tentative_anchor_, tentative_slab_)
and are orphaned — not in corpus, not in library/tentative, not a live
DRAFT_UNAUTHORIZED packet.

Leaves tentative_concept_* and tentative_bundle_* alone (those are
session-scoped working memory, not library artifacts).

Also rejects any PROPOSED/ACCEPTED proposed_edges pointing at ghost ids.

Usage:
  python scripts/cleanup_ghosts.py           # dry run (scan + report)
  python scripts/cleanup_ghosts.py --apply   # actually write
"""
from pathlib import Path
import json
import sys

# Make `src` importable when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.services.corpus import CorpusRegistry
from src.services.session_store import SessionStore

APPLY = "--apply" in sys.argv
LIB_FLOW_PREFIXES = ("mined_", "tentative_anchor_", "tentative_slab_")

LIB_TENT = Path("app/library/tentative")
library_ids = {p.stem for p in LIB_TENT.glob("*.json")} if LIB_TENT.exists() else set()

reg = CorpusRegistry()
reg.load_all()
merged = reg.merged
corpus_ids = set(merged.anchors) | set(merged.slabs) | set(merged.bundles)
print(f"[SCAN] {len(library_ids)} library records, {len(corpus_ids)} corpus ids")

ss = SessionStore()
if not ss.root.exists():
    print("[SCAN] No sessions dir")
    sys.exit(0)

total_ghosts = 0
total_sessions = 0
total_edges = 0

for sess_dir in sorted(ss.root.iterdir()):
    if not sess_dir.is_dir():
        continue
    sid = sess_dir.name

    # Which drafts are still awaiting review in this session?
    live_draft_ids = set()
    drafts_dir = sess_dir / "drafts"
    if drafts_dir.exists():
        for f in drafts_dir.glob("*.json"):
            name = f.stem
            if name.endswith("_raw") or name.endswith(".enriched"):
                continue
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if data.get("status") == "DRAFT_UNAUTHORIZED":
                    live_draft_ids.add(name)
            except Exception:
                pass

    reg_data, edges = ss.load_registry(sid)
    frame = ss.load_frame(sid)
    if not reg_data and not frame:
        continue

    ghost_names = []
    for concept_name, info in (reg_data or {}).items():
        if not isinstance(info, dict):
            continue
        nid = info.get("id")
        if not nid or not nid.startswith(LIB_FLOW_PREFIXES):
            continue
        if nid in corpus_ids or nid in library_ids or nid in live_draft_ids:
            continue
        ghost_names.append((concept_name, nid))

    if not ghost_names:
        continue

    ghost_ids = {nid for _, nid in ghost_names}
    total_sessions += 1
    total_ghosts += len(ghost_names)
    print(f"[{sid}] {len(ghost_names)} ghost(s)")
    for name, nid in ghost_names:
        print(f"    - {nid}  ({name!r})")

    if not APPLY:
        continue

    # Purge registry
    for name, _ in ghost_names:
        reg_data.pop(name, None)

    # Purge frame state (all the dict-valued maps keyed by node_id)
    if frame is not None:
        frame.active_nodes = [n for n in frame.active_nodes if n not in ghost_ids]
        for nid in ghost_ids:
            for d in (
                frame.salience_now,
                frame.salience_smoothed,
                frame.structural_weight,
                frame.activation_sources,
                frame.active_anchors,
                frame.active_bundles,
                frame.active_slabs,
                frame.active_concepts,
                frame.corpus_hits,
                frame.corpus_last_hit,
                frame.truth_pressure,
            ):
                d.pop(nid, None)

    # Filter tentative_edges (session-scoped, dict form)
    remaining_edges = [
        e for e in (edges or [])
        if e.get("from") not in ghost_ids and e.get("to") not in ghost_ids
    ]
    ss.save_registry(sid, reg_data, remaining_edges)
    if frame is not None:
        ss.save_frame(sid, frame)

    # Reject proposed_edges that reference ghost ids
    edges_path = sess_dir / "proposed_edges.json"
    rejected = 0
    if edges_path.exists():
        try:
            pdata = json.loads(edges_path.read_text(encoding="utf-8"))
            for e in pdata.get("edges", []):
                if e.get("status") in ("PROPOSED", "ACCEPTED"):
                    if e.get("from_node") in ghost_ids or e.get("to_node") in ghost_ids:
                        e["status"] = "REJECTED"
                        rejected += 1
            edges_path.write_text(json.dumps(pdata, indent=2), encoding="utf-8")
        except Exception as exc:
            print(f"    [warn] proposed_edges rewrite failed: {exc}")

    total_edges += rejected
    if rejected:
        print(f"    -> rejected {rejected} proposed edge(s)")

print(
    f"\n[SUMMARY] {total_ghosts} ghost(s) + {total_edges} edge(s) rejected "
    f"across {total_sessions} session(s)"
)
if not APPLY:
    print("[DRY RUN] Re-run with --apply to clean up.")
else:
    print("[APPLIED]")
