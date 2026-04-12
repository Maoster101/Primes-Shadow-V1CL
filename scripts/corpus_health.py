"""Corpus health report — v1 (utilization + structural integrity).

Answers the question: "how effective is the corpus?"

Reads the on-disk corpus via CorpusRegistry and aggregates corpus_hits
across every session frame under app/workbench/sessions/*/frame.json.
Pure computation — no LLM, no embeddings, runs in well under a second.

Usage:
    python scripts/corpus_health.py

Sections produced:
    A.  Utilization
        - Fire distribution across corpus
        - Dead / cold / hot node partition
        - Gini coefficient (inequality of usage)
        - Per-collection breakdown
        - Full dead-node list with source file provenance
    A.5 Review Queue (drafts awaiting human sign-off)
        - Draft counts by DraftStatus
        - Pending list per session with packet_type, age, confidence
    B.  Structural integrity
        - Orphan nodes (no incoming or outgoing connections)
        - Unsupported bundles (no anchor invokes them)
        - Dangling references (depends_on / links / invokes → missing)
        - Connected components (graph islands)
        - Average degree by node type
    B.5 Committed-orphan audit
        - Former-draft nodes that passed DRAFT_UNAUTHORIZED→COMMITTED
          but never had outgoing structural wiring added.
        - The "Terra Preta pattern": commit != integration.

The dead-node list is what makes this useful. Counts alone tell you the
shape of the problem; seeing which specific nodes are unused tells you
whether governance is working as intended (protecting rare doctrine) or
failing (mining pipeline too permissive).
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict, deque
from datetime import date
from pathlib import Path
from typing import Any

# Force UTF-8 stdout on Windows so box-drawing characters (U+2550 etc.)
# don't trip cp1252. Same workaround we use elsewhere in the project for
# log lines that contain em-dashes. .reconfigure() is Python 3.7+.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

# Make src.* importable when invoked as `python scripts/corpus_health.py`
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.services.corpus import CorpusRegistry  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
#  Data aggregation
# ─────────────────────────────────────────────────────────────────────


def load_hit_data(sessions_root: Path) -> tuple[Counter, dict[str, int], int]:
    """Aggregate corpus_hits across every session frame.json on disk.

    Returns:
        hit_counter:       Counter[node_id -> total fires]
        last_hit:          dict[node_id -> highest turn number seen]
        sessions_with_data: int, number of sessions that had any hit data
    """
    hit_counter: Counter = Counter()
    last_hit: dict[str, int] = {}
    sessions_with_data = 0

    if not sessions_root.exists():
        return hit_counter, last_hit, 0

    for frame_path in sessions_root.glob("*/frame.json"):
        try:
            data = json.loads(frame_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        hits: dict[str, int] = data.get("corpus_hits", {}) or {}
        last: dict[str, int] = data.get("corpus_last_hit", {}) or {}
        if not hits:
            continue
        sessions_with_data += 1
        for nid, count in hits.items():
            hit_counter[nid] += int(count)
        for nid, turn in last.items():
            if int(turn) > last_hit.get(nid, 0):
                last_hit[nid] = int(turn)

    return hit_counter, last_hit, sessions_with_data


def load_drafts(sessions_root: Path) -> list[dict[str, Any]]:
    """Walk every session's drafts/ directory and return a flat draft list.

    Only includes files that look like real DraftPacket JSON (must have
    ``id`` and ``status`` keys). Skips the following sidecar filename
    suffixes because they live alongside the authoritative packet but
    aren't themselves review-queue entries:

        *_raw.json          — miner's raw proposal, pre-wrap
        *.enriched.json     — dreaming-pass rewrite proposal; reviewer
                              promotes it by copying over the packet.json
                              (see draft.dream_log.md conventions)

    Also skips any JSON that fails to parse.

    Each returned entry is a dict:
        {
          "session_id": str,
          "path": Path,
          "mtime": float (unix seconds),
          "packet_type": "anchor" | "bundle" | "slab" | ...,
          "status": DraftStatus string,
          "confidence": float,
          "justification": str,
          "label": best-effort human-readable label,
          "source_chat_id": str,
          "raw": original dict,
        }
    """
    drafts: list[dict[str, Any]] = []
    if not sessions_root.exists():
        return drafts

    for draft_path in sessions_root.glob("*/drafts/*.json"):
        if draft_path.name.endswith("_raw.json"):
            continue
        if draft_path.name.endswith(".enriched.json"):
            continue
        try:
            data = json.loads(draft_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "id" not in data or "status" not in data:
            continue

        # Session id is the directory two levels above the draft file:
        # .../sessions/<session_id>/drafts/<file>.json
        session_id = draft_path.parent.parent.name

        # Best-effort label extraction from inline anchor/slab/bundle object
        label = data.get("id", "")
        if data.get("anchor"):
            label = (
                data["anchor"].get("canonical_phrase")
                or data["anchor"].get("title")
                or label
            )
        elif data.get("slab"):
            label = (
                data["slab"].get("title")
                or (data["slab"].get("canonical_text") or "").split("\n", 1)[0]
                or label
            )
        elif data.get("bundle"):
            payload = data["bundle"].get("payload") or {}
            intents = payload.get("intent") or []
            label = intents[0] if intents else label

        drafts.append({
            "session_id": session_id,
            "path": draft_path,
            "mtime": draft_path.stat().st_mtime,
            "packet_type": data.get("packet_type", "anchor"),
            "status": data.get("status", "DRAFT_UNAUTHORIZED"),
            "confidence": float(data.get("confidence") or 0.0),
            "justification": data.get("justification", ""),
            "label": str(label)[:60],
            "source_chat_id": data.get("source_chat_id", ""),
            "raw": data,
        })

    drafts.sort(key=lambda d: d["mtime"], reverse=True)
    return drafts


def collect_corpus_nodes(registry: CorpusRegistry) -> dict[str, dict[str, Any]]:
    """Walk every ACTIVE collection and return a provenance table.

    Returns a dict keyed by node_id with:
        {
          "type": "anchor" | "slab" | "bundle",
          "collection": collection_id,
          "label": short human-readable label,
          "obj": the pydantic model instance,
        }

    Inactive collections are intentionally excluded — they're absorbed
    into a condensate slab (see CorpusRegistry._deactivate_absorbed_
    collections) and dead-weight analysis on them would pollute the
    signal. Only nodes the matcher actually sees count toward health.
    """
    table: dict[str, dict[str, Any]] = {}
    for cid in sorted(registry.active_ids):
        store = registry.collections.get(cid)
        if not store:
            continue
        for a in store.anchors.values():
            table[a.id] = {
                "type": "anchor",
                "collection": cid,
                "label": a.canonical_phrase,
                "obj": a,
            }
        for s in store.slabs.values():
            label = s.title or (s.canonical_text or "").split("\n", 1)[0]
            table[s.id] = {
                "type": "slab",
                "collection": cid,
                "label": label[:60],
                "obj": s,
            }
        for b in store.bundles.values():
            intent = b.payload.intent or []
            label = intent[0] if intent else b.id
            table[b.id] = {
                "type": "bundle",
                "collection": cid,
                "label": label[:60],
                "obj": b,
            }
    return table


# ─────────────────────────────────────────────────────────────────────
#  Math helpers
# ─────────────────────────────────────────────────────────────────────


def gini(values: list[int]) -> float:
    """Gini coefficient on a list of non-negative counts.

    0.0 = perfect equality (every node fires equally)
    1.0 = maximal inequality (one node gets all fires)

    Uses the rank-weighted formulation which is O(n log n) and stable
    when values contain zeros or the total is zero.
    """
    if not values:
        return 0.0
    sorted_v = sorted(values)
    n = len(sorted_v)
    total = sum(sorted_v)
    if total == 0:
        return 0.0
    cum = 0
    for i, v in enumerate(sorted_v, start=1):
        cum += i * v
    return (2 * cum) / (n * total) - (n + 1) / n


# ─────────────────────────────────────────────────────────────────────
#  Section A — Utilization
# ─────────────────────────────────────────────────────────────────────


def section_a_utilization(
    nodes: dict[str, dict[str, Any]],
    hit_counter: Counter,
    sessions_with_data: int,
) -> dict[str, Any]:
    """Print Section A and return computed stats for downstream use."""
    total = len(nodes)
    by_type = Counter(info["type"] for info in nodes.values())
    by_collection = Counter(info["collection"] for info in nodes.values())

    # Fire counts restricted to nodes that actually live in the active
    # corpus. Stray keys in corpus_hits (tentatives, deleted nodes, old
    # IDs from prior corpus versions) are ignored so the denominator is
    # honest.
    live_fires = {nid: hit_counter.get(nid, 0) for nid in nodes}
    fired_nodes = {nid: c for nid, c in live_fires.items() if c > 0}
    dead_nodes = {nid: info for nid, info in nodes.items() if live_fires[nid] == 0}

    total_fires = sum(live_fires.values())
    gini_score = gini(list(live_fires.values()))

    print("═" * 68)
    print("  PRIME'S SHADOW — CORPUS HEALTH REPORT")
    print(f"  {date.today().isoformat()}")
    print("═" * 68)
    print()
    print("■ SECTION A — UTILIZATION")
    print(
        f"  Sessions with hit data: {sessions_with_data}    "
        f"Total fire events: {total_fires}"
    )
    print(
        f"  Corpus size: {total} nodes  "
        f"({by_type.get('anchor', 0)} anchors · "
        f"{by_type.get('slab', 0)} slabs · "
        f"{by_type.get('bundle', 0)} bundles)"
    )
    print(
        f"  Unique nodes ever fired: {len(fired_nodes)}  "
        f"({100.0 * len(fired_nodes) / total:.1f}% of corpus)"
        if total
        else "  (empty corpus)"
    )
    dead_flag = "  [!]" if total and len(dead_nodes) / total > 0.5 else ""
    print(
        f"  Dead nodes (never fired): {len(dead_nodes)}  "
        f"({100.0 * len(dead_nodes) / total:.1f}%){dead_flag}"
    )
    print(f"  Gini coefficient: {gini_score:.3f}  ", end="")
    if gini_score < 0.3:
        print("[near-uniform usage]")
    elif gini_score < 0.6:
        print("[moderate concentration]")
    else:
        print("[heavy concentration — few nodes dominate]")
    print()

    # Per-collection breakdown
    print("  Per-collection:")
    for cid in sorted(by_collection):
        col_nodes = [nid for nid, info in nodes.items() if info["collection"] == cid]
        col_fired = sum(1 for nid in col_nodes if live_fires[nid] > 0)
        pct = 100.0 * col_fired / len(col_nodes) if col_nodes else 0.0
        print(
            f"    {cid:<22} {len(col_nodes):>3} nodes   "
            f"{col_fired:>3} fired  ({pct:5.1f}%)"
        )
    print()

    # Top hot nodes
    top_n = 8
    hot = sorted(fired_nodes.items(), key=lambda kv: -kv[1])[:top_n]
    if hot:
        print(f"  Top {len(hot)} hot nodes:")
        for nid, count in hot:
            info = nodes[nid]
            pct = 100.0 * count / total_fires if total_fires else 0.0
            label = info["label"][:48]
            print(
                f"    {count:>4} fires ({pct:4.1f}%)  "
                f"[{info['type'][:6]:<6}]  {nid}"
            )
            print(f"         {label!r}")
    print()

    # Full dead-node list, grouped by collection then type
    if dead_nodes:
        print(f"  DEAD NODE LIST ({len(dead_nodes)} nodes, grouped by collection):")
        dead_by_col: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
        for nid, info in dead_nodes.items():
            dead_by_col[info["collection"]].append((nid, info))
        for cid in sorted(dead_by_col):
            rows = dead_by_col[cid]
            print(f"    [{cid}]  {len(rows)} dead:")
            # Sort by type then id for stable legible output
            rows.sort(key=lambda kv: (kv[1]["type"], kv[0]))
            for nid, info in rows:
                label = info["label"][:50]
                print(f"      {info['type']:<7}  {nid}")
                print(f"               {label!r}")
    print()

    return {
        "total": total,
        "by_type": dict(by_type),
        "fired": len(fired_nodes),
        "dead": len(dead_nodes),
        "gini": gini_score,
        "total_fires": total_fires,
    }


# ─────────────────────────────────────────────────────────────────────
#  Section B — Structural integrity
# ─────────────────────────────────────────────────────────────────────


def section_b_structural(
    nodes: dict[str, dict[str, Any]],
    registry: CorpusRegistry,
) -> dict[str, Any]:
    """Compute graph-shape metrics across the merged active corpus.

    The merged view is what the matcher actually sees at runtime, so
    graph health at that layer is the relevant layer to audit. Per-
    collection structural issues show up implicitly through the merged
    view.
    """
    merged = registry.merged

    all_ids: set[str] = set(nodes.keys())

    # Undirected adjacency assembled from three sources:
    #   (1) explicit Edge records
    #   (2) Anchor.invokes lists
    #   (3) Slab.depends_on and Slab.links.{anchors,bundles}
    #   (4) KeyBundle.depends_on and KeyBundle.supports
    # Invokes/depends_on/links are structural connections that matter
    # for "is this node reachable" even if no explicit Edge exists.
    adj: dict[str, set[str]] = {nid: set() for nid in all_ids}

    def link(a: str, b: str) -> None:
        if a in adj and b in adj and a != b:
            adj[a].add(b)
            adj[b].add(a)

    dangling: list[tuple[str, str, str]] = []  # (owner_id, field, missing_id)

    for eid, e in merged.edges.items():
        if e.from_node not in all_ids:
            dangling.append((eid, "edge.from", e.from_node))
        if e.to_node not in all_ids:
            dangling.append((eid, "edge.to", e.to_node))
        link(e.from_node, e.to_node)

    for a in merged.anchors.values():
        for target in a.invokes:
            if target not in all_ids:
                dangling.append((a.id, "invokes", target))
            else:
                link(a.id, target)
        for dep in a.depends_on:
            if dep not in all_ids:
                dangling.append((a.id, "depends_on", dep))
            else:
                link(a.id, dep)

    for s in merged.slabs.values():
        for dep in s.depends_on:
            if dep not in all_ids:
                dangling.append((s.id, "depends_on", dep))
            else:
                link(s.id, dep)
        for la in s.links.anchors:
            if la not in all_ids:
                dangling.append((s.id, "links.anchors", la))
            else:
                link(s.id, la)
        for lb in s.links.bundles:
            if lb not in all_ids:
                dangling.append((s.id, "links.bundles", lb))
            else:
                link(s.id, lb)

    for b in merged.bundles.values():
        for dep in b.depends_on:
            if dep not in all_ids:
                dangling.append((b.id, "depends_on", dep))
            else:
                link(b.id, dep)
        for sup in b.supports:
            if sup not in all_ids:
                dangling.append((b.id, "supports", sup))
            else:
                link(b.id, sup)

    # Orphans: nodes with zero connections in either direction
    orphans = sorted(nid for nid, neigh in adj.items() if not neigh)

    # Unsupported bundles: not referenced by any anchor.invokes or
    # any edge's to_node or any slab.links.bundles
    bundle_ids = {nid for nid, info in nodes.items() if info["type"] == "bundle"}
    referenced_bundles: set[str] = set()
    for a in merged.anchors.values():
        referenced_bundles.update(bid for bid in a.invokes if bid in bundle_ids)
    for s in merged.slabs.values():
        referenced_bundles.update(bid for bid in s.links.bundles if bid in bundle_ids)
    for e in merged.edges.values():
        if e.to_node in bundle_ids:
            referenced_bundles.add(e.to_node)
    unsupported_bundles = sorted(bundle_ids - referenced_bundles)

    # Connected components via BFS over undirected adjacency
    seen: set[str] = set()
    components: list[list[str]] = []
    for nid in all_ids:
        if nid in seen:
            continue
        comp: list[str] = []
        q: deque[str] = deque([nid])
        seen.add(nid)
        while q:
            cur = q.popleft()
            comp.append(cur)
            for nb in adj[cur]:
                if nb not in seen:
                    seen.add(nb)
                    q.append(nb)
        components.append(comp)
    components.sort(key=len, reverse=True)

    # Average degree by type
    deg_by_type: dict[str, list[int]] = defaultdict(list)
    for nid, info in nodes.items():
        deg_by_type[info["type"]].append(len(adj[nid]))

    # ── Print ────────────────────────────────────────────────────────
    print("■ SECTION B — STRUCTURAL INTEGRITY")
    orphan_flag = "  [!]" if orphans else ""
    print(f"  Orphan nodes: {len(orphans)}{orphan_flag}")
    for nid in orphans[:10]:
        info = nodes[nid]
        print(f"    {info['type']:<7}  {nid}    [{info['collection']}]")
    if len(orphans) > 10:
        print(f"    ... and {len(orphans) - 10} more")

    unsup_flag = "  [!]" if unsupported_bundles else ""
    print(f"  Unsupported bundles: {len(unsupported_bundles)}{unsup_flag}")
    for bid in unsupported_bundles:
        print(f"    {bid}    [{nodes[bid]['collection']}]")

    dangling_flag = "  [!]" if dangling else "  ok"
    print(f"  Dangling references: {len(dangling)}{dangling_flag}")
    for owner, field, missing in dangling[:10]:
        print(f"    {owner}.{field} → {missing}  (missing)")
    if len(dangling) > 10:
        print(f"    ... and {len(dangling) - 10} more")

    print(f"  Connected components: {len(components)}")
    for i, comp in enumerate(components[:5]):
        print(f"    #{i + 1}  {len(comp)} nodes")
        if len(comp) <= 3:
            for nid in comp:
                print(f"         {nodes[nid]['type']:<7}  {nid}")
    if len(components) > 5:
        print(f"    ... and {len(components) - 5} smaller components")

    print("  Average degree (undirected):")
    for t in ("anchor", "slab", "bundle"):
        degs = deg_by_type.get(t, [])
        avg = sum(degs) / len(degs) if degs else 0.0
        print(f"    {t:<7}  {avg:4.2f}  (n={len(degs)})")
    print()

    return {
        "orphans": orphans,
        "unsupported_bundles": unsupported_bundles,
        "dangling": dangling,
        "components": len(components),
    }


# ─────────────────────────────────────────────────────────────────────
#  Section A.5 — Review queue (drafts awaiting sign-off)
# ─────────────────────────────────────────────────────────────────────


# DraftStatus values considered "pending" (user action required).
# Sourced from src/models/enums.py:DraftStatus. Hard-coded as strings
# to avoid importing the enum when the JSON already stores raw strings.
PENDING_STATUSES = {
    "DRAFT_UNAUTHORIZED",
    "PROVISIONAL",
    "STALE_REVIEW_REQUIRED",
}
TERMINAL_STATUSES = {"COMMITTED", "REJECTED"}


def _format_age(mtime: float) -> str:
    """Human-readable age from a file mtime. Short form: 2h, 3d, 12d."""
    delta = max(0.0, time.time() - mtime)
    if delta < 3600:
        return f"{int(delta // 60)}m"
    if delta < 86400:
        return f"{int(delta // 3600)}h"
    return f"{int(delta // 86400)}d"


def _looks_ambiguous(d: dict[str, Any]) -> str:
    """Return a short explanation if the draft's anchor-vs-slab type
    decision looks questionable. Empty string = no complaint.

    Heuristics (conservative — only flag obvious mismatches):
      1. The ID prefix names a different type than packet_type. The
         DraftManager mints ids as f"tentative_{type}_..." so a file
         named "tentative_slab_*" with packet_type "anchor" is a
         self-inconsistency that the reviewer must resolve before
         commit.
      2. packet_type == "anchor" but inline slab field is populated,
         or vice versa — conflicting declared intent.
      3. packet_type == "anchor" but label is long (> 60 chars) — long
         canonical phrases are a slab smell.
      4. packet_type == "slab" but label is short (< 60 chars) and has
         no newlines in canonical_text — short single-line slabs are
         usually anchors in disguise.
      5. packet_type == "anchor" but no inline anchor payload at all —
         there's literally nothing to commit; the draft is unusable as
         written.
    """
    ptype = d.get("packet_type", "anchor")
    label = d.get("label", "")
    raw = d.get("raw", {})
    anchor_obj = raw.get("anchor")
    slab_obj = raw.get("slab")
    draft_id = raw.get("id", "")

    # ID-prefix disagrees with declared type
    if "tentative_anchor_" in draft_id and ptype != "anchor":
        return f"id prefix says anchor, packet_type={ptype}"
    if "tentative_slab_" in draft_id and ptype != "slab":
        return f"id prefix says slab, packet_type={ptype}"
    if "tentative_bundle_" in draft_id and ptype != "bundle":
        return f"id prefix says bundle, packet_type={ptype}"

    # Conflicting inline payload
    if ptype == "anchor" and slab_obj and not anchor_obj:
        return "declared anchor but inline slab populated"
    if ptype == "slab" and anchor_obj and not slab_obj:
        return "declared slab but inline anchor populated"

    # No inline payload at all — nothing to commit
    if ptype == "anchor" and not anchor_obj:
        return "no inline anchor payload — nothing to commit"
    if ptype == "slab" and not slab_obj:
        return "no inline slab payload — nothing to commit"
    if ptype == "bundle" and not raw.get("bundle"):
        return "no inline bundle payload — nothing to commit"

    # Length mismatches
    if ptype == "anchor" and len(label) > 60:
        return "long canonical phrase — may belong as slab"

    if ptype == "slab":
        text = ""
        if slab_obj:
            text = slab_obj.get("canonical_text") or ""
        if 0 < len(text) < 60 and "\n" not in text:
            return "short single-line slab — may belong as anchor"

    return ""


def section_a5_review_queue(drafts: list[dict[str, Any]]) -> dict[str, Any]:
    """Print Section A.5 — pending drafts needing human sign-off.

    Surfaces packet_type prominently so reviewers can see at a glance
    whether each candidate is being proposed as an anchor, a slab, or
    a bundle. Drafts that trip the `_looks_ambiguous` heuristic get a
    [TYPE?] marker asking for explicit type confirmation before commit.
    """
    print("■ SECTION A.5 — REVIEW QUEUE (drafts awaiting sign-off)")

    if not drafts:
        print("  (no draft files found under app/workbench/sessions/*/drafts/)")
        print()
        return {"pending": 0, "committed": 0, "rejected": 0}

    by_status: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for d in drafts:
        by_status[d["status"]].append(d)

    pending_count = sum(len(by_status.get(s, [])) for s in PENDING_STATUSES)
    committed_count = len(by_status.get("COMMITTED", []))
    rejected_count = len(by_status.get("REJECTED", []))
    other_count = sum(
        len(v) for s, v in by_status.items()
        if s not in PENDING_STATUSES and s not in TERMINAL_STATUSES
    )

    pending_flag = "  [!]" if pending_count > 10 else ""
    print(
        f"  Total drafts on disk: {len(drafts)}    "
        f"pending: {pending_count}{pending_flag}    "
        f"committed: {committed_count}    rejected: {rejected_count}"
        + (f"    other: {other_count}" if other_count else "")
    )

    # Per-status summary with counts-by-packet-type
    print()
    print("  Status breakdown (counts by packet_type):")
    for status in sorted(by_status):
        items = by_status[status]
        ptypes = Counter(d["packet_type"] for d in items)
        ptype_str = "  ".join(f"{pt}={c}" for pt, c in sorted(ptypes.items()))
        marker = "→" if status in PENDING_STATUSES else " "
        print(f"    {marker} {status:<24} {len(items):>3}   {ptype_str}")
    print()

    # The actual review queue — pending items, newest first, with type
    # prominently displayed and ambiguity flags.
    pending: list[dict[str, Any]] = []
    for s in PENDING_STATUSES:
        pending.extend(by_status.get(s, []))
    pending.sort(key=lambda d: d["mtime"], reverse=True)

    if pending:
        print(f"  PENDING DRAFTS ({len(pending)}, newest first):")
        print(
            "  " + "─" * 66
        )
        # Show up to 20 most recent; tail summarised if longer.
        shown = pending[:20]
        for d in shown:
            ambiguous = _looks_ambiguous(d)
            type_tag = f"[{d['packet_type'].upper():<8}]"
            if ambiguous:
                type_tag += " [TYPE?]"
            age = _format_age(d["mtime"])
            print(
                f"    {type_tag}  conf={d['confidence']:.2f}  "
                f"age={age:<4}  status={d['status']}"
            )
            print(f"        id:      {d['raw'].get('id', '?')}")
            print(f"        label:   {d['label']!r}")
            print(
                f"        session: {d['session_id']}    "
                f"chat: {d['source_chat_id']}"
            )
            just = (d["justification"] or "").strip().replace("\n", " ")
            if just:
                print(f"        reason:  {just[:100]}")
            if ambiguous:
                print(f"        [TYPE?]  {ambiguous}")
            print()
        if len(pending) > 20:
            print(f"    ... and {len(pending) - 20} more pending drafts")
            print()
    else:
        print("  No drafts currently pending — review queue empty.")
        print()

    return {
        "pending": pending_count,
        "committed": committed_count,
        "rejected": rejected_count,
        "pending_drafts": pending,
    }


# ─────────────────────────────────────────────────────────────────────
#  Section B.5 — Committed-orphan audit (Terra Preta pattern)
# ─────────────────────────────────────────────────────────────────────


# Prefixes marking former-draft provenance. Both forms exist on disk:
#   - `tentative_*` from current DraftManager (src/services/draft_manager.py)
#   - `mined_*`     from older mining-pipeline commits
# Neither prefix is rewritten on commit, so they persist as a historical
# marker that the node entered via the proposal pipeline rather than
# hand-authored corpus edits.
FORMER_DRAFT_PREFIXES = ("tentative_", "mined_")


def _is_former_draft(node_id: str) -> bool:
    return node_id.startswith(FORMER_DRAFT_PREFIXES)


def section_b5_committed_orphans(
    nodes: dict[str, dict[str, Any]],
    registry: CorpusRegistry,
) -> dict[str, Any]:
    """Print Section B.5 — former-draft nodes that committed but never
    acquired structural wiring ("commit != integration").

    A clean commit should either:
      - wire the new anchor into invokes chains (anchor → bundle/slab), or
      - link a new slab to existing anchors via links.anchors, or
      - be intentionally standalone (e.g. a seed topic, marked as such).

    This section flags former-draft nodes where NONE of those happened,
    which is the "Terra Preta pattern": valuable content that entered
    the corpus, fires on the matcher, and then sits in its own graph
    island with nothing pointing to it and nothing it points to.
    """
    print("■ SECTION B.5 — COMMITTED-ORPHAN AUDIT")
    print("  (former-draft nodes that committed without structural wiring)")

    merged = registry.merged
    all_ids = set(nodes.keys())

    # Build incoming-reference set per node id from the authoritative
    # graph sources. This is a one-pass index, separate from Section B's
    # undirected adjacency, because here we care about *directionality*:
    # "does anything point AT this node?" is the integration question.
    incoming: dict[str, set[str]] = defaultdict(set)
    outgoing: dict[str, set[str]] = defaultdict(set)

    for a in merged.anchors.values():
        for tgt in a.invokes:
            if tgt in all_ids:
                incoming[tgt].add(a.id)
                outgoing[a.id].add(tgt)
        for dep in a.depends_on:
            if dep in all_ids:
                incoming[dep].add(a.id)
                outgoing[a.id].add(dep)

    for s in merged.slabs.values():
        for dep in s.depends_on:
            if dep in all_ids:
                incoming[dep].add(s.id)
                outgoing[s.id].add(dep)
        for la in s.links.anchors:
            if la in all_ids:
                incoming[la].add(s.id)
                outgoing[s.id].add(la)
        for lb in s.links.bundles:
            if lb in all_ids:
                incoming[lb].add(s.id)
                outgoing[s.id].add(lb)

    for b in merged.bundles.values():
        for dep in b.depends_on:
            if dep in all_ids:
                incoming[dep].add(b.id)
                outgoing[b.id].add(dep)
        for sup in b.supports:
            if sup in all_ids:
                incoming[sup].add(b.id)
                outgoing[b.id].add(sup)

    for e in merged.edges.values():
        if e.from_node in all_ids and e.to_node in all_ids:
            incoming[e.to_node].add(e.from_node)
            outgoing[e.from_node].add(e.to_node)

    former_drafts = [nid for nid in all_ids if _is_former_draft(nid)]
    former_drafts.sort()

    # Classify each former draft
    fully_orphan: list[str] = []
    no_outgoing: list[str] = []
    no_incoming: list[str] = []
    healthy: list[str] = []

    for nid in former_drafts:
        has_out = bool(outgoing.get(nid))
        has_in = bool(incoming.get(nid))
        if not has_out and not has_in:
            fully_orphan.append(nid)
        elif not has_out:
            no_outgoing.append(nid)
        elif not has_in:
            no_incoming.append(nid)
        else:
            healthy.append(nid)

    total = len(former_drafts)
    print(
        f"  Former-draft nodes in corpus: {total}  "
        f"(prefixes: {', '.join(FORMER_DRAFT_PREFIXES)})"
    )
    if total == 0:
        print("  (no former-draft nodes found — nothing to audit)")
        print()
        return {"total": 0}

    fully_flag = "  [!]" if fully_orphan else ""
    print(f"    fully orphan (no in, no out):    {len(fully_orphan)}{fully_flag}")
    print(f"    no outgoing wiring:              {len(no_outgoing)}")
    print(f"    no incoming references:          {len(no_incoming)}")
    print(f"    healthy (both directions wired): {len(healthy)}")
    print()

    def _emit(label: str, ids: list[str]) -> None:
        if not ids:
            return
        print(f"  {label}:")
        for nid in ids:
            info = nodes.get(nid)
            if not info:
                continue
            ntype = info["type"]
            nlabel = info["label"][:48]
            print(f"    {ntype:<7}  {nid}")
            print(f"             {nlabel!r}")
            # Show direction hint explicitly so reviewer knows what's missing
            if nid in fully_orphan:
                hint = "needs both: add invokes-to AND references-from"
            elif nid in no_outgoing:
                hint = "needs outgoing: anchor.invokes/slab.links/bundle.supports"
            else:
                hint = "needs incoming: wire an existing anchor or slab to point here"
            print(f"             → {hint}")
        print()

    _emit("Fully orphan former drafts", fully_orphan)
    _emit("No outgoing wiring", no_outgoing)
    _emit("No incoming references", no_incoming)

    return {
        "total": total,
        "fully_orphan": fully_orphan,
        "no_outgoing": no_outgoing,
        "no_incoming": no_incoming,
        "healthy": healthy,
    }


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────


def main() -> int:
    registry = CorpusRegistry()
    registry.load_all()

    sessions_root = ROOT / "app" / "workbench" / "sessions"
    hit_counter, last_hit, sessions_with_data = load_hit_data(sessions_root)
    drafts = load_drafts(sessions_root)

    nodes = collect_corpus_nodes(registry)
    if not nodes:
        print("No active corpus nodes — nothing to report.")
        return 1

    section_a_utilization(nodes, hit_counter, sessions_with_data)
    section_a5_review_queue(drafts)
    section_b_structural(nodes, registry)
    section_b5_committed_orphans(nodes, registry)

    print("═" * 68)
    print("  End of report.")
    print("═" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
