"""Auto-reject pending drafts that re-mine an already-absorbed collection.

Background — the "absorbed collection" pattern:
    A slabify promotion (POST /collections/{id}/promote) takes an entire
    collection (say `vin_disel_skit`) and collapses it into a single
    condensate slab in the parent corpus whose `canonical_text` starts
    with the string `"Neighborhood: <collection_id>"`. That marker is
    the authoritative signal that `<collection_id>` has been absorbed —
    its raw anchors stay on disk for possible future re-expansion, but
    the matcher and frame builder ignore them (see
    `CorpusRegistry._deactivate_absorbed_collections` for the self-heal).

What this means for the review queue:
    Any pending draft that was mined AGAINST an already-absorbed
    collection and whose `canonical_phrase` already exists verbatim
    as an anchor in that collection's on-disk `anchors.yaml` is pure
    pipeline noise — it is a duplicate of content that has already
    been summarized into a condensate slab, and committing it would
    double-fire the frame. This script finds those drafts and flips
    them to `REJECTED` with a clear `stale_reason` so that the review
    dashboard stops showing them as pending work.

Non-duplicate drafts (e.g. slab-type summaries whose titles do NOT
match any existing anchor) are left alone — those represent fresh
higher-level structure that a human should still review.

This is a first sketch of the wider "dreaming phase" architecture:
async, cold-path, full-context, selective-forgetting.

Usage:
    python scripts/reject_absorbed_duplicates.py            # dry run
    python scripts/reject_absorbed_duplicates.py --apply    # actually write
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

# Force UTF-8 stdout on Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.enums import DraftStatus  # noqa: E402
from src.models.schemas import DraftPacket  # noqa: E402

SESSIONS_ROOT = ROOT / "app" / "workbench" / "sessions"
CORPUS_ROOT = ROOT / "app" / "corpus"
CORPORA_ROOT = ROOT / "app" / "corpora"

NEIGHBORHOOD_PREFIX = "Neighborhood: "


def _load_yaml(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, list) else []


def discover_absorbed_collections() -> dict[str, tuple[str, str]]:
    """Scan all slabs.yaml files for 'Neighborhood: <id>' markers.

    Returns a map {absorbed_cid: (owner_cid, slab_id)} so we can report
    which condensate slab caused the absorption. Mirrors the logic in
    `CorpusRegistry._deactivate_absorbed_collections`.
    """
    absorbed: dict[str, tuple[str, str]] = {}

    # Scan every slabs.yaml under app/corpus and app/corpora/*/objects
    candidates: list[tuple[Path, str]] = []
    legacy = CORPUS_ROOT / "objects" / "slabs.yaml"
    if legacy.exists():
        candidates.append((legacy, "default"))
    if CORPORA_ROOT.exists():
        for child in sorted(CORPORA_ROOT.iterdir()):
            if not child.is_dir():
                continue
            sf = child / "objects" / "slabs.yaml"
            if sf.exists():
                candidates.append((sf, child.name))

    for sf, owner_cid in candidates:
        for entry in _load_yaml(sf):
            text = (entry.get("canonical_text") or "").strip()
            if not text.startswith(NEIGHBORHOOD_PREFIX):
                continue
            absorbed_cid = text[len(NEIGHBORHOOD_PREFIX):].split("\n", 1)[0].strip()
            if not absorbed_cid or absorbed_cid == owner_cid:
                continue
            # First-writer-wins if multiple condensates target the same cid
            absorbed.setdefault(absorbed_cid, (owner_cid, entry.get("id", "?")))
    return absorbed


def load_collection_phrases(collection_id: str) -> dict[str, str]:
    """Return {canonical_phrase_lower: anchor_id} for a collection's anchors.

    We also fold in slab titles so a re-mined slab whose title matches an
    existing slab is detectable — though the typical collision is on the
    anchor-phrase axis.
    """
    root = CORPORA_ROOT / collection_id
    if not (root / "objects").exists():
        if collection_id == "default" and (CORPUS_ROOT / "objects").exists():
            root = CORPUS_ROOT
        else:
            return {}

    phrases: dict[str, str] = {}
    for anchor in _load_yaml(root / "objects" / "anchors.yaml"):
        cp = (anchor.get("canonical_phrase") or "").strip()
        if cp:
            phrases[cp.lower()] = anchor.get("id", "?")
    for slab in _load_yaml(root / "objects" / "slabs.yaml"):
        title = (slab.get("title") or "").strip()
        if title:
            phrases.setdefault(title.lower(), slab.get("id", "?"))
    return phrases


def target_collection_of(raw: dict[str, Any]) -> str | None:
    """Return the collection this draft was mined against, if known."""
    # Miner writes either '_target_collection' or 'target_collection'
    return raw.get("_target_collection") or raw.get("target_collection")


def process(apply: bool) -> int:
    absorbed = discover_absorbed_collections()
    if not absorbed:
        print("No absorbed collections detected — nothing to do.")
        return 0

    print("Absorbed collections detected:")
    for cid, (owner, slab_id) in sorted(absorbed.items()):
        print(f"  {cid}  →  {slab_id}  (in collection '{owner}')")
    print()

    # Preload canonical-phrase sets for each absorbed collection
    phrase_index: dict[str, dict[str, str]] = {
        cid: load_collection_phrases(cid) for cid in absorbed
    }
    for cid, phrases in phrase_index.items():
        print(f"  {cid}: {len(phrases)} on-disk canonical phrases indexed")
    print()

    stats = {
        "scanned": 0,
        "non_target": 0,  # no target_collection at all
        "target_live": 0,  # target is NOT absorbed
        "target_absorbed_novel": 0,  # absorbed but phrase didn't match
        "rejected": 0,
        "already_rejected": 0,
        "not_pending": 0,
        "error": 0,
    }

    mode = "APPLY" if apply else "DRY RUN"
    print(f"═══ Auto-reject absorbed duplicates — {mode} ═══")
    print()

    for session_dir in sorted(SESSIONS_ROOT.iterdir()):
        if not session_dir.is_dir():
            continue
        drafts_dir = session_dir / "drafts"
        if not drafts_dir.exists():
            continue

        session_hits: list[str] = []

        for packet_path in sorted(drafts_dir.glob("*.json")):
            if packet_path.name.endswith("_raw.json"):
                continue
            stats["scanned"] += 1

            try:
                packet_data = json.loads(packet_path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"  [error] could not read {packet_path.name}: {e}")
                stats["error"] += 1
                continue

            if "status" not in packet_data:
                continue

            status = packet_data.get("status")
            # Only touch drafts still waiting for sign-off.
            if status != DraftStatus.DRAFT_UNAUTHORIZED.value:
                if status == DraftStatus.REJECTED.value:
                    stats["already_rejected"] += 1
                else:
                    stats["not_pending"] += 1
                continue

            draft_id = packet_data.get("id", packet_path.stem)
            raw_path = drafts_dir / f"{draft_id}_raw.json"
            if not raw_path.exists():
                stats["non_target"] += 1
                continue

            try:
                raw = json.loads(raw_path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"  [error] could not read raw {raw_path.name}: {e}")
                stats["error"] += 1
                continue

            tc = target_collection_of(raw)
            if not tc:
                stats["non_target"] += 1
                continue
            if tc not in absorbed:
                stats["target_live"] += 1
                continue

            # The raw was mined against an absorbed collection. Check
            # whether the canonical_phrase (or slab title) already exists
            # verbatim in that collection's on-disk contents.
            phrase = (
                raw.get("canonical_phrase")
                or raw.get("title")
                or ""
            ).strip()
            key = phrase.lower()
            existing_id = phrase_index.get(tc, {}).get(key)

            if not existing_id:
                stats["target_absorbed_novel"] += 1
                session_hits.append(
                    f"  [keep] {draft_id}  novel in absorbed '{tc}':  {phrase!r}"
                )
                continue

            # Confirmed duplicate. Build the new status + reason.
            owner_cid, slab_id = absorbed[tc]
            stale_reason = (
                f"Target collection '{tc}' was absorbed into "
                f"{slab_id} (in '{owner_cid}'). "
                f"canonical_phrase already exists as {existing_id}."
            )

            packet_data["status"] = DraftStatus.REJECTED.value
            packet_data["stale_reason"] = stale_reason

            # Round-trip through Pydantic to catch any schema drift
            # before writing.
            try:
                packet = DraftPacket(**packet_data)
            except Exception as e:
                print(
                    f"  [error] schema validation failed on {draft_id}: {e}"
                )
                stats["error"] += 1
                continue

            stats["rejected"] += 1
            session_hits.append(
                f"  [reject] {draft_id}  →  {existing_id}    {phrase!r}"
            )

            if apply:
                packet_path.write_text(
                    json.dumps(packet.model_dump(mode="json"), indent=2),
                    encoding="utf-8",
                )

        if session_hits:
            print(f"── session {session_dir.name} ──")
            for line in session_hits:
                print(line)
            print()

    print("═" * 48)
    print(f"  Packets scanned:                  {stats['scanned']}")
    print(f"  Rejected (absorbed duplicate):    {stats['rejected']}")
    print(f"  Kept (absorbed but novel):        {stats['target_absorbed_novel']}")
    print(f"  Kept (target still live):         {stats['target_live']}")
    print(f"  Skipped (no target collection):   {stats['non_target']}")
    print(f"  Skipped (not pending):            {stats['not_pending']}")
    print(f"  Skipped (already rejected):       {stats['already_rejected']}")
    print(f"  Errors:                           {stats['error']}")
    print("═" * 48)
    if not apply and stats["rejected"] > 0:
        print("  [DRY RUN — run with --apply to write changes]")
    return 0 if stats["error"] == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write rejections. Without this flag, dry-run only.",
    )
    args = parser.parse_args()

    if not SESSIONS_ROOT.exists():
        print(f"No sessions directory at {SESSIONS_ROOT}")
        return 1

    return process(args.apply)


if __name__ == "__main__":
    sys.exit(main())
