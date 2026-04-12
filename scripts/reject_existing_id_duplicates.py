"""Auto-reject pending drafts whose canonical_phrase is an existing anchor ID.

Background — the "ID-as-phrase" miner bug:
    When the miner detects that a chat references an existing corpus anchor
    (e.g. turn 12 strongly activates `ANCHOR_ARCHER_CHANT_v1`), it should
    report that as a hit on the existing anchor. Instead, in some code
    path, it emits a NEW draft whose `canonical_phrase` is the matched
    anchor's ID string itself. The resulting draft looks like:

        {"type": "anchor",
         "canonical_phrase": "ANCHOR_ARCHER_CHANT_v1",
         "canonical_text": "... analogous to 'I am the bone of my sword' ..."}

    This is pure pipeline noise — the correct anchor is already in the
    corpus, and committing the draft would either collide on ID (if the
    validator catches it) or spawn a sibling anchor with a non-sense
    canonical_phrase (if it doesn't).

Detection rule — strict by design:
    A pending draft is auto-rejectable iff its `canonical_phrase` (case-
    insensitive, whitespace-trimmed) exactly matches the id of any existing
    anchor in the merged corpus view. Slab drafts are ALSO checked against
    slab IDs and titles, but in practice the bug only hits anchors.

    We deliberately do NOT fuzzy-match — a draft whose phrase is
    `"i am the bone of my sword"` (the real phrase, which is the
    existing anchor's canonical_phrase, not its ID) would NOT be caught
    here. That's a separate class of duplicate and belongs to a different
    script (or ideally to a matcher fix in the miner).

Usage:
    python scripts/reject_existing_id_duplicates.py            # dry run
    python scripts/reject_existing_id_duplicates.py --apply    # actually write
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

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


def _load_yaml(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, list) else []


def build_existing_id_index() -> dict[str, tuple[str, str]]:
    """Return {lowercase_id: (kind, owner_collection)} for every anchor + slab
    across every on-disk collection. First-writer-wins on collision (default
    collection takes priority if it appears first)."""
    index: dict[str, tuple[str, str]] = {}

    candidates: list[tuple[Path, str]] = []
    # Legacy path
    if (CORPUS_ROOT / "objects").exists():
        candidates.append((CORPUS_ROOT, "default"))
    # Multi-collection path
    if CORPORA_ROOT.exists():
        for child in sorted(CORPORA_ROOT.iterdir()):
            if child.is_dir() and (child / "objects").exists():
                candidates.append((child, child.name))

    for root, cid in candidates:
        for anchor in _load_yaml(root / "objects" / "anchors.yaml"):
            aid = (anchor.get("id") or "").strip()
            if aid:
                index.setdefault(aid.lower(), ("anchor", cid))
        for slab in _load_yaml(root / "objects" / "slabs.yaml"):
            sid = (slab.get("id") or "").strip()
            if sid:
                index.setdefault(sid.lower(), ("slab", cid))
    return index


def process(apply: bool) -> int:
    index = build_existing_id_index()
    print(f"Existing corpus IDs indexed: {len(index)}")
    print()

    stats = {
        "scanned": 0,
        "rejected": 0,
        "not_pending": 0,
        "no_phrase": 0,
        "no_id_match": 0,
        "error": 0,
    }

    mode = "APPLY" if apply else "DRY RUN"
    print(f"═══ Auto-reject ID-as-phrase duplicates — {mode} ═══")
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
                data = json.loads(packet_path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"  [error] could not read {packet_path.name}: {e}")
                stats["error"] += 1
                continue

            if "status" not in data:
                continue
            if data.get("status") != DraftStatus.DRAFT_UNAUTHORIZED.value:
                stats["not_pending"] += 1
                continue

            # Pull the phrase from the inline payload — post-backfill
            # this is populated for every packet. Fall back to the raw
            # sidecar if for some reason it isn't.
            phrase = ""
            if data.get("anchor"):
                phrase = (data["anchor"].get("canonical_phrase") or "").strip()
            elif data.get("slab"):
                phrase = (data["slab"].get("title") or "").strip()
            elif data.get("bundle"):
                # Bundles don't have a natural phrase; skip.
                pass

            if not phrase:
                # Last-ditch: read raw sidecar
                draft_id = data.get("id", packet_path.stem)
                raw_path = drafts_dir / f"{draft_id}_raw.json"
                if raw_path.exists():
                    try:
                        raw = json.loads(raw_path.read_text(encoding="utf-8"))
                        phrase = (
                            raw.get("canonical_phrase") or raw.get("title") or ""
                        ).strip()
                    except Exception:
                        pass

            if not phrase:
                stats["no_phrase"] += 1
                continue

            match = index.get(phrase.lower())
            if not match:
                stats["no_id_match"] += 1
                continue

            existing_kind, owner_cid = match
            draft_id = data.get("id", packet_path.stem)
            stale_reason = (
                f"canonical_phrase '{phrase}' is the ID of existing "
                f"{existing_kind} in collection '{owner_cid}'. "
                f"Miner emitted corpus-ID-as-phrase (known bug)."
            )

            data["status"] = DraftStatus.REJECTED.value
            data["stale_reason"] = stale_reason

            try:
                packet = DraftPacket(**data)
            except Exception as e:
                print(f"  [error] schema validation failed on {draft_id}: {e}")
                stats["error"] += 1
                continue

            stats["rejected"] += 1
            session_hits.append(
                f"  [reject] {draft_id}  →  existing {existing_kind} '{phrase}' "
                f"in '{owner_cid}'"
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
    print(f"  Rejected (ID-as-phrase):          {stats['rejected']}")
    print(f"  Skipped (not pending):            {stats['not_pending']}")
    print(f"  Skipped (no phrase at all):       {stats['no_phrase']}")
    print(f"  Kept (phrase not an existing ID): {stats['no_id_match']}")
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
