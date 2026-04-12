"""Backfill inline anchor/slab/bundle payloads on existing DraftPackets.

Before the fix in draft_manager.extract_proposals, every mined draft was
written with `anchor=null / slab=null / bundle=null` even when the miner
had produced a full payload — the raw proposal was stashed in a sibling
`{draft_id}_raw.json` file and the typed inline fields on the packet
were never populated. This left the whole review queue unreadable
without knowing about the sidecar.

This script walks every session's drafts/ directory and, for each
DraftPacket whose inline fields are all null, loads the matching raw
sidecar and backfills:

    - packet_type            ← raw.type (fixes anchor/slab type downgrade)
    - anchor / slab / bundle ← one of them, shaped to match the schema
    - source_turns           ← raw.source_turns OR raw.source_pairs

Already-populated packets are left alone. Drafts with no raw sidecar
or unrecognised types are reported and skipped (no silent data loss).

Usage:
    python scripts/backfill_draft_payloads.py            # dry run, reports only
    python scripts/backfill_draft_payloads.py --apply    # actually write
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Force UTF-8 stdout on Windows (same workaround as corpus_health.py)
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.schemas import DraftPacket  # noqa: E402


SESSIONS_ROOT = ROOT / "app" / "workbench" / "sessions"


def build_inline_payload(
    draft_id: str, raw: dict[str, Any]
) -> tuple[str, dict | None, dict | None, dict | None]:
    """Return (packet_type, anchor_dict, slab_dict, bundle_dict).

    Mirrors the eager-construction logic in draft_manager.extract_proposals
    so that backfilled packets match packets that get created after the fix.
    """
    prop_type = raw.get("type", "anchor")

    if prop_type == "anchor":
        return (
            "anchor",
            {
                "id": draft_id,
                "canonical_phrase": raw.get("canonical_phrase", "") or "",
                "aliases": list(raw.get("aliases") or []),
                "invokes": [],
                "notes": raw.get("justification", "") or "",
            },
            None,
            None,
        )

    if prop_type == "slab":
        return (
            "slab",
            None,
            {
                "id": draft_id,
                "title": (
                    raw.get("title") or raw.get("canonical_phrase") or ""
                ),
                "canonical_text": raw.get("canonical_text", "") or "",
                "links": {"anchors": [], "bundles": []},
                "version": "v1",
            },
            None,
        )

    if prop_type == "bundle":
        payload = raw.get("payload") or {}
        if "intent" not in payload or not payload["intent"]:
            justification = (raw.get("justification") or "").strip()
            payload["intent"] = [justification] if justification else [draft_id]
        return (
            "bundle",
            None,
            None,
            {
                "id": draft_id,
                "payload": payload,
                "version": "v1",
            },
        )

    return (prop_type, None, None, None)


def normalize_source_turns(raw: dict[str, Any], fallback: list[int]) -> list[int]:
    """Collapse source_turns + source_pairs into a single list."""
    st = raw.get("source_turns") or raw.get("source_pairs")
    if not st:
        return fallback
    return list(st)


def process_session(session_dir: Path, apply: bool) -> dict[str, int]:
    """Process one session's drafts/. Returns counts of backfilled/skipped."""
    stats = {
        "total": 0,
        "already_populated": 0,
        "backfilled": 0,
        "no_raw": 0,
        "type_corrected": 0,
        "error": 0,
    }

    drafts_dir = session_dir / "drafts"
    if not drafts_dir.exists():
        return stats

    for packet_path in sorted(drafts_dir.glob("*.json")):
        if packet_path.name.endswith("_raw.json"):
            continue
        stats["total"] += 1

        try:
            data = json.loads(packet_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  [error] could not read {packet_path.name}: {e}")
            stats["error"] += 1
            continue

        if "id" not in data or "status" not in data:
            # Not a DraftPacket — probably some other metadata file
            continue

        draft_id = data.get("id", packet_path.stem)
        already_has_inline = any(
            data.get(k) for k in ("anchor", "slab", "bundle")
        )
        if already_has_inline:
            stats["already_populated"] += 1
            continue

        raw_path = drafts_dir / f"{draft_id}_raw.json"
        if not raw_path.exists():
            # No raw sibling — packet was either hand-authored, from a
            # prior schema version, or the raw file got deleted. Skip
            # silently unless verbose, but count for the report.
            stats["no_raw"] += 1
            continue

        try:
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  [error] could not read raw {raw_path.name}: {e}")
            stats["error"] += 1
            continue

        packet_type, anchor_dict, slab_dict, bundle_dict = build_inline_payload(
            draft_id, raw
        )

        old_packet_type = data.get("packet_type", "anchor")
        if old_packet_type != packet_type:
            stats["type_corrected"] += 1

        # Mutate the packet dict in place
        data["packet_type"] = packet_type
        data["anchor"] = anchor_dict
        data["slab"] = slab_dict
        data["bundle"] = bundle_dict

        # Correct source_turns if it looks like the default fallback
        # and raw has something better. Don't overwrite real values.
        if data.get("source_turns") in ([0], []):
            data["source_turns"] = normalize_source_turns(
                raw, data.get("source_turns") or [0]
            )

        # Round-trip through Pydantic to verify schema validity, then
        # serialize the validated model back to disk. If validation
        # fails, report and skip rather than writing corrupt data.
        try:
            packet = DraftPacket(**data)
        except Exception as e:
            print(
                f"  [error] schema validation failed on {draft_id}: {e}"
            )
            stats["error"] += 1
            continue

        stats["backfilled"] += 1
        if apply:
            packet_path.write_text(
                json.dumps(packet.model_dump(mode="json"), indent=2),
                encoding="utf-8",
            )

        # Console report — one line per backfilled draft
        label = ""
        if anchor_dict:
            label = anchor_dict.get("canonical_phrase", "") or ""
        elif slab_dict:
            label = slab_dict.get("title", "") or ""
        print(
            f"  {packet_type:<6}  {draft_id}  "
            f"{'(type: '+old_packet_type+'→'+packet_type+')' if old_packet_type != packet_type else ''}"
        )
        if label:
            print(f"           {label!r}")

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write backfilled packets. Without this flag, dry-run only.",
    )
    args = parser.parse_args()

    if not SESSIONS_ROOT.exists():
        print(f"No sessions directory at {SESSIONS_ROOT}")
        return 1

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"═══ Draft payload backfill — {mode} ═══")
    print()

    grand = {
        "total": 0, "already_populated": 0, "backfilled": 0,
        "no_raw": 0, "type_corrected": 0, "error": 0,
    }

    for session_dir in sorted(SESSIONS_ROOT.iterdir()):
        if not session_dir.is_dir():
            continue
        drafts_dir = session_dir / "drafts"
        if not drafts_dir.exists():
            continue

        print(f"── session {session_dir.name} ──")
        stats = process_session(session_dir, args.apply)
        for k, v in stats.items():
            grand[k] += v
        print()

    print("═" * 48)
    print(f"  Total packets scanned:   {grand['total']}")
    print(f"  Already populated:       {grand['already_populated']}")
    print(f"  Backfilled:              {grand['backfilled']}")
    print(f"    ↳ type corrected:      {grand['type_corrected']}")
    print(f"  Skipped (no raw file):   {grand['no_raw']}")
    print(f"  Errors:                  {grand['error']}")
    print("═" * 48)
    if not args.apply and grand["backfilled"] > 0:
        print("  [DRY RUN — run with --apply to write changes]")
    return 0 if grand["error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
