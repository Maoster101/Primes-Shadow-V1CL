"""Verify dreaming's canonical_phrase / title identity lock held.

Walks a session's drafts directory, compares each `*.enriched.json`
against its `*_raw.json` (the original mined proposal), and checks
that:

  - For anchor drafts: enriched.anchor.canonical_phrase ==
                       raw.canonical_phrase
  - For slab drafts:   enriched.slab.title == raw.title

Any mismatch = identity lock breach. Prints a per-draft summary +
aggregate counts. Exit code 0 if lock held everywhere, 1 if any breach.

Usage:
    python scripts/verify_identity_lock.py                          # most recent session
    python scripts/verify_identity_lock.py 39d8c3d8                 # specific session id
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def find_session_dir(arg_id: str | None) -> Path:
    sessions = Path("app/workbench/sessions")
    if not sessions.exists():
        print("No sessions dir found", file=sys.stderr)
        sys.exit(2)
    if arg_id:
        d = sessions / arg_id
        if not d.exists():
            print(f"Session {arg_id} not found", file=sys.stderr)
            sys.exit(2)
        return d
    # Pick the most recently modified session dir
    candidates = [d for d in sessions.iterdir() if d.is_dir() and (d / "drafts").exists()]
    if not candidates:
        print("No session dirs with drafts found", file=sys.stderr)
        sys.exit(2)
    return max(candidates, key=lambda d: d.stat().st_mtime)


def main():
    sid = sys.argv[1] if len(sys.argv) > 1 else None
    sdir = find_session_dir(sid)
    drafts_dir = sdir / "drafts"
    print(f"Verifying identity lock for session: {sdir.name}")
    print(f"Drafts dir: {drafts_dir}")
    print()

    enriched_files = sorted(drafts_dir.glob("*.enriched.json"))
    if not enriched_files:
        print("No enriched files in this session — nothing to verify.")
        sys.exit(0)

    breaches = []
    preserved = []
    skipped = []

    for ep in enriched_files:
        draft_id = ep.stem.replace(".enriched", "")
        raw_path = drafts_dir / f"{draft_id}_raw.json"
        if not raw_path.exists():
            skipped.append((draft_id, "no _raw.json"))
            continue

        with open(ep, encoding="utf-8") as f:
            enriched = json.load(f)
        with open(raw_path, encoding="utf-8") as f:
            raw = json.load(f)

        ptype = raw.get("type", "?")

        if ptype == "anchor":
            before = (raw.get("canonical_phrase") or "").strip()
            after_dict = enriched.get("anchor") or {}
            after = (after_dict.get("canonical_phrase") or "").strip()
            field = "canonical_phrase"
        elif ptype == "slab":
            before = (raw.get("title") or "").strip()
            after_dict = enriched.get("slab") or {}
            after = (after_dict.get("title") or "").strip()
            field = "title"
        else:
            skipped.append((draft_id, f"type={ptype}"))
            continue

        if before == after:
            preserved.append((draft_id, ptype, field, before))
        else:
            breaches.append((draft_id, ptype, field, before, after))

    print(f"=== RESULTS ===")
    print(f"  Enriched drafts examined: {len(enriched_files)}")
    print(f"  Identity preserved:       {len(preserved)}")
    print(f"  Identity breaches:        {len(breaches)}")
    print(f"  Skipped:                  {len(skipped)}")
    print()

    if breaches:
        print(f"=== BREACHES ({len(breaches)}) — LOCK FAILED ===")
        for did, ptype, field, before, after in breaches:
            print(f"  ! {did} ({ptype})")
            print(f"      {field} BEFORE: {before!r}")
            print(f"      {field} AFTER:  {after!r}")
        print()

    if preserved:
        print(f"=== PRESERVED ({len(preserved)}) — these are the rewrites that respected the lock ===")
        for did, ptype, field, value in preserved:
            print(f"  v {did} ({ptype}): {field}={value!r}")
        print()

    if skipped:
        print(f"=== SKIPPED ({len(skipped)}) ===")
        for did, reason in skipped:
            print(f"  - {did}: {reason}")
        print()

    if breaches:
        print("RESULT: LOCK FAILED")
        sys.exit(1)
    else:
        print("RESULT: LOCK HELD on all enriched rewrites")
        sys.exit(0)


if __name__ == "__main__":
    main()
