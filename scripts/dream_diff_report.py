"""Diff every dream rewrite against its original draft.

Walks a session's drafts dir, for each `*.enriched.json` shows the
field-level diff against the corresponding `*_raw.json` (the original
mined proposal). Useful for:

  - Confirming the identity lock held (canonical_phrase / title
    unchanged everywhere)
  - Confirming the productive rewrites still happened (notes
    expansion, canonical_text refinement, ungrounded_references
    pruning, alias additions)
  - Spotting regressions where the lock killed legitimate work

Usage:
    python scripts/dream_diff_report.py                # most recent session
    python scripts/dream_diff_report.py 39d8c3d8       # specific session
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def find_session_dir(arg_id: str | None) -> Path:
    sessions = Path("app/workbench/sessions")
    if arg_id:
        return sessions / arg_id
    candidates = [d for d in sessions.iterdir() if d.is_dir() and (d / "drafts").exists()]
    return max(candidates, key=lambda d: d.stat().st_mtime)


def main():
    sid = sys.argv[1] if len(sys.argv) > 1 else None
    sdir = find_session_dir(sid)
    drafts_dir = sdir / "drafts"
    print(f"Session: {sdir.name}")
    print(f"Drafts: {drafts_dir}")
    print()

    enriched_files = sorted(drafts_dir.glob("*.enriched.json"))
    if not enriched_files:
        print("No enriched files in this session.")
        sys.exit(0)

    # Aggregate counters
    identity_changes = 0
    notes_changes = 0
    ctext_changes = 0
    aliases_changes = 0
    refs_changes = 0
    enriched_packet_lock_notes = 0

    for ep in enriched_files:
        draft_id = ep.stem.replace(".enriched", "")
        raw_path = drafts_dir / f"{draft_id}_raw.json"
        if not raw_path.exists():
            continue

        with open(ep, encoding="utf-8") as f:
            enriched = json.load(f)
        with open(raw_path, encoding="utf-8") as f:
            raw = json.load(f)

        ptype = raw.get("type", "?")
        # Track if the enriched packet's justification mentions identity lock
        if "Identity lock blocked" in (enriched.get("justification") or ""):
            enriched_packet_lock_notes += 1

        if ptype == "anchor":
            after = enriched.get("anchor") or {}
            before_phrase = (raw.get("canonical_phrase") or "").strip()
            after_phrase = (after.get("canonical_phrase") or "").strip()
            before_aliases = list(raw.get("aliases") or [])
            after_aliases = list(after.get("aliases") or [])
            before_notes = (raw.get("justification") or "").strip()
            after_notes = (after.get("notes") or "").strip()

            print(f"--- {draft_id} (anchor) ---")
            phrase_status = "PRESERVED" if before_phrase == after_phrase else "CHANGED!"
            print(f'  canonical_phrase  [{phrase_status}]: {before_phrase!r}')
            if before_phrase != after_phrase:
                print(f'                        -> {after_phrase!r}')
                identity_changes += 1
            if before_aliases != after_aliases:
                added = [a for a in after_aliases if a not in before_aliases]
                removed = [a for a in before_aliases if a not in after_aliases]
                if added or removed:
                    print(f'  aliases changed: + {added}  - {removed}')
                    aliases_changes += 1
            if before_notes != after_notes:
                print(f'  notes BEFORE ({len(before_notes)} chars): {before_notes[:120]}{"..." if len(before_notes)>120 else ""}')
                print(f'  notes AFTER  ({len(after_notes)} chars): {after_notes[:120]}{"..." if len(after_notes)>120 else ""}')
                notes_changes += 1
            print()

        elif ptype == "slab":
            after = enriched.get("slab") or {}
            before_title = (raw.get("title") or "").strip()
            after_title = (after.get("title") or "").strip()
            before_text = (raw.get("canonical_text") or "").strip()
            after_text = (after.get("canonical_text") or "").strip()
            before_refs = (raw.get("references_anchors") or [])
            after_links = (after.get("links") or {}).get("anchors") or []

            print(f"--- {draft_id} (slab) ---")
            title_status = "PRESERVED" if before_title == after_title else "CHANGED!"
            print(f'  title  [{title_status}]: {before_title!r}')
            if before_title != after_title:
                print(f'             -> {after_title!r}')
                identity_changes += 1
            if before_text != after_text:
                print(f'  canonical_text BEFORE ({len(before_text)} chars): {before_text[:140]}{"..." if len(before_text)>140 else ""}')
                print(f'  canonical_text AFTER  ({len(after_text)} chars): {after_text[:140]}{"..." if len(after_text)>140 else ""}')
                ctext_changes += 1
            if list(before_refs) != list(after_links):
                added = [r for r in after_links if r not in before_refs]
                removed = [r for r in before_refs if r not in after_links]
                if added or removed:
                    print(f'  links.anchors: + {added}  - {removed}')
                    refs_changes += 1
            print()

    print(f"=" * 78)
    print(f"AGGREGATE")
    print(f"  enriched files examined:           {len(enriched_files)}")
    print(f"  identity changes (canonical/title): {identity_changes}  {'(LOCK FAILED)' if identity_changes else '(LOCK HELD)'}")
    print(f"  notes changes (anchors):           {notes_changes}")
    print(f"  canonical_text changes (slabs):    {ctext_changes}")
    print(f"  aliases changes:                   {aliases_changes}")
    print(f"  links.anchors changes (slabs):     {refs_changes}")
    print(f"  enriched packets with lock note:   {enriched_packet_lock_notes}  (model tried to rename, parser blocked)")


if __name__ == "__main__":
    main()
