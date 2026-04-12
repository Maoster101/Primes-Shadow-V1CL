"""Commit the salience anchor+slab pair through the patched review_draft path.

This exercises the dreaming-commit gap fix in draft_manager.py. The two
drafts live in session 1ed6e74d, both with their .enriched.json already
promoted over the live .json. Order matters: anchor first (no deps), then
slab (which carries links.anchors -> the just-committed anchor id).

Post-commit verification loads the merged corpus fresh from disk and
asserts the enriched content strings all landed — if the fallback ladder
is doing its job, none of the confabulated "runtime header" text should
be present and slab.links.anchors must point at the committed anchor.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Windows cp1252 stdout cannot encode U+2500 etc.; UTF-8 is the lingua franca
# for every other tool in this project.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.services.corpus import CorpusRegistry
from src.services.draft_manager import DraftManager
from src.services.session_store import SessionStore

SESSION_ID = "1ed6e74d"
ANCHOR_DRAFT_ID = "tentative_anchor_73d88b7b_v1"
SLAB_DRAFT_ID = "tentative_slab_a9049516_v1"


def build_services():
    registry = CorpusRegistry()
    results = registry.load_all()  # dict[collection_id, list[error_str]]
    total_errs = sum(len(v) for v in (results or {}).values())
    if total_errs:
        print(f"[load] WARNING: {total_errs} validation errors on load")
        for cid, errs in (results or {}).items():
            for e in errs[:3]:
                print(f"  - [{cid}] {e}")
    corpus = registry.merged
    print(
        f"Merged corpus pre-commit: {len(corpus.anchors)} anchors, "
        f"{len(corpus.slabs)} slabs, {len(corpus.bundles)} bundles"
    )
    print(f"Active collections: {sorted(registry.active_ids)}")

    session_store = SessionStore(PROJECT_ROOT / "app" / "workbench" / "sessions")
    draft_manager = DraftManager(
        corpus=corpus,
        session_store=session_store,
    )
    draft_manager._registry = registry
    return registry, corpus, draft_manager


async def commit_draft(dm: DraftManager, draft_id: str) -> dict:
    print(f"\n--- Committing {draft_id} ---")
    result = await dm.review_draft(
        session_id=SESSION_ID,
        draft_id=draft_id,
        action="promote_corpus",
        oli_mode="ON",
        drift_severity="low",
    )
    print(f"  result: {result}")
    return result


async def main():
    registry, corpus, dm = build_services()

    # Sanity: confirm both drafts are visible to the session store
    anchor_packet = dm.session_store.load_draft_packet(SESSION_ID, ANCHOR_DRAFT_ID)
    slab_packet = dm.session_store.load_draft_packet(SESSION_ID, SLAB_DRAFT_ID)
    assert anchor_packet is not None, "anchor draft not found on disk"
    assert slab_packet is not None, "slab draft not found on disk"
    assert anchor_packet.anchor is not None, "anchor packet inline dict missing!"
    assert slab_packet.slab is not None, "slab packet inline dict missing!"
    assert slab_packet.slab["links"]["anchors"] == [ANCHOR_DRAFT_ID], (
        f"slab.links.anchors not wired: {slab_packet.slab['links']}"
    )
    print(
        f"Pre-commit sanity OK: anchor.canonical_phrase="
        f"{anchor_packet.anchor['canonical_phrase']!r}, "
        f"slab.links.anchors={slab_packet.slab['links']['anchors']}"
    )

    anchor_result = await commit_draft(dm, ANCHOR_DRAFT_ID)
    if "error" in anchor_result:
        print(f"ABORT: anchor commit failed: {anchor_result}")
        return 1

    slab_result = await commit_draft(dm, SLAB_DRAFT_ID)
    if "error" in slab_result:
        print(f"ABORT: slab commit failed: {slab_result}")
        return 1

    # --- Verification: fresh reload from disk ---
    print("\n=== Verification: fresh reload ===")
    registry2 = CorpusRegistry()
    results2 = registry2.load_all()
    total_errs2 = sum(len(v) for v in (results2 or {}).values())
    if total_errs2:
        print(f"[reload] {total_errs2} errors")
        for cid, errs in (results2 or {}).items():
            for e in errs[:3]:
                print(f"  - [{cid}] {e}")
    corpus2 = registry2.merged
    print(
        f"Merged corpus post-commit: {len(corpus2.anchors)} anchors, "
        f"{len(corpus2.slabs)} slabs, {len(corpus2.bundles)} bundles"
    )

    committed_anchor = corpus2.anchors.get(ANCHOR_DRAFT_ID)
    committed_slab = corpus2.slabs.get(SLAB_DRAFT_ID)
    assert committed_anchor is not None, f"anchor {ANCHOR_DRAFT_ID} missing after reload"
    assert committed_slab is not None, f"slab {SLAB_DRAFT_ID} missing after reload"

    # Anchor content assertions
    print(f"\nAnchor canonical_phrase: {committed_anchor.canonical_phrase!r}")
    print(f"Anchor aliases ({len(committed_anchor.aliases)}): {committed_anchor.aliases}")
    assert committed_anchor.canonical_phrase == "node salience", (
        f"canonical_phrase wrong: {committed_anchor.canonical_phrase!r}"
    )
    for needle in ("salience_now", "salience_smoothed", "alpha=0.30", "EWA", "frame_manager"):
        assert needle in committed_anchor.notes, f"anchor.notes missing {needle!r}"
    print("Anchor notes acceptance test: all 5 needles present OK")

    # Slab content assertions
    print(f"\nSlab title: {committed_slab.title!r}")
    print(f"Slab links.anchors: {committed_slab.links.anchors}")
    assert committed_slab.title == "Frame Dynamics: Salience Estimation and EWA Smoothing"
    assert ANCHOR_DRAFT_ID in committed_slab.links.anchors, (
        f"slab.links.anchors missing the anchor id: {committed_slab.links.anchors}"
    )
    for needle in ("alpha", "EWA", "re_detect_salience", "re_hit_boost"):
        assert needle in committed_slab.canonical_text, f"slab.canonical_text missing {needle!r}"
    print("Slab canonical_text acceptance test: all 4 needles present OK")

    # Contamination test: the confabulated strings must NOT appear
    for poison in ("runtime_header_salience", "resonance is a", "processing focus"):
        assert poison not in committed_anchor.notes, f"anchor.notes still contains {poison!r}"
        assert poison not in committed_slab.canonical_text, (
            f"slab.canonical_text still contains {poison!r}"
        )
    print("Contamination test: confabulated strings not present OK")

    print("\n=== ALL CHECKS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
