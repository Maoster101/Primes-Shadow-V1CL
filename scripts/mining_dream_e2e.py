"""End-to-end timing test: mine + dream against a fixture.

Simulates the full user flow: mining a document into proposals,
pushing them as DraftPackets, then dreaming them. Measures wallclock
for each phase + the totalised E2E latency the user actually
experiences from "click Mine" to "all drafts dreamed."

Run:
    python scripts/mining_dream_e2e.py                                  # vin diesel skit
    FIXTURE=tests/fixtures/mirror_paper_excerpt.txt python scripts/mining_dream_e2e.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
import uuid
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.services.narrative_miner import NarrativeMiner
from src.services.corpus import CorpusStore
from src.services.session_store import SessionStore
from src.services.chat_store import ChatStore
from src.services.dreaming import dream_all_pending
from src.services import ollama
from src.models.schemas import DraftPacket, DraftStack, ChatMessage
from src.models.enums import DraftStatus


def _proposals_to_drafts(
    proposals: list[dict],
    session_id: str,
    chat_id: str,
    session_store: SessionStore,
):
    """Replicate /push-mined's proposal→DraftPacket conversion in-process.

    Same shape as src/api/mining.py::push_mined_proposals so the dream
    pass sees the same packet structure it'd get from the real API path.
    """
    stack = DraftStack(session_id=session_id)
    created = []
    for prop in proposals:
        ptype = prop.get("type", "anchor")
        text = prop.get("canonical_phrase") or prop.get("canonical_text", "") or prop.get("title", "")
        if not text:
            continue
        draft_id = f"mined_{ptype}_{uuid.uuid4().hex[:8]}_v1"
        inline_anchor = None
        inline_slab = None
        if ptype == "anchor":
            inline_anchor = {
                "id": draft_id,
                "canonical_phrase": prop.get("canonical_phrase", "") or "",
                "aliases": list(prop.get("aliases") or []),
                "invokes": [],
                "notes": prop.get("justification", "") or "",
            }
        elif ptype == "slab":
            inline_slab = {
                "id": draft_id,
                "title": prop.get("title") or prop.get("canonical_phrase") or "",
                "canonical_text": prop.get("canonical_text", "") or "",
                "links": {"anchors": [], "bundles": []},
                "version": "v1",
            }
        elif ptype == "bundle":
            # Bundles aren't dreamed today; skip but track count
            continue
        packet = DraftPacket(
            id=draft_id,
            packet_type=ptype,
            source_chat_id=chat_id,
            source_turns=[0],
            proposed_nodes=[draft_id],
            justification=prop.get("justification", ""),
            confidence=prop.get("confidence", 0.5),
            status=DraftStatus.DRAFT_UNAUTHORIZED,
            anchor=inline_anchor,
            slab=inline_slab,
        )
        session_store.save_draft_packet(session_id, packet)
        # Store raw proposal so dreaming has the original for fallback
        raw_path = session_store.drafts_dir(session_id) / f"{draft_id}_raw.json"
        session_store.write_json(raw_path, prop)
        stack.packets.append(draft_id)
        created.append(packet)
    session_store.save_draft_stack(session_id, stack)
    return created


async def main():
    fixture = Path(os.environ.get("FIXTURE", "tests/fixtures/vin_diesel_skit.txt"))
    if not fixture.exists():
        print(f"Fixture not found: {fixture}", file=sys.stderr)
        sys.exit(2)

    raw = fixture.read_text(encoding="utf-8")

    # ── Setup: synthetic chat + session ─────────────────────────────
    # Dreaming retrieves source_turns from the chat referenced in each
    # DraftPacket. We create a minimal chat with the fixture as the
    # initial user message so the audit has something real to ground
    # against (matches the production /push-mined → background dream
    # flow shape).
    corpus = CorpusStore()
    session_store = SessionStore()
    chat_store = ChatStore()

    chat = chat_store.create_chat(
        title=f"E2E test ({fixture.name})",
        collection_id="default",
    )
    chat_id = chat.id
    chat_store.append_message(chat_id, ChatMessage(role="user", content=raw, turn=0))
    session_id = session_store.create_session(chat_id)

    print(f"Fixture:  {fixture}")
    print(f"Length:   {len(raw)} chars")
    print(f"Model:    {ollama.CHAT_MODEL}")
    print(f"Dream model: {os.environ.get('PS_DREAM_MODEL', 'llama3.2:latest (default)')}")
    print(f"Extract num_ctx:    {ollama._EXTRACT_NUM_CTX}")
    print(f"Mining concurrency: {os.environ.get('PS_MINING_PARALLEL', '2 (default)')}")
    print(f"Dream concurrency:  {os.environ.get('PS_DREAMING_PARALLEL', '2 (default)')}")
    print(f"Synthetic chat:     {chat_id}")
    print(f"Synthetic session:  {session_id}")
    print()

    # ── Phase 1: mine ───────────────────────────────────────────────
    print(f"[mine] Starting NarrativeMiner.mine()...")
    miner = NarrativeMiner(corpus)
    t_mine_start = time.perf_counter()
    mine_result = await miner.mine(raw, source_label=fixture.name)
    mine_elapsed = time.perf_counter() - t_mine_start
    print(f"[mine] Done in {mine_elapsed:.1f}s")

    proposals = mine_result.get("proposals", [])
    by_type: dict[str, int] = {}
    for p in proposals:
        by_type[p.get("type", "?")] = by_type.get(p.get("type", "?"), 0) + 1
    print(f"[mine] {len(proposals)} proposals: {by_type}")
    print(f"[mine] {len(mine_result.get('edges', []))} edges")
    print()

    # ── Phase 2: push to drafts ────────────────────────────────────
    print(f"[push] Converting proposals to DraftPackets...")
    t_push_start = time.perf_counter()
    created = _proposals_to_drafts(proposals, session_id, chat_id, session_store)
    push_elapsed = time.perf_counter() - t_push_start
    n_anchor = sum(1 for p in created if p.packet_type == "anchor")
    n_slab = sum(1 for p in created if p.packet_type == "slab")
    print(f"[push] {len(created)} drafts created in {push_elapsed:.2f}s "
          f"({n_anchor} anchors + {n_slab} slabs; bundles aren't dreamed)")
    print()

    # ── Phase 3: dream all pending ─────────────────────────────────
    print(f"[dream] Starting dream_all_pending() — concurrency={os.environ.get('PS_DREAMING_PARALLEL', '2')}...")
    t_dream_start = time.perf_counter()
    dream_results = await dream_all_pending(
        corpus, session_store, chat_store, session_id,
    )
    dream_elapsed = time.perf_counter() - t_dream_start
    by_status: dict[str, int] = {}
    for r in dream_results:
        by_status[r.get("status", "?")] = by_status.get(r.get("status", "?"), 0) + 1
    print(f"[dream] {len(dream_results)} drafts processed in {dream_elapsed:.1f}s")
    print(f"[dream] Statuses: {by_status}")
    if dream_results:
        avg = dream_elapsed / max(1, len(dream_results))
        print(f"[dream] Avg per-draft: {avg:.1f}s wallclock "
              f"(serial-equivalent: {avg * len(dream_results):.1f}s)")
    print()

    # ── Phase 4: summary ───────────────────────────────────────────
    total = mine_elapsed + push_elapsed + dream_elapsed
    print(f"=== E2E TIMING SUMMARY ===")
    print(f"  mine:  {mine_elapsed:>6.1f}s")
    print(f"  push:  {push_elapsed:>6.2f}s")
    print(f"  dream: {dream_elapsed:>6.1f}s")
    print(f"  ----")
    print(f"  total: {total:>6.1f}s for {len(created)} drafts dreamed")
    print()

    # ── Phase 5: spot-check audit + rewrite quality ────────────────
    rewritten = sum(1 for r in dream_results if r.get("status") == "enriched")
    grounded = sum(1 for r in dream_results if r.get("status") == "already_grounded")
    errored = sum(1 for r in dream_results if r.get("status") == "error")
    skipped = sum(1 for r in dream_results if r.get("status") == "skipped")
    print(f"=== DREAM VERDICT BREAKDOWN ===")
    print(f"  already_grounded (audit clean, no rewrite): {grounded}")
    print(f"  enriched (rewrite produced):                {rewritten}")
    print(f"  errored:                                    {errored}")
    print(f"  skipped (already enriched):                 {skipped}")
    print()

    if dream_results:
        # Show first audit verdict + ungrounded_references for spot check
        first_with_audit = next(
            (r for r in dream_results if r.get("audit")),
            None,
        )
        if first_with_audit:
            audit = first_with_audit["audit"]
            ungrounded = audit.get("ungrounded_references") or []
            print(f"=== SAMPLE AUDIT (draft_id={first_with_audit.get('draft_id')}) ===")
            print(f"  verdict: {audit.get('verdict')}")
            print(f"  rewrite_needed: {audit.get('rewrite_needed')}")
            print(f"  grounded_claims: {len(audit.get('grounded_claims') or [])}")
            print(f"  confabulated_claims: {len(audit.get('confabulated_claims') or [])}")
            print(f"  ungrounded_references: {len(ungrounded)}")
            if ungrounded:
                for ref in ungrounded[:5]:
                    print(f"    - {ref!r}")
            redundant = audit.get("redundant_with") or []
            if redundant:
                print(f"  redundant_with: {redundant}")
            print(f"  notes: {audit.get('notes', '')[:200]}")


if __name__ == "__main__":
    asyncio.run(main())
