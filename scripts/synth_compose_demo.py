"""Synthesis Step 2 demo — compose layer + optional chat-model run.

End-to-end pipeline for the synthesis feature:
  1. Mine a doc into an in-memory corpus  (or load committed corpus)
  2. Run synthesize() to get a SynthesisResult
  3. Compose the system prompt
  4. Print the prompt (and optionally run it through the chat model)

Two modes:

  --print-only   (default): just compose and print the prompt. Fast.
  --run-model    : also send the prompt + query through gemma3:12b and
                   stream the response. Slower (60-120s), shows the
                   end-to-end synthesis.

Run:
    python scripts/synth_compose_demo.py "Sovereign logic OS raw.txt" "neural side vs symbolic side"
    python scripts/synth_compose_demo.py "Sovereign logic OS raw.txt" "OLI" --run-model
    python scripts/synth_compose_demo.py "Sovereign logic OS raw.txt" "OLI" --cite
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
import uuid
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.services.convo_miner import ConversationMiner
from src.services.corpus import CorpusStore
from src.services.label_resolver import build_index, resolve_label
from src.services.synthesis import synthesize, SynthesisResult
from src.services.synthesis_compose import compose_synthesis_prompt
from src.services import ollama
from src.models.schemas import (
    Anchor, Slab, KeyBundle, Edge, AnchorMatchPolicy, BundlePayload, AnchorMeta,
)
from src.models.enums import EdgeType, SlabType


# Borrow the materialisation helper from mine_then_synth.py — same shape.
def _materialise(corpus: CorpusStore, mined: dict) -> tuple[int, int, int, int]:
    """Convert mining output into in-memory corpus entries with fuzzy
    edge label resolution. Returns counts for diagnostics."""
    proposals = mined.get("proposals", [])
    edges = mined.get("edges", [])

    label_entries: list[tuple[str, str]] = []
    n_a = n_s = n_b = 0
    for p in proposals:
        ptype = p.get("type", "")
        nid = f"mined_{ptype}_{uuid.uuid4().hex[:8]}_v1"
        if ptype == "anchor":
            phrase = (p.get("canonical_phrase") or "").strip()
            if not phrase:
                continue
            corpus.anchors[nid] = Anchor(
                id=nid, canonical_phrase=phrase,
                aliases=list(p.get("aliases") or []),
                notes=p.get("justification", ""),
                match_policy=AnchorMatchPolicy(),
                meta=AnchorMeta(),
            )
            label_entries.append((phrase, nid))
            for al in p.get("aliases") or []:
                label_entries.append((al, nid))
            n_a += 1
        elif ptype == "slab":
            title = (p.get("title") or "").strip() or f"Slab {nid}"
            ctext = (p.get("canonical_text") or "").strip()
            if not ctext:
                continue
            corpus.slabs[nid] = Slab(
                id=nid, type=SlabType.REFERENCE,
                title=title, canonical_text=ctext,
                meta=AnchorMeta(),
            )
            label_entries.append((title, nid))
            n_s += 1
        elif ptype == "bundle":
            label = (p.get("label") or "").strip() or f"Bundle {nid}"
            corpus.bundles[nid] = KeyBundle(
                id=nid,
                payload=BundlePayload(intent=[label]),
                meta=AnchorMeta(),
            )
            label_entries.append((label, nid))
            n_b += 1

    label_to_id = build_index(label_entries)
    n_e = 0
    for e in edges:
        etype_raw = (e.get("type") or "LINKS").upper()
        if etype_raw not in {"INVOKES", "SUPPORTS", "CONFLICTS", "TENSIONS",
                              "LINKS", "SEQUENCE", "PARENT_OF"}:
            etype_raw = "LINKS"
        from_label = (e.get("from") or e.get("from_label") or "").strip()
        to_label = (e.get("to") or e.get("to_label") or "").strip()
        from_id, _ = resolve_label(from_label, label_to_id)
        to_id, _ = resolve_label(to_label, label_to_id)
        if not from_id or not to_id or from_id == to_id:
            continue
        eid = f"mined_edge_{uuid.uuid4().hex[:8]}_v1"
        try:
            corpus.edges[eid] = Edge(
                id=eid, type=EdgeType(etype_raw),
                from_node=from_id, to_node=to_id,
                weight=float(e.get("confidence", 0.5)),
                confidence=float(e.get("confidence", 0.5)),
            )
            n_e += 1
        except Exception:
            continue
    return n_a, n_s, n_b, n_e


async def _stream_chat_response(system_prompt: str, user_query: str):
    """Run the chat model on (system_prompt, user_query) and stream tokens.

    No Mirror / OLI gating in this harness — that's Step 3's job.
    For demo purposes we want to see what the bare composed prompt
    produces. Production wiring (Step 3) routes through pipeline.py
    with full chat-pipeline gating active.
    """
    print("\n" + "=" * 78)
    print("MODEL RESPONSE (gemma3:12b, no Mirror gating in this demo)")
    print("=" * 78)
    print()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]
    full_response = []
    async for chunk in ollama.chat_stream(messages, temperature=0.5):
        if chunk.get("done"):
            break
        content = chunk.get("content", "")
        if content:
            full_response.append(content)
            print(content, end="", flush=True)
    print()
    print()


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("doc_path", help="Path to source doc to mine")
    parser.add_argument("query", help="User's natural-language query")
    parser.add_argument("--budget", type=int, default=10000,
                        help="Synthesis selection token budget")
    parser.add_argument("--cite", action="store_true",
                        help="Include citation appendix + inline-cite instructions")
    parser.add_argument("--run-model", action="store_true",
                        help="Also stream the chat model's response")
    parser.add_argument("--print-only", action="store_true",
                        help="Print only the composed prompt, skip the model run")
    args = parser.parse_args()

    doc_path = Path(args.doc_path)
    if not doc_path.exists():
        print(f"Doc not found: {doc_path}", file=sys.stderr)
        sys.exit(2)
    raw = doc_path.read_text(encoding="utf-8", errors="replace")

    print(f"Doc:    {doc_path}")
    print(f"Length: {len(raw)} chars")
    print(f"Query:  {args.query!r}")
    print(f"Cite:   {args.cite}")
    print()

    # ── Mine ───────────────────────────────────────────────────────
    corpus = CorpusStore()
    miner = ConversationMiner(corpus)
    print("[1/4] Mining...")
    t0 = time.perf_counter()
    mined = await miner.mine(raw, source_label=doc_path.name, min_confidence=0.4)
    mine_secs = time.perf_counter() - t0
    print(f"      {len(mined.get('proposals', []))} proposals, "
          f"{len(mined.get('edges', []))} edges in {mine_secs:.1f}s")

    print("[2/4] Materialising into corpus...")
    n_a, n_s, n_b, n_e = _materialise(corpus, mined)
    print(f"      {n_a} anchors, {n_s} slabs, {n_b} bundles, {n_e} edges")

    # ── Synthesise ─────────────────────────────────────────────────
    print(f"[3/4] Synthesising ({args.budget} token budget)...")
    t0 = time.perf_counter()
    result = await synthesize(args.query, corpus, budget_tokens=args.budget)
    synth_secs = time.perf_counter() - t0
    print(f"      done in {synth_secs:.1f}s | budget used: "
          f"{result.budget_used_tokens}/{result.budget_tokens} tokens")
    print(f"      seeds={len(result.seeds)} t1sup={len(result.tier1_supported)} "
          f"t1conf={len(result.tier1_conflicts)} t2sup={len(result.tier2_supports)} "
          f"t2div={len(result.tier2_divergent)}")

    # ── Compose ────────────────────────────────────────────────────
    print(f"[4/4] Composing prompt...")
    prompt = compose_synthesis_prompt(result, include_citations=args.cite)
    print(f"      prompt length: {len(prompt)} chars (~{len(prompt) // 4} tokens)")
    print()

    print("=" * 78)
    print("COMPOSED SYSTEM PROMPT")
    print("=" * 78)
    print(prompt)

    # ── Optional model run ────────────────────────────────────────
    if args.run_model and not args.print_only:
        await _stream_chat_response(prompt, args.query)


if __name__ == "__main__":
    asyncio.run(main())
