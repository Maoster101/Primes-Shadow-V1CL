# Prime's Shadow

**A full-stack LLM desktop application implementing the Mirror architecture — persistent epistemic integrity, graph-of-graphs semantic memory, and live user-controlled salience steering.**

> *"I am the bone of my sword."*

---

## What this is

Prime's Shadow is a local AI assistant built around a novel architecture for maintaining epistemic integrity and semantic context across long-horizon LLM interactions. It is the working implementation of [Mirror](./preprint), a neuro-symbolic system documented in a technical preprint currently under external review.

Three things make it architecturally distinct from any existing LLM interface:

1. **Live user-controlled salience steering** — users directly manipulate model attention weights at runtime by interacting with the corpus graph. Double-click a node to heat it (increase salience), cool or dismiss via context menu. The model sees the updated frame header on the next turn. Dismissed nodes are not deleted — they are evicted from the active frame but remain in the corpus and can be organically re-detected if the conversation warrants it. No known prior implementation of this interaction primitive exists.

2. **Persistent graph-of-graphs semantic memory (Corpus)** — concepts detected across sessions are stored as a typed graph (Anchor → Bundle → Slab) with edge semantics (INVOKES / SUPPORTS / CONFLICTS), versioning, and cascade propagation. The corpus persists across sessions and is validated on startup. Session state is restored on chat select, including tentative nodes, edges, and corpus access patterns.

3. **Three-layer constitutional constraint injection (OLI)** — a ten-layer epistemic constraint system injected at three levels: constitutional prompt (~2,900 tokens), per-turn runtime header (~50 tokens), and code-side post-generation validator. Enforces hard-gated claim admissibility (FACT / INFERENCE / HYPOTHESIS / UNKNOWN), layer integrity bounds (L0–L4), and surfaces epistemic rigour decay in real time.

---

## Runtime Stack

| Layer | Technology | Purpose |
|---|---|---|
| LLM | GPT-OSS 20B via Ollama | Primary face model — MoE, 128k context, 3.6B active params/token |
| Embeddings | nomic-embed-text via Ollama | 768-dim embeddings for semantic X-axis positioning, anchor matching, dedup |
| Backend | Python 3.13 + FastAPI + Uvicorn | API server on port 8420 |
| Frontend | Single-page HTML + vanilla JS | Two canvas renderers with custom 3D projection + force-directed physics |
| Storage | YAML (corpus) + JSON (chats, sessions, frames) | All local, file-backed, no database |
| Web Search | DDG (default) / Google Gemini / OpenAI / Serper.dev | Configurable, zero-key fallback to DDG |

---

## Architecture

### Three-Layer OLI Injection

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Constitutional Prompt (model-side, ~2,900 tokens) │
│  Full OLI-0 to OLI-9 + LI-0 to LI-4 + BMD + Philosophy    │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: Per-Turn Runtime Header (model-side, ~50 tokens)  │
│  oli_mode, gate_state, frame_summary, drift, enforcement    │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: Post-Generation Validator (code-side)             │
│  Pattern matching against L0-L4 violation signatures        │
└─────────────────────────────────────────────────────────────┘
```

### Per-Turn Pipeline

```
User message
  → classify_message()          — Function gate (6 types)
  → anchor_matcher.match_all()  — Two-phase: string → semantic, soft * format gate
  → asyncio.gather(
      frame_manager.update_turn(),  — Activate nodes, salience, decay, corpus hits
      estimate_drift()              — Affect density, claim volatility, rigor drop
    )
  → build_runtime_header()      — Gate state + frame summary + drift for model
  → build_system_prompt()       — Constitutional + BMD + header
  → ollama.chat_stream()        — Streaming response (web search if mode=on/auto)
  → oli_validator.validate_output() — Code-side L0-L4 flag-and-surface
  → yield SSE chunks to frontend
  → background: draft_manager.extract_proposals() every 8 turns
```

---

## Corpus & Data Model

### Corpus Objects (YAML, authoritative)
- **Anchors** — invocation handles; named, characterised, callable by the * operator
- **Bundles** — structured semantic neighbourhoods grouping related concepts
- **Slabs** — heavyweight canonical content nodes (50+ line markdown); the atomic unit of committed knowledge
- **Edges** — typed: INVOKES, SUPPORTS, CONFLICTS, LINKS, PARENT_OF with weight, confidence, tension, conditions

### Runtime Objects (JSON, session-scoped)
- **FrameState** — active nodes by type, salience_now/smoothed, structural weight, decay model (0.85/turn), corpus access patterns (hit_count + last_hit_turn)
- **DraftStack** — tentative proposals, max 10/session
- **Tentative Registry** — concept detection, parent/child hierarchy, promotion state
- **Tentative Edges** — PARENT_OF between tentative nodes

### Promotion Chain
```
conversation turn
  → concept detected (ephemeral, no authority)
  → if similar to existing: PARENT_OF edge (auto-hierarchical)
  → 3+ children: events feed suggests bundling
  → user Ctrl+select → right-click → Create bundle
  → user right-click bundle → Promote to anchor
  → slab proposed ONLY on explicit request or session end
  → corpus commit requires OLI ON + FACT verification
  → auto-generates minimum coherence bundle at slab commit
```

---

## Frontend

### Layout
**Sidebar** (280px) | **Chat pane** (420px, resizable) | **Right column** (flex)

### Right Column — Dual Canvas
**Active Frame Canvas** (top) — live session graph showing 2-hop neighbourhood of active nodes. Semantic X-axis via nomic-embed-text embeddings (Creative ←→ Rigorous). Temperature/salience drives radiance glow and pulse animation.

**Cold Corpus Canvas** (bottom, tabs: Events / Cold Corpus) — independent canvas showing full persistent corpus with access pattern encoding. Hit count → node size/glow. Recency → intensity. 30-turn slow decay. Cross-canvas: corpus nodes glow when referenced in the active session.

### Node Visual Encoding — Two Orthogonal Axes

| Axis | Encodes | Visual |
|---|---|---|
| Shape | Structural type | Anchor: solid + concentric rings. Bundle: hexagon. Slab: hollow + centre dot. Concept: dashed circle. |
| Colour + Size | Salience / temperature | Cold (deep blue, small) → warm (white) → hot (amber) → fire (red-orange, large) |

Tentative nodes: yellow dashed ring overlay. Rejected: grey dashed ring.

### Interactions

| Gesture | Action |
|---|---|
| Hover | Quick tooltip: type, status, salience, bonds, hits |
| Click | Full-height slide-in inspector: deep data, slab canonical_text, bundle payloads |
| Double-click | Heat node — updates backend salience, model sees it next turn |
| Right-click | Context menu: heat / cool / inspect / bundle / promote / commit / reject / remove |
| Ctrl+click | Multi-select for manual bundle creation |
| Drag | Orbit |
| Middle-drag | Pan |
| Scroll | Zoom to cursor |

### Sidebar Controls
- OLI toggle (ON / OFF)
- Web search mode (Off / On / Auto)
- Thinking level (Off / Low / Med / High → Ollama reasoning depth)
- Search provider (DDG / Serper / Google / OpenAI / Perplexity / Brave) + API key field

---

## Prompt Architecture

**OLI OFF** (~420 tokens): BMD scaffold (5 personas + routing + example), Layer Integrity compact reference (LI-0 to LI-4), OLI layer names (reference only), tempo rules, base Mirror identity.

**OLI ON** (~2,900 tokens): Full constitutional with two clearly separated systems:
- **Layer Integrity (LI-0 to LI-4)** — conversation depth boundaries: Banter → Descriptive → Evaluative → Bounded prescriptive → Operationalisation
- **OLI v2.1 (OLI-0 to OLI-9)** — epistemic enforcement: Epistemic floor → Claim admissibility → Domain separation → Interaction style → Memory → Operators → Durability → Traceability → Interrogation → Refinement → Versioning
- **System Philosophy**: human-as-loop, absence of command = hard deny, summaries = zero authority
- **\* operator**: three functions — anchor invocation, semantic depth, stabilised analysis

**BMD (Bones Mode Distribution)**: Brennan 0.60 / Zack 0.15 / Booth 0.10 / Angela 0.10 / Hodgins 0.05. Routing adjusts contextually.

---

## Persistence

| Object | Storage | Lifetime |
|---|---|---|
| Chats | JSON files | Forever, until user archives |
| FrameState | JSON per session | Restored on chat select (includes tentative registry, edges, corpus access patterns) |
| Corpus | YAML files | Persistent, validated on startup (6 checks) |
| Events | Append-only JSONL logs | Gate, match, drift, frame, proposal, verification events |

---

## Prerequisites

- [Ollama](https://ollama.ai) with `gpt-oss:20b` and `nomic-embed-text` pulled
- Python 3.13+

```bash
ollama pull gpt-oss:20b
ollama pull nomic-embed-text
```

## Installation

```bash
git clone https://github.com/[your-handle]/primes-shadow
cd primes-shadow
pip install -r requirements.txt
python -m uvicorn src.app:app --host 0.0.0.0 --port 8420
```

Open `http://localhost:8420` in your browser.

---

## Relation to Mirror

Prime's Shadow is the working implementation of the Mirror architecture. The Mirror preprint — *"Mirror: A Neuro-Symbolic Architecture for Persistent Epistemic Integrity in LLM Interactions"* — documents the theoretical foundations. The independent convergence between Mirror's Behavioural Mode Distribution (formalised February 16, 2026) and Anthropic's Persona Selection Model paper (published February 23, 2026) is discussed in the preprint.

---

## Status

Active development. Fast-iteration phase. Not yet packaged for general distribution — local deployment only.

---

*Prime's Shadow. The saddle, not the horse.*
