# Corpus Graph-of-Graphs: Structure and Layers

Working doc. Snapshot of the corpus schema as it stands, alongside the
planned **semantic-zoom redesign** that introduces pillars as a curated
top tier sitting above the existing Anchor/Bundle/Slab graph. Planned
additions are flagged inline as `// PLANNED`. Mirror vocab assumed
throughout.

The redesign is motivated by the current graph being mechanically
correct but visually illegible at scale — too many nodes of similar
visual weight, no clear "what is this corpus about" view. The semantic-
zoom layer (§6) addresses this without demolishing the existing schema:
pillars are a persisted overlay, the underlying Anchor/Bundle/Slab
content graph is untouched.

---

## 1. Node taxonomy

Four node kinds (see `NodeType` enum). Three are structural; one is
peripheral.

### 1.1 Anchor — retrieval handle

Short, hashable, semantically distinctive. Exists to be matched against
incoming utterances via hybrid (exact + fuzzy + semantic) matching.
Anchors are the *doorbell*, not the content behind the door.

Key fields (from `Anchor`):

- `canonical_phrase` — the matchable string
- `aliases` — alternate surface forms
- `invokes` — list of node IDs activated when this anchor fires
  (this is the anchor's outbound activation list, semantically equivalent
  to an INVOKES edge but stored on the node for fast lookup)
- `notes` — short grounding text, not the full payload
- `match_policy` — gate requirements, allowed message functions,
  exact-vs-fuzzy confidence thresholds (defaults: 0.70 exact, 0.90 fuzzy)
- `lifecycle_status` — ACTIVE / DORMANT / DEPRECATED
- `depends_on`, `assumptions` — prerequisites and implicit assumptions
- `meta` — version, supersedes, semantic_change flag, created_at

### 1.2 KeyBundle — structured constraint payload

**Note on role evolution:** Bundles originally functioned as groupings of
co-occurring anchors used to trigger slab formation. From the document-
mining perspective they've taken on a second role: a structured packet
of framing constraints attached to a domain or slab cluster — the
`BundlePayload` carries `intent`, `invariants`, `non_assumptions`,
`warnings`, `heuristics`, `markers`, `rules`, `activation_clause`,
`canonical_quote_handles`, and `closing_clause`.

A bundle is the right node when the extracted content is a coordinated
*directive packet* rather than freestanding prose — i.e. when intent +
invariants + warnings + rules belong together as one unit of framing.

Key fields (from `KeyBundle` + `BundlePayload`):

- `payload.intent` — 1–5 statements of purpose
- `payload.invariants` — up to 8 things that must hold
- `payload.non_assumptions` — up to 6 things explicitly *not* assumed
- `payload.warnings` — up to 8 failure modes / pitfalls
- `payload.heuristics` — up to 8 rules of thumb
- `payload.rules`, `payload.markers`, `payload.activation_clause`
- `payload.canonical_quote_handles` — short retrieval handles into
  associated slab content
- `supports` — slabs this bundle frames

### 1.3 Slab — substantive content

The meaty content. Where actual prose, claims, evidence, beats, and
procedures live. Slabs are what gets *returned* when retrieval lands
on an anchor.

Key fields (from `Slab`):

- `canonical_text` — the prose payload
- `title` — short label
- `type` (`SlabType`) — CONSTITUTIONAL / INVARIANT / REFERENCE / CANONICAL
- `links.anchors` — anchor IDs that point into this slab
- `links.bundles` — bundles framing this slab
- `links.anchors_inline` — anchors demoted from corpus level by the
  consolidation pass (see §3.1); preserves canonical_phrase, aliases,
  notes, and ID for traceability and potential re-promotion
- `intent`, `invariants` — slab-level metadata inlined from the former
  1:1 "coherence bundle" (which was structurally redundant)
- `depends_on`, `assumptions` — prerequisites
- `provenance_refs` — where this slab came from (workbench / library,
  conversation_segment / synthesis_notes / etc.)
- `requires_oli_mode` — gating on OLI mode for retrieval
- `lifecycle_status`, `meta` — same lifecycle / versioning as anchor

### 1.4 Concept — peripheral

A fourth NodeType exists (`CONCEPT`) for nodes that don't fit the
anchor/bundle/slab roles cleanly. Used sparingly; not part of the
load-bearing spine.

---

## 2. Edge taxonomy

Edges are reified (`Edge` model). Each carries `weight`, `confidence`,
optional `tension`, optional `justification`, and an optional
`dialectic_subtype` for CONFLICTS / TENSIONS edges.

### 2.1 Weight vs. confidence — semantically distinct

This is a load-bearing distinction worth restating because the seed
corpus initially conflated them:

- **`weight`** — *signal gain*. How strongly activation cascades along
  this edge when fired. Scales the target's inherited activation. Set
  by the curator (hand-authored) or proposed by the LLM at mine time.
- **`confidence`** — *epistemic belief that the edge exists and is
  correctly typed*. 1.0 for hand-curated / CONSTITUTIONAL edges,
  lower for LLM-mined edges awaiting corroboration. Scales cascade as
  a multiplier.

Effective cascade strength = `weight × confidence × kernel_for_type`
(per-edge-type kernel from `frame_manager.CASCADE_KERNEL`).

### 2.2 Edge types (current — from `EdgeType` enum)

- **INVOKES** — A activates B when fired. The activation primitive.
  Anchor→Bundle and Anchor→Slab are the common patterns; also stored
  inline on `Anchor.invokes` for fast lookup.
- **SUPPORTS** — A provides evidence / grounding for B.
- **CONFLICTS** — direct opposition. A rejects or contradicts B.
- **TENSIONS** — productive tension. A and B counterbalance, both
  valid simultaneously (e.g. mercy ↔ justice). Distinct from CONFLICTS;
  not a resolution-pending state but a permanent dialectical pair.
- **LINKS** — associative catch-all. Softer than INVOKES, doesn't
  imply activation cascade.
- **PARENT_OF** — structural containment.
- **SEQUENCE** — narrative ordering; A precedes B in the story spine.
  Used heavily by the narrative miner.

### 2.3 Dialectic subtype (CONFLICTS / TENSIONS refinement)

Optional refinement of CONFLICTS / TENSIONS edges, recording the
mechanism that produced them. From `DialecticSubtype`:

Document / static patterns:
- **OPPOSITION** — source positions one side as wrong
- **COUNTERBALANCE** — source holds both valid (e.g. mercy ↔ justice)

Temporal / conversational patterns (live mining):
- **DRIFT** — position evolved across turns
- **RETRACTION** — position explicitly withdrawn
- **LIVE_CORRECTION** — mid-conversation rephrasing without retraction

Subtype is None for non-dialectic edges and for pre-live-mining edges
where the mechanism wasn't recorded.

### 2.4 Edge auxiliary fields

- `tension` — optional scalar, conflict intensity for CONFLICTS edges
- `justification` — free-text rationale, produced at mining time;
  surfaced to synthesis alongside CONFLICTS / TENSIONS edges so the
  LLM sees the original reasoning, not just the bare edge
- `source_turn` — turn that triggered the edge (for live-mined edges)
- `conditions.allowed_functions` — gating by message function

---

## 3. Lifecycle and consolidation passes

The corpus is not produced in one shot. Nodes and edges move through
several stages of refinement.

### 3.1 Anchor consolidation — demotion to inline

Anchors that fail the **cross-slab-reach criterion** — touched by fewer
than 2 slabs AND not referenced in any multi-slab bundle — get demoted
from `corpus.anchors` into the parent slab's `links.anchors_inline`
list (`InlineAnchor` records).

Rationale: an anchor that only ever fires inside one slab isn't doing
retrieval work that justifies a corpus-level node. It's just metadata
on that slab. Inline storage preserves the text content (canonical
phrase, aliases, notes) and the original ID, so re-promotion is
possible later if another doc introduces a second touching slab.

### 3.2 Slab coherence inlining

Slab-level `intent` and `invariants` used to live in a 1:1 "coherence
bundle" node. That bundle was structurally redundant — same content,
derived only from one slab, no cross-slab role. The fields are now
inlined on the slab directly. Pre-migration slabs default to empty
lists for backwards compatibility.

The pattern here matches §3.1: when a node is doing no cross-structure
work, it should be inlined rather than maintained as a separate node.

### 3.3 Live mining trajectory canonicalization

Drafts produced during a live conversation carry an `epistemic_status`
(from `EpistemicStatus` enum): NEWLY_RAISED → CONFIRMED / DRIFTED /
RETRACTED / UNTOUCHED → CANONICALIZED.

End-of-conversation pass walks the ghost stack, looks at each draft's
`referenced_in_turns`, `salience_trajectory`, and `superseded_by`, and
classifies. Drift and retraction edges (with appropriate
`dialectic_subtype`) get written. Confirmed drafts promote to corpus;
untouched drafts drop unless salience trajectory justifies preservation.

---

## 4. Runtime / session layer (FrameState)

Separate from corpus storage. Tracks the *current attentional state*
during a live conversation. Reset per session, restored on chat select.

Key dimensions tracked per node:

- `active_anchors` / `active_bundles` / `active_slabs` — weight maps
- `salience_now` — instantaneous heat
- `salience_smoothed` — EWA-smoothed heat (more stable for rendering)
- `structural_weight` — currently a runtime dimension for conversation
  scoring; not yet derived from corpus topology
  // PLANNED: once PillarDefinitions exist, this field can read
  // structural prominence directly from pillar membership rather
  // than recomputing — a node is structurally weighted if it is a
  // pillar member or itself a pillar (see §6)
- `truth_pressure` — per-node epistemic tension accumulator;
  incremented when claims are questioned. High pressure triggers
  targeted gauntlet checks even when global mismatch is low.
- `corpus_hits` — total times each corpus node has been pulled into
  this session
- `corpus_last_hit`, `corpus_hit_log` — for session-trajectory replay

Decay model: `FrameDecay(per_turn=0.85, floor=0.10)` — geometric
decay each turn, never below the floor (so nodes stay weakly
present rather than vanishing entirely).

Distinction worth keeping clear:
- **Salience** — temporal, observer-relative, decays. Live-conversation
  attention state.
- **Corpus hits** — session-cumulative access count. Wear, not heat.
- **Structural weight** — currently runtime; planned to be backed by
  corpus topology (see §6).

---

## 5. Mining and ingestion pipeline

`Mine → Dream → Review → Commit`, with miner strategy selected per
ingestion source.

### 5.1 Current miners

- **ConversationMiner** — live-chat trajectory mining, emits drafts
  with epistemic status tracking
- **NarrativeMiner** — paragraph-based segmentation, SEQUENCE edges
  between consecutive slabs, condensation / reconstitution to
  corpus-appropriate density

### 5.2 Grounding modes (`DomainMode`)

- **INTERNAL** — grounds against existing corpus and conversation
- **EXTERNAL** — grounds against external source (code, document, etc.)

Confabulated content is detected during the Dream pass and rewritten
before human Review.

### 5.3 Planned miners

// PLANNED — see §6.6 for the genre-specific miner split (paper_miner,
// docs_miner, reference_miner). Miner selection is driven by an
// ingestion-time genre hint, and the same hint feeds into the
// tier-construction pass's per-genre weighting.

---

## 6. Planned redesign — semantic zoom and pillars

The corpus already supports semantic zoom: `static/index.html` carries
a drill stack, wheel-triggered drill in/out, and cross-fade transitions
that walk a hierarchical cluster tree. `_loadClusterView()` synthesizes
cluster meta-nodes (`_isCluster: true`, `_clusterPath`, `_clusterMembers`)
which share the `cNodes` array with real anchors/slabs — the renderer
doesn't distinguish them, semantics live in metadata flags. Top zoom
shows cluster meta-graph; drilling into a leaf cluster reveals members.

What's currently sitting at the top zoom level for any corpus is an
**algorithmically derived partition** — useful for arbitrary content
but generic. It tells you "these 47 nodes cluster together by similarity"
rather than "this paper argues X via three pillars." For text-mined
corpora we want the top zoom to be **semantically curated**, not
algorithmically derived.

### 6.1 Pillars as persisted cluster meta-nodes

The core insight: a pillar is structurally the same shape as the cluster
meta-nodes the view layer already synthesizes — a container with a
label, members, optional children, hasChildren flag. The difference is
*provenance*: synthesized clusters are computed at view time from a
backend partition; pillars are **persisted, curated, and authored
during ingestion**.

This means:

- No new rendering work — pillars present to the canvas as cluster
  meta-nodes via the existing `_isCluster` machinery
- No new drill stack work — `_drillStack` already walks arbitrary-depth
  zoom hierarchies
- No demolition of the Anchor/Bundle/Slab content graph — pillars sit
  *alongside* it as a navigational overlay
- Mixed corpora are handled naturally — render persisted pillars where
  they exist, fall back to algorithmic clustering elsewhere

### 6.2 PillarDefinition — the persisted overlay

// PLANNED schema addition — sits alongside Anchor/Bundle/Slab,
// does not modify them

```python
class PillarDefinition(BaseModel):
    id: str
    label: str                    # short navigable headline
    summary: str                  # 1–3 sentences shown at top zoom
    members: list[str]            # node IDs at the next tier down
    children: list[str] = []      # nested PillarDefinition IDs for
                                  # multi-level hierarchy (paper >
                                  # section > sub-claim)
    cross_edges: list[dict] = []  # lifted pillar-to-pillar relations
                                  # (see §6.4)
    origin: str                   # source document / ingestion event
    pillar_role: Optional[str]    # genre-dependent: claim / beat /
                                  # concept / section / etc.
    meta: AnchorMeta              # versioning, supersedes, etc.
```

Pillars reference existing nodes by ID via `members` and `children`.
They never store content directly — the slab/bundle/anchor stays the
source of truth. A pillar's summary is a *distillation* generated at
construction time, persisted because regenerating it from members on
every view would be expensive.

Pillars are scarce by design: paper ~5–10, narrative ~3–7 major beats,
docs ~5–15. Sub-pillars (one nest level deeper) may exist for longer
documents where a single tier of pillars would be too dense.

### 6.3 Tier-construction pass — building the overlay

// PLANNED pipeline stage — fits between Commit and live runtime,
// re-runnable

Runs over the committed corpus graph for a given ingestion scope:

1. Identify load-bearing clusters of Anchor/Bundle/Slab nodes
   - INVOKES in-degree weighted by invoker distinctness
   - SUPPORTS in-degree (evidence convergence)
   - SEQUENCE centrality for narrative (downstream chain length)
   - Existing "anchor referenced by ≥2 slabs" heuristic as the cheap
     baseline signal
2. Apply genre-specific weighting based on ingestion origin
3. For each load-bearing cluster, create a PillarDefinition:
   - members = node IDs in the cluster
   - summary = LLM call grounded against cluster content
   - label = short title (LLM or extracted from highest-importance
     anchor's canonical_phrase)
4. Lift cross-cluster edges to pillar-level cross_edges (§6.4)
5. Optionally nest pillars into a multi-tier hierarchy if cluster
   sizes warrant it

Pass is **idempotent** and **non-destructive**. Re-running it produces
a fresh set of PillarDefinitions; old ones get superseded (`meta.
supersedes`). The Anchor/Bundle/Slab graph is untouched at every step.
Pillar definitions can be thrown out and regenerated with different
heuristics without risk to content.

### 6.4 Edge lifting — cross-pillar relations

When two slabs in different pillar-clusters share an edge, the edge
may *lift* to a pillar-level cross_edge. Lifting rules differ by edge
type:

- **SUPPORTS** — lifts naturally. Multiple underlying SUPPORTS edges
  between clusters → pillar-level SUPPORTS with aggregate weight.
- **SEQUENCE** — lifts naturally when pillars correspond to ordered
  units (story acts, doc procedure phases).
- **CONFLICTS / TENSIONS** — higher lift threshold. One localised
  disagreement doesn't make two pillars conflict; multiple underlying
  conflicts (or one near a pillar's core claim) lifts to pillar-level.
- **INVOKES** — generally does *not* lift. Activation cascade is a
  runtime concern; pillar-level navigation shouldn't be cluttered with
  it.
- **LINKS** — does not lift. Too weak to carry pillar-level meaning.

Lifting is **lossy by design** — Tier-0 is meant to be a compression.
The lifting rule is roughly "aggregate weight of underlying Tier-1
edges, normalised by cluster size, gated by an edge-type-specific
threshold."

### 6.5 Dispatch — when to render pillars

// PLANNED extension to `loadCorpusGraph()` dispatch

Current dispatch (paraphrased from `static/index.html:4815-4831`):

```
clusters mode + empty stack       → top-level cluster meta-graph
clusters mode + non-leaf top      → sub-cluster meta-graph at path
clusters mode + leaf top          → leaf member nodes (drilled)
nodes mode                        → full /corpus/full
```

New cases:

```
clusters mode + corpus has pillars + empty stack
                                  → render pillar meta-graph
clusters mode + drilled into pillar (non-leaf)
                                  → render pillar's members
                                    (slabs/bundles/sub-pillars)
clusters mode + drilled into sub-pillar
                                  → recurse as above
```

Pillars and algorithmic clusters can coexist within a single corpus.
A pillar-rich subgraph (e.g. a recently ingested paper) renders by
pillars; a pillar-less subgraph (e.g. accumulated conversation drafts)
falls back to algorithmic clustering. The drill stack doesn't care
which produced the meta-nodes.

### 6.6 Genre-specific miners (still planned)

// PLANNED — text-decomposition miner split:
// - `paper_miner` — section-aware chunking, citation-grounded,
//   optimised for claim / evidence / rebuttal extraction
// - `docs_miner` — procedure-aware chunking, code-grounded,
//   optimised for concept / step / pitfall extraction
// - `reference_miner` (later) — for flat reference material
//   (specs, glossaries) that has no thesis or arc

All produce the same Anchor/Bundle/Slab node and edge types. Genre
hint at ingestion determines miner selection AND drives the per-genre
weighting and cap in the tier-construction pass (§6.3).

### 6.7 Minimum viable slice for validation

Smallest end-to-end build that proves the pillar layer works:

1. Add `PillarDefinition` to schemas as a pure data class
2. Hand-author or LLM-stub ~5–10 PillarDefinitions for one paper-mined
   corpus
3. Modify `_loadClusterView` dispatch to consult PillarDefinitions
   when present, synthesize cluster meta-nodes from them
4. Render — verify that walking the pillar tree at top zoom reads
   like the paper's outline

Step 3 is the only real code change. If step 4 *feels right* the
redesign is validated and the proper miner-driven tier-construction
pass becomes worth building. If it doesn't, minimal effort spent and
concrete failure modes to learn from.

---

## 7. Open questions / known soft spots

- **Re-running the tier-construction pass.** When does it re-run?
  Per-commit is expensive on a large corpus; scheduled risks staleness;
  on-demand requires explicit trigger. Probably scheduled + on-demand,
  with per-commit deferred via a dirty-flag mechanism per ingestion
  scope.
- **Pillar regeneration vs. user edits.** If a user manually adjusts
  a pillar's label or summary, does re-running the construction pass
  overwrite their edits? Likely need a `user_curated` flag that
  protects manual changes from regeneration.
- **`structural_weight` in FrameState** — currently runtime-only. With
  pillars persisted, structural importance is now a queryable corpus
  property rather than a derived score, so the runtime field can read
  directly from PillarDefinition membership rather than recomputing.
- **Bundles in mined content** — the dual role (conversation co-
  occurrence trigger vs. document constraint packet) remains
  unresolved at the schema level. May need either two bundle subtypes
  or origin-based interpretation. Watch as more document corpora land.
- **Edge-lifting thresholds** — the cutoffs for when underlying edges
  lift to pillar-level are currently sketched but not calibrated.
  Wants empirical tuning against a few sample corpora before being
  fixed.
- **Reference material (specs, glossaries)** — flat content with no
  thesis or arc. May not benefit from pillar overlay at all; the
  algorithmic clustering fallback is probably correct for these.
  Watch for failure cases.
- **Cross-corpus pillars.** A concept that appears as a pillar in two
  separately ingested papers — same pillar or two? Likely two with a
  LINKS edge between them; merging risks losing the per-source
  context.

---

## 8. Quick-reference cheat sheet

```
NODE TYPES (content tier — Tier-1 / Tier-2)
  anchor      — retrieval handle (terse, hashable, match target)
  key_bundle  — structured constraint payload (intent/invariants/...)
  slab        — substantive content payload
  concept     — peripheral, sparse use

NODE TYPES (// PLANNED navigational overlay — Tier-0)
  pillar_definition  — persisted cluster meta-node; references
                       content nodes via members + children; carries
                       label, summary, cross_edges

EDGE TYPES
  INVOKES     — activation cascade (also inline on Anchor.invokes)
  SUPPORTS    — A is evidence for B
  CONFLICTS   — A rejects B (with dialectic_subtype)
  TENSIONS    — A and B counterbalance (with dialectic_subtype)
  LINKS       — associative, no cascade
  PARENT_OF   — structural containment
  SEQUENCE    — narrative ordering (heavily used by narrative miner)

EDGE WEIGHTS
  weight      — signal gain on cascade
  confidence  — epistemic belief edge exists and is typed correctly
  effective   — weight × confidence × type_kernel

LIFECYCLE
  ACTIVE / DORMANT / DEPRECATED

CONSOLIDATION (current)
  demote anchors with <2 slab reach → inline
  inline 1:1 coherence-bundle content → slab fields

TIER CONSTRUCTION (// PLANNED)
  identify load-bearing clusters in content graph
  → persist PillarDefinitions referencing those clusters
  → lift cross-cluster edges to pillar-level cross_edges
  → render via existing cluster meta-node machinery

SEMANTIC ZOOM (existing infra, extended by pillars)
  Tier-0  pillar meta-graph (// PLANNED, persisted)
          or algorithmic cluster meta-graph (current fallback)
  Tier-1  slabs / bundles within a drilled pillar/cluster
  Tier-2  anchors / inline machinery within a drilled slab
  drill stack walks tiers; renderer is tier-agnostic

RUNTIME (FrameState, per-session)
  salience_now / salience_smoothed (decay 0.85/turn, floor 0.10)
  corpus_hits / corpus_last_hit (session wear)
  truth_pressure (per-node epistemic tension)
  structural_weight (// PLANNED: read from PillarDefinition
                     membership rather than recomputed)
```
