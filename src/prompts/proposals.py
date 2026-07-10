"""Structured extraction prompts for tentative anchor/slab proposals
and existing-to-existing relationship mining.

Conservative by design: only propose load-bearing concepts, not every
passing mention. The corpus must stay sparse and authoritative.

Phase 3: prompts emit RELATIONSHIPS between concepts. The output
format is a wrapper object with ``proposals`` (nodes) and ``edges``
(relations). The parser in draft_manager tolerates the legacy
bare-array format too, so if the model regresses to the old schema,
nothing breaks — edges just silently won't be captured for that turn.

Phase 4: existing-to-existing relationship miner (RELATIONSHIP_MINING_PROMPT).
Runs as a second pass after the main extractor. Its job is different —
not "extract new concepts" but "detect latent relationships between
nodes that already exist in the corpus." The LLM is shown a label index
(anchors + slab titles + bundle-intent synthesized labels) and told both
endpoints MUST be from that list. The index is the top-K nodes most
semantically related to the conversation (DraftManager._semantic_catalogs),
not the whole corpus — a full dump overflows the extraction context and
buries the relevant candidates. This surfaces connections the main miner
under-proposes because the main miner is oriented toward novelty, not
cross-reference.
"""

# Shared edge-type definitions so all three prompts stay in sync.
_EDGE_TYPES_BLOCK = """\
  - INVOKES: A summons or activates B (structural dependency, e.g. an anchor points to the slab it describes).
  - SUPPORTS: A provides evidence, grounding, or justification for B.
  - CONFLICTS: A and B make incompatible claims. The speaker positions one as wrong/incompatible with the other. Both endpoints must be preserved — seeing one without the other produces biased reasoning.
  - TENSIONS: A and B counterbalance each other but BOTH remain valid. Use this for dialectical pairs and tradeoffs where neither side cancels the other (e.g. mercy and justice, creative breadth and rigorous depth, user agency and system safety).
  - LINKS: soft association — A and B are related but no stronger claim can be made.
  - SEQUENCE: A precedes B in a narrative or causal chain (story order, derivation step, temporal priority).
  - PARENT_OF: A is a container/bundle whose scope includes B (hierarchical composition).

Choosing CONFLICTS vs TENSIONS vs LINKS — pick the STRONGEST that fits; do NOT default to LINKS:
  - Opposed AND one is wrong / rejected / incompatible ("X is wrong, Y is right", "X rules out Y") → CONFLICTS.
  - Opposed but BOTH valid, to be balanced or traded off ("pushes against Y but both must coexist", "tension between X and Y", "balance X against Y") → TENSIONS.
  - LINKS is ONLY for weak, unstated association. It is NOT a safe fallback when the speaker has stated a real opposition.
Do NOT downgrade a stated opposition to LINKS just because you can imagine a way to reconcile the two ideas (e.g. reading one as merely an extension of the other). Honor the speaker's framing: if they say two things are in tension or conflict, encode that as TENSIONS/CONFLICTS even if you personally see a reconciliation."""


PROPOSAL_EXTRACTION_PROMPT = """\
You are a semantic extraction engine for a knowledge corpus. Analyze the \
recent conversation below and identify at most 1-2 genuinely LOAD-BEARING \
concepts that deserve to become permanent semantic objects.

Rules:
- Only propose things that are clearly novel and reused across multiple turns.
- Do NOT propose every passing mention, casual reference, or fleeting idea.
- A slab = durable canonical knowledge (a definition, principle, or stable frame). \
  Only propose a slab if the concept is load-bearing and would be referenced again.
- An anchor = a persistent invocation handle (a named concept with aliases). \
  Only propose an anchor if the concept was named, invoked, or referenced repeatedly.
- Tag each proposal: FACT (needs external verification), INFERENCE (model logical step), \
  HYPOTHESIS (conjecture), or UNKNOWN.
- If nothing in the conversation is genuinely novel or load-bearing, return \
  {"proposals": [], "edges": []}.

FOIL / OPPOSING ANCHORS (when applicable):
When the speaker contrasts THEIR concept with an opposing force, default behaviour, \
or characterisation they're arguing against, extract BOTH SIDES as anchors. The \
speaker's concept AND the foil are equally valid corpus hooks — without both in \
the corpus, downstream CONFLICTS / TENSIONS edges have nothing to connect.

Examples:
- "fighting a statistical war against the Internet Average" → both `statistical war` \
  AND `Internet Average` (the opposing force the speaker is positioning against)
- "Neural Gravity pulls AI away from my Private Logic" → both `Neural Gravity` AND \
  `Private Logic`, then a CONFLICTS edge between them
- "mercy and justice both have moral weight" → both anchors, then a TENSIONS edge

Foil anchors don't need to be coined — characterisations like "Probabilistic Engine", \
"the average user", "Standard English" are valid foils when they appear as something \
the speaker positions against. Single-side extraction loses the dialectic structure \
the corpus is designed to capture.

Then, if you've extracted a foil pair (or if the conversation explicitly contrasts \
two concepts), emit the corresponding CONFLICTS / TENSIONS edge between them.

RELATIONSHIPS (Phase 3): if the conversation asserts a clear relationship \
between two concepts, also emit edges connecting them. Use concept labels \
(not IDs) — the system will resolve them. Valid endpoints:
  - Proposals from this turn's own proposals list, OR
  - Existing corpus anchors (from the list below), OR
  - Existing corpus slabs (from the list below, referenced by their title), OR
  - Existing corpus bundles (from the list below, referenced by their label), OR
  - Any concept explicitly named in the conversation.

IMPORTANT: Relationships between TWO EXISTING nodes (no new proposal \
needed) are valuable and often missed. If the conversation connects two \
things that already exist in the corpus — e.g. "X supports Y" or "these \
are in sequence" — emit that edge even if you aren't proposing either X \
or Y as new.

CRITICAL — collection names are NOT valid edge endpoints. Edges connect \
nodes (anchors / slabs / bundles), not collections (which are containers \
of nodes). The following are COLLECTION IDs, never use them as \
from_label or to_label:
$ACTIVE_COLLECTIONS
If the user says something like "v5 conflicts with v6" or "this \
collection supports that one," find the specific slab TITLE or anchor \
PHRASE inside those collections that the claim is actually about, and \
use that as the endpoint. Picking a specific slab is always better \
than picking a collection name.

Edge types:
$EDGE_TYPES

Only emit edges that are explicitly asserted or strongly implied. No speculation.

Existing corpus anchors (do NOT re-propose these, but you MAY reference \
them as edge endpoints by their canonical phrase):
$EXISTING_ANCHORS

Existing corpus slabs (reference by title, same rule — no re-proposal):
$EXISTING_SLABS

Existing corpus bundles (reference by label — descriptive intent summary):
$EXISTING_BUNDLES

Return ONLY raw JSON — an object with `proposals` and `edges` arrays:
{
  "proposals": [
    {"type": "anchor" or "slab",
     "canonical_phrase": "short name" (for anchors),
     "canonical_text": "full text" (for slabs),
     "aliases": ["alt name", ...],
     "justification": "why this is load-bearing",
     "source_turns": [int, ...],
     "claim_tag": "FACT" or "INFERENCE" or "HYPOTHESIS" or "UNKNOWN"}
  ],
  "edges": [
    {"type": "INVOKES|SUPPORTS|CONFLICTS|TENSIONS|LINKS|SEQUENCE|PARENT_OF",
     "from_label": "concept name A",
     "to_label": "concept name B",
     "confidence": 0.0-1.0,
     "justification": "why this relationship is asserted"}
  ]
}

Recent conversation:
$CONVERSATION

Proposals: """.replace("$EDGE_TYPES", _EDGE_TYPES_BLOCK)


PROPOSAL_EXPLICIT_PROMPT = """\
The user has explicitly asked to save or anchor a concept from the conversation. \
Extract the specific concept they are referring to.

If the user's request implies a relationship to other concepts in the \
conversation (e.g. "save this as a slab that supports X", "anchor this — \
it conflicts with Y", "this comes after Z in the narrative"), also emit \
edges linking them. Use concept labels (not IDs). Valid endpoints are:
  - The proposal you're creating now, OR
  - Any existing corpus anchor/slab/bundle from the lists below, OR
  - Any concept named in the conversation.

CRITICAL — these are COLLECTION IDs, never use them as endpoints:
$ACTIVE_COLLECTIONS
Edges connect nodes, not collections. Find the specific slab TITLE or \
anchor PHRASE that the claim is about.

Edge types:
$EDGE_TYPES

Existing corpus anchors (reference by canonical phrase):
$EXISTING_ANCHORS

Existing corpus slabs (reference by title):
$EXISTING_SLABS

Existing corpus bundles (reference by label):
$EXISTING_BUNDLES

Return ONLY raw JSON — an object with `proposals` (a single-element array) \
and `edges` arrays:
{
  "proposals": [
    {"type": "anchor" or "slab",
     "canonical_phrase": "short name" (for anchors),
     "canonical_text": "full text" (for slabs),
     "aliases": ["alt name", ...],
     "justification": "why the user wants this saved",
     "source_turns": [int, ...],
     "claim_tag": "FACT" or "INFERENCE" or "HYPOTHESIS" or "UNKNOWN"}
  ],
  "edges": [
    {"type": "INVOKES|SUPPORTS|CONFLICTS|TENSIONS|LINKS|SEQUENCE|PARENT_OF",
     "from_label": "concept name A",
     "to_label": "concept name B",
     "confidence": 0.0-1.0,
     "justification": "why this relationship is asserted"}
  ]
}

Recent conversation:
$CONVERSATION

User's request: """.replace("$EDGE_TYPES", _EDGE_TYPES_BLOCK)


RELATIONSHIP_MINING_PROMPT = """\
You are a RELATIONSHIP DETECTION engine — a complementary pass to the \
main concept extractor. Your job is narrow and specific: given a \
recent conversation AND a catalog of existing corpus nodes, identify \
RELATIONSHIPS between those existing nodes that the conversation \
asserts or strongly implies.

This is different from concept extraction. You do NOT propose new \
nodes. You look ONLY for connections between things that already \
exist in the catalog below. Both endpoints of every edge MUST be \
drawn from the lists provided.

Goal: flesh out the graph's relational structure. The corpus has \
many nodes, and many of them are related in ways that have never \
been formally recorded. When the conversation discusses two existing \
corpus concepts together — explaining one in terms of another, \
noting a contradiction, placing them in narrative order, showing \
that one grounds the other — that is an edge worth surfacing.

Rules:
- Only edges between existing nodes. If a concept in the conversation \
  doesn't match anything in the catalog below, skip it — that's a \
  job for the concept extractor, not you.
- Exact or near-exact label match. If the conversation mentions \
  "terra preta" and the catalog has "Terra Preta" as an anchor, use \
  the anchor's canonical phrase verbatim.
- Be conservative. Only emit edges that are explicitly asserted or \
  very strongly implied. Speculation pollutes the graph.
- If the conversation touches nothing in the catalog, or only one \
  thing, return {"edges": []}.
- A relationship that was previously mined doesn't need to be \
  re-emitted — the system dedupes on (from, to, type) — so when in \
  doubt, emit.

CRITICAL — these are COLLECTION IDs, NEVER use them as from_label or \
to_label. They are containers, not nodes. Edges must connect nodes:
$ACTIVE_COLLECTIONS
If the user says "v5 conflicts with v6" or "this collection relates \
to that one," identify which SPECIFIC slab or anchor WITHIN those \
collections the claim is actually about, and use those as endpoints. \
If you cannot identify a specific node-level referent, skip the edge \
rather than using a collection name.

Edge types:
$EDGE_TYPES

Existing corpus anchors (reference by canonical phrase):
$EXISTING_ANCHORS

Existing corpus slabs (reference by title):
$EXISTING_SLABS

Existing corpus bundles (reference by label):
$EXISTING_BUNDLES

Return ONLY raw JSON — an object with an `edges` array:
{
  "edges": [
    {"type": "INVOKES|SUPPORTS|CONFLICTS|TENSIONS|LINKS|SEQUENCE|PARENT_OF",
     "from_label": "exact label from catalog",
     "to_label": "exact label from catalog",
     "confidence": 0.0-1.0,
     "justification": "the specific conversational moment that asserts this relationship"}
  ]
}

Recent conversation:
$CONVERSATION

Edges: """.replace("$EDGE_TYPES", _EDGE_TYPES_BLOCK)
