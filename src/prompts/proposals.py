"""Structured extraction prompts for tentative anchor/slab proposals.

Conservative by design: only propose load-bearing concepts, not every
passing mention. The corpus must stay sparse and authoritative.

Phase 3: prompts now also emit RELATIONSHIPS between concepts. The output
format is a wrapper object with `proposals` (nodes) and `edges` (relations).
The parser in draft_manager tolerates the legacy bare-array format too, so
if the model regresses to the old schema, nothing breaks — edges just
silently won't be captured for that turn.
"""

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

RELATIONSHIPS (Phase 3): if the conversation asserts a clear relationship \
between two concepts, also emit edges connecting them. Use concept labels \
(not IDs) — the system will resolve them. Reference:
  - Proposals from this turn's own proposals list, OR
  - Existing corpus anchors (from the list below), OR
  - Any concept mentioned by name in the conversation.
Edge types:
  - INVOKES: A summons or activates B (structural dependency)
  - SUPPORTS: A provides evidence or grounding for B
  - CONFLICTS: A and B make incompatible claims
  - LINKS: soft association, no stronger claim available
Only emit edges that are explicitly asserted or strongly implied. No speculation.

Existing corpus anchors (do NOT re-propose these, but you MAY reference \
them as edge endpoints):
$EXISTING_ANCHORS

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
    {"type": "INVOKES|SUPPORTS|CONFLICTS|LINKS",
     "from_label": "concept name A",
     "to_label": "concept name B",
     "confidence": 0.0-1.0,
     "justification": "why this relationship is asserted"}
  ]
}

Recent conversation:
$CONVERSATION

Proposals: """

PROPOSAL_EXPLICIT_PROMPT = """\
The user has explicitly asked to save or anchor a concept from the conversation. \
Extract the specific concept they are referring to.

If the user's request implies a relationship to other concepts in the \
conversation (e.g. "save this as a slab that supports X", "anchor this — \
it conflicts with Y"), also emit edges linking them. Use concept labels \
(not IDs). Edge types: INVOKES, SUPPORTS, CONFLICTS, LINKS.

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
    {"type": "INVOKES|SUPPORTS|CONFLICTS|LINKS",
     "from_label": "concept name A",
     "to_label": "concept name B",
     "confidence": 0.0-1.0,
     "justification": "why this relationship is asserted"}
  ]
}

Recent conversation:
$CONVERSATION

User's request: """
