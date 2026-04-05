"""Structured extraction prompts for tentative anchor/slab proposals.

Conservative by design: only propose load-bearing concepts, not every
passing mention. The corpus must stay sparse and authoritative.
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
- If nothing in the conversation is genuinely novel or load-bearing, return [].

Existing corpus anchors (do NOT re-propose these):
$EXISTING_ANCHORS

Return ONLY raw JSON — an array (may be empty):
[{"type": "anchor" or "slab", "canonical_phrase": "short name" (for anchors), \
"canonical_text": "full text" (for slabs), "aliases": ["alt name", ...], \
"justification": "why this is load-bearing", "source_turns": [int, ...], \
"claim_tag": "FACT" or "INFERENCE" or "HYPOTHESIS" or "UNKNOWN"}]

Recent conversation:
$CONVERSATION

Proposals (at most 1-2, or empty array): """

PROPOSAL_EXPLICIT_PROMPT = """\
The user has explicitly asked to save or anchor a concept from the conversation. \
Extract the specific concept they are referring to.

Return ONLY raw JSON — a single object:
{"type": "anchor" or "slab", "canonical_phrase": "short name" (for anchors), \
"canonical_text": "full text" (for slabs), "aliases": ["alt name", ...], \
"justification": "why the user wants this saved", "source_turns": [int, ...], \
"claim_tag": "FACT" or "INFERENCE" or "HYPOTHESIS" or "UNKNOWN"}

Recent conversation:
$CONVERSATION

User's request: """
