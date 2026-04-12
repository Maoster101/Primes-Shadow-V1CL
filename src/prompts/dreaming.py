"""Dreaming-phase prompts — grounding audit + content rewrite.

The dreaming pass sits between mining (extract_proposals) and commit
(review_draft). Its job is to check whether the miner's output is
grounded in reality and rewrite it if not.

Two grounding modes:
  - CODE:   draft references system internals → ground against source code
  - DOMAIN: draft captures domain knowledge from conversation → ground
            against the conversation itself + existing corpus

Two-stage prompt:
  1. GROUNDING_AUDIT — decide what's real vs confabulated/vague
  2. REWRITE — produce corrected/refined content for the draft
"""

# ---------------------------------------------------------------------------
# Stage 1: Grounding audit
# ---------------------------------------------------------------------------
# Input slots:
#   $DRAFT_TYPE        — "anchor" or "slab"
#   $DRAFT_ID          — e.g. "tentative_anchor_73d88b7b_v1"
#   $DRAFT_CONTENT     — JSON of the current inline dict (anchor or slab)
#   $JUSTIFICATION     — the packet-level justification string
#   $CHAT_CONTEXT      — the original chat turns that sourced this draft
#   $GROUNDING_MODE    — "CODE" or "DOMAIN"
#   $REFERENCE_MATERIAL — code snippets (CODE mode) or expanded chat (DOMAIN mode)
#   $CORPUS_CONTEXT    — summaries of related existing corpus nodes

GROUNDING_AUDIT_PROMPT = """\
You are a grounding auditor for a knowledge corpus. A mining pass extracted \
a draft {$DRAFT_TYPE} from a conversation. The miner only sees chat text, \
which may contain model confabulation, vague paraphrasing, or missing context.

**Grounding mode: $GROUNDING_MODE**

$MODE_INSTRUCTIONS

## Draft being audited

ID: $DRAFT_ID
Type: $DRAFT_TYPE
Content:
```json
$DRAFT_CONTENT
```
Justification: $JUSTIFICATION

## Original chat context (where this was mined from)
$CHAT_CONTEXT

## Reference material (ground truth for this mode)
$REFERENCE_MATERIAL

## Related existing corpus nodes (check for redundancy)
$CORPUS_CONTEXT

## Output

Output ONLY raw JSON:
{
  "verdict": "GROUNDED" | "PARTIALLY_GROUNDED" | "CONFABULATED",
  "grounded_claims": ["list of claims that are verified or well-supported"],
  "confabulated_claims": ["list of claims that are wrong, vague, or unsupported"],
  "real_terms": {
    "inaccurate_term": "corrected_term",
    ...
  },
  "grounding_sources": ["references to where ground truth lives"],
  "redundant_with": ["corpus node IDs this duplicates, if any"],
  "rewrite_needed": true | false,
  "notes": "brief explanation of the overall grounding situation"
}
"""

# Mode-specific instruction blocks inserted into $MODE_INSTRUCTIONS
MODE_INSTRUCTIONS_CODE = """\
This draft appears to reference **system internals** — code, config, schemas, \
or architectural concepts from the codebase. Your job: compare the draft's \
claims against the ACTUAL CODE provided below.

For each claim, check:
1. Is this term/phrase used in the actual code? Where exactly?
2. Does the described behavior match what the code actually does?
3. Are there real field names, function names, or config keys that should \
   replace confabulated or imprecise ones?
4. Does an existing corpus node already cover this concept? (check redundancy)

Mark as CONFABULATED if the code doesn't support the claim at all. \
Mark as PARTIALLY_GROUNDED if the concept is real but details are wrong. \
Mark as GROUNDED only if the draft accurately reflects the code."""

MODE_INSTRUCTIONS_DOMAIN = """\
This draft captures **domain knowledge** from a natural conversation — \
it's about an external topic, not about the system's own codebase. \
Your job: assess the quality and accuracy of the extracted knowledge.

For each claim, check:
1. Is this a coherent, well-formed statement of knowledge? Or is it \
   a vague paraphrase that lost meaning during extraction?
2. Does the conversation actually support this claim, or did the miner \
   hallucinate connections that aren't in the source turns?
3. Are key details, caveats, or qualifications missing that were present \
   in the original conversation?
4. Does an existing corpus node already cover this concept? (check redundancy)
5. Is the canonical_phrase/title specific enough to be useful as a \
   retrieval handle, or is it too generic?

Mark as CONFABULATED if the conversation doesn't support the claim. \
Mark as PARTIALLY_GROUNDED if the concept is real but extraction lost \
important nuance, specificity, or context. \
Mark as GROUNDED if the draft faithfully captures the conversational insight."""

# ---------------------------------------------------------------------------
# Stage 2: Content rewrite
# ---------------------------------------------------------------------------
# Input slots:
#   $DRAFT_TYPE        — "anchor" or "slab"
#   $DRAFT_ID          — e.g. "tentative_anchor_73d88b7b_v1"
#   $DRAFT_CONTENT     — JSON of the current inline dict
#   $AUDIT_RESULT      — JSON output from stage 1
#   $GROUNDING_MODE    — "CODE" or "DOMAIN"
#   $REFERENCE_MATERIAL — code snippets or expanded chat
#   $COMPANION_DRAFTS  — other drafts from same session (for cross-linking)

REWRITE_PROMPT = """\
You are a corpus content rewriter. A grounding audit found that a draft \
{$DRAFT_TYPE} needs revision. Your job: rewrite the content fields to be \
accurate, specific, and well-grounded.

**Grounding mode: $GROUNDING_MODE**

$MODE_REWRITE_INSTRUCTIONS

## General rules
- Preserve the draft ID exactly: $DRAFT_ID
- Preserve structural fields (id, version, meta) — only rewrite CONTENT fields
- For anchors: rewrite canonical_phrase, aliases, notes, invokes
- For slabs: rewrite title, canonical_text, links.anchors (if companions exist)
- Keep aliases broad: include both corrected terms AND the original terms \
  (so the matcher still resolves old references to this node)
- If a companion draft exists that this draft should link to, include \
  its ID in links.anchors (for slabs) or invokes (for anchors)

## Grounding audit result
```json
$AUDIT_RESULT
```

## Current draft content
```json
$DRAFT_CONTENT
```

## Reference material
$REFERENCE_MATERIAL

## Companion drafts from same session
$COMPANION_DRAFTS

## Output

Return ONLY the rewritten inline dict as raw JSON. For an anchor:
{"id": "...", "canonical_phrase": "...", "aliases": [...], "invokes": [...], \
"notes": "..."}

For a slab:
{"id": "...", "title": "...", "canonical_text": "...", \
"links": {"anchors": [...], "bundles": [...]}, "version": "v1"}

Rewritten content: """

MODE_REWRITE_CODE = """\
This is a CODE-mode rewrite. The draft references system internals.
- Use REAL function names, field names, config keys from the code
- Include specific file:line references in notes/canonical_text
- Replace any confabulated technical terms with the actual code terms
- If the code implements something the draft describes vaguely, be precise \
  about the mechanism (e.g., "EWA with alpha=0.30" not "smoothing")"""

MODE_REWRITE_DOMAIN = """\
This is a DOMAIN-mode rewrite. The draft captures external knowledge.
- Recover any nuance, caveats, or specificity that the miner lost
- Make the canonical_text a clear, self-contained statement of knowledge \
  that would be useful to someone who hasn't read the original conversation
- If the conversation cited specific numbers, dates, constraints, or \
  references, include them
- Make the title/canonical_phrase specific enough to be a useful retrieval \
  handle — avoid generic phrases like "important concept" or "key idea"
- If the draft is about a process or procedure, capture the steps
- If the draft is about a distinction or comparison, make both sides explicit"""
