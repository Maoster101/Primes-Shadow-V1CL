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
  "ungrounded_references": ["anchor canonical_phrase that's referenced (in slab.references_anchors or anchor.invokes) but isn't supported by the source chat turns or reference material"],
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
Your job: assess whether the extraction is *grounded* in what the source \
chat actually said.

Important context: the mining pipeline already handles surface-quality \
concerns at extraction time — alias capture, generic-phrase filtering, \
markdown stripping, and anchor-slab structural links. Don't re-litigate \
those. Your job is the deeper grounding question only you have access \
to: did the miner's claims actually come from the source turns, or did \
the model extrapolate beyond them?

For each claim, check:
1. Does the canonical_text or notes assert anything that goes BEYOND \
   what the source turns explicitly said or clearly implied? Mining can \
   produce over-confident summarisations — flag claims that aren't \
   chat-supported as confabulated.
2. **For each anchor referenced** (in slab.references_anchors or \
   anchor.invokes), does that anchor's concept actually appear in or \
   get clearly implied by the source turns? An anchor reference the \
   chat doesn't support is an invented structural link — list it in \
   `ungrounded_references`.
3. Does the conversation provide details, caveats, or qualifications \
   that the extraction lost? Recoverable nuance counts as \
   PARTIALLY_GROUNDED — the rewrite stage will fill them back in.
4. Does an existing corpus node already cover this concept? (check \
   redundancy — list duplicate node IDs in `redundant_with`)

Mark as CONFABULATED if the chat doesn't support the core claim. \
Mark as PARTIALLY_GROUNDED if the concept is real but key details, \
nuance, or anchor references are not chat-supported. \
Mark as GROUNDED if every claim and every anchor reference traces \
to something the source turns said."""

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
{$DRAFT_TYPE} needs revision. Your job: revise ONLY the fields the audit \
flagged. Preserve everything else verbatim.

**Grounding mode: $GROUNDING_MODE**

$MODE_REWRITE_INSTRUCTIONS

## General rules

The mining pipeline already produces structurally good output: aliases \
captured from source surface, anchor-slab links, bundle membership, \
canonical phrase forms with markdown stripped. **Don't re-derive what \
mining already produced.** Your job is narrowly scoped to what the \
audit specifically flagged.

- Preserve the draft ID exactly: $DRAFT_ID
- Preserve structural fields (id, version, meta) — never rewrite these
- **Where the audit was silent on a field, preserve it verbatim from the draft.**
- **Aliases**: only ADD aliases the audit explicitly identified in source \
  turns. Preserve mining-produced aliases unless the audit flagged them \
  as confabulated. Do not speculatively broaden the alias list.
- **Anchor references** (slab.links.anchors / slab.references_anchors / \
  anchor.invokes): only REMOVE references that appear in the audit's \
  `ungrounded_references` list. Don't generally re-derive these — Pass 3 \
  produced them with structural intent. ADD a reference only if the \
  audit explicitly identified a missing one.
- If a companion draft exists that this draft should link to AND the \
  audit flagged the missing link, include the companion's ID in the \
  appropriate field.

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
This is a DOMAIN-mode rewrite. Mining produced the structural fields \
already (canonical_phrase, aliases, references_anchors). Your scope is \
narrow: fix only what the audit flagged.

Focus on:
- **canonical_text accuracy** (slabs): if the audit flagged claims as \
  going beyond the source turns, tighten the text to what the chat \
  actually supports. Recover nuance, caveats, numbers, dates, or \
  qualifications the miner missed.
- **notes** (anchors): mining doesn't populate this. If the source \
  turns give context — origin, scope, or limitations of the term — \
  fill it in.
- **invokes** (anchors): if the audit flagged a missing bundle link \
  AND a related bundle exists in companions or corpus, add it.
- **redundancy resolution**: if the audit found redundancy with an \
  existing corpus node (in `redundant_with`), prefer to enrich the \
  existing node rather than creating a duplicate — make this explicit \
  in the notes / canonical_text.
- **ungrounded references**: remove any anchor reference the audit \
  listed in `ungrounded_references` from invokes / links.anchors.

Do NOT rewrite (mining already handled these well):
- canonical_phrase / title: mining's negative-space rules filtered \
  generic phrases at extraction time.
- aliases: mining captures variants from source surface. Only add \
  aliases the audit explicitly identified in the chat.
- references_anchors / links.anchors: Pass 3 produced these as \
  structural intent. Only remove items the audit listed in \
  `ungrounded_references`."""
