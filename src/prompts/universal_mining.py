"""Universal mining prompt shared by every AI Mine 2 source profile.

Source-specific code is responsible only for normalization and segmentation.
The ontology, evidence contract, and JSON shape stay identical across chats,
narratives, documents, and papers.
"""
from __future__ import annotations

import json


_PROFILES = {
    "chat": (
        "Treat user-authored statements and assistant contributions explicitly adopted, "
        "developed, or confirmed by the user as authoritative. Do not canonize an "
        "unacknowledged assistant suggestion. Later corrections override earlier claims. "
        "Conversation order matters for corrections and decisions, but adjacency alone "
        "does not imply a SEQUENCE edge."
    ),
    "narrative": (
        "Treat the supplied narrative as authoritative within its fictional, "
        "autobiographical, or argumentative frame. Preserve genuine event and beat order. "
        "A named character or entity that recurs across multiple events, drives the action, "
        "or is referred to by more than one name IS an anchor — extract it as one and fold "
        "its variant names (given name, surname, epithet, the name others call it by) into "
        "aliases; do not dissolve the protagonists into event slabs while anchoring only "
        "incidental scenery. Only genuine walk-ons named once stay unanchored. This "
        "recurring-character rule is specific to narrative sources and must not be "
        "generalized to documents or papers (a document naming many parts, steps, or fields "
        "does not thereby make each one an anchor)."
    ),
    "document": (
        "Treat the document as the source authority for what it states, while leaving "
        "external truth unverified. Preserve supplied section paths. Extract definitions, "
        "principles, procedures, constraints, decisions, limitations, and conclusions."
    ),
    "paper": (
        "Treat the paper as authoritative only for what its authors report or argue. "
        "Distinguish claim, method, result, interpretation, comparison, and limitation. "
        "Preserve sample conditions, uncertainty, scope, and negative results."
    ),
    "unknown": (
        "Use conservative source-authored authority. Preserve explicit structure and order "
        "only where the source makes them meaningful. Do not infer a genre-specific claim."
    ),
}


def build_universal_prompt(
    *,
    task: str,
    source_kind: str,
    source_path: str,
    existing_catalog: list[dict],
    source_material: str,
) -> str:
    """Build one stage of the universal EXTRACT -> CONSOLIDATE -> RELATE protocol."""
    kind = source_kind if source_kind in _PROFILES else "unknown"
    task = task.upper().strip()
    catalog_json = json.dumps(existing_catalog, ensure_ascii=False)

    task_rules = {
        "EXTRACT": """
Extract grounded candidate slabs and anchors from this source segment. Emit only
relationships directly evidenced inside the segment. Do not merge against unseen
segments. A dense segment may yield several objects; a short segment may yield none.
""",
        "CONSOLIDATE": """
The SOURCE MATERIAL is a JSON array of candidates from every segment. Merge semantic
duplicates, preserve every source_ref, choose one discriminative canonical phrase/title,
and keep materially different conditions or claims separate. Normalize source_path into
shared descriptive semantic headings across segments; remove all segment/chunk/page and
bare-number path atoms. Compare finalists with the existing catalog. If a candidate is already represented, omit it and record a warning.

ANCHOR DEDUP — decide merges on IDENTITY, not surface similarity. Use each anchor's
description to judge whether two anchors are the SAME referent. If they are (Harry Potter
/ Harry / the boy who lived), COLLAPSE them into one anchor: keep the most discriminative
canonical_phrase, fold the others into aliases, keep the clearest identity description.
If two anchors are distinct concepts that merely co-occur or point at the same entity —
a contested title ("The Chosen One") or a category term/slur ("Mudblood") whose identity
is its own — DO NOT merge them, even when lexically or topically close; a merge here is
destructive. Preserve them as separate anchors and express the relation as an INVOKES
relationship in the RELATE stage. Every surviving anchor must keep a description.
Return the surviving candidates; relationships are not required in this stage.
""",
        "RELATE": """
The SOURCE MATERIAL contains finalized candidates plus compact source evidence. Create
only relationships supported by that evidence. Do not create nodes. Endpoints must be a
candidate temp_id or exact existing catalog ID. Preserve meaningful opposition and do not
use LINKS merely because two ideas are topically similar.
""",
    }.get(task, "Perform the requested mining task conservatively.")

    return f"""You are the universal semantic mining engine for Prime's Shadow.

Your purpose is to convert source material into a sparse, authoritative, searchable
knowledge graph. Source profiles change interpretation, never the ontology or schema.

TASK: {task}
SOURCE KIND: {kind}
SOURCE LOCATOR (evidence only, never a hierarchy label): {source_path or '(root)'}
PROFILE GUIDANCE: {_PROFILES[kind]}

TASK RULES
{task_rules.strip()}

UNIVERSAL ONTOLOGY

SLAB -- the primary knowledge unit. A slab records one durable semantic MOVE: a
definition, claim, principle, method, result, interpretation, decision, constraint,
limitation, comparison, narrative event, or world rule. Material making the same move
belongs together; material making a different move belongs in another slab.

Every slab must have a specific 3-8 word title and complete, self-contained canonical
text. Preserve conditions, qualifications, exceptions, uncertainty, and negative
results. Never add unsupported completion. Do not leave dangling references such as
"this" or "the above" without restating their referent.

Every slab must also have a DESCRIPTION: one dense sentence written FOR RETRIEVAL,
like an encyclopedia or wiki index entry. Say what the slab establishes and what
questions it answers, in the third person about the slab ("Defines the airflow
assumptions governing scent dispersion and when perceptible wind becomes opt-in"),
NOT a restatement of the canonical text. Someone scanning only descriptions should
be able to tell whether this slab is the one they need. Keep it self-contained: no
"this slab", no pronouns without a referent, no reliance on the title being read too.

ANCHOR -- a reusable invocation handle for an entity OR a key concept. Create one only
for a distinctive phrase the source names, coins, repeatedly invokes, or clearly defines
and that a user might say later to retrieve it. Do not create anchors for headings,
generic nouns, every entity, or phrases useful only inside one slab.

Every anchor must have a DESCRIPTION: one dense sentence stating its IDENTITY — what
kind of thing it IS and what would let a reader recognize it's the concept they mean.
Third person, self-contained, stable across the corpus, not tied to one incident. This
is NOT the alias framing ("refers to X") and NOT relational context. Do NOT name which
characters or slabs invoke it — that is edge information, not identity. Example (concept,
category term): "A pejorative for a witch or wizard born to non-magical parents, used by
blood-status supremacists as an insult" — note it names no specific target.

Aliases must be lexical alternatives for the SAME referent (Harry Potter / Harry / the
boy who lived). Subtypes, examples, populations, consequences, descriptions, and related
principles are not aliases. Critically, do NOT fold a DISTINCT concept into an entity as
an alias just because they co-occur. A contested title ("The Chosen One" — a prophesied
role ambiguously contested between two candidates) and a category term ("Mudblood") are
their own anchors with their own identity descriptions, related to the entity by an
INVOKES edge, never merged into it. Write each description precisely enough that "same
referent?" is answerable from the descriptions alone.

Do not invent pillars or bundles during extraction. For source_path, preserve the source's
semantic heading hierarchy using short descriptive heading text. Never put transport
locators such as source labels, segment-3, chunk 2, page numbers, paragraph numbers, or a
bare section number in source_path. If the source has no headings, derive a stable 2-6 word
descriptive topic path grounded in the material. Hierarchy and directive bundles are
handled after semantic extraction.

RELATIONSHIPS
- INVOKES: a handle directly activates a knowledge object
- SUPPORTS: A provides evidence, grounding, or justification for B
- CONFLICTS: A and B are incompatible; one rejects or rules out the other
- TENSIONS: A and B remain valid but must be balanced
- SEQUENCE: A precedes B temporally, procedurally, narratively, or derivationally
- PARENT_OF: A structurally contains B
- LINKS: a meaningful association exists but no stronger type is justified

Use the strongest supported type. Uncertainty is a reason to omit an edge, not to emit
LINKS. Endpoints must be a temp_id emitted in this response or an exact existing node ID.
Never use collection names or free-form labels as endpoints.

GROUNDING AND AUTHORITY
Every candidate needs at least one source_ref with an exact supplied locator, a short
verbatim quote, and authority status. Do not emit a candidate without evidence. For chat,
assistant_unadopted material is ineligible. A correction or final decision supersedes an
earlier alternative unless the rejected alternative remains independently useful.

CLAIM TAGS
- FACT: externally verifiable claim
- SOURCE_CLAIM: asserted by the source, not independently verified
- INFERENCE: logically derived from the source
- HYPOTHESIS: conjectural or proposed
- NORMATIVE: preference, value, rule, or prescription
- FICTIONAL_CANON: true inside a fictional/constructed setting
- UNKNOWN: cannot safely classify

Confidence measures extraction confidence, not truth. Omit candidates below 0.50.

EXISTING CORPUS CATALOG
Do not recreate these nodes. Reference them only by exact ID:
{catalog_json}

OUTPUT
Return ONLY valid JSON:
{{
  "status": "OK|EMPTY_VALID|UNCERTAIN",
  "candidates": [
    {{
      "temp_id": "candidate_001",
      "type": "anchor|slab",
      "canonical_phrase": "required for anchor, null for slab",
      "aliases": ["same-referent lexical alternatives"],
      "title": "required for slab, null for anchor",
      "canonical_text": "required for slab, null for anchor",
      "description": "required for BOTH: for a slab, one dense retrieval sentence — what it establishes and the questions it answers; for an anchor, one dense IDENTITY sentence — what the concept/entity IS, no relational context, no invoking characters",
      "semantic_role": "definition|claim|principle|method|result|interpretation|decision|constraint|limitation|comparison|narrative_event|world_rule|other",
      "source_path": ["part", "section", "subsection"],
      "source_refs": [
        {{"locator": "exact supplied locator", "quote": "short verbatim evidence", "authority": "user_authored|user_adopted|source_authored|assistant_unadopted|inferred"}}
      ],
      "claim_tag": "FACT|SOURCE_CLAIM|INFERENCE|HYPOTHESIS|NORMATIVE|FICTIONAL_CANON|UNKNOWN",
      "confidence": 0.0,
      "justification": "why this is durable and independently retrievable"
    }}
  ],
  "relationships": [
    {{
      "type": "INVOKES|SUPPORTS|CONFLICTS|TENSIONS|SEQUENCE|PARENT_OF|LINKS",
      "from_ref": "candidate temp_id or exact existing node ID",
      "to_ref": "candidate temp_id or exact existing node ID",
      "source_refs": [{{"locator": "source locator", "quote": "relationship evidence"}}],
      "confidence": 0.0,
      "justification": "specific relationship asserted by the evidence"
    }}
  ],
  "warnings": [{{"locator": "source locator or null", "issue": "ambiguity, missing context, conflict, merge, or truncation"}}]
}}

SOURCE MATERIAL
{source_material}
"""

