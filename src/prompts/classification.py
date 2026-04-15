"""Structured extraction prompts for the function gate and drift estimation."""

FUNCTION_GATE_PROMPT = """\
You are a message classifier for a semantic reasoning system. Classify the user's \
message into exactly one function category.

Categories:
- context_compression: User is referencing prior context, anchors, or semantic \
structure for reuse. Includes explicit "use this as context" signals AND requests \
to persist/save/anchor/bundle work into the corpus ("make a slab of this", \
"turn this plan into X", "save this as an anchor" — these compress ongoing work \
into reusable structure, which is exactly context compression).
- object_of_work: User is working on or discussing a specific object, concept, \
plan, design, or artifact. The substance of focused work. When the user is \
building something or walking through a concrete topic in depth.
- affect_release: User is processing emotions, venting, expressing personal \
experience. Internal domain.
- rigor_work: User is engaging in rigorous analysis, testing, challenging, \
or requesting adversarial pushback.
- meta_schema: User is discussing the system itself, its rules, its structure, \
or how it should behave (not content-level work — only when they're talking \
ABOUT the Mirror/OLI/rules).
- neutral: General conversation, greetings, clarifications, short acknowledgements. \
Only use this when NONE of the above fit. Substantive content belongs in \
object_of_work even if the user is being casual about it.

ADDITIONALLY, if the user's message references a concept, anchor, frame, or \
bundle by name or canonical phrase, classify HOW they are referencing it \
(mention_type). Otherwise leave mention_type null.

mention_type categories:
- reference: user is passively mentioning or alluding to a concept without \
asking about it or invoking it ("anyway, this reminds me of the archer thing"). \
No activation intent; the mention is incidental to the main message.
- question: user is asking about a concept, seeking information or clarification \
("what's the archer bundle about?", "how does epistemic floor work?"). They want \
to understand it, not load it as a live frame.
- deliberate_invocation: user is explicitly activating a concept as a live frame \
for the current turn ("*i am the bone of my sword*", "activate archer frame", \
"keep the epistemic floor rules on for this"). They want the referenced slab/\
bundle loaded into active context.

If no concept is referenced at all, mention_type is null.

Return ONLY raw JSON:
{
  "function": "<category>",
  "confidence": <float 0-1>,
  "explicit": <bool — true if the user explicitly asks to persist, save, anchor, \
bundle, instantiate, create, mint, draft, promote, or turn something into a \
slab/anchor/bundle/concept; also true for explicit "use this as context" / \
"remember this" / "add this to the corpus" signals. Match on INTENT not exact \
wording — any verb that means "materialize this as a persistent graph object" \
counts. False otherwise.>,
  "mention_type": "<reference | question | deliberate_invocation | null>",
  "notes": "<string or null — surface when confidence is low>"
}

Examples of explicit=true:
- "save this as a slab"
- "turn this plan into a slab called X"
- "anchor this concept"
- "remember this for later"
- "commit this to the corpus"
- "use this as context going forward"
- "instantiate a tentative slab with this"
- "lets instantiate a tentative slab"
- "create a slab for this"
- "make this an anchor"
- "draft a bundle from X and Y"
- "mint a tentative anchor from the above"
- "promote this to the corpus"
- "add this as a slab"
Examples of explicit=false:
- "what do you think about X"
- "explain Y"
- "help me with Z"
- Any general discussion without a persistence verb.

Examples of mention_type:
- "*i am the bone of my sword*" → deliberate_invocation (archer frame)
- "what's the epistemic floor again?" → question
- "anyway this reminds me of the archer thing" → reference
- "how does claim admissibility work?" → question
- "engage the regulation gates for this conversation" → deliberate_invocation
- "hi, how are you?" → null (no concept referenced)

User message: """

DRIFT_ESTIMATE_PROMPT = """\
You are a meta-cognitive signal estimator. Analyze the conversation turn below \
in the context of the session so far. Estimate these signals:

- affect_density (0-1): How much emotional/affective content is in this turn. \
0 = purely analytical. 1 = entirely emotional.
- claim_volatility (0-1): How much the user's position has shifted compared to \
their recent statements. 0 = stable. 1 = complete reversal.
- rigor_drop (0-1): Whether the discourse quality has dropped from earlier in \
the session. 0 = maintained or improved. 1 = significant degradation.
- domain_mode: "internal" if this is personal/affective/subjective work. \
"external" if this is about external/measurable/objective reality.

Return ONLY raw JSON with these four fields: affect_density, claim_volatility, rigor_drop, domain_mode.

Session context (recent turns):
$CONTEXT

Current turn: """


# ── Combined classify + drift (single LLM call) ──────────────
COMBINED_CLASSIFY_DRIFT_PROMPT = """\
You are a message analyzer for a semantic reasoning system. Perform TWO tasks on the user message below.

TASK 1 — MESSAGE CLASSIFICATION
Classify the dominant function:
- context_compression: referencing prior context, anchors, or asking to persist/save/anchor/bundle work
- object_of_work: working on a specific object, concept, plan, design, or artifact
- affect_release: processing emotions, venting, expressing personal experience
- rigor_work: rigorous analysis, testing, challenging, requesting adversarial pushback
- meta_schema: discussing the system itself, its rules, its structure
- neutral: general conversation, greetings, clarifications

Also determine mention_type if the user references a concept by name:
- reference: passive mention, no activation intent
- question: asking about a concept
- deliberate_invocation: explicitly activating a concept as live frame
- null: no concept referenced

explicit = true if user explicitly asks to persist/save/anchor/bundle/\
instantiate/create/mint/draft/promote something into a slab, anchor, bundle, \
or concept. Match on INTENT not exact wording: any verb meaning "materialize \
this as a persistent graph object" counts. Examples: "save this as a slab", \
"anchor this", "lets instantiate a tentative slab", "create a slab for X", \
"make this an anchor", "promote this to the corpus", "draft a bundle".

TASK 2 — DRIFT ESTIMATION
Estimate these meta-cognitive signals for the current turn:
- affect_density (0-1): emotional content level
- claim_volatility (0-1): position shift from recent statements
- rigor_drop (0-1): discourse quality degradation
- domain_mode: "internal" (personal/subjective) or "external" (objective/measurable)

Return ONLY raw JSON combining both tasks:
{
  "function": "<category>",
  "confidence": <float 0-1>,
  "explicit": <bool>,
  "mention_type": "<reference | question | deliberate_invocation | null>",
  "notes": "<string or null>",
  "affect_density": <float 0-1>,
  "claim_volatility": <float 0-1>,
  "rigor_drop": <float 0-1>,
  "domain_mode": "<internal | external>"
}

Session context:
$CONTEXT

User message: """

SALIENCE_PROMPT = """\
You are a salience estimator for a semantic graph. Given the active frame nodes \
and the current conversation turn, estimate how salient (relevant/important) each \
node is to this turn. 0 = irrelevant, 1 = central to the turn.

Active nodes (id and description):
$NODES

Return ONLY raw JSON — a dict mapping node_id to salience (0.0-1.0):
{"node_id_1": 0.8, "node_id_2": 0.3}

Current turn: """

CONCEPT_DETECT_PROMPT = """\
You are a concept extractor for a semantic knowledge system. Identify the 1-2 \
PRIMARY concepts, entities, or named things the user is focused on in this turn.

EXTRACT when the user:
- names a specific topic, project, place, person, system, framework, or concept \
(even in passing — "I'm thinking about terra preta australis" → extract "terra preta australis")
- introduces a technical term, methodology, artifact, or domain
- brings up a new subject that could plausibly become a reusable knowledge node
- asks to save/anchor/slab something (extract what they want saved)

DO NOT extract when the user is:
- greeting ("hi", "hey", "thanks")
- purely meta about the chat itself ("can you repeat that", "what did you mean")
- giving short acknowledgements ("yes", "ok", "go on")
- venting emotion with no topical content

When in doubt, EXTRACT — the code downstream deduplicates against existing \
corpus and consolidates near-identical tentative nodes, so extra extraction \
is cheap. Missing a concept is expensive.

Return ONLY raw JSON — an array of 0-2 objects:
[{"concept": "short canonical name", "description": "one sentence definition or context"}]

User message: """
