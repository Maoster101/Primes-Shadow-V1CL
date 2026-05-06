"""Synthesis intent detection — heuristic gate for routing chat turns
through the synthesis pipeline.

Step 3 of the synthesis feature stack:
  Step 1 — synthesis.synthesize     (selection / graph walk)
  Step 2 — synthesis_compose        (selection -> system prompt)
  Step 3 — synthesis_intent         (when to route a turn through 1+2)

Design note — why heuristic and not an LLM classifier:
  ``classify_and_drift`` already runs a real LLM call in the
  background of every turn. Adding a *second* LLM call on the
  critical path just to decide "is this a synthesis question?" would
  double pre-stream latency for every chat message. A regex + anchor-
  count heuristic is cheap, deterministic, and recoverable on both
  sides:

    - False positive: synthesis composes a prompt that's appended on
      top of the standard one. The model still answers the question;
      it just has more curated context. No harm.
    - False negative: turn falls back to standard RAG retrieval,
      which is what would have happened anyway.

The two signals combine because each alone has a known weakness:

  Pattern hit:   catches "tell me about X", "explain X", "summarise X"
                 — explicit synthesis-y framings. Misses paraphrased
                 queries like "how do mercy and justice relate?".
  Anchor count:  ≥2 anchors fire on the user's message means the
                 message is multi-topic / topic-focused — the kind of
                 query that benefits from graph-aware retrieval.
                 Misses queries that name no canonical concepts.

Either signal is enough; neither is required of the other. Citation
detection rides on top — only fires when synthesis already fires.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


# ── Pattern banks ─────────────────────────────────────────────────
#
# Phrases that strongly suggest the user is asking for a corpus-
# grounded synthesis rather than a quick lookup. Ordered roughly by
# specificity — more specific patterns first so the matcher returns
# early on the strongest signal.
#
# Tuning principle: prefer FALSE NEGATIVES over FALSE POSITIVES.
# A missed synthesis falls back gracefully to standard RAG; a
# spurious synthesis adds compute (~500ms embedding + walk) with no
# user-visible bug, but does add latency. Patterns are kept conservative.

_SYNTH_PATTERNS = [
    r"\btell me (?:about|more about)\b",
    r"\bsynthe[sz]i[sz]e\b",
    r"\bgive me (?:an? )?overview\b",
    r"\bwalk me through\b",
    r"\bexplain (?:the |my |your |this )?(?:concept|idea|thinking|view|stance|position|reasoning|framework|model)\b",
    r"\bwhat (?:is|are) (?:my |the |your )?(?:view|stance|position|take|thinking|reasoning|framework)\b",
    r"\bhow (?:do|does|did) .{2,40}\b(?:work|relate|connect|differ|compare)\b",
    r"\bcompare\b.{2,60}\b(?:and|with|to|against|vs)\b",
    r"\bcontrast\b",
    r"\bwhat do (?:I|you|we) think about\b",
    r"\bsummari[sz]e\b",
    r"\bbreak (?:this|that|it) down\b",
    r"\bwhat'?s (?:the |my )?(?:case|argument|reasoning|point) (?:for|against|behind)\b",
    r"\bdialectic\b",
    r"\btrade[- ]?offs?\b",
]

# Phrases that indicate the user wants citation markers in the
# composed output. Detected only when synthesis already fires —
# citation flag has no meaning otherwise.
_CITE_PATTERNS = [
    r"\bwith (?:sources|citations|references)\b",
    r"\bcite (?:sources|specific|the)",
    r"\bshow (?:me )?(?:the )?source",
    r"\breferences?\b.{0,20}\bplease\b",
    r"\b(?:include|add) (?:source|citation|reference)s?\b",
]


_synth_re = re.compile("|".join(_SYNTH_PATTERNS), re.IGNORECASE)
_cite_re = re.compile("|".join(_CITE_PATTERNS), re.IGNORECASE)


# ── Tunables ──────────────────────────────────────────────────────

ANCHOR_COUNT_THRESHOLD = 2
"""Minimum anchors fired on the user's message to fire synthesis on
the anchor-count signal alone. Single anchor mentions are too common
to treat as synthesis intent (a user dropping a coined term in casual
chat shouldn't trigger a 10k-token graph walk). Two-plus anchors
strongly indicates a multi-concept query that benefits from the
graph-aware tier structure."""

MAX_USER_TEXT_PEEK = 2000
"""Cap on user_text length we scan with the regex matchers. Longer
prompts (paste-bombs, etc.) are unlikely to be synthesis queries —
synthesis-y messages are typically short questions. Capping protects
against pathological regex backtracking on huge inputs."""


# ── Result type ───────────────────────────────────────────────────


@dataclass
class SynthesisIntent:
    """Structured result of intent detection.

    Returning a dataclass (rather than a tuple) because callers want
    to log WHY synthesis fired — pattern hit vs anchor count is
    diagnostic for tuning. The compose layer also benefits from
    knowing when the pattern that fired hints at citations.
    """
    triggered: bool
    include_citations: bool
    pattern_hit: bool
    anchor_signal: bool
    anchor_count: int


# ── Public entry ──────────────────────────────────────────────────


def detect_synthesis_intent(
    user_text: str,
    *,
    anchor_match_count: int = 0,
) -> SynthesisIntent:
    """Decide whether to route this turn through the synthesis pipeline.

    Args:
      user_text: the user's raw message for this turn.
      anchor_match_count: number of anchors the matcher fired on
        this turn (auto-activate + candidates, deduplicated). Pass 0
        when no matcher is wired — the pattern signal alone still works.

    Returns:
      SynthesisIntent with ``triggered`` true when synthesis should
      fire, ``include_citations`` true when the user asked for
      sources, and the individual signals exposed for diagnostics.

    Synthesis fires if EITHER signal is present:
      - pattern_hit:   regex match against synthesis-y phrasings
      - anchor_signal: anchor_match_count >= ANCHOR_COUNT_THRESHOLD

    Citations only fire when synthesis fires and a citation phrase
    is also present. Without the synthesis fire, the citation flag
    is meaningless (no compose layer to consume it).
    """
    if not user_text or not user_text.strip():
        return SynthesisIntent(
            triggered=False,
            include_citations=False,
            pattern_hit=False,
            anchor_signal=False,
            anchor_count=anchor_match_count,
        )

    text = user_text.strip()[:MAX_USER_TEXT_PEEK]
    pattern_hit = bool(_synth_re.search(text))
    anchor_signal = anchor_match_count >= ANCHOR_COUNT_THRESHOLD
    triggered = pattern_hit or anchor_signal
    include_citations = triggered and bool(_cite_re.search(text))

    return SynthesisIntent(
        triggered=triggered,
        include_citations=include_citations,
        pattern_hit=pattern_hit,
        anchor_signal=anchor_signal,
        anchor_count=anchor_match_count,
    )
