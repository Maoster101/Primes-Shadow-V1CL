"""Fast keyword-based pre-classifier for gate function detection.

Runs synchronously before the system prompt is built, so function-dependent
signals (currently ``review_mode``) reach the model on the turn they're
needed — before the LLM-based ``classify_and_drift`` background task
completes post-stream.

Kept intentionally conservative: CORPUS_REVIEW requires BOTH a review verb
AND a corpus noun. Missed detections fall through to the regular classifier
and get the normal behavior; false positives would inject the review-mode
addendum where it isn't warranted, which is the larger failure mode —
framing an unrelated question as corpus review mis-shapes the response.
"""
from __future__ import annotations
import re
from typing import Optional

from ..models.enums import MessageFunction


# Review verbs / adjectives — editorial / curation signals.
# Intentionally does NOT include generic "look at" / "check" / "good" —
# those are too broad and would fire on ordinary review-of-content
# questions. Adjectives like ``redundant`` and ``duplicate`` are included
# because they work as standalone review signals: "which anchor is
# redundant?" has no verb but clear editorial intent.
_REVIEW_VERBS = re.compile(
    r"\b("
    r"compare|contrast|review|assess|evaluate|critique|curate|prune|"
    r"deduplicate|dedupe|merge|consolidate|"
    r"redundant|duplicates?|"
    r"which\s+(?:should|would|do)\s+i\s+keep|"
    r"which\s+(?:is|are)\s+(?:better|more\s+useful)|"
    r"worth\s+(?:keeping|deleting|promoting|discarding)|"
    r"should\s+i\s+(?:keep|delete|merge|promote|discard|commit|purge)|"
    r"clean\s+up|tidy\s+up"
    r")\b",
    re.IGNORECASE,
)

# Corpus nouns — objects or collections in the knowledge base.
# Plural forms handled by the trailing s? on each root. Intentionally
# includes "mined" since the user often says "the mined proposals" or
# "the mined slabs" when reviewing mining output.
_CORPUS_NOUNS = re.compile(
    r"\b("
    r"anchors?|slabs?|bundles?|edges?|"
    r"collections?|corpus|corpora|"
    r"drafts?|tentatives?|proposals?|"
    r"mined|mining\s+(?:output|result|results|pass|passes)|"
    r"(?:graph\s+)?nodes?|entries|entry"
    r")\b",
    re.IGNORECASE,
)


def preclassify_function(user_text: str) -> Optional[MessageFunction]:
    """Return a MessageFunction if keyword heuristics fire, else None.

    Currently only detects CORPUS_REVIEW. Other gate types stay with the
    full LLM classifier since their signals are more nuanced (e.g.
    distinguishing RIGOR_WORK from OBJECT_OF_WORK requires understanding
    whether the user is challenging or building).
    """
    if not user_text or len(user_text.strip()) < 10:
        return None

    has_verb = bool(_REVIEW_VERBS.search(user_text))
    has_noun = bool(_CORPUS_NOUNS.search(user_text))

    if has_verb and has_noun:
        return MessageFunction.CORPUS_REVIEW

    return None
