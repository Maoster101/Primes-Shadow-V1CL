"""Anchor matcher — §8 + OP_01 (ASTERISK WRAP).

The `*` operator in Prime's Shadow is heavily overloaded. A single
wrapped span can simultaneously be an anchor invocation, a stage
direction, a cultural reference, and an affect signal — e.g.
`*oh my GAAAHHHD*` is a stage direction, a paralinguistic exclamation,
a probable imitation (Ironmouse-style vocal tic), and a strong affect
marker all at once.

Earlier designs tried to dispatch each wrapped span into exactly one
of {anchor_hit, unresolved, depth, correction, stage_direction}. That
is wrong: the categories overlap. This module instead extracts a set
of FEATURES from every wrapped span and emits a single WrappedSpan
record carrying all applicable features. The runtime header shows the
feature set plus a derived `primary_reading` label; the Mirror is
taught (in the constitutional prompt) how to compose features into an
interpretation.

Features:
  structural:
    - anchor_hit            one hard anchor match
    - ambiguous_anchor      two or more hard anchor matches
  content (lexical):
    - correction_cue        matches CORRECTION_CUES
    - emote_vocab           matches EMOTE_VERBS or EMOTE_NOUNS
  content (orthographic):
    - extended_vowels       three or more consecutive same vowel
    - extended_consonants   four or more consecutive same consonant
    - affect_caps           word of three or more consecutive caps
    - mixed_caps            mid-word case shift (mOcK, jeSAAAAAAHHs)
    - repeated_punct        !!!, ???, ... (three or more)
  shape:
    - short                 <= 5 words
    - long                  > 10 words

Primary reading precedence (for quick display only; the feature set is
the source of truth):
  ambiguous_anchor > anchor_hit > correction_cue
    > (affect_caps | extended_vowels | mixed_caps) -> performed_imitation
    > (emote_vocab & short)                         -> stage_direction
    > (repeated_punct & short)                      -> strong_affect
    > long                                          -> semantic_depth
    > default                                       -> semantic_depth

Hard parse rules (OP_01):
  - `*...*` must be contiguous and closed.
  - `**` runs are plain text (markdown passthrough).
  - Nested `*` inside a wrap closes at the inner `*`.
  - An unclosed trailing wrap emits `correction_hint` (user is mid-
    correction) and the tail is also handed to residue fuzzy so outer
    matching can still see it.
  - Residue (unwrapped plain segments joined by single spaces) is the
    ONLY text fed to fuzzy matching. Wrapped span text is excluded.
"""
from __future__ import annotations
import logging
import re
from typing import Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

from ..models.schemas import Anchor, MessageClassification, Gate
from ..models.enums import MatchTier, MessageFunction, GateOutcome, GateStage
from .corpus import CorpusStore
from .gate_eval import GateEvaluator, PipelineResult
from . import ollama
from .event_log import EventLog

_event_log = EventLog()


# ======================================================================
# Feature vocabularies and regexes
# ======================================================================

CORRECTION_CUES = {
    "scratch that", "never mind", "nevermind", "forget that", "forget it",
    "hold on", "wait", "actually", "oops", "correction",
    "i retract", "retract that", "i meant", "let me rephrase",
    "let me restart", "strike that", "disregard that", "my mistake",
    "no wait", "actually no", "ignore that", "take that back",
}

EMOTE_VERBS = {
    "laughs", "giggles", "chuckles", "cackles", "snickers",
    "smiles", "grins", "beams", "smirks", "frowns",
    "sighs", "groans", "moans", "yawns",
    "nods", "shakes", "shrugs", "tilts",
    "blinks", "stares", "squints", "winks",
    "cries", "weeps", "sobs",
    "blushes", "flushes", "pales",
    "thinks", "ponders", "considers", "wonders",
    "gasps", "gulps", "swallows",
    "waves", "points", "gestures",
    "pauses", "stops", "freezes",
    "leans", "paces", "wanders",
    "facepalms", "facepalm",
    "raises", "lowers", "crosses", "uncrosses",
    "taps", "rubs", "scratches",
    "breathes", "exhales", "inhales", "sniffs", "snorts", "scoffs",
    "rolls", "stretches",
    "laughing", "smiling", "grinning", "sighing", "shrugging",
    "nodding", "staring", "blinking", "thinking",
}

EMOTE_NOUNS = {
    "shock", "surprise", "concern", "relief", "confusion",
    "laugh", "laughter", "sigh", "groan", "yawn",
    "nod", "shrug", "blink", "stare", "wink",
    "cry", "sob", "blush", "gasp",
    "wave", "pause", "breath",
    "smile", "grin", "frown", "smirk",
    "scoff", "stretch",
    "eyebrow", "brow", "eye", "eyes", "head", "hand", "hands",
    "shoulders", "lips", "mouth",
}

# Negative descriptors — vocabulary of dismissive/frustrated adjectives
# and nouns. Split into MILD and HARSH because the response behaviour
# changes when the affect is directed at the model: mild exasperation
# wants playful register-matching, harsh complaint wants actual course
# correction. The user treats the model with decency by default, so
# MILD is the expected register for model-directed affect.
MILD_DESCRIPTORS = {
    "dumb", "stupid", "silly", "foolish", "dense", "clueless",
    "weird", "strange", "odd", "goofy", "confused",
    "annoying", "ridiculous", "absurd", "cringe", "cringey",
    "embarrassing", "embarrassed",
}
HARSH_DESCRIPTORS = {
    "useless", "broken", "terrible", "awful", "horrible", "hopeless",
    "worthless", "garbage", "trash", "cursed", "pointless",
    "wrong", "mistake", "messed", "screwed", "failed",
    "idiot", "idiotic", "moron", "moronic", "fool",
}
NEGATIVE_DESCRIPTORS = MILD_DESCRIPTORS | HARSH_DESCRIPTORS

# Exasperation / sigh interjections — affective markers that function
# like emotes but specifically signal mild, often playful, frustration.
# Combined with second-person they signal affectionate exasperation at
# the model rather than hostile complaint.
SIGH_INTERJECTIONS = {
    "urgh", "ugh", "argh", "grr", "grrr", "hmph", "tsk",
    "pfft", "bleh", "meh", "oof", "yeesh", "sheesh",
}

# Self-directed expletive/frustration tokens — conventionally self-
# referential without needing an explicit pronoun.
SELF_EXPLETIVES = {
    "fml", "kms", "smh", "facepalm",
}

# Idiom prefixes that contain "my" but are NOT self-reference. Stripped
# before first-person detection so `*oh my god english is stupid*`
# correctly reads as world-directed rather than self-directed.
INTERJECTION_IDIOMS = (
    "oh my god", "oh my goodness", "oh my gosh",
    "my god", "my goodness", "good god", "good lord",
    "jesus christ", "oh jesus", "holy shit", "holy cow",
    "for god's sake", "for fuck's sake", "for crying out loud",
)

# Orthographic regexes
_RE_EXT_VOWEL = re.compile(r'([aeiouAEIOU])\1{2,}')
_RE_EXT_CONS = re.compile(r'([bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ])\1{3,}')
_RE_AFFECT_CAPS = re.compile(r'\b[A-Z]{3,}\b')
_RE_MIXED_CAPS = re.compile(r'[a-z][A-Z]|[A-Z]{2,}[a-z][A-Z]')
_RE_REPEATED_PUNCT = re.compile(r'[!?]{2,}|\.{3,}')


# ======================================================================
# OP_01 parser output
# ======================================================================

class Segment(BaseModel):
    kind: str  # "wrap" | "plain"
    text: str


class ParseResult(BaseModel):
    segments: list[Segment] = Field(default_factory=list)
    unclosed_tail: Optional[str] = None
    residue: str = ""


# ======================================================================
# Match records
# ======================================================================

class AnchorMatch(BaseModel):
    anchor_id: str
    tier: MatchTier
    confidence: float
    matched_phrase: str
    match_method: str  # "hard_exact" | "hard_partial" | "exact" | "partial" | "semantic"
    span_index: Optional[int] = None


class WrappedSpan(BaseModel):
    """Unified record for any closed `*...*` span.

    Carries every feature the span triggered, plus a derived
    `primary_reading` label for quick display. The Mirror composes
    interpretations from the full feature set — the label is not
    authoritative.
    """
    span_index: int
    text: str
    anchor_hits: list[AnchorMatch] = Field(default_factory=list)
    features: list[str] = Field(default_factory=list)
    primary_reading: str = "semantic_depth"


class CorpusRef(BaseModel):
    """A reference to a non-anchor corpus object (slab or bundle)."""
    node_id: str
    node_type: str          # "slab" | "bundle"
    matched_phrase: str
    confidence: float = 1.0

class AnchorMatchResult(BaseModel):
    auto_activate: list[AnchorMatch] = Field(default_factory=list)
    candidates: list[AnchorMatch] = Field(default_factory=list)
    weak: list[AnchorMatch] = Field(default_factory=list)
    wrapped_spans: list[WrappedSpan] = Field(default_factory=list)
    corpus_refs: list[CorpusRef] = Field(default_factory=list)
    correction_hint: Optional[str] = None


# ======================================================================
# Feature extractor
# ======================================================================

INTERROGATIVES = {"why", "how", "what", "where", "when", "who", "wtf", "wth"}


def _detect_directionality_and_pragmatics(text: str) -> list[str]:
    """Detect who an affective wrap is aimed at, plus pragmatic modifiers.

    Directionality features (at most one fires):
      - self_directed_affect   "*fml im so dumb*"
      - model_directed_affect  "*urgh why are u the way u are*"
      - world_directed_affect  "*god english is stupid*"

    Pragmatic modifiers (can fire alongside directionality):
      - sigh_interjection      `urgh`, `ugh`, `argh`, `hmph`, `oof`…
      - rhetorical_question    interrogative + affect — NOT asking for
                               an answer; the user knows there isn't one
                               and is venting/teasing

    Ambient emotes (`*sighs*`, `*mild shock*`) return nothing.
    """
    low = text.lower()
    word_set = set(re.findall(r"[a-z']+", low))

    has_neg = bool(word_set & NEGATIVE_DESCRIPTORS)
    has_harsh = bool(word_set & HARSH_DESCRIPTORS)
    has_self_expletive = bool(word_set & SELF_EXPLETIVES)
    has_my_bad = any(p in low for p in ("my bad", "my fault", "my mistake"))
    has_sigh = bool(word_set & SIGH_INTERJECTIONS)

    has_im = any(
        re.search(r"\b" + p + r"\b", low)
        for p in (r"i'm", r"im", r"i\s+am", r"i've", r"ive", r"i'd", r"id", r"i\s+feel")
    )
    low_stripped = low
    for idiom in INTERJECTION_IDIOMS:
        low_stripped = low_stripped.replace(idiom, "")
    has_my = bool(re.search(r"\bmy\b", low_stripped)) or "myself" in low_stripped

    # Second-person: full forms plus "u"/"ur" as standalone tokens.
    has_you = (
        bool(word_set & {"you", "your", "yours", "youre", "u", "ur", "yer"})
        or "you're" in low
    )

    features: list[str] = []

    # Sigh interjection is a standalone affect marker.
    if has_sigh:
        features.append("sigh_interjection")

    # Rhetorical question: the wrap contains an interrogative AND
    # carries affect (sigh, directionality target, negative descriptor).
    # The user isn't asking a literal question — they're venting.
    has_interrogative = bool(word_set & INTERROGATIVES)
    if has_interrogative and (has_sigh or has_neg or has_you or has_im or has_my):
        features.append("rhetorical_question")

    # Directionality — at most one fires.
    if has_self_expletive or has_my_bad or ((has_im or has_my) and has_neg):
        features.append("self_directed_affect")
    elif has_you and (has_neg or has_sigh or has_interrogative):
        # Second-person + (negative descriptor OR sigh interjection OR
        # interrogative) → model-directed. The sigh/interrogative path
        # catches affectionate exasperation like `*urgh why are u the
        # way u are*` which has no negative descriptor at all.
        features.append("model_directed_affect")
        if has_harsh:
            features.append("harsh_descriptor")  # escalates reading
    elif has_neg and not (has_im or has_my) and not has_you:
        features.append("world_directed_affect")

    return features


def _extract_features(text: str, anchor_hit_count: int) -> tuple[list[str], str]:
    """Return (features, primary_reading) for a closed wrapped span."""
    features: list[str] = []
    lower = text.lower()
    words = lower.split()
    word_count = len(words)

    # Structural
    if anchor_hit_count >= 2:
        features.append("ambiguous_anchor")
    elif anchor_hit_count == 1:
        features.append("anchor_hit")

    # Shape
    if word_count <= 5:
        features.append("short")
    elif word_count > 10:
        features.append("long")

    # Lexical — correction cues
    if any(cue in lower for cue in CORRECTION_CUES):
        features.append("correction_cue")

    # Lexical — emote vocabulary
    word_set = set(re.findall(r"[a-z']+", lower))
    if word_set & EMOTE_VERBS or word_set & EMOTE_NOUNS:
        features.append("emote_vocab")

    # Orthographic — operate on ORIGINAL text (case-sensitive)
    if _RE_EXT_VOWEL.search(text):
        features.append("extended_vowels")
    if _RE_EXT_CONS.search(text):
        features.append("extended_consonants")
    if _RE_AFFECT_CAPS.search(text):
        features.append("affect_caps")
    if _RE_MIXED_CAPS.search(text):
        features.append("mixed_caps")
    if _RE_REPEATED_PUNCT.search(text):
        features.append("repeated_punct")

    # Directionality + pragmatic modifiers (sigh, rhetorical_question).
    features.extend(_detect_directionality_and_pragmatics(text))

    # Derive primary_reading. Directionality beats generic affect
    # readings because response behaviour differs sharply by target.
    # Model-directed defaults to EXASPERATION (mild, affectionate) and
    # only escalates to COMPLAINT when a harsh descriptor is present —
    # the user treats the model with decency by default, so hostile
    # complaints are rare and should not be the assumed register.
    fs = set(features)
    if "ambiguous_anchor" in fs:
        primary = "ambiguous_invocation"
    elif "anchor_hit" in fs:
        primary = "explicit_invocation"
    elif "correction_cue" in fs:
        primary = "self_correction"
    elif "self_directed_affect" in fs:
        primary = "self_directed_frustration"
    elif "model_directed_affect" in fs:
        primary = (
            "model_directed_complaint" if "harsh_descriptor" in fs
            else "model_directed_exasperation"
        )
    elif "world_directed_affect" in fs:
        primary = "world_directed_frustration"
    elif fs & {"affect_caps", "extended_vowels", "mixed_caps", "extended_consonants"}:
        primary = "performed_imitation"
    elif "emote_vocab" in fs and "short" in fs:
        primary = "stage_direction"
    elif "sigh_interjection" in fs:
        primary = "ambient_sigh"
    elif "repeated_punct" in fs and "short" in fs:
        primary = "strong_affect"
    else:
        primary = "semantic_depth"

    return features, primary


def compute_wrap_affect_boost(result: "AnchorMatchResult") -> float:
    """Convert orthographic/emote features into an affect_density bump.

    Called from the pipeline after the LLM drift estimate is produced.
    Wrapped spans are a more reliable affect channel than the prose
    itself because users reach for `*sighs*` or `*oh my GAAAHHHD*`
    precisely when their prose is staying neutral but their reaction
    isn't. Each affect-laden wrapped span contributes a small bump,
    capped so a single emote can't dominate the drift window.
    """
    bump = 0.0
    for span in result.wrapped_spans:
        fs = set(span.features)
        if fs & {"affect_caps", "extended_vowels", "mixed_caps",
                 "extended_consonants", "repeated_punct"}:
            bump += 0.12
        elif "emote_vocab" in fs:
            bump += 0.08
    return min(0.35, bump)  # cap so emotes boost but don't dominate


# ======================================================================
# Main matcher
# ======================================================================

class AnchorMatcher:

    def __init__(self, corpus: CorpusStore, *, use_declarative_gates: bool = False):
        self.corpus = corpus
        self._embed_cache: dict[str, list[tuple[str, list[float]]]] = {}
        self._gate_eval = GateEvaluator()
        self._use_declarative = use_declarative_gates
        # Cache the ordered gate pipeline from corpus (sorted by stage order)
        self._gate_pipeline: list[Gate] = []
        self._build_gate_pipeline()

    def _build_gate_pipeline(self) -> None:
        """Build the ordered 3-stage gate pipeline from corpus gates."""
        stage_order = {
            GateStage.FUNCTION: 0,
            GateStage.EXPLICITNESS: 1,
            GateStage.CONFIDENCE: 2,
        }
        self._gate_pipeline = sorted(
            self.corpus.gates.values(),
            key=lambda g: stage_order.get(g.stage, 99),
        )

    async def warm_cache(self, only_new: bool = False) -> None:
        """Embed anchor canonical phrases + aliases for fast matching.

        Args:
          only_new: when True, skip anchors that already have a cache
            entry. Use during bulk promotion — each commit only embeds
            the just-promoted anchor instead of rebuilding the entire
            cache. Turns per-commit cost from O(N) to O(1).
            Default False preserves the original full-rebuild semantics
            for startup and collection-activation callers, where the
            corpus view may have changed under the cache.

        Pre-fix history: this used to walk the entire corpus on every
        call, including from drafts.py after each individual anchor
        promotion. With N anchors in the corpus, the M-th promotion
        re-embedded all M anchors, giving O(M²) total embed work to
        promote M drafts. On the pod run (~800 anchors), that was
        observed at ~51s per anchor wall-time, ~7-8 hours for the
        batch. With ``only_new=True``, each commit embeds exactly one
        anchor — the new one — and the batch finishes in minutes.
        """
        for anchor in self.corpus.anchors.values():
            if only_new and anchor.id in self._embed_cache:
                continue
            phrases = [anchor.canonical_phrase] + anchor.aliases
            embeddings = await ollama.embed(phrases)
            self._embed_cache[anchor.id] = list(zip(phrases, embeddings))

    def _invalidate_cache(self, anchor_id: str) -> None:
        self._embed_cache.pop(anchor_id, None)

    def gate_check(self, anchor: Anchor, classification: MessageClassification) -> bool:
        """§8 — Pure code gate (imperative path).

        Gate policy: anchors should match during normal conversation.
        The gate only BLOCKS matching when:
        - gate_required=True AND the function is not in allowed_functions
          AND the function is not a 'safe default' (object_of_work, neutral,
          rigor_work, meta_schema all allow matching — only affect_release
          is blocked by default since corpus activation during emotional
          venting is usually unwanted).
        """
        if not anchor.match_policy.gate_required:
            return True
        if classification.explicit:
            return True
        if classification.function in anchor.match_policy.allowed_functions:
            return True
        # Safe defaults: allow matching for substantive message types
        # even if not explicitly listed in allowed_functions
        from ..models.enums import MessageFunction
        safe_functions = {
            MessageFunction.OBJECT_OF_WORK,
            MessageFunction.NEUTRAL,
            MessageFunction.RIGOR_WORK,
            MessageFunction.META_SCHEMA,
        }
        if classification.function in safe_functions:
            return True
        return False

    def declarative_gate_check(
        self, anchor: Anchor, classification: MessageClassification,
    ) -> PipelineResult:
        """§8.4 — Declarative gate pipeline (parallel path).

        Runs the 3-stage gate pipeline (FUNCTION → EXPLICITNESS →
        CONFIDENCE) using the simpleeval evaluator. Returns a full
        PipelineResult with outcome, matched rules, and trace.
        """
        context = {"anchor": anchor, "classification": classification}
        return self._gate_eval.run_pipeline(self._gate_pipeline, context)

    def _effective_gate_check(
        self, anchor: Anchor, classification: MessageClassification,
    ) -> bool:
        """Run both gate paths, log divergence, return the authoritative result.

        When `use_declarative_gates` is True, the declarative pipeline is
        authoritative. Otherwise the imperative gate_check() is
        authoritative and the declarative result is logged for comparison.
        """
        imperative = self.gate_check(anchor, classification)

        if not self._gate_pipeline:
            return imperative

        try:
            declarative = self.declarative_gate_check(anchor, classification)
            decl_pass = declarative.allowed
        except Exception as exc:
            logger.error(
                "Declarative gate failed for %s, falling back to imperative: %s",
                anchor.id, exc,
            )
            return imperative

        if imperative != decl_pass:
            logger.warning(
                "Gate divergence on %s: imperative=%s declarative=%s (%s via %s)",
                anchor.id, imperative, declarative.final_outcome.value,
                declarative.final_outcome.value,
                " -> ".join(
                    f"{sr.gate_id}:{sr.outcome.value}"
                    for sr in declarative.stage_results
                ),
            )

        return decl_pass if self._use_declarative else imperative

    # ------------------------------------------------------------------
    # OP_01 parser
    # ------------------------------------------------------------------

    def _parse_segments(self, text: str) -> ParseResult:
        segments: list[Segment] = []
        i = 0
        n = len(text)
        buf = ""
        in_wrap = False
        wrap_buf = ""
        unclosed_tail: Optional[str] = None

        def flush_plain():
            nonlocal buf
            if buf:
                segments.append(Segment(kind="plain", text=buf))
                buf = ""

        while i < n:
            ch = text[i]
            if ch == "*":
                # Treat `**` as plain (markdown bold passthrough).
                if i + 1 < n and text[i + 1] == "*":
                    if in_wrap:
                        wrap_buf += "**"
                    else:
                        buf += "**"
                    i += 2
                    continue

                if not in_wrap:
                    flush_plain()
                    in_wrap = True
                    wrap_buf = ""
                    i += 1
                    continue
                else:
                    content = wrap_buf.strip()
                    if content:
                        segments.append(Segment(kind="wrap", text=content))
                    else:
                        buf += "*" + wrap_buf + "*"
                    in_wrap = False
                    wrap_buf = ""
                    i += 1
                    continue
            else:
                if in_wrap:
                    wrap_buf += ch
                else:
                    buf += ch
                i += 1

        if in_wrap:
            tail = wrap_buf.strip()
            if tail:
                unclosed_tail = tail
                buf += tail
        flush_plain()

        residue = " ".join(s.text for s in segments if s.kind == "plain").strip()
        return ParseResult(
            segments=segments,
            unclosed_tail=unclosed_tail,
            residue=residue,
        )

    # ------------------------------------------------------------------
    # Entrypoint
    # ------------------------------------------------------------------

    async def match_all(
        self,
        user_text: str,
        classification: MessageClassification,
    ) -> AnchorMatchResult:
        result = AnchorMatchResult()

        parsed = self._parse_segments(user_text)
        if parsed.unclosed_tail:
            result.correction_hint = parsed.unclosed_tail

        wrapped_hits: set[str] = set()  # anchors already activated via a span

        # --- Pass A: per-span hard match + feature extraction ---
        wrap_index = -1
        for seg in parsed.segments:
            if seg.kind != "wrap":
                continue
            wrap_index += 1

            span_matches: list[AnchorMatch] = []
            for anchor in self.corpus.anchors.values():
                # Wrapped spans bypass the function gate — the asterisk wrap
                # IS the format gate. Only format_gate="asterisk_wrapped_only"
                # anchors are restricted to wraps, but once inside a wrap,
                # the function classification is irrelevant.
                if anchor.match_policy.format_gate != "asterisk_wrapped_only":
                    if not self._effective_gate_check(anchor, classification):
                        continue
                m = self._hard_match_span(seg.text, anchor)
                if m is not None:
                    m.span_index = wrap_index
                    span_matches.append(m)
            span_matches.sort(key=lambda x: x.confidence, reverse=True)

            # Count only the top-tier hits for ambiguity.
            effective_hits: list[AnchorMatch] = []
            if span_matches:
                top_conf = span_matches[0].confidence
                effective_hits = [
                    m for m in span_matches
                    if abs(m.confidence - top_conf) < 0.05
                ]

            features, primary = _extract_features(
                seg.text, anchor_hit_count=len(effective_hits)
            )

            ws = WrappedSpan(
                span_index=wrap_index,
                text=seg.text,
                anchor_hits=effective_hits,
                features=features,
                primary_reading=primary,
            )
            result.wrapped_spans.append(ws)

            # Activation policy by primary_reading:
            if primary == "explicit_invocation":
                top = effective_hits[0]
                top.tier = MatchTier.EXACT_OR_PARTIAL
                result.auto_activate.append(top)
                wrapped_hits.add(top.anchor_id)
            elif primary == "ambiguous_invocation":
                # Do not auto-activate; surface as candidates for the user.
                for m in effective_hits:
                    m.tier = MatchTier.AMBIGUOUS_FUZZY
                    result.candidates.append(m)
            # Other readings (self_correction, performed_imitation,
            # stage_direction, strong_affect, semantic_depth) carry no
            # activation — the header will inform the Mirror how to
            # interpret them.

        # --- Pass B: residue fuzzy matching ---
        residue = parsed.residue
        if residue:
            for anchor in self.corpus.anchors.values():
                if anchor.id in wrapped_hits:
                    continue
                if not self._effective_gate_check(anchor, classification):
                    continue
                # Hard format gate: asterisk_wrapped_only anchors can
                # only enter via a wrapped span.
                if anchor.match_policy.format_gate == "asterisk_wrapped_only":
                    continue

                match = self._string_match(residue, anchor)
                if match is None:
                    match = await self._semantic_match(residue, anchor)
                if match is None:
                    continue

                policy = anchor.match_policy
                if match.confidence >= policy.min_confidence_exact:
                    match.tier = MatchTier.EXACT_OR_PARTIAL
                    result.auto_activate.append(match)
                elif match.confidence >= policy.min_confidence_fuzzy:
                    match.tier = MatchTier.AMBIGUOUS_FUZZY
                    result.candidates.append(match)
                else:
                    match.tier = MatchTier.WEAK_SEMANTIC
                    result.weak.append(match)

        # --- Pass D: corpus-wide semantic sweep (embedding similarity) ---
        # Catches paraphrases, cultural quotes, and thematic resonance that
        # substring matching can never reach. Bypasses the function gate —
        # like wrapped spans, semantic resonance IS the signal — but requires
        # a higher confidence threshold to compensate for the lack of
        # explicit invocation.
        #
        # Example: "when you can do what I do and you don't, and the bad
        # things happen, they happen because of you" should light up
        # ANCHOR_SPIDERMAN_RESP_v1 ("great power, great responsibility")
        # even though it shares zero keywords.
        already_matched = (
            {m.anchor_id for m in result.auto_activate}
            | {m.anchor_id for m in result.candidates}
            | wrapped_hits
        )
        await self._semantic_sweep(user_text, already_matched, result)

        # --- Pass C: corpus-wide name matching (slabs + bundles) ---
        # Check if the user referenced any slab or bundle by title/name.
        # This catches references like "epistemic floor", "pushback invariant",
        # "archer bundle" that aren't anchors but are still corpus objects.
        full_text_lower = user_text.lower()
        matched_ids = {m.anchor_id for m in result.auto_activate + result.candidates}

        for slab in self.corpus.slabs.values():
            if slab.id in matched_ids:
                continue
            # Match against slab title (cleaned) and ID-derived name
            title = (slab.title or "").lower()
            id_name = slab.id.lower().replace("slab_", "").replace("_v1", "").replace("_", " ")
            for phrase in [title, id_name]:
                if phrase and len(phrase) > 3 and phrase in full_text_lower:
                    result.corpus_refs.append(CorpusRef(
                        node_id=slab.id, node_type="slab",
                        matched_phrase=phrase, confidence=0.9,
                    ))
                    matched_ids.add(slab.id)
                    break

        for bundle in self.corpus.bundles.values():
            if bundle.id in matched_ids:
                continue
            # Match against bundle intent phrases and ID-derived name
            intents = bundle.payload.intent if hasattr(bundle.payload, 'intent') else []
            id_name = bundle.id.lower().replace("bundle_", "").replace("_v1", "").replace("_", " ")
            for phrase in list(intents) + [id_name]:
                phrase_lower = phrase.lower() if phrase else ""
                if phrase_lower and len(phrase_lower) > 3 and phrase_lower in full_text_lower:
                    result.corpus_refs.append(CorpusRef(
                        node_id=bundle.id, node_type="bundle",
                        matched_phrase=phrase_lower, confidence=0.9,
                    ))
                    matched_ids.add(bundle.id)
                    break

        result.auto_activate.sort(key=lambda m: m.confidence, reverse=True)
        result.candidates.sort(key=lambda m: m.confidence, reverse=True)

        # Log (tolerant to older event_log signatures).
        if (result.auto_activate or result.candidates
                or result.wrapped_spans or result.correction_hint):
            try:
                _event_log.log_match_event(
                    auto=[m.model_dump() for m in result.auto_activate],
                    candidates=[m.model_dump() for m in result.candidates],
                    weak_count=len(result.weak),
                    wrapped_spans=[w.model_dump() for w in result.wrapped_spans],
                    correction_hint=result.correction_hint,
                )
            except TypeError:
                _event_log.log_match_event(
                    auto=[m.model_dump() for m in result.auto_activate],
                    candidates=[m.model_dump() for m in result.candidates],
                    weak_count=len(result.weak),
                )

        return result

    # ------------------------------------------------------------------
    # Hard-gated span matcher
    # ------------------------------------------------------------------

    def _hard_match_span(self, span_text: str, anchor: Anchor) -> Optional[AnchorMatch]:
        """OP_01 hard matcher — exact or dominant partial quote only.

        No semantic / embedding matching. No cross-word substring match.
        """
        span_lower = span_text.lower().strip()
        if not span_lower:
            return None
        span_word_count = len(span_lower.split())
        phrases = [anchor.canonical_phrase] + anchor.aliases

        for phrase in phrases:
            phrase_lower = phrase.lower().strip()
            if not phrase_lower:
                continue

            if phrase_lower == span_lower:
                return AnchorMatch(
                    anchor_id=anchor.id,
                    tier=MatchTier.EXACT_OR_PARTIAL,
                    confidence=1.0,
                    matched_phrase=phrase,
                    match_method="hard_exact",
                )
            if re.search(
                r'(^|\b)' + re.escape(phrase_lower) + r'(\b|$)',
                span_lower,
            ):
                return AnchorMatch(
                    anchor_id=anchor.id,
                    tier=MatchTier.EXACT_OR_PARTIAL,
                    confidence=0.95,
                    matched_phrase=phrase,
                    match_method="hard_exact",
                )

            phrase_words = phrase_lower.split()
            if phrase_words and all(
                re.search(r'\b' + re.escape(w) + r'\b', span_lower)
                for w in phrase_words
            ):
                dominance = len(phrase_words) / max(1, span_word_count)
                if dominance >= 0.5:
                    return AnchorMatch(
                        anchor_id=anchor.id,
                        tier=MatchTier.EXACT_OR_PARTIAL,
                        confidence=0.9,
                        matched_phrase=phrase,
                        match_method="hard_partial",
                    )
        return None

    # ------------------------------------------------------------------
    # Residue matchers (unchanged from prior behaviour)
    # ------------------------------------------------------------------

    def _string_match(self, user_text: str, anchor: Anchor) -> Optional[AnchorMatch]:
        text_lower = user_text.lower()
        phrases = [anchor.canonical_phrase] + anchor.aliases

        for phrase in phrases:
            phrase_lower = phrase.lower()
            if phrase_lower in text_lower:
                return AnchorMatch(
                    anchor_id=anchor.id,
                    tier=MatchTier.EXACT_OR_PARTIAL,
                    confidence=1.0,
                    matched_phrase=phrase,
                    match_method="exact",
                )
            words = phrase_lower.split()
            if len(words) > 1 and all(
                re.search(r'\b' + re.escape(w) + r'\b', text_lower) for w in words
            ):
                return AnchorMatch(
                    anchor_id=anchor.id,
                    tier=MatchTier.EXACT_OR_PARTIAL,
                    confidence=0.85,
                    matched_phrase=phrase,
                    match_method="partial",
                )
        return None

    async def _semantic_match(
        self, user_text: str, anchor: Anchor
    ) -> Optional[AnchorMatch]:
        import numpy as np

        user_vec = await ollama.embed_single(user_text)
        user_arr = np.array(user_vec)

        best_sim = 0.0
        best_phrase = ""

        cached = self._embed_cache.get(anchor.id)
        if cached:
            for phrase, emb in cached:
                anc_arr = np.array(emb)
                sim = float(np.dot(user_arr, anc_arr) / (
                    np.linalg.norm(user_arr) * np.linalg.norm(anc_arr)
                ))
                if sim > best_sim:
                    best_sim = sim
                    best_phrase = phrase
        else:
            from . import embeddings
            phrases = [anchor.canonical_phrase] + anchor.aliases
            for phrase in phrases:
                sim = await embeddings.cosine_similarity(user_text, phrase)
                if sim > best_sim:
                    best_sim = sim
                    best_phrase = phrase

        if best_sim >= anchor.match_policy.min_confidence_fuzzy * 0.8:
            return AnchorMatch(
                anchor_id=anchor.id,
                tier=MatchTier.WEAK_SEMANTIC,
                confidence=best_sim,
                matched_phrase=best_phrase,
                match_method="semantic",
            )
        return None

    # ------------------------------------------------------------------
    # Pass D — corpus-wide semantic sweep
    # ------------------------------------------------------------------

    async def _semantic_sweep(
        self,
        user_text: str,
        already_matched: set[str],
        result: AnchorMatchResult,
    ) -> None:
        """Embed full user text once; batch-compare against ALL anchor
        embeddings. Gate-free — semantic resonance is the signal.

        Tiered thresholds:
          ≥ 0.72  → auto_activate  (strong thematic match)
          ≥ 0.58  → candidates     (probable reference)
          ≥ 0.45  → weak           (faint echo, logged only)

        These are deliberately higher than Pass B thresholds because
        Pass D bypasses the function gate. The higher bar compensates.

        Uses the warm_cache when available — if anchors were pre-embedded
        at startup, only the user text needs embedding (1 vector vs N+1).
        """
        import numpy as np

        anchors = [
            a for a in self.corpus.anchors.values()
            if a.id not in already_matched
        ]
        if not anchors:
            return

        # Embed user text (always needed)
        try:
            user_vec = np.array(await ollama.embed_single(user_text))
        except Exception:
            logger.warning("Pass D: user embedding failed, skipping")
            return
        user_norm = np.linalg.norm(user_vec)
        if user_norm < 1e-8:
            return

        # Collect anchor phrase embeddings — prefer warm cache
        # If cache is cold for some anchors, batch-embed the uncached ones
        cached_pairs: list[tuple[str, str, np.ndarray]] = []  # (anchor_id, phrase, vec)
        uncached_phrases: list[str] = []
        uncached_map: list[tuple[str, str]] = []  # (anchor_id, phrase)

        for anchor in anchors:
            if anchor.id in self._embed_cache:
                for phrase, emb in self._embed_cache[anchor.id]:
                    cached_pairs.append((anchor.id, phrase, np.array(emb)))
            else:
                for phrase in [anchor.canonical_phrase] + anchor.aliases:
                    if phrase:
                        uncached_phrases.append(phrase)
                        uncached_map.append((anchor.id, phrase))

        # Batch-embed any uncached phrases
        if uncached_phrases:
            try:
                uncached_vecs = await ollama.embed(uncached_phrases)
                for i, (anchor_id, phrase) in enumerate(uncached_map):
                    cached_pairs.append((anchor_id, phrase, np.array(uncached_vecs[i])))
            except Exception:
                logger.warning("Pass D: anchor embedding failed for %d uncached phrases", len(uncached_phrases))

        logger.debug("[Pass D] %d anchors to sweep, %d cached pairs, %d uncached",
                     len(anchors), len(cached_pairs), len(uncached_phrases))

        # Find best similarity per anchor
        best_per_anchor: dict[str, tuple[float, str]] = {}
        for anchor_id, phrase, vec in cached_pairs:
            norm = np.linalg.norm(vec)
            if norm < 1e-8:
                continue
            sim = float(np.dot(user_vec, vec) / (user_norm * norm))
            prev = best_per_anchor.get(anchor_id)
            if prev is None or sim > prev[0]:
                best_per_anchor[anchor_id] = (sim, phrase)

        # Debug: per-anchor similarities. Cheap-skip when DEBUG isn't enabled
        # so we don't burn the sort + format on every turn at INFO level.
        if logger.isEnabledFor(logging.DEBUG):
            for aid, (sim, phrase) in sorted(best_per_anchor.items(), key=lambda x: -x[1][0]):
                logger.debug("[Pass D]   %s: %.4f (%s)", aid, sim, phrase[:50])

        # Tier the results
        # Thresholds tuned for nomic-embed-text 768-dim cosine similarity.
        # Paraphrases of the same idea score ~0.55-0.65; exact quotes ~0.85+.
        # The candidate band (0.55-0.70) catches "same theme, different words"
        # which is exactly where cultural references and indirect invocations live.
        THRESHOLD_AUTO = 0.70
        THRESHOLD_CANDIDATE = 0.55
        THRESHOLD_WEAK = 0.42

        sweep_hits = []
        for anchor_id, (sim, phrase) in best_per_anchor.items():
            if sim < THRESHOLD_WEAK:
                continue
            match = AnchorMatch(
                anchor_id=anchor_id,
                confidence=round(sim, 4),
                matched_phrase=phrase,
                match_method="semantic_sweep",
                tier=MatchTier.WEAK_SEMANTIC,  # default, overridden below
            )
            if sim >= THRESHOLD_AUTO:
                match.tier = MatchTier.EXACT_OR_PARTIAL
                result.auto_activate.append(match)
            elif sim >= THRESHOLD_CANDIDATE:
                match.tier = MatchTier.AMBIGUOUS_FUZZY
                result.candidates.append(match)
            else:
                result.weak.append(match)
            sweep_hits.append(match)

        if sweep_hits:
            logger.info(
                "Pass D semantic sweep: %d hits — %s",
                len(sweep_hits),
                ", ".join(
                    f"{m.anchor_id}@{m.confidence:.2f}"
                    for m in sorted(sweep_hits, key=lambda x: -x.confidence)[:5]
                ),
            )
