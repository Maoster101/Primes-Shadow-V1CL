"""Conversation miner — extract corpus proposals from conversation exports.

Inspired by MemPalace's mining pipeline, adapted for Prime's Shadow's
corpus ontology (anchors, slabs, bundles).

Pipeline:
  1. Detect format (Claude JSON, ChatGPT JSON, markdown, plain text)
  2. Normalize to exchange pairs [{role, content}]
  3. Chunk into topically coherent segments
  4. Score each chunk against topic keywords
  5. Extract structured proposals (anchors, slabs, bundles)
  6. Return proposals for user review via draft stack

Usage:
    from src.services.convo_miner import ConversationMiner
    miner = ConversationMiner(corpus)
    proposals = await miner.mine(text_or_path, source_label="claude-export")
"""
from __future__ import annotations
import json
import re
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from . import ollama

logger = logging.getLogger(__name__)


# ── Format detection & normalization ───────────────────────────

@dataclass
class Exchange:
    """A single message in normalized form."""
    role: str          # "user" | "assistant" | "system"
    content: str
    index: int = 0     # position in conversation


@dataclass
class ExchangePair:
    """A user message + assistant response paired together."""
    user: Exchange
    assistant: Optional[Exchange] = None
    pair_index: int = 0


def detect_format(raw: str) -> str:
    """Detect conversation export format.

    Returns one of: 'claude_json', 'chatgpt_json', 'markdown', 'plaintext'
    """
    stripped = raw.strip()

    # JSON formats
    if stripped.startswith(("{", "[")):
        try:
            data = json.loads(stripped)

            # Claude export: array of objects with "uuid" and "chat_messages"
            if isinstance(data, list) and data and "chat_messages" in data[0]:
                return "claude_json"

            # Claude export variant: single conversation object
            if isinstance(data, dict) and "chat_messages" in data:
                return "claude_json"

            # ChatGPT export: array of objects with "mapping" key
            if isinstance(data, list) and data and "mapping" in data[0]:
                return "chatgpt_json"

            # ChatGPT single conversation
            if isinstance(data, dict) and "mapping" in data:
                return "chatgpt_json"

            # Generic JSON with messages array
            if isinstance(data, dict) and "messages" in data:
                return "chatgpt_json"
            if isinstance(data, list) and data and isinstance(data[0], dict) and "role" in data[0]:
                return "chatgpt_json"

        except json.JSONDecodeError:
            pass

    # Markdown format: look for ## Human / ## Assistant or **Human:** patterns
    if re.search(r"^#{1,3}\s*(Human|User|Assistant|Claude)\b", stripped, re.MULTILINE):
        return "markdown"
    if re.search(r"^\*\*(Human|User|Assistant|Claude)\*\*:", stripped, re.MULTILINE):
        return "markdown"

    return "plaintext"


def normalize_claude_json(raw: str) -> list[Exchange]:
    """Parse Claude JSON export into exchanges."""
    data = json.loads(raw.strip())

    # Handle array of conversations — take the first or flatten
    if isinstance(data, list):
        conversations = data
    else:
        conversations = [data]

    exchanges = []
    idx = 0
    for convo in conversations:
        messages = convo.get("chat_messages", [])
        for msg in messages:
            sender = msg.get("sender", "").lower()
            text = msg.get("text", "")
            if not text:
                # Some Claude exports nest content in "content" array
                content_blocks = msg.get("content", [])
                if isinstance(content_blocks, list):
                    text = " ".join(
                        b.get("text", "") for b in content_blocks
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                elif isinstance(content_blocks, str):
                    text = content_blocks

            if not text.strip():
                continue

            role = "user" if sender in ("human", "user") else "assistant"
            exchanges.append(Exchange(role=role, content=text.strip(), index=idx))
            idx += 1

    return exchanges


def normalize_chatgpt_json(raw: str) -> list[Exchange]:
    """Parse ChatGPT JSON export into exchanges."""
    data = json.loads(raw.strip())

    # ChatGPT export with "mapping" structure
    if isinstance(data, dict) and "mapping" in data:
        data = [data]
    if isinstance(data, list) and data and "mapping" in data[0]:
        exchanges = []
        idx = 0
        for convo in data:
            mapping = convo.get("mapping", {})
            # Sort by create_time to get chronological order
            nodes = sorted(
                mapping.values(),
                key=lambda n: (n.get("message", {}) or {}).get("create_time") or 0,
            )
            for node in nodes:
                msg = node.get("message")
                if not msg:
                    continue
                role_raw = msg.get("author", {}).get("role", "")
                content = msg.get("content", {})
                if isinstance(content, dict):
                    parts = content.get("parts", [])
                    text = " ".join(str(p) for p in parts if isinstance(p, str))
                elif isinstance(content, str):
                    text = content
                else:
                    continue
                if not text.strip() or role_raw == "system":
                    continue
                role = "user" if role_raw == "user" else "assistant"
                exchanges.append(Exchange(role=role, content=text.strip(), index=idx))
                idx += 1
        return exchanges

    # Simple messages array format
    if isinstance(data, dict) and "messages" in data:
        messages = data["messages"]
    elif isinstance(data, list) and data and "role" in data[0]:
        messages = data
    else:
        return []

    exchanges = []
    for idx, msg in enumerate(messages):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(c) for c in content)
        if not content.strip() or role == "system":
            continue
        exchanges.append(Exchange(role=role, content=content.strip(), index=idx))
    return exchanges


def normalize_markdown(raw: str) -> list[Exchange]:
    """Parse markdown conversation format into exchanges."""
    # Split on role headers: ## Human, ## Assistant, **Human:**, etc.
    pattern = r"(?:^|\n)(?:#{1,3}\s*|\*\*)(Human|User|Assistant|Claude)(?:\*\*)?[:\s]*\n?"
    parts = re.split(pattern, raw, flags=re.IGNORECASE)

    exchanges = []
    idx = 0
    # parts[0] is text before first header (skip), then alternating role/content
    i = 1
    while i + 1 < len(parts):
        role_label = parts[i].strip().lower()
        content = parts[i + 1].strip()
        if content:
            role = "user" if role_label in ("human", "user") else "assistant"
            exchanges.append(Exchange(role=role, content=content, index=idx))
            idx += 1
        i += 2
    return exchanges


def normalize_plaintext(raw: str) -> list[Exchange]:
    """Best-effort parse of unstructured text. Treat as single user message."""
    # Try to split on "User:" / "Assistant:" style patterns
    pattern = r"\n(User|Human|Me|Assistant|AI|Claude|Bot)\s*:\s*"
    parts = re.split(pattern, raw, flags=re.IGNORECASE)

    if len(parts) > 2:
        exchanges = []
        idx = 0
        i = 1
        while i + 1 < len(parts):
            role_label = parts[i].strip().lower()
            content = parts[i + 1].strip()
            if content:
                role = "user" if role_label in ("user", "human", "me") else "assistant"
                exchanges.append(Exchange(role=role, content=content, index=idx))
                idx += 1
            i += 2
        if exchanges:
            return exchanges

    # Fallback: entire text as a single user message
    return [Exchange(role="user", content=raw.strip(), index=0)]


NORMALIZERS = {
    "claude_json": normalize_claude_json,
    "chatgpt_json": normalize_chatgpt_json,
    "markdown": normalize_markdown,
    "plaintext": normalize_plaintext,
}


def normalize(raw: str) -> tuple[str, list[Exchange]]:
    """Detect format and normalize to exchanges. Returns (format_name, exchanges)."""
    fmt = detect_format(raw)
    exchanges = NORMALIZERS[fmt](raw)
    return fmt, exchanges


# ── Exchange pairing ───────────────────────────────────────────

def pair_exchanges(exchanges: list[Exchange]) -> list[ExchangePair]:
    """Group exchanges into user-assistant pairs."""
    pairs = []
    pair_idx = 0
    i = 0
    while i < len(exchanges):
        ex = exchanges[i]
        if ex.role == "user":
            pair = ExchangePair(user=ex, pair_index=pair_idx)
            # Look ahead for assistant response
            if i + 1 < len(exchanges) and exchanges[i + 1].role == "assistant":
                pair.assistant = exchanges[i + 1]
                i += 2
            else:
                i += 1
            pairs.append(pair)
            pair_idx += 1
        else:
            # Orphaned assistant message — create a synthetic pair
            pairs.append(ExchangePair(
                user=Exchange(role="user", content="(context)", index=ex.index),
                assistant=ex,
                pair_index=pair_idx,
            ))
            pair_idx += 1
            i += 1
    return pairs


# ── Topic chunking ─────────────────────────────────────────────

# Keyword banks for topic scoring. Each topic maps to keywords that
# suggest the conversation segment relates to a corpus concept.

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "identity": [
        "identity", "self", "who i am", "personality", "values",
        "character", "core belief", "principle", "fundamental",
    ],
    "regulation": [
        "regulate", "regulation", "boundary", "limit", "constraint",
        "rule", "policy", "governance", "control", "moderation",
        "should not", "must not", "never", "always",
    ],
    "epistemics": [
        "know", "believe", "evidence", "proof", "claim", "hypothesis",
        "fact", "inference", "uncertain", "confidence", "epistemic",
        "truth", "verify", "source", "citation",
    ],
    "affect": [
        "feel", "emotion", "frustrat", "happy", "sad", "anger",
        "joy", "anxious", "stress", "overwhelm", "excite", "passion",
        "care about", "matters to me",
    ],
    "methodology": [
        "method", "process", "workflow", "pipeline", "system",
        "framework", "approach", "technique", "strategy", "pattern",
        "architecture", "design", "implement",
    ],
    "decisions": [
        "decide", "decision", "chose", "choice", "tradeoff",
        "trade-off", "priority", "prioritize", "commit", "settled on",
        "going with", "picked", "opted",
    ],
    "preferences": [
        "prefer", "preference", "like", "dislike", "want",
        "style", "taste", "rather", "favorite", "habit",
    ],
    "problems": [
        "problem", "issue", "bug", "error", "broken", "fix",
        "wrong", "fail", "struggle", "challenge", "difficult",
        "stuck", "blocker",
    ],
}


@dataclass
class ScoredChunk:
    """A segment of conversation scored against topic keywords."""
    pairs: list[ExchangePair]
    topics: dict[str, float]   # topic -> score
    top_topic: str = ""
    combined_text: str = ""


def score_chunk(pairs: list[ExchangePair]) -> ScoredChunk:
    """Score a group of exchange pairs against topic keywords."""
    # Combine all text in this chunk
    texts = []
    for p in pairs:
        texts.append(p.user.content)
        if p.assistant:
            texts.append(p.assistant.content)
    combined = " ".join(texts).lower()
    word_count = max(len(combined.split()), 1)

    # Score each topic
    scores = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in combined)
        # Normalize by number of keywords in the bank (not by text length,
        # since longer texts naturally hit more keywords)
        scores[topic] = count / len(keywords)

    top = max(scores, key=scores.get) if scores else "methodology"
    return ScoredChunk(
        pairs=pairs,
        topics=scores,
        top_topic=top,
        combined_text=combined[:3000],
    )


def chunk_pairs(pairs: list[ExchangePair], chunk_size: int = 4) -> list[ScoredChunk]:
    """Split exchange pairs into overlapping chunks and score each.

    Uses a sliding window with 50% overlap so topic boundaries
    that fall mid-chunk are still captured.
    """
    if not pairs:
        return []

    chunks = []
    step = max(1, chunk_size // 2)  # 50% overlap
    for start in range(0, len(pairs), step):
        window = pairs[start:start + chunk_size]
        if not window:
            break
        chunks.append(score_chunk(window))

    return chunks


# ── Proposal extraction ────────────────────────────────────────

@dataclass
class MiningProposal:
    """A proposed corpus object extracted from conversation."""
    proposal_type: str         # "anchor" | "slab" | "bundle"
    canonical_phrase: str = "" # for anchors
    canonical_text: str = ""   # for slabs
    title: str = ""
    label: str = ""            # for bundles
    aliases: list[str] = field(default_factory=list)
    source_topic: str = ""
    confidence: float = 0.0
    source_pairs: list[int] = field(default_factory=list)  # pair indices
    justification: str = ""


EXTRACTION_PROMPT = """You are a corpus extraction specialist for Prime's Shadow, a neuro-symbolic personal knowledge system.

Given a conversation segment about "{topic}", extract meaningful corpus objects:

1. **Anchors** — Short canonical phrases (2-8 words) that represent durable, reusable concepts the user cares about. These are the "hooks" that the system matches against in future conversations. Include 1-3 aliases (alternative phrasings).

2. **Slabs** — Longer canonical texts (2-10 sentences) that capture a regulation, principle, or domain context the user has articulated. These are the "rules" that shape model behavior.

3. **Bundles** — Groups of related concepts that co-activate (e.g., "identity + regulation" often appear together). Only propose if the segment shows clear conceptual clustering.

Return ONLY valid JSON:
{{
  "proposals": [
    {{
      "type": "anchor",
      "canonical_phrase": "short phrase",
      "aliases": ["alt1", "alt2"],
      "justification": "why this matters",
      "confidence": 0.0-1.0
    }},
    {{
      "type": "slab",
      "title": "brief title",
      "canonical_text": "the full text of the regulation/principle",
      "justification": "why this matters",
      "confidence": 0.0-1.0
    }},
    {{
      "type": "bundle",
      "label": "bundle label",
      "members": ["phrase1", "phrase2"],
      "justification": "why these co-activate",
      "confidence": 0.0-1.0
    }}
  ]
}}

Rules:
- Only extract things the USER stated or clearly endorses (not things the assistant suggested)
- Confidence 0.8+ = user explicitly stated this as a principle/belief
- Confidence 0.5-0.8 = user implied this through behavior or repeated patterns
- Confidence <0.5 = tentative, needs confirmation
- Skip generic/obvious statements that don't reflect personal knowledge
- Maximum 5 proposals per segment

Conversation segment:
{text}
"""


async def extract_proposals_from_chunk(
    chunk: ScoredChunk,
    existing_anchors: list[str],
) -> list[MiningProposal]:
    """Use the LLM to extract structured proposals from a conversation chunk."""
    # Build readable text from the chunk
    text_parts = []
    for pair in chunk.pairs:
        text_parts.append(f"User: {pair.user.content[:500]}")
        if pair.assistant:
            text_parts.append(f"Assistant: {pair.assistant.content[:500]}")
    segment_text = "\n\n".join(text_parts)

    # Skip very short chunks
    if len(segment_text) < 100:
        return []

    prompt = EXTRACTION_PROMPT.format(
        topic=chunk.top_topic,
        text=segment_text[:4000],
    )

    # Add existing anchors as context to avoid duplicates
    if existing_anchors:
        prompt += f"\n\nExisting anchors (avoid duplicating): {', '.join(existing_anchors[:20])}"

    try:
        result = await ollama.structured_extract(prompt)
    except Exception as e:
        logger.warning("Extraction failed for chunk: %s", e)
        return []

    proposals = []
    raw_proposals = result.get("proposals", [])
    if not isinstance(raw_proposals, list):
        return []

    pair_indices = [p.pair_index for p in chunk.pairs]

    for raw in raw_proposals:
        ptype = raw.get("type", "")
        confidence = float(raw.get("confidence", 0.5))

        if ptype == "anchor":
            proposals.append(MiningProposal(
                proposal_type="anchor",
                canonical_phrase=raw.get("canonical_phrase", ""),
                aliases=raw.get("aliases", []),
                source_topic=chunk.top_topic,
                confidence=confidence,
                source_pairs=pair_indices,
                justification=raw.get("justification", ""),
            ))
        elif ptype == "slab":
            proposals.append(MiningProposal(
                proposal_type="slab",
                title=raw.get("title", ""),
                canonical_text=raw.get("canonical_text", ""),
                source_topic=chunk.top_topic,
                confidence=confidence,
                source_pairs=pair_indices,
                justification=raw.get("justification", ""),
            ))
        elif ptype == "bundle":
            proposals.append(MiningProposal(
                proposal_type="bundle",
                label=raw.get("label", ""),
                aliases=raw.get("members", []),
                source_topic=chunk.top_topic,
                confidence=confidence,
                source_pairs=pair_indices,
                justification=raw.get("justification", ""),
            ))

    return proposals


# ── Deduplication ──────────────────────────────────────────────

def deduplicate_proposals(proposals: list[MiningProposal]) -> list[MiningProposal]:
    """Remove near-duplicate proposals, keeping the highest confidence version."""
    seen: dict[str, MiningProposal] = {}
    for p in proposals:
        # Create a dedup key from type + normalized phrase/title
        if p.proposal_type == "anchor":
            key = f"anchor:{p.canonical_phrase.lower().strip()}"
        elif p.proposal_type == "slab":
            key = f"slab:{p.title.lower().strip()}"
        else:
            key = f"bundle:{p.label.lower().strip()}"

        if key not in seen or p.confidence > seen[key].confidence:
            seen[key] = p

    return list(seen.values())


# ── Main miner class ──────────────────────────────────────────

class ConversationMiner:
    """Mine conversation exports for corpus proposals."""

    def __init__(self, corpus):
        self.corpus = corpus

    async def mine(
        self,
        raw_text: str,
        source_label: str = "import",
        min_confidence: float = 0.4,
        chunk_size: int = 4,
    ) -> dict:
        """Full mining pipeline.

        Args:
            raw_text: Raw conversation export (JSON, markdown, or plaintext)
            source_label: Label for provenance tracking
            min_confidence: Minimum confidence threshold for proposals
            chunk_size: Number of exchange pairs per chunk

        Returns:
            Dict with format info, stats, and proposals
        """
        # Step 1: Detect and normalize
        fmt, exchanges = normalize(raw_text)
        if not exchanges:
            return {
                "format": fmt,
                "exchanges": 0,
                "proposals": [],
                "error": "No exchanges found in input",
            }

        # Step 2: Pair exchanges
        pairs = pair_exchanges(exchanges)

        # Step 3: Chunk and score
        chunks = chunk_pairs(pairs, chunk_size=chunk_size)

        # Filter to chunks with meaningful topic scores
        # (at least one topic scores above threshold)
        meaningful_chunks = [
            c for c in chunks
            if max(c.topics.values(), default=0) > 0.1
        ]

        # If no meaningful chunks, use all chunks (fallback)
        if not meaningful_chunks:
            meaningful_chunks = chunks[:10]  # cap at 10

        # Step 4: Extract proposals from each chunk
        existing_phrases = [
            a.canonical_phrase for a in self.corpus.anchors.values()
        ]

        all_proposals = []
        for chunk in meaningful_chunks:
            chunk_proposals = await extract_proposals_from_chunk(chunk, existing_phrases)
            all_proposals.extend(chunk_proposals)

        # Step 5: Deduplicate and filter
        all_proposals = deduplicate_proposals(all_proposals)
        all_proposals = [p for p in all_proposals if p.confidence >= min_confidence]

        # Sort by confidence descending
        all_proposals.sort(key=lambda p: p.confidence, reverse=True)

        # Step 6: Build topic summary
        topic_counts: dict[str, int] = {}
        for chunk in chunks:
            topic_counts[chunk.top_topic] = topic_counts.get(chunk.top_topic, 0) + 1

        return {
            "format": fmt,
            "exchanges": len(exchanges),
            "pairs": len(pairs),
            "chunks_analyzed": len(meaningful_chunks),
            "topic_distribution": topic_counts,
            "proposals": [
                {
                    "type": p.proposal_type,
                    "canonical_phrase": p.canonical_phrase,
                    "canonical_text": p.canonical_text[:200] if p.canonical_text else "",
                    "title": p.title,
                    "label": p.label,
                    "aliases": p.aliases,
                    "source_topic": p.source_topic,
                    "confidence": round(p.confidence, 2),
                    "justification": p.justification,
                    "source_pairs": p.source_pairs,
                }
                for p in all_proposals
            ],
            "proposal_count": len(all_proposals),
            "source_label": source_label,
        }

    async def mine_file(self, path: str | Path, **kwargs) -> dict:
        """Mine a file from disk."""
        path = Path(path)
        if not path.exists():
            return {"error": f"File not found: {path}"}
        raw = path.read_text(encoding="utf-8")
        kwargs.setdefault("source_label", path.name)
        return await self.mine(raw, **kwargs)
