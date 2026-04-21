"""Dreaming pass — grounding audit + content rewrite for draft proposals.

Sits between mining (extract_proposals) and commit (review_draft).
The miner sees only chat text, so its output may contain confabulated
field names, vague paraphrases, or missing context. The dreaming pass:

  1. Detects grounding mode (CODE vs DOMAIN)
  2. Retrieves the original chat context (source_turns)
  3. Retrieves reference material:
     - CODE mode: embeds draft text, retrieves top-K code/config chunks
     - DOMAIN mode: expands chat window for full conversational context
  4. Runs a two-stage model call: grounding audit → content rewrite
  5. Writes .enriched.json + .dream_log.md alongside the draft

Two grounding modes:

  CODE   — draft references system internals (functions, config, schemas).
           Ground against actual source code via embedding retrieval.
           Detect confabulated field names, wrong behaviors, invented APIs.

  DOMAIN — draft captures external knowledge from natural conversation
           (e.g., terra preta soil science, narrative structure analysis).
           Ground against the conversation itself + existing corpus.
           Detect lost nuance, vague paraphrasing, missed specificity.

Mode detection: embed the draft text, retrieve top code chunks. If the
best chunk scores below a threshold, the draft isn't about code → DOMAIN.

The enriched packet is NOT auto-promoted — the reviewer must still
explicitly accept it (cp enriched → json) before commit. This keeps
the human in the loop for the grounding judgment.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..models.schemas import DraftPacket
from ..prompts.dreaming import (
    GROUNDING_AUDIT_PROMPT,
    REWRITE_PROMPT,
    MODE_INSTRUCTIONS_CODE,
    MODE_INSTRUCTIONS_DOMAIN,
    MODE_REWRITE_CODE,
    MODE_REWRITE_DOMAIN,
)
from .session_store import SessionStore
from .corpus import CorpusStore
from .atomic_io import atomic_write_text
from . import ollama

# Dedicated small/fast model for dream enrichment. Dreams are text-to-text
# transforms (grounding audit + rewrite) that don't need the reasoning
# strength of the main chat model. Routing them to a small model (Llama
# 3.2 3B ≈ 2 GB) instead of gemma3:12b gives roughly 5-10× throughput at
# minimal quality cost, AND fits alongside the chat model in 16 GB VRAM
# so Ollama doesn't have to swap models per call.
#
# Overridable via env var for easy experimentation without editing code.
# Set to empty string to disable override (fall back to main chat model).
import os as _os
DREAM_MODEL: Optional[str] = _os.environ.get("PS_DREAM_MODEL", "llama3.2:latest") or None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Code directories to index for retrieval
_CODE_DIRS = [
    "src/services",
    "src/models",
    "src/prompts",
    "src/api",
]
_CONFIG_GLOBS = [
    "app/corpora/*/state/*.yaml",
    "app/corpora/*/objects/*.yaml",
]
# File extensions to index
_CODE_EXTENSIONS = {".py", ".yaml", ".yml"}
# Maximum chunk size (characters) for code snippets
_CHUNK_MAX_CHARS = 1500
# Number of top-K code chunks to retrieve
_TOP_K_CHUNKS = 8
# Number of top-K corpus nodes to include as context
_TOP_K_CORPUS = 5
# Code-relevance threshold: if the best code chunk scores below this,
# the draft is about domain knowledge, not system internals.
# Calibrated from observed scores: code-referencing drafts score 0.65+,
# domain drafts score 0.45-0.55 against code chunks.
_CODE_MODE_THRESHOLD = 0.60
# How many extra chat turns to fetch in DOMAIN mode (wider window)
_DOMAIN_CHAT_WINDOW = 20


# ---------------------------------------------------------------------------
# Code chunking
# ---------------------------------------------------------------------------

def _chunk_python_file(path: Path, content: str) -> list[dict]:
    """Split a Python file into function/class-level chunks.

    Each chunk is a dict with:
      - path: str (relative file path)
      - name: str (function/class name or "module_header")
      - text: str (the code text)
      - start_line: int
    """
    lines = content.split("\n")
    chunks: list[dict] = []
    current_chunk_lines: list[str] = []
    current_name = "module_header"
    current_start = 1

    def _flush():
        if current_chunk_lines:
            text = "\n".join(current_chunk_lines)
            if len(text.strip()) > 20:  # skip trivially empty chunks
                chunks.append({
                    "path": str(path),
                    "name": current_name,
                    "text": text[:_CHUNK_MAX_CHARS],
                    "start_line": current_start,
                })

    for i, line in enumerate(lines, 1):
        stripped = line.lstrip()
        # New function or class boundary
        if stripped.startswith(("def ", "class ", "async def ")):
            _flush()
            current_chunk_lines = [line]
            current_start = i
            # Extract name
            for prefix in ("async def ", "def ", "class "):
                if stripped.startswith(prefix):
                    rest = stripped[len(prefix):]
                    current_name = rest.split("(")[0].split(":")[0].strip()
                    break
        else:
            current_chunk_lines.append(line)

    _flush()
    return chunks


def _chunk_yaml_file(path: Path, content: str) -> list[dict]:
    """Split a YAML file into section-level chunks.

    Uses top-level keys as section boundaries. For large YAML files
    (like anchors.yaml), each top-level list item becomes a chunk.
    """
    # For corpus object files (list of items), chunk by item
    lines = content.split("\n")
    chunks: list[dict] = []

    if content.lstrip().startswith("- "):
        # List-style YAML (anchors, slabs, etc.)
        current_lines: list[str] = []
        current_start = 1
        for i, line in enumerate(lines, 1):
            if line.startswith("- ") and current_lines:
                text = "\n".join(current_lines)
                if len(text.strip()) > 20:
                    # Extract id if present
                    name = "item"
                    for cl in current_lines:
                        if "id:" in cl:
                            name = cl.split("id:")[1].strip().strip("'\"")
                            break
                    chunks.append({
                        "path": str(path),
                        "name": name,
                        "text": text[:_CHUNK_MAX_CHARS],
                        "start_line": current_start,
                    })
                current_lines = [line]
                current_start = i
            else:
                current_lines.append(line)
        # Flush last item
        if current_lines:
            text = "\n".join(current_lines)
            if len(text.strip()) > 20:
                name = "item"
                for cl in current_lines:
                    if "id:" in cl:
                        name = cl.split("id:")[1].strip().strip("'\"")
                        break
                chunks.append({
                    "path": str(path),
                    "name": name,
                    "text": text[:_CHUNK_MAX_CHARS],
                    "start_line": current_start,
                })
    else:
        # Map-style YAML (frame_policy.yaml, etc.) — treat whole file as one chunk
        chunks.append({
            "path": str(path),
            "name": path.stem,
            "text": content[:_CHUNK_MAX_CHARS * 2],
            "start_line": 1,
        })

    return chunks


# ---------------------------------------------------------------------------
# Code index (built on demand, cached per session)
# ---------------------------------------------------------------------------

class _CodeIndex:
    """Lightweight in-memory index of code chunks with embeddings.

    Built lazily on first use, then cached. For the project's current
    size (~50 source files) this takes ~2-3 seconds to embed everything.
    """

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.chunks: list[dict] = []  # each has path, name, text, start_line
        self.embeddings: list[list[float]] = []  # parallel to chunks
        self._built = False

    def _discover_files(self) -> list[Path]:
        """Find all indexable code/config files."""
        files: list[Path] = []
        for code_dir in _CODE_DIRS:
            d = self.project_root / code_dir
            if d.exists():
                for f in sorted(d.rglob("*")):
                    if f.suffix in _CODE_EXTENSIONS and f.is_file():
                        files.append(f)
        # Config files via globs
        for pattern in _CONFIG_GLOBS:
            for f in sorted(self.project_root.glob(pattern)):
                if f.is_file() and f not in files:
                    files.append(f)
        return files

    def _chunk_file(self, path: Path) -> list[dict]:
        """Read and chunk a single file."""
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return []
        rel = str(path.relative_to(self.project_root)).replace("\\", "/")
        if path.suffix == ".py":
            chunks = _chunk_python_file(Path(rel), content)
        elif path.suffix in (".yaml", ".yml"):
            chunks = _chunk_yaml_file(Path(rel), content)
        else:
            chunks = [{
                "path": rel,
                "name": path.stem,
                "text": content[:_CHUNK_MAX_CHARS],
                "start_line": 1,
            }]
        return chunks

    async def build(self) -> None:
        """Discover files, chunk them, and embed all chunks."""
        if self._built:
            return
        files = self._discover_files()
        all_chunks: list[dict] = []
        for f in files:
            all_chunks.extend(self._chunk_file(f))

        if not all_chunks:
            self._built = True
            return

        # Embed in batches (ollama embed supports batch input)
        texts = [f"{c['path']}::{c['name']}\n{c['text'][:500]}" for c in all_chunks]
        batch_size = 32
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                embs = await ollama.embed(batch)
                all_embeddings.extend(embs)
            except Exception as e:
                print(f"[DREAM] Embedding batch {i}-{i+len(batch)} failed: {e}")
                # Fill with zeros so indices stay aligned
                all_embeddings.extend([[0.0] * 768] * len(batch))

        self.chunks = all_chunks
        self.embeddings = all_embeddings
        self._built = True
        print(f"[DREAM] Code index built: {len(self.chunks)} chunks from {len(files)} files")

    async def retrieve(self, query_text: str, top_k: int = _TOP_K_CHUNKS) -> list[dict]:
        """Retrieve top-K code chunks most similar to the query text.

        Returns chunks sorted by descending cosine similarity, each
        augmented with a 'score' field.
        """
        if not self._built:
            await self.build()
        if not self.chunks:
            return []

        query_emb = await ollama.embed_single(query_text)
        scored: list[tuple[float, dict]] = []
        for i, chunk in enumerate(self.chunks):
            emb = self.embeddings[i]
            score = _cosine_sim(query_emb, emb)
            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, chunk in scored[:top_k]:
            results.append({**chunk, "score": round(score, 4)})
        return results


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Corpus context builder
# ---------------------------------------------------------------------------

def _build_corpus_context(
    corpus: CorpusStore, query_text: str, top_k: int = _TOP_K_CORPUS
) -> str:
    """Build a text summary of the most relevant existing corpus nodes.

    Simple keyword overlap scoring (not embedding-based) — fast and
    good enough for finding related nodes when the draft text contains
    real terms that also appear in committed nodes.
    """
    query_words = set(query_text.lower().split())

    scored: list[tuple[float, str]] = []
    for anchor in corpus.anchors.values():
        text = f"{anchor.canonical_phrase} {' '.join(anchor.aliases)} {anchor.notes}"
        words = set(text.lower().split())
        overlap = len(query_words & words)
        if overlap > 0:
            summary = (
                f"[ANCHOR] {anchor.id}\n"
                f"  phrase: {anchor.canonical_phrase}\n"
                f"  aliases: {anchor.aliases}\n"
                f"  notes: {anchor.notes[:200]}"
            )
            scored.append((overlap, summary))

    for slab in corpus.slabs.values():
        text = f"{slab.title} {slab.canonical_text}"
        words = set(text.lower().split())
        overlap = len(query_words & words)
        if overlap > 0:
            summary = (
                f"[SLAB] {slab.id}\n"
                f"  title: {slab.title}\n"
                f"  text: {slab.canonical_text[:200]}"
            )
            scored.append((overlap, summary))

    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return "(no related corpus nodes found)"
    return "\n\n".join(s for _, s in scored[:top_k])


# ---------------------------------------------------------------------------
# DreamingPass service
# ---------------------------------------------------------------------------

# Module-level index cache (rebuilt once per process lifetime)
_code_index: Optional[_CodeIndex] = None


class DreamingPass:
    """Grounding audit + content rewrite for draft proposals.

    Usage:
        dreamer = DreamingPass(corpus, session_store, chat_store)
        result = await dreamer.dream(session_id, draft_id)

    Result dict:
        {"status": "enriched", "audit": {...}, "rewrite": {...}}
        {"status": "already_grounded", "audit": {...}}
        {"status": "error", "error": "..."}
    """

    def __init__(
        self,
        corpus: CorpusStore,
        session_store: SessionStore,
        chat_store,  # ChatStore — imported lazily to avoid circular deps
        project_root: Optional[Path] = None,
    ):
        self.corpus = corpus
        self.session_store = session_store
        self.chat_store = chat_store
        self.project_root = project_root or Path(__file__).resolve().parent.parent.parent

    async def _get_code_index(self) -> _CodeIndex:
        """Get or build the code index (cached at module level)."""
        global _code_index
        if _code_index is None:
            _code_index = _CodeIndex(self.project_root)
        if not _code_index._built:
            await _code_index.build()
        return _code_index

    def _load_chat_context(
        self, chat_id: str, source_turns: list[int], expanded: bool = False
    ) -> str:
        """Load chat turns that sourced this draft.

        Args:
            chat_id: The chat to load from.
            source_turns: Specific turn indices the miner cited.
            expanded: If True (DOMAIN mode), load a wider window around
                the source turns to capture full conversational context
                that the miner may have summarized too aggressively.
        """
        try:
            messages = self.chat_store.get_messages(chat_id)
        except Exception:
            return "(chat not found or unreadable)"

        if not messages:
            return "(empty chat)"

        if expanded and source_turns:
            # DOMAIN mode: load a wide window around the cited turns
            min_turn = max(0, min(source_turns) - 3)
            max_turn = min(len(messages), max(source_turns) + 4)
            # Also include last N turns for context
            window_indices = set(range(min_turn, max_turn))
            tail_start = max(0, len(messages) - _DOMAIN_CHAT_WINDOW)
            window_indices |= set(range(tail_start, len(messages)))
            indices = sorted(window_indices)
        elif source_turns:
            indices = sorted(source_turns)
        else:
            # No source turns at all — use last N
            indices = list(range(max(0, len(messages) - 8), len(messages)))

        lines = []
        char_limit = 800 if expanded else 500
        for idx in indices:
            if 0 <= idx < len(messages):
                m = messages[idx]
                role = getattr(m, "role", "unknown")
                content = getattr(m, "content", str(m))
                cited = " [CITED]" if idx in (source_turns or []) else ""
                lines.append(
                    f"[turn {idx}] [{role}]{cited} {content[:char_limit]}"
                )

        if not lines:
            for i, m in enumerate(messages[-6:]):
                role = getattr(m, "role", "unknown")
                content = getattr(m, "content", str(m))
                lines.append(f"[turn {len(messages)-6+i}] [{role}] {content[:400]}")

        return "\n".join(lines)

    def _get_companion_drafts(
        self, session_id: str, exclude_id: str
    ) -> list[DraftPacket]:
        """Get other drafts from the same session (for cross-linking)."""
        all_packets = self.session_store.list_draft_packets(session_id)
        return [p for p in all_packets if p.id != exclude_id]

    def _format_companions(self, companions: list[DraftPacket]) -> str:
        """Format companion drafts as context for the rewrite prompt."""
        if not companions:
            return "(no companion drafts in this session)"
        lines = []
        for p in companions:
            inline = p.anchor or p.slab or p.bundle
            lines.append(
                f"[{p.packet_type or 'unknown'}] {p.id}\n"
                f"  content: {json.dumps(inline, default=str)[:300]}"
            )
        return "\n\n".join(lines)

    async def dream(self, session_id: str, draft_id: str) -> dict:
        """Run the full dreaming pass on a single draft.

        Automatically detects grounding mode:
          - CODE mode if the draft text is semantically close to source code
          - DOMAIN mode if the draft is about external knowledge

        Returns a result dict with status and details. Does NOT
        auto-promote the enriched output — the reviewer must still
        accept it explicitly.
        """
        t0 = time.time()

        # --- Load draft ---
        packet = self.session_store.load_draft_packet(session_id, draft_id)
        if not packet:
            return {"status": "error", "error": f"Draft {draft_id} not found"}

        draft_type = packet.packet_type or "anchor"
        inline = packet.anchor if draft_type == "anchor" else packet.slab
        if not inline:
            # Try to build inline from raw
            raw_path = (
                self.session_store.drafts_dir(session_id)
                / f"{draft_id}_raw.json"
            )
            raw = self.session_store.read_json(raw_path) or {}
            if not raw:
                return {"status": "error", "error": "No inline dict or raw data"}
            inline = raw  # use raw as fallback

        # --- Build query text for retrieval ---
        if draft_type == "anchor":
            query = (
                f"{inline.get('canonical_phrase', '')} "
                f"{' '.join(inline.get('aliases', []))} "
                f"{inline.get('notes', '')}"
            )
        else:
            query = (
                f"{inline.get('title', '')} "
                f"{inline.get('canonical_text', '')}"
            )
        query = query[:600]  # cap for embedding

        # --- Retrieve code snippets + detect grounding mode ---
        code_index = await self._get_code_index()
        code_chunks = await code_index.retrieve(query, top_k=_TOP_K_CHUNKS)
        top_score = code_chunks[0]["score"] if code_chunks else 0.0
        grounding_mode = "CODE" if top_score >= _CODE_MODE_THRESHOLD else "DOMAIN"
        print(
            f"[DREAM] Mode detection for {draft_id}: "
            f"top_code_score={top_score:.4f} -> {grounding_mode}"
        )

        # --- Build reference material based on mode ---
        if grounding_mode == "CODE":
            reference_material = "\n\n".join(
                f"### {c['path']}::{c['name']} (line {c['start_line']}, score={c['score']})\n"
                f"```\n{c['text']}\n```"
                for c in code_chunks
            )
            mode_instructions = MODE_INSTRUCTIONS_CODE
            mode_rewrite = MODE_REWRITE_CODE
        else:
            # DOMAIN mode: reference material is the expanded chat context.
            # Code chunks are still available but demoted — the primary
            # grounding source is the conversation itself.
            expanded_chat = self._load_chat_context(
                packet.source_chat_id,
                packet.source_turns or [],
                expanded=True,
            )
            reference_material = (
                "### Expanded conversation context\n"
                f"{expanded_chat}"
            )
            # Append top 3 code chunks with low relevance warning
            if code_chunks and top_score > 0.40:
                reference_material += (
                    "\n\n### Potentially relevant code (low confidence)\n"
                    + "\n\n".join(
                        f"#### {c['path']}::{c['name']} (score={c['score']})\n"
                        f"```\n{c['text'][:500]}\n```"
                        for c in code_chunks[:3]
                    )
                )
            mode_instructions = MODE_INSTRUCTIONS_DOMAIN
            mode_rewrite = MODE_REWRITE_DOMAIN

        # --- Load chat context (always included, narrower in CODE mode) ---
        chat_context = self._load_chat_context(
            packet.source_chat_id,
            packet.source_turns or [],
            expanded=(grounding_mode == "DOMAIN"),
        )

        # --- Build corpus context ---
        corpus_context = _build_corpus_context(self.corpus, query)

        # --- Get companion drafts ---
        companions = self._get_companion_drafts(session_id, draft_id)
        companions_text = self._format_companions(companions)

        # =====================================================================
        # Stage 1: Grounding audit
        # =====================================================================
        audit_prompt = (
            GROUNDING_AUDIT_PROMPT
            .replace("$GROUNDING_MODE", grounding_mode)
            .replace("$MODE_INSTRUCTIONS", mode_instructions)
            .replace("$DRAFT_TYPE", draft_type)
            .replace("$DRAFT_ID", draft_id)
            .replace("$DRAFT_CONTENT", json.dumps(inline, indent=2, default=str))
            .replace("$JUSTIFICATION", packet.justification or "")
            .replace("$CHAT_CONTEXT", chat_context)
            .replace("$REFERENCE_MATERIAL", reference_material)
            .replace("$CORPUS_CONTEXT", corpus_context)
            .replace("{$DRAFT_TYPE}", draft_type)
        )

        print(f"[DREAM] Stage 1: grounding audit for {draft_id} ({grounding_mode})"
              + (f" [model={DREAM_MODEL}]" if DREAM_MODEL else "")
              + "...")
        try:
            audit_result = await ollama.structured_extract(audit_prompt, model=DREAM_MODEL)
        except Exception as e:
            return {"status": "error", "error": f"Audit model call failed: {e}"}

        verdict = audit_result.get("verdict", "UNKNOWN")
        rewrite_needed = audit_result.get("rewrite_needed", False)
        print(f"[DREAM] Audit verdict: {verdict}, rewrite_needed: {rewrite_needed}")

        # If already grounded, skip rewrite
        if verdict == "GROUNDED" and not rewrite_needed:
            self._write_dream_log(
                session_id, draft_id, draft_type, grounding_mode,
                audit_result, None, code_chunks, time.time() - t0,
            )
            return {
                "status": "already_grounded",
                "grounding_mode": grounding_mode,
                "audit": audit_result,
            }

        # =====================================================================
        # Stage 2: Content rewrite
        # =====================================================================
        rewrite_prompt = (
            REWRITE_PROMPT
            .replace("$GROUNDING_MODE", grounding_mode)
            .replace("$MODE_REWRITE_INSTRUCTIONS", mode_rewrite)
            .replace("$DRAFT_TYPE", draft_type)
            .replace("$DRAFT_ID", draft_id)
            .replace("$DRAFT_CONTENT", json.dumps(inline, indent=2, default=str))
            .replace("$AUDIT_RESULT", json.dumps(audit_result, indent=2, default=str))
            .replace("$REFERENCE_MATERIAL", reference_material)
            .replace("$COMPANION_DRAFTS", companions_text)
            .replace("{$DRAFT_TYPE}", draft_type)
        )

        print(f"[DREAM] Stage 2: rewriting {draft_id} ({grounding_mode})"
              + (f" [model={DREAM_MODEL}]" if DREAM_MODEL else "")
              + "...")
        try:
            rewritten = await ollama.structured_extract(rewrite_prompt, model=DREAM_MODEL)
        except Exception as e:
            return {
                "status": "error",
                "error": f"Rewrite model call failed: {e}",
                "audit": audit_result,
            }

        # --- Build enriched packet ---
        enriched = packet.model_dump(mode="json")
        enriched["justification"] = (
            f"Rewritten by dreaming pass ({grounding_mode} mode). "
            f"Audit verdict: {verdict}. "
            f"See {draft_id}.dream_log.md for details."
        )

        if draft_type == "anchor":
            enriched["anchor"] = rewritten
        elif draft_type == "slab":
            enriched["slab"] = rewritten

        # Write .enriched.json
        enriched_path = (
            self.session_store.drafts_dir(session_id)
            / f"{draft_id}.enriched.json"
        )
        self.session_store.write_json(enriched_path, enriched)

        # Write dream log
        self._write_dream_log(
            session_id, draft_id, draft_type, grounding_mode,
            audit_result, rewritten, code_chunks, time.time() - t0,
        )

        elapsed = time.time() - t0
        print(f"[DREAM] Done: {draft_id} enriched in {elapsed:.1f}s ({grounding_mode})")

        return {
            "status": "enriched",
            "grounding_mode": grounding_mode,
            "audit": audit_result,
            "rewrite": rewritten,
            "enriched_path": str(enriched_path),
            "elapsed_seconds": round(elapsed, 1),
        }

    def _write_dream_log(
        self,
        session_id: str,
        draft_id: str,
        draft_type: str,
        grounding_mode: str,
        audit: dict,
        rewrite: Optional[dict],
        code_chunks: list[dict],
        elapsed: float,
    ) -> None:
        """Write a human-readable dream log alongside the draft."""
        log_path = (
            self.session_store.drafts_dir(session_id)
            / f"{draft_id}.dream_log.md"
        )
        lines = [
            f"# Dream log — `{draft_id}`",
            f"",
            f"- **timestamp:** {datetime.now(timezone.utc).isoformat()}Z",
            f"- **elapsed:** {elapsed:.1f}s",
            f"- **type:** {draft_type}",
            f"- **grounding_mode:** {grounding_mode}",
            f"- **verdict:** {audit.get('verdict', 'UNKNOWN')}",
            f"- **rewrite_needed:** {audit.get('rewrite_needed', False)}",
            f"",
            f"## Grounding audit",
            f"",
            f"### Grounded claims",
        ]
        for claim in audit.get("grounded_claims", []):
            lines.append(f"- {claim}")
        lines.append("")
        lines.append("### Confabulated claims")
        for claim in audit.get("confabulated_claims", []):
            lines.append(f"- {claim}")
        lines.append("")
        lines.append("### Term corrections")
        for old, new in audit.get("real_terms", {}).items():
            lines.append(f"- `{old}` → `{new}`")
        lines.append("")
        lines.append("### Grounding sources")
        for src in audit.get("grounding_sources", []):
            lines.append(f"- `{src}`")
        lines.append("")
        redundant = audit.get("redundant_with", [])
        if redundant:
            lines.append("### Redundancy warning")
            lines.append(f"This draft may overlap with: {', '.join(redundant)}")
            lines.append("")
        lines.append(f"### Notes")
        lines.append(audit.get("notes", "(none)"))
        lines.append("")

        if rewrite:
            lines.append("## Rewritten content")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(rewrite, indent=2, default=str))
            lines.append("```")
            lines.append("")

        lines.append("## Code chunks retrieved")
        lines.append("")
        for c in code_chunks[:5]:
            lines.append(
                f"- `{c['path']}::{c['name']}` (line {c['start_line']}, "
                f"score={c['score']})"
            )

        lines.append("")
        lines.append("## Reviewer instructions")
        lines.append("")
        lines.append("To accept this rewrite:")
        lines.append("```bash")
        lines.append(
            f'cp "app/workbench/sessions/{session_id}/drafts/{draft_id}.enriched.json" \\'
        )
        lines.append(
            f'   "app/workbench/sessions/{session_id}/drafts/{draft_id}.json"'
        )
        lines.append("```")
        lines.append("")
        lines.append("To reject: delete `.enriched.json` and `.dream_log.md`.")
        lines.append("")

        atomic_write_text(log_path, "\n".join(lines))


async def dream_all_pending(
    corpus: CorpusStore,
    session_store: SessionStore,
    chat_store,
    session_id: str,
    project_root: Optional[Path] = None,
) -> list[dict]:
    """Run the dreaming pass on all DRAFT_UNAUTHORIZED packets in a session.

    Convenience function for batch processing. Returns a list of
    result dicts, one per draft.
    """
    dreamer = DreamingPass(corpus, session_store, chat_store, project_root)
    packets = session_store.list_draft_packets(session_id)
    results = []

    for packet in packets:
        if packet.status.value != "DRAFT_UNAUTHORIZED":
            continue
        # Skip if already enriched
        enriched_path = (
            session_store.drafts_dir(session_id)
            / f"{packet.id}.enriched.json"
        )
        if enriched_path.exists():
            results.append({
                "draft_id": packet.id,
                "status": "skipped",
                "reason": "enriched file already exists",
            })
            continue

        result = await dreamer.dream(session_id, packet.id)
        result["draft_id"] = packet.id
        results.append(result)

    return results
