"""AI Mine 2: one source-adapted, graph-native mining pipeline."""
from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

from ..prompts.universal_mining import build_universal_prompt
from . import mining_progress, ollama
from .convo_miner import detect_format, normalize

SOURCE_KINDS = {"auto", "chat", "narrative", "document", "paper"}
EDGE_TYPES = {"INVOKES", "SUPPORTS", "CONFLICTS", "TENSIONS", "SEQUENCE", "PARENT_OF", "LINKS"}

# AI Mine 2 needs a far larger context window than the legacy miners. Its
# segments cap at 6k-14k chars (vs the legacy ~1.8k), the ontology prompt is
# large, the existing-catalog dump rides along, extraction runs with think=True,
# and CONSOLIDATE/RELATE pass the whole candidate set / 16k of evidence back in.
# The global PS_EXTRACT_NUM_CTX default (4096, sized for legacy segments) would
# truncate the model mid-JSON — an unterminated-string parse crash. Size the
# window here instead so we don't inflate KV-cache VRAM on the legacy paths.
# Override with PS_MINE2_NUM_CTX if a large source still truncates (bigger) or
# VRAM is tight on concurrent slots (smaller, e.g. with PS_MINING_PARALLEL=1).
_MINE2_NUM_CTX = int(os.environ.get("PS_MINE2_NUM_CTX", "16384"))


def detect_source_kind(raw: str, requested: str = "auto") -> str:
    requested = (requested or "auto").lower()
    if requested in SOURCE_KINDS - {"auto"}:
        return requested
    fmt = detect_format(raw)
    if fmt != "plaintext" or re.search(r"(?im)^(user|human|assistant|claude|ai)\s*:", raw):
        return "chat"
    if re.search(r"(?im)^\s*(abstract|methods?|methodology|results?|discussion|references)\s*$", raw):
        return "paper"
    if len(re.findall(r"(?m)^#{1,6}\s+|^\d+(?:\.\d+)*[.)]?\s+[A-Z]", raw)) >= 2:
        return "document"
    return "narrative"


def normalize_source(raw: str, kind: str) -> str:
    if kind != "chat":
        return raw.strip()
    _fmt, exchanges = normalize(("\n" + raw) if detect_format(raw) == "plaintext" else raw)
    if not exchanges:
        return raw.strip()
    return "\n\n".join(
        f"[turn {ex.index}] [{ex.role}]\n{ex.content.strip()}"
        for ex in exchanges if ex.content.strip()
    )


def segment_source(text: str, max_chars: int) -> list[str]:
    max_chars = max(2_000, min(20_000, max_chars))
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    pieces: list[str] = []
    for block in blocks or [text.strip()]:
        while len(block) > max_chars:
            cut = block.rfind("\n", 0, max_chars)
            if cut < max_chars // 2:
                cut = block.rfind(". ", 0, max_chars)
                cut = cut + 1 if cut >= max_chars // 2 else max_chars
            pieces.append(block[:cut].strip())
            block = block[cut:].strip()
        if block:
            pieces.append(block)
    segments: list[str] = []
    buf = ""
    for piece in pieces:
        candidate = f"{buf}\n\n{piece}".strip() if buf else piece
        if buf and len(candidate) > max_chars:
            segments.append(buf)
            buf = piece
        else:
            buf = candidate
    if buf:
        segments.append(buf)
    return segments

_MECHANICAL_PATH_ATOM = re.compile(
    r"^(?:(?:segment|chunk|page|paragraph)\s*[-_ ]?\s*\d+|\d+(?:\.\d+)*|part\s+[ivxlcdm]+)$",
    re.IGNORECASE,
)


def normalize_semantic_path(value: Any, fallback_label: str, source_label: str = "") -> list[str]:
    """Return descriptive hierarchy atoms, never transport/chunk locators.

    The extraction transport uses segment numbers for evidence location, but those
    identifiers are not semantic structure and must never become pillar labels.
    """
    if isinstance(value, str):
        raw_parts = [part.strip() for part in re.split(r"\s*/\s*", value)]
    elif isinstance(value, (list, tuple)):
        raw_parts = [str(part).strip() for part in value if part is not None]
    else:
        raw_parts = []

    clean: list[str] = []
    source_key = (source_label or "").strip().casefold()
    for part in raw_parts:
        if not part or part.casefold() == source_key or _MECHANICAL_PATH_ATOM.fullmatch(part):
            continue
        clean.append(part)

    if clean:
        return clean
    fallback = re.sub(r"\s+", " ", str(fallback_label or "Extracted concept")).strip()
    return [fallback]

def _source_ref_key(ref: dict) -> str:
    locator = re.sub(r"\s+", " ", str(ref.get("locator") or "")).strip().casefold()
    quote = re.sub(r"\s+", " ", str(ref.get("quote") or "")).strip().casefold()
    return f"{locator}\0{quote}"


def evidence_source_order(item: dict, segment: str, segment_index: int, fallback_index: int) -> int:
    """Locate candidate evidence in the server-owned source segment."""
    folded = segment.casefold()
    offsets: list[int] = []
    for ref in item.get("source_refs") or []:
        if not isinstance(ref, dict):
            continue
        quote = str(ref.get("quote") or "").strip().casefold()
        if not quote:
            continue
        fragments = [part.strip(" .…") for part in re.split(r"\.{3}|…", quote)]
        for fragment in fragments:
            if len(fragment) < 16:
                continue
            position = folded.find(fragment)
            if position >= 0:
                offsets.append(position)
                break
    offset = min(offsets) if offsets else fallback_index * 1_000
    return (segment_index + 1) * 1_000_000_000 + offset

def candidate_source_order(item: dict, fallback_index: int = 0) -> int:
    """Recover stable document order without asking the model to infer it."""
    explicit = item.get("_source_order")
    try:
        if explicit is not None:
            return int(explicit)
    except (TypeError, ValueError):
        pass
    temp_id = str(item.get("temp_id") or "")
    match = re.match(r"^s(\d+).*?(\d+)$", temp_id, re.IGNORECASE)
    if match:
        return int(match.group(1)) * 1_000_000_000 + int(match.group(2))
    return 10**12 + fallback_index


def deterministic_sequence_edges(candidates: list[dict]) -> list[dict]:
    """Connect finalized slabs in source order; model judgment is not involved."""
    slabs = [item for item in candidates if str(item.get("type", "")).lower() == "slab"]
    slabs.sort(key=lambda item: candidate_source_order(item))
    edges: list[dict] = []
    for source, target in zip(slabs, slabs[1:]):
        source_ref = str(source.get("temp_id") or "")
        target_ref = str(target.get("temp_id") or "")
        source_label = str(source.get("title") or "").strip()
        target_label = str(target.get("title") or "").strip()
        if not source_ref or not target_ref or source_ref == target_ref or not source_label or not target_label:
            continue
        edges.append({
            "type": "SEQUENCE",
            "from_ref": source_ref,
            "to_ref": target_ref,
            "from": source_label,
            "to": target_label,
            "confidence": 0.98,
            "justification": "Deterministic order of consecutive mined slabs in the source.",
            "deterministic": True,
        })
    return edges

def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9_'-]{3,}", text.lower())}


def relevant_catalog(store: Any, text: str, limit: int = 40) -> list[dict]:
    query, rows = _words(text), []
    for node_type, objects in (("anchor", store.anchors.values()), ("slab", store.slabs.values())):
        for obj in objects:
            label = getattr(obj, "canonical_phrase", None) or getattr(obj, "title", None) or ""
            summary = getattr(obj, "canonical_text", None) or ""
            score = len(query & _words(f"{label} {summary}"))
            if score:
                rows.append((score, {"id": obj.id, "type": node_type, "label": label, "summary": summary[:300]}))
    rows.sort(key=lambda row: (-row[0], row[1]["id"]))
    return [row for _score, row in rows[:limit]]


def consolidate_exact(candidates: list[dict]) -> list[dict]:
    merged: dict[tuple[str, str], dict] = {}
    for item in candidates:
        kind = str(item.get("type", "")).lower()
        label = item.get("canonical_phrase") if kind == "anchor" else item.get("title")
        key = (kind, re.sub(r"\s+", " ", str(label or "")).strip().casefold())
        if kind not in {"anchor", "slab"} or not key[1]:
            continue
        if key not in merged:
            merged[key] = item
            continue
        prior = merged[key]
        prior["source_refs"] = prior.get("source_refs", []) + item.get("source_refs", [])
        prior["aliases"] = list(dict.fromkeys(prior.get("aliases", []) + item.get("aliases", [])))
        prior["confidence"] = max(float(prior.get("confidence", 0)), float(item.get("confidence", 0)))
    return list(merged.values())


class UniversalMiner:
    def __init__(self, store: Any):
        self.store = store

    async def mine(self, raw_text: str, source_label: str, source_kind: str = "auto",
                   min_confidence: float = 0.5, segment_chars: int = 0) -> dict:
        mining_progress.reset("universal")
        try:
            kind = detect_source_kind(raw_text, source_kind)
            normalized = normalize_source(raw_text, kind)
            cap = segment_chars or (14_000 if ollama.extract_is_hosted() else 6_000)
            segments = segment_source(normalized, cap)
            catalog = relevant_catalog(self.store, normalized)
            mining_progress.set_phase("extracting", len(segments))
            semaphore = asyncio.Semaphore(ollama.mining_parallelism())

            async def extract_one(index: int, segment: str) -> dict:
                async with semaphore:
                    try:
                        result = await ollama.structured_extract(build_universal_prompt(
                            task="EXTRACT", source_kind=kind,
                            source_path=f"{source_label}/segment-{index + 1}",
                            existing_catalog=catalog,
                            source_material=f"[segment {index + 1}]\n{segment}",
                        ), think=True, num_ctx=_MINE2_NUM_CTX)
                    except Exception as exc:
                        # A single segment the model mangled (e.g. truncated JSON
                        # on a dense passage) must not sink the whole mine. Skip
                        # it with a warning; the other segments still land.
                        result = {"candidates": [], "relationships": [], "warnings": [
                            {"locator": f"segment {index + 1}", "issue": f"Extraction skipped: {exc}"}
                        ]}
                    mining_progress.increment()
                    return result

            raw_results = await asyncio.gather(*(
                extract_one(index, segment) for index, segment in enumerate(segments)
            ))
            candidates: list[dict] = []
            relationships: list[dict] = []
            warnings: list[Any] = []
            source_order_by_ref: dict[str, int] = {}
            for segment_index, result in enumerate(raw_results):
                warnings.extend(result.get("warnings", []))
                for item_index, item in enumerate(result.get("candidates", [])):
                    if not isinstance(item, dict):
                        continue
                    refs = item.get("source_refs") or []
                    if not refs or all(
                        ref.get("authority") == "assistant_unadopted"
                        for ref in refs if isinstance(ref, dict)
                    ):
                        continue
                    try:
                        confidence = float(item.get("confidence", 0))
                    except (TypeError, ValueError):
                        confidence = 0
                    if confidence < min_confidence:
                        continue
                    item["confidence"] = confidence
                    label = item.get("canonical_phrase") or item.get("title") or "Extracted concept"
                    item["source_path"] = normalize_semantic_path(
                        item.get("source_path"), str(label), source_label,
                    )
                    item["_source_order"] = evidence_source_order(
                        item, segments[segment_index], segment_index, item_index,
                    )
                    for ref in item.get("source_refs") or []:
                        if isinstance(ref, dict):
                            key = _source_ref_key(ref)
                            previous = source_order_by_ref.get(key)
                            if previous is None or item["_source_order"] < previous:
                                source_order_by_ref[key] = item["_source_order"]
                    item["temp_id"] = f"s{segment_index + 1}_{item.get('temp_id') or item_index + 1}"
                    candidates.append(item)
                relationships.extend(rel for rel in result.get("relationships", []) if isinstance(rel, dict))
            candidates = consolidate_exact(candidates)

            if len(segments) > 1 and len(candidates) > 1:
                mining_progress.set_phase("consolidating", 1)
                try:
                    result = await ollama.structured_extract(build_universal_prompt(
                        task="CONSOLIDATE", source_kind=kind, source_path=source_label,
                        existing_catalog=catalog, source_material=json.dumps(candidates, ensure_ascii=False),
                    ), think=True, num_ctx=_MINE2_NUM_CTX)
                    if isinstance(result.get("candidates"), list):
                        candidates = consolidate_exact(result["candidates"])
                        for fallback_index, item in enumerate(candidates):
                            label = item.get("canonical_phrase") or item.get("title") or "Extracted concept"
                            item["source_path"] = normalize_semantic_path(
                                item.get("source_path"), str(label), source_label,
                            )
                            recovered_orders = [
                                source_order_by_ref[_source_ref_key(ref)]
                                for ref in item.get("source_refs") or []
                                if isinstance(ref, dict) and _source_ref_key(ref) in source_order_by_ref
                            ]
                            item["_source_order"] = (
                                min(recovered_orders) if recovered_orders
                                else candidate_source_order(item, fallback_index)
                            )
                    warnings.extend(result.get("warnings", []))
                except Exception as exc:
                    warnings.append({"locator": None, "issue": f"Consolidation fallback used: {exc}"})
                mining_progress.increment()

            for fallback_index, item in enumerate(candidates):
                item["_source_order"] = candidate_source_order(item, fallback_index)

            if candidates:
                mining_progress.set_phase("edge_extraction", 1)
                try:
                    result = await ollama.structured_extract(build_universal_prompt(
                        task="RELATE", source_kind=kind, source_path=source_label,
                        existing_catalog=catalog,
                        source_material=json.dumps({"candidates": candidates, "evidence": normalized[:16_000]}, ensure_ascii=False),
                    ), think=True, num_ctx=_MINE2_NUM_CTX)
                    relationships.extend(rel for rel in result.get("relationships", []) if isinstance(rel, dict))
                    warnings.extend(result.get("warnings", []))
                except Exception as exc:
                    warnings.append({"locator": None, "issue": f"Relationship pass skipped: {exc}"})
                mining_progress.increment()

            proposals, labels = self._compat_proposals(candidates, source_label)
            # Source order is structural, not an LLM judgment. Discard any
            # model-proposed SEQUENCE edges and replace them with one exact spine.
            edges = [
                edge for edge in self._compat_edges(relationships, labels, catalog)
                if edge.get("type") != "SEQUENCE"
            ]
            existing = {(edge.get("type"), edge.get("from"), edge.get("to")) for edge in edges}
            for edge in deterministic_sequence_edges(candidates):
                key = (edge["type"], edge["from"], edge["to"])
                if key not in existing:
                    edges.append(edge)
                    existing.add(key)
            mining_progress.mark_done()
            return {
                "status": "OK" if proposals else "EMPTY_VALID", "source_kind": kind,
                "source_format": detect_format(raw_text), "extract_model": ollama.EXTRACT_MODEL,
                "hosted": ollama.extract_is_hosted(), "segments": len(segments),
                "candidates": candidates, "relationships": relationships,
                "proposals": proposals, "edges": edges, "warnings": warnings,
                "stats": {"segments": len(segments), "candidates": len(proposals), "relationships": len(edges)},
            }
        except Exception as exc:
            mining_progress.mark_error(str(exc))
            raise

    @staticmethod
    def _compat_proposals(candidates: list[dict], source_label: str) -> tuple[list[dict], dict[str, str]]:
        proposals, labels = [], {}
        for item in candidates:
            kind = str(item.get("type", "")).lower()
            label = item.get("canonical_phrase") if kind == "anchor" else item.get("title")
            text = item.get("canonical_text")
            if (kind == "anchor" and not label) or (kind == "slab" and (not label or not text)):
                continue
            labels[str(item.get("temp_id"))] = str(label)
            proposal = dict(item)
            proposal.update({"type": kind, "source_label": source_label,
                             "source_topic": item.get("source_path") or [source_label]})
            proposals.append(proposal)
        return proposals, labels

    @staticmethod
    def _compat_edges(relationships: list[dict], labels: dict[str, str], catalog: list[dict]) -> list[dict]:
        known = {row["id"]: row["label"] for row in catalog}
        known.update(labels)
        out, seen = [], set()
        for rel in relationships:
            edge_type = str(rel.get("type", "")).upper()
            source = known.get(str(rel.get("from_ref")))
            target = known.get(str(rel.get("to_ref")))
            key = (edge_type, source, target)
            if edge_type not in EDGE_TYPES or not source or not target or source == target or key in seen:
                continue
            seen.add(key)
            out.append({"type": edge_type, "from": source, "to": target,
                        "confidence": rel.get("confidence", 0.5), "justification": rel.get("justification", "")})
        return out
