"""MCP Server — expose Prime's Shadow as Model Context Protocol tools.

Implements JSON-RPC 2.0 over stdin/stdout (no library needed).
Any MCP client (Claude Code, Cursor, etc.) can discover and call
Prime's Shadow corpus, frame, and event tools.

Usage:
    python -m src.mcp_server

Protocol flow:
    Client -> initialize       -> Server returns capabilities
    Client -> tools/list       -> Server returns tool definitions
    Client -> tools/call       -> Server dispatches to handler, returns result
    Client -> notifications/*  -> Server ignores (no-op)
"""
from __future__ import annotations
import json
import sys
import logging
from pathlib import Path
from typing import Any, Callable

# ── Bootstrap imports ──────────────────────────────────────────
# We import lazily inside handlers to avoid slow startup on tool
# discovery calls (MCP clients call tools/list before tools/call).
_corpus = None
_event_log = None
_frame_manager = None
_anchor_matcher = None


def _boot():
    """Lazy-init service instances on first tools/call."""
    global _corpus, _event_log, _frame_manager, _anchor_matcher
    if _corpus is not None:
        return

    from src.services.corpus import CorpusStore
    from src.services.event_log import EventLog
    from src.services.frame_manager import FrameManager
    from src.services.anchor_matcher import AnchorMatcher

    _corpus = CorpusStore()
    errors = _corpus.load()
    if errors:
        logging.warning("Corpus validation errors: %s", errors)

    _event_log = EventLog()
    _frame_manager = FrameManager(_corpus)
    _anchor_matcher = AnchorMatcher(_corpus)


# ── Tool registry ──────────────────────────────────────────────
# Each tool: {description, input_schema (JSON Schema), handler}

TOOLS: dict[str, dict[str, Any]] = {}


def _tool(name: str, description: str, schema: dict):
    """Decorator: register a function as an MCP tool."""
    def decorator(fn: Callable):
        TOOLS[name] = {
            "description": description,
            "input_schema": schema,
            "handler": fn,
        }
        return fn
    return decorator


# ── Tool implementations ───────────────────────────────────────

@_tool(
    "ps_status",
    "Get Prime's Shadow corpus status: object counts, validation errors, health.",
    {"type": "object", "properties": {}, "required": []},
)
def ps_status(args: dict) -> dict:
    _boot()
    errors = _corpus.validate()
    return {
        "anchors": len(_corpus.anchors),
        "slabs": len(_corpus.slabs),
        "bundles": len(_corpus.bundles),
        "edges": len(_corpus.edges),
        "gates": len(_corpus.gates),
        "valid": len(errors) == 0,
        "errors": errors[:10],
    }


@_tool(
    "ps_list_anchors",
    "List all anchors in the corpus with their canonical phrases, invokes targets, and match policy.",
    {
        "type": "object",
        "properties": {
            "include_match_policy": {
                "type": "boolean",
                "description": "Include confidence thresholds in output",
                "default": False,
            }
        },
        "required": [],
    },
)
def ps_list_anchors(args: dict) -> dict:
    _boot()
    include_mp = args.get("include_match_policy", False)
    result = []
    for aid, a in _corpus.anchors.items():
        entry = {
            "id": aid,
            "canonical_phrase": a.canonical_phrase,
            "invokes": a.invokes,
            "aliases": a.aliases,
        }
        if include_mp:
            entry["match_policy"] = {
                "min_confidence_exact": a.match_policy.min_confidence_exact,
                "min_confidence_fuzzy": a.match_policy.min_confidence_fuzzy,
                "gate_required": a.match_policy.gate_required,
            }
        result.append(entry)
    return {"anchors": result, "count": len(result)}


@_tool(
    "ps_list_slabs",
    "List all slabs with their type, lifecycle status, and OLI mode gating.",
    {"type": "object", "properties": {}, "required": []},
)
def ps_list_slabs(args: dict) -> dict:
    _boot()
    result = []
    for sid, s in _corpus.slabs.items():
        result.append({
            "id": sid,
            "title": s.title,
            "type": str(s.type.value) if hasattr(s.type, "value") else str(s.type),
            "lifecycle_status": str(s.lifecycle_status.value) if hasattr(s.lifecycle_status, "value") else str(s.lifecycle_status),
            "requires_oli_mode": s.requires_oli_mode.value if s.requires_oli_mode else None,
            "version": s.version,
            "text_length": len(s.canonical_text),
        })
    return {"slabs": result, "count": len(result)}


@_tool(
    "ps_list_bundles",
    "List all key bundles with their supports targets.",
    {"type": "object", "properties": {}, "required": []},
)
def ps_list_bundles(args: dict) -> dict:
    _boot()
    result = []
    for bid, b in _corpus.bundles.items():
        result.append({
            "id": bid,
            "intent": b.payload.intent if hasattr(b, "payload") else [],
            "supports": b.supports,
            "version": b.version,
        })
    return {"bundles": result, "count": len(result)}


@_tool(
    "ps_get_slab",
    "Get a slab's full canonical text and metadata by ID.",
    {
        "type": "object",
        "properties": {
            "slab_id": {"type": "string", "description": "The slab ID to retrieve"}
        },
        "required": ["slab_id"],
    },
)
def ps_get_slab(args: dict) -> dict:
    _boot()
    sid = args["slab_id"]
    s = _corpus.slabs.get(sid)
    if not s:
        return {"error": f"Slab not found: {sid}", "available": list(_corpus.slabs.keys())}
    return {
        "id": s.id,
        "title": s.title,
        "canonical_text": s.canonical_text,
        "type": str(s.type.value) if hasattr(s.type, "value") else str(s.type),
        "lifecycle_status": str(s.lifecycle_status.value) if hasattr(s.lifecycle_status, "value") else str(s.lifecycle_status),
        "requires_oli_mode": s.requires_oli_mode.value if s.requires_oli_mode else None,
        "version": s.version,
        "links": {
            "anchors": s.links.anchors if hasattr(s.links, "anchors") else [],
            "bundles": s.links.bundles if hasattr(s.links, "bundles") else [],
        },
        "depends_on": s.depends_on,
        "assumptions": s.assumptions,
    }


@_tool(
    "ps_match_anchor",
    "Run anchor matching against user text. Returns matched anchors with confidence scores, features, and tier.",
    {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to match against anchors"}
        },
        "required": ["text"],
    },
)
def ps_match_anchor(args: dict) -> dict:
    """Keyword-based anchor matching (synchronous, no gate checks).

    For full async matching with classification gates, use the
    HTTP API POST /chats/{id}/messages instead.
    """
    _boot()
    text = args["text"].lower()

    matches = []
    for aid, anchor in _corpus.anchors.items():
        # Check canonical phrase
        phrase = anchor.canonical_phrase.lower()
        if phrase in text or text in phrase:
            matches.append({
                "anchor_id": aid,
                "canonical_phrase": anchor.canonical_phrase,
                "match_type": "canonical",
                "invokes": anchor.invokes,
            })
            continue
        # Check aliases
        for alias in anchor.aliases:
            if alias.lower() in text:
                matches.append({
                    "anchor_id": aid,
                    "canonical_phrase": anchor.canonical_phrase,
                    "matched_alias": alias,
                    "match_type": "alias",
                    "invokes": anchor.invokes,
                })
                break

    return {"matches": matches, "match_count": len(matches)}


@_tool(
    "ps_corpus_search",
    "Search corpus objects (anchors, slabs, bundles) by keyword in IDs, phrases, and text content.",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search keyword"},
            "object_types": {
                "type": "array",
                "items": {"type": "string", "enum": ["anchors", "slabs", "bundles", "gates"]},
                "description": "Which object types to search (default: all)",
            }
        },
        "required": ["query"],
    },
)
def ps_corpus_search(args: dict) -> dict:
    _boot()
    query = args["query"].lower()
    types = args.get("object_types", ["anchors", "slabs", "bundles", "gates"])
    results = []

    if "anchors" in types:
        for aid, a in _corpus.anchors.items():
            searchable = f"{aid} {a.canonical_phrase} {' '.join(a.aliases)}".lower()
            if query in searchable:
                results.append({"type": "anchor", "id": aid, "label": a.canonical_phrase})

    if "slabs" in types:
        for sid, s in _corpus.slabs.items():
            searchable = f"{sid} {s.title} {s.canonical_text[:200]}".lower()
            if query in searchable:
                results.append({"type": "slab", "id": sid, "label": s.title or sid})

    if "bundles" in types:
        for bid, b in _corpus.bundles.items():
            intent_str = " ".join(b.payload.intent) if hasattr(b, "payload") else ""
            searchable = f"{bid} {intent_str}".lower()
            if query in searchable:
                results.append({"type": "bundle", "id": bid, "label": bid})

    if "gates" in types:
        for gid, g in _corpus.gates.items():
            if query in gid.lower():
                results.append({"type": "gate", "id": gid, "label": gid})

    return {"results": results, "count": len(results)}


@_tool(
    "ps_get_frame",
    "Get the current frame state for a session: active nodes, salience scores, anchors, bundles, slabs.",
    {
        "type": "object",
        "properties": {
            "session_id": {"type": "string", "description": "Session ID to inspect"}
        },
        "required": ["session_id"],
    },
)
def ps_get_frame(args: dict) -> dict:
    _boot()
    sid = args["session_id"]
    frame = _frame_manager._frames.get(sid)
    if not frame:
        return {"error": f"No active frame for session: {sid}"}
    return {
        "session_id": sid,
        "active_nodes": frame.active_nodes,
        "active_anchors": dict(frame.active_anchors),
        "active_bundles": dict(frame.active_bundles),
        "active_slabs": dict(frame.active_slabs),
        "active_concepts": dict(frame.active_concepts),
        "salience": {k: round(v, 3) for k, v in frame.salience_smoothed.items()},
        "mismatch_score": round(frame.mismatch_score, 3),
        "node_count": len(frame.active_nodes),
    }


@_tool(
    "ps_query_events",
    "Read recent events from a log file (push_events, drift_events, gate_events, match_events, frame_events, etc.).",
    {
        "type": "object",
        "properties": {
            "log_name": {
                "type": "string",
                "description": "Log file name (e.g. 'push_events.jsonl', 'drift_events.jsonl', 'gate_events.jsonl')",
            },
            "last_n": {
                "type": "integer",
                "description": "Number of recent events to return (default 20)",
                "default": 20,
            }
        },
        "required": ["log_name"],
    },
)
def ps_query_events(args: dict) -> dict:
    _boot()
    log_name = args["log_name"]
    last_n = args.get("last_n", 20)
    events = _event_log.read_recent(log_name, last_n=last_n)
    return {"log_name": log_name, "events": events, "count": len(events)}


@_tool(
    "ps_get_anchor",
    "Get full details of a single anchor by ID.",
    {
        "type": "object",
        "properties": {
            "anchor_id": {"type": "string", "description": "The anchor ID to retrieve"}
        },
        "required": ["anchor_id"],
    },
)
def ps_get_anchor(args: dict) -> dict:
    _boot()
    aid = args["anchor_id"]
    a = _corpus.anchors.get(aid)
    if not a:
        return {"error": f"Anchor not found: {aid}", "available": list(_corpus.anchors.keys())}
    return {
        "id": a.id,
        "canonical_phrase": a.canonical_phrase,
        "invokes": a.invokes,
        "aliases": a.aliases,
        "match_policy": {
            "gate_required": a.match_policy.gate_required,
            "min_confidence_exact": a.match_policy.min_confidence_exact,
            "min_confidence_fuzzy": a.match_policy.min_confidence_fuzzy,
            "allowed_functions": [f.value for f in a.match_policy.allowed_functions],
        },
        "depends_on": a.depends_on,
        "version": a.meta.version if hasattr(a, "meta") else None,
    }


@_tool(
    "ps_reverse_deps",
    "Get all objects that depend on or reference a given node ID (cascade index).",
    {
        "type": "object",
        "properties": {
            "node_id": {"type": "string", "description": "Node ID to find dependents for"}
        },
        "required": ["node_id"],
    },
)
def ps_reverse_deps(args: dict) -> dict:
    _boot()
    nid = args["node_id"]
    deps = _corpus.get_reverse_deps(nid)
    return {"node_id": nid, "dependents": deps, "count": len(deps)}


@_tool(
    "ps_validate_corpus",
    "Run all 6 corpus validation checks from section 24.1 and return any errors.",
    {"type": "object", "properties": {}, "required": []},
)
def ps_validate_corpus(args: dict) -> dict:
    _boot()
    errors = _corpus.validate()
    return {"valid": len(errors) == 0, "errors": errors, "error_count": len(errors)}


@_tool(
    "ps_base_set",
    "Get the base set of slabs for a given OLI mode (the slabs that will be included in every prompt).",
    {
        "type": "object",
        "properties": {
            "oli_mode": {
                "type": "string",
                "enum": ["ON", "OFF"],
                "description": "OLI mode (default OFF)",
                "default": "OFF",
            }
        },
        "required": [],
    },
)
def ps_base_set(args: dict) -> dict:
    _boot()
    from src.models.enums import OLIMode
    mode = OLIMode(args.get("oli_mode", "OFF"))
    slabs = _corpus.base_set_slabs(mode)
    return {
        "oli_mode": mode.value,
        "slabs": [
            {"id": s.id, "title": s.title, "type": s.type.value if hasattr(s.type, "value") else str(s.type)}
            for s in slabs
        ],
        "count": len(slabs),
    }


# ── JSON-RPC 2.0 dispatch ─────────────────────────────────────

SERVER_INFO = {
    "name": "primes-shadow",
    "version": "1.0.0",
}

SERVER_CAPABILITIES = {
    "tools": {},  # we support tools/list and tools/call
}


def _make_response(id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": id, "result": result}


def _make_error(id: Any, code: int, message: str, data: Any = None) -> dict:
    err = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": id, "error": err}


def handle_request(req: dict) -> dict | None:
    """Dispatch a single JSON-RPC request. Returns response dict or None for notifications."""
    method = req.get("method", "")
    id_ = req.get("id")  # None for notifications
    params = req.get("params", {})

    # Notifications (no id) — acknowledge silently
    if id_ is None:
        return None

    if method == "initialize":
        return _make_response(id_, {
            "protocolVersion": "2024-11-05",
            "capabilities": SERVER_CAPABILITIES,
            "serverInfo": SERVER_INFO,
        })

    elif method == "tools/list":
        tool_list = []
        for name, spec in TOOLS.items():
            tool_list.append({
                "name": name,
                "description": spec["description"],
                "inputSchema": spec["input_schema"],
            })
        return _make_response(id_, {"tools": tool_list})

    elif method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})

        if tool_name not in TOOLS:
            return _make_error(id_, -32602, f"Unknown tool: {tool_name}")

        try:
            result = TOOLS[tool_name]["handler"](tool_args)
            return _make_response(id_, {
                "content": [{"type": "text", "text": json.dumps(result, default=str, ensure_ascii=False)}],
            })
        except Exception as e:
            return _make_response(id_, {
                "content": [{"type": "text", "text": json.dumps({"error": str(e)})}],
                "isError": True,
            })

    elif method == "ping":
        return _make_response(id_, {})

    else:
        return _make_error(id_, -32601, f"Method not found: {method}")


def main():
    """Main loop: read JSON-RPC from stdin, write responses to stdout."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,  # MCP logs go to stderr, not stdout
    )

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            err = _make_error(None, -32700, f"Parse error: {e}")
            sys.stdout.write(json.dumps(err) + "\n")
            sys.stdout.flush()
            continue

        response = handle_request(req)
        if response is not None:
            sys.stdout.write(json.dumps(response, default=str, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
