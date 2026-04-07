"""Ollama inference service — GPT-OSS 20B + nomic-embed-text.

LLM = sensor, code = actuator. This service is the sensor interface.
The model proposes; calling code decides what to do with proposals.
"""
from __future__ import annotations
import json
import logging
import os
import re
import httpx
from typing import Optional

logger = logging.getLogger(__name__)

OLLAMA_BASE = "http://localhost:11434"
CHAT_MODEL = os.environ.get("PS_CHAT_MODEL", "gpt-oss:20b")
EMBED_MODEL = "nomic-embed-text"

# ── Performance tuning ─────────────────────────────────────────
# These options are merged into every Ollama request.
#
#   num_ctx    — Context window size in tokens. Must match our budget.
#                20B model + 32k ctx fits in 16GB VRAM comfortably.
#                Higher values (64k/128k) need more VRAM for KV cache.
#   num_gpu    — Number of model layers to offload to GPU.
#                -1 = all layers (full GPU offload). The 13GB model
#                fits in 16GB VRAM with room for KV cache at 32k ctx.
#   num_batch  — Prompt evaluation batch size. Higher = faster prompt
#                processing at cost of slightly more VRAM. 1024 is
#                good for a 16GB card.
#   num_thread — CPU threads for non-GPU ops. Match physical cores.
#
# Environment overrides (set before starting the server):
#   OLLAMA_FLASH_ATTENTION=1  — Enables flash attention (faster, less VRAM)
#   OLLAMA_KEEP_ALIVE=-1      — Never unload model from VRAM
#   OLLAMA_NUM_PARALLEL=2     — Concurrent request slots

_NUM_CTX = int(os.environ.get("PS_NUM_CTX", "16384"))  # 16k default (saves VRAM for KV cache)

# num_gpu: layers offloaded to GPU. -1 = all, 0 = CPU only.
# Auto-detect: check free VRAM and allocate layers proportionally.
# Override with PS_NUM_GPU env var (e.g. PS_NUM_GPU=8).
_NUM_GPU = int(os.environ.get("PS_NUM_GPU", "-1"))

def _detect_gpu_layers() -> int:
    """Auto-detect how many of the model's 24 layers fit in free VRAM."""
    if _NUM_GPU != -1:
        return _NUM_GPU
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        ).strip()
        free_mb = int(out.split("\n")[0])
        # ~550 MB per layer for 20B model.
        # Reserve 1200 MB for embed model (~600MB) + KV cache + system.
        usable = max(0, free_mb - 1200)
        layers = min(24, usable // 550)
        if layers < 1:
            return 0  # CPU only — not enough VRAM
        return layers
    except Exception:
        return 0  # fallback to CPU if detection fails

_AUTO_GPU_LAYERS = _detect_gpu_layers()

# Detect physical CPU cores for thread count
def _detect_cpu_threads() -> int:
    try:
        import os
        # Use physical cores (not hyperthreads) for Ollama
        cores = os.cpu_count() or 8
        return max(4, cores)
    except Exception:
        return 8

MODEL_OPTIONS: dict = {
    "num_ctx": _NUM_CTX,
    "num_gpu": _AUTO_GPU_LAYERS,
    "num_batch": 1024,      # higher batch = faster prompt eval
    "num_thread": _detect_cpu_threads(),
}

# Reusable client — partial GPU offload means slow cold starts (~60s model load)
# connect=30s, read=300s (5 min for first-token during cold load + slow CPU layers)
_client = httpx.AsyncClient(base_url=OLLAMA_BASE, timeout=httpx.Timeout(300.0, connect=30.0))


def _opts(temperature: float = 0.7, **extra) -> dict:
    """Build options dict: MODEL_OPTIONS + per-call overrides."""
    o = {**MODEL_OPTIONS, "temperature": temperature}
    o.update(extra)
    return o


async def generate(
    prompt: str,
    system: Optional[str] = None,
    temperature: float = 0.7,
    raw_json: bool = False,
) -> str:
    """Single-shot generation. Use for structured extraction tasks."""
    payload: dict = {
        "model": CHAT_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": _opts(temperature),
    }
    if system:
        payload["system"] = system
    if raw_json:
        payload["format"] = "json"

    resp = await _client.post("/api/generate", json=payload)
    resp.raise_for_status()
    return resp.json()["response"]


async def chat(
    messages: list[dict],
    temperature: float = 0.7,
    think: bool = False,
) -> dict:
    """Multi-turn chat completion. Returns full response object.

    Args:
        messages: List of {role, content} dicts.
        think: Enable extended reasoning (harmony format 'Reasoning: high').
    """
    payload: dict = {
        "model": CHAT_MODEL,
        "messages": messages,
        "stream": False,
        "options": _opts(temperature),
    }
    if think:
        payload["options"]["think"] = True

    resp = await _client.post("/api/chat", json=payload)
    resp.raise_for_status()
    return resp.json()


BROWSER_TOOLS = [
    {"type": "function", "function": {
        "name": "browser.search",
        "description": "Search the web for current information",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "search query"},
        }, "required": ["query"]},
    }},
]


async def chat_stream(
    messages: list[dict],
    temperature: float = 0.7,
    think: bool = False,
    think_level: str = "medium",
    web_mode: str = "off",
):
    """Streaming chat completion. Yields content chunks.

    web_mode: "off" = no search, "on" = always search first,
              "auto" = pass tools, let model decide
    """
    payload: dict = {
        "model": CHAT_MODEL,
        "messages": list(messages),
        "stream": True,
        "options": _opts(temperature),
    }
    if think:
        payload["think"] = True
        if think_level in ("low", "medium", "high"):
            payload["options"]["think_level"] = think_level

    if web_mode == "on":
        # Always search — inject results into context before streaming
        from . import web_search as ws
        user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        if user_msg:
            yield {"content": f"[searching: {user_msg[:60]}]\n", "done": False}
            results = await ws.search(user_msg, top_n=5)
            search_context = "\n".join(
                f"[{i+1}] {r['title']}\n    {r['url']}\n    {r['snippet']}"
                for i, r in enumerate(results) if r.get("title")
            )
            if search_context:
                payload["messages"] = list(messages)
                payload["messages"].insert(-1, {
                    "role": "system",
                    "content": f"[WEB SEARCH RESULTS for: {user_msg[:80]}]\n{search_context}\n[/WEB SEARCH RESULTS]\nUse these results to ground your response with current information. Cite sources where applicable.",
                })

    elif web_mode == "auto":
        # Let model decide — pass tool definitions, handle calls if they come
        payload["tools"] = BROWSER_TOOLS
        payload["stream"] = False
        async for chunk in _chat_with_tools(payload):
            yield chunk
        return

    try:
        async with _client.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if chunk.get("done"):
                    yield chunk
                    return
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield {"content": content, "done": False}
    except httpx.HTTPStatusError as e:
        logger.error("Ollama HTTP error: %s", e)
        yield {"content": f"\n\n[Ollama error: {e.response.status_code} — the model may have run out of VRAM or hit a context limit. Try sending a shorter message or restarting Ollama.]", "done": False}
        yield {"done": True}
    except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError) as e:
        logger.error("Ollama connection error: %s", e)
        yield {"content": f"\n\n[Connection to Ollama lost: {type(e).__name__}. Is Ollama still running?]", "done": False}
        yield {"done": True}


async def _chat_with_tools(payload: dict):
    """Non-streaming chat with tool call loop.

    GPT-OSS may respond with tool_calls instead of content.
    We execute the tool, feed results back, then request final response
    WITHOUT tools so the model generates content instead of more tool calls.
    """
    from . import web_search

    msgs = list(payload["messages"])

    # Round 1: call with tools — let model decide if it needs to search
    resp = await _client.post("/api/chat", json=payload)
    resp.raise_for_status()
    data = resp.json()
    msg = data.get("message", {})

    tool_calls = msg.get("tool_calls", [])
    content = msg.get("content", "")

    if not tool_calls:
        # No search needed — model decided it has enough knowledge
        yield {"content": content, "done": False, "auto_search": False}
        yield {"done": True}
        return

    # Execute all tool calls
    msgs.append(msg)
    all_results = []

    for tc in tool_calls:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        args = fn.get("arguments", {})

        if name == "browser.search":
            query = args.get("query", "")
            yield {"content": f"[searching: {query}]\n", "done": False}
            results = await web_search.search(query, top_n=5)
            tool_result = "\n".join(
                f"[{i+1}] {r['title']}\n    {r['url']}\n    {r['snippet']}"
                for i, r in enumerate(results)
            )
            all_results.append(tool_result)
            msgs.append({
                "role": "tool",
                "content": tool_result or "No results found.",
            })
        else:
            msgs.append({
                "role": "tool",
                "content": f"Tool '{name}' not available.",
            })

    # Round 2: call WITHOUT tools — force content generation from search results
    final_payload = {
        "model": payload["model"],
        "messages": msgs,
        "stream": False,
        "options": payload.get("options", {}),
        # No "tools" key — model must generate content
    }
    resp2 = await _client.post("/api/chat", json=final_payload)
    resp2.raise_for_status()
    data2 = resp2.json()
    final_content = data2.get("message", {}).get("content", "")

    if final_content:
        yield {"content": final_content, "done": False}
    yield {"done": True}


async def structured_extract(
    prompt: str,
    system: Optional[str] = None,
) -> dict:
    """Generate and parse structured JSON from the model.

    Strips markdown fences if present and parses to dict.
    Falls back to format=json if free-form extraction fails.
    """
    raw = await generate(prompt, system=system, temperature=0.3)
    return _parse_json_response(raw)


def _parse_json_response(raw: str) -> dict:
    """Parse JSON from model output, handling markdown fences and comments."""
    text = raw.strip()
    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    # Strip single-line comments (// ...) that GPT-OSS sometimes adds
    text = re.sub(r"//[^\n]*", "", text)
    text = text.strip()
    return json.loads(text)


async def embed(texts: list[str]) -> list[list[float]]:
    """Get embeddings from nomic-embed-text. Returns list of 768-dim vectors."""
    resp = await _client.post("/api/embed", json={
        "model": EMBED_MODEL,
        "input": texts,
    })
    resp.raise_for_status()
    return resp.json()["embeddings"]


async def embed_single(text: str) -> list[float]:
    """Embed a single text. Convenience wrapper."""
    vecs = await embed([text])
    return vecs[0]


async def preload_model() -> None:
    """Preload the chat model into VRAM so the first request isn't cold.

    Sends a minimal generate request with keep_alive=-1 (never unload).
    This loads model weights + allocates KV cache upfront.
    """
    try:
        resp = await _client.post("/api/generate", json={
            "model": CHAT_MODEL,
            "prompt": "",
            "stream": False,
            "keep_alive": -1,
            "options": MODEL_OPTIONS,
        })
        resp.raise_for_status()
        print(f"[OLLAMA] Model preloaded: {CHAT_MODEL} "
              f"(ctx={_NUM_CTX}, gpu_layers={_AUTO_GPU_LAYERS}/24, batch={MODEL_OPTIONS['num_batch']})")
    except Exception as e:
        print(f"[OLLAMA] Preload failed (gpu_layers={_AUTO_GPU_LAYERS}): {e}")


async def health_check() -> dict:
    """Verify Ollama is running and models are available."""
    resp = await _client.get("/api/tags")
    resp.raise_for_status()
    models = {m["name"] for m in resp.json().get("models", [])}
    return {
        "ollama": True,
        "chat_model": CHAT_MODEL in models or any(CHAT_MODEL in m for m in models),
        "embed_model": EMBED_MODEL in models or any(EMBED_MODEL in m for m in models),
    }
