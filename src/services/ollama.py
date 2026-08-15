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

# ── Host / auth ────────────────────────────────────────────────
# Two deployment modes:
#   LOCAL  (default): talk to ollama daemon on this machine.
#                     PS_OLLAMA_HOST=http://localhost:11434 (default)
#                     PS_OLLAMA_API_KEY unset.
#   CLOUD:            talk to Ollama-hosted inference.
#                     PS_OLLAMA_HOST=https://ollama.com
#                     PS_OLLAMA_API_KEY=<your key from ollama.com>
#                     Model tags must end in `-cloud` (e.g. gpt-oss:120b-cloud).
#
# Cloud silently ignores inference-engine flags (num_ctx, num_gpu, num_batch,
# keep_alive) — those are llama.cpp knobs on a local daemon. We strip them
# from cloud payloads so logs aren't misleading and payloads stay small.
OLLAMA_BASE = os.environ.get("PS_OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_API_KEY = os.environ.get("PS_OLLAMA_API_KEY", "").strip()
IS_CLOUD = "ollama.com" in OLLAMA_BASE or bool(OLLAMA_API_KEY)

# Embeddings can target a different host than chat. Common case: chat in
# cloud, embeddings on local daemon (Ollama Cloud doesn't host
# nomic-embed-text, and per-request embed latency adds up fast in matching).
# Defaults to the primary host so the simple case stays simple.
OLLAMA_EMBED_BASE = os.environ.get("PS_OLLAMA_EMBED_HOST", OLLAMA_BASE).rstrip("/")
# Embed key defaults: same as chat key UNLESS the embed host is local — in
# that case sending a cloud Bearer token to localhost is wasteful (the local
# daemon silently ignores it). Explicitly override with PS_OLLAMA_EMBED_API_KEY.
_explicit_embed_key = os.environ.get("PS_OLLAMA_EMBED_API_KEY")
if _explicit_embed_key is not None:
    OLLAMA_EMBED_API_KEY = _explicit_embed_key.strip()
elif "ollama.com" in OLLAMA_EMBED_BASE:
    OLLAMA_EMBED_API_KEY = OLLAMA_API_KEY
else:
    OLLAMA_EMBED_API_KEY = ""

CHAT_MODEL = os.environ.get("PS_CHAT_MODEL", "gemma3:12b")
EMBED_MODEL = os.environ.get("PS_EMBED_MODEL", "nomic-embed-text")
# Authoring/extraction model — the recall/chat split. Recall runs on the
# fast CHAT_MODEL; structured extraction (end-of-chat ghost sweep, /mine,
# proposal + relationship extraction) is a high-rigor reasoning task that
# a small chat model handles poorly (empty proposals). Point this at a
# capable model (e.g. gpt-oss:20b, qwen3:14b) so authoring gets real
# reasoning without slowing recall. Defaults to CHAT_MODEL = no change.
# Callers passing an explicit model= (e.g. dream enrichment routing to a
# small model) still win over this default.
EXTRACT_MODEL = os.environ.get("PS_EXTRACT_MODEL", CHAT_MODEL)


def extract_is_hosted() -> bool:
    """True when the extraction model runs on a hosted/cloud backend.

    Detected from the model TAG (``…:…-cloud``) or an app-side frontier key /
    cloud host — NOT the app's IS_CLOUD alone, which is False in the common
    setup (local Ollama daemon + a cloud-tagged model routed via Ollama's own
    sign-in). Hosted backends have huge contexts and no local VRAM limit, so
    the end-of-chat mine can feed the WHOLE conversation rather than the tight
    context-budgeted window a local model needs. Read at call time so a
    runtime /models/switch-extract takes effect immediately.
    """
    return "cloud" in (EXTRACT_MODEL or "").lower() or bool(OLLAMA_API_KEY) or IS_CLOUD


def mining_parallelism() -> int:
    """How many extraction calls a doc miner may run concurrently.

    Local extraction is VRAM-bound: two concurrent gpt-oss:20b / gemma
    generations already saturate a 16GB GPU, and over-subscribing makes
    Ollama queue — or crash, as gemma3:12b did. So the local default stays a
    conservative 2 (overridable via PS_MINING_PARALLEL for bigger cards).

    Hosted/cloud extraction has no local VRAM ceiling; the bottleneck is
    per-request latency, which fanning out hides. A section-heavy document that
    drills 30 leaves serially at ~4s each is 2 min; at width 16 it is ~8s. So
    hosted gets a much wider default. PS_MINING_PARALLEL, when set, still wins
    in either regime.

    Read at call time (not import) so a runtime extract-model switch re-tunes
    the next mine without a restart, and so the value reflects the model that
    is actually active when a mine starts.
    """
    env = os.environ.get("PS_MINING_PARALLEL")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return 16 if extract_is_hosted() else 2


def _auth_headers() -> dict:
    """HTTP headers for chat requests. Adds bearer auth when key is set."""
    headers = {}
    if OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"
    return headers


def _embed_auth_headers() -> dict:
    """HTTP headers for embed requests. May differ from chat headers."""
    headers = {}
    if OLLAMA_EMBED_API_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_EMBED_API_KEY}"
    return headers

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

_NUM_CTX = int(os.environ.get("PS_NUM_CTX", "16384"))  # Chat context. Default halved from 32k -> 16k so 26B models leave VRAM headroom for embeddings + KV overhead. Set PS_NUM_CTX=32768 to restore the older larger window on smaller models.
_EXTRACT_NUM_CTX = int(os.environ.get("PS_EXTRACT_NUM_CTX", "4096"))  # Structured extraction (mining, proposals, concept detection). Lowered from 8192 — mining segments cap at ~1800 chars (~450 tokens), so 4096 is generous and halves the per-slot KV cache cost. With OLLAMA_NUM_PARALLEL=2 this leaves enough VRAM headroom for the chat model on a 16GB card.

# num_gpu: layers offloaded to GPU.
#   99  = "offload as many layers as fit in VRAM" (Ollama's convention)
#   -1  = same as 99 in newer Ollama versions
#    0  = CPU only
#
# We default to 99 — let Ollama figure out the optimal split at model
# load time. This avoids the stale-detection bug where our import-time
# check sees different VRAM than what's available when Ollama actually
# loads the model.
#
# Override with PS_NUM_GPU env var if needed (e.g. PS_NUM_GPU=0 for CPU).
_NUM_GPU = int(os.environ.get("PS_NUM_GPU", "99"))

# Detect physical CPU cores for thread count
def _detect_cpu_threads() -> int:
    try:
        import os
        # Use physical cores (not hyperthreads) for Ollama
        cores = os.cpu_count() or 8
        return max(4, cores)
    except Exception:
        return 8

_NUM_BATCH = int(os.environ.get("PS_NUM_BATCH", "512"))  # Prompt-eval batch size. Lowered from 1024 so tight-VRAM 26B configs don't peak-spill during prompt processing. Bump to 1024+ on 24GB+ cards.

MODEL_OPTIONS: dict = {
    "num_ctx": _NUM_CTX,
    "num_gpu": _NUM_GPU,    # 99 = max GPU offload (Ollama picks optimal split)
    "num_batch": _NUM_BATCH,
    "num_thread": _detect_cpu_threads(),
}

logger.info(
    "Ollama config: model=%s num_gpu=%s chat_ctx=%s extract_ctx=%s num_batch=%s threads=%s",
    CHAT_MODEL, _NUM_GPU, _NUM_CTX, _EXTRACT_NUM_CTX, _NUM_BATCH, MODEL_OPTIONS["num_thread"],
)

# Reusable client — partial GPU offload means slow cold starts (~60s model load)
# connect=30s, read=300s (5 min for first-token during cold load + slow CPU layers).
# Cloud calls go over TLS to ollama.com; the same timeout works for both.
_client = httpx.AsyncClient(
    base_url=OLLAMA_BASE,
    timeout=httpx.Timeout(300.0, connect=30.0),
    headers=_auth_headers(),
)

# Separate client for embeddings — usually points at the same host, but lets
# you keep embeddings local while chat goes to cloud (see PS_OLLAMA_EMBED_HOST).
_embed_client = (
    _client
    if OLLAMA_EMBED_BASE == OLLAMA_BASE and OLLAMA_EMBED_API_KEY == OLLAMA_API_KEY
    else httpx.AsyncClient(
        base_url=OLLAMA_EMBED_BASE,
        timeout=httpx.Timeout(60.0, connect=10.0),
        headers=_embed_auth_headers(),
    )
)

logger.info(
    "Ollama transport: chat_host=%s embed_host=%s cloud=%s auth=%s",
    OLLAMA_BASE, OLLAMA_EMBED_BASE, IS_CLOUD, "yes" if OLLAMA_API_KEY else "no",
)


from . import model_profiles


def _opts(temperature: float = 0.7, **extra) -> dict:
    """Build options dict: MODEL_OPTIONS + per-call overrides.

    In cloud mode, llama.cpp-runtime flags (num_ctx, num_gpu, num_batch,
    num_thread) are stripped — the hosted runtime ignores them anyway,
    and stripping keeps payloads small + logs honest.
    """
    if IS_CLOUD:
        o = {"temperature": temperature}
    else:
        o = {**MODEL_OPTIONS, "temperature": temperature}
    # Per-call overrides win — but cloud still won't honor runtime flags.
    # We let callers pass num_ctx etc. through (no-op on cloud) so call
    # sites don't need to branch on IS_CLOUD.
    o.update(extra)
    return o


def _payload_base(**fields) -> dict:
    """Build a request payload with model (and keep_alive for local) set.

    `keep_alive=-1` pins the model in VRAM on a local daemon. Cloud manages
    its own lifecycle, so we omit it there.
    """
    base: dict = {"model": CHAT_MODEL}
    if not IS_CLOUD:
        base["keep_alive"] = -1
    base.update(fields)
    return base


async def generate(
    prompt: str,
    system: Optional[str] = None,
    temperature: float = 0.7,
    raw_json: bool = False,
    num_ctx: Optional[int] = None,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    think: bool = False,
) -> str:
    """Single-shot generation. Use for structured extraction tasks.

    `num_ctx` override: structured extraction rarely needs the full chat
    context budget. Passing a smaller value (e.g. 8192 for mining) avoids
    reserving a huge KV cache for a short prompt.

    `model` override: if provided, routes this call to a different Ollama
    model than the active chat model. Used by dream enrichment to run on
    a small/fast model (e.g. llama3.2:latest) without evicting the chat
    model from VRAM. Falls back to ``CHAT_MODEL`` when unset.

    `timeout` override: per-call read-timeout in seconds. The shared
    client defaults to 300s; a local model can exceed that on a large
    synthesis prompt (e.g. whole-document outlining). Passing a larger
    value lets the call finish rather than raising ReadTimeout.
    """
    opts = _opts(temperature)
    if num_ctx is not None:
        opts["num_ctx"] = num_ctx
    payload_kwargs: dict = {"prompt": prompt, "stream": False, "options": opts}
    if model is not None:
        payload_kwargs["model"] = model
    payload: dict = _payload_base(**payload_kwargs)
    if think:
        # `think` is a TOP-LEVEL Ollama field, NOT an options key — in options
        # it's silently ignored (verified: thinking_len=0). Top-level makes the
        # model reason into a separate `thinking` field, leaving `response`
        # clean JSON for the parser. No-op on models that can't think.
        payload["think"] = True
    if system:
        payload["system"] = system
    if raw_json:
        payload["format"] = "json"

    post_kwargs: dict = {"json": payload}
    if timeout is not None:
        post_kwargs["timeout"] = httpx.Timeout(timeout, connect=30.0)
    resp = await _client.post("/api/generate", **post_kwargs)
    if resp.status_code != 200:
        print(f"[OLLAMA GENERATE ERROR] {resp.status_code}: {resp.text[:300]}")
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
    payload = _payload_base(
        messages=messages,
        stream=False,
        options=_opts(temperature),
    )
    if think:
        payload["think"] = True  # top-level, not options (which is ignored)

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
    payload = _payload_base(
        messages=list(messages),
        stream=True,
        options=_opts(temperature),
    )
    if think and model_profiles.active().supports_think:
        # `think` is a TOP-LEVEL field and takes a bool OR a level string
        # ("low"/"medium"/"high") — and the level genuinely controls reasoning
        # depth (verified: gpt-oss low -> ~12 chars, high -> ~1200). Pass it
        # directly. The old options["think_level"] key is not a real Ollama
        # field and was silently ignored, so the UI selector only ever toggled
        # on/off, never the depth.
        payload["think"] = think_level if think_level in ("low", "medium", "high") else True

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
                    "content": f"[WEB SEARCH RESULTS for: {user_msg[:80]}]\n{search_context}\n[/WEB SEARCH RESULTS]\nUse these results as background knowledge to inform your response. Synthesize the information naturally — do NOT copy snippets verbatim, do NOT list URLs inline, do NOT dump raw search results. Write a clear, conversational response in your own words. If the user wants sources, they can ask.",
                })

    elif web_mode == "auto":
        if model_profiles.active().supports_tools:
            # Let model decide — pass tool definitions, handle calls if they come
            payload["tools"] = BROWSER_TOOLS
            payload["stream"] = False
            async for chunk in _chat_with_tools(payload):
                yield chunk
            return
        else:
            # Model doesn't support tools — fall back to "on" (always search)
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
                        "content": f"[WEB SEARCH RESULTS for: {user_msg[:80]}]\n{search_context}\n[/WEB SEARCH RESULTS]\nUse these results as background knowledge to inform your response. Synthesize the information naturally — do NOT copy snippets verbatim, do NOT list URLs inline, do NOT dump raw search results. Write a clear, conversational response in your own words. If the user wants sources, they can ask.",
                    })

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
        body = e.response.text[:500] if hasattr(e.response, 'text') else str(e)
        logger.error("Ollama HTTP error: %s — body: %s", e.response.status_code, body)
        print(f"[OLLAMA ERROR] {e.response.status_code}: {body}")
        yield {"content": f"\n\n[Ollama error: {e.response.status_code} — {body[:200]}]", "done": False}
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
    final_payload = _payload_base(
        messages=msgs,
        stream=False,
        options=payload.get("options", {}),
        # No "tools" key — model must generate content
    )
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
    num_ctx: Optional[int] = None,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    think: bool = False,
) -> dict:
    """Generate and parse structured JSON from the model.

    Strips markdown fences if present and parses to dict.
    Falls back to format=json if free-form extraction fails.

    `num_ctx` default: if caller doesn't specify, we use the value of
    PS_EXTRACT_NUM_CTX (default 8192). Structured extraction prompts are
    short-to-medium; reserving a full chat-sized KV cache for them wastes
    VRAM that the chat model needs.

    `model` override: optional per-call model swap (see ``generate``).
    Used by dream enrichment to route to a small fast model.

    `timeout` override: per-call read-timeout in seconds (see
    ``generate``). Large synthesis prompts on local models can exceed
    the 300s default.
    """
    if num_ctx is None:
        num_ctx = _EXTRACT_NUM_CTX
    # Default to the authoring model (PS_EXTRACT_MODEL) unless the caller
    # explicitly routed elsewhere. Recall never calls this path.
    if model is None:
        model = EXTRACT_MODEL
    try:
        raw = await generate(
            prompt, system=system, temperature=0.3, num_ctx=num_ctx,
            model=model, timeout=timeout, think=think,
        )
        return _parse_json_response(raw)
    except Exception:
        if not think:
            raise
        # Best-effort thinking: the model may not support it, or emitted the
        # reasoning inline and broke JSON parsing — retry plain (no think).
        raw = await generate(
            prompt, system=system, temperature=0.3, num_ctx=num_ctx,
            model=model, timeout=timeout, think=False,
        )
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


# Ollama's /api/embed rejects large batches: on 0.31.2 a batch of ~250+
# inputs 400s with an internal "tokenize" subprocess connection error
# (batches <=100 succeed). Chunk client-side to stay well under that — this
# keeps every bulk caller (slab warm-up, dedup cache, Pass D sweep) working
# regardless of corpus size and is more robust to future Ollama limits.
_EMBED_MAX_BATCH = 64


async def embed(texts: list[str]) -> list[list[float]]:
    """Get embeddings from nomic-embed-text. Returns list of 768-dim vectors.

    Large inputs are split into sub-batches of _EMBED_MAX_BATCH and the
    results concatenated in order — Ollama's /api/embed fails on big
    batches (see _EMBED_MAX_BATCH note).

    Uses the embed-specific transport (which may differ from chat — see
    PS_OLLAMA_EMBED_HOST). Defaults to the same host as chat.
    """
    if not texts:
        return []
    out: list[list[float]] = []
    for i in range(0, len(texts), _EMBED_MAX_BATCH):
        chunk = texts[i:i + _EMBED_MAX_BATCH]
        resp = await _embed_client.post("/api/embed", json={
            "model": EMBED_MODEL,
            "input": chunk,
        })
        resp.raise_for_status()
        out.extend(resp.json()["embeddings"])
    return out


async def embed_single(text: str) -> list[float]:
    """Embed a single text. Convenience wrapper."""
    vecs = await embed([text])
    return vecs[0]


async def preload_model() -> None:
    """Warm the chat model so the first user request isn't cold.

    On local: sends a minimal generate with keep_alive=-1 (pin in VRAM).
    On cloud: still hits /api/generate to validate auth + model availability,
              but skips local-only diagnostics (/api/ps, GPU split).
    """
    try:
        if IS_CLOUD:
            # Cloud: cheap validation call. No options, no keep_alive.
            payload = {"model": CHAT_MODEL, "prompt": "", "stream": False}
            resp = await _client.post("/api/generate", json=payload)
            resp.raise_for_status()
            print(f"[OLLAMA] Cloud transport ready: host={OLLAMA_BASE} model={CHAT_MODEL}")
        else:
            resp = await _client.post("/api/generate", json=_payload_base(
                prompt="",
                stream=False,
                options=MODEL_OPTIONS,
            ))
            resp.raise_for_status()
            # Check actual GPU split from Ollama
            try:
                ps_resp = await _client.get("/api/ps")
                ps_data = ps_resp.json()
                for m in ps_data.get("models", []):
                    total = m.get("size", 0) / 1e9
                    vram = m.get("size_vram", 0) / 1e9
                    pct = (vram / total * 100) if total > 0 else 0
                    print(f"[OLLAMA] Model loaded: {m['name']} — "
                          f"{vram:.1f}GB VRAM / {total:.1f}GB total ({pct:.0f}% GPU)")
            except Exception:
                pass
            print(f"[OLLAMA] Config: ctx={_NUM_CTX}, num_gpu={_NUM_GPU}, batch={MODEL_OPTIONS['num_batch']}")

        # Auto-detect model capabilities from Ollama (works on both transports)
        profile = await model_profiles.set_active(CHAT_MODEL)
        print(f"[OLLAMA] Profile: family={profile.family}, params={profile.parameter_size}, "
              f"think={profile.supports_think}, tools={profile.supports_tools}, "
              f"vision={profile.supports_vision}, layers={profile.block_count}")
    except Exception as e:
        print(f"[OLLAMA] Preload failed (cloud={IS_CLOUD}, num_gpu={_NUM_GPU}): {e}")


async def health_check() -> dict:
    """Verify Ollama is reachable and the configured models are available.

    When chat and embed are on different hosts (cloud + local hybrid), each
    is checked against its own /api/tags. Empty model lists from cloud are
    expected — cloud /api/tags returns only models pulled to the account,
    so we treat "any model containing the name" as a soft match.
    """
    resp = await _client.get("/api/tags")
    resp.raise_for_status()
    chat_models = {m["name"] for m in resp.json().get("models", [])}

    if _embed_client is _client:
        embed_models = chat_models
    else:
        try:
            r2 = await _embed_client.get("/api/tags")
            r2.raise_for_status()
            embed_models = {m["name"] for m in r2.json().get("models", [])}
        except Exception:
            embed_models = set()

    return {
        "ollama": True,
        "transport": "cloud" if IS_CLOUD else "local",
        "chat_host": OLLAMA_BASE,
        "embed_host": OLLAMA_EMBED_BASE,
        "chat_model": CHAT_MODEL in chat_models or any(CHAT_MODEL in m for m in chat_models),
        "embed_model": EMBED_MODEL in embed_models or any(EMBED_MODEL in m for m in embed_models),
    }
