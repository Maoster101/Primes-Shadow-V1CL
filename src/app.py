"""Prime's Shadow — Main application entry point."""
import logging
import sys as _sys

# Force UTF-8 stdout/stderr. Windows consoles and redirected pipes default to
# cp1252, which raises UnicodeEncodeError on any non-latin1 char a model emits
# (arrows, CJK, …). That was silently killing edge persistence when a proposed
# edge's justification contained "→" — a crash in a diagnostic print() aborted
# the whole edge-resolution step. errors="replace" keeps prints from ever
# taking down the request path.
for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from .api.routes import router
from .api import deps
from .api.deps import anchor_matcher, registry

# ── Application logging ───────────────────────────────────────────
# Under uvicorn, only uvicorn's own loggers are configured — app-level
# logger.info()/logger.warning() calls (pipeline timing, matcher/frame
# diagnostics) are dropped at the default WARNING root level. Attach a
# dedicated handler to the "src" namespace so app logs surface, scoped
# so third-party libraries aren't pulled to INFO. propagate=False keeps
# these off the root logger to avoid double-printing under uvicorn.
_app_logger = logging.getLogger("src")
if not _app_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    _app_logger.addHandler(_handler)
    _app_logger.setLevel(logging.INFO)
    _app_logger.propagate = False

app = FastAPI(title="Prime's Shadow", version="0.1.0")
app.include_router(router, prefix="/api")

# Serve static frontend files
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(static_dir / "index.html"))


@app.on_event("startup")
async def startup():
    # Load all corpus collections via registry (§24.1)
    all_errors = registry.load_all()
    for cid, errors in all_errors.items():
        if errors:
            print(f"[CORPUS] Collection '{cid}' — {len(errors)} validation errors:")
            for e in errors[:5]:
                print(f"  - {e}")
        else:
            store = registry.get_store(cid)
            print(f"[CORPUS] Collection '{cid}' — {len(store.anchors)} anchors, "
                  f"{len(store.slabs)} slabs, {len(store.bundles)} bundles, "
                  f"{len(store.gates)} gates")

    # Rebind all services to merged view
    deps.rebind_corpus()
    merged = registry.merged
    print(f"[CORPUS] Merged view: {len(merged.anchors)} anchors, "
          f"{len(merged.slabs)} slabs, {len(merged.bundles)} bundles, "
          f"{len(merged.gates)} gates "
          f"(from {len(registry.active_ids)} collections)")

    # Pre-embed all anchor phrases for fast matching
    await anchor_matcher.warm_cache()
    print(f"[MATCHER] Anchor embedding cache warmed ({len(deps.corpus.anchors)} anchors)")

    # Pre-embed all REFERENCE slab canonical_texts for retrieval-augmented
    # system prompts. The model sees a compact catalog always; the full text
    # of semantically-relevant slabs gets injected on the turns they matter.
    await deps.slab_matcher.warm_cache()
    print(
        f"[MATCHER] Slab embedding cache warmed "
        f"({len(deps.slab_matcher._embed_cache)} REFERENCE slabs, "
        f"{len(deps.slab_matcher._global_pr)} PR nodes)"
    )

    # Reconcile dangling state across the five draft/tentative locations.
    # Commits ACCEPTED edges whose endpoints have since landed, prunes
    # orphan registry entries from the library→delete flow, rejects
    # fully-orphan proposed edges. Safe no-op on a clean system.
    sweep = deps.lifecycle.run_startup_sweep()
    print(f"[LIFECYCLE] Startup sweep: {sweep}")

    # Preload chat model into VRAM (eliminates cold-start on first message).
    # Fire-and-forget: a model swap (e.g. switching families between boots)
    # can take 30-60s, and blocking startup on it made the server "unreachable"
    # until the swap completed. The first real request warms the model anyway,
    # so preload is a latency optimization, not a correctness dependency.
    import asyncio
    from .services import ollama
    asyncio.create_task(ollama.preload_model())
