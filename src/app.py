"""Prime's Shadow — Main application entry point."""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from .api.routes import router, corpus, anchor_matcher, gauntlet_engine, registry, _rebind_corpus

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
    _rebind_corpus()
    merged = registry.merged
    print(f"[CORPUS] Merged view: {len(merged.anchors)} anchors, "
          f"{len(merged.slabs)} slabs, {len(merged.bundles)} bundles, "
          f"{len(merged.gates)} gates "
          f"(from {len(registry.active_ids)} collections)")

    # Pre-embed all anchor phrases for fast matching
    await anchor_matcher.warm_cache()
    print(f"[MATCHER] Anchor embedding cache warmed ({len(corpus.anchors)} anchors)")

    # Preload chat model into VRAM (eliminates cold-start on first message)
    from .services import ollama
    await ollama.preload_model()
