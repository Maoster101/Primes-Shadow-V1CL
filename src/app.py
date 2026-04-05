"""Prime's Shadow — Main application entry point."""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from .api.routes import router, corpus, anchor_matcher

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
    # Load and validate corpus on startup (§24.1)
    errors = corpus.load()
    if errors:
        print(f"[CORPUS] Validation errors on startup ({len(errors)}):")
        for e in errors[:10]:
            print(f"  - {e}")
    else:
        print(f"[CORPUS] Loaded OK: {len(corpus.anchors)} anchors, "
              f"{len(corpus.slabs)} slabs, {len(corpus.bundles)} bundles")

    # Pre-embed all anchor phrases for fast matching
    await anchor_matcher.warm_cache()
    print(f"[MATCHER] Anchor embedding cache warmed ({len(corpus.anchors)} anchors)")
