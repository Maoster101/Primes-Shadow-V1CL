"""Universal AI Mine 2 API."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..services import ollama
from ..services.universal_miner import SOURCE_KINDS, UniversalMiner
from .deps import registry

router = APIRouter()


class UniversalMineRequest(BaseModel):
    text: str
    source_label: str = "universal"
    source_kind: str = "auto"
    min_confidence: float = Field(default=0.5, ge=0, le=1)
    segment_chars: int = Field(default=0, ge=0, le=20_000)
    target_collection: str = "default"


@router.get("/mine-v2/config")
async def mine_v2_config():
    return {"extract_model": ollama.EXTRACT_MODEL, "hosted": ollama.extract_is_hosted(),
            "source_kinds": sorted(SOURCE_KINDS)}


@router.post("/mine-v2")
async def mine_v2(req: UniversalMineRequest):
    if not req.text.strip():
        raise HTTPException(400, "Empty text")
    if len(req.text) > 5_000_000:
        raise HTTPException(413, "Text too large (max 5MB)")
    if req.source_kind not in SOURCE_KINDS:
        raise HTTPException(400, f"Unknown source kind: {req.source_kind}")
    store = registry.get_store(req.target_collection)
    if store is None:
        raise HTTPException(404, f"Collection not found: {req.target_collection}")
    result = await UniversalMiner(store).mine(
        req.text, req.source_label, req.source_kind, req.min_confidence, req.segment_chars,
    )
    result["target_collection"] = req.target_collection
    return result
