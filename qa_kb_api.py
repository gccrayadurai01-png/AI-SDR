from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import config
import qa_kb


router = APIRouter(prefix="/api/learn", tags=["learn"])


@router.get("/status")
async def status():
    config.reload_secrets()
    s = qa_kb.stats()
    return {
        "enabled": bool(config.QA_KB_ENABLED),
        "min_score": float(config.QA_KB_MIN_SCORE),
        **s,
    }


@router.get("/items")
async def items(limit: int = 200, offset: int = 0):
    return {"items": qa_kb.list_items(limit=limit, offset=offset)}


class SearchBody(BaseModel):
    q: str = ""
    limit: int = 25


@router.post("/search")
async def search(body: SearchBody):
    q = (body.q or "").strip()
    if not q:
        return {"items": []}
    return {"items": qa_kb.search(q, limit=body.limit)}


@router.delete("/clear")
async def clear():
    qa_kb.clear()
    return {"status": "cleared"}

