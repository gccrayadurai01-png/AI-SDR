"""
Call-learned Q/A knowledge base.

- Persists (question, answer) pairs to data/qa_kb.json
- Retrieves best matching prior Q/A for repeated or related questions
- Designed to be lightweight: no extra dependencies, no embeddings
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from storage import DATA_DIR

logger = logging.getLogger(__name__)

KB_FILE = DATA_DIR / "qa_kb.json"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load() -> dict[str, Any]:
    if not KB_FILE.exists():
        return {"items": []}
    try:
        raw = json.loads(KB_FILE.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {"items": []}
        raw.setdefault("items", [])
        if not isinstance(raw["items"], list):
            raw["items"] = []
        return raw
    except Exception:
        logger.exception("qa_kb: failed to read %s", KB_FILE)
        return {"items": []}


def _save(store: dict[str, Any]) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    KB_FILE.write_text(json.dumps(store, indent=2, ensure_ascii=False), encoding="utf-8")


_STOP = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "is",
    "are",
    "am",
    "was",
    "were",
    "be",
    "been",
    "it",
    "this",
    "that",
    "these",
    "those",
    "we",
    "you",
    "i",
    "they",
    "he",
    "she",
    "them",
    "me",
    "my",
    "your",
    "our",
    "as",
    "at",
    "from",
    "by",
    "about",
    "so",
    "just",
    "like",
    "ok",
    "okay",
    "yeah",
    "yep",
    "no",
    "yes",
    "hi",
    "hello",
    "hey",
}


def _norm(s: str) -> str:
    t = (s or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t


def _tokens(s: str) -> set[str]:
    t = _norm(s)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    parts = [p for p in t.split() if p and p not in _STOP and len(p) > 1]
    return set(parts)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    uni = len(a | b)
    return inter / uni if uni else 0.0


def similarity(q1: str, q2: str) -> float:
    """
    Return 0..1 similarity using a hybrid of:
    - token Jaccard (semantic-ish)
    - character SequenceMatcher (typos / phrasing)
    """
    n1 = _norm(q1)
    n2 = _norm(q2)
    if not n1 or not n2:
        return 0.0
    if n1 == n2:
        return 1.0
    tj = _jaccard(_tokens(n1), _tokens(n2))
    cr = SequenceMatcher(None, n1, n2).ratio()
    return 0.62 * tj + 0.38 * cr


def _two_sentences_max(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return t
    # Soft cap: keep at most 2 sentences for phone.
    parts = re.split(r"(?<=[.!?])\s+", t)
    out = " ".join(p.strip() for p in parts[:2] if p and p.strip()).strip()
    return out if out else t


@dataclass
class Match:
    score: float
    item: dict[str, Any]


def find_best(question: str, min_score: float = 0.82) -> Match | None:
    store = _load()
    best: Match | None = None
    for it in store.get("items") or []:
        if not isinstance(it, dict):
            continue
        q = it.get("question") or ""
        sc = similarity(question, q)
        if sc >= min_score and (best is None or sc > best.score):
            best = Match(score=sc, item=it)
    return best


def answer_for(question: str, min_score: float = 0.82) -> tuple[str | None, float]:
    m = find_best(question, min_score=min_score)
    if not m:
        return None, 0.0
    ans = (m.item.get("answer") or "").strip()
    if not ans:
        return None, 0.0
    return _two_sentences_max(ans), float(m.score)


def add_qa(
    question: str,
    answer: str,
    *,
    call_control_id: str | None = None,
    source: str = "call_turn",
) -> None:
    q = (question or "").strip()
    a = (answer or "").strip()
    if len(q) < 4 or len(a) < 2:
        return
    store = _load()
    now = _utcnow_iso()

    # If very similar question exists, update it (keeps KB compact).
    m = find_best(q, min_score=0.93)
    if m:
        it = m.item
        it["answer"] = a  # latest answer wins
        it["last_seen_at"] = now
        it["count"] = int(it.get("count") or 0) + 1
        it["source"] = it.get("source") or source
        if call_control_id:
            it["last_call_control_id"] = call_control_id
        _save(store)
        return

    item = {
        "id": uuid.uuid4().hex[:16],
        "question": q,
        "answer": a,
        "created_at": now,
        "last_seen_at": now,
        "count": 1,
        "source": source,
        "last_call_control_id": call_control_id or "",
    }
    store.setdefault("items", []).insert(0, item)
    # Keep file bounded (simple LRU-ish by recency of insertion).
    max_items = 4000
    if len(store["items"]) > max_items:
        store["items"] = store["items"][:max_items]
    _save(store)


def stats() -> dict[str, Any]:
    store = _load()
    items = store.get("items") or []
    return {
        "total_items": len(items),
        "file": str(KB_FILE),
    }


def list_items(limit: int = 200, offset: int = 0) -> list[dict[str, Any]]:
    store = _load()
    items = [it for it in (store.get("items") or []) if isinstance(it, dict)]
    o = max(0, int(offset))
    l = max(1, min(2000, int(limit)))
    return items[o : o + l]


def search(query: str, limit: int = 25) -> list[dict[str, Any]]:
    q = (query or "").strip()
    if not q:
        return []
    store = _load()
    scored: list[tuple[float, dict[str, Any]]] = []
    for it in store.get("items") or []:
        if not isinstance(it, dict):
            continue
        sc = similarity(q, str(it.get("question") or ""))
        if sc >= 0.45:
            scored.append((sc, it))
    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[dict[str, Any]] = []
    for sc, it in scored[: max(1, min(200, int(limit)))]:
        row = dict(it)
        row["_score"] = round(float(sc), 3)
        out.append(row)
    return out


def clear() -> None:
    _save({"items": []})

