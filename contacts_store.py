"""Persistent contacts / leads for CRM-style UI."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent / "data"
CONTACTS_FILE = DATA_DIR / "contacts.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_file() -> dict[str, Any]:
    return {
        "contacts": [
            {
                "id": "seed-1",
                "name": "Sarah Chen",
                "email": "sarah@techcorp.com",
                "phone": "+1 415-555-0101",
                "company": "TechCorp",
                "title": "VP Engineering",
                "status": "qualified",
                "score": 85,
                "tags": ["Enterprise", "Tech"],
                "call_history": [
                    {
                        "at": "2026-03-28T17:34:00+00:00",
                        "outcome": "Interested",
                        "summary": "Discussed platform capabilities. Prospect showed interest in AI calling features. Requested demo.",
                    },
                    {
                        "at": "2026-03-20T14:00:00+00:00",
                        "outcome": "Follow-up",
                        "summary": "Left voicemail. Asked to reconnect this week.",
                    },
                ],
                "emails_sent": [
                    {
                        "subject": "Follow-up: Meeting Request",
                        "sent_at": "2026-03-20T10:00:00+00:00",
                        "opened": True,
                    }
                ],
            },
            {
                "id": "seed-2",
                "name": "Marcus Webb",
                "email": "marcus@dataflow.io",
                "phone": "+1 415-555-0142",
                "company": "DataFlow",
                "title": "CTO",
                "status": "contacted",
                "score": 62,
                "tags": ["Mid-Market"],
                "call_history": [
                    {
                        "at": "2026-03-25T16:10:00+00:00",
                        "outcome": "No Answer",
                        "summary": "No pickup. Scheduled retry.",
                    }
                ],
                "emails_sent": [],
            },
            {
                "id": "seed-3",
                "name": "Elena Park",
                "email": "elena@nimbus.ai",
                "phone": "+1 650-555-0199",
                "company": "Nimbus AI",
                "title": "Head of IT",
                "status": "meeting_booked",
                "score": 92,
                "tags": ["AI", "Enterprise"],
                "call_history": [],
                "emails_sent": [
                    {
                        "subject": "CloudFuze — quick intro",
                        "sent_at": "2026-03-22T09:30:00+00:00",
                        "opened": True,
                    }
                ],
            },
            {
                "id": "seed-4",
                "name": "James O'Neil",
                "email": "j.oneil@legacy.com",
                "phone": "+1 212-555-0177",
                "company": "Legacy Systems Inc",
                "title": "Director Ops",
                "status": "new",
                "score": 34,
                "tags": [],
                "call_history": [],
                "emails_sent": [],
            },
            {
                "id": "seed-5",
                "name": "Priya Shah",
                "email": "priya@bright.co",
                "phone": "+1 408-555-0123",
                "company": "Bright Co",
                "title": "CISO",
                "status": "closed",
                "score": 78,
                "tags": ["Security"],
                "call_history": [
                    {
                        "at": "2026-03-15T11:00:00+00:00",
                        "outcome": "Interested",
                        "summary": "Wants pricing follow-up next quarter.",
                    }
                ],
                "emails_sent": [],
            },
            {
                "id": "seed-6",
                "name": "David Kim",
                "email": "dkim@vertex.net",
                "phone": "+1 206-555-0165",
                "company": "Vertex Net",
                "title": "IT Manager",
                "status": "qualified",
                "score": 71,
                "tags": ["Tech"],
                "call_history": [],
                "emails_sent": [],
            },
        ]
    }


def _load() -> dict[str, Any]:
    DATA_DIR.mkdir(exist_ok=True)
    if not CONTACTS_FILE.exists():
        data = _default_file()
        _save(data)
        return data
    try:
        return json.loads(CONTACTS_FILE.read_text(encoding="utf-8"))
    except Exception:
        data = _default_file()
        _save(data)
        return data


def _save(data: dict[str, Any]) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    CONTACTS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def list_contacts() -> list[dict[str, Any]]:
    return list(_load().get("contacts") or [])


def get_contact(contact_id: str) -> dict[str, Any] | None:
    for c in list_contacts():
        if c.get("id") == contact_id:
            return dict(c)
    return None


def create_contact(payload: dict[str, Any]) -> dict[str, Any]:
    data = _load()
    contacts = data.setdefault("contacts", [])
    row = {
        "id": str(uuid.uuid4()),
        "name": (payload.get("name") or "").strip() or "Unknown",
        "email": (payload.get("email") or "").strip(),
        "phone": (payload.get("phone") or "").strip(),
        "company": (payload.get("company") or "").strip(),
        "title": (payload.get("title") or "").strip(),
        "status": (payload.get("status") or "new").strip().lower().replace(" ", "_"),
        "score": int(payload.get("score") or 0),
        "tags": payload.get("tags") if isinstance(payload.get("tags"), list) else [],
        "call_history": [],
        "emails_sent": [],
        "created_at": _now_iso(),
    }
    contacts.append(row)
    _save(data)
    return row


def update_contact(contact_id: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    data = _load()
    contacts = data.setdefault("contacts", [])
    for i, c in enumerate(contacts):
        if c.get("id") != contact_id:
            continue
        merged = {**c, **{k: v for k, v in payload.items() if k != "id"}}
        if "score" in payload:
            merged["score"] = int(payload["score"])
        contacts[i] = merged
        _save(data)
        return dict(contacts[i])
    return None


def import_contacts_replace(payload: dict[str, Any]) -> int:
    """Replace entire contacts list (e.g. JSON import). Returns count."""
    contacts = payload.get("contacts")
    if not isinstance(contacts, list):
        return 0
    cleaned: list[dict[str, Any]] = []
    for it in contacts:
        if not isinstance(it, dict):
            continue
        row = dict(it)
        if not row.get("id"):
            row["id"] = str(uuid.uuid4())
        cleaned.append(row)
    _save({"contacts": cleaned})
    return len(cleaned)


def delete_contact(contact_id: str) -> bool:
    data = _load()
    contacts = data.setdefault("contacts", [])
    new_list = [c for c in contacts if c.get("id") != contact_id]
    if len(new_list) == len(contacts):
        return False
    data["contacts"] = new_list
    _save(data)
    return True


def _digits_only(phone: str) -> str:
    return re.sub(r"\D", "", phone or "")


def find_email_by_phone_e164(phone: str) -> str | None:
    """Match saved contacts by phone (last 10 digits for NANP)."""
    d = _digits_only(phone)
    if len(d) < 10:
        return None
    tail = d[-10:]
    for c in list_contacts():
        cd = _digits_only(str(c.get("phone") or ""))
        if len(cd) >= 10 and cd[-10:] == tail:
            em = (c.get("email") or "").strip()
            if em and "@" in em:
                return em
    return None


# --- status labels for UI ---
STATUS_LABELS = {
    "qualified": "Qualified",
    "contacted": "Contacted",
    "meeting_booked": "Meeting Booked",
    "new": "New",
    "closed": "Closed",
}
