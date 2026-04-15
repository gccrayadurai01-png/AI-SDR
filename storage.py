"""
Simple JSON-based persistence for calls, script config, and settings.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

DATA_DIR   = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

CALLS_FILE  = DATA_DIR / "calls.json"
SCRIPT_FILE = DATA_DIR / "script.json"

DEFAULT_SCRIPT: dict[str, Any] = {
    "sdr_name":                "Alex",
    "company_name":            "Your Company",
    "call_objective":          "Book a 15-minute discovery call",
    "target_persona":          "Decision makers at mid-market and enterprise companies",
    "value_proposition":       "We help companies solve their biggest challenges with our solution.",
    "pain_points":             "High costs\nManual processes\nLack of visibility\nScalability challenges",
    "company_website":         "",
    "product_services":        "",
    "competitive_advantage":   "",
    "call_flow":               "1. Opening & rapport\n2. Discovery questions\n3. Pain identification\n4. Value connection\n5. Booking/next steps\n6. Professional close",
    "end_goal":                "Schedule a 15-minute discovery call to explore fit",
    "opening_line":            "Hi {name}, this is {sdr_name} from {company}...",
    "discovery_questions":     "What tools are you currently using?\nWhat's the biggest challenge you're facing?\nHow are you measuring success today?\nWho else is involved in decisions like this?",
    "objection_handling":      "Not interested: Totally fair — can I ask what you're using today?\nToo busy: When would be a better time to chat?\nSend email: A quick 15-min call might be more valuable.\nWe have a solution: How's that working for you? Any gaps?",
    "booking_phrase":          "Would Tuesday at 2pm or Thursday at 10am work better?",
    "voicemail_message":       "Hey {name}, this is {sdr_name} from {company}. I was reaching out because we help companies like yours tackle key challenges. I'd love to set up a quick 15-minute call to see if it makes sense for your team. Feel free to call me back or I'll try you again soon. Have a great day!",
    "knowledge_base_notes":    "",
}


# ─── helpers ────────────────────────────────────────────────
def _load(path: Path, default: Any) -> Any:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return default

def _save(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


# ─── calls ──────────────────────────────────────────────────
def load_calls() -> list[dict]:
    return _load(CALLS_FILE, [])

def save_call(call: dict) -> None:
    calls = load_calls()
    for i, c in enumerate(calls):
        if c.get("call_control_id") == call.get("call_control_id"):
            calls[i] = call
            _save(CALLS_FILE, calls)
            return
    call.setdefault("started_at", datetime.utcnow().isoformat())
    calls.insert(0, call)
    _save(CALLS_FILE, calls)

def update_call(call_control_id: str, **kwargs: Any) -> None:
    calls = load_calls()
    for c in calls:
        if c.get("call_control_id") == call_control_id:
            c.update(kwargs)
            _save(CALLS_FILE, calls)
            return


def get_call_by_control_id(call_control_id: str) -> dict[str, Any] | None:
    """Latest row from calls.json for this id (used when active_calls was cleared)."""
    if not call_control_id:
        return None
    for c in load_calls():
        if c.get("call_control_id") == call_control_id:
            return dict(c)
    return None


def finalize_call_end(call_control_id: str, **kwargs: Any) -> bool:
    """
    Update calls.json row by call_control_id. Returns True if a row was updated.
    Used for call.hangup so we persist even when in-memory active_calls was cleared.
    """
    if not call_control_id:
        return False
    calls = load_calls()
    for c in calls:
        if c.get("call_control_id") == call_control_id:
            c.update(kwargs)
            _save(CALLS_FILE, calls)
            return True
    return False


def mark_stale_initiated_calls(max_age_hours: float = 1.0) -> list[str]:
    """
    Mark old `initiated` rows as ended when Telnyx never sent answered/hangup webhooks
    (wrong APP_BASE_URL, ngrok down, etc.). Returns call_control_ids updated.
    """
    calls = load_calls()
    updated: list[str] = []
    now = datetime.utcnow()
    for c in calls:
        if c.get("state") != "initiated":
            continue
        started = c.get("started_at")
        if not started:
            continue
        try:
            raw = str(started).replace("Z", "+00:00")
            t = datetime.fromisoformat(raw)
            if t.tzinfo is not None:
                t = t.replace(tzinfo=None)
            age_sec = (now - t).total_seconds()
        except Exception:
            continue
        if age_sec <= max_age_hours * 3600:
            continue
        cid = c.get("call_control_id")
        if not cid:
            continue
        c["state"] = "ended"
        c["ended_at"] = now.isoformat()
        c["duration_seconds"] = 0
        c["ended_reason"] = "stale_no_webhook"
        updated.append(cid)
    if updated:
        _save(CALLS_FILE, calls)
    return updated


# ─── script ─────────────────────────────────────────────────
def load_script() -> dict[str, Any]:
    return _load(SCRIPT_FILE, DEFAULT_SCRIPT.copy())

def save_script(script: dict[str, Any]) -> None:
    _save(SCRIPT_FILE, script)


# ─── tasks ─────────────────────────────────────────────────
TASKS_FILE = DATA_DIR / "tasks.json"

def load_tasks() -> list[dict]:
    return _load(TASKS_FILE, [])

def save_task(task: dict) -> None:
    tasks = load_tasks()
    for i, t in enumerate(tasks):
        if t.get("id") == task.get("id"):
            tasks[i] = task
            _save(TASKS_FILE, tasks)
            return
    tasks.insert(0, task)
    _save(TASKS_FILE, tasks)

def delete_task(task_id: str) -> None:
    tasks = load_tasks()
    tasks = [t for t in tasks if t.get("id") != task_id]
    _save(TASKS_FILE, tasks)

def update_task(task_id: str, **kwargs: Any) -> None:
    tasks = load_tasks()
    for t in tasks:
        if t.get("id") == task_id:
            t.update(kwargs)
            _save(TASKS_FILE, tasks)
            return
