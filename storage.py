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
    "sdr_name": "Anthony",
    "company_name": "CloudFuze",
    "call_objective": "Book a 15-minute discovery call to demo CloudFuze Manage",
    "target_persona": "VP/Director of IT, SaaS Ops, Security or Cloud at mid-market/enterprise",
    "value_proposition": "CloudFuze Manage gives teams full visibility into their SaaS and AI app stack — saving ~30% on license spend, catching shadow IT, and closing security gaps before AI rollouts like Copilot or Gemini.",
    "opening": "Hey {name}, Anthony here from CloudFuze – quick question, got a minute?",
    "upfront_contract": "I'll keep it to 15 minutes. Sound okay?",
    "discovery_questions": [
        "Can you tell me a bit about your upcoming migration? What's the hardest part?",
        "How are you handling permissions and shared files in that move?",
        "What happens if a shared link accidentally goes to the new system?",
        "If you had a magic wand, what's one thing you'd fix about cloud security?",
        "On a scale of 1–10, how urgent is this for you?",
        "What usually makes you say yes to a new solution?"
    ],
    "negative_reversals": [
        "I know you're probably thinking this isn't a big deal…",
        "Maybe this isn't an issue for you right now, is it?",
        "I'm not here to sell anything, just to explore if there's a fit."
    ],
    "insight_value_points": [
        "We often see clients spend 20–30% less on apps after we show them the waste.",
        "Most prospects are surprised by how many public links we find in their cloud.",
        "Preventing a single data leak could pay for itself – is that risk on your radar?",
        "Average CloudFuze customer cleans up ~200,000 hidden links per year.",
        "With Copilot/Gemini on the horizon, companies MUST clean permissions first — those tools surface everything a user can access."
    ],
    "soft_close_cta": [
        "Does spending 15 minutes make sense to see if CloudFuze can help?",
        "If I could help solve that, would it be worth a quick call?"
    ],
    "multi_thread": [
        "Is anyone else on your team involved in this decision? Who should join the call?",
        "Who else should I send info to before we talk?"
    ],
    "objection_handling": {
        "busy": "Totally understand, when's a better time to chat?",
        "not_interested": "I understand. Just so I'm clear, is it the timing or the solution?",
        "has_tool": "Oh nice, how's that working for you? What does it not do?",
        "no_budget": "Makes sense — usually this saves money within the first few months. Might not be the right time though.",
        "send_email": "Happy to — but a quick 15-min call is usually more helpful. Worth a shot?"
    },
    "voicemail_message": "Hi {name}, this is {sdr_name} from {company}. We help IT teams avoid surprises during cloud migrations and AI rollouts. I'd love to ask you one quick question — do you have a few seconds? If not, you can reach me at this number. Thank you!",
    "knowledge_manage": "CloudFuze Manage: SaaS/AI app management platform. Full visibility and control over all cloud apps and permissions. License optimization (~30% savings), shadow IT detection, access governance, permission risk scoring, automated remediation (one-click revoke public links, reassign files), AI-readiness tools (scan for hot content before enabling AI bots).",
    "knowledge_pain_context": "Data Sprawl & Oversharing: many orgs have rampant permission sprawl, AI exacerbates it. Migration Complexity: moving 100-300TB, downtime risk, permission creep, lost files. AI Exposure: Copilot/GPT indexes what people can see — permission hygiene is crucial.",
    "knowledge_metrics": "Avg customer cleans ~200,000 hidden links/year. Cut 20-30% unused licenses within first 3 months. Supports 40+ cloud platforms and 100+ SaaS apps. Web3 company cut shadow IT by 40% with CloudFuze Manage.",
    "knowledge_competitive": "CloudFuze Manage adds migration-related permission cleanup layer beyond typical license mgmt tools. If prospect has a tool, ask what it doesn't do — Sandler disqualification technique.",
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
