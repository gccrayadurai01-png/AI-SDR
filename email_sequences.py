"""
Email automation sequences — isolated from outbound Telnyx calls.

Stores templates + enrollments in data/email_sequences.json.
Background tick sends due steps via EMAIL_PROVIDER: SMTP, SendGrid, Resend, or Mailgun (configure in .env).

API prefix: /api/email-sequences
"""

from __future__ import annotations

import asyncio
import logging
import re
import smtplib
import uuid
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import config
from storage import DATA_DIR

logger = logging.getLogger(__name__)

DATA_FILE = DATA_DIR / "email_sequences.json"

router = APIRouter(prefix="/api/email-sequences", tags=["email-sequences"])

_scheduler_task: asyncio.Task[None] | None = None
_scheduler_stop = asyncio.Event()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _default_store() -> dict[str, Any]:
    return {
        "templates": [
            {
                "id": "default_3touch",
                "name": "3-touch intro (example)",
                "steps": [
                    {
                        "subject": "Hi {{name}} — quick question",
                        "body_text": "Hi {{name}},\n\nI noticed {{company}} and wanted to reach out briefly.\n\n— {{sdr_name}}",
                        "delay_hours_after_previous": 0,
                    },
                    {
                        "subject": "Following up",
                        "body_text": "Hi {{name}},\n\nCircling back on my last note. Open to a 10-min chat this week?\n\n— {{sdr_name}}",
                        "delay_hours_after_previous": 48,
                    },
                    {
                        "subject": "Last note from me",
                        "body_text": "Hi {{name}},\n\nI'll assume timing isn't right — if pipeline is ever a priority, reply anytime.\n\n— {{sdr_name}}",
                        "delay_hours_after_previous": 72,
                    },
                ],
            }
        ],
        "enrollments": [],
    }


def load_store() -> dict[str, Any]:
    if not DATA_FILE.exists():
        return _default_store()
    try:
        import json

        raw = json.loads(DATA_FILE.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return _default_store()
        raw.setdefault("templates", _default_store()["templates"])
        raw.setdefault("enrollments", [])
        return raw
    except Exception:
        logger.exception("email_sequences: corrupt %s — using defaults", DATA_FILE)
        return _default_store()


def save_store(data: dict[str, Any]) -> None:
    import json

    DATA_DIR.mkdir(exist_ok=True)
    DATA_FILE.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _template_by_id(store: dict[str, Any], tid: str) -> dict[str, Any] | None:
    for t in store.get("templates") or []:
        if isinstance(t, dict) and t.get("id") == tid:
            return t
    return None


def _render(text: str, vars_: dict[str, str]) -> str:
    out = text or ""
    for k, v in vars_.items():
        out = out.replace("{{" + k + "}}", v)
    return out


def _effective_provider() -> str:
    config.reload_secrets()
    p = (config.EMAIL_PROVIDER or "smtp").strip().lower()
    if p not in ("smtp", "sendgrid", "resend", "mailgun", "gmail_oauth", "outlook_oauth"):
        return "smtp"
    return p


def smtp_ready() -> bool:
    """True when SMTP host + From are set (for SMTP test / legacy checks)."""
    config.reload_secrets()
    return bool((config.SMTP_HOST or "").strip() and (config.EMAIL_FROM or "").strip())


def email_delivery_ready() -> bool:
    """True when the active provider can send (per config)."""
    config.reload_secrets()
    p = _effective_provider()
    if p == "smtp":
        return smtp_ready()
    if p == "sendgrid":
        return bool((config.SENDGRID_API_KEY or "").strip() and (config.EMAIL_FROM or "").strip())
    if p == "resend":
        return bool((config.RESEND_API_KEY or "").strip() and (config.EMAIL_FROM or "").strip())
    if p == "mailgun":
        return bool(
            (config.MAILGUN_API_KEY or "").strip()
            and (config.MAILGUN_DOMAIN or "").strip()
            and (config.EMAIL_FROM or "").strip()
        )
    if p == "gmail_oauth":
        try:
            from email_oauth import oauth_account_ready

            return oauth_account_ready("google") and bool((config.EMAIL_FROM or "").strip())
        except Exception:
            return False
    if p == "outlook_oauth":
        try:
            from email_oauth import oauth_account_ready

            return oauth_account_ready("microsoft") and bool((config.EMAIL_FROM or "").strip())
        except Exception:
            return False
    return False


def _parse_from_for_apis(raw: str) -> tuple[str, str | None]:
    """Return (email, display_name_or_none) from EMAIL_FROM."""
    s = (raw or "").strip()
    if not s:
        raise RuntimeError("EMAIL_FROM is not set")
    m = re.match(r"^(.+?)\s*<([^>]+)>$", s)
    if m:
        name = m.group(1).strip().strip("\"'")
        email = m.group(2).strip()
        return email, (name or None)
    if "@" in s:
        return s, None
    raise RuntimeError("EMAIL_FROM must include a valid email address")


def _send_smtp_sync(to_email: str, subject: str, body: str) -> None:
    host = (config.SMTP_HOST or "").strip()
    if not host:
        raise RuntimeError("SMTP_HOST is not set")
    from_addr = (config.EMAIL_FROM or "").strip()
    if not from_addr:
        raise RuntimeError("EMAIL_FROM is not set")
    user = (config.SMTP_USER or "").strip()
    password = (config.SMTP_PASSWORD or "") or ""
    port = int(config.SMTP_PORT or 587)
    use_tls = bool(config.SMTP_USE_TLS)

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_email

    payload = msg.as_string()

    if port == 465:
        with smtplib.SMTP_SSL(host, port, timeout=45) as s:
            if user:
                s.login(user, password)
            s.sendmail(from_addr, [to_email], payload)
    else:
        with smtplib.SMTP(host, port, timeout=45) as s:
            s.ehlo()
            if use_tls:
                s.starttls()
                s.ehlo()
            if user or password:
                s.login(user, password)
            s.sendmail(from_addr, [to_email], payload)


def _send_sendgrid_sync(to_email: str, subject: str, body: str) -> None:
    key = (config.SENDGRID_API_KEY or "").strip()
    if not key:
        raise RuntimeError("SENDGRID_API_KEY is not set")
    from_email, from_name = _parse_from_for_apis(config.EMAIL_FROM or "")
    from_obj: dict[str, str] = {"email": from_email}
    if from_name:
        from_obj["name"] = from_name
    payload = {
        "personalizations": [{"to": [{"email": to_email}]}],
        "from": from_obj,
        "subject": subject,
        "content": [{"type": "text/plain", "value": body}],
    }
    with httpx.Client(timeout=45.0) as client:
        r = client.post(
            "https://api.sendgrid.com/v3/mail/send",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
        )
    if r.status_code >= 400:
        raise RuntimeError(f"SendGrid HTTP {r.status_code}: {r.text[:500]}")


def _send_resend_sync(to_email: str, subject: str, body: str) -> None:
    key = (config.RESEND_API_KEY or "").strip()
    if not key:
        raise RuntimeError("RESEND_API_KEY is not set")
    from_raw = (config.EMAIL_FROM or "").strip()
    if not from_raw:
        raise RuntimeError("EMAIL_FROM is not set")
    payload = {
        "from": from_raw,
        "to": [to_email],
        "subject": subject,
        "text": body,
    }
    with httpx.Client(timeout=45.0) as client:
        r = client.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
        )
    if r.status_code >= 400:
        raise RuntimeError(f"Resend HTTP {r.status_code}: {r.text[:500]}")


def _send_mailgun_sync(to_email: str, subject: str, body: str) -> None:
    domain = (config.MAILGUN_DOMAIN or "").strip()
    key = (config.MAILGUN_API_KEY or "").strip()
    if not domain or not key:
        raise RuntimeError("MAILGUN_DOMAIN and MAILGUN_API_KEY are required")
    base = (config.MAILGUN_API_BASE or "https://api.mailgun.net").strip().rstrip("/")
    from_raw = (config.EMAIL_FROM or "").strip()
    if not from_raw:
        raise RuntimeError("EMAIL_FROM is not set")
    url = f"{base}/v3/{domain}/messages"
    data = {"from": from_raw, "to": to_email, "subject": subject, "text": body}
    with httpx.Client(timeout=45.0) as client:
        r = client.post(url, auth=("api", key), data=data)
    if r.status_code >= 400:
        raise RuntimeError(f"Mailgun HTTP {r.status_code}: {r.text[:500]}")


def _send_email_sync(to_email: str, subject: str, body: str) -> None:
    config.reload_secrets()
    p = _effective_provider()
    if p == "sendgrid":
        _send_sendgrid_sync(to_email, subject, body)
    elif p == "resend":
        _send_resend_sync(to_email, subject, body)
    elif p == "mailgun":
        _send_mailgun_sync(to_email, subject, body)
    elif p == "gmail_oauth":
        from email_oauth import send_via_gmail_api

        send_via_gmail_api(to_email, subject, body)
    elif p == "outlook_oauth":
        from email_oauth import send_via_microsoft_graph

        send_via_microsoft_graph(to_email, subject, body)
    else:
        _send_smtp_sync(to_email, subject, body)


async def send_email_async(to_email: str, subject: str, body: str) -> None:
    await asyncio.to_thread(_send_email_sync, to_email, subject, body)


def test_smtp_connection() -> dict[str, Any]:
    """Connect and optionally login — does not send mail."""
    config.reload_secrets()
    if not smtp_ready():
        return {"ok": False, "error": "Set SMTP_HOST and EMAIL_FROM first"}
    host = (config.SMTP_HOST or "").strip()
    port = int(config.SMTP_PORT or 587)
    use_tls = bool(config.SMTP_USE_TLS)
    user = (config.SMTP_USER or "").strip()
    password = (config.SMTP_PASSWORD or "") or ""
    try:
        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=25) as s:
                if user or password:
                    s.login(user, password)
        else:
            with smtplib.SMTP(host, port, timeout=25) as s:
                s.ehlo()
                if use_tls:
                    s.starttls()
                    s.ehlo()
                if user or password:
                    s.login(user, password)
        return {"ok": True, "message": "SMTP connection OK"}
    except Exception as e:
        logger.warning("SMTP test failed: %s", e)
        return {"ok": False, "error": str(e)}


def test_email_delivery() -> dict[str, Any]:
    """Verify the configured EMAIL_PROVIDER (no message sent)."""
    config.reload_secrets()
    p = _effective_provider()
    if p == "smtp":
        return test_smtp_connection()
    if p == "sendgrid":
        key = (config.SENDGRID_API_KEY or "").strip()
        if not key:
            return {"ok": False, "error": "SENDGRID_API_KEY is not set"}
        if not (config.EMAIL_FROM or "").strip():
            return {"ok": False, "error": "EMAIL_FROM is not set"}
        try:
            with httpx.Client(timeout=20.0) as client:
                r = client.get(
                    "https://api.sendgrid.com/v3/user/profile",
                    headers={"Authorization": f"Bearer {key}"},
                )
            if r.status_code >= 400:
                return {"ok": False, "error": f"SendGrid HTTP {r.status_code}: {r.text[:300]}"}
            data = r.json()
            return {"ok": True, "message": "SendGrid API key OK", "provider": "sendgrid", "account": data.get("email")}
        except Exception as e:
            logger.warning("SendGrid test failed: %s", e)
            return {"ok": False, "error": str(e)}
    if p == "resend":
        key = (config.RESEND_API_KEY or "").strip()
        if not key:
            return {"ok": False, "error": "RESEND_API_KEY is not set"}
        if not (config.EMAIL_FROM or "").strip():
            return {"ok": False, "error": "EMAIL_FROM is not set"}
        try:
            with httpx.Client(timeout=20.0) as client:
                r = client.get(
                    "https://api.resend.com/domains",
                    headers={"Authorization": f"Bearer {key}"},
                )
            if r.status_code >= 400:
                return {"ok": False, "error": f"Resend HTTP {r.status_code}: {r.text[:300]}"}
            return {"ok": True, "message": "Resend API key OK", "provider": "resend"}
        except Exception as e:
            logger.warning("Resend test failed: %s", e)
            return {"ok": False, "error": str(e)}
    if p == "mailgun":
        domain = (config.MAILGUN_DOMAIN or "").strip()
        key = (config.MAILGUN_API_KEY or "").strip()
        base = (config.MAILGUN_API_BASE or "https://api.mailgun.net").strip().rstrip("/")
        if not domain or not key:
            return {"ok": False, "error": "MAILGUN_DOMAIN and MAILGUN_API_KEY are required"}
        if not (config.EMAIL_FROM or "").strip():
            return {"ok": False, "error": "EMAIL_FROM is not set"}
        url = f"{base}/v3/domains/{domain}"
        try:
            with httpx.Client(timeout=20.0) as client:
                r = client.get(url, auth=("api", key))
            if r.status_code >= 400:
                return {"ok": False, "error": f"Mailgun HTTP {r.status_code}: {r.text[:300]}"}
            return {"ok": True, "message": "Mailgun domain OK", "provider": "mailgun"}
        except Exception as e:
            logger.warning("Mailgun test failed: %s", e)
            return {"ok": False, "error": str(e)}
    if p == "gmail_oauth":
        from email_oauth import test_gmail_connection

        return test_gmail_connection()
    if p == "outlook_oauth":
        from email_oauth import test_microsoft_connection

        return test_microsoft_connection()
    return {"ok": False, "error": "Unknown EMAIL_PROVIDER"}


def _parse_iso(dt: str | None) -> datetime | None:
    if not dt:
        return None
    try:
        s = str(dt).replace("Z", "+00:00")
        t = datetime.fromisoformat(s)
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return t
    except Exception:
        return None


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


async def process_due_enrollments() -> None:
    if not config.EMAIL_AUTOMATION_ENABLED:
        return
    if not email_delivery_ready():
        return

    store = load_store()
    now = _utcnow()
    changed = False

    for e in list(store.get("enrollments") or []):
        if not isinstance(e, dict):
            continue
        if e.get("status") != "active":
            continue
        nxt = _parse_iso(e.get("next_send_at"))
        if nxt is None or nxt > now:
            continue

        tid = e.get("template_id") or ""
        tmpl = _template_by_id(store, tid)
        if not tmpl:
            e["status"] = "error"
            e["last_error"] = f"template not found: {tid}"
            changed = True
            continue

        steps = tmpl.get("steps") or []
        idx = int(e.get("step_index") or 0)
        if idx >= len(steps):
            e["status"] = "completed"
            e["completed_at"] = _iso(now)
            changed = True
            continue

        step = steps[idx]
        if not isinstance(step, dict):
            e["status"] = "error"
            e["last_error"] = "invalid step"
            changed = True
            continue

        vars_: dict[str, str] = {
            "name": str(e.get("prospect_name") or "there"),
            "email": str(e.get("email") or ""),
            "company": str(e.get("company") or ""),
            "first_name": str(e.get("first_name") or e.get("prospect_name") or "there"),
            "sdr_name": (config.SDR_NAME or "Alex"),
        }
        extra = e.get("vars") or e.get("custom_vars")
        if isinstance(extra, dict):
            for k, v in extra.items():
                if isinstance(k, str) and v is not None:
                    vars_[k] = str(v)

        subj = _render(str(step.get("subject") or "Hello"), vars_)
        body = _render(str(step.get("body_text") or ""), vars_)
        to = str(e.get("email") or "").strip()
        if not to or not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", to):
            e["status"] = "error"
            e["last_error"] = "invalid enrollment email"
            changed = True
            continue

        try:
            await send_email_async(to, subj, body)
            e["last_sent_at"] = _iso(now)
            e["last_sent_subject"] = subj
            e["last_error"] = ""
            logger.info("Email sequence sent step %s to %s (%s)", idx, to, subj[:60])
        except Exception as ex:
            e["last_error"] = str(ex)
            logger.error("Email sequence send failed: %s", ex)
            changed = True
            continue

        next_idx = idx + 1
        e["step_index"] = next_idx
        if next_idx >= len(steps):
            e["status"] = "completed"
            e["completed_at"] = _iso(now)
            e.pop("next_send_at", None)
        else:
            delay_h = float((steps[next_idx] or {}).get("delay_hours_after_previous") or 0)
            e["next_send_at"] = _iso(now + timedelta(hours=max(0.0, delay_h)))
        changed = True

    if changed:
        save_store(store)


async def _scheduler_loop() -> None:
    tick = max(15, int(config.EMAIL_SEQUENCE_TICK_SEC or 60))
    while not _scheduler_stop.is_set():
        try:
            config.reload_secrets()
            await process_due_enrollments()
        except Exception:
            logger.exception("email sequence scheduler tick failed")
        try:
            await asyncio.wait_for(_scheduler_stop.wait(), timeout=float(tick))
            break
        except asyncio.TimeoutError:
            continue


def start_email_scheduler() -> None:
    global _scheduler_task
    if _scheduler_task is not None and not _scheduler_task.done():
        return
    _scheduler_stop.clear()
    _scheduler_task = asyncio.create_task(_scheduler_loop())
    logger.info("Email sequence scheduler started")


def stop_email_scheduler() -> None:
    global _scheduler_task
    _scheduler_stop.set()
    t = _scheduler_task
    _scheduler_task = None
    if t and not t.done():
        t.cancel()


# ─── API models ────────────────────────────────────────────


class SequenceStep(BaseModel):
    subject: str = ""
    body_text: str = ""
    delay_hours_after_previous: float = 0


class TemplateCreate(BaseModel):
    id: str | None = None
    name: str = "Untitled sequence"
    steps: list[SequenceStep] = Field(default_factory=list)


class EnrollBody(BaseModel):
    template_id: str
    email: str
    prospect_name: str = "there"
    company: str = ""
    first_name: str = ""
    custom_vars: dict[str, str] | None = None


@router.get("/status")
async def seq_status():
    config.reload_secrets()
    return {
        "enabled": bool(config.EMAIL_AUTOMATION_ENABLED),
        "smtp_configured": email_delivery_ready(),
        "email_provider": _effective_provider(),
        "tick_seconds": int(config.EMAIL_SEQUENCE_TICK_SEC or 60),
        "scheduler_running": _scheduler_task is not None and not _scheduler_task.done(),
    }


@router.get("/templates")
async def list_templates():
    return {"templates": load_store().get("templates", [])}


@router.post("/templates")
async def create_template(body: TemplateCreate):
    store = load_store()
    tid = (body.id or "").strip() or uuid.uuid4().hex[:12]
    for t in store.get("templates") or []:
        if isinstance(t, dict) and t.get("id") == tid:
            raise HTTPException(status_code=409, detail="Template id already exists")
    steps = [s.model_dump() for s in body.steps]
    store.setdefault("templates", []).append({"id": tid, "name": body.name, "steps": steps})
    save_store(store)
    return {"id": tid, "status": "saved"}


@router.get("/enrollments")
async def list_enrollments():
    return {"enrollments": load_store().get("enrollments", [])}


@router.post("/enroll")
async def enroll(body: EnrollBody):
    store = load_store()
    if not _template_by_id(store, body.template_id):
        raise HTTPException(status_code=404, detail="Unknown template_id")
    eid = uuid.uuid4().hex
    now = _utcnow()
    tmpl = _template_by_id(store, body.template_id) or {}
    steps = tmpl.get("steps") or []
    if not steps:
        raise HTTPException(status_code=400, detail="Template has no steps")

    first_delay = float((steps[0] or {}).get("delay_hours_after_previous") or 0)
    next_at = now + timedelta(hours=max(0.0, first_delay))

    row = {
        "id": eid,
        "template_id": body.template_id,
        "email": body.email.strip(),
        "prospect_name": body.prospect_name.strip() or "there",
        "company": body.company.strip(),
        "first_name": (body.first_name or "").strip(),
        "vars": body.custom_vars or {},
        "step_index": 0,
        "next_send_at": _iso(next_at),
        "status": "active",
        "created_at": _iso(now),
        "last_error": "",
    }
    store.setdefault("enrollments", []).append(row)
    save_store(store)
    return {"id": eid, "next_send_at": row["next_send_at"]}


@router.post("/enrollments/{eid}/pause")
async def pause_enrollment(eid: str):
    store = load_store()
    for e in store.get("enrollments") or []:
        if isinstance(e, dict) and e.get("id") == eid:
            if e.get("status") == "active":
                e["status"] = "paused"
                save_store(store)
            return {"id": eid, "status": e.get("status")}
    raise HTTPException(status_code=404, detail="Enrollment not found")


@router.post("/enrollments/{eid}/resume")
async def resume_enrollment(eid: str):
    store = load_store()
    now = _utcnow()
    for e in store.get("enrollments") or []:
        if isinstance(e, dict) and e.get("id") == eid:
            if e.get("status") == "paused":
                e["status"] = "active"
                if not e.get("next_send_at"):
                    e["next_send_at"] = _iso(now)
                save_store(store)
            return {"id": eid, "status": e.get("status")}
    raise HTTPException(status_code=404, detail="Enrollment not found")


@router.post("/enrollments/{eid}/cancel")
async def cancel_enrollment(eid: str):
    store = load_store()
    for e in store.get("enrollments") or []:
        if isinstance(e, dict) and e.get("id") == eid:
            e["status"] = "cancelled"
            e.pop("next_send_at", None)
            save_store(store)
            return {"id": eid, "status": "cancelled"}
    raise HTTPException(status_code=404, detail="Enrollment not found")
