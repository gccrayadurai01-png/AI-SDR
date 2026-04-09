"""
OAuth2 mail sending: Gmail API + Microsoft Graph (same pattern as desktop/mobile mail apps).

Requires OAuth app credentials in .env (Google Cloud + Azure app registration).
Tokens stored in data/email_oauth_tokens.json (refresh tokens — protect this file).
"""

from __future__ import annotations

import base64
import json
import logging
import re
import secrets
import time
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse

import config
from storage import DATA_DIR

logger = logging.getLogger(__name__)

TOKEN_FILE = DATA_DIR / "email_oauth_tokens.json"
# state -> (provider "google"|"microsoft", expiry_unix)
_pending_oauth: dict[str, tuple[str, float]] = {}
_STATE_TTL_SEC = 600.0

router = APIRouter(prefix="/api/email-oauth", tags=["email-oauth"])


def _token_store_path() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return TOKEN_FILE


def load_token_store() -> dict[str, Any]:
    p = _token_store_path()
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        logger.exception("email_oauth: corrupt token file")
        return {}


def save_token_store(store: dict[str, Any]) -> None:
    p = _token_store_path()
    p.write_text(json.dumps(store, indent=2, default=str), encoding="utf-8")


def oauth_account_ready(which: str) -> bool:
    """which: google | microsoft"""
    st = load_token_store()
    key = "google" if which == "google" else "microsoft"
    row = st.get(key)
    if not isinstance(row, dict):
        return False
    return bool((row.get("refresh_token") or "").strip())


def oauth_connection_status() -> dict[str, Any]:
    config.reload_secrets()
    st = load_token_store()
    g = st.get("google") if isinstance(st.get("google"), dict) else {}
    m = st.get("microsoft") if isinstance(st.get("microsoft"), dict) else {}
    return {
        "google_connected": bool((g.get("refresh_token") or "").strip()),
        "google_email": (g.get("email") or "").strip(),
        "microsoft_connected": bool((m.get("refresh_token") or "").strip()),
        "microsoft_email": (m.get("email") or "").strip(),
        "google_oauth_configured": bool(
            (config.GOOGLE_OAUTH_CLIENT_ID or "").strip() and (config.GOOGLE_OAUTH_CLIENT_SECRET or "").strip()
        ),
        "microsoft_oauth_configured": bool(
            (config.MICROSOFT_OAUTH_CLIENT_ID or "").strip()
            and (config.MICROSOFT_OAUTH_CLIENT_SECRET or "").strip()
        ),
        "google_oauth_client_id": (config.GOOGLE_OAUTH_CLIENT_ID or "").strip(),
        "google_oauth_client_secret_set": config.env_file_nonempty("GOOGLE_OAUTH_CLIENT_SECRET"),
        "microsoft_oauth_client_id": (config.MICROSOFT_OAUTH_CLIENT_ID or "").strip(),
        "microsoft_oauth_client_secret_set": config.env_file_nonempty("MICROSOFT_OAUTH_CLIENT_SECRET"),
        "microsoft_oauth_tenant": (config.MICROSOFT_OAUTH_TENANT or "common").strip(),
    }


def _redirect_base() -> str:
    config.reload_secrets()
    return (config.APP_BASE_URL or "http://localhost:8000").strip().rstrip("/")


def _patch_env_email_from(email: str) -> None:
    if not email or "@" not in email:
        return
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        env_path.write_text(f"EMAIL_FROM={email}\n", encoding="utf-8")
        config.reload_secrets()
        return
    text = env_path.read_text(encoding="utf-8")
    line = re.compile(r"^EMAIL_FROM\s*=.*$", re.MULTILINE)
    if line.search(text):
        text = line.sub(f"EMAIL_FROM={email}", text)
    else:
        sep = "" if text.endswith("\n") else "\n"
        text = text.rstrip() + sep + f"EMAIL_FROM={email}\n"
    env_path.write_text(text, encoding="utf-8")
    config.reload_secrets()


def _pop_state(state: str | None, expected: str) -> None:
    if not state or state not in _pending_oauth:
        raise HTTPException(status_code=400, detail="Invalid or expired OAuth state — try Connect again")
    prov, exp = _pending_oauth.pop(state)
    if prov != expected or time.time() > exp:
        raise HTTPException(status_code=400, detail="OAuth state expired — try Connect again")


# --- Google ---

GOOGLE_AUTH = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN = "https://oauth2.googleapis.com/token"
GMAIL_SEND = "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"
GMAIL_PROFILE = "https://gmail.googleapis.com/gmail/v1/users/me/profile"


def _google_refresh_access_token(refresh_token: str) -> dict[str, Any]:
    cid = (config.GOOGLE_OAUTH_CLIENT_ID or "").strip()
    sec = (config.GOOGLE_OAUTH_CLIENT_SECRET or "").strip()
    if not cid or not sec:
        raise RuntimeError("GOOGLE_OAUTH_CLIENT_ID / GOOGLE_OAUTH_CLIENT_SECRET not set")
    data = {
        "client_id": cid,
        "client_secret": sec,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }
    with httpx.Client(timeout=30.0) as client:
        r = client.post(GOOGLE_TOKEN, data=data)
    if r.status_code >= 400:
        raise RuntimeError(f"Google token refresh failed: HTTP {r.status_code} {r.text[:400]}")
    return r.json()


def google_get_access_token() -> str:
    config.reload_secrets()
    st = load_token_store()
    g = st.get("google")
    if not isinstance(g, dict) or not (g.get("refresh_token") or "").strip():
        raise RuntimeError("Gmail not connected — use Connect Gmail in Settings")
    rt = g["refresh_token"].strip()
    exp = float(g.get("expires_at") or 0)
    now = time.time()
    if (g.get("access_token") or "").strip() and exp > now + 90:
        return str(g["access_token"])
    tok = _google_refresh_access_token(rt)
    access = str(tok.get("access_token") or "")
    if not access:
        raise RuntimeError("Google did not return access_token")
    expires_in = int(tok.get("expires_in") or 3600)
    g["access_token"] = access
    g["expires_at"] = now + max(60, expires_in - 120)
    st["google"] = g
    save_token_store(st)
    return access


def send_via_gmail_api(to_email: str, subject: str, body: str) -> None:
    config.reload_secrets()
    st = load_token_store()
    g = st.get("google") if isinstance(st.get("google"), dict) else {}
    from_email = (g.get("email") or "").strip() or (config.EMAIL_FROM or "").strip()
    m = re.match(r"^(.+?)\s*<([^>]+)>$", from_email)
    if m:
        from_email = m.group(2).strip()
    if not from_email or "@" not in from_email:
        raise RuntimeError("Set EMAIL_FROM or reconnect Gmail so the From address is known")
    raw = _gmail_raw(from_email, to_email, subject, body)
    token = google_get_access_token()
    payload = {"raw": raw}
    with httpx.Client(timeout=45.0) as client:
        r = client.post(
            GMAIL_SEND,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
        )
    if r.status_code >= 400:
        raise RuntimeError(f"Gmail API HTTP {r.status_code}: {r.text[:500]}")


def _gmail_raw(from_email: str, to_email: str, subject: str, body: str) -> str:
    msg = MIMEText(body, "plain", "utf-8")
    msg["To"] = to_email
    msg["From"] = from_email
    msg["Subject"] = subject
    return base64.urlsafe_b64encode(msg.as_bytes()).decode()


def test_gmail_connection() -> dict[str, Any]:
    try:
        token = google_get_access_token()
        with httpx.Client(timeout=20.0) as client:
            r = client.get(GMAIL_PROFILE, headers={"Authorization": f"Bearer {token}"})
        if r.status_code >= 400:
            return {"ok": False, "error": f"Gmail HTTP {r.status_code}: {r.text[:300]}"}
        data = r.json()
        return {"ok": True, "message": "Gmail API OK", "provider": "gmail_oauth", "email": data.get("emailAddress")}
    except Exception as e:
        logger.warning("Gmail test failed: %s", e)
        return {"ok": False, "error": str(e)}


# --- Microsoft Graph ---

def _ms_tenant() -> str:
    t = (config.MICROSOFT_OAUTH_TENANT or "common").strip().lower()
    return t if t else "common"


def _ms_authority_base() -> str:
    return f"https://login.microsoftonline.com/{_ms_tenant()}"


def _microsoft_refresh_access_token(refresh_token: str) -> dict[str, Any]:
    cid = (config.MICROSOFT_OAUTH_CLIENT_ID or "").strip()
    sec = (config.MICROSOFT_OAUTH_CLIENT_SECRET or "").strip()
    if not cid or not sec:
        raise RuntimeError("MICROSOFT_OAUTH_CLIENT_ID / MICROSOFT_OAUTH_CLIENT_SECRET not set")
    data = {
        "client_id": cid,
        "client_secret": sec,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
        "scope": "offline_access Mail.Send User.Read",
    }
    url = f"{_ms_authority_base()}/oauth2/v2.0/token"
    with httpx.Client(timeout=30.0) as client:
        r = client.post(url, data=data)
    if r.status_code >= 400:
        raise RuntimeError(f"Microsoft token refresh failed: HTTP {r.status_code} {r.text[:400]}")
    return r.json()


def microsoft_get_access_token() -> str:
    config.reload_secrets()
    st = load_token_store()
    m = st.get("microsoft")
    if not isinstance(m, dict) or not (m.get("refresh_token") or "").strip():
        raise RuntimeError("Outlook not connected — use Connect Outlook in Settings")
    rt = m["refresh_token"].strip()
    exp = float(m.get("expires_at") or 0)
    now = time.time()
    if (m.get("access_token") or "").strip() and exp > now + 90:
        return str(m["access_token"])
    tok = _microsoft_refresh_access_token(rt)
    access = str(tok.get("access_token") or "")
    if not access:
        raise RuntimeError("Microsoft did not return access_token")
    expires_in = int(tok.get("expires_in") or 3600)
    m["access_token"] = access
    m["expires_at"] = now + max(60, expires_in - 120)
    if tok.get("refresh_token"):
        m["refresh_token"] = str(tok["refresh_token"])
    st["microsoft"] = m
    save_token_store(st)
    return access


def send_via_microsoft_graph(to_email: str, subject: str, body: str) -> None:
    token = microsoft_get_access_token()
    payload = {
        "message": {
            "subject": subject,
            "body": {"contentType": "Text", "content": body},
            "toRecipients": [{"emailAddress": {"address": to_email}}],
        },
        "saveToSentItems": True,
    }
    with httpx.Client(timeout=45.0) as client:
        r = client.post(
            "https://graph.microsoft.com/v1.0/me/sendMail",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
        )
    if r.status_code >= 400:
        raise RuntimeError(f"Microsoft Graph HTTP {r.status_code}: {r.text[:500]}")


def test_microsoft_connection() -> dict[str, Any]:
    try:
        token = microsoft_get_access_token()
        with httpx.Client(timeout=20.0) as client:
            r = client.get(
                "https://graph.microsoft.com/v1.0/me",
                headers={"Authorization": f"Bearer {token}"},
            )
        if r.status_code >= 400:
            return {"ok": False, "error": f"Graph HTTP {r.status_code}: {r.text[:300]}"}
        data = r.json()
        mail = (data.get("mail") or data.get("userPrincipalName") or "") if isinstance(data, dict) else ""
        return {"ok": True, "message": "Microsoft Graph OK", "provider": "outlook_oauth", "email": mail}
    except Exception as e:
        logger.warning("Microsoft Graph test failed: %s", e)
        return {"ok": False, "error": str(e)}


# --- HTTP routes ---

def _frontend_redirect(query: str) -> RedirectResponse:
    base = _redirect_base()
    return RedirectResponse(url=f"{base}/?{query}", status_code=302)


@router.get("/google/start")
async def google_start():
    config.reload_secrets()
    cid = (config.GOOGLE_OAUTH_CLIENT_ID or "").strip()
    if not cid:
        raise HTTPException(status_code=400, detail="Set GOOGLE_OAUTH_CLIENT_ID in .env (Google Cloud OAuth client)")
    redirect_uri = _redirect_base() + "/api/email-oauth/google/callback"
    state = secrets.token_urlsafe(32)
    _pending_oauth[state] = ("google", time.time() + _STATE_TTL_SEC)
    params = {
        "client_id": cid,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "https://www.googleapis.com/auth/gmail.send",
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
    }
    url = f"{GOOGLE_AUTH}?{urlencode(params)}"
    return RedirectResponse(url=url, status_code=302)


@router.get("/google/callback")
async def google_callback(code: str | None = None, state: str | None = None, error: str | None = None):
    if error:
        return _frontend_redirect(f"email_oauth=error&detail={error}")
    if not code:
        return _frontend_redirect("email_oauth=error&detail=no_code")
    _pop_state(state, "google")
    config.reload_secrets()
    cid = (config.GOOGLE_OAUTH_CLIENT_ID or "").strip()
    sec = (config.GOOGLE_OAUTH_CLIENT_SECRET or "").strip()
    redirect_uri = _redirect_base() + "/api/email-oauth/google/callback"
    data = {
        "code": code,
        "client_id": cid,
        "client_secret": sec,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }
    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.post(GOOGLE_TOKEN, data=data)
        if r.status_code >= 400:
            return _frontend_redirect(f"email_oauth=error&detail=token_{r.status_code}")
        tok = r.json()
        rt = tok.get("refresh_token")
        at = tok.get("access_token")
        if not rt and not at:
            return _frontend_redirect("email_oauth=error&detail=no_token")
        st = load_token_store()
        g = st.get("google") if isinstance(st.get("google"), dict) else {}
        if rt:
            g["refresh_token"] = rt
        elif not (g.get("refresh_token") or "").strip():
            return _frontend_redirect("email_oauth=error&detail=no_refresh_token")
        if at:
            g["access_token"] = at
            g["expires_at"] = time.time() + float(tok.get("expires_in") or 3600) - 120
        email = ""
        if at:
            with httpx.Client(timeout=15.0) as client:
                r2 = client.get(GMAIL_PROFILE, headers={"Authorization": f"Bearer {at}"})
            if r2.status_code == 200:
                email = str(r2.json().get("emailAddress") or "")
        if email:
            g["email"] = email
            _patch_env_email_from(email)
        st["google"] = g
        save_token_store(st)
        logger.info("Gmail OAuth connected: %s", email or "(email pending)")
        return _frontend_redirect("email_oauth=google")
    except Exception as e:
        logger.exception("Google callback failed")
        return _frontend_redirect(f"email_oauth=error&detail={str(e)[:80]}")


@router.get("/microsoft/start")
async def microsoft_start():
    config.reload_secrets()
    cid = (config.MICROSOFT_OAUTH_CLIENT_ID or "").strip()
    if not cid:
        raise HTTPException(status_code=400, detail="Set MICROSOFT_OAUTH_CLIENT_ID in .env (Azure app registration)")
    redirect_uri = _redirect_base() + "/api/email-oauth/microsoft/callback"
    state = secrets.token_urlsafe(32)
    _pending_oauth[state] = ("microsoft", time.time() + _STATE_TTL_SEC)
    params = {
        "client_id": cid,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "response_mode": "query",
        "scope": "offline_access Mail.Send User.Read",
        "state": state,
    }
    url = f"{_ms_authority_base()}/oauth2/v2.0/authorize?{urlencode(params)}"
    return RedirectResponse(url=url, status_code=302)


@router.get("/microsoft/callback")
async def microsoft_callback(code: str | None = None, state: str | None = None, error: str | None = None):
    if error:
        return _frontend_redirect(f"email_oauth=error&detail={error}")
    if not code:
        return _frontend_redirect("email_oauth=error&detail=no_code")
    _pop_state(state, "microsoft")
    config.reload_secrets()
    cid = (config.MICROSOFT_OAUTH_CLIENT_ID or "").strip()
    sec = (config.MICROSOFT_OAUTH_CLIENT_SECRET or "").strip()
    redirect_uri = _redirect_base() + "/api/email-oauth/microsoft/callback"
    data = {
        "code": code,
        "client_id": cid,
        "client_secret": sec,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
        "scope": "offline_access Mail.Send User.Read",
    }
    try:
        url = f"{_ms_authority_base()}/oauth2/v2.0/token"
        with httpx.Client(timeout=30.0) as client:
            r = client.post(url, data=data)
        if r.status_code >= 400:
            return _frontend_redirect(f"email_oauth=error&detail=token_{r.status_code}")
        tok = r.json()
        rt = tok.get("refresh_token")
        at = tok.get("access_token")
        st = load_token_store()
        m = st.get("microsoft") if isinstance(st.get("microsoft"), dict) else {}
        if rt:
            m["refresh_token"] = rt
        elif not (m.get("refresh_token") or "").strip():
            return _frontend_redirect("email_oauth=error&detail=no_refresh_token")
        if at:
            m["access_token"] = at
            m["expires_at"] = time.time() + float(tok.get("expires_in") or 3600) - 120
        if at:
            with httpx.Client(timeout=15.0) as client:
                r2 = client.get(
                    "https://graph.microsoft.com/v1.0/me",
                    headers={"Authorization": f"Bearer {at}"},
                )
            if r2.status_code == 200:
                u = r2.json()
                em = str(u.get("mail") or u.get("userPrincipalName") or "")
                if em:
                    m["email"] = em
                    _patch_env_email_from(em)
        st["microsoft"] = m
        save_token_store(st)
        logger.info("Microsoft OAuth connected")
        return _frontend_redirect("email_oauth=outlook")
    except Exception as e:
        logger.exception("Microsoft callback failed")
        return _frontend_redirect(f"email_oauth=error&detail={str(e)[:80]}")


@router.get("/status")
async def http_oauth_status():
    config.reload_secrets()
    return oauth_connection_status()


@router.post("/disconnect")
async def disconnect(body: dict[str, Any] | None = None):
    payload = body or {}
    which = str(payload.get("provider") or "").strip().lower()
    if which not in ("google", "microsoft"):
        raise HTTPException(status_code=400, detail='provider must be "google" or "microsoft"')
    st = load_token_store()
    st.pop(which, None)
    save_token_store(st)
    return {"ok": True, "disconnected": which}
