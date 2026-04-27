"""
Twilio Voice Call Handler — drop-in alternative to Telnyx Call Control.

Architecture difference:
  - Telnyx: Stateful Call Control (make call → webhook events → call actions)
  - Twilio: TwiML-based (make call with URL → Twilio POSTs to our URL → we return XML)

Pricing (2026, US local outbound):
  - Outbound voice: $0.013/min  (Telnyx ~$0.007/min; Twilio costs more but is more reliable)
  - Inbound voice:  $0.0085/min
  - TTS (Polly via <Say>): FREE — included in voice pricing
  - STT (Gather speech): $0.000/min — Twilio includes it
  - No auto-disable on zero balance — charges payment method instead

Setup:
  1. Sign up at twilio.com (free trial = $15 credit)
  2. Get a phone number from Twilio
  3. Set env vars:
       TWILIO_ACCOUNT_SID   (from console.twilio.com)
       TWILIO_AUTH_TOKEN    (from console.twilio.com)
       TWILIO_PHONE_NUMBER  (+15551234567)
  4. Set VOICE_PROVIDER=twilio in Railway env vars

TTS Voice options (Polly neural, no extra cost):
  - Polly.Matthew-Neural  (US English, male)
  - Polly.Joanna-Neural   (US English, female)
  - Polly.Amy-Neural      (British English, female)
  - Polly.Brian-Neural    (British English, male)
  Set TWILIO_TTS_VOICE env var to change (default: Polly.Matthew-Neural)
"""

from __future__ import annotations

import logging
import os
import xml.sax.saxutils as saxutils
from typing import Any

logger = logging.getLogger(__name__)

_twilio_client: Any = None


def _get_client():
    """Lazy-init Twilio REST Client."""
    global _twilio_client
    if _twilio_client is not None:
        return _twilio_client
    try:
        from twilio.rest import Client as TwilioClient
    except ImportError as e:
        raise RuntimeError(
            "twilio package not installed. Run: pip install twilio"
        ) from e
    sid = os.environ.get("TWILIO_ACCOUNT_SID", "").strip()
    token = os.environ.get("TWILIO_AUTH_TOKEN", "").strip()
    if not sid or not token:
        raise RuntimeError(
            "TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN must be set in environment variables."
        )
    _twilio_client = TwilioClient(sid, token)
    return _twilio_client


def is_configured() -> bool:
    """True if Twilio env vars are set."""
    return bool(
        os.environ.get("TWILIO_ACCOUNT_SID")
        and os.environ.get("TWILIO_AUTH_TOKEN")
        and os.environ.get("TWILIO_PHONE_NUMBER")
    )


def get_tts_voice() -> str:
    """TwiML <Say> voice — Polly neural at no extra cost."""
    return os.environ.get("TWILIO_TTS_VOICE", "Polly.Matthew-Neural").strip()


# ─────────────────────────────────────────────
# 1. OUTBOUND CALL
# ─────────────────────────────────────────────
async def make_outbound_call(
    to_number: str,
    from_number: str | None = None,
    base_url: str | None = None,
) -> dict:
    """
    Place outbound call via Twilio REST API.
    Returns dict with `call_control_id` = Twilio CallSid (compatible with existing active_calls structure).
    """
    import config
    from_num = (from_number or os.environ.get("TWILIO_PHONE_NUMBER") or "").strip()
    if not from_num:
        raise RuntimeError("TWILIO_PHONE_NUMBER is not set.")
    webhook_base = (base_url or config.APP_BASE_URL).rstrip("/")
    client = _get_client()
    call = client.calls.create(
        to=to_number,
        from_=from_num,
        url=f"{webhook_base}/webhooks/twilio/answered",
        status_callback=f"{webhook_base}/webhooks/twilio/status",
        status_callback_method="POST",
        status_callback_event=["initiated", "ringing", "answered", "completed"],
        method="POST",
        # 60s ring timeout — hang up if not answered
        timeout=60,
    )
    logger.info(
        "Twilio call initiated: %s -> %s | SID=%s status=%s",
        from_num, to_number, call.sid, call.status,
    )
    return {
        "status": "initiated",
        "call_control_id": call.sid,   # Use SID as call_control_id for compat
        "call_leg_id": call.sid,
        "to": to_number,
        "provider": "twilio",
    }


# ─────────────────────────────────────────────
# 2. CALL CONTROL ACTIONS
# ─────────────────────────────────────────────
async def hangup_call(call_sid: str) -> None:
    """Terminate a Twilio call by updating its status to 'completed'."""
    try:
        client = _get_client()
        call = client.calls(call_sid).update(status="completed")
        logger.info("Twilio hangup: %s -> %s", call_sid, call.status)
    except Exception as e:
        logger.warning("Twilio hangup failed for %s: %s", call_sid, e)


# ─────────────────────────────────────────────
# 3. TWIML BUILDERS
# ─────────────────────────────────────────────
def _esc(text: str) -> str:
    """Escape text for TwiML XML."""
    return saxutils.escape(str(text or ""))


def make_twiml_gather(
    say_text: str,
    action_url: str,
    timeout: int = 10,
    speech_timeout: str = "auto",
    voice: str | None = None,
) -> str:
    """
    TwiML: <Say> greeting inside <Gather speech> so we listen after speaking.
    Falls back to pure XML if twilio package unavailable.
    """
    v = voice or get_tts_voice()
    try:
        from twilio.twiml.voice_response import VoiceResponse, Gather
        response = VoiceResponse()
        gather = Gather(
            input="speech",
            action=action_url,
            timeout=timeout,
            speech_timeout=speech_timeout,
            method="POST",
            language="en-US",
        )
        gather.say(say_text, voice=v)
        response.append(gather)
        # If Gather times out with no input, redirect back so we keep listening
        response.redirect(action_url, method="POST")
        return str(response)
    except ImportError:
        # Manual XML fallback
        escaped = _esc(say_text)
        url = _esc(action_url)
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<Response>\n"
            f'  <Gather input="speech" action="{url}" timeout="{timeout}" '
            f'speechTimeout="{speech_timeout}" method="POST" language="en-US">\n'
            f'    <Say voice="{v}">{escaped}</Say>\n'
            "  </Gather>\n"
            f'  <Redirect method="POST">{url}</Redirect>\n'
            "</Response>"
        )


def make_twiml_say_only(say_text: str, voice: str | None = None) -> str:
    """TwiML: <Say> then continue to gather (used when we just need to speak + keep listening)."""
    v = voice or get_tts_voice()
    try:
        from twilio.twiml.voice_response import VoiceResponse
        response = VoiceResponse()
        response.say(say_text, voice=v)
        return str(response)
    except ImportError:
        escaped = _esc(say_text)
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<Response>\n"
            f'  <Say voice="{v}">{escaped}</Say>\n'
            "</Response>"
        )


def make_twiml_say_hangup(say_text: str, voice: str | None = None) -> str:
    """TwiML: <Say> something and hang up."""
    v = voice or get_tts_voice()
    try:
        from twilio.twiml.voice_response import VoiceResponse
        response = VoiceResponse()
        response.say(say_text, voice=v)
        response.hangup()
        return str(response)
    except ImportError:
        escaped = _esc(say_text)
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<Response>\n"
            f'  <Say voice="{v}">{escaped}</Say>\n'
            "  <Hangup/>\n"
            "</Response>"
        )


def make_twiml_hangup() -> str:
    """TwiML: immediate hangup."""
    try:
        from twilio.twiml.voice_response import VoiceResponse
        r = VoiceResponse()
        r.hangup()
        return str(r)
    except ImportError:
        return '<?xml version="1.0" encoding="UTF-8"?><Response><Hangup/></Response>'


def make_twiml_pause_gather(
    pause_seconds: float,
    action_url: str,
    timeout: int = 8,
    voice: str | None = None,
) -> str:
    """TwiML: pause then gather (when we just need to listen after a short delay)."""
    v = voice or get_tts_voice()
    try:
        from twilio.twiml.voice_response import VoiceResponse, Gather, Pause
        response = VoiceResponse()
        gather = Gather(
            input="speech",
            action=action_url,
            timeout=timeout,
            speech_timeout="auto",
            method="POST",
            language="en-US",
        )
        gather.pause(length=int(pause_seconds))
        response.append(gather)
        response.redirect(action_url, method="POST")
        return str(response)
    except ImportError:
        url = _esc(action_url)
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<Response>\n"
            f'  <Gather input="speech" action="{url}" timeout="{timeout}" '
            f'speechTimeout="auto" method="POST">\n'
            f'    <Pause length="{int(pause_seconds)}"/>\n'
            "  </Gather>\n"
            f'  <Redirect method="POST">{url}</Redirect>\n'
            "</Response>"
        )


# ─────────────────────────────────────────────
# 4. DIAGNOSTICS
# ─────────────────────────────────────────────
def run_twilio_diagnostics() -> dict[str, Any]:
    """Check Twilio credentials and phone number availability."""
    issues: list[str] = []
    out: dict[str, Any] = {
        "provider": "twilio",
        "account_sid_set": bool(os.environ.get("TWILIO_ACCOUNT_SID")),
        "auth_token_set": bool(os.environ.get("TWILIO_AUTH_TOKEN")),
        "phone_number": os.environ.get("TWILIO_PHONE_NUMBER"),
    }
    if not out["account_sid_set"]:
        issues.append("TWILIO_ACCOUNT_SID is not set.")
    if not out["auth_token_set"]:
        issues.append("TWILIO_AUTH_TOKEN is not set.")
    if not out["phone_number"]:
        issues.append("TWILIO_PHONE_NUMBER is not set.")
    if issues:
        out["issues"] = issues
        out["ok"] = False
        return out
    try:
        client = _get_client()
        account = client.api.accounts(os.environ["TWILIO_ACCOUNT_SID"]).fetch()
        out["account_friendly_name"] = account.friendly_name
        out["account_status"] = account.status
        if account.status != "active":
            issues.append(f"Twilio account status is '{account.status}' (expected 'active').")
    except Exception as e:
        issues.append(f"Could not fetch Twilio account: {e}")
    out["issues"] = issues
    out["ok"] = len(issues) == 0
    return out
