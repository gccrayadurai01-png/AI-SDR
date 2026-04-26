"""
Telnyx Call Control Handler — v4 SDK
Uses Telnyx server-side transcription (no WebSocket audio streaming needed).
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from typing import Any

import httpx
import telnyx
from telnyx import omit

import config

logger = logging.getLogger(__name__)

# Dedupe duplicate Telnyx webhook deliveries / interim+final same text (cc_id -> (text, monotonic_ts))
_transcription_last: dict[str, tuple[str, float]] = {}

_tx: telnyx.Telnyx | None = None
_tx_sig: tuple[str, str, str, str] | None = None


def format_telnyx_exception(e: Exception) -> str:
    """
    Telnyx often returns a vague top-level message (e.g. 'internal call error').
    Prefer the JSON `errors[]` detail from APIStatusError.body when present.
    """
    if isinstance(e, telnyx.APIStatusError):
        parts: list[str] = [f"HTTP {e.status_code}"]
        base = str(e).strip()
        if base:
            parts.append(base)
        body = e.body
        if isinstance(body, dict):
            errs = body.get("errors")
            if isinstance(errs, list) and errs:
                for item in errs[:8]:
                    if isinstance(item, dict):
                        chunk = item.get("detail") or item.get("title") or item.get("code")
                        meta = item.get("meta") if isinstance(item.get("meta"), dict) else None
                        if meta:
                            chunk = f"{chunk} {meta}" if chunk else str(meta)
                        if chunk:
                            parts.append(str(chunk))
                    elif item:
                        parts.append(str(item))
            else:
                dumped = json.dumps(body, default=str)
                if dumped and dumped != "{}":
                    parts.append(dumped[:1800])
        elif body is not None:
            parts.append(str(body)[:1200])
        try:
            rt = (e.response.text or "").strip()
            if rt and all(rt not in p for p in parts):
                parts.append("raw=" + rt[:2000])
        except Exception:
            pass
        msg = " | ".join(p for p in parts if p)
    else:
        msg = str(e)
    if "internal call error" in msg.lower():
        msg += (
            " — Check Telnyx Mission Control: the Call Control `connection_id` must match the app "
            "that owns your `from` number; `from` must be E.164 (+country…); "
            "`APP_BASE_URL` must be reachable (HTTPS + /webhooks/telnyx) for call control."
        )
    return msg


def run_telnyx_diagnostics() -> dict[str, Any]:
    """
    Compare Mission Control to .env: Call Control App id, webhook URL, and
    phone number → connection_id (common cause of dial 'internal call error').
    """
    config.reload_secrets()
    issues: list[str] = []
    expected_wh = f"{config.APP_BASE_URL}/webhooks/telnyx"
    out: dict[str, Any] = {
        "expected_webhook_url": expected_wh,
        "app_base_url": config.APP_BASE_URL,
        "env_connection_id": config.TELNYX_CONNECTION_ID,
        "env_from_number": config.TELNYX_PHONE_NUMBER,
    }
    if not config.TELNYX_API_KEY:
        issues.append("TELNYX_API_KEY is not set.")
        out["issues"] = issues
        out["ok"] = False
        return out
    cid = (config.TELNYX_CONNECTION_ID or "").strip()
    if not cid:
        issues.append("TELNYX_CONNECTION_ID is empty — use the Call Control Application ID from Telnyx.")
        out["issues"] = issues
        out["ok"] = False
        return out
    tx = _client()
    try:
        r = tx.call_control_applications.retrieve(id=cid)
        d = r.data
        if d:
            ob = d.outbound
            out["call_control_application"] = {
                "id": d.id,
                "application_name": d.application_name,
                "active": d.active,
                "webhook_event_url": d.webhook_event_url,
                "webhook_api_version": d.webhook_api_version,
                "outbound_voice_profile_id": getattr(ob, "outbound_voice_profile_id", None) if ob else None,
                "outbound_channel_limit": getattr(ob, "channel_limit", None) if ob else None,
            }
            if ob and not getattr(ob, "outbound_voice_profile_id", None):
                issues.append(
                    "Call Control Application has no Outbound Voice Profile — assign one under Voice in Mission Control."
                )
            if d.active is False:
                issues.append(
                    "Call Control Application is inactive — enable it in Telnyx Mission Control."
                )
            wurl = (d.webhook_event_url or "").strip().rstrip("/")
            exp = expected_wh.rstrip("/")
            if wurl and wurl != exp:
                issues.append(
                    f"Mission Control Webhook URL is '{d.webhook_event_url}' but this app dials with "
                    f"'{expected_wh}'. Set the Call Control Application webhook to exactly: {expected_wh}"
                )
            if d.webhook_api_version is not None and str(d.webhook_api_version) != "2":
                issues.append(
                    f"Set webhook API version to 2 in the Call Control Application (currently {d.webhook_api_version})."
                )
    except Exception as e:
        out["call_control_application_error"] = format_telnyx_exception(e)
        issues.append(
            "Could not retrieve Call Control Application — wrong TELNYX_CONNECTION_ID or API key."
        )

    fn = config.TELNYX_PHONE_NUMBER or ""
    digits = "".join(c for c in fn if c.isdigit())
    if len(digits) < 10:
        issues.append("TELNYX_PHONE_NUMBER is missing or not a valid number.")
    else:
        try:
            lst = tx.phone_numbers.list(filter={"phone_number": digits}, page_size=20)
            found = None
            for p in lst:
                pd = "".join(c for c in (p.phone_number or "") if c.isdigit())
                if pd.endswith(digits) or digits.endswith(pd):
                    found = p
                    break
            if found:
                st = getattr(found, "status", None)
                out["phone_number"] = {
                    "phone_number": found.phone_number,
                    "connection_id": found.connection_id,
                    "status": st,
                }
                if st and str(st).lower() not in ("active",):
                    issues.append(f"Phone number status is '{st}' (expected active).")
                pid = (found.connection_id or "").strip()
                if pid and pid != cid:
                    issues.append(
                        f"MISMATCH: Your Telnyx number is assigned to connection_id {pid}, but "
                        f"TELNYX_CONNECTION_ID in .env is {cid}. In Mission Control → Numbers → your DID → Voice, "
                        "link the number to the same Call Control Application as in .env."
                    )
            else:
                issues.append(
                    f"No purchased number matching {fn} in this Telnyx account (searched digits {digits})."
                )
        except Exception as e:
            out["phone_number_error"] = format_telnyx_exception(e)
            issues.append("Could not look up phone numbers via Telnyx API.")

    out["issues"] = issues
    out["ok"] = len(issues) == 0
    return out


def _client() -> telnyx.Telnyx:
    """Lazy Telnyx client; rebuilds when API key or connection settings change."""
    global _tx, _tx_sig
    config.reload_secrets()  # pick up .env after Save Settings
    k = config.TELNYX_API_KEY
    if not k:
        raise RuntimeError("TELNYX_API_KEY is not set")
    sig = (
        k,
        config.TELNYX_PHONE_NUMBER or "",
        config.TELNYX_CONNECTION_ID or "",
        config.APP_BASE_URL or "",
    )
    if _tx is None or _tx_sig != sig:
        _tx = telnyx.Telnyx(api_key=k)
        _tx_sig = sig
    return _tx


# ─────────────────────────────────────────────
# 1. OUTBOUND CALL (no stream_url — we use transcription instead)
# ─────────────────────────────────────────────
async def make_outbound_call(to_number: str, from_number: str | None = None, connection_id: str | None = None) -> dict:
    """Place outbound call. Optional `from_number` and `connection_id` override
    config defaults — used for per-tenant phone routing in Phase 3."""
    try:
        result = _client().calls.dial(
            connection_id=connection_id or config.TELNYX_CONNECTION_ID,
            to=to_number,
            from_=from_number or config.TELNYX_PHONE_NUMBER,
            webhook_url=f"{config.APP_BASE_URL}/webhooks/telnyx",
            webhook_url_method="POST",
            answering_machine_detection="disabled",
        )
        data = result.data
        logger.info(f"Call initiated -> {to_number} | {data.call_control_id}")
        return {
            "status": "initiated",
            "call_control_id": data.call_control_id,
            "call_leg_id": data.call_leg_id,
            "to": to_number,
        }
    except Exception as e:
        logger.error("Call failed: %s", format_telnyx_exception(e))
        raise


# ─────────────────────────────────────────────
# 2. CALL CONTROL ACTIONS
# ─────────────────────────────────────────────
async def answer_call(call_control_id: str):
    _client().calls.actions.answer(call_control_id=call_control_id)
    logger.info(f"Answered: {call_control_id}")


async def hangup_call(call_control_id: str):
    _client().calls.actions.hangup(call_control_id=call_control_id)
    logger.info(f"Hung up: {call_control_id}")


_MAX_SPEAK_CHARS = 2800


def estimate_tts_playback_seconds(text: str) -> float:
    """
    Rough spoken duration for scheduling listen resume when call.speak.ended is delayed or missing.
    Biased slightly long so we rarely restart STT while our side is still talking (echo).
    """
    t = (text or "").strip()
    if not t:
        return 3.0
    words = len(t.split())
    return min(26.0, max(3.0, words * 0.52 + 1.0))


async def _speak_via_telnyx_speak(call_control_id: str, payload: str) -> None:
    """Telnyx native speak() — Polly/Azure or ElevenLabs via Telnyx integration secret."""
    voice = config.telnyx_speak_voice_for_api()
    vs = config.elevenlabs_voice_settings()
    kwargs: dict = {
        "call_control_id": call_control_id,
        "payload": payload,
        "voice": voice,
        "language": omit if vs else config.TELNYX_SPEAK_LANGUAGE,
        "service_level": config.TELNYX_SPEAK_SERVICE_LEVEL,
    }
    if vs:
        kwargs["voice_settings"] = vs
        logger.info(
            "TTS ElevenLabs (Telnyx secret): voice=%s api_key_ref=%s",
            voice,
            config.ELEVENLABS_API_KEY_REF,
        )
    try:
        _client().calls.actions.speak(**kwargs)
    except telnyx.APIStatusError as e:
        logger.error(
            "Telnyx speak failed (%s): %s — body=%s",
            getattr(e, "status_code", "?"),
            e,
            getattr(e, "body", None),
        )
        raise
    except Exception as e:
        logger.error("Telnyx speak failed: %s", e, exc_info=True)
        raise
    logger.info("Speaking (%s): '%s...'", voice, payload[:80])


async def _speak_via_telnyx_polly_only(call_control_id: str, payload: str) -> None:
    """Polly/Azure only — used when ElevenLabs direct failed. Do NOT use Telnyx+ElevenLabs speak here: it often defaults to a female voice."""
    config.reload_secrets()
    voice = config.TELNYX_SPEAK_VOICE or "AWS.Polly.Matthew-Neural"
    kwargs: dict = {
        "call_control_id": call_control_id,
        "payload": payload,
        "voice": voice,
        "language": config.TELNYX_SPEAK_LANGUAGE,
        "service_level": config.TELNYX_SPEAK_SERVICE_LEVEL,
    }
    logger.warning("TTS Polly fallback (voice=%s) — ElevenLabs direct failed; skipping Telnyx ElevenLabs integration", voice)
    try:
        _client().calls.actions.speak(**kwargs)
    except telnyx.APIStatusError as e:
        logger.error(
            "Polly fallback speak failed (%s): %s — body=%s",
            getattr(e, "status_code", "?"),
            e,
            getattr(e, "body", None),
        )
        raise


async def _elevenlabs_direct_to_playback(call_control_id: str, payload: str) -> bool:
    """
    ElevenLabs REST → MP3 → Telnyx start_playback. Tries stream vs non-stream if the first attempt fails.
    Returns True on success.
    """
    vid_raw = (config.ELEVENLABS_VOICE_ID or "").strip()
    if not vid_raw:
        return False
    # Voice ID goes in URL path (alphanumeric + hyphens; no extra encoding needed for standard EL IDs)
    vid_enc = vid_raw
    el_body = {
        "text": payload,
        "model_id": config.ELEVENLABS_MODEL_ID or "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.70,
            "similarity_boost": 0.85,
        },
    }
    headers = {
        "xi-api-key": config.ELEVENLABS_API_KEY or "",
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    params: dict[str, str] = {"output_format": "mp3_22050_32"}

    async def _fetch_mp3(use_stream: bool) -> bytes:
        url = (
            f"https://api.elevenlabs.io/v1/text-to-speech/{vid_enc}/stream"
            if use_stream
            else f"https://api.elevenlabs.io/v1/text-to-speech/{vid_enc}"
        )
        p = dict(params)
        if use_stream:
            p["optimize_streaming_latency"] = "2"
        async with httpx.AsyncClient(timeout=15.0) as ac:
            if use_stream:
                async with ac.stream("POST", url, json=el_body, headers=headers, params=p) as r:
                    r.raise_for_status()
                    parts: list[bytes] = []
                    async for chunk in r.aiter_bytes():
                        if chunk:
                            parts.append(chunk)
                    return b"".join(parts)
            r = await ac.post(url, json=el_body, headers=headers, params=p)
            r.raise_for_status()
            return r.content

    preferred = config.ELEVENLABS_HTTP_STREAM
    for use_stream in (preferred, not preferred):
        try:
            mp3 = await _fetch_mp3(use_stream)
            if len(mp3) < 64:
                raise RuntimeError(f"ElevenLabs returned empty/short audio ({len(mp3)} bytes)")
            b64 = base64.b64encode(mp3).decode("ascii")
            logger.info(
                "TTS ElevenLabs direct → start_playback (voice_id=%s, stream=%s, bytes=%d)",
                vid_raw,
                use_stream,
                len(mp3),
            )
            last_err: Exception | None = None
            for attempt in range(2):
                try:
                    _client().calls.actions.start_playback(
                        call_control_id=call_control_id,
                        playback_content=b64,
                        audio_type="mp3",
                    )
                    return True
                except Exception as e:
                    last_err = e
                    logger.warning("start_playback attempt %s: %s", attempt + 1, e)
                    if attempt == 0:
                        await asyncio.sleep(0.12)
            logger.error("start_playback failed after ElevenLabs OK: %s", last_err, exc_info=last_err)
        except Exception as e:
            logger.warning(
                "ElevenLabs direct attempt stream=%s failed: %s",
                use_stream,
                e,
                exc_info=True,
            )
    return False


async def speak_on_call(call_control_id: str, text: str):
    """
    1) ElevenLabs REST (xi-api-key) → Telnyx start_playback when configured.
    2) If (1) was configured but failed: Polly only — NOT Telnyx speak+ElevenLabs (often sounds like a default female).
    3) If no direct API key: Telnyx speak + ElevenLabs integration secret, else Polly.
    """
    config.reload_secrets()
    payload = (text or "").strip()
    if not payload:
        return
    if len(payload) > _MAX_SPEAK_CHARS:
        payload = payload[: _MAX_SPEAK_CHARS - 3] + "..."

    want_direct = bool(
        config.ELEVENLABS_DIRECT_FIRST
        and config.ELEVENLABS_API_KEY
        and config.ELEVENLABS_VOICE_ID
    )
    if want_direct:
        ok = await _elevenlabs_direct_to_playback(call_control_id, payload)
        if ok:
            return
        logger.error(
            "ElevenLabs direct failed completely — using Polly only. "
            "Check logs above (401/403 = bad API key; 404 = wrong voice_id). "
            "Telnyx+ElevenLabs speak fallback is disabled here because it often uses a default female voice."
        )
        await _speak_via_telnyx_polly_only(call_control_id, payload)
        return

    await _speak_via_telnyx_speak(call_control_id, payload)


async def start_transcription(call_control_id: str):
    """Start Telnyx server-side transcription. Sends call.transcription webhooks."""
    try:
        _client().calls.actions.start_transcription(
            call_control_id=call_control_id,
            transcription_tracks="inbound",
        )
        logger.info(f"Transcription started: {call_control_id}")
    except Exception as e:
        logger.error(f"Transcription start failed: {e}")
        try:
            _client().calls.actions.start_transcription(
                call_control_id=call_control_id,
            )
            logger.info(f"Transcription started (default): {call_control_id}")
        except Exception as e2:
            logger.error(f"Transcription fallback also failed: {e2}")


async def stop_transcription(call_control_id: str):
    try:
        _client().calls.actions.stop_transcription(
            call_control_id=call_control_id,
        )
    except Exception as e:
        logger.warning(f"Transcription stop: {e}")


async def start_recording(call_control_id: str):
    try:
        _client().calls.actions.start_recording(
            call_control_id=call_control_id,
            format="mp3",
            channels="single",
        )
        logger.info(f"Recording started: {call_control_id}")
    except Exception as e:
        logger.warning(f"Recording start failed (non-fatal): {e}")


async def stop_recording(call_control_id: str):
    try:
        _client().calls.actions.stop_recording(call_control_id=call_control_id)
    except Exception as e:
        logger.warning(f"Recording stop: {e}")


# ─────────────────────────────────────────────
# 3. WEBHOOK PARSER
# ─────────────────────────────────────────────
def parse_transcription_from_payload(pl: dict) -> tuple[str, bool]:
    """
    Extract (text, is_final) from Telnyx call.transcription payload.

    If `is_final` / `final` is omitted, treat as final (Telnyx often omits on single-shot).
    Using default False for missing is_final caused zero replies (only opener heard).
    """
    td = pl.get("transcription_data")
    if td is None:
        td = pl.get("transcription") or {}
    if isinstance(td, list) and td:
        td = td[0]
    if isinstance(td, str):
        text = td.strip()
        is_final = True
        return text, is_final
    if not isinstance(td, dict):
        td = {}
    text = (
        (td.get("transcript") or td.get("transcription_text") or td.get("text") or "")
        .strip()
    )
    alts = td.get("alternatives")
    if not text and isinstance(alts, list) and alts:
        a0 = alts[0]
        if isinstance(a0, dict):
            text = (a0.get("transcript") or a0.get("text") or "").strip()
    if "is_final" in td:
        is_final = bool(td.get("is_final"))
    elif "final" in td:
        is_final = bool(td.get("final"))
    else:
        is_final = True
    return text, is_final


def extract_call_control_id_from_body(body: dict) -> str | None:
    """Resolve call_control_id — required for speak/stop; some events omit it in nested payload."""
    for top in (body.get("call_control_id"), body.get("call_leg_id")):
        if isinstance(top, str) and top.strip():
            return top.strip()
    data = body.get("data") or {}
    pl = data.get("payload")
    if not isinstance(pl, dict):
        pl = {}
    cid = (
        pl.get("call_control_id")
        or data.get("call_control_id")
        or pl.get("call_leg_id")
        or data.get("call_leg_id")
    )
    if isinstance(cid, str) and cid.strip():
        return cid.strip()
    return None


def normalize_telnyx_event_type(etype: str) -> str:
    """Map call.transcription.* variants to call.transcription for a single handler."""
    if not etype or etype == "unknown":
        return etype or "unknown"
    if etype.startswith("call.transcription"):
        return "call.transcription"
    return etype


def parse_call_transcription_event(body: dict) -> tuple[str, bool, str | None]:
    """
    Parse full Telnyx POST body for call.transcription.
    Returns (text, is_final, call_control_id).
    """
    data = body.get("data") or {}
    pl = data.get("payload")
    if not isinstance(pl, dict):
        pl = {}
    cc_id = extract_call_control_id_from_body(body)

    text, is_final = parse_transcription_from_payload(pl)

    if not text:
        # Some API versions nest transcription only under data
        slim = {
            k: v
            for k, v in data.items()
            if k not in ("record_type", "event_type", "id", "occurred_at", "payload")
        }
        text, is_final = parse_transcription_from_payload(slim)

    if not text:
        td = data.get("transcription_data")
        if isinstance(td, dict):
            text, is_final = parse_transcription_from_payload({"transcription_data": td})
        elif isinstance(td, str) and td.strip():
            text, is_final = td.strip(), True

    if not text:
        for key in ("transcription_text", "text", "transcript"):
            v = data.get(key)
            if isinstance(v, str) and v.strip():
                text, is_final = v.strip(), True
                break

    return text, is_final, cc_id


def should_emit_transcription_reply(cc_id: str, text: str, is_final: bool) -> bool:
    """
    Decide whether to run the LLM reply for this webhook.
    When Telnyx only sends is_final=false, optional interim mode + dedupe avoids silence.
    """
    t = (text or "").strip()
    if not t:
        return False
    if is_final:
        pass
    else:
        if not config.TRANSCRIPTION_REPLY_ON_INTERIM:
            return False
        # Telnyx often sends short interim fragments ("yes", "hi", "ok") before final; len>=4 was too strict.
        if len(t) < 2:
            return False

    now = time.monotonic()
    key = cc_id or ""
    prev = _transcription_last.get(key)
    if prev and prev[0] == t and (now - prev[1]) < 3.0:
        logger.info("Transcription dedupe skip (same text within 3s): %r", t[:80])
        return False
    _transcription_last[key] = (t, now)
    return True


def parse_webhook_event(payload: dict) -> dict:
    try:
        data    = payload.get("data", {})
        event   = data.get("event_type", "unknown")
        pl      = data.get("payload", {})
        return {
            "event_type":      event,
            "call_control_id": pl.get("call_control_id"),
            "call_leg_id":     pl.get("call_leg_id"),
            "direction":       pl.get("direction"),
            "from":            pl.get("from"),
            "to":              pl.get("to"),
            "state":           pl.get("state"),
            "raw":             pl,
        }
    except Exception as e:
        logger.error(f"Webhook parse error: {e}")
        return {"event_type": "parse_error", "error": str(e)}
