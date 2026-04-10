"""
AI SDR — FastAPI server
Uses Telnyx server-side transcription (no WebSocket audio streaming).
Flow: call.answered → speak → start_transcription → call.transcription → Claude → speak → loop
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path

from typing import Any

from fastapi import BackgroundTasks, Body, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import anthropic
from anthropic import AsyncAnthropic

import apollo_client
import campaign as campaign_lib
from campaign import (
    normalize_phone,
    pause_campaign,
    prospect_display_name,
    resume_campaign,
    signal_call_ended,
    start_campaign,
    stop_campaign,
)
import config
import qa_kb
from email_sequences import (
    router as email_sequences_router,
    email_delivery_ready,
    smtp_ready,
    start_email_scheduler,
    stop_email_scheduler,
    test_email_delivery,
    test_smtp_connection,
)
from qa_kb_api import router as learn_router
from email_oauth import router as email_oauth_router, oauth_connection_status
from knowledge_base import get_full_knowledge, UPLOADED_DOCS_KNOWLEDGE, CLOUDFUZE_KNOWLEDGE
from prospect_import import parse_csv_bytes, parse_xlsx_bytes
from sdr_agent import (
    join_streamed_reply_parts,
    opening_line,
    next_sdr_reply,
    stream_sdr_reply_sentences,
)
from telnyx_handler import (
    format_telnyx_exception,
    make_outbound_call,
    run_telnyx_diagnostics,
    answer_call,
    hangup_call,
    speak_on_call,
    start_transcription,
    start_recording,
    stop_transcription,
    parse_webhook_event,
    parse_call_transcription_event,
    should_emit_transcription_reply,
    extract_call_control_id_from_body,
    normalize_telnyx_event_type,
    estimate_tts_playback_seconds,
)
import contacts_store
from post_call_email import resolve_prospect_email, run_post_call_followup_email
from storage import (
    load_calls,
    save_call,
    update_call,
    get_call_by_control_id,
    finalize_call_end,
    mark_stale_initiated_calls,
    load_script,
    save_script,
    load_tasks,
    save_task,
    delete_task,
    update_task,
)

# ─── logging ────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def _app_lifespan(app: FastAPI):
    """Starts email sequence scheduler; outbound Telnyx flow is unchanged."""
    start_email_scheduler()
    yield
    stop_email_scheduler()


# ─── app ────────────────────────────────────────────────────
app = FastAPI(title="AI SDR", version="3.0.0", lifespan=_app_lifespan)
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.include_router(email_sequences_router)
app.include_router(learn_router)
app.include_router(email_oauth_router)

# In-memory active calls: { call_control_id: {...} }
active_calls: dict = {}

# Conversation history per call: { call_control_id: [{"role": ..., "content": ...}] }
conversations: dict[str, list] = {}

# Track if AI is currently speaking (prevent interruption loops)
speaking_calls: set = set()

# Track calls that already got the opening line (prevent repeats)
opened_calls: set = set()

# Track AI Assistant activity (watchdog: detect silent assistant)
_ai_assistant_first_event: dict[str, float] = {}  # cc_id -> timestamp of first AI event

# ─── Telnyx AI Assistant (speech-to-speech) ───────────────
import telnyx
import httpx as _httpx

ASSISTANT_ID = "assistant-0a0eb33a-5fc0-4dd1-b366-8f0f1432be42"

_tx: telnyx.Telnyx | None = None
_tx_sig: tuple[str, str, str, str] | None = None


def _get_tx() -> telnyx.Telnyx:
    global _tx, _tx_sig
    config.reload_secrets()
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


# Pre-cached filler audio (ElevenLabs Anthony voice, generated at startup)
_filler_audio_cache: dict[str, str] = {}  # filler_text -> base64 MP3
_filler_playing: dict[str, bool] = {}     # cc_id -> True if filler playing
_last_filler_time: dict[str, float] = {}  # cc_id -> timestamp

# ── Hot-path caches (rebuilt on script save + startup) ──
_cached_script: dict[str, Any] = {}
_cached_knowledge_history: list[dict] = []
_cached_voice_kwargs: dict[str, Any] = {}  # voice + voice_settings pre-built


def _rebuild_hot_cache():
    """Rebuild all hot-path caches from current script + config. Called on startup + script save."""
    global _cached_script, _cached_knowledge_history, _cached_voice_kwargs
    _cached_script = load_script()
    _cached_knowledge_history = get_knowledge_message_history()
    vkw: dict[str, Any] = {}
    vid = config.ELEVENLABS_VOICE_ID
    ref = config.ELEVENLABS_API_KEY_REF
    if vid and ref:
        vkw["voice"] = f"ElevenLabs.eleven_multilingual_v2.{vid}"
        vkw["voice_settings"] = {
            "type": "elevenlabs",
            "api_key_ref": ref,
            "voice_speed": 0.9,
        }
    else:
        vkw["voice"] = config.TELNYX_SPEAK_VOICE or "AWS.Polly.Matthew-Neural"
    _cached_voice_kwargs = vkw
    logger.info("Hot cache rebuilt: script=%d keys, knowledge=%d msgs, voice=%s",
                len(_cached_script), len(_cached_knowledge_history), vkw.get("voice", "?")[:40])


def get_system_prompt() -> str:
    """Sandler system prompt for Telnyx AI Assistant -- MUST stay under 1500 chars."""
    s = load_script()
    sdr = s.get("sdr_name", "Anthony")
    co = s.get("company_name", "CloudFuze")
    obj = s.get("call_objective", "Book a 15-minute discovery call")
    persona = s.get("target_persona", "IT / SaaS decision makers")

    return f"""You={sdr}, SDR at {co}. LIVE cold call. Never say you're AI.
Goal: {obj}. Target: {persona}.

SANDLER SELLING SYSTEM -- run the call autonomously:
1. PATTERN INTERRUPT: "Got a quick second?" Warm, casual, disarming.
2. UP-FRONT CONTRACT: "I'll keep it to 15 min. If it doesn't fit, totally cool. Fair?"
3. PAIN DISCOVERY: Ask about their world. ONE question per turn. Listen. Reflect back. Go deeper. Share a brief insight when natural, then ask the next question.
4. NEGATIVE REVERSE: If they resist, gently challenge: "Maybe this isn't a big deal for you right now?"
5. QUALIFY: Pain level, budget, decision makers. "Who else weighs in?"
6. SOFT CLOSE: "Does 15 min make sense to explore this?" Propose a day/time.
7. POST-SELL: Lock it, send invite, thank them.

STYLE:
- 1-2 sentences max per turn. ONE question at a time.
- EXCHANGE: they share, you give a relevant insight back, then next question.
- Sound curious, human, conversational. "yeah", "got it", "makes sense".
- Answer their questions honestly and briefly, then steer back.
- Never list features. Never monologue. Never pitch-dump.
OBJECTIONS: Busy=when's better. Not interested=ask what's behind it. Has tool=what's missing? No budget=usually saves money.
END: Confirm, thank, bye.""".strip()


def get_opening_line(name: str = "there", title: str = "", company: str = "") -> str:
    """Generate opening line — AI decides the style, we just fill in the names."""
    s = load_script()
    sdr = s.get("sdr_name", "Alex")
    co = s.get("company_name", "CloudFuze")
    # Simple pattern interrupt opener — the AI will take it from here
    line = f"Hey {name}, this is {sdr} from {co} -- did I catch you at a bad time?"
    return line


def _get_compact_knowledge() -> str:
    """Product knowledge injected as message_history -- keeps system prompt short for Telnyx."""
    s = load_script()
    manage = s.get("knowledge_manage", "")
    pain = s.get("knowledge_pain_context", "")
    metrics = s.get("knowledge_metrics", "")
    competitive = s.get("knowledge_competitive", "")

    # Also load the Sandler talking blocks for reference
    discovery = s.get("discovery_questions", [])
    reversals = s.get("negative_reversals", [])
    insights = s.get("insight_value_points", [])
    closes = s.get("soft_close_cta", [])
    objections = s.get("objection_handling", {})

    discovery_str = " | ".join(discovery[:4]) if isinstance(discovery, list) else str(discovery)
    reversals_str = " | ".join(reversals[:3]) if isinstance(reversals, list) else str(reversals)
    insights_str = " | ".join(insights[:4]) if isinstance(insights, list) else str(insights)
    closes_str = " | ".join(closes[:2]) if isinstance(closes, list) else str(closes)
    obj_str = " | ".join(f"{k}: {v}" for k,v in objections.items()) if isinstance(objections, dict) else str(objections)

    return f"""CLOUDFUZE KNOWLEDGE -- weave into conversation naturally, never dump as a list.

PRODUCT: {manage}

PAIN CONTEXT: {pain}

METRICS (use for credibility): {metrics}

COMPETITIVE: {competitive}

SAMPLE DISCOVERY Qs (adapt, don't read verbatim): {discovery_str}

NEGATIVE REVERSALS (use when they resist): {reversals_str}

INSIGHTS TO SHARE (weave in naturally): {insights_str}

SOFT CLOSE OPTIONS: {closes_str}

OBJECTION RESPONSES: {obj_str}"""


def get_knowledge_message_history() -> list[dict]:
    """Return compact knowledge as message_history for AI Assistant — optimized for low latency."""
    knowledge = _get_compact_knowledge()
    if not knowledge:
        return []
    return [
        {"role": "user", "content": f"[INTERNAL — never read aloud]\n{knowledge}"},
        {"role": "assistant", "content": "Got it."},
    ]


def sync_assistant_to_script():
    """Push current script config to the Telnyx AI Assistant + tune for minimum latency."""
    try:
        instructions = get_system_prompt()
        _get_tx().ai.assistants.update(
            assistant_id=ASSISTANT_ID,
            instructions=instructions,
        )
        logger.info("Synced script to Telnyx AI Assistant")
    except Exception as e:
        logger.error(f"Failed to sync assistant: {e}")

    # Tune assistant via raw HTTP PATCH (voice + transcription + model)
    try:
        import httpx
        api_key = config.TELNYX_API_KEY
        if api_key:
            instructions = get_system_prompt()
            patch_body: dict[str, Any] = {
                "instructions": instructions,
                "model": "anthropic/claude-haiku-4-5",
                "transcription": {"model": "distil-whisper/distil-large-v2"},
                "llm_temperature": 0.7,
                "telephony_settings": {
                    "user_idle_timeout_seconds": 15,
                },
            }
            voice_id = config.ELEVENLABS_VOICE_ID
            api_key_ref = config.ELEVENLABS_API_KEY_REF
            if voice_id and api_key_ref:
                patch_body["voice_settings"] = {
                    "voice": f"ElevenLabs.eleven_multilingual_v2.{voice_id}",
                    "api_key_ref": api_key_ref,
                    "voice_speed": 0.9,
                    "similarity_boost": 0.85,
                    "style": 0.35,
                    "use_speaker_boost": True,
                }
            r = httpx.patch(
                f"https://api.telnyx.com/v2/ai/assistants/{ASSISTANT_ID}",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=patch_body,
                timeout=15.0,
            )
            if r.status_code < 300:
                logger.info("Assistant synced: claude-haiku-4-5, distil-whisper, ElevenLabs")
            else:
                logger.warning("Assistant PATCH %s: %s", r.status_code, r.text[:300])
    except Exception as e:
        logger.warning("Assistant PATCH failed (non-fatal): %s", e)


async def _precache_filler_audio():
    """Pre-generate filler phrases via ElevenLabs at startup. Cached = instant playback."""
    import httpx
    config.reload_secrets()
    voice_id = config.ELEVENLABS_VOICE_ID
    api_key = config.ELEVENLABS_API_KEY
    if not voice_id or not api_key:
        logger.warning("ElevenLabs not configured — filler audio not pre-cached")
        return

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {"xi-api-key": api_key, "Content-Type": "application/json", "Accept": "audio/mpeg"}
    params = {"output_format": "mp3_22050_32"}

    async with httpx.AsyncClient(timeout=30.0) as ac:
        for phrase in config.PHONE_FILLER_UTTERANCES:
            try:
                r = await ac.post(url, json={
                    "text": phrase,
                    "model_id": "eleven_turbo_v2_5",
                }, headers=headers, params=params)
                r.raise_for_status()
                import base64
                b64 = base64.b64encode(r.content).decode("ascii")
                _filler_audio_cache[phrase] = b64
                logger.info("Cached filler: \"%s\" (%d bytes)", phrase, len(r.content))
            except Exception as e:
                logger.warning("Failed to cache filler \"%s\": %s", phrase, e)

    logger.info("Pre-cached %d/%d filler phrases (Anthony turbo)", len(_filler_audio_cache), len(config.PHONE_FILLER_UTTERANCES))


import re as _re

_GOODBYE_PATTERNS = _re.compile(
    r'\b(goodbye|good\s*bye|bye\s*bye|talk\s*soon|have\s*a\s*great|'
    r'take\s*care|appreciate\s*your\s*time|thanks\s*for\s*your\s*time|'
    r'nice\s*talking|nice\s*chatting|have\s*a\s*good\s*one|'
    r'catch\s*you\s*later|speak\s*soon|cheers|so\s*long)\b',
    _re.IGNORECASE,
)

# Track pending auto-hangup tasks so we can cancel if conversation continues
_auto_hangup_tasks: dict[str, asyncio.Task] = {}


def _is_goodbye(text: str) -> bool:
    """Check if text contains a natural conversation-ending phrase."""
    return bool(_GOODBYE_PATTERNS.search(text or ""))


async def _auto_hangup_after_goodbye(cc_id: str):
    """Wait a few seconds after goodbye, then auto-hangup if no new speech."""
    try:
        await asyncio.sleep(4.0)  # Give 4s for any follow-up
        if cc_id not in active_calls:
            return
        if active_calls[cc_id].get("state") == "ended":
            return
        logger.info("Auto-hangup: conversation ended naturally for %s", cc_id)
        try:
            await hangup_call(cc_id)
        except Exception as e:
            logger.error("Auto-hangup failed: %s", e)
    except asyncio.CancelledError:
        pass  # Conversation continued — hangup cancelled
    finally:
        _auto_hangup_tasks.pop(cc_id, None)


async def _ai_assistant_watchdog(cc_id: str, greeting: str):
    """Watchdog: if no AI Assistant event within 8s, fall back to TTS pipeline."""
    await asyncio.sleep(8)
    if cc_id not in active_calls:
        return  # Call already ended
    if cc_id in _ai_assistant_first_event:
        return  # AI Assistant is working — all good
    rec = active_calls.get(cc_id, {})
    if not rec.get("ai_assistant"):
        return  # Already fell back

    logger.warning("WATCHDOG: AI Assistant silent for 8s on %s — falling back to TTS pipeline", cc_id)
    rec["ai_assistant"] = False

    # Try speaking the greeting via Polly + start transcription as TTS fallback
    try:
        speak_on_call(cc_id, greeting)
        speaking_calls.add(cc_id)
    except Exception as e:
        logger.error("Watchdog TTS fallback speak failed: %s", e)


async def _check_assistant_health():
    """Startup check: verify AI Assistant exists and ElevenLabs integration is reachable."""
    try:
        api_key = config.TELNYX_API_KEY
        if not api_key:
            logger.error("HEALTH: TELNYX_API_KEY not set!")
            return

        async with _httpx.AsyncClient(timeout=10.0) as ac:
            # Check assistant exists
            r = await ac.get(
                f"https://api.telnyx.com/v2/ai/assistants/{ASSISTANT_ID}",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            if r.status_code == 200:
                data = r.json().get("data", {})
                logger.info("HEALTH: Assistant OK — id=%s, model=%s", data.get("id"), data.get("model"))
                vs = data.get("voice_settings", {})
                if vs:
                    logger.info("HEALTH: Assistant voice_settings: %s", json.dumps(vs)[:200])
                else:
                    logger.warning("HEALTH: No voice_settings on assistant — voice may not work")
            else:
                logger.error("HEALTH: Assistant check failed HTTP %s: %s", r.status_code, r.text[:200])

        # Check ElevenLabs API key works
        el_key = config.ELEVENLABS_API_KEY
        if el_key:
            async with _httpx.AsyncClient(timeout=10.0) as ac:
                r = await ac.get(
                    "https://api.elevenlabs.io/v1/user",
                    headers={"xi-api-key": el_key},
                )
                if r.status_code == 200:
                    logger.info("HEALTH: ElevenLabs API key valid")
                else:
                    logger.error("HEALTH: ElevenLabs API key INVALID (HTTP %s)", r.status_code)
        else:
            logger.warning("HEALTH: ELEVENLABS_API_KEY not set — direct TTS fallback unavailable")

        logger.info("HEALTH: Integration secret ref = '%s' (must match Telnyx Mission Control secret)", config.ELEVENLABS_API_KEY_REF)
    except Exception as e:
        logger.warning("Health check error (non-fatal): %s", e)


@app.on_event("startup")
async def on_startup():
    """Sync AI Assistant + pre-cache filler audio + hot caches on every server start."""
    _get_tx()  # Pre-init Telnyx client — no cold start on first call
    _rebuild_hot_cache()
    try:
        sync_assistant_to_script()
    except Exception as e:
        logger.warning("Startup assistant sync failed (non-fatal): %s", e)
    await _check_assistant_health()
    await _precache_filler_audio()


async def _play_filler_for_ai_assistant(cc_id: str) -> None:
    """Play a pre-cached filler phrase instantly while AI Assistant thinks."""
    import random

    now = time.time()
    last = _last_filler_time.get(cc_id, 0)
    if now - last < 3.0:
        return
    if not _filler_audio_cache:
        return

    phrase = random.choice(list(_filler_audio_cache.keys()))
    b64 = _filler_audio_cache[phrase]

    try:
        _get_tx().calls.actions.start_playback(
            call_control_id=cc_id,
            playback_content=b64,
            audio_type="mp3",
        )
        _filler_playing[cc_id] = True
        _last_filler_time[cc_id] = now
        logger.info("AI-ASST filler: \"%s\" — INSTANT", phrase)
    except Exception as e:
        logger.debug("Filler playback failed (non-fatal): %s", e)


def _stop_filler_if_playing(cc_id: str) -> None:
    """Stop filler playback when AI Assistant starts speaking."""
    if _filler_playing.get(cc_id):
        try:
            _get_tx().calls.actions.stop_playback(call_control_id=cc_id)
        except Exception:
            pass
        _filler_playing[cc_id] = False


# ════════════════════════════════════════════════════════════
#  FRONTEND
# ════════════════════════════════════════════════════════════
@app.get("/")
async def serve_dashboard():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"status": "AI SDR running - dashboard not found"})


# ════════════════════════════════════════════════════════════
#  API — HEALTH & STATUS
# ════════════════════════════════════════════════════════════
@app.get("/api/health")
async def health():
    return {"status": "ok", "active_calls": len(active_calls), "base_url": config.APP_BASE_URL}


@app.get("/api/telnyx/diagnostics")
async def telnyx_diagnostics():
    return run_telnyx_diagnostics()


@app.get("/api/status")
async def api_status():
    config.reload_secrets()
    flags = config.dashboard_connection_flags()
    return {
        "telnyx":    flags["telnyx"],
        "deepgram":  flags["deepgram"],
        "anthropic": flags["anthropic"],
        "apollo":    flags["apollo"],
        "email":     flags["email"],
        "anthropic_model": config.ANTHROPIC_MODEL,
        "telnyx_phone":      config.TELNYX_PHONE_NUMBER or "",
        "telnyx_connection": config.TELNYX_CONNECTION_ID or "",
        "base_url":          config.APP_BASE_URL,
        "env_file":          str(config._ENV_FILE),
    }


@app.get("/api/test/assistant")
async def test_assistant():
    """Diagnostic: check AI Assistant config, ElevenLabs key, and integration secret."""
    results = {}
    config.reload_secrets()

    # 1. Check assistant exists
    try:
        api_key = config.TELNYX_API_KEY
        async with _httpx.AsyncClient(timeout=10.0) as ac:
            r = await ac.get(
                f"https://api.telnyx.com/v2/ai/assistants/{ASSISTANT_ID}",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            if r.status_code == 200:
                data = r.json().get("data", {})
                results["assistant"] = {
                    "ok": True,
                    "id": data.get("id"),
                    "model": data.get("model"),
                    "voice_settings": data.get("voice_settings"),
                }
            else:
                results["assistant"] = {"ok": False, "error": f"HTTP {r.status_code}: {r.text[:200]}"}
    except Exception as e:
        results["assistant"] = {"ok": False, "error": str(e)}

    # 2. Check ElevenLabs direct API key
    el_key = config.ELEVENLABS_API_KEY
    if el_key:
        try:
            async with _httpx.AsyncClient(timeout=10.0) as ac:
                r = await ac.get("https://api.elevenlabs.io/v1/user", headers={"xi-api-key": el_key})
                results["elevenlabs_api"] = {"ok": r.status_code == 200, "status": r.status_code}
        except Exception as e:
            results["elevenlabs_api"] = {"ok": False, "error": str(e)}
    else:
        results["elevenlabs_api"] = {"ok": False, "error": "ELEVENLABS_API_KEY not set"}

    # 3. Check integration secret ref
    results["integration_secret_ref"] = config.ELEVENLABS_API_KEY_REF or "(not set)"
    results["voice_id"] = config.ELEVENLABS_VOICE_ID or "(not set)"
    results["voice_string"] = f"ElevenLabs.{config.ELEVENLABS_MODEL_ID or 'eleven_multilingual_v2'}.{config.ELEVENLABS_VOICE_ID}"

    # 4. Check Telnyx integration secrets
    try:
        api_key = config.TELNYX_API_KEY
        async with _httpx.AsyncClient(timeout=10.0) as ac:
            r = await ac.get(
                "https://api.telnyx.com/v2/integration_secrets",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            if r.status_code == 200:
                secrets = r.json().get("data", [])
                secret_names = [s.get("identifier") or s.get("name") for s in secrets]
                ref = config.ELEVENLABS_API_KEY_REF
                found = ref in secret_names if ref else False
                results["telnyx_integration_secrets"] = {
                    "ok": found,
                    "available_secrets": secret_names,
                    "looking_for": ref,
                    "found": found,
                }
                if not found:
                    results["telnyx_integration_secrets"]["fix"] = (
                        f"Go to Telnyx Mission Control → Settings → Integration Secrets → "
                        f"Add new secret with identifier '{ref}' and paste your ElevenLabs API key as the value"
                    )
            else:
                results["telnyx_integration_secrets"] = {"ok": False, "error": f"HTTP {r.status_code}"}
    except Exception as e:
        results["telnyx_integration_secrets"] = {"ok": False, "error": str(e)}

    return results


@app.get("/api/test/apollo")
async def test_apollo():
    config.reload_secrets()
    if not config.APOLLO_API_KEY:
        return JSONResponse(
            {"ok": False, "error": "APOLLO_API_KEY is not set"},
            status_code=400,
        )
    try:
        data = await apollo_client.test_connection()
        return data
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=502)


@app.get("/api/test/anthropic")
async def test_anthropic():
    """Verify the key and model with a minimal API call (dashboard Test button)."""
    config.reload_secrets()
    if not config.ANTHROPIC_API_KEY:
        return JSONResponse(
            {"ok": False, "error": "ANTHROPIC_API_KEY is not set"},
            status_code=400,
        )
    try:
        client = AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
        await client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=1,
            messages=[{"role": "user", "content": "hi"}],
        )
        return {"ok": True, "model": config.ANTHROPIC_MODEL}
    except anthropic.APIStatusError as e:
        err_body = getattr(e, "body", None)
        msg = str(e)
        if isinstance(err_body, dict):
            inner = err_body.get("error") or {}
            if isinstance(inner, dict) and inner.get("message"):
                msg = inner["message"]
        code = getattr(e, "status_code", None)
        return JSONResponse(
            {"ok": False, "error": msg, "model": config.ANTHROPIC_MODEL, "http_status": code},
            status_code=502,
        )
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e), "model": config.ANTHROPIC_MODEL}, status_code=502)


@app.get("/api/settings")
async def get_settings():
    config.reload_secrets()
    return {
        "telnyx_key_set":    config.env_file_nonempty("TELNYX_API_KEY"),
        "anthropic_key_set": config.env_file_nonempty("ANTHROPIC_API_KEY"),
        "apollo_key_set":    config.env_file_nonempty("APOLLO_API_KEY"),
        "phone_number":      config.TELNYX_PHONE_NUMBER or "",
        "connection_id":     config.TELNYX_CONNECTION_ID or "",
        "base_url":          config.APP_BASE_URL or "",
        "anthropic_max_tokens_reply": config.ANTHROPIC_MAX_TOKENS_REPLY,
        "anthropic_live_model": config.ANTHROPIC_LIVE_MODEL or "",
        "anthropic_phone_model_effective": config.phone_reply_model(),
        "env_file":          str(config._ENV_FILE),
        "telnyx_speak_voice": config.TELNYX_SPEAK_VOICE or "",
        "elevenlabs_voice_id": config.ELEVENLABS_VOICE_ID or "",
        "elevenlabs_model_id": config.ELEVENLABS_MODEL_ID or "",
        "elevenlabs_api_key_ref": config.ELEVENLABS_API_KEY_REF or "",
        "elevenlabs_direct_first": bool(config.ELEVENLABS_DIRECT_FIRST),
        "elevenlabs_key_set": bool(config.ELEVENLABS_API_KEY),
        "tts_mode_summary": config.tts_mode_description(),
        "tts_voice_effective": config.telnyx_speak_voice_effective(),
        "smtp_host":               (config.SMTP_HOST or "").strip(),
        "smtp_port":               int(config.SMTP_PORT or 587),
        "smtp_user":               (config.SMTP_USER or "").strip(),
        "smtp_password_set":       config.env_file_nonempty("SMTP_PASSWORD"),
        "email_from":              (config.EMAIL_FROM or "").strip(),
        "smtp_use_tls":            bool(config.SMTP_USE_TLS),
        "email_automation_enabled": bool(config.EMAIL_AUTOMATION_ENABLED),
        "post_call_followup_email_enabled": bool(config.POST_CALL_FOLLOWUP_EMAIL_ENABLED),
        "post_call_followup_delay_sec": int(config.POST_CALL_FOLLOWUP_DELAY_SEC or 300),
        "email_provider":          (config.EMAIL_PROVIDER or "smtp").strip().lower(),
        "sendgrid_api_key_set":    config.env_file_nonempty("SENDGRID_API_KEY"),
        "resend_api_key_set":      config.env_file_nonempty("RESEND_API_KEY"),
        "mailgun_api_key_set":     config.env_file_nonempty("MAILGUN_API_KEY"),
        "mailgun_domain":          (config.MAILGUN_DOMAIN or "").strip(),
        "mailgun_api_base":        (config.MAILGUN_API_BASE or "").strip(),
        "smtp_ready":              smtp_ready(),
        "email_ready":             email_delivery_ready(),
        **oauth_connection_status(),
    }


@app.get("/api/tts-config")
async def api_tts_config():
    """Debug: which TTS path is active (no secrets). Open in browser after editing .env."""
    config.reload_secrets()
    direct = bool(
        config.ELEVENLABS_DIRECT_FIRST
        and config.ELEVENLABS_API_KEY
        and config.ELEVENLABS_VOICE_ID
    )
    return {
        "mode": config.tts_mode_description(),
        "voice_effective": config.telnyx_speak_voice_effective(),
        "elevenlabs_voice_id_configured": bool(config.ELEVENLABS_VOICE_ID),
        "telnyx_integration_secret_configured": bool(config.ELEVENLABS_API_KEY_REF),
        "direct_elevenlabs_active": direct,
        "note": "Custom voice uses ElevenLabs direct API when ELEVENLABS_API_KEY is set in .env. "
        "Paste your xi-api key from elevenlabs.io → Profile, restart server, then confirm direct_elevenlabs_active is true.",
    }


@app.post("/api/settings")
async def save_settings(request: Request):
    import re

    body = await request.json()
    env_path = Path(__file__).parent / ".env"
    env_text = env_path.read_text(encoding="utf-8") if env_path.exists() else ""

    def set_env(text: str, key: str, val: str) -> str:
        if re.search(rf"^{re.escape(key)}\s*=", text, re.MULTILINE):
            return re.sub(rf"^({re.escape(key)}\s*=).*", rf"\g<1>{val}", text, flags=re.MULTILINE)
        return text + f"\n{key}={val}"

    def patch_env_line(text: str, key: str, val: str) -> str:
        """Set key=val, or remove the line if val is empty."""
        line = re.compile(rf"^{re.escape(key)}\s*=.*$", re.MULTILINE)
        v = val.strip()
        if line.search(text):
            if v:
                return line.sub(f"{key}={v}", text)
            return line.sub("", text)
        if v:
            sep = "" if text.endswith("\n") else "\n"
            return text.rstrip() + sep + f"{key}={v}\n"
        return text

    mapping = {
        "telnyx_api_key":       "TELNYX_API_KEY",
        "anthropic_api_key":    "ANTHROPIC_API_KEY",
        "apollo_api_key":       "APOLLO_API_KEY",
        "telnyx_phone_number":  "TELNYX_PHONE_NUMBER",
        "telnyx_connection_id": "TELNYX_CONNECTION_ID",
        "app_base_url":         "APP_BASE_URL",
        "anthropic_max_tokens_reply": "ANTHROPIC_MAX_TOKENS_REPLY",
        "anthropic_live_model": "ANTHROPIC_LIVE_MODEL",
    }
    for field, env_key in mapping.items():
        val = body.get(field, "").strip()
        if val:
            env_text = set_env(env_text, env_key, val)
    # Dashboard sends phone_number / connection_id / base_url
    for ui_key, env_key in (
        ("phone_number", "TELNYX_PHONE_NUMBER"),
        ("connection_id", "TELNYX_CONNECTION_ID"),
        ("base_url", "APP_BASE_URL"),
    ):
        val = str(body.get(ui_key) or "").strip()
        if val:
            env_text = set_env(env_text, env_key, val)

    voice_map = {
        "telnyx_speak_voice": "TELNYX_SPEAK_VOICE",
        "elevenlabs_voice_id": "ELEVENLABS_VOICE_ID",
        "elevenlabs_model_id": "ELEVENLABS_MODEL_ID",
        "elevenlabs_api_key_ref": "ELEVENLABS_API_KEY_REF",
        "elevenlabs_api_key": "ELEVENLABS_API_KEY",
    }
    for field, env_key in voice_map.items():
        if field not in body:
            continue
        if field == "elevenlabs_api_key" and not str(body.get(field, "") or "").strip():
            # Omit empty: do not wipe existing key on every save when field left blank in UI.
            continue
        env_text = patch_env_line(env_text, env_key, str(body.get(field, "") or ""))
    if "elevenlabs_direct_first" in body:
        on = bool(body.get("elevenlabs_direct_first"))
        env_text = patch_env_line(env_text, "ELEVENLABS_DIRECT_FIRST", "1" if on else "0")

    smtp_fields = {
        "smtp_host": "SMTP_HOST",
        "smtp_port": "SMTP_PORT",
        "smtp_user": "SMTP_USER",
        "email_from": "EMAIL_FROM",
    }
    for field, env_key in smtp_fields.items():
        if field not in body:
            continue
        env_text = patch_env_line(env_text, env_key, str(body.get(field) or "").strip())
    if "smtp_password" in body and str(body.get("smtp_password") or "").strip():
        env_text = patch_env_line(env_text, "SMTP_PASSWORD", str(body.get("smtp_password") or "").strip())
    if "smtp_use_tls" in body:
        env_text = patch_env_line(
            env_text, "SMTP_USE_TLS", "1" if bool(body.get("smtp_use_tls")) else "0"
        )
    if "email_automation_enabled" in body:
        env_text = patch_env_line(
            env_text, "EMAIL_AUTOMATION_ENABLED", "1" if bool(body.get("email_automation_enabled")) else "0"
        )
    if "post_call_followup_email_enabled" in body:
        env_text = patch_env_line(
            env_text,
            "POST_CALL_FOLLOWUP_EMAIL_ENABLED",
            "1" if bool(body.get("post_call_followup_email_enabled")) else "0",
        )
    if "post_call_followup_delay_sec" in body:
        try:
            delay = max(60, int(str(body.get("post_call_followup_delay_sec") or "300").strip()))
        except ValueError:
            delay = 300
        env_text = patch_env_line(env_text, "POST_CALL_FOLLOWUP_DELAY_SEC", str(delay))

    if "email_provider" in body:
        ep = str(body.get("email_provider") or "smtp").strip().lower()
        if ep not in ("smtp", "sendgrid", "resend", "mailgun", "gmail_oauth", "outlook_oauth"):
            ep = "smtp"
        env_text = patch_env_line(env_text, "EMAIL_PROVIDER", ep)
    direct_fields = {
        "sendgrid_api_key": "SENDGRID_API_KEY",
        "resend_api_key": "RESEND_API_KEY",
        "mailgun_api_key": "MAILGUN_API_KEY",
        "mailgun_domain": "MAILGUN_DOMAIN",
        "mailgun_api_base": "MAILGUN_API_BASE",
    }
    for field, env_key in direct_fields.items():
        if field not in body:
            continue
        if field.endswith("_api_key") and not str(body.get(field) or "").strip():
            continue
        env_text = patch_env_line(env_text, env_key, str(body.get(field) or "").strip())

    oauth_env = {
        "google_oauth_client_id": "GOOGLE_OAUTH_CLIENT_ID",
        "google_oauth_client_secret": "GOOGLE_OAUTH_CLIENT_SECRET",
        "microsoft_oauth_client_id": "MICROSOFT_OAUTH_CLIENT_ID",
        "microsoft_oauth_client_secret": "MICROSOFT_OAUTH_CLIENT_SECRET",
        "microsoft_oauth_tenant": "MICROSOFT_OAUTH_TENANT",
    }
    for field, env_key in oauth_env.items():
        if field not in body:
            continue
        if field.endswith("_secret") and not str(body.get(field) or "").strip():
            continue
        env_text = patch_env_line(env_text, env_key, str(body.get(field) or "").strip())

    env_path.write_text(env_text, encoding="utf-8")
    config.reload_secrets()
    return {
        "status": "saved",
        "note": "Keys saved and reloaded — Test buttons use new values immediately.",
    }


@app.post("/api/settings/test-smtp")
async def settings_test_smtp():
    """Verify SMTP host + login (no email sent)."""
    return test_smtp_connection()


@app.post("/api/settings/test-email")
async def settings_test_email():
    """Verify active EMAIL_PROVIDER (SMTP handshake or HTTP API; no email sent)."""
    return test_email_delivery()


# ════════════════════════════════════════════════════════════
#  API — SCRIPT
# ════════════════════════════════════════════════════════════
@app.get("/api/script")
async def get_script():
    return load_script()


@app.post("/api/script")
async def post_script(request: Request):
    body = await request.json()
    save_script(body)
    _rebuild_hot_cache()
    sync_assistant_to_script()
    return {"status": "saved", "synced": True}


def _enrich_contact_with_calls(contact: dict[str, Any]) -> dict[str, Any]:
    """Merge matching rows from calls.json into call_history for the contact detail view."""
    import re

    c = dict(contact)
    ph = str(contact.get("phone") or "")
    norm = re.sub(r"\D", "", ph)
    if len(norm) < 10:
        return c
    tail = norm[-10:]
    merged: list[dict[str, Any]] = list(c.get("call_history") or [])
    seen: set[str] = set()
    for row in merged:
        key = f"{row.get('at')}|{row.get('summary')}"
        seen.add(key)
    try:
        for call in load_calls():
            to_raw = str(call.get("to") or "")
            to_d = re.sub(r"\D", "", to_raw)
            if len(to_d) < 10 or to_d[-10:] != tail:
                continue
            tr = call.get("transcript") or []
            snippet = ""
            if isinstance(tr, list) and tr:
                snippet = " ".join(
                    str(x.get("text") or "") for x in tr[-6:] if isinstance(x, dict)
                ).strip()[:280]
            entry = {
                "at": call.get("ended_at") or call.get("started_at") or "",
                "outcome": str(call.get("status") or "completed").replace("_", " ").title(),
                "summary": snippet or "Call recorded in AI SDR.",
                "source": "calls",
            }
            key = f"{entry['at']}|{entry['summary'][:40]}"
            if key not in seen:
                seen.add(key)
                merged.append(entry)
    except Exception:
        pass
    merged.sort(key=lambda x: str(x.get("at") or ""), reverse=True)
    c["call_history"] = merged[:30]
    return c


@app.get("/api/contacts")
async def api_contacts_list():
    rows = contacts_store.list_contacts()
    return {"contacts": rows, "total": len(rows)}


@app.post("/api/contacts")
async def api_contacts_create(request: Request):
    body = await request.json()
    return contacts_store.create_contact(body)


@app.get("/api/contacts/{contact_id}")
async def api_contacts_get(contact_id: str):
    c = contacts_store.get_contact(contact_id)
    if not c:
        raise HTTPException(status_code=404, detail="Contact not found")
    return _enrich_contact_with_calls(c)


@app.patch("/api/contacts/{contact_id}")
async def api_contacts_patch(contact_id: str, request: Request):
    body = await request.json()
    updated = contacts_store.update_contact(contact_id, body)
    if not updated:
        raise HTTPException(status_code=404, detail="Contact not found")
    return updated


@app.delete("/api/contacts/{contact_id}")
async def api_contacts_delete(contact_id: str):
    if not contacts_store.delete_contact(contact_id):
        raise HTTPException(status_code=404, detail="Contact not found")
    return {"status": "ok"}


@app.post("/api/contacts/import")
async def api_contacts_import(request: Request):
    body = await request.json()
    if isinstance(body, list):
        n = contacts_store.import_contacts_replace({"contacts": body})
    elif isinstance(body, dict) and isinstance(body.get("contacts"), list):
        n = contacts_store.import_contacts_replace(body)
    else:
        raise HTTPException(status_code=400, detail="Expected JSON array or {contacts: [...]}")
    return {"status": "ok", "imported": n}


@app.post("/api/script/suggest")
async def script_suggest(request: Request):
    body = await request.json()
    company = body.get("company_name", "")
    persona = body.get("target_persona", "")
    value = body.get("value_proposition", "")
    objective = body.get("call_objective", "")
    sdr_name = body.get("sdr_name", "Sarah")

    prompt = f"""You are helping an SDR set up their cold-calling script. Generate suggestions based on:
Company: {company}
Target persona: {persona}
Value proposition: {value}
Objective: {objective}
SDR name: {sdr_name}

Return JSON with these fields:
- discovery_questions: array of 5-7 short questions; prioritize CloudFuze Manage (SaaS visibility, licenses, shadow IT); include 1-2 about cloud migration only as follow-ups
- objections: object with keys: not_interested, send_email, call_back, have_solution, no_budget, manage_fine (each a short 1-sentence response). For manage_fine: prospect says they do not need Manage or app management is all good — response should pivot to CloudFuze Migrate (cloud-to-cloud migration) with one question
- booking_phrase: a natural way to ask for a meeting
- opening_line: a casual, permission-based opener using {{name}}, {{sdr_name}}, {{company}} placeholders

Return ONLY valid JSON, no markdown."""

    if not config.ANTHROPIC_API_KEY:
        raise HTTPException(status_code=400, detail="ANTHROPIC_API_KEY is not set")
    try:
        client = AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
        resp = await client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        suggestion = json.loads(text)
        return {"suggestion": suggestion}
    except Exception as e:
        logger.error("AI suggest failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


# ════════════════════════════════════════════════════════════
#  API — CALL HISTORY
# ════════════════════════════════════════════════════════════
@app.get("/api/calls/history")
async def call_history():
    calls = load_calls()
    out: list[dict[str, Any]] = []
    for c in calls:
        if isinstance(c, dict):
            row = dict(c)
            row["summary_preview"] = _summary_preview_for_history(row)
            out.append(row)
        else:
            out.append(c)  # type: ignore[arg-type]
    return {"total": len(out), "calls": out}


@app.post("/api/calls/{call_control_id}/recompute-insights")
async def recompute_call_insights(call_control_id: str):
    """Re-run Claude (or Telnyx fallback) on stored transcript + telnyx_insights. Use after upgrading insight logic."""
    rec = get_call_by_control_id(call_control_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Call not found")
    asyncio.create_task(_generate_call_insights(call_control_id))
    return {"status": "queued", "call_control_id": call_control_id}


@app.post("/api/calls/{call_control_id}/insights")
async def update_call_insights(call_control_id: str, request: Request):
    """
    Edit insights tags from the UI (short_tag/outcome/next_step).
    Does not affect call handling logic.
    """
    body = await request.json()
    rec = get_call_by_control_id(call_control_id) or active_calls.get(call_control_id)
    if not isinstance(rec, dict):
        raise HTTPException(status_code=404, detail="Call not found")
    ins = rec.get("insights") if isinstance(rec.get("insights"), dict) else {}
    ins = dict(ins or {})
    for k in ("short_tag", "outcome", "next_step"):
        if k in body:
            ins[k] = str(body.get(k) or "").strip()
    update_call(call_control_id, insights=ins)
    if call_control_id in active_calls:
        active_calls[call_control_id]["insights"] = ins
    return {"status": "saved", "insights": ins}


@app.post("/api/calls/cleanup-stale")
async def cleanup_stale_calls(payload: dict = Body(default={})):
    """Mark old 'initiated' calls as ended (fixes dashboard stuck on Pending). Optional body: {\"max_age_minutes\": 15}."""
    mins = float((payload or {}).get("max_age_minutes", 15))
    ids = mark_stale_initiated_calls(max_age_hours=max(0.05, mins) / 60.0)
    now_iso = datetime.utcnow().isoformat()
    for cid in ids:
        rec = active_calls.get(cid)
        if rec:
            rec["state"] = "ended"
            rec["ended_at"] = now_iso
            rec["ended_reason"] = "stale_no_webhook"
    return {"updated": len(ids), "call_control_ids": ids}


# ════════════════════════════════════════════════════════════
#  APOLLO + PROSPECTS + CAMPAIGN
# ════════════════════════════════════════════════════════════
class ApolloSearchBody(BaseModel):
    page: int = 1
    per_page: int = 25
    q_keywords: str = ""
    person_titles: list[str] = []
    person_locations: list[str] = []
    organization_locations: list[str] = []
    person_seniorities: list[str] = []
    organization_num_employees_ranges: list[str] = []
    q_organization_domains_list: list[str] = []
    include_similar_titles: bool = True


@app.post("/api/apollo/search")
async def apollo_search(body: ApolloSearchBody):
    try:
        data = await apollo_client.search_people(
            page=body.page,
            per_page=body.per_page,
            q_keywords=body.q_keywords or None,
            person_titles=body.person_titles or None,
            person_locations=body.person_locations or None,
            organization_locations=body.organization_locations or None,
            person_seniorities=body.person_seniorities or None,
            organization_num_employees_ranges=body.organization_num_employees_ranges or None,
            q_organization_domains_list=body.q_organization_domains_list or None,
            include_similar_titles=body.include_similar_titles,
        )
        people = data.get("people") or []
        rows = []
        for p in people:
            org = p.get("organization") or {}
            rows.append(
                {
                    "apollo_person_id": p.get("id") or "",
                    "first_name": p.get("first_name") or "",
                    "last_name": p.get("last_name") or "",
                    "title": p.get("title") or "",
                    "company": org.get("name") if isinstance(org, dict) else "",
                    "phone": "",
                    "notes": "From Apollo search (add phone via CSV or enrichment)",
                    "email": "",
                }
            )
        return {
            "people": rows,
            "pagination": data.get("pagination") or {},
            "total_entries": data.get("pagination", {}).get("total_entries")
            or data.get("total_entries"),
        }
    except Exception as e:
        logger.exception("Apollo search failed")
        raise HTTPException(status_code=502, detail=str(e)) from e


@app.post("/api/prospects/import-file")
async def prospects_import_file(file: UploadFile = File(...)):
    raw = await file.read()
    name = (file.filename or "").lower()
    if name.endswith(".csv"):
        rows, warnings = parse_csv_bytes(raw)
    elif name.endswith(".xlsx"):
        rows, warnings = parse_xlsx_bytes(raw)
    else:
        raise HTTPException(
            status_code=400,
            detail="Upload a .csv or .xlsx file (Excel .xls not supported).",
        )
    return {"rows": rows, "warnings": warnings, "count": len(rows)}


class CampaignStartBody(BaseModel):
    prospects: list[dict[str, Any]]
    spacing_seconds: float = Field(60.0, ge=0, le=86400)


@app.post("/api/campaign/start")
async def campaign_start(body: CampaignStartBody):
    if not body.prospects:
        raise HTTPException(status_code=400, detail="No prospects in queue")

    async def dial_one(p: dict[str, Any]) -> str | None:
        phone = normalize_phone(p.get("phone"))
        if not phone:
            return None
        name = prospect_display_name(p)
        parts: list[str] = []
        if p.get("title"):
            parts.append(f"Title: {p['title']}")
        if p.get("email"):
            parts.append(f"Email: {p['email']}")
        if p.get("apollo_person_id"):
            parts.append(f"Apollo: {p['apollo_person_id']}")
        if p.get("notes"):
            parts.append(p["notes"])
        notes = " | ".join(parts)
        req = CallRequest(
            to_number=phone,
            prospect_name=name,
            company=p.get("company") or "",
            notes=notes,
            prospect_email=str(p.get("email") or "").strip(),
        )
        result = await place_outbound_call(req)
        return result.get("call_control_id")

    ok = start_campaign(body.prospects, body.spacing_seconds, dial_one)
    if not ok:
        raise HTTPException(
            status_code=409,
            detail="Campaign already running or paused — stop it before starting a new one",
        )
    return {"status": "started", "total": len(body.prospects)}


@app.post("/api/campaign/pause")
async def campaign_pause():
    pause_campaign()
    return {"status": campaign_lib.state.status}


@app.post("/api/campaign/resume")
async def campaign_resume():
    resume_campaign()
    return {"status": campaign_lib.state.status}


@app.post("/api/campaign/stop")
async def campaign_stop():
    stop_campaign()
    return {"status": campaign_lib.state.status}


@app.get("/api/campaign/status")
async def campaign_status():
    st = campaign_lib.state
    return {
        "status": st.status,
        "index": st.index,
        "total": st.total,
        "spacing_seconds": st.spacing_seconds,
        "last_error": st.last_error,
        "last_to": st.last_to,
        "skipped": st.skipped,
    }


# ════════════════════════════════════════════════════════════
#  TASKS (dashboard parity with server.py)
# ════════════════════════════════════════════════════════════
@app.get("/api/tasks")
async def get_tasks():
    return {"tasks": load_tasks()}


@app.post("/api/tasks")
async def create_task(request: Request):
    body = await request.json()
    task = {
        "id": str(uuid.uuid4())[:8],
        "prospect_name": body.get("prospect_name", ""),
        "phone": body.get("phone", ""),
        "company": body.get("company", ""),
        "type": body.get("type", "callback"),
        "due_date": body.get("due_date", ""),
        "notes": body.get("notes", ""),
        "status": "pending",
        "call_control_id": body.get("call_control_id", ""),
        "created_at": datetime.utcnow().isoformat(),
    }
    save_task(task)
    return {"status": "created", "task": task}


@app.post("/api/tasks/{task_id}")
async def update_task_endpoint(task_id: str, request: Request):
    body = await request.json()
    update_task(task_id, **body)
    return {"status": "updated"}


@app.delete("/api/tasks/{task_id}")
async def delete_task_endpoint(task_id: str):
    delete_task(task_id)
    return {"status": "deleted"}


def check_callback_request(
    transcript: list,
    prospect_name: str,
    phone: str,
    company: str,
    cc_id: str,
) -> None:
    callback_phrases = [
        "call me back",
        "call back",
        "try me again",
        "call later",
        "not a good time",
        "busy right now",
        "call tomorrow",
        "call next week",
    ]
    for entry in transcript:
        if entry.get("role") == "prospect":
            text_lower = (entry.get("text") or "").lower()
            if any(phrase in text_lower for phrase in callback_phrases):
                task = {
                    "id": str(uuid.uuid4())[:8],
                    "prospect_name": prospect_name,
                    "phone": phone,
                    "company": company,
                    "type": "callback",
                    "due_date": "",
                    "notes": f'Prospect said: "{entry.get("text", "")}"',
                    "status": "pending",
                    "call_control_id": cc_id,
                    "created_at": datetime.utcnow().isoformat(),
                }
                save_task(task)
                logger.info("Auto-task created for callback: %s", prospect_name)
                return


def _ensure_task_for_meeting(call_control_id: str, rec: dict, insights: dict) -> None:
    """Create a single meeting task for this call if not present."""
    try:
        tasks = load_tasks()
        for t in tasks:
            if (t.get("call_control_id") or "") == call_control_id and (t.get("type") or "") == "meeting":
                return
        prospect_name = rec.get("prospect_name", "")
        phone = rec.get("to", "") or rec.get("phone", "")
        company = rec.get("company", "")
        due = str(insights.get("meeting_time") or insights.get("meeting_time_utc") or "").strip()
        notes = (insights.get("short_tag") or "").strip()
        summ = (insights.get("summary") or "").strip()
        if summ:
            notes = (notes + " — " if notes else "") + summ
        task = {
            "id": str(uuid.uuid4())[:8],
            "prospect_name": prospect_name,
            "phone": phone,
            "company": company,
            "type": "meeting",
            "due_date": due,
            "notes": notes or "Meeting booked on call.",
            "status": "pending",
            "call_control_id": call_control_id,
            "created_at": datetime.utcnow().isoformat(),
        }
        save_task(task)
        logger.info("Auto-task created for meeting: %s", prospect_name)
    except Exception:
        logger.exception("Failed to create meeting task for %s", call_control_id)


# ════════════════════════════════════════════════════════════
#  OUTBOUND CALL
# ════════════════════════════════════════════════════════════
class CallRequest(BaseModel):
    to_number:       str
    prospect_name:   str = "there"
    company:         str = ""
    notes:           str = ""
    prospect_email:  str = ""  # for automatic post-call recap email (SMTP + Claude)


async def place_outbound_call(req: CallRequest) -> dict:
    to = normalize_phone(req.to_number)
    if not to:
        raise HTTPException(
            status_code=400,
            detail="Invalid phone number — use E.164 (+15551234567) or 10-digit US/CA.",
        )
    if not config.TELNYX_PHONE_NUMBER:
        raise HTTPException(
            status_code=400,
            detail="TELNYX_PHONE_NUMBER is missing or invalid in .env (need +E.164).",
        )
    if not config.TELNYX_CONNECTION_ID:
        raise HTTPException(status_code=400, detail="TELNYX_CONNECTION_ID is not set in .env.")
    logger.info(f"Dialing {to} ({req.prospect_name})")
    # Fire research in background — don't wait for it before dialing
    if req.prospect_name or req.company:
        asyncio.create_task(research_prospect(req.prospect_name, "", req.company))
    result = await make_outbound_call(to)
    em = (req.prospect_email or "").strip()
    rec = {
        "call_control_id": result["call_control_id"],
        "call_leg_id":     result.get("call_leg_id"),
        "state":           "initiated",
        "to":              to,
        "prospect_name":   req.prospect_name,
        "company":         req.company,
        "notes":           req.notes,
        "prospect_email":  em,
        "transcript":      [],
        "started_at":      datetime.utcnow().isoformat(),
        "recording_url":   None,
    }
    active_calls[result["call_control_id"]] = rec
    save_call(rec)
    return result


@app.post("/call/outbound")
async def trigger_outbound_call(req: CallRequest):
    try:
        result = await place_outbound_call(req)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Outbound dial failed")
        raise HTTPException(status_code=502, detail=format_telnyx_exception(e)) from e


# ════════════════════════════════════════════════════════════
#  TELNYX WEBHOOK — handles all call events
# ════════════════════════════════════════════════════════════
async def _backup_start_transcription_after_opening(cc_id: str) -> None:
    """If call.playback/speak.ended is never delivered, STT never starts — user hears only the opener."""
    try:
        name = (active_calls.get(cc_id) or {}).get("prospect_name") or "there"
        delay = max(3.0, estimate_tts_playback_seconds(opening_line(name)) + 1.0)
        await asyncio.sleep(delay)
        if cc_id not in active_calls or active_calls[cc_id].get("state") == "ended":
            return
        await start_transcription(cc_id)
        logger.info(
            "Backup start_transcription after opening (cc_id=%s, delay_s=%.1f)",
            cc_id,
            delay,
        )
    except Exception:
        logger.exception("Backup start_transcription failed cc_id=%s", cc_id)


async def _main_bg_opening(cc_id: str) -> None:
    """Defer TTS until after Telnyx webhook HTTP 200 (avoids dropped audio / timeouts)."""
    try:
        await asyncio.sleep(0.05)
        name = (active_calls.get(cc_id) or {}).get("prospect_name") or "there"
        await speak_on_call(cc_id, opening_line(name))
        await start_recording(cc_id)
        asyncio.create_task(_backup_start_transcription_after_opening(cc_id))
    except Exception:
        logger.exception("Opening TTS failed cc_id=%s", cc_id)
        speaking_calls.discard(cc_id)


async def _resume_listen_fallback(cc_id: str, reply: str) -> None:
    """If call.speak.ended / playback.ended never arrive, still clear guard and restart STT."""
    try:
        await asyncio.sleep(estimate_tts_playback_seconds(reply))
    finally:
        speaking_calls.discard(cc_id)
        try:
            if cc_id in active_calls and active_calls[cc_id].get("state") != "ended":
                await start_transcription(cc_id)
        except Exception:
            logger.exception("listen fallback start_transcription failed cc_id=%s", cc_id)


async def _main_bg_transcription_reply(cc_id: str, text: str) -> None:
    reply = ""
    try:
        await stop_transcription(cc_id)
        rec = active_calls.get(cc_id)
        if rec:
            rec.setdefault("transcript", []).append({"role": "prospect", "text": text})
        conv = conversations.setdefault(cc_id, [])
        conv.append({"role": "user", "content": text})
        logger.info(f"Claude thinking... ({len(conv)} turns)")
        t0 = time.monotonic()

        # Q/A KB shortcut: if this is a repeated/related question, reuse the prior best answer.
        if config.QA_KB_ENABLED:
            kb_ans, kb_score = qa_kb.answer_for(text, min_score=config.QA_KB_MIN_SCORE)
            if kb_ans:
                reply = kb_ans
                conv.append({"role": "assistant", "content": reply})
                logger.info("KB HIT score=%.2f reply=%r", kb_score, reply[:120])
                t1 = time.monotonic()
                await speak_on_call(cc_id, reply)
                logger.info("LATENCY speak_api_ms=%.0f (Telnyx HTTP)", (time.monotonic() - t1) * 1000)
                if rec:
                    rec["transcript"].append({"role": "agent", "text": reply})
                    save_call(rec)
                qa_kb.add_qa(text, reply, call_control_id=cc_id, source="kb_hit_reply")
                return

        if config.STREAM_SPEECH_PIPELINE:
            parts: list[str] = []
            if config.should_play_think_filler(text):
                await speak_on_call(cc_id, config.phone_think_filler_phrase())
            async for sent in stream_sdr_reply_sentences(conv):
                parts.append(sent)
                await speak_on_call(cc_id, sent)
            reply = join_streamed_reply_parts(parts)
            if not reply:
                reply = "Sorry — bad connection on my end. Can I call you back tomorrow?"
            conv.append({"role": "assistant", "content": reply})
            if not parts:
                await speak_on_call(cc_id, reply)
        elif config.should_play_think_filler(text):
            llm_task = asyncio.create_task(next_sdr_reply(conv))
            await speak_on_call(cc_id, config.phone_think_filler_phrase())
            reply = await llm_task
            conv.append({"role": "assistant", "content": reply})
            t1 = time.monotonic()
            await speak_on_call(cc_id, reply)
            logger.info("LATENCY speak_api_ms=%.0f (Telnyx HTTP)", (time.monotonic() - t1) * 1000)
        else:
            reply = await next_sdr_reply(conv)
            conv.append({"role": "assistant", "content": reply})
            t1 = time.monotonic()
            await speak_on_call(cc_id, reply)
            logger.info("LATENCY speak_api_ms=%.0f (Telnyx HTTP)", (time.monotonic() - t1) * 1000)

        t_llm_ms = (time.monotonic() - t0) * 1000
        logger.info(f"AI says: \"{reply}\"")
        logger.info(
            "LATENCY turn_ms=%.0f model=%s max_tokens=%s stream=%s",
            t_llm_ms,
            config.phone_reply_model(),
            config.ANTHROPIC_MAX_TOKENS_REPLY,
            config.STREAM_SPEECH_PIPELINE,
        )
        if rec:
            rec["transcript"].append({"role": "agent", "text": reply})
            # Persist transcript to disk immediately (not just on hangup)
            save_call(rec)
        if config.QA_KB_ENABLED and reply:
            qa_kb.add_qa(text, reply, call_control_id=cc_id, source="call_turn")
    except Exception as e:
        logger.error(f"Claude/speak error: {e}")
    finally:
        asyncio.create_task(_resume_listen_fallback(cc_id, reply))


async def _start_ai_assistant_fast(cc_id: str, name: str, title: str, company: str, background_tasks: BackgroundTasks):
    """Start AI Assistant off the main webhook thread — zero blocking on call.answered.
    Uses pre-cached values + Telnyx SDK for reliable start."""
    loop = asyncio.get_event_loop()

    # ── Build greeting from CACHED script (no disk read) ──
    s = _cached_script or load_script()
    sdr = s.get("sdr_name", "Alex")
    co = s.get("company_name", "CloudFuze")
    greeting = f"Hey {name}, this is {sdr} from {co} -- did I catch you at a bad time?"

    # ── Build message_history from CACHED knowledge (no recompute) ──
    msg_history = list(_cached_knowledge_history)  # shallow copy
    research = get_cached_research(name, company)
    if research:
        msg_history.append({"role": "user", "content": f"[BRIEFING]\n{research[:500]}"})
        msg_history.append({"role": "assistant", "content": "Got it."})

    # ── Build kwargs using CACHED voice settings (no config lookups) ──
    ai_kwargs: dict[str, Any] = {
        "call_control_id": cc_id,
        "assistant": {"id": ASSISTANT_ID},
        "greeting": greeting,
        "transcription": {"model": "distil-whisper/distil-large-v2"},
        "interruption_settings": {"enable": True},
    }
    ai_kwargs.update(_cached_voice_kwargs)
    if msg_history:
        ai_kwargs["message_history"] = msg_history

    tx = _get_tx()

    # ── Fire recording detached (don't block greeting) ──
    async def _fire_recording_detached():
        try:
            await loop.run_in_executor(None, lambda: tx.calls.actions.start_recording(
                call_control_id=cc_id, format="mp3", channels="single"))
        except Exception as e:
            logger.warning("Recording start failed (non-fatal): %s", e)

    try:
        t0 = time.monotonic()
        await loop.run_in_executor(None, lambda: tx.calls.actions.start_ai_assistant(**ai_kwargs))
        latency_ms = (time.monotonic() - t0) * 1000
        active_calls.setdefault(cc_id, {})["ai_assistant"] = True
        logger.info("AI Assistant started in %.0fms — greeting: %s", latency_ms, greeting[:60])
        # Recording in background — don't block greeting
        asyncio.create_task(_fire_recording_detached())
        asyncio.create_task(_ai_assistant_watchdog(cc_id, greeting))
    except Exception as e:
        logger.exception("AI Assistant failed: %s — falling back to TTS", e)
        active_calls.setdefault(cc_id, {})["ai_assistant"] = False
        speaking_calls.add(cc_id)
        background_tasks.add_task(_main_bg_opening, cc_id)


@app.post("/webhooks/telnyx")
async def telnyx_webhook(request: Request, background_tasks: BackgroundTasks):
    raw = await request.body()
    try:
        body = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        logger.error(
            "WEBHOOK non-JSON len=%s ct=%s preview=%r",
            len(raw),
            request.headers.get("content-type"),
            raw[:1200],
        )
        return JSONResponse(content={"status": "ignored", "reason": "invalid_json"}, status_code=200)
    try:
        event = parse_webhook_event(body)
        raw_type = (body.get("data") or {}).get("event_type") or event.get("event_type") or "unknown"
        etype = normalize_telnyx_event_type(raw_type)
        # Prefer full-body resolution — payload-only parse misses some Telnyx shapes (hangup, etc.)
        cc_id = extract_call_control_id_from_body(body) or (event.get("call_control_id") or "").strip() or None

        logger.info(f"WEBHOOK: {etype} | {cc_id}")

        pl = (body.get("data") or {}).get("payload") or {}

        # ── INBOUND CALL ──────────────────────────────
        if etype == "call.initiated":
            if (pl.get("direction") or event.get("direction")) == "incoming":
                await answer_call(cc_id)
                rec = {"call_control_id": cc_id, "state": "answered", "transcript": [],
                       "direction": "inbound", "started_at": datetime.utcnow().isoformat()}
                active_calls[cc_id] = rec
                save_call(rec)

        # ── CALL ANSWERED → Start Telnyx AI Assistant (speech-to-speech) ──
        elif etype == "call.answered":
            if not cc_id:
                logger.error("call.answered missing call_control_id")
                return JSONResponse(content={"status": "ok"})
            if cc_id in opened_calls:
                logger.info(f"SKIP duplicate call.answered: {cc_id}")
                return JSONResponse(content={"status": "ok"})
            opened_calls.add(cc_id)

            if cc_id in active_calls:
                active_calls[cc_id]["state"] = "answered"
                # Defer disk write — don't block the greeting
                background_tasks.add_task(update_call, cc_id, state="answered")

            rec = active_calls.get(cc_id) or {}
            name = rec.get("prospect_name", "there")
            title = rec.get("title", "") or ""
            company = rec.get("company", "") or ""
            conversations[cc_id] = []

            # Fire AI Assistant start in background — return 200 to Telnyx ASAP
            asyncio.create_task(_start_ai_assistant_fast(cc_id, name, title, company, background_tasks))

        # ── SPEAK/PLAYBACK ENDED → start transcription (only for TTS fallback) ──
        elif etype in ("call.speak.ended", "call.playback.ended"):
            logger.info(f"Speak/playback done: {cc_id}")
            speaking_calls.discard(cc_id)
            # Auto-hangup after voicemail message
            if active_calls.get(cc_id, {}).get("voicemail"):
                logger.info("Voicemail message delivered — auto-hanging up %s", cc_id)
                await asyncio.sleep(0.5)  # Small pause after message
                try:
                    await hangup_call(cc_id)
                except Exception as e:
                    logger.error("Voicemail hangup failed: %s", e)
            # If AI Assistant is active, it handles everything — skip
            elif active_calls.get(cc_id, {}).get("ai_assistant"):
                pass
            elif cc_id in active_calls and active_calls.get(cc_id, {}).get("state") != "ended":
                try:
                    await start_transcription(cc_id)
                except Exception as e:
                    logger.error(f"Transcription start failed: {e}")

        # ── AI ASSISTANT TRANSCRIPTION → capture for call history ──
        elif etype in ("call.ai_assistant.transcription", "call.ai_assistant.partial_transcription"):
            # Mark that AI Assistant is alive (disarm watchdog)
            if cc_id and cc_id not in _ai_assistant_first_event:
                _ai_assistant_first_event[cc_id] = time.time()
                logger.info("AI Assistant alive — first event for %s", cc_id)
            ai_text = (pl.get("text") or pl.get("transcript") or "").strip()
            ai_role = pl.get("role", "")  # "user" or "assistant"
            if ai_text and cc_id:
                rec = active_calls.get(cc_id)
                if rec:
                    tlist = rec.setdefault("transcript", [])
                    role = "prospect" if ai_role == "user" else "agent"
                    tlist.append({"role": role, "text": ai_text})
                    # Persist transcript immediately
                    save_call(rec)
                    if ai_role == "user":
                        logger.info(f"AI-ASST HEARD: \"{ai_text}\"")
                        if config.PHONE_THINK_FILLER and config.should_play_think_filler(ai_text):
                            asyncio.create_task(_play_filler_for_ai_assistant(cc_id))
                        # If prospect says goodbye, don't cancel — let auto-hangup proceed
                        if _is_goodbye(ai_text):
                            logger.info("Prospect said goodbye — keeping auto-hangup active")
                        elif cc_id in _auto_hangup_tasks:
                            # Prospect said something that's NOT goodbye — cancel pending hangup
                            _auto_hangup_tasks[cc_id].cancel()
                            _auto_hangup_tasks.pop(cc_id, None)
                    else:
                        logger.info(f"AI-ASST SAID: \"{ai_text}\"")
                        _stop_filler_if_playing(cc_id)
                        # Detect goodbye from AI → schedule auto-hangup
                        if _is_goodbye(ai_text):
                            logger.info("Goodbye detected in AI response — scheduling auto-hangup in 4s")
                            if cc_id in _auto_hangup_tasks:
                                _auto_hangup_tasks[cc_id].cancel()
                            _auto_hangup_tasks[cc_id] = asyncio.create_task(_auto_hangup_after_goodbye(cc_id))

        # ── AI ASSISTANT SPEAKING → stop filler ──
        elif etype in ("call.ai_assistant.speaking_started", "call.ai_assistant.response_started"):
            if cc_id:
                # Mark alive (disarm watchdog)
                if cc_id not in _ai_assistant_first_event:
                    _ai_assistant_first_event[cc_id] = time.time()
                    logger.info("AI Assistant alive (speaking) — first event for %s", cc_id)
                _stop_filler_if_playing(cc_id)
                logger.info("AI Assistant speaking — filler stopped")

        # ── AI ASSISTANT ERROR → fall back to TTS pipeline ──
        elif etype == "call.ai_assistant.error":
            logger.error("AI Assistant error: %s", pl)
            if cc_id and cc_id in active_calls:
                active_calls[cc_id]["ai_assistant"] = False
                logger.info("Falling back to TTS pipeline for %s", cc_id)
                speaking_calls.add(cc_id)
                background_tasks.add_task(_main_bg_opening, cc_id)

        # ── TRANSCRIPTION → prospect speaking (TTS fallback only) ──
        elif etype == "call.transcription":
            # Only used when AI Assistant is NOT active (fallback mode)
            if active_calls.get(cc_id, {}).get("ai_assistant"):
                return JSONResponse(content={"status": "ok"})
            text, is_final, cc_resolved = parse_call_transcription_event(body)
            cc_id = cc_resolved or cc_id
            if not cc_id or not text:
                return JSONResponse(content={"status": "ok"})

            logger.info(f"HEARD ({'final' if is_final else 'interim'}): \"{text}\"")

            if not should_emit_transcription_reply(cc_id, text, is_final):
                return JSONResponse(content={"status": "ok"})
            if cc_id in speaking_calls:
                return JSONResponse(content={"status": "ok"})

            speaking_calls.add(cc_id)
            asyncio.create_task(_main_bg_transcription_reply(cc_id, text))
            return JSONResponse(content={"status": "ok"})

        # ── CONVERSATION ENDED → auto-hangup + generate AI insights ──
        elif etype == "call.conversation.ended":
            if cc_id:
                logger.info("Conversation ended for %s — auto-hanging up + generating insights", cc_id)
                asyncio.create_task(_generate_call_insights(cc_id))
                # Auto-hangup: Telnyx says conversation is done
                try:
                    await hangup_call(cc_id)
                    logger.info("Auto-hangup after conversation.ended for %s", cc_id)
                except Exception as e:
                    logger.warning("Auto-hangup on conversation.ended failed (may already be hung up): %s", e)

        # ── TELNYX CONVERSATION INSIGHTS ──
        elif etype == "call.conversation_insights.generated":
            insights_data = pl.get("insights") or pl
            if cc_id:
                logger.info("Conversation insights received for %s", cc_id)
                rec = active_calls.get(cc_id)
                if rec:
                    rec["telnyx_insights"] = insights_data
                update_call(cc_id, telnyx_insights=insights_data)
                # Re-run Claude insights using Telnyx summary when local transcript is empty.
                asyncio.create_task(_generate_call_insights(cc_id))

        # ── RECORDING SAVED ────────────────────────────
        elif etype == "call.recording.saved":
            url = event["raw"].get("recording_urls", {}).get("mp3") or \
                  event["raw"].get("public_recording_urls", {}).get("mp3")
            if url:
                logger.info(f"Recording saved: {url}")
                if cc_id in active_calls:
                    active_calls[cc_id]["recording_url"] = url
                update_call(cc_id, recording_url=url)

        # ── CALL ENDED ─────────────────────────────────
        elif etype == "call.hangup":
            hang_cc = extract_call_control_id_from_body(body) or cc_id
            if not hang_cc:
                logger.error(
                    "call.hangup: missing call_control_id — cannot persist end state. data.payload keys=%s",
                    list((body.get("data") or {}).get("payload") or {}),
                )
                return JSONResponse(content={"status": "ok"})
            signal_call_ended(hang_cc)
            ended_at = datetime.utcnow().isoformat()
            rec = active_calls.get(hang_cc)
            duration_seconds = None
            transcript: list = []
            if rec and rec.get("state") == "ended":
                return JSONResponse(content={"status": "ok"})
            if rec:
                rec["state"] = "ended"
                rec["ended_at"] = ended_at
                try:
                    started = datetime.fromisoformat(rec.get("started_at", ended_at))
                    duration_seconds = int((datetime.utcnow() - started).total_seconds())
                    rec["duration_seconds"] = duration_seconds
                except Exception:
                    duration_seconds = rec.get("duration_seconds")
                transcript = rec.get("transcript", []) or []
                active_calls[hang_cc] = rec
                check_callback_request(
                    transcript,
                    rec.get("prospect_name", ""),
                    rec.get("to", ""),
                    rec.get("company", ""),
                    hang_cc,
                )
            if not finalize_call_end(
                hang_cc,
                state="ended",
                ended_at=ended_at,
                duration_seconds=duration_seconds if duration_seconds is not None else 0,
                transcript=transcript,
            ):
                logger.warning(
                    "call.hangup: no calls.json row for %s — check dial save_call / call_control_id",
                    hang_cc,
                )
            if rec and (rec.get("prospect_email") or "").strip():
                update_call(hang_cc, prospect_email=(rec.get("prospect_email") or "").strip())
            asyncio.create_task(_generate_call_insights(hang_cc))
            merged_row = dict(get_call_by_control_id(hang_cc) or {})
            if rec:
                merged_row.update(rec)
            if config.POST_CALL_FOLLOWUP_EMAIL_ENABLED and resolve_prospect_email(merged_row):
                asyncio.create_task(run_post_call_followup_email(hang_cc))
            asyncio.create_task(_remove_ended_call_after(hang_cc))
            conversations.pop(hang_cc, None)
            _ai_assistant_first_event.pop(hang_cc, None)
            task = _auto_hangup_tasks.pop(hang_cc, None)
            if task:
                task.cancel()
            speaking_calls.discard(hang_cc)
            opened_calls.discard(hang_cc)
            _filler_playing.pop(hang_cc, None)
            _last_filler_time.pop(hang_cc, None)

            # Q/A KB: also ingest pairs from the full transcript at call end (best-effort).
            if config.QA_KB_ENABLED:
                try:
                    trec = get_call_by_control_id(hang_cc) or rec or {}
                    tlist = (trec.get("transcript") or []) if isinstance(trec, dict) else []
                    if isinstance(tlist, list) and tlist:
                        pending_q: str | None = None
                        for turn in tlist:
                            if not isinstance(turn, dict):
                                continue
                            role = (turn.get("role") or "").lower()
                            txt = (turn.get("text") or "").strip()
                            if not txt:
                                continue
                            if role in ("prospect", "user"):
                                pending_q = txt
                            elif role in ("agent", "assistant") and pending_q:
                                qa_kb.add_qa(
                                    pending_q,
                                    txt,
                                    call_control_id=hang_cc,
                                    source="call_end_transcript",
                                )
                                pending_q = None
                except Exception:
                    logger.exception("QA KB transcript ingest failed for %s", hang_cc)

        elif etype == "call.machine.detection.ended":
            result_val = (pl.get("result") or event["raw"].get("result") or "").lower()
            logger.info("AMD result for %s: %s", cc_id, result_val)
            if "machine" in result_val or "beep" in result_val:
                # Voicemail detected — stop AI Assistant if running, leave voicemail pitch, hangup
                rec = active_calls.get(cc_id) or {}

                # Stop AI Assistant if it was started (AMD can arrive after call.answered)
                if rec.get("ai_assistant"):
                    try:
                        await asyncio.get_event_loop().run_in_executor(
                            None, lambda: _get_tx().calls.actions.stop_ai_assistant(call_control_id=cc_id)
                        )
                        logger.info("Stopped AI Assistant for voicemail on %s", cc_id)
                    except Exception as e:
                        logger.warning("Stop AI Assistant for voicemail failed (non-fatal): %s", e)

                name = rec.get("prospect_name") or "there"
                s = load_script()
                sdr = s.get("sdr_name", "Alex")
                co = s.get("company_name", "our company")
                vm_tmpl = s.get("voicemail_message") or f"Hey {{name}}, this is {{sdr_name}} from {{company}}. I was reaching out because we help companies get full visibility into their SaaS and AI app stack. I'd love to set up a quick 15-minute call. I'll try you again soon — have a great day!"
                vm_msg = vm_tmpl.replace("{name}", name).replace("{sdr_name}", sdr).replace("{company}", co)
                logger.info("Voicemail detected — leaving pitch (%d chars): %s...", len(vm_msg), vm_msg[:100])
                try:
                    await speak_on_call(cc_id, vm_msg)
                    # Mark as voicemail so we auto-hangup after speak ends
                    active_calls.setdefault(cc_id, {})["voicemail"] = True
                    active_calls.setdefault(cc_id, {})["ai_assistant"] = False
                    rec["transcript"] = [{"role": "agent", "text": f"[VOICEMAIL] {vm_msg}"}]
                    update_call(cc_id, outcome="voicemail")
                    save_call(rec)
                except Exception as e:
                    logger.error("Voicemail speak failed: %s — hanging up", e)
                    await hangup_call(cc_id)
            else:
                logger.info("AMD: human detected for %s — AI Assistant continues", cc_id)

        return JSONResponse(content={"status": "ok"})

    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return JSONResponse(content={"status": "error", "detail": str(e)}, status_code=500)


# ════════════════════════════════════════════════════════════
#  MISSING UTILITY FUNCTIONS
# ════════════════════════════════════════════════════════════
def _telnyx_conversation_summary(telnyx: Any) -> str:
    """Extract human-readable summary text from Telnyx conversation_insights payload."""
    if not isinstance(telnyx, dict):
        return ""
    results = telnyx.get("results")
    if not isinstance(results, list):
        return ""
    parts: list[str] = []
    for r in results:
        if isinstance(r, dict):
            t = r.get("result") or r.get("text") or ""
            t = str(t).strip()
            if t:
                parts.append(t)
    return " ".join(parts).strip()


def _insights_summary_is_placeholder(ins: Any) -> bool:
    if not isinstance(ins, dict):
        return True
    s = (ins.get("summary") or "").strip().lower()
    if not s:
        return True
    return (
        "too short for meaningful" in s
        or "no conversation text captured" in s
    )


def _summary_preview_for_history(rec: dict[str, Any]) -> str:
    """One line for call history: prefer real insights, else Telnyx narrative."""
    ins = rec.get("insights") if isinstance(rec.get("insights"), dict) else {}
    s = (ins.get("summary") or "").strip()
    if s and not _insights_summary_is_placeholder(ins):
        return s[:200] + ("…" if len(s) > 200 else "")
    tx = _telnyx_conversation_summary(rec.get("telnyx_insights"))
    if tx:
        return tx[:200] + ("…" if len(tx) > 200 else "")
    return s[:200] if s else ""


def _merge_rec_for_insights(cc_id: str) -> dict[str, Any] | None:
    """Prefer on-disk row + freshest transcript from active_calls."""
    rec = get_call_by_control_id(cc_id)
    if not rec:
        rec = active_calls.get(cc_id)
    if not rec:
        return None
    rec = dict(rec)
    ac = active_calls.get(cc_id)
    if ac:
        t_disk = rec.get("transcript") or []
        t_mem = ac.get("transcript") or []
        if len(t_mem) > len(t_disk):
            rec["transcript"] = t_mem
        if not rec.get("telnyx_insights") and ac.get("telnyx_insights"):
            rec["telnyx_insights"] = ac["telnyx_insights"]
    return rec


async def _remove_ended_call_after(cc_id: str, delay: float = 180.0) -> None:
    """Remove ended call from active_calls dict after a delay (keeps it visible briefly)."""
    try:
        await asyncio.sleep(delay)
        rec = active_calls.get(cc_id)
        if rec and rec.get("state") == "ended":
            active_calls.pop(cc_id, None)
    except Exception:
        logger.exception("_remove_ended_call_after failed cc_id=%s", cc_id)


async def _generate_call_insights(cc_id: str) -> None:
    """After call ends, use Claude to analyze transcript (or Telnyx conversation summary) and generate insights."""
    try:
        rec = _merge_rec_for_insights(cc_id)
        if not rec:
            return

        transcript = rec.get("transcript", []) or []
        telnyx_txt = _telnyx_conversation_summary(rec.get("telnyx_insights"))

        transcript_text = ""
        if len(transcript) >= 2:
            transcript_text = "\n".join(
                f"{'AI SDR' if t.get('role') == 'agent' else 'Prospect'}: {t.get('text', '')}"
                for t in transcript
                if isinstance(t, dict)
            )
        elif telnyx_txt:
            transcript_text = (
                "[SOURCE: Telnyx AI conversation summary — local transcript was empty]\n" + telnyx_txt
            )
        else:
            insights = {
                "summary": "No conversation text captured yet. If you use Telnyx AI Assistant, insights appear when Telnyx sends conversation insights.",
                "short_tag": "No transcript",
                "outcome": "no_conversation",
                "sentiment": "neutral",
                "action_items": [],
                "key_points": [],
                "objections": [],
                "next_step": "Check webhook / transcript settings",
                "interest_level": 0,
            }
            update_call(cc_id, insights=insights)
            if cc_id in active_calls:
                active_calls[cc_id]["insights"] = insights
            return

        if not config.ANTHROPIC_API_KEY:
            insights = {
                "summary": (telnyx_txt or transcript_text)[:1200],
                "short_tag": "Summary (Telnyx)",
                "outcome": "interested",
                "sentiment": "neutral",
                "action_items": [],
                "key_points": [],
                "objections": [],
                "next_step": "Review recording",
                "interest_level": 50,
            }
            update_call(cc_id, insights=insights)
            if cc_id in active_calls:
                active_calls[cc_id]["insights"] = insights
            return

        prompt = f"""Analyze this sales call. Return ONLY valid JSON (no markdown, no code blocks).

CONVERSATION (transcript and/or platform summary):
{transcript_text}

Return this exact JSON structure:
{{"short_tag": "2-3 words describing what happened (e.g. 'Meeting booked', 'Not interested', 'Call back', 'No answer')",
"summary": "2-3 sentence summary of the call",
"outcome": "one of: meeting_booked, callback_scheduled, interested, not_interested, no_answer, voicemail, gatekeeper, hung_up",
"sentiment": "one of: very_positive, positive, neutral, negative, very_negative",
"interest_level": 0-100,
"action_items": ["list of follow-up actions needed"],
"key_points": ["key things discussed or learned"],
"objections": ["any objections the prospect raised"],
"next_step": "recommended next action",
"meeting_time_utc": "ISO8601 UTC timestamp if meeting was booked, else empty string",
"prospect_pain_points": ["pain points mentioned by prospect"],
"buying_signals": ["any positive buying signals detected"]}}"""

        client = AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
        response = await client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

        insights = json.loads(raw)
        if isinstance(insights, dict):
            # Normalize missing fields for UI and downstream logic.
            insights.setdefault("short_tag", "")
            insights.setdefault("meeting_time_utc", "")
        update_call(cc_id, insights=insights)
        if cc_id in active_calls:
            active_calls[cc_id]["insights"] = insights
        logger.info("Call insights generated for %s: outcome=%s, interest=%s%%",
                     cc_id[:20], insights.get("outcome"), insights.get("interest_level"))
        if isinstance(insights, dict) and insights.get("outcome") == "meeting_booked":
            _ensure_task_for_meeting(cc_id, rec if isinstance(rec, dict) else {}, insights)
    except Exception as e:
        logger.exception("Failed to generate call insights for %s: %s", cc_id, e)


# ════════════════════════════════════════════════════════════
#  PROSPECT RESEARCH (pre-call)
# ════════════════════════════════════════════════════════════
_prospect_research_cache: dict[str, str] = {}


def _research_key(name: str, company: str) -> str:
    return f"{(name or '').strip().lower()}|{(company or '').strip().lower()}"


async def research_prospect(name: str, title: str, company: str) -> str:
    """Research prospect via Apollo. Cache result for instant use during call."""
    key = _research_key(name, company)
    if key in _prospect_research_cache:
        return _prospect_research_cache[key]

    if not company and not name:
        return ""

    try:
        results = await apollo_client.search_people(
            person_titles=[title] if title else [],
            q_keywords=company or name,
            per_page=1,
        )
        people = results.get("people", [])
        org = None
        person = None
        if people:
            person = people[0]
            org = person.get("organization")

        parts = [f"PROSPECT INFO:"]
        parts.append(f"- {name}, {title} at {company}")

        if person:
            if person.get("headline"):
                parts.append(f"- Headline: {person['headline'][:150]}")

        if org:
            if org.get("short_description"):
                parts.append(f"- Company: {org['short_description'][:200]}")
            if org.get("estimated_num_employees"):
                parts.append(f"- Size: ~{org['estimated_num_employees']} employees")
            if org.get("industry"):
                parts.append(f"- Industry: {org['industry']}")
            if org.get("keywords") and isinstance(org["keywords"], list):
                parts.append(f"- Tech: {', '.join(org['keywords'][:8])}")
            if org.get("annual_revenue_printed"):
                parts.append(f"- Revenue: {org['annual_revenue_printed']}")
            if org.get("website_url"):
                parts.append(f"- Website: {org['website_url']}")
        else:
            parts.append(f"- No detailed info found. Ask about their team and tools.")

        research = "\n".join(parts)
        _prospect_research_cache[key] = research
        logger.info("Researched %s @ %s: %d chars", name, company, len(research))
        return research
    except Exception as e:
        logger.warning("Research failed for %s @ %s: %s", name, company, e)
        fallback = f"PROSPECT INFO:\n- {name}, {title} at {company}"
        _prospect_research_cache[key] = fallback
        return fallback


def get_cached_research(name: str, company: str) -> str:
    return _prospect_research_cache.get(_research_key(name, company), "")


@app.post("/api/prospect/research")
async def api_research_prospect(request: Request):
    """Research a prospect BEFORE calling. Caches result for instant use."""
    body = await request.json()
    name = body.get("name", "")
    title = body.get("title", "")
    company = body.get("company", "")
    if not name and not company:
        return JSONResponse({"error": "Need name or company"}, status_code=400)
    research = await research_prospect(name, title, company)
    return {"status": "ok", "research": research, "cached": True}


@app.post("/api/prospect/research-batch")
async def api_research_batch(request: Request):
    """Research multiple prospects at once (e.g., before campaign start)."""
    body = await request.json()
    prospects = body.get("prospects", [])
    tasks = []
    for p in prospects[:50]:
        tasks.append(research_prospect(
            p.get("name", ""), p.get("title", ""), p.get("company", "")
        ))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    ok = sum(1 for r in results if isinstance(r, str) and r)
    return {"status": "ok", "researched": ok, "total": len(prospects)}


@app.get("/api/prospect/research-cache")
async def api_research_cache():
    return {"cache": _prospect_research_cache, "count": len(_prospect_research_cache)}


# ════════════════════════════════════════════════════════════
#  KNOWLEDGE BASE
# ════════════════════════════════════════════════════════════
@app.post("/api/knowledge/upload")
async def upload_knowledge_doc(file: UploadFile = File(...)):
    """Upload a document (txt, pdf, csv, docx) to the AI knowledge base."""
    try:
        raw = await file.read()
        fname = file.filename or "unknown"
        text = ""
        if fname.endswith(".txt") or fname.endswith(".md"):
            text = raw.decode("utf-8", errors="ignore")
        elif fname.endswith(".csv"):
            text = raw.decode("utf-8", errors="ignore")
        elif fname.endswith(".pdf"):
            try:
                import io
                try:
                    import pypdf
                    reader = pypdf.PdfReader(io.BytesIO(raw))
                    text = "\n".join(p.extract_text() or "" for p in reader.pages)
                except ImportError:
                    text = f"[PDF uploaded: {fname} — install pypdf to extract text]"
            except Exception:
                text = f"[PDF uploaded: {fname} — could not extract text]"
        elif fname.endswith(".docx"):
            try:
                import io, zipfile
                zf = zipfile.ZipFile(io.BytesIO(raw))
                import xml.etree.ElementTree as ET
                doc_xml = zf.read("word/document.xml")
                tree = ET.fromstring(doc_xml)
                ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                text = "\n".join(node.text for node in tree.iter(f"{{{ns['w']}}}t") if node.text)
            except Exception:
                text = f"[DOCX uploaded: {fname} — could not extract text]"
        else:
            text = raw.decode("utf-8", errors="ignore")

        if text.strip():
            doc_entry = f"--- Document: {fname} ---\n{text[:3000]}"
            UPLOADED_DOCS_KNOWLEDGE.append(doc_entry)
            logger.info("Knowledge doc uploaded: %s (%d chars)", fname, len(text))
            return {"status": "ok", "filename": fname, "chars": len(text[:3000]),
                    "total_docs": len(UPLOADED_DOCS_KNOWLEDGE)}
        else:
            return JSONResponse({"error": "Could not extract text from file"}, status_code=400)
    except Exception as e:
        logger.error(f"Knowledge upload failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/knowledge")
async def get_knowledge():
    return {
        "website_knowledge": CLOUDFUZE_KNOWLEDGE[:500] + "...",
        "uploaded_docs": len(UPLOADED_DOCS_KNOWLEDGE),
        "doc_names": [d.split("\n")[0] for d in UPLOADED_DOCS_KNOWLEDGE],
    }


@app.delete("/api/knowledge")
async def clear_knowledge():
    UPLOADED_DOCS_KNOWLEDGE.clear()
    return {"status": "ok", "message": "Knowledge base cleared"}


# ════════════════════════════════════════════════════════════
#  ACTIVE CALLS CRUD
# ════════════════════════════════════════════════════════════
@app.get("/calls")
async def list_calls():
    return JSONResponse(content={"total": len(active_calls), "calls": active_calls})


@app.get("/calls/{call_control_id}")
async def get_call(call_control_id: str):
    call = active_calls.get(call_control_id) or get_call_by_control_id(call_control_id)
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    return JSONResponse(content=call)


@app.delete("/calls/{call_control_id}")
async def end_call(call_control_id: str):
    await hangup_call(call_control_id)
    return JSONResponse(content={"status": "hung up", "call_control_id": call_control_id})


# ════════════════════════════════════════════════════════════
#  ENTRYPOINT
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    logger.info(f"AI SDR starting on port {config.PORT}")
    logger.info(f"Dashboard: http://localhost:{config.PORT}")
    # Pass the app object directly to avoid accidental module-resolution collisions.
    # (When passing an app object, uvicorn can't use reload reliably.)
    uvicorn.run(app, host="0.0.0.0", port=config.PORT, reload=False)
