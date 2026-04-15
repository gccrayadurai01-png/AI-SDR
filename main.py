"""
Knight AI SDR — FastAPI server
Uses Telnyx server-side transcription (no WebSocket audio streaming).
Flow: call.answered → speak → start_transcription → call.transcription → Claude → speak → loop
"""
from __future__ import annotations

import asyncio
import os
from collections import Counter, defaultdict
from contextlib import asynccontextmanager
import json
import logging
import re as _re
import time
import uuid
from datetime import datetime, timedelta
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
from qa_kb_api import router as learn_router
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
    # Ensure data directory and files exist (for fresh deployments)
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "research").mkdir(exist_ok=True)
    defaults = {
        "calls.json": "[]",
        "contacts.json": '{"contacts": []}',
        "script.json": json.dumps({
            "sdr_name": "Anthony",
            "company_name": "",
            "call_objective": "Book a 15-minute discovery call",
            "target_persona": "",
            "value_proposition": "",
            "opening_line": "Hi {name}, this is {sdr_name} from {company} — did I catch you at a bad time?",
            "discovery_questions": "What tools are you currently using?\nWhat's your biggest challenge?\nHow do you measure success?\nWho else is involved?",
            "objection_handling": "Not interested: Totally fair — can I ask what you're using today?\nToo busy: When would be a better time?\nSend email: A quick 15-min call might be more valuable.\nWe have a solution: How's that working for you?",
            "booking_phrase": "Would you have 15 minutes this week or next for a quick chat?",
            "voicemail_message": "Hey {name}, this is {sdr_name} from {company}. I'd love to set up a quick call. Feel free to call me back!",
        }),
        "email_sequences.json": '{"sequences": []}',
    }
    for fname, default_content in defaults.items():
        fpath = data_dir / fname
        if not fpath.exists():
            fpath.write_text(default_content)
    yield


# ─── app ────────────────────────────────────────────────────
app = FastAPI(title="Knight AI SDR", version="3.0.0", lifespan=_app_lifespan)
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.include_router(learn_router)

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

# ─── Health check background task ────────────────────────
_health_check_task: asyncio.Task | None = None
_last_health_check: dict[str, Any] = {"status": "pending", "last_run": None, "result": None}

# ─── Telnyx AI Assistant (speech-to-speech) ───────────────
import telnyx
import httpx as _httpx

ASSISTANT_ID = "assistant-7b0da1f0-aaff-44c0-ae3e-a75cc89f1ddc"

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


SALES_TECHNIQUES = {
    "sandler": {
        "name": "Sandler",
        "description": "Low-pressure, Pain-based, Consultant approach",
        "prompt": """SANDLER METHOD:
1. BOND: Wait for response. Then low-pressure: "Not sure if this is relevant -- got a sec?"
2. CONTRACT: "Quick question or two, if it doesn't fit, no worries. Fair?"
3. PAIN: Ask what tools they use, what's broken. Listen, reflect, go deeper.
4. BUDGET: "Is that something with budget?" "Who else weighs in?"
5. BOOK: "Worth a quick 15-min look? Thursday or Friday?"
6. CLOSE: Confirm, thank, bye.""",
        "opening_template": "Hi {name}, this is {sdr_name} from {company} — did I catch you at a bad time?",
        "objection_handling": "Not interested: Totally fair — can I ask what you're using today?\nToo busy: When would be a better time to chat?\nSend email: A quick 15-min call might be more valuable.\nWe have a solution: How's that working for you? Any gaps?",
    },
    "gap": {
        "name": "Gap Selling",
        "description": "Quantifying Current vs. Future State",
        "prompt": """GAP SELLING METHOD:
1. CURRENT STATE: Ask about their current situation. "Walk me through how you handle X today."
2. IDENTIFY PROBLEMS: "What's not working?" "What does that cost you?"
3. FUTURE STATE: "If you could wave a magic wand, what would it look like?"
4. THE GAP: Quantify the gap between current and future. "So that gap is costing you roughly X?"
5. BRIDGE: Position your solution as the bridge. "We close exactly that gap."
6. BOOK: "Let me show you how — 15 minutes this week?"
""",
        "opening_template": "Hi {name}, this is {sdr_name} from {company}. I help companies like yours close the gap between where they are and where they want to be. Got a quick minute?",
        "objection_handling": "Not interested: What does your current process cost you monthly?\nToo busy: Totally get it. When's a better time?\nHave a solution: How close is it getting you to your ideal state?\nNo budget: Usually the gap we close pays for itself. Worth a quick look?",
    },
    "challenger": {
        "name": "Challenger",
        "description": "Teach, Tailor, Take Control",
        "prompt": """CHALLENGER METHOD:
1. TEACH: Lead with an insight they don't know. "We've been seeing a trend with companies like yours..."
2. TAILOR: Connect the insight to THEIR specific situation. "For a company your size, that usually means..."
3. TAKE CONTROL: Be direct about next steps. Don't be afraid to push back respectfully.
4. REFRAME: If they object, reframe the problem. "Most teams think X, but actually Y."
5. COMMERCIAL TEACHING: Share a perspective that leads to your solution naturally.
6. BOOK: "I've got a framework for this. 15 minutes — I'll share the data."
""",
        "opening_template": "Hi {name}, this is {sdr_name} from {company}. I've been researching companies like {company_prospect} and found something interesting. Got 30 seconds?",
        "objection_handling": "Not interested: Fair — but most of our customers said that before seeing the data. What if I shared one insight?\nToo busy: I'll be quick — one question: are you seeing X trend?\nHave a solution: Interesting — are you getting Y result? Most aren't.\nNo budget: This usually saves more than it costs. Worth validating?",
    },
    "spin": {
        "name": "SPIN",
        "description": "Questioning to discover needs",
        "prompt": """SPIN METHOD:
1. SITUATION: Ask about their current setup. "What are you using for X right now?"
2. PROBLEM: Identify issues. "What challenges do you run into with that?"
3. IMPLICATION: Explore impact. "When that happens, what does it cost you?"
4. NEED-PAYOFF: Get them to articulate value. "If you could fix that, what would it mean for your team?"
5. SOLUTION: Brief connection to your offering. Only AFTER they articulate the need.
6. BOOK: "Sounds like it's worth exploring. 15 minutes this week?"
""",
        "opening_template": "Hi {name}, this is {sdr_name} from {company}. I work with teams that handle {target_area} — mind if I ask a quick question?",
        "objection_handling": "Not interested: No worries — quick question though, how are you handling X today?\nToo busy: Totally understand. When would work better?\nHave a solution: Great — how's it performing on Y?\nNo budget: If it saved you Z hours a week, would that change things?",
    },
    "meddic": {
        "name": "MEDDIC",
        "description": "Rigorous Qualification Framework",
        "prompt": """MEDDIC METHOD:
1. METRICS: What metrics matter to them? "How do you measure success in this area?"
2. ECONOMIC BUYER: "Who signs off on decisions like this?"
3. DECISION CRITERIA: "What factors matter most when evaluating solutions?"
4. DECISION PROCESS: "Walk me through how you'd evaluate something like this."
5. IDENTIFY PAIN: "What's the #1 thing you'd fix if you could?"
6. CHAMPION: Build them as your internal champion. "Would you be open to exploring this with your team?"
7. BOOK: "Let me put together something specific for your metrics. 15 minutes?"
""",
        "opening_template": "Hi {name}, this is {sdr_name} from {company}. We help teams hit their KPIs in {target_area}. Quick question — how are you tracking success there today?",
        "objection_handling": "Not interested: Understood — what metrics are you focused on this quarter?\nToo busy: When would be better? I want to make sure this is relevant.\nHave a solution: How's it tracking against your KPIs?\nNo budget: If we could show measurable ROI, who would need to see it?",
    },
}


def _get_technique_prompt(technique: str) -> str:
    """Return the sales technique prompt block for the given technique key."""
    tech = SALES_TECHNIQUES.get(technique, SALES_TECHNIQUES["sandler"])
    return tech["prompt"]


def get_system_prompt() -> str:
    """Compact sales prompt for Telnyx AI Assistant -- must stay SHORT for voice to work."""
    s = load_script()
    sdr = s.get("sdr_name", "Alex")
    co = s.get("company_name", "Your Company")
    obj = s.get("call_objective", "Book a 15-minute discovery call")
    persona = s.get("target_persona", "IT / SaaS decision makers")
    vp = s.get("value_proposition", "")

    technique = s.get("sales_technique", "sandler")
    technique_block = _get_technique_prompt(technique)

    return f"""You={sdr}, SDR at {co}. LIVE cold call. You are a real human, never say you're AI.
Goal: {obj}. Target: {persona}.

IMPORTANT: The greeting has ALREADY been spoken. Do NOT repeat it. Do NOT say hi again. Wait for the prospect to respond, then continue the conversation.

RULES:
- Reply in 1 sentence max. NEVER more than 15 words per turn.
- Respond INSTANTLY. No pauses, no thinking delays.
- ONE question at a time. Wait for their answer.
- Sound natural: "yeah", "got it", "makes sense", "totally".
- After greeting, WAIT for them to speak first. Then respond.

{technique_block}

OBJECTIONS: Busy="When's better?" Not interested=one probe then respect. Has tool="How's that working?"
If they say stop/not interested/hang up after probe: "Totally understand, appreciate your time. Have a great day!" then END.
END: Say goodbye naturally and stop.""".strip()


def get_opening_line(name: str = "there", title: str = "", company: str = "") -> str:
    """Generate opening line — AI decides the style, we just fill in the names."""
    s = load_script()
    sdr = s.get("sdr_name", "Alex")
    co = s.get("company_name", "Your Company")
    # Simple pattern interrupt opener — the AI will take it from here
    line = f"Hey {name}, this is {sdr} from {co} -- did I catch you at a bad time?"
    return line


def _get_compact_knowledge() -> str:
    """Product knowledge injected as message_history -- keeps system prompt short for Telnyx.
    Pulls from the uploaded knowledge base / script config rather than hardcoded content."""
    full = get_full_knowledge()
    if full and full.strip():
        # Truncate to keep AI Assistant message_history lean
        return full[:2000]
    # Fallback: minimal from script
    s = load_script()
    vp = s.get("value_proposition", "")
    co = s.get("company_name", "Knight")
    return f"PRODUCT KNOWLEDGE for {co} -- weave into conversation naturally.\n\n{vp}" if vp else ""


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
                    "user_idle_timeout_secs": 90,
                    "max_duration_secs": 1800,
                },
                "interruption_settings": {
                    "enable": True,
                    "start_speaking_plan": {
                        "wait_seconds": 0.5,
                    },
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
                    "model_id": "eleven_multilingual_v2",
                }, headers=headers, params=params)
                r.raise_for_status()
                import base64
                b64 = base64.b64encode(r.content).decode("ascii")
                _filler_audio_cache[phrase] = b64
                logger.info("Cached filler: \"%s\" (%d bytes)", phrase, len(r.content))
            except Exception as e:
                logger.warning("Failed to cache filler \"%s\": %s", phrase, e)

    logger.info("Pre-cached %d/%d filler phrases (Anthony turbo)", len(_filler_audio_cache), len(config.PHONE_FILLER_UTTERANCES))


_GOODBYE_PATTERNS = _re.compile(
    r'\b(goodbye|good\s*bye|bye\s*bye|talk\s*soon|have\s*a\s*great|'
    r'take\s*care|appreciate\s*your\s*time|thanks\s*for\s*your\s*time|'
    r'nice\s*talking|nice\s*chatting|have\s*a\s*good\s*one|'
    r'catch\s*you\s*later|speak\s*soon|cheers|so\s*long|'
    r'not\s*interested|stop\s*calling|remove\s*me|do\s*not\s*call|'
    r"don'?t\s*call|hang\s*up|go\s*away|leave\s*me\s*alone)\b",
    _re.IGNORECASE,
)

# Track pending auto-hangup tasks so we can cancel if conversation continues
_auto_hangup_tasks: dict[str, asyncio.Task] = {}

# Track last speech time per call for silence detection
_last_speech_time: dict[str, float] = {}
_silence_watchdog_tasks: dict[str, asyncio.Task] = {}
_ai_assistant_started: set[str] = set()  # cc_ids where AI Assistant was started — NEVER auto-hangup these


def _is_goodbye(text: str) -> bool:
    """Check if text contains a natural conversation-ending phrase."""
    return bool(_GOODBYE_PATTERNS.search(text or ""))


async def _silence_watchdog(cc_id: str):
    """Monitor call silence — only log, NEVER auto-hangup.
    Telnyx AI Assistant manages its own conversation lifecycle.
    We must not kill calls — data collection requires full conversations."""
    try:
        while cc_id in active_calls and active_calls[cc_id].get("state") != "ended":
            await asyncio.sleep(30)
            last = _last_speech_time.get(cc_id, 0)
            if last and (time.time() - last) > 60:
                logger.info("SILENCE MONITOR: No speech for 60s on %s — logging only (NOT hanging up)", cc_id)
    except asyncio.CancelledError:
        pass
    finally:
        _silence_watchdog_tasks.pop(cc_id, None)


async def _auto_hangup_after_goodbye(cc_id: str):
    """DISABLED — previously hung up calls mid-conversation. Now just logs.
    Telnyx AI Assistant handles conversation ending on its own."""
    try:
        await asyncio.sleep(5.0)
        if cc_id not in active_calls:
            return
        if active_calls[cc_id].get("state") == "ended":
            return
        logger.info("Goodbye detected for %s — logging only (NOT hanging up)", cc_id)
    except asyncio.CancelledError:
        pass
    finally:
        _auto_hangup_tasks.pop(cc_id, None)


async def _ai_assistant_watchdog(cc_id: str, greeting: str):
    """Watchdog DISABLED — Telnyx AI Assistant manages its own lifecycle.
    Keeping function stub so callers don't break."""
    return


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
    global _health_check_task
    logger.info("=== STARTUP: APP_BASE_URL = %s ===", config.APP_BASE_URL)
    logger.info("=== STARTUP: Webhook URL = %s/webhooks/telnyx ===", config.APP_BASE_URL)
    if "ngrok" in config.APP_BASE_URL and os.environ.get("RAILWAY_ENVIRONMENT"):
        logger.error("!!! CRITICAL: APP_BASE_URL points to ngrok but running on Railway — webhooks will FAIL !!!")
        logger.error("!!! Set APP_BASE_URL in Railway env vars or enable RAILWAY_PUBLIC_DOMAIN !!!")
    _get_tx()  # Pre-init Telnyx client — no cold start on first call
    _rebuild_hot_cache()
    try:
        sync_assistant_to_script()
    except Exception as e:
        logger.warning("Startup assistant sync failed (non-fatal): %s", e)
    await _check_assistant_health()
    await _precache_filler_audio()
    _health_check_task = asyncio.create_task(_health_check_loop())


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
    return JSONResponse({"status": "Knight AI SDR running - dashboard not found"})


# ════════════════════════════════════════════════════════════
#  API — HEALTH & STATUS
# ════════════════════════════════════════════════════════════
@app.get("/login")
async def serve_login():
    login_page = STATIC_DIR / "login.html"
    return FileResponse(str(login_page))


@app.get("/api/health")
async def health():
    return {"status": "ok", "active_calls": len(active_calls), "base_url": config.APP_BASE_URL}


@app.get("/api/telnyx/diagnostics")
async def telnyx_diagnostics():
    return run_telnyx_diagnostics()


@app.get("/api/status")
async def api_status():
    config.reload_secrets()
    # Use actual config values — works on Railway (env vars) AND local (.env file)
    return {
        "telnyx":    bool(config.TELNYX_API_KEY and config.TELNYX_CONNECTION_ID and config.TELNYX_PHONE_NUMBER),
        "deepgram":  bool(os.environ.get("DEEPGRAM_API_KEY")),
        "anthropic": bool(config.ANTHROPIC_API_KEY),
        "apollo":    bool(config.APOLLO_API_KEY),
        "email":     bool(config.SMTP_HOST and config.EMAIL_FROM) or bool(config.SENDGRID_API_KEY),
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


@app.get("/api/voices")
async def api_list_voices():
    """Return available ElevenLabs voices for campaign and quick-dial voice selection."""
    voices = [
        {"id": "49uxf7RPOcr58oowljKn", "name": "Anthony - American Voice", "description": "American English voice"},
        {"id": "a9paacvZxTlkONiCPzfC", "name": "Indian Voice", "description": "Indian accent voice"},
    ]
    return {"voices": voices}


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

    env_path.write_text(env_text, encoding="utf-8")
    config.reload_secrets()
    return {
        "status": "saved",
        "note": "Keys saved and reloaded — Test buttons use new values immediately.",
    }


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
- discovery_questions: array of 5-7 short questions tailored to the company's value proposition and target persona
- objections: object with keys: not_interested, send_email, call_back, have_solution, no_budget (each a short 1-sentence response)
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
#  API — MULTI-AGENT MANAGEMENT
# ════════════════════════════════════════════════════════════
AGENTS_FILE = Path(__file__).parent / "data" / "agents.json"

def _load_agents() -> list[dict]:
    if AGENTS_FILE.exists():
        try:
            return json.loads(AGENTS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []

def _save_agents(agents: list[dict]) -> None:
    AGENTS_FILE.write_text(json.dumps(agents, indent=2, default=str), encoding="utf-8")

@app.get("/api/sales-techniques")
async def list_sales_techniques():
    """List all available sales techniques."""
    return [{"id": k, "name": v["name"], "description": v["description"]} for k, v in SALES_TECHNIQUES.items()]

@app.get("/api/agents")
async def list_agents():
    agents = _load_agents()
    return {"agents": agents, "total": len(agents)}

@app.post("/api/agents")
async def create_agent(request: Request):
    body = await request.json()
    agents = _load_agents()
    agent = {
        "id": body.get("id") or str(uuid.uuid4())[:8],
        "name": body.get("name", "New Agent"),
        "type": body.get("type", "outbound"),
        "status": body.get("status", "active"),
        "created_at": datetime.utcnow().isoformat(),
        "sdr_name": body.get("sdr_name", ""),
        "company_name": body.get("company_name", ""),
        "call_objective": body.get("call_objective", ""),
        "target_persona": body.get("target_persona", ""),
        "value_proposition": body.get("value_proposition", ""),
        "opening_line": body.get("opening_line", ""),
        "discovery_questions": body.get("discovery_questions", ""),
        "objection_handling": body.get("objection_handling", ""),
        "booking_phrase": body.get("booking_phrase", ""),
        "voicemail_message": body.get("voicemail_message", ""),
        "website": body.get("website", ""),
        "sales_technique": body.get("sales_technique", "sandler"),
    }
    agents.append(agent)
    _save_agents(agents)
    return {"status": "created", "agent": agent}

@app.patch("/api/agents/{agent_id}")
async def update_agent(agent_id: str, request: Request):
    body = await request.json()
    agents = _load_agents()
    for a in agents:
        if a.get("id") == agent_id:
            a.update(body)
            a["updated_at"] = datetime.utcnow().isoformat()
            _save_agents(agents)
            return {"status": "updated", "agent": a}
    raise HTTPException(status_code=404, detail="Agent not found")

@app.delete("/api/agents/{agent_id}")
async def delete_agent(agent_id: str):
    agents = _load_agents()
    agents = [a for a in agents if a.get("id") != agent_id]
    _save_agents(agents)
    return {"status": "deleted"}

@app.post("/api/agents/{agent_id}/activate")
async def activate_agent(agent_id: str):
    """Set this agent as the active script and sync to Telnyx AI Assistant."""
    agents = _load_agents()
    agent = None
    for a in agents:
        if a.get("id") == agent_id:
            agent = a
            break
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    # Copy agent fields to script.json (makes it the active calling agent)
    script_data = {
        "sdr_name": agent.get("sdr_name", ""),
        "company_name": agent.get("company_name", ""),
        "call_objective": agent.get("call_objective", ""),
        "target_persona": agent.get("target_persona", ""),
        "value_proposition": agent.get("value_proposition", ""),
        "opening_line": agent.get("opening_line", ""),
        "discovery_questions": agent.get("discovery_questions", ""),
        "objection_handling": agent.get("objection_handling", ""),
        "booking_phrase": agent.get("booking_phrase", ""),
        "voicemail_message": agent.get("voicemail_message", ""),
        "sales_technique": agent.get("sales_technique", "sandler"),
    }
    save_script(script_data)
    _rebuild_hot_cache()
    sync_assistant_to_script()
    return {"status": "activated", "agent_id": agent_id}

@app.post("/api/agents/build-from-website")
async def build_agent_from_website(request: Request):
    """Scrape a website URL and use Claude to auto-generate a full agent config."""
    body = await request.json()
    url = (body.get("url") or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")
    if not config.ANTHROPIC_API_KEY:
        raise HTTPException(status_code=400, detail="ANTHROPIC_API_KEY is not set")

    # Fetch the website content
    import httpx as _httpx
    try:
        async with _httpx.AsyncClient(follow_redirects=True, timeout=15) as http:
            resp = await http.get(url, headers={"User-Agent": "Mozilla/5.0 Knight-AI-SDR/1.0"})
            resp.raise_for_status()
            html_content = resp.text[:30000]  # Limit to 30k chars
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch website: {e}") from e

    # Strip HTML tags for cleaner text
    import re
    text = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()[:12000]

    prompt = f"""You are an expert sales strategist. Analyze this company's website content and create a complete AI SDR agent configuration for cold-calling their potential customers.

Website URL: {url}
Website Content:
{text}

Generate a JSON object with these exact fields:
- "name": A short agent name (e.g. "Enterprise Closer" or "Product Demo Setter")
- "company_name": The company name from the website
- "sdr_name": Suggest a professional first name for the AI SDR
- "call_objective": What the call should achieve (e.g. "Book a 15-minute demo")
- "target_persona": Who should be called (job titles, company size, industry)
- "value_proposition": 2-3 sentence value prop based on the product/service
- "opening_line": A natural cold-call opener using {{name}}, {{sdr_name}}, {{company}} placeholders. Use a pattern-interrupt style.
- "discovery_questions": 5-7 qualifying questions (newline separated) tailored to this product
- "objection_handling": Handle common objections: not_interested, send_email, have_solution, no_budget, call_back (one paragraph covering all)
- "booking_phrase": Natural way to ask for a meeting
- "voicemail_message": A 30-second voicemail script using {{name}}, {{sdr_name}}, {{company}} placeholders

Return ONLY valid JSON, no markdown fences, no explanation."""

    try:
        client = AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
        resp = await client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        agent_data = json.loads(raw)
        agent_data["website"] = url
        agent_data["type"] = "outbound"
        agent_data["status"] = "draft"
        return {"status": "ok", "agent": agent_data}
    except Exception as e:
        logger.error("Build from website failed: %s", e)
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
    agent_id: str = ""
    auto_mode: bool = False
    auto_mode_break_seconds: float = Field(300.0, ge=0, le=86400)


@app.post("/api/campaign/start")
async def campaign_start(body: CampaignStartBody):
    if not body.prospects:
        raise HTTPException(status_code=400, detail="No prospects in queue")

    # If agent_id provided, activate that agent before starting calls
    if body.agent_id:
        agents = _load_agents()
        agent = next((a for a in agents if a.get("id") == body.agent_id), None)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {body.agent_id} not found")
        script_data = {
            "sdr_name": agent.get("sdr_name", ""),
            "company_name": agent.get("company_name", ""),
            "call_objective": agent.get("call_objective", ""),
            "target_persona": agent.get("target_persona", ""),
            "value_proposition": agent.get("value_proposition", ""),
            "opening_line": agent.get("opening_line", ""),
            "discovery_questions": agent.get("discovery_questions", ""),
            "objection_handling": agent.get("objection_handling", ""),
            "booking_phrase": agent.get("booking_phrase", ""),
            "voicemail_message": agent.get("voicemail_message", ""),
        }
        save_script(script_data)
        _rebuild_hot_cache()
        sync_assistant_to_script()

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


# ════════════════════════════════════════════════════════════
#  NAMED CAMPAIGNS CRUD
# ════════════════════════════════════════════════════════════
_CAMPAIGNS_FILE = Path(__file__).parent / "data" / "campaigns.json"
_active_campaign_id: str | None = None


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
        "campaign_id": _active_campaign_id,
    }

def _load_campaigns() -> list[dict]:
    if _CAMPAIGNS_FILE.exists():
        try:
            return json.loads(_CAMPAIGNS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []

def _save_campaigns(data: list[dict]) -> None:
    _CAMPAIGNS_FILE.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

def _get_campaign(camp_id: str) -> dict | None:
    return next((c for c in _load_campaigns() if c.get("id") == camp_id), None)

def _update_campaign(camp_id: str, updates: dict) -> dict | None:
    camps = _load_campaigns()
    for c in camps:
        if c.get("id") == camp_id:
            c.update(updates)
            _save_campaigns(camps)
            return c
    return None


@app.get("/api/campaigns/list")
async def list_campaigns():
    return _load_campaigns()


@app.get("/api/campaigns/history")
async def get_campaigns_history():
    """Backwards compat — returns same as list."""
    return _load_campaigns()


@app.post("/api/campaigns/create")
async def create_campaign(request: Request):
    body = await request.json()
    name = body.get("name", "").strip()
    if not name:
        raise HTTPException(400, "Campaign name is required")
    prospects = body.get("prospects", [])
    # Normalize prospects
    for p in prospects:
        p.setdefault("outcome", "")
        p.setdefault("status", "queued")
    camp = {
        "id": str(uuid.uuid4())[:8],
        "name": name,
        "created_at": datetime.now().isoformat(),
        "status": "draft",
        "agent_id": body.get("agent_id", ""),
        "voice_id": body.get("voice_id", ""),
        "spacing_seconds": body.get("spacing_seconds", 60),
        "auto_mode": body.get("auto_mode", True),
        "prospects": prospects,
        "dialed": 0,
        "current_index": 0,
        "outcomes": {},
    }
    camps = _load_campaigns()
    camps.insert(0, camp)
    _save_campaigns(camps)

    if body.get("start_now"):
        await _start_named_campaign(camp["id"])

    return camp


@app.get("/api/campaigns/{campaign_id}")
async def get_campaign(campaign_id: str):
    # If this is the active running campaign, enrich with live state
    c = _get_campaign(campaign_id)
    if not c:
        raise HTTPException(404, "Campaign not found")
    if _active_campaign_id == campaign_id and campaign_lib.state.status in ("running", "paused"):
        c["status"] = campaign_lib.state.status
        c["dialed"] = campaign_lib.state.index
        c["current_index"] = campaign_lib.state.index
    return c


@app.post("/api/campaigns/{campaign_id}/start")
async def start_named_campaign_endpoint(campaign_id: str):
    ok = await _start_named_campaign(campaign_id)
    if not ok:
        raise HTTPException(409, "Campaign already running or no prospects")
    return {"status": "started"}


@app.post("/api/campaigns/{campaign_id}/add-prospects")
async def add_prospects_to_campaign(campaign_id: str, request: Request):
    body = await request.json()
    new_prospects = body.get("prospects", [])
    c = _get_campaign(campaign_id)
    if not c:
        raise HTTPException(404, "Campaign not found")
    for p in new_prospects:
        p.setdefault("outcome", "")
        p.setdefault("status", "queued")
    existing = c.get("prospects", [])
    existing.extend(new_prospects)
    _update_campaign(campaign_id, {"prospects": existing})
    return {"status": "ok", "total": len(existing)}


@app.delete("/api/campaigns/{campaign_id}")
@app.delete("/api/campaigns/history/{campaign_id}")
async def delete_campaign_record(campaign_id: str):
    camps = _load_campaigns()
    camps = [c for c in camps if c.get("id") != campaign_id]
    _save_campaigns(camps)
    return {"ok": True}


async def _start_named_campaign(camp_id: str) -> bool:
    """Start a named campaign using the existing campaign runner."""
    global _active_campaign_id
    c = _get_campaign(camp_id)
    if not c:
        return False
    prospects = c.get("prospects", [])
    if not prospects:
        return False

    # Activate agent if specified
    agent_id = c.get("agent_id", "")
    if agent_id:
        agents = _load_agents()
        agent = next((a for a in agents if a.get("id") == agent_id), None)
        if agent:
            script_data = {
                "sdr_name": agent.get("sdr_name", ""),
                "company_name": agent.get("company_name", ""),
                "call_objective": agent.get("call_objective", ""),
                "target_persona": agent.get("target_persona", ""),
                "value_proposition": agent.get("value_proposition", ""),
                "opening_line": agent.get("opening_line", ""),
                "discovery_questions": agent.get("discovery_questions", ""),
                "objection_handling": agent.get("objection_handling", ""),
                "booking_phrase": agent.get("booking_phrase", ""),
                "voicemail_message": agent.get("voicemail_message", ""),
            }
            save_script(script_data)
            _rebuild_hot_cache()
            sync_assistant_to_script()

    spacing = c.get("spacing_seconds", 60)

    # ── Voice override: if campaign specifies a voice_id, use it for all calls ──
    _original_voice_id = config.ELEVENLABS_VOICE_ID
    campaign_voice_id = (c.get("voice_id") or "").strip()
    if campaign_voice_id:
        config.ELEVENLABS_VOICE_ID = campaign_voice_id
        logger.info("Campaign voice override: %s → %s", _original_voice_id, campaign_voice_id)

    _update_campaign(camp_id, {"status": "running", "started_at": datetime.now().isoformat()})
    _active_campaign_id = camp_id

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

    async def _on_done():
        """Called when campaign runner finishes — update persisted state."""
        global _active_campaign_id
        # Restore original voice when campaign ends
        if campaign_voice_id:
            config.ELEVENLABS_VOICE_ID = _original_voice_id
            logger.info("Campaign ended — voice restored to: %s", _original_voice_id)
        st = campaign_lib.state
        updates = {
            "dialed": st.index,
            "current_index": st.index,
            "status": st.status if st.status in ("completed", "stopped") else "completed",
        }
        _update_campaign(camp_id, updates)
        _active_campaign_id = None

    ok = start_campaign(prospects, spacing, dial_one)
    if not ok:
        _update_campaign(camp_id, {"status": "draft"})
        _active_campaign_id = None
        return False

    # Monitor completion in background
    async def _monitor():
        while campaign_lib.state.status in ("running", "paused"):
            # Update live dialed count periodically
            _update_campaign(camp_id, {
                "dialed": campaign_lib.state.index,
                "current_index": campaign_lib.state.index,
                "status": campaign_lib.state.status,
            })
            await asyncio.sleep(5)
        await _on_done()

    asyncio.create_task(_monitor())
    return True


# ════════════════════════════════════════════════════════════
#  AUTH
# ════════════════════════════════════════════════════════════
KNIGHT_USERS = {
    "admin": "Knight2024!",
    "roy": "knight123",
}

@app.post("/api/auth/login")
async def login(request: Request):
    body = await request.json()
    username = body.get("username", "").strip().lower()
    password = body.get("password", "")
    if KNIGHT_USERS.get(username) == password:
        return {"ok": True, "username": username, "token": f"knight-{username}-session"}
    raise HTTPException(status_code=401, detail="Invalid username or password")


# ════════════════════════════════════════════════════════════
#  DATABASE RESET
# ════════════════════════════════════════════════════════════
@app.post("/api/admin/reset-db")
async def reset_database(request: Request):
    body = await request.json()
    if body.get("confirm") != "RESET":
        raise HTTPException(400, "Send {\"confirm\": \"RESET\"} to confirm")
    data_dir = Path(__file__).parent / "data"
    files_cleared = []
    for fname in ["calls.json", "contacts.json", "tasks.json", "campaigns.json"]:
        f = data_dir / fname
        if f.exists():
            f.unlink()
            files_cleared.append(fname)
    # Reset script to blank defaults
    (data_dir / "script.json").write_text(json.dumps({
        "sdr_name": "Alex",
        "company_name": "Your Company",
        "call_objective": "Book a 15-minute discovery call",
        "target_persona": "Decision makers at mid-market companies",
        "value_proposition": "We help companies solve their biggest challenges.",
        "opening_line": "Hi {name}, this is {sdr_name} from {company} — did I catch you at a bad time?",
        "discovery_questions": "What tools are you currently using?\nWhat's the biggest challenge you're facing?",
        "objection_handling": "Not interested: Totally fair — can I ask what you're using today?\nToo busy: When would be a better time?",
        "booking_phrase": "Would you have 15 minutes this week for a quick chat?",
        "voicemail_message": "Hey {name}, this is {sdr_name} from {company}. Would love to connect — call me back or I'll try again. Have a great day!",
    }, indent=2), encoding="utf-8")
    _rebuild_hot_cache()
    return {"ok": True, "cleared": files_cleared}


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
    voice_id:        str = ""  # ElevenLabs voice ID override for this call


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
    # Apply voice override for this call if specified
    voice_id = (req.voice_id or "").strip()
    if voice_id:
        config.ELEVENLABS_VOICE_ID = voice_id
        logger.info("Voice override for call: %s", voice_id)
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
    """Start AI Assistant off the main webhook thread — zero blocking on call.answered."""
    loop = asyncio.get_event_loop()

    # ── Build greeting from CACHED script (no disk read) ──
    s = _cached_script or load_script()
    sdr = s.get("sdr_name", "Alex")
    co = s.get("company_name", "Your Company")
    greeting = f"Hey {name}, this is {sdr} from {co} -- did I catch you at a bad time?"

    # ── Build message_history from CACHED knowledge (no recompute) ──
    msg_history = list(_cached_knowledge_history)  # shallow copy
    research = get_cached_research(name, company)
    if research:
        msg_history.append({"role": "user", "content": f"[BRIEFING]\n{research[:500]}"})
        msg_history.append({"role": "assistant", "content": "Got it."})

    # ── Build AI Assistant kwargs — let assistant use its own voice config ──
    ai_kwargs: dict[str, Any] = {
        "call_control_id": cc_id,
        "assistant": {"id": ASSISTANT_ID},
        "greeting": greeting,
        "transcription": {"model": "distil-whisper/distil-large-v2"},
        "interruption_settings": {"enable": True},
    }
    if msg_history:
        ai_kwargs["message_history"] = msg_history

    tx = _get_tx()

    # ── Fire recording detached — delay 5s so it doesn't overlap greeting ──
    async def _fire_recording_detached():
        await asyncio.sleep(5)
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
        _ai_assistant_started.add(cc_id)
        logger.info("AI Assistant started in %.0fms — greeting: %s", latency_ms, greeting[:60])
        asyncio.create_task(_fire_recording_detached())
    except Exception as e:
        err_str = str(e)
        # 422 "already in progress" means Telnyx already auto-started it — treat as success
        if "90061" in err_str or "already in progress" in err_str.lower():
            logger.info("AI Assistant already running (auto-started by Telnyx) — continuing normally")
            active_calls.setdefault(cc_id, {})["ai_assistant"] = True
            _ai_assistant_started.add(cc_id)
            asyncio.create_task(_fire_recording_detached())
        else:
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

            # Start silence watchdog — auto-hangup if no speech for 30s
            _last_speech_time[cc_id] = time.time()
            if cc_id in _silence_watchdog_tasks:
                _silence_watchdog_tasks[cc_id].cancel()
            _silence_watchdog_tasks[cc_id] = asyncio.create_task(_silence_watchdog(cc_id))

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
            # Update silence watchdog — speech detected
            if cc_id:
                _last_speech_time[cc_id] = time.time()
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

        # ── CONVERSATION ENDED → generate insights; Telnyx handles hangup ──
        elif etype == "call.conversation.ended":
            if cc_id:
                logger.info("Conversation ended for %s — generating insights", cc_id)
                asyncio.create_task(_generate_call_insights(cc_id))

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
            # Clean up watchdogs
            _last_speech_time.pop(hang_cc, None)
            task = _silence_watchdog_tasks.pop(hang_cc, None)
            if task:
                task.cancel()
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
            asyncio.create_task(_remove_ended_call_after(hang_cc))
            conversations.pop(hang_cc, None)
            _ai_assistant_first_event.pop(hang_cc, None)
            _ai_assistant_started.discard(hang_cc)
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
        "website_knowledge": (CLOUDFUZE_KNOWLEDGE or "")[:500] + "...",
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
#  BELLA AI SEARCH
# ════════════════════════════════════════════════════════════
@app.post("/api/bella/search")
async def bella_search(request: Request):
    """Bella AI - conversational prospect search assistant.
    Accepts a natural language query, uses Claude to extract ICP criteria,
    then searches Apollo and enriches results."""
    body = await request.json()
    query = body.get("query", "")
    conversation = body.get("conversation", [])

    if not query:
        raise HTTPException(400, "query is required")

    config.reload_secrets()
    api_key = config.ANTHROPIC_API_KEY
    if not api_key:
        raise HTTPException(500, "Anthropic API key not configured")
    client = AsyncAnthropic(api_key=api_key)

    # Build conversation for Bella
    messages: list[dict[str, str]] = []
    for msg in conversation:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": query})

    bella_system = """You are Bella, an AI sales research assistant for Knight. You help find ideal prospects.

When the user describes their ideal customer, extract search criteria and respond in TWO parts:
1. A friendly conversational response acknowledging what they want
2. A JSON block with Apollo search parameters

Always respond with this exact format:
<response>Your friendly message here</response>
<search>{"q_keywords": "", "person_titles": [], "person_locations": [], "organization_locations": [], "person_seniorities": [], "organization_num_employees_ranges": [], "q_organization_domains_list": []}</search>

If you don't have enough info yet, just use <response> without <search> and ask qualifying questions like:
- What industry are you targeting?
- What job titles should I look for?
- Any specific company size?
- Geographic preferences?
- Any specific companies or domains?

Be warm, helpful, and conversational. Guide them to give you enough info for a good search."""

    resp = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=bella_system,
        messages=messages,
    )

    reply_text = resp.content[0].text

    # Parse response and search parts
    response_match = _re.search(r"<response>(.*?)</response>", reply_text, _re.DOTALL)
    search_match = _re.search(r"<search>(.*?)</search>", reply_text, _re.DOTALL)

    bella_reply = response_match.group(1).strip() if response_match else reply_text
    search_results = None

    if search_match:
        try:
            search_params = json.loads(search_match.group(1))
            # Actually search Apollo
            raw = await apollo_client.search_people(**search_params)
            people_raw = raw.get("people", [])
            people = []
            for p in people_raw:
                org = p.get("organization", {}) or {}
                people.append({
                    "apollo_person_id": p.get("id", ""),
                    "first_name": p.get("first_name", ""),
                    "last_name": p.get("last_name", ""),
                    "title": p.get("title", ""),
                    "company": org.get("name", ""),
                    "phone": p.get("phone_number") or "",
                    "email": p.get("email") or "",
                    "linkedin_url": p.get("linkedin_url") or "",
                    "location": p.get("city") or "",
                })
            search_results = {
                "people": people,
                "total": raw.get("pagination", {}).get("total_entries", 0),
            }
        except Exception as e:
            logger.warning("Bella Apollo search failed: %s", e)
            bella_reply += f"\n\nI tried searching but hit an issue: {e}. Could you refine your criteria?"

    return {
        "reply": bella_reply,
        "search_results": search_results,
        "has_search": search_match is not None,
    }


# ════════════════════════════════════════════════════════════
#  HEALTH CHECK (background + on-demand)
# ════════════════════════════════════════════════════════════
@app.get("/api/health-check/status")
async def health_check_status():
    return _last_health_check


@app.post("/api/health-check/run")
async def run_health_check_now():
    """Manually trigger a health check."""
    result = await _perform_health_check()
    return result


async def _perform_health_check():
    """Run system health check -- verify Telnyx, Anthropic, Apollo connections."""
    global _last_health_check
    checks: dict[str, Any] = {}

    # Check Telnyx
    try:
        _get_tx()
        checks["telnyx"] = {"status": "ok", "phone": config.TELNYX_PHONE_NUMBER}
    except Exception as e:
        checks["telnyx"] = {"status": "error", "error": str(e)}

    # Check Anthropic
    try:
        client = AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
        await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content": "ping"}],
        )
        checks["anthropic"] = {"status": "ok"}
    except Exception as e:
        checks["anthropic"] = {"status": "error", "error": str(e)}

    # Check Apollo
    try:
        result = await apollo_client.test_connection()
        checks["apollo"] = {"status": "ok" if result.get("ok") else "error"}
    except Exception as e:
        checks["apollo"] = {"status": "error", "error": str(e)}

    all_ok = all(c["status"] == "ok" for c in checks.values())
    _last_health_check = {
        "status": "ok" if all_ok else "degraded",
        "last_run": datetime.now().isoformat(),
        "checks": checks,
    }
    return _last_health_check


async def _health_check_loop():
    """Run health check every 30 minutes."""
    while True:
        try:
            await _perform_health_check()
            logger.info("Health check completed: %s", _last_health_check["status"])
        except Exception as e:
            logger.error("Health check failed: %s", e)
        await asyncio.sleep(1800)  # 30 minutes


# ════════════════════════════════════════════════════════════
#  CALLBACK SCHEDULING
# ════════════════════════════════════════════════════════════
@app.post("/api/callbacks/schedule")
async def schedule_callback(request: Request):
    """Schedule a callback from conversation context."""
    body = await request.json()
    phone = body.get("phone", "")
    name = body.get("name", "")
    company = body.get("company", "")
    callback_time = body.get("callback_time", "")
    notes = body.get("notes", "")
    call_control_id = body.get("call_control_id", "")

    if not phone or not callback_time:
        raise HTTPException(400, "phone and callback_time required")

    task = {
        "id": str(uuid.uuid4())[:8],
        "type": "callback",
        "prospect_name": name,
        "phone": phone,
        "company": company,
        "due_date": callback_time,
        "notes": notes or "Callback requested during call",
        "status": "pending",
        "call_control_id": call_control_id,
        "auto_dial": True,
        "created_at": datetime.now().isoformat(),
    }
    save_task(task)

    # Schedule the auto-dial
    asyncio.create_task(_auto_callback_worker(task))

    return {"ok": True, "task": task}


async def _auto_callback_worker(task: dict):
    """Wait until callback time, then auto-dial."""
    try:
        target = datetime.fromisoformat(task["due_date"])
        now = datetime.now()
        delay = (target - now).total_seconds()
        if delay > 0:
            logger.info("Callback scheduled for %s (%d seconds from now)", task["due_date"], delay)
            await asyncio.sleep(delay)

        # Check if task still pending
        tasks_list = load_tasks()
        for t in tasks_list:
            if t["id"] == task["id"] and t["status"] == "pending":
                # Auto-dial
                logger.info("Auto-callback: dialing %s for %s", task["phone"], task["prospect_name"])
                req = CallRequest(
                    to_number=task["phone"],
                    prospect_name=task["prospect_name"],
                    notes=task.get("notes", ""),
                )
                await place_outbound_call(req)
                update_task(task["id"], status="completed")
                break
    except Exception as e:
        logger.error("Auto-callback failed: %s", e)


@app.post("/api/calls/{call_control_id}/parse-callback")
async def parse_callback_from_transcript(call_control_id: str):
    """Use Claude to parse callback request from call transcript."""
    call = get_call_by_control_id(call_control_id)
    if not call:
        raise HTTPException(404, "Call not found")

    transcript = call.get("transcript", "")
    if not transcript:
        return {"found": False}

    # Build transcript text
    if isinstance(transcript, list):
        transcript_text = "\n".join(
            f"{'AI' if t.get('role') == 'agent' else 'Prospect'}: {t.get('text', '')}"
            for t in transcript if isinstance(t, dict)
        )
    else:
        transcript_text = str(transcript)

    client = AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
    resp = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        system="Extract callback scheduling requests from this call transcript. If the prospect mentioned a specific time to call back (e.g., 'call me tomorrow at 5pm', 'try me next Tuesday'), return a JSON with: {\"found\": true, \"callback_time\": \"ISO datetime\", \"raw_text\": \"what they said\"}. If no callback was mentioned, return {\"found\": false}. Use today's date as reference and assume the prospect's local timezone.",
        messages=[{"role": "user", "content": f"Today is {datetime.now().strftime('%Y-%m-%d %H:%M')}. Transcript:\n{transcript_text}"}],
    )

    try:
        result = json.loads(resp.content[0].text)
        return result
    except Exception:
        return {"found": False}


# ════════════════════════════════════════════════════════════
#  DASHBOARD STATS
# ════════════════════════════════════════════════════════════
@app.get("/api/dashboard/stats")
async def dashboard_stats():
    """Rich dashboard stats for charts and graphs."""
    calls = load_calls()
    tasks_list = load_tasks()
    contacts = contacts_store.list_contacts()

    # Call stats by day (last 7 days)
    daily_calls: dict[str, int] = defaultdict(int)
    daily_answered: dict[str, int] = defaultdict(int)
    outcomes: Counter = Counter()
    total_duration = 0

    for c in calls:
        dt_str = c.get("start_time") or c.get("created_at", "")
        if dt_str:
            try:
                day = dt_str[:10]
                daily_calls[day] += 1
                if c.get("status") not in ("initiated", "failed"):
                    daily_answered[day] += 1
            except Exception:
                pass
        outcome = c.get("outcome") or c.get("insights", {}).get("outcome", "unknown")
        outcomes[outcome] += 1
        dur = c.get("duration_seconds", 0)
        if isinstance(dur, (int, float)):
            total_duration += dur

    # Sort by date, last 7 days
    last_7 = []
    for i in range(6, -1, -1):
        d = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        last_7.append({
            "date": d,
            "calls": daily_calls.get(d, 0),
            "answered": daily_answered.get(d, 0),
        })

    # Task stats
    pending_tasks = sum(1 for t in tasks_list if t.get("status") == "pending")
    completed_tasks = sum(1 for t in tasks_list if t.get("status") == "completed")
    callbacks = sum(1 for t in tasks_list if t.get("type") == "callback" and t.get("status") == "pending")

    return {
        "total_calls": len(calls),
        "total_contacts": len(contacts),
        "total_duration_minutes": round(total_duration / 60, 1),
        "outcomes": dict(outcomes),
        "daily_calls": last_7,
        "pending_tasks": pending_tasks,
        "completed_tasks": completed_tasks,
        "pending_callbacks": callbacks,
        "answer_rate": round(sum(daily_answered.values()) / max(sum(daily_calls.values()), 1) * 100, 1),
    }


# ════════════════════════════════════════════════════════════
#  ENTRYPOINT
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Knight AI SDR starting on port {config.PORT}")
    logger.info(f"Dashboard: http://localhost:{config.PORT}")
    # Pass the app object directly to avoid accidental module-resolution collisions.
    # (When passing an app object, uvicorn can't use reload reliably.)
    uvicorn.run(app, host="0.0.0.0", port=config.PORT, reload=False)
