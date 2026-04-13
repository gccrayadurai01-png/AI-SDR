"""
AI SDR Server — Clean single-file version.
Telnyx speak → transcription → Claude → speak → loop.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx as _httpx
import telnyx
import anthropic
import apollo_client
from anthropic import AsyncAnthropic
from fastapi import BackgroundTasks, Body, FastAPI, Request, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
import campaign as campaign_lib
from email_sequences import (
    router as email_sequences_router,
    email_delivery_ready,
    smtp_ready,
    start_email_scheduler,
    stop_email_scheduler,
    test_email_delivery,
    test_smtp_connection,
)
from email_oauth import router as email_oauth_router, oauth_connection_status
from knowledge_base import get_full_knowledge
from sdr_agent import join_streamed_reply_parts, pop_first_speakable_chunk
from campaign import signal_call_ended, start_campaign, pause_campaign, resume_campaign, stop_campaign, normalize_phone, prospect_display_name
from post_call_email import resolve_prospect_email, run_post_call_followup_email
from prospect_import import parse_csv_bytes, parse_xlsx_bytes
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
from telnyx_handler import (
    format_telnyx_exception,
    run_telnyx_diagnostics,
    speak_on_call,
    start_transcription,
    stop_transcription,
    estimate_tts_playback_seconds,
    parse_call_transcription_event,
    should_emit_transcription_reply,
    extract_call_control_id_from_body,
    normalize_telnyx_event_type,
)

# ─── Setup ───────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

# ─── Pre-cached filler audio (ElevenLabs Anthony voice, generated at startup) ───
_filler_audio_cache: dict[str, str] = {}  # filler_text -> base64 MP3

from contextlib import asynccontextmanager

@asynccontextmanager
async def _app_lifespan(app_instance: FastAPI):
    start_email_scheduler()
    yield
    stop_email_scheduler()

app = FastAPI(title="AI SDR", version="3.0", lifespan=_app_lifespan)
STATIC = Path(__file__).parent / "static"
if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

app.include_router(email_sequences_router)
app.include_router(email_oauth_router)

# ─── Telnyx client (refreshes when .env changes via Save Settings) ───
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

# ─── State ───────────────────────────────────────────
active_calls: dict = {}          # call_control_id → call record
conversations: dict = {}         # call_control_id → [{"role":..., "content":...}]
answered_calls: set = set()      # prevent duplicate call.answered
speaking_now: set = set()        # calls where AI is currently speaking


# ─── Helpers ─────────────────────────────────────────
ASSISTANT_ID = "assistant-0a0eb33a-5fc0-4dd1-b366-8f0f1432be42"

# Pre-call research cache: "name|company" -> research string
# Populated BEFORE calling via UI or campaign start. Zero latency on call.
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
        log.info("Researched %s @ %s: %d chars", name, company, len(research))
        return research
    except Exception as e:
        log.warning("Research failed for %s @ %s: %s", name, company, e)
        fallback = f"PROSPECT INFO:\n- {name}, {title} at {company}"
        _prospect_research_cache[key] = fallback
        return fallback


def get_cached_research(name: str, company: str) -> str:
    """Get cached research — returns empty string if not yet researched."""
    return _prospect_research_cache.get(_research_key(name, company), "")


def get_opening_line(name: str = "there", title: str = "", company: str = "") -> str:
    s = load_script()
    sdr = s.get("sdr_name", "Alex")
    co = s.get("company_name", "our company")
    tmpl = s.get("opening_line", "Hey {name}, this is {sdr_name} from {company} -- did I catch you at a bad time?")
    line = tmpl.replace("{name}", name).replace("{sdr_name}", sdr).replace("{company}", co)
    line = line.replace("{title}", title).replace("{prospect_company}", company)
    return line


def get_system_prompt() -> str:
    """Short system prompt — keeps LLM fast. Knowledge goes in message_history per-call."""
    s = load_script()
    sdr = s.get("sdr_name", "Alex")
    co = s.get("company_name", "Your Company")
    obj = s.get("call_objective", "Book a 15-minute discovery call")
    vp = s.get("value_proposition", "")
    extra = s.get("additional_instructions", "")

    return f"""You={sdr}, SDR at {co}. LIVE phone call.
Goal:{obj}. Offer:{vp[:150]}

VOICE STYLE — sound like a REAL human on the phone:
- Add natural filler sounds: "hmm", "uh", "so yeah", "um", "you know", "right right"
- Leave small pauses with "..." between thoughts: "So... I was curious about..."
- Keep sentences SHORT. 1-2 max per turn. Break up longer thoughts with pauses.
- Sometimes slightly rephrase mid-sentence: "We help with— well actually, we manage your SaaS tools"
- Mirror their energy. If they're casual, be casual. If they're formal, match it.
- Sound slightly imperfect — real people don't speak in perfect sentences.
- Use casual connectors: "yeah so", "honestly", "I mean", "look"
- NEVER sound scripted or robotic. NEVER reveal you are AI.

RULES: 1 question per turn. Ask ONE thing, then WAIT.
Flow: rapport→soft questions→discovery→pitch only after clear pain→book meeting.
Objections: Busy=offer callback. Not interested=ask how they handle it now. Has tool=ask what's missing. No budget=we save money month 1.
{extra[:200]}""".strip()


def get_knowledge_message_history() -> list[dict]:
    """Return knowledge as message_history — loaded once at call start, not every LLM turn."""
    knowledge = get_full_knowledge()
    if not knowledge:
        return []
    return [
        {"role": "user", "content": f"[INTERNAL — never read aloud]\n{knowledge}"},
        {"role": "assistant", "content": "Got it, I know the product details."},
    ]


def sync_assistant_to_script():
    """Push current script config to the Telnyx AI Assistant."""
    try:
        instructions = get_system_prompt()
        _get_tx().ai.assistants.update(
            assistant_id=ASSISTANT_ID,
            instructions=instructions,
        )
        log.info("✅ Synced script to Telnyx AI Assistant")
    except Exception as e:
        log.error(f"Failed to sync assistant: {e}")

    # Tune assistant for MINIMUM latency via raw HTTP PATCH
    try:
        import httpx
        api_key = config.TELNYX_API_KEY
        if api_key:
            r = httpx.patch(
                f"https://api.telnyx.com/v2/ai/assistants/{ASSISTANT_ID}",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "instructions": instructions,
                    "model": "anthropic/claude-haiku-4-5",
                    # Fastest interruption + response settings
                    "interruption_settings": {
                        "enable": True,
                        "start_speaking_plan": {
                            "wait_seconds": 0.05,
                            "transcription_endpointing_plan": {
                                "on_punctuation_seconds": 0.2,
                                "on_no_speech_seconds": 0.6,
                            },
                        },
                    },
                    "voice_settings": {
                        "voice": f"ElevenLabs.eleven_multilingual_v2.{config.ELEVENLABS_VOICE_ID}",
                        "api_key_ref": config.ELEVENLABS_API_KEY_REF,
                        "voice_speed": 0.9,
                        "similarity_boost": 0.5,
                        "style": 0.0,
                        "use_speaker_boost": True,
                    },
                },
                timeout=15.0,
            )
            if r.status_code < 300:
                log.info("✅ Assistant tuned: wait=0.1s, turbo TTS, speed=1.1x")
            else:
                log.warning("Assistant PATCH returned %s: %s", r.status_code, r.text[:200])
    except Exception as e:
        log.warning("Assistant latency tune failed (non-fatal): %s", e)


def _startup_sync_assistant():
    """Sync assistant on server startup so voice + instructions are always current."""
    try:
        sync_assistant_to_script()
    except Exception as e:
        log.warning("Startup assistant sync failed (non-fatal): %s", e)


async def _precache_filler_audio():
    """Pre-generate all filler phrases via ElevenLabs at startup. Cached = instant playback during calls."""
    import httpx
    config.reload_secrets()
    voice_id = config.ELEVENLABS_VOICE_ID
    api_key = config.ELEVENLABS_API_KEY
    if not voice_id or not api_key:
        log.warning("ElevenLabs not configured — filler audio not pre-cached")
        return

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {"xi-api-key": api_key, "Content-Type": "application/json", "Accept": "audio/mpeg"}
    # Use turbo model + smaller mp3 for fastest possible filler playback
    params = {"output_format": "mp3_22050_32"}

    async with httpx.AsyncClient(timeout=30.0) as ac:
        for phrase in config.PHONE_FILLER_UTTERANCES:
            try:
                r = await ac.post(url, json={
                    "text": phrase,
                    "model_id": "eleven_turbo_v2_5",  # Turbo = fastest generation
                }, headers=headers, params=params)
                r.raise_for_status()
                import base64
                b64 = base64.b64encode(r.content).decode("ascii")
                _filler_audio_cache[phrase] = b64
                log.info("✅ Cached filler: \"%s\" (%d bytes)", phrase, len(r.content))
            except Exception as e:
                log.warning("Failed to cache filler \"%s\": %s", phrase, e)

    log.info("🎤 Pre-cached %d/%d filler phrases in Anthony's voice (turbo)", len(_filler_audio_cache), len(config.PHONE_FILLER_UTTERANCES))


@app.on_event("startup")
async def on_startup():
    """Sync assistant + pre-cache filler audio on every server start."""
    _startup_sync_assistant()
    await _precache_filler_audio()


# Persistent Claude client (no recreation per turn = faster)
_claude_client: AsyncAnthropic | None = None
_claude_key: str = ""


def _get_claude() -> AsyncAnthropic:
    global _claude_client, _claude_key
    config.reload_secrets()
    key = config.ANTHROPIC_API_KEY or ""
    if _claude_client is None or key != _claude_key:
        _claude_client = AsyncAnthropic(api_key=key)
        _claude_key = key
    return _claude_client


async def ask_claude(cc_id: str, prospect_text: str) -> str:
    """Stream Claude response for minimum latency."""
    conv = conversations.setdefault(cc_id, [])
    conv.append({"role": "user", "content": prospect_text})

    try:
        client = _get_claude()
        sys_text = get_system_prompt()
        model = config.phone_reply_model()
        t0 = asyncio.get_event_loop().time()

        # Use streaming for faster time-to-first-token
        full_text = ""
        async with client.messages.stream(
            model=model,
            max_tokens=config.ANTHROPIC_MAX_TOKENS_REPLY,
            system=sys_text,
            messages=conv,
        ) as stream:
            async for chunk in stream.text_stream:
                full_text += chunk

        elapsed = asyncio.get_event_loop().time() - t0
        log.info("⚡ Claude %s replied in %.1fs (%d tokens)", model, elapsed, len(full_text.split()))
        reply = full_text.strip()
        # Clean wrapping quotes
        if len(reply) >= 2 and reply[0] in '"\'':
            if reply[-1] == reply[0]:
                reply = reply[1:-1].strip()
        conv.append({"role": "assistant", "content": reply})
        return reply
    except Exception as e:
        log.error(f"Claude error: {e}")
        return "Sorry, bad connection on my end. Can I call you back tomorrow?"


async def iter_claude_reply_chunks(cc_id: str, prospect_text: str):
    """Append user, stream assistant text in speakable chunks (does not append assistant)."""
    conv = conversations.setdefault(cc_id, [])
    conv.append({"role": "user", "content": prospect_text})
    client = _get_claude()
    sys_text = get_system_prompt()
    system_blocks = [
        {"type": "text", "text": sys_text, "cache_control": {"type": "ephemeral"}},
    ]
    buffer = ""

    async def _run(sys: str | list):
        nonlocal buffer
        async with client.messages.stream(
            model=config.phone_reply_model(),
            max_tokens=config.ANTHROPIC_MAX_TOKENS_REPLY,
            system=sys,
            messages=conv,
        ) as stream:
            async for chunk in stream.text_stream:
                buffer += chunk
                while True:
                    spoken, buffer = pop_first_speakable_chunk(buffer)
                    if spoken:
                        yield spoken
                    else:
                        break

    try:
        async for part in _run(system_blocks):
            yield part
    except Exception as e:
        log.error(f"Claude stream chunks: {e}")
        buffer = ""
        async for part in _run(sys_text):
            yield part
    if buffer.strip():
        yield buffer.strip()


async def iter_claude_stream_after_user(cc_id: str):
    """Like iter_claude_reply_chunks but user message is already the last message in conv."""
    conv = conversations.setdefault(cc_id, [])
    client = _get_claude()
    sys_text = get_system_prompt()
    system_blocks = [
        {"type": "text", "text": sys_text, "cache_control": {"type": "ephemeral"}},
    ]
    buffer = ""

    async def _run(sys: str | list):
        nonlocal buffer
        async with client.messages.stream(
            model=config.phone_reply_model(),
            max_tokens=config.ANTHROPIC_MAX_TOKENS_REPLY,
            system=sys,
            messages=conv,
        ) as stream:
            async for chunk in stream.text_stream:
                buffer += chunk
                while True:
                    spoken, buffer = pop_first_speakable_chunk(buffer)
                    if spoken:
                        yield spoken
                    else:
                        break

    try:
        async for part in _run(system_blocks):
            yield part
    except Exception as e:
        log.error(f"Claude stream chunks: {e}")
        buffer = ""
        async for part in _run(sys_text):
            yield part
    if buffer.strip():
        yield buffer.strip()


async def _backup_start_transcription_after_speak(cc_id: str, wait_seconds: float) -> None:
    """Safety net: if call.playback.ended / call.speak.ended never arrives, start transcription after estimated TTS duration."""
    try:
        await asyncio.sleep(wait_seconds)
        if cc_id not in active_calls or active_calls[cc_id].get("state") == "ended":
            return
        # Only start if we're still in speaking_now (meaning the .ended webhook didn't fire)
        if cc_id in speaking_now:
            speaking_now.discard(cc_id)
            await start_transcription(cc_id)
            log.info("Backup start_transcription (no .ended webhook after %.1fs) cc_id=%s", wait_seconds, cc_id)
    except Exception:
        log.exception("Backup start_transcription failed cc_id=%s", cc_id)


def _speak_polly_fast(cc_id: str, text: str) -> None:
    """Speak via Telnyx Polly — INSTANT, no external API call. Best for low latency."""
    _get_tx().calls.actions.speak(
        call_control_id=cc_id,
        payload=text,
        voice=config.TELNYX_SPEAK_VOICE or "AWS.Polly.Matthew-Neural",
        language=config.TELNYX_SPEAK_LANGUAGE or "en-US",
        service_level="premium",
    )


async def _bg_start_call_with_opener(cc_id: str, greeting: str) -> None:
    """Speak opener via ElevenLabs direct (Anthony voice) then start transcription."""
    try:
        try:
            _get_tx().calls.actions.start_recording(call_control_id=cc_id, format="mp3", channels="single")
        except Exception:
            pass

        await speak_on_call(cc_id, greeting)
        log.info("✅ Opener spoken (ElevenLabs): %s", greeting[:60])

        est = estimate_tts_playback_seconds(greeting)
        asyncio.create_task(_backup_start_transcription_after_speak(cc_id, est + 1.0))

    except Exception:
        log.exception("Opener failed cc_id=%s", cc_id)
        speaking_now.discard(cc_id)
        try:
            await start_transcription(cc_id)
        except Exception:
            pass


def _start_ai_assistant_for_call(cc_id: str, greeting: str = "") -> None:
    """Start Telnyx AI Assistant for speech-to-speech conversation."""
    config.reload_secrets()
    ai_kwargs: dict[str, Any] = {
        "call_control_id": cc_id,
        "assistant": {"id": ASSISTANT_ID},
        "interruption_settings": {"enable": True},
        "transcription": {"model": "distil-whisper/distil-large-v2"},
    }
    if greeting:
        ai_kwargs["greeting"] = greeting
    # Configure ElevenLabs voice
    voice_id = config.ELEVENLABS_VOICE_ID
    api_key_ref = config.ELEVENLABS_API_KEY_REF
    if voice_id and api_key_ref:
        ai_kwargs["voice"] = f"ElevenLabs.{config.ELEVENLABS_MODEL_ID or 'eleven_multilingual_v2'}.{voice_id}"
        ai_kwargs["voice_settings"] = {"type": "elevenlabs", "api_key_ref": api_key_ref}
    _get_tx().calls.actions.start_ai_assistant(**ai_kwargs)
    log.info("✅ AI Assistant started (speech-to-speech) for %s", cc_id)


# Track filler playback state per call
_filler_playing: dict[str, bool] = {}  # cc_id -> True if filler is currently playing
_last_filler_time: dict[str, float] = {}  # cc_id -> timestamp of last filler played


async def _play_filler_for_ai_assistant(cc_id: str) -> None:
    """Play a pre-cached filler phrase instantly while AI Assistant thinks."""
    import time
    import random

    # Don't spam fillers — min 3s gap
    now = time.time()
    last = _last_filler_time.get(cc_id, 0)
    if now - last < 3.0:
        return

    if not _filler_audio_cache:
        return

    # Pick a random cached filler
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
        log.info("🗣️ AI-ASST filler: \"%s\" — INSTANT", phrase)
    except Exception as e:
        log.debug("Filler playback failed (non-fatal): %s", e)


def _stop_filler_if_playing(cc_id: str) -> None:
    """Stop filler playback when AI Assistant starts speaking."""
    if _filler_playing.get(cc_id):
        try:
            _get_tx().calls.actions.stop_playback(call_control_id=cc_id)
        except Exception:
            pass
        _filler_playing[cc_id] = False


async def _bg_opening_line(cc_id: str, greeting: str) -> None:
    """Fallback: speak opener via ElevenLabs direct + start transcription pipeline."""
    try:
        await asyncio.sleep(0.02)
        try:
            _get_tx().calls.actions.start_recording(call_control_id=cc_id, format="mp3", channels="single")
        except Exception:
            pass
        await speak_on_call(cc_id, greeting)
        log.info("Opening spoken (fallback) — voice: %s", config.telnyx_speak_voice_effective())
        est = estimate_tts_playback_seconds(greeting)
        asyncio.create_task(_backup_start_transcription_after_speak(cc_id, est + 1.0))
    except Exception:
        log.exception("Opening TTS failed cc_id=%s", cc_id)
        speaking_now.discard(cc_id)
        try:
            await start_transcription(cc_id)
        except Exception:
            pass


async def _resume_listen_fallback_server(cc_id: str, reply: str) -> None:
    try:
        await asyncio.sleep(estimate_tts_playback_seconds(reply))
    finally:
        speaking_now.discard(cc_id)
        try:
            if cc_id in active_calls and active_calls[cc_id].get("state") != "ended":
                await start_transcription(cc_id)
        except Exception:
            log.exception("listen fallback start_transcription failed cc_id=%s", cc_id)


async def _bg_transcription_turn(cc_id: str, prospect_text: str) -> None:
    reply = ""
    try:
        import time
        t_start = time.time()

        # Stop transcription in background (DON'T await — it's slow)
        asyncio.create_task(stop_transcription(cc_id))

        rec = active_calls.get(cc_id)
        if rec:
            rec.setdefault("transcript", []).append({"role": "prospect", "text": prospect_text})

        # ── STEP 1: Play filler INSTANTLY (pre-cached, zero latency) ──
        filler = config.phone_think_filler_phrase()
        cached_b64 = _filler_audio_cache.get(filler)
        if cached_b64 and config.should_play_think_filler(prospect_text):
            try:
                _get_tx().calls.actions.start_playback(
                    call_control_id=cc_id,
                    playback_content=cached_b64,
                    audio_type="mp3",
                )
                _filler_playing[cc_id] = True
                log.info("🗣️ Filler: \"%s\" — INSTANT (%.0fms after speech)", filler, (time.time() - t_start) * 1000)
            except Exception:
                pass

        # ── STEP 2: Claude Haiku in parallel (runs while filler plays) ──
        reply = await ask_claude(cc_id, prospect_text)
        t_claude = time.time()
        log.info("⚡ Claude done in %.0fms: \"%s\"", (t_claude - t_start) * 1000, reply[:60])

        if rec:
            rec["transcript"].append({"role": "agent", "text": reply})
            # Persist transcript to disk immediately (not just on hangup)
            save_call(rec)

        # ── STEP 3: Stop filler + speak reply ──
        if _filler_playing.get(cc_id):
            try:
                _get_tx().calls.actions.stop_playback(call_control_id=cc_id)
            except Exception:
                pass
            _filler_playing[cc_id] = False
            await asyncio.sleep(0.05)  # Tiny gap so stop registers

        # Reply via ElevenLabs turbo (Anthony voice)
        await speak_on_call(cc_id, reply)
        log.info("🔊 Total turn: %.0fms (heard→reply playing)", (time.time() - t_start) * 1000)

    except Exception:
        log.exception("Transcription reply failed cc_id=%s", cc_id)
        try:
            await speak_on_call(cc_id, "Sorry, bad connection. Can I try you back tomorrow?")
        except Exception:
            pass
    finally:
        _filler_playing.pop(cc_id, None)
        asyncio.create_task(_resume_listen_fallback_server(cc_id, reply))


# ═══════════════════════════════════════════════════════
#  POST-CALL AI INSIGHTS
# ═══════════════════════════════════════════════════════
async def _generate_call_insights(cc_id: str) -> None:
    """After call ends, use Claude to analyze transcript and generate insights."""
    try:
        rec = active_calls.get(cc_id)
        if not rec:
            # Try loading from storage
            rec = get_call_by_control_id(cc_id)
        if not rec:
            return

        transcript = rec.get("transcript", [])
        if not transcript or len(transcript) < 2:
            # Too short to analyze
            insights = {
                "summary": "Call was too short for meaningful analysis.",
                "outcome": "no_conversation",
                "sentiment": "neutral",
                "action_items": [],
                "key_points": [],
                "objections": [],
                "next_step": "Retry call later",
                "interest_level": 0,
            }
            update_call(cc_id, insights=insights)
            if cc_id in active_calls:
                active_calls[cc_id]["insights"] = insights
            return

        # Format transcript for Claude
        transcript_text = "\n".join(
            f"{'AI SDR' if t['role']=='agent' else 'Prospect'}: {t['text']}"
            for t in transcript
        )

        prompt = f"""Analyze this sales call transcript. Return ONLY valid JSON (no markdown, no code blocks).

TRANSCRIPT:
{transcript_text}

Return this exact JSON structure:
{{"summary": "2-3 sentence summary of the call",
"outcome": "one of: meeting_booked, callback_scheduled, interested, not_interested, no_answer, voicemail, gatekeeper, hung_up",
"sentiment": "one of: very_positive, positive, neutral, negative, very_negative",
"interest_level": 0-100,
"action_items": ["list of follow-up actions needed"],
"key_points": ["key things discussed or learned"],
"objections": ["any objections the prospect raised"],
"next_step": "recommended next action",
"prospect_pain_points": ["pain points mentioned by prospect"],
"buying_signals": ["any positive buying signals detected"]}}"""

        client = AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
        response = await client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        # Clean markdown code blocks if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

        insights = json.loads(raw)
        update_call(cc_id, insights=insights)
        if cc_id in active_calls:
            active_calls[cc_id]["insights"] = insights
        log.info("📊 Call insights generated for %s: outcome=%s, interest=%s%%",
                 cc_id[:20], insights.get("outcome"), insights.get("interest_level"))
    except Exception as e:
        log.exception("Failed to generate call insights for %s: %s", cc_id, e)


# ═══════════════════════════════════════════════════════
#  FRONTEND
# ═══════════════════════════════════════════════════════
@app.get("/")
async def dashboard():
    f = STATIC / "index.html"
    return FileResponse(str(f)) if f.exists() else JSONResponse({"status": "running"})


# ═══════════════════════════════════════════════════════
#  API ENDPOINTS (for dashboard)
# ═══════════════════════════════════════════════════════
@app.get("/api/health")
async def health():
    return {"status": "ok", "active_calls": len(active_calls), "base_url": config.APP_BASE_URL}

@app.get("/api/status")
async def status():
    config.reload_secrets()
    flags = config.dashboard_connection_flags()
    return {
        "telnyx": flags["telnyx"],
        "deepgram": flags["deepgram"],
        "anthropic": flags["anthropic"],
        "apollo": flags["apollo"],
        "email": flags["email"],
        "anthropic_model": config.ANTHROPIC_MODEL,
        "telnyx_phone": config.TELNYX_PHONE_NUMBER or "",
        "telnyx_connection": config.TELNYX_CONNECTION_ID or "",
        "base_url": config.APP_BASE_URL,
        "env_file": str(config._ENV_FILE),
    }

@app.get("/api/test/apollo")
async def test_apollo():
    config.reload_secrets()
    if not config.APOLLO_API_KEY:
        return JSONResponse(
            {"ok": False, "error": "APOLLO_API_KEY is not set"},
            status_code=400,
        )
    try:
        return await apollo_client.test_connection()
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=502)


@app.get("/api/test/anthropic")
async def test_anthropic():
    config.reload_secrets()
    if not config.ANTHROPIC_API_KEY:
        return JSONResponse(
            {"ok": False, "error": "ANTHROPIC_API_KEY is not set"},
            status_code=400,
        )
    try:
        c = AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
        await c.messages.create(
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
        "telnyx_key_set": config.env_file_nonempty("TELNYX_API_KEY"),
        "deepgram_key_set": config.env_file_nonempty("DEEPGRAM_API_KEY"),
        "anthropic_key_set": config.env_file_nonempty("ANTHROPIC_API_KEY"),
        "apollo_key_set": config.env_file_nonempty("APOLLO_API_KEY"),
        "phone_number": config.TELNYX_PHONE_NUMBER or "",
        "connection_id": config.TELNYX_CONNECTION_ID or "",
        "base_url": config.APP_BASE_URL or "",
        "anthropic_max_tokens_reply": config.ANTHROPIC_MAX_TOKENS_REPLY,
        "anthropic_live_model": config.ANTHROPIC_LIVE_MODEL or "",
        "anthropic_phone_model_effective": config.phone_reply_model(),
        "env_file": str(config._ENV_FILE),
        "telnyx_speak_voice": config.TELNYX_SPEAK_VOICE or "",
        "elevenlabs_voice_id": config.ELEVENLABS_VOICE_ID or "",
        "elevenlabs_model_id": config.ELEVENLABS_MODEL_ID or "",
        "elevenlabs_api_key_ref": config.ELEVENLABS_API_KEY_REF or "",
        "elevenlabs_direct_first": bool(config.ELEVENLABS_DIRECT_FIRST),
        "elevenlabs_key_set": bool(config.ELEVENLABS_API_KEY),
        "tts_mode_summary": config.tts_mode_description(),
        "tts_voice_effective": config.telnyx_speak_voice_effective(),
        "smtp_host": (config.SMTP_HOST or "").strip(),
        "smtp_port": int(config.SMTP_PORT or 587),
        "smtp_user": (config.SMTP_USER or "").strip(),
        "smtp_password_set": config.env_file_nonempty("SMTP_PASSWORD"),
        "email_from": (config.EMAIL_FROM or "").strip(),
        "smtp_use_tls": bool(config.SMTP_USE_TLS),
        "email_automation_enabled": bool(config.EMAIL_AUTOMATION_ENABLED),
        "post_call_followup_email_enabled": bool(config.POST_CALL_FOLLOWUP_EMAIL_ENABLED),
        "post_call_followup_delay_sec": int(config.POST_CALL_FOLLOWUP_DELAY_SEC or 300),
        "email_provider": (config.EMAIL_PROVIDER or "smtp").strip().lower(),
        "sendgrid_api_key_set": config.env_file_nonempty("SENDGRID_API_KEY"),
        "resend_api_key_set": config.env_file_nonempty("RESEND_API_KEY"),
        "mailgun_api_key_set": config.env_file_nonempty("MAILGUN_API_KEY"),
        "mailgun_domain": (config.MAILGUN_DOMAIN or "").strip(),
        "mailgun_api_base": (config.MAILGUN_API_BASE or "").strip(),
        "smtp_ready": smtp_ready(),
        "email_ready": email_delivery_ready(),
        **oauth_connection_status(),
    }

@app.post("/api/settings")
async def save_settings_endpoint(request: Request):
    import re
    body = await request.json()
    env_path = Path(__file__).parent / ".env"
    env_text = env_path.read_text(encoding="utf-8") if env_path.exists() else ""

    def set_env(text: str, key: str, val: str) -> str:
        if re.search(rf"^{re.escape(key)}\s*=", text, re.MULTILINE):
            return re.sub(rf"^({re.escape(key)}\s*=).*", rf"\g<1>{val}", text, flags=re.MULTILINE)
        return text + f"\n{key}={val}"

    def patch_env_line(text: str, key: str, val: str) -> str:
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

    mapping = {"telnyx_api_key": "TELNYX_API_KEY", "deepgram_api_key": "DEEPGRAM_API_KEY",
               "anthropic_api_key": "ANTHROPIC_API_KEY", "apollo_api_key": "APOLLO_API_KEY",
               "telnyx_phone_number": "TELNYX_PHONE_NUMBER",
               "telnyx_connection_id": "TELNYX_CONNECTION_ID", "app_base_url": "APP_BASE_URL",
               "anthropic_max_tokens_reply": "ANTHROPIC_MAX_TOKENS_REPLY",
               "anthropic_live_model": "ANTHROPIC_LIVE_MODEL"}
    for field, env_key in mapping.items():
        val = body.get(field, "").strip()
        if val:
            env_text = set_env(env_text, env_key, val)
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
    return {"status": "saved", "note": "Keys saved and reloaded."}


@app.post("/api/settings/test-smtp")
async def settings_test_smtp():
    return test_smtp_connection()


@app.post("/api/settings/test-email")
async def settings_test_email():
    return test_email_delivery()

@app.get("/api/script")
async def get_script_endpoint():
    return load_script()

@app.post("/api/script")
async def save_script_endpoint(request: Request):
    save_script(await request.json())
    sync_assistant_to_script()
    return {"status": "saved", "synced": True}

@app.get("/api/calls/history")
async def history():
    return {"total": len(load_calls()), "calls": load_calls()}


@app.post("/api/calls/cleanup-stale")
async def cleanup_stale_calls(payload: dict = Body(default={})):
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


# ═══════════════════════════════════════════════════════
#  OUTBOUND CALL
# ═══════════════════════════════════════════════════════
class CallRequest(BaseModel):
    to_number: str
    prospect_name: str = "there"
    company: str = ""
    title: str = ""
    notes: str = ""
    prospect_email: str = ""

@app.post("/call/outbound")
async def outbound(req: CallRequest):
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
    log.info(f"DIAL: {to} ({req.prospect_name})")
    try:
        result = _get_tx().calls.dial(
            connection_id=config.TELNYX_CONNECTION_ID,
            to=to,
            from_=config.TELNYX_PHONE_NUMBER,
            webhook_url=f"{config.APP_BASE_URL}/webhooks/telnyx",
            webhook_url_method="POST",
        )
    except Exception as e:
        log.exception("Telnyx dial failed")
        raise HTTPException(status_code=502, detail=format_telnyx_exception(e)) from e
    cc_id = result.data.call_control_id
    em = (req.prospect_email or "").strip()
    rec = {
        "call_control_id": cc_id, "call_leg_id": result.data.call_leg_id,
        "state": "initiated", "to": to, "prospect_name": req.prospect_name,
        "company": req.company, "title": req.title, "notes": req.notes,
        "prospect_email": em, "transcript": [],
        "started_at": datetime.utcnow().isoformat(), "recording_url": None,
    }
    active_calls[cc_id] = rec
    save_call(rec)
    return JSONResponse({"status": "initiated", "call_control_id": cc_id,
                         "call_leg_id": result.data.call_leg_id, "to": to})


async def _remove_ended_call_after(cc_id: str, delay: float = 180.0) -> None:
    try:
        await asyncio.sleep(delay)
        rec = active_calls.get(cc_id)
        if rec and rec.get("state") == "ended":
            active_calls.pop(cc_id, None)
    except Exception:
        log.exception("_remove_ended_call_after failed cc_id=%s", cc_id)


# ═══════════════════════════════════════════════════════
#  TELNYX WEBHOOK — the core call loop
# ═══════════════════════════════════════════════════════
@app.post("/webhooks/telnyx")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    body = await request.json()
    data = body.get("data", {})
    raw_etype = data.get("event_type", "?")
    etype = normalize_telnyx_event_type(raw_etype)
    pl = data.get("payload", {}) or {}
    cc_id = extract_call_control_id_from_body(body) or (pl.get("call_control_id") or "").strip() or None

    log.info(f"EVENT: {etype} | {cc_id}")

    try:
        # ── INBOUND CALL ───────────────────────────
        if etype == "call.initiated":
            if pl.get("direction") == "incoming":
                _get_tx().calls.actions.answer(call_control_id=cc_id)
                active_calls[cc_id] = {"call_control_id": cc_id, "state": "answered",
                    "transcript": [], "started_at": datetime.utcnow().isoformat()}
                save_call(active_calls[cc_id])

        # ── CALL ANSWERED → Telnyx AI Assistant (speech-to-speech) ─────
        elif etype == "call.answered":
            if cc_id in answered_calls:
                return JSONResponse({"status": "ok"})
            answered_calls.add(cc_id)

            if cc_id in active_calls:
                active_calls[cc_id]["state"] = "answered"
                update_call(cc_id, state="answered")

            rec = active_calls.get(cc_id) or {}
            name = rec.get("prospect_name", "there")
            title = rec.get("title", "") or ""
            company = rec.get("company", "") or ""
            greeting = get_opening_line(name, title=title, company=company)

            # Get pre-cached research — NO API call here, instant lookup
            research = get_cached_research(name, company)

            # Build message_history: knowledge base + prospect research
            msg_history = get_knowledge_message_history()
            if research:
                msg_history.append({
                    "role": "user",
                    "content": f"[BRIEFING — don't read aloud]\n{research}\nUse naturally. Personalize questions to their role/company.",
                })
                msg_history.append({
                    "role": "assistant",
                    "content": "Got it.",
                })
                log.info("Injected research (%d chars)", len(research))

            conversations[cc_id] = []
            if not cc_id:
                log.error("call.answered missing call_control_id; payload keys=%s", list(pl.keys()))
            else:
                log.info("Call answered for %s — starting speech-to-speech", name)
                # Recording in background — don't block greeting
                asyncio.get_event_loop().run_in_executor(
                    None, lambda: _get_tx().calls.actions.start_recording(call_control_id=cc_id, format="mp3", channels="single")
                )
                try:
                    config.reload_secrets()
                    ai_kwargs = {
                        "call_control_id": cc_id,
                        "assistant": {"id": ASSISTANT_ID},
                        "greeting": greeting,
                        "transcription": {"model": "distil-whisper/distil-large-v2"},
                        "interruption_settings": {"enable": True},
                    }
                    # Pass ElevenLabs voice settings inline (critical for audio output)
                    voice_id = config.ELEVENLABS_VOICE_ID
                    api_key_ref = config.ELEVENLABS_API_KEY_REF
                    if voice_id and api_key_ref:
                        ai_kwargs["voice"] = f"ElevenLabs.{config.ELEVENLABS_MODEL_ID or 'eleven_multilingual_v2'}.{voice_id}"
                        ai_kwargs["voice_settings"] = {"type": "elevenlabs", "api_key_ref": api_key_ref}
                    if msg_history:
                        ai_kwargs["message_history"] = msg_history
                    _get_tx().calls.actions.start_ai_assistant(**ai_kwargs)
                    active_calls.setdefault(cc_id, {})["ai_assistant"] = True
                    log.info("AI Assistant started — greeting: %s", greeting[:60])
                except Exception as e:
                    log.exception("AI Assistant failed: %s — falling back", e)
                    active_calls.setdefault(cc_id, {})["ai_assistant"] = False
                    speaking_now.add(cc_id)
                    background_tasks.add_task(_bg_start_call_with_opener, cc_id, greeting)

        # ── SPEAK/PLAYBACK ENDED → start transcription (only for fallback pipeline) ────
        elif etype in ("call.speak.ended", "call.playback.ended"):
            speaking_now.discard(cc_id)
            if active_calls.get(cc_id, {}).get("ai_assistant"):
                pass  # AI Assistant handles everything
            elif cc_id in active_calls and active_calls.get(cc_id, {}).get("state") != "ended":
                log.info("SPEAK/PLAYBACK DONE → start transcription")
                try:
                    await start_transcription(cc_id)
                except Exception as e:
                    log.error(f"Transcription start failed: {e}")

        # ── TRANSCRIPTION → prospect spoke ─────────
        elif etype == "call.transcription":
            text, is_final, cc_resolved = parse_call_transcription_event(body)
            cc_id = cc_resolved or cc_id

            # LOG THE RAW PAYLOAD so we can debug empty transcriptions
            log.info("TRANSCRIPTION RAW: text=%r is_final=%s cc=%s payload_keys=%s",
                     text, is_final, cc_id, list((data.get("payload") or {}).keys()) if isinstance(data.get("payload"), dict) else "N/A")

            if not cc_id:
                return JSONResponse({"status": "ok"})
            if not text:
                return JSONResponse({"status": "ok"})

            rec = active_calls.get(cc_id, {})

            if not should_emit_transcription_reply(cc_id, text, is_final):
                return JSONResponse({"status": "ok"})

            log.info(f"HEARD: \"{text}\"")

            if cc_id in speaking_now:
                return JSONResponse({"status": "ok"})

            speaking_now.add(cc_id)
            asyncio.create_task(_bg_transcription_turn(cc_id, text))
            return JSONResponse({"status": "ok"})

        # ── AI ASSISTANT EVENTS (speech-to-speech transcript capture + filler) ───
        elif etype in ("call.ai_assistant.transcription", "call.ai_assistant.partial_transcription"):
            # Capture transcript from AI Assistant for call history
            ai_text = (pl.get("text") or pl.get("transcript") or "").strip()
            ai_role = pl.get("role", "")  # "user" or "assistant"
            if ai_text and cc_id:
                rec = active_calls.get(cc_id)
                if rec:
                    tlist = rec.setdefault("transcript", [])
                    role = "prospect" if ai_role == "user" else "agent"
                    tlist.append({"role": role, "text": ai_text})
                    # Persist transcript to disk immediately
                    save_call(rec)
                    if ai_role == "user":
                        log.info(f"👂 AI-ASST HEARD: \"{ai_text}\"")
                        # Play filler INSTANTLY while AI thinks (pre-cached Anthony voice)
                        if config.PHONE_THINK_FILLER and config.should_play_think_filler(ai_text):
                            asyncio.create_task(_play_filler_for_ai_assistant(cc_id))
                    else:
                        log.info(f"🤖 AI-ASST SAID: \"{ai_text}\"")
                        # Stop filler when AI starts speaking
                        _stop_filler_if_playing(cc_id)

        # ── AI ASSISTANT SPEAKING → stop filler immediately ───
        elif etype in ("call.ai_assistant.speaking_started", "call.ai_assistant.response_started"):
            if cc_id:
                _stop_filler_if_playing(cc_id)
                log.info("🎙️ AI Assistant speaking — filler stopped")

        elif etype == "call.ai_assistant.error":
            log.error("AI Assistant error: %s", pl)
            # Fall back to ElevenLabs direct pipeline if AI Assistant crashes mid-call
            if cc_id and cc_id in active_calls:
                active_calls[cc_id]["ai_assistant"] = False
                log.info("Falling back to ElevenLabs direct pipeline for %s", cc_id)
                speaking_now.add(cc_id)
                background_tasks.add_task(_bg_opening_line, cc_id, "Hey sorry about that — I had a little glitch. Where were we?")

        # ── RECORDING SAVED ────────────────────────
        elif etype == "call.recording.saved":
            url = pl.get("recording_urls", {}).get("mp3") or \
                  pl.get("public_recording_urls", {}).get("mp3")
            if url and cc_id in active_calls:
                active_calls[cc_id]["recording_url"] = url
                update_call(cc_id, recording_url=url)

        # ── TELNYX CONVERSATION INSIGHTS ──────────
        elif etype == "call.conversation_insights.generated":
            insights_data = pl.get("insights") or pl
            if cc_id:
                log.info("📊 Conversation insights received for %s", cc_id)
                rec = active_calls.get(cc_id)
                if rec:
                    rec["telnyx_insights"] = insights_data
                update_call(cc_id, telnyx_insights=insights_data)

        # ── CONVERSATION ENDED → generate AI insights ──
        elif etype == "call.conversation.ended":
            if cc_id:
                log.info("💬 Conversation ended for %s — generating insights", cc_id)
                asyncio.create_task(_generate_call_insights(cc_id))

        # ── HANGUP ─────────────────────────────────
        elif etype == "call.hangup":
            hang_cc = extract_call_control_id_from_body(body) or cc_id
            if not hang_cc:
                log.error(
                    "call.hangup: missing call_control_id; payload keys=%s",
                    list(pl.keys()) if isinstance(pl, dict) else pl,
                )
                return JSONResponse({"status": "ok"})
            log.info("CALL ENDED %s", hang_cc)
            signal_call_ended(hang_cc)
            ended_at = datetime.utcnow().isoformat()
            rec = active_calls.get(hang_cc)
            duration_seconds = None
            transcript: list = []
            if rec and rec.get("state") == "ended":
                return JSONResponse({"status": "ok"})
            if rec:
                rec["state"] = "ended"
                rec["ended_at"] = ended_at
                try:
                    started = datetime.fromisoformat(rec.get("started_at", ""))
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
                log.warning("call.hangup: no calls.json row for %s", hang_cc)
            if rec and (rec.get("prospect_email") or "").strip():
                update_call(hang_cc, prospect_email=(rec.get("prospect_email") or "").strip())
            merged_row = dict(get_call_by_control_id(hang_cc) or {})
            if rec:
                merged_row.update(rec)
            if config.POST_CALL_FOLLOWUP_EMAIL_ENABLED and resolve_prospect_email(merged_row):
                asyncio.create_task(run_post_call_followup_email(hang_cc))
            asyncio.create_task(_remove_ended_call_after(hang_cc))
            answered_calls.discard(hang_cc)
            speaking_now.discard(hang_cc)
            conversations.pop(hang_cc, None)
            _filler_playing.pop(hang_cc, None)
            _last_filler_time.pop(hang_cc, None)

    except Exception as e:
        log.exception("WEBHOOK ERROR: %s", e)

    return JSONResponse({"status": "ok"})


# ═══════════════════════════════════════════════════════
#  CALL MANAGEMENT
# ═══════════════════════════════════════════════════════
@app.get("/calls")
async def list_calls():
    return JSONResponse({"total": len(active_calls), "calls": active_calls})

@app.get("/calls/{cc_id}")
async def get_call(cc_id: str):
    call = active_calls.get(cc_id) or get_call_by_control_id(cc_id)
    if not call:
        raise HTTPException(404, "Not found")
    return JSONResponse(call)

@app.delete("/calls/{cc_id}")
async def end_call(cc_id: str):
    _get_tx().calls.actions.hangup(call_control_id=cc_id)
    return JSONResponse({"status": "hung up"})


# ═══════════════════════════════════════════════════════
#  CAMPAIGN — auto-dial prospects one by one
# ═══════════════════════════════════════════════════════
class CampaignStartBody(BaseModel):
    prospects: list[dict[str, Any]]
    spacing_seconds: float = Field(30.0, ge=0, le=86400)


@app.post("/api/campaign/start")
async def campaign_start(body: CampaignStartBody):
    if not body.prospects:
        raise HTTPException(400, "No prospects in queue")
    if campaign_lib.state.status == "running":
        raise HTTPException(409, "Campaign already running")

    # Pre-research ALL prospects before first call
    research_tasks = []
    for p in body.prospects[:50]:
        name = prospect_display_name(p)
        research_tasks.append(research_prospect(name, p.get("title", ""), p.get("company", "")))
    if research_tasks:
        await asyncio.gather(*research_tasks, return_exceptions=True)
        log.info("Pre-researched %d prospects for campaign", len(research_tasks))

    async def dial_one(p: dict[str, Any]) -> str | None:
        phone = normalize_phone(p.get("phone"))
        if not phone:
            return None
        name = prospect_display_name(p)
        company = p.get("company", "")
        title = p.get("title", "")

        try:
            result = _get_tx().calls.dial(
                connection_id=config.TELNYX_CONNECTION_ID,
                to=phone,
                from_=config.TELNYX_PHONE_NUMBER,
                webhook_url=f"{config.APP_BASE_URL}/webhooks/telnyx",
                webhook_url_method="POST",
            )
            cc_id = result.data.call_control_id
            rec = {
                "call_control_id": cc_id, "call_leg_id": result.data.call_leg_id,
                "state": "initiated", "to": phone, "prospect_name": name,
                "company": company, "title": title, "notes": p.get("notes", ""),
                "prospect_email": str(p.get("email") or "").strip(),
                "transcript": [],
                "started_at": datetime.utcnow().isoformat(), "recording_url": None,
            }
            active_calls[cc_id] = rec
            save_call(rec)
            return cc_id
        except Exception as e:
            log.error("Campaign dial failed: %s", format_telnyx_exception(e))
            return None

    ok = start_campaign(body.prospects, body.spacing_seconds, dial_one)
    if not ok:
        raise HTTPException(409, "Could not start campaign")
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
        "status": st.status, "index": st.index, "total": st.total,
        "spacing_seconds": st.spacing_seconds, "last_error": st.last_error,
        "last_to": st.last_to, "skipped": st.skipped,
    }


# ═══════════════════════════════════════════════════════
#  APOLLO SEARCH
# ═══════════════════════════════════════════════════════
@app.post("/api/apollo/search")
async def apollo_search(request: Request):
    body = await request.json()
    try:
        data = await apollo_client.search_people(
            page=body.get("page", 1),
            per_page=body.get("per_page", 25),
            q_keywords=body.get("q_keywords") or None,
            person_titles=body.get("person_titles") or None,
            person_locations=body.get("person_locations") or None,
            organization_locations=body.get("organization_locations") or None,
            person_seniorities=body.get("person_seniorities") or None,
            organization_num_employees_ranges=body.get("organization_num_employees_ranges") or None,
            q_organization_domains_list=body.get("q_organization_domains_list") or None,
            include_similar_titles=body.get("include_similar_titles", True),
        )
        people = data.get("people") or []
        rows = []
        for p in people:
            org = p.get("organization") or {}
            rows.append({
                "apollo_person_id": p.get("id", ""),
                "first_name": p.get("first_name", ""),
                "last_name": p.get("last_name", ""),
                "title": p.get("title", ""),
                "company": org.get("name") if isinstance(org, dict) else "",
                "phone": p.get("phone_number", "") or "",
                "email": p.get("email", "") or "",
                "notes": "",
            })
        return {
            "people": rows,
            "pagination": data.get("pagination", {}),
            "total_entries": data.get("pagination", {}).get("total_entries"),
        }
    except Exception as e:
        log.error(f"Apollo search failed: {e}")
        raise HTTPException(502, str(e))


# ═══════════════════════════════════════════════════════
#  PROSPECT FILE IMPORT
# ═══════════════════════════════════════════════════════
@app.post("/api/prospects/import-file")
async def prospects_import_file(file: UploadFile = File(...)):
    raw = await file.read()
    name = (file.filename or "").lower()
    if name.endswith(".csv"):
        rows, warnings = parse_csv_bytes(raw)
    elif name.endswith(".xlsx"):
        rows, warnings = parse_xlsx_bytes(raw)
    else:
        raise HTTPException(400, "Upload a .csv or .xlsx file")
    return {"rows": rows, "warnings": warnings, "count": len(rows)}


# ═══════════════════════════════════════════════════════
#  TASKS
# ═══════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════
#  PROSPECT RESEARCH (pre-call)
# ═══════════════════════════════════════════════════════
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
    for p in prospects[:50]:  # Max 50 at once
        tasks.append(research_prospect(
            p.get("name", ""), p.get("title", ""), p.get("company", "")
        ))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    ok = sum(1 for r in results if isinstance(r, str) and r)
    return {"status": "ok", "researched": ok, "total": len(prospects)}


@app.get("/api/prospect/research-cache")
async def api_research_cache():
    """Return all cached research."""
    return {"cache": _prospect_research_cache, "count": len(_prospect_research_cache)}


# ═══════════════════════════════════════════════════════
#  AI SCRIPT SUGGEST + KNOWLEDGE
# ═══════════════════════════════════════════════════════
@app.post("/api/knowledge/upload")
async def upload_knowledge_doc(file: UploadFile = File(...)):
    """Upload a document (txt, pdf, csv, docx) to the AI knowledge base."""
    from knowledge_base import UPLOADED_DOCS_KNOWLEDGE
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
            # Limit to 3000 chars per doc to keep prompt manageable
            doc_entry = f"--- Document: {fname} ---\n{text[:3000]}"
            UPLOADED_DOCS_KNOWLEDGE.append(doc_entry)
            # Re-sync assistant with new knowledge
            sync_assistant_to_script()
            log.info("Knowledge doc uploaded: %s (%d chars)", fname, len(text))
            return {"status": "ok", "filename": fname, "chars": len(text[:3000]),
                    "total_docs": len(UPLOADED_DOCS_KNOWLEDGE)}
        else:
            return JSONResponse({"error": "Could not extract text from file"}, status_code=400)
    except Exception as e:
        log.error(f"Knowledge upload failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/knowledge")
async def get_knowledge():
    """Return current knowledge base contents."""
    from knowledge_base import UPLOADED_DOCS_KNOWLEDGE, CLOUDFUZE_KNOWLEDGE
    return {
        "website_knowledge": CLOUDFUZE_KNOWLEDGE[:500] + "...",
        "uploaded_docs": len(UPLOADED_DOCS_KNOWLEDGE),
        "doc_names": [d.split("\n")[0] for d in UPLOADED_DOCS_KNOWLEDGE],
    }


@app.delete("/api/knowledge")
async def clear_knowledge():
    """Clear all uploaded documents from knowledge base."""
    from knowledge_base import UPLOADED_DOCS_KNOWLEDGE
    UPLOADED_DOCS_KNOWLEDGE.clear()
    sync_assistant_to_script()
    return {"status": "ok", "message": "Knowledge base cleared"}


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
- discovery_questions: array of 4-5 short questions to uncover pain points
- objections: object with keys: not_interested, send_email, call_back, have_solution, no_budget (each a short 1-sentence response)
- booking_phrase: a natural way to ask for a meeting
- opening_line: a casual, permission-based opener using {{name}}, {{sdr_name}}, {{company}} placeholders

Return ONLY valid JSON, no markdown."""

    try:
        client = AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
        resp = await client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        # Parse JSON from response
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        suggestion = json.loads(text)
        return {"suggestion": suggestion}
    except Exception as e:
        log.error(f"AI suggest failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ═══════════════════════════════════════════════════════
#  AUTO-CREATE TASK ON CALL END (detect "call me back")
# ═══════════════════════════════════════════════════════
def check_callback_request(transcript: list, prospect_name: str, phone: str, company: str, cc_id: str):
    """Check if prospect asked for a callback and auto-create a task."""
    callback_phrases = ["call me back", "call back", "try me again", "call later", "not a good time", "busy right now", "call tomorrow", "call next week"]
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
                    "notes": f"Prospect said: \"{entry.get('text', '')}\"",
                    "status": "pending",
                    "call_control_id": cc_id,
                    "created_at": datetime.utcnow().isoformat(),
                }
                save_task(task)
                log.info(f"Auto-task created for callback: {prospect_name}")
                return
    return


# ═══════════════════════════════════════════════════════
#  START
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    import os
    import uvicorn

    _reload = os.environ.get("UVICORN_RELOAD", "").strip().lower() in ("1", "true", "yes")
    log.info(f"AI SDR starting on port {config.PORT}")
    log.info(f"Dashboard: http://localhost:{config.PORT}")
    if not _reload:
        log.info("Uvicorn reload is OFF (stable on Windows). Set UVICORN_RELOAD=1 to enable auto-reload.")
    uvicorn.run("server:app", host="0.0.0.0", port=config.PORT, reload=_reload)
