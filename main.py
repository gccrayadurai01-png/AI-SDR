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
from knowledge_base import get_full_knowledge, UPLOADED_DOCS_KNOWLEDGE, CLOUDFUZE_KNOWLEDGE, _persist_uploaded_docs
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
    """Structured sales prompt grounded in script content — prevents hallucination."""
    s = load_script()
    sdr = s.get("sdr_name", "Alex")
    co = s.get("company_name", "Your Company")
    obj = s.get("call_objective", "Book a 15-minute discovery call")
    persona = s.get("target_persona", "IT / SaaS decision makers")
    vp = s.get("value_proposition", "")
    pain = s.get("pain_points", "")
    website = s.get("company_website", "")
    products = s.get("product_services", "")
    advantage = s.get("competitive_advantage", "")
    call_flow = s.get("call_flow", "")
    end_goal = s.get("end_goal", obj)
    discovery = s.get("discovery_questions", "")
    objections = s.get("objection_handling", "")
    booking = s.get("booking_phrase", "")
    kb_notes = s.get("knowledge_base_notes", "")

    technique = s.get("sales_technique", "sandler")
    technique_block = _get_technique_prompt(technique)

    # Build structured sections
    sections = []
    sections.append(f"You={sdr}, SDR at {co}. LIVE cold call. You are a real human, never say you're AI.")
    sections.append(f"CALL OBJECTIVE: {obj}")
    sections.append(f"END GOAL: {end_goal}")
    sections.append(f"TARGET: {persona}")

    sections.append("""IMPORTANT: The greeting has ALREADY been spoken. Do NOT repeat it. Do NOT say hi again. Wait for the prospect to respond, then continue the conversation.""")

    sections.append("""ANTI-HALLUCINATION RULES:
- ONLY discuss what is in your PRODUCT KNOWLEDGE below. NEVER invent features, pricing, stats, or claims.
- If asked something you don't know, say: "Great question — I'd want to get you the exact details on that. Let me set up a quick call with our specialist."
- NEVER make up case studies, customer names, percentages, or ROI numbers unless explicitly provided below.
- If the prospect asks about pricing, say: "Pricing depends on your specific setup — that's exactly what we'd cover in the demo."
- Stick to the script sections below. Do not freelance.""")

    sections.append("""VOICE RULES:
- Reply in 1-2 sentences max. NEVER more than 20 words per turn.
- Respond INSTANTLY. No pauses, no thinking delays.
- ONE question at a time. Wait for their answer.
- Sound natural: "yeah", "got it", "makes sense", "totally".
- After greeting, WAIT for them to speak first. Then respond.""")

    if vp:
        sections.append(f"VALUE PROPOSITION:\n{vp}")

    if products:
        sections.append(f"PRODUCTS/SERVICES (only mention these):\n{products}")

    if pain:
        sections.append(f"PAIN POINTS TO PROBE:\n{pain}")

    if advantage:
        sections.append(f"COMPETITIVE ADVANTAGE:\n{advantage}")

    if website:
        sections.append(f"COMPANY WEBSITE: {website} — refer prospects here for more details.")

    if discovery:
        sections.append(f"DISCOVERY QUESTIONS (use these):\n{discovery}")

    if objections:
        sections.append(f"OBJECTION HANDLING:\n{objections}")

    if booking:
        sections.append(f"BOOKING: {booking}")

    if call_flow:
        sections.append(f"CALL FLOW:\n{call_flow}")

    if kb_notes:
        sections.append(f"KNOWLEDGE BASE:\n{kb_notes}")

    # Include uploaded-document snippets (persisted) so the Assistant grounds on them too.
    if UPLOADED_DOCS_KNOWLEDGE:
        uploaded_blob = "\n\n".join(UPLOADED_DOCS_KNOWLEDGE)[:3500]
        sections.append(f"UPLOADED DOCUMENTS (cite these facts only, do not invent):\n{uploaded_blob}")

    sections.append(technique_block)

    sections.append("""END CALL: If they say stop/not interested after one probe: "Totally understand, appreciate your time. Have a great day!" then END. Say goodbye naturally and stop.""")

    return "\n\n".join(sections).strip()


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
    s = load_script()

    parts = []

    # Script-based knowledge
    vp = s.get("value_proposition", "")
    products = s.get("product_services", "")
    pain = s.get("pain_points", "")
    advantage = s.get("competitive_advantage", "")
    website = s.get("company_website", "")
    kb_notes = s.get("knowledge_base_notes", "")
    co = s.get("company_name", "Knight")

    if vp:
        parts.append(f"VALUE PROPOSITION: {vp}")
    if products:
        parts.append(f"PRODUCTS/SERVICES: {products}")
    if pain:
        parts.append(f"PAIN POINTS: {pain}")
    if advantage:
        parts.append(f"COMPETITIVE EDGE: {advantage}")
    if website:
        parts.append(f"WEBSITE: {website}")
    if kb_notes:
        parts.append(f"ADDITIONAL KNOWLEDGE: {kb_notes}")

    if full and full.strip():
        parts.append(full[:1500])

    if not parts:
        return f"PRODUCT KNOWLEDGE for {co} — weave into conversation naturally.\n\n{vp}" if vp else ""

    header = f"PRODUCT KNOWLEDGE for {co} — ONLY use this information. NEVER invent details.\n\n"
    return (header + "\n\n".join(parts))[:2000]


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
                    # 300s user-silence cap: long enough that a prospect
                    # thinking / checking a calendar / putting us briefly on
                    # hold does NOT cause Telnyx to silently terminate the
                    # AI session (which leaves the call alive with zero
                    # audio — the dreaded "mid-call silence"). Voicemail
                    # protection is still handled by answering_machine_
                    # detection at dial time.
                    "user_idle_timeout_secs": 300,
                    "max_duration_secs": 1800,
                },
                "interruption_settings": {
                    "enable": True,
                    "start_speaking_plan": {
                        # Require ~1.5s of continuous prospect speech before
                        # barging in. 0.5s was too twitchy — a cough or
                        # background noise would pause the AI mid-sentence
                        # and the resume webhook sometimes never arrived,
                        # leaving the AI stuck.
                        "wait_seconds": 1.5,
                    },
                },
            }
            voice_id = config.ELEVENLABS_VOICE_ID
            api_key_ref = config.ELEVENLABS_API_KEY_REF
            if voice_id and api_key_ref:
                # CRITICAL: `type: elevenlabs` must be present at the top of
                # voice_settings — without it Telnyx stores the settings but
                # does NOT route audio through ElevenLabs, so calls go silent
                # even though `start_ai_assistant` returns 200. This was the
                # direct cause of the "no voice on the call" incident.
                patch_body["voice_settings"] = {
                    "type": "elevenlabs",
                    "voice": f"ElevenLabs.eleven_multilingual_v2.{voice_id}",
                    "api_key_ref": api_key_ref,
                    "voice_speed": 0.9,
                    # ── Stable voice profile — reduces artifacts/choppiness ──
                    # similarity_boost too high (>0.8) + style > 0 + no stability
                    # floor produces vocal drift / fabling / choppy artifacts over
                    # generations longer than ~1 min. ElevenLabs-recommended stable
                    # values: stability 0.5+, similarity 0.75, style 0.0.
                    "stability": 0.55,
                    "similarity_boost": 0.75,
                    "style": 0.0,
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
    # Only match unambiguous call-ending phrases. Do NOT include "not
    # interested"/"stop calling" etc. — prospects say those mid-conversation
    # ("I'm not interested YET, tell me more") and the AI handles them via
    # objection flow. Hard-stop phrases live in _HARD_STOP_PATTERNS below.
    r'\b(goodbye|good\s*bye|bye\s*bye|talk\s*soon|have\s*a\s*great\s*(day|one|week)|'
    r'take\s*care|appreciate\s*your\s*time|thanks\s*for\s*your\s*time|'
    r'nice\s*talking|nice\s*chatting|have\s*a\s*good\s*one|'
    r'catch\s*you\s*later|speak\s*soon|cheers|so\s*long)\b',
    _re.IGNORECASE,
)

# Track pending auto-hangup tasks so we can cancel if conversation continues
_auto_hangup_tasks: dict[str, asyncio.Task] = {}

# Track last speech time per call for silence detection
_last_speech_time: dict[str, float] = {}
_silence_watchdog_tasks: dict[str, asyncio.Task] = {}
_ai_assistant_started: set[str] = set()  # cc_ids where AI Assistant was started — NEVER auto-hangup these
_ai_user_turn_times: dict[str, list[float]] = {}  # timestamps of prospect speaking turns
_ai_agent_turn_times: dict[str, list[float]] = {}  # timestamps of AI speaking turns
_voicemail_watchdog_tasks: dict[str, asyncio.Task] = {}  # cc_id -> watchdog task


def _is_goodbye(text: str) -> bool:
    """Check if text contains a natural conversation-ending phrase."""
    return bool(_GOODBYE_PATTERNS.search(text or ""))


# ── Hard stops from prospect: definitive "end the call NOW" signals ──
# These are unambiguous disinterest/rejection. When the PROSPECT says one
# of these, we should wrap up quickly — not keep probing. Tighter than
# _GOODBYE_PATTERNS so ordinary speech never false-matches.
_HARD_STOP_PATTERNS = _re.compile(
    # Keep ONLY unambiguous "never call me again" phrases. Removed soft
    # brush-offs like "not interested right now" / "not interested at
    # all" / "I'm not interested" — prospects use these mid-conversation
    # ("I'm not interested in X, but tell me about Y") and a match used
    # to schedule an auto-hangup that silenced the call.
    r"(?:\bstop\s+calling\s+me\b|\bplease\s+stop\s+calling\b|"
    r"\btake\s+me\s+off\s+(?:your|the)\s+list\b|"
    r"\bremove\s+me\s+from\s+(?:your|the)\s+list\b|"
    r"\bdo\s+not\s+call\s+(?:me\s+)?(?:again|back)\b|"
    r"\bdon'?t\s+call\s+(?:me\s+)?(?:again|back)\b|"
    r"\bnever\s+call\s+(?:me\s+)?again\b|"
    r"\bleave\s+me\s+alone\b)",
    _re.IGNORECASE,
)


def _is_hard_stop(text: str) -> bool:
    """Prospect issued an unambiguous 'end call' signal."""
    return bool(_HARD_STOP_PATTERNS.search(text or ""))


# ── Booking-confirmed signals: meeting is locked, wrap up politely ──
# Fires on AI or prospect saying clear meeting-booked phrases. After
# these, exchange farewell and end.
_BOOKING_CONFIRMED_PATTERNS = _re.compile(
    # Require explicit, hard-to-false-match booking confirmation language.
    # Removed "you're set for" and "looking forward to our/the…" — these
    # fire on ordinary meeting discussion ("are you set for the demo?")
    # and used to silence live calls by scheduling an auto-hangup.
    r"(?:\bcalendar\s+invite\s+(?:sent|is\s+on\s+the\s+way|is\s+coming)|"
    r"\bi'?ll\s+send\s+(?:you\s+)?(?:the\s+|a\s+|an\s+)?calendar\s+invite\b|"
    r"\bmeeting\s+is\s+(?:booked|confirmed|scheduled)\b|"
    r"\bwe'?re\s+(?:booked|confirmed)\s+for\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|next\s+week)\b|"
    r"\bsee\s+you\s+(?:on\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|next\s+week)\b|"
    r"\btalk\s+to\s+you\s+(?:on\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|next\s+week)\b)",
    _re.IGNORECASE,
)


def _is_booking_confirmed(text: str) -> bool:
    """Meeting has been explicitly locked in — exchange farewell and end."""
    return bool(_BOOKING_CONFIRMED_PATTERNS.search(text or ""))


async def _silence_watchdog(cc_id: str):
    """Mid-call silence monitor + AI resurrection.

    Watches for the specific failure mode: AI Assistant went silent but the
    call is still alive (Telnyx killed the AI session on idle / interrupt /
    transient error, and the bot never spoke again). If no AGENT turn has
    been observed for ~25s, we attempt exactly ONE restart of the AI
    Assistant to wake it back up. NEVER hangs up.
    """
    try:
        nudged_at = 0.0
        while cc_id in active_calls and active_calls[cc_id].get("state") != "ended":
            await asyncio.sleep(10)
            rec = active_calls.get(cc_id) or {}
            if not rec.get("ai_assistant"):
                continue  # Already in TTS fallback — don't touch
            agent_turns = _ai_agent_turn_times.get(cc_id) or []
            last_agent = agent_turns[-1] if agent_turns else _ai_assistant_first_event.get(cc_id, 0)
            if not last_agent:
                continue
            idle = time.time() - last_agent
            # Only nudge once per call, only after real silence
            if idle > 25 and (time.time() - nudged_at) > 60 and not rec.get("_ai_nudge_done"):
                rec["_ai_nudge_done"] = True
                nudged_at = time.time()
                logger.warning("MID-CALL SILENCE: %s has been quiet %.0fs — restarting AI Assistant", cc_id, idle)
                name = rec.get("prospect_name") or "there"
                title = rec.get("prospect_title") or ""
                company = rec.get("company") or ""
                try:
                    # Schedule restart in background — reuses the same fast path
                    from fastapi import BackgroundTasks as _BT
                    asyncio.create_task(_start_ai_assistant_fast(cc_id, name, title, company, _BT()))
                except Exception as e:
                    logger.warning("Silence nudge restart failed: %s", e)
    except asyncio.CancelledError:
        pass
    finally:
        _silence_watchdog_tasks.pop(cc_id, None)


async def _auto_hangup_after_goodbye(cc_id: str):
    """Hang up call ~5s after a natural goodbye is detected.
    Critical for bulk dialing — prevents zombie calls after conversation ends."""
    try:
        await asyncio.sleep(5.0)
        if cc_id not in active_calls:
            return
        if active_calls[cc_id].get("state") == "ended":
            return
        # Only hang up if no new user speech arrived after goodbye (within last 4s)
        last_user = 0.0
        turns = _ai_user_turn_times.get(cc_id) or []
        if turns:
            last_user = turns[-1]
        if last_user and (time.time() - last_user) < 4.0:
            logger.info("Goodbye auto-hangup CANCELLED for %s — user spoke again", cc_id)
            return
        logger.info("Goodbye detected for %s — hanging up naturally", cc_id)
        try:
            await hangup_call(cc_id)
        except Exception as e:
            logger.warning("Goodbye hangup failed (non-fatal): %s", e)
    except asyncio.CancelledError:
        pass
    finally:
        _auto_hangup_tasks.pop(cc_id, None)


async def _voicemail_watchdog(cc_id: str):
    """DISABLED at the code level — we cannot reliably detect voicemail from
    webhooks because the Telnyx AI Assistant processes user speech internally
    and does not always emit call.ai_assistant.transcription events to us.

    Voicemail handling is delegated to Telnyx itself via:
      - answering_machine_detection on the outbound call
      - telephony_settings.user_idle_timeout_secs on the Assistant (90s)
      - telephony_settings.time_limit_secs (1800s hard cap)

    Kept as a stub so callers don't break."""
    return


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
    # Read directly from os.environ — works on Railway AND local
    telnyx_key  = os.environ.get("TELNYX_API_KEY", "").strip()
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    apollo_key  = os.environ.get("APOLLO_API_KEY", "").strip()
    elevenlabs_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    deepgram_key = os.environ.get("DEEPGRAM_API_KEY", "").strip()
    smtp_host   = os.environ.get("SMTP_HOST", "").strip()
    email_from  = os.environ.get("EMAIL_FROM", "").strip()
    sendgrid    = os.environ.get("SENDGRID_API_KEY", "").strip()
    return {
        "telnyx":    bool(telnyx_key and config.TELNYX_CONNECTION_ID and config.TELNYX_PHONE_NUMBER),
        "deepgram":  bool(deepgram_key),
        "anthropic": bool(anthropic_key),
        "apollo":    bool(apollo_key),
        "email":     bool((smtp_host and email_from) or sendgrid),
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
                raw = r.json()
                # Telnyx's /v2/ai/assistants/{id} may return the assistant at
                # the root level OR wrapped in {"data": {...}} depending on
                # SDK version / endpoint. Try both.
                data = raw.get("data") if isinstance(raw.get("data"), dict) else raw
                results["assistant"] = {
                    "ok": True,
                    "id": data.get("id"),
                    "model": data.get("model"),
                    "voice_settings": data.get("voice_settings"),
                    "instructions_len": len(data.get("instructions") or ""),
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
        "company_website": body.get("company_website", ""),
        "pain_points": body.get("pain_points", ""),
        "product_services": body.get("product_services", ""),
        "competitive_advantage": body.get("competitive_advantage", ""),
        "call_flow": body.get("call_flow", ""),
        "end_goal": body.get("end_goal", ""),
        "knowledge_base_notes": body.get("knowledge_base_notes", ""),
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
        "company_website": agent.get("company_website", ""),
        "pain_points": agent.get("pain_points", ""),
        "product_services": agent.get("product_services", ""),
        "competitive_advantage": agent.get("competitive_advantage", ""),
        "call_flow": agent.get("call_flow", ""),
        "end_goal": agent.get("end_goal", ""),
        "knowledge_base_notes": agent.get("knowledge_base_notes", ""),
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

    prompt = f"""You are an expert sales strategist. Analyze this company's website and create a COMPLETE AI SDR agent configuration.

Website URL: {url}
Website Content:
{text}

Generate a JSON object with ALL these fields:
- "name": Short agent name (e.g. "Enterprise Closer")
- "company_name": Company name from website
- "company_website": "{url}"
- "sdr_name": Professional first name for the AI SDR
- "call_objective": What the call should achieve
- "target_persona": Who to call (titles, company size, industry)
- "value_proposition": 2-3 sentence value prop from the website content ONLY
- "product_services": List the actual products/services from the website (bullet points, newline separated)
- "pain_points": 4-6 pain points this product solves (newline separated)
- "competitive_advantage": What makes this company different (from website only)
- "end_goal": Ultimate call outcome (e.g. "Book a 15-min demo")
- "call_flow": Numbered call flow steps (newline separated)
- "opening_line": Natural cold-call opener using {{name}}, {{sdr_name}}, {{company}} placeholders
- "discovery_questions": 5-7 qualifying questions (newline separated)
- "objection_handling": Handle objections: not_interested, send_email, have_solution, no_budget (newline separated)
- "booking_phrase": Natural meeting request
- "voicemail_message": 30-second voicemail using {{name}}, {{sdr_name}}, {{company}} placeholders
- "knowledge_base_notes": Key facts from the website the AI should know (features, integrations, use cases)

IMPORTANT: Only extract real information from the website. Do NOT invent features, pricing, or claims.

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
def _strip_briefing_from_transcript(turns: Any) -> list[dict]:
    """
    Defensive filter: remove internal [BRIEFING] + synthetic acknowledgements
    from stored transcripts so the UI never shows priming junk. Works on
    transcripts saved before the backend filter was added.
    """
    if not isinstance(turns, list):
        return []
    cleaned: list[dict] = []
    for t in turns:
        if not isinstance(t, dict):
            continue
        txt = (t.get("text") or "").strip()
        if not txt:
            continue
        upper = txt.upper()
        if "[BRIEFING]" in upper or upper.startswith("PROSPECT INFO") or upper.startswith("RESEARCH SUMMARY"):
            continue
        if txt.lower() in ("got it.", "got it", "okay.", "okay", "understood.", "understood"):
            continue
        cleaned.append(t)
    return cleaned


@app.get("/api/calls/history")
async def call_history():
    calls = load_calls()
    out: list[dict[str, Any]] = []
    for c in calls:
        if isinstance(c, dict):
            row = dict(c)
            # Strip briefing junk from stored transcripts (backward-compat fix
            # for rows saved before the fetcher was filtered).
            if row.get("transcript"):
                row["transcript"] = _strip_briefing_from_transcript(row.get("transcript"))
            row["summary_preview"] = _summary_preview_for_history(row)
            out.append(row)
        else:
            out.append(c)  # type: ignore[arg-type]
    return {"total": len(out), "calls": out}


@app.post("/api/admin/restore-recordings")
async def restore_all_recordings(request: Request):
    """
    Walk all historical calls and attempt to re-fetch + persist each
    recording from Telnyx. Recordings Telnyx still holds server-side
    (within the retention window) will be downloaded to local disk so
    the UI can play them forever after. Returns per-call status.

    Body: optional {\"limit\": N} to cap (default 500).
    """
    try:
        body = await request.json() if (await request.body()) else {}
    except Exception:
        body = {}
    limit = int((body or {}).get("limit", 500))

    calls = load_calls() or []
    results = {"total": 0, "already_local": 0, "restored": 0, "not_found_in_telnyx": 0, "failed": 0}
    details: list[dict] = []

    # Process newest-first (most likely to still be in retention)
    calls_sorted = sorted(
        calls,
        key=lambda c: (c.get("started_at") or "") if isinstance(c, dict) else "",
        reverse=True,
    )[:limit]

    for c in calls_sorted:
        if not isinstance(c, dict):
            continue
        cc = c.get("call_control_id")
        if not cc:
            continue
        results["total"] += 1
        disk = _recording_disk_path(cc)
        if disk.exists() and disk.stat().st_size > 0:
            results["already_local"] += 1
            continue
        try:
            fresh = await _refresh_telnyx_recording_url(cc, c)
            if not fresh:
                results["not_found_in_telnyx"] += 1
                details.append({"cc_id": cc[:30], "status": "not_in_telnyx"})
                continue
            await _persist_recording_to_disk(cc, fresh)
            if disk.exists() and disk.stat().st_size > 0:
                update_call(cc, recording_url=fresh, recording_local=True)
                results["restored"] += 1
                details.append({"cc_id": cc[:30], "status": "restored", "bytes": disk.stat().st_size})
            else:
                results["failed"] += 1
                details.append({"cc_id": cc[:30], "status": "download_failed"})
        except Exception as e:
            results["failed"] += 1
            details.append({"cc_id": cc[:30], "status": "error", "error": str(e)[:120]})

    logger.info("Recording restore complete: %s", results)
    return {"summary": results, "details": details[:100]}


@app.post("/api/calls/{call_control_id}/refresh-recording")
async def refresh_recording_endpoint(call_control_id: str):
    """
    Manual refresh: re-fetch a fresh Telnyx recording URL and persist to disk.
    Use from UI when a call's recording shows as broken.
    """
    rec = get_call_by_control_id(call_control_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Call not found")
    fresh = await _refresh_telnyx_recording_url(call_control_id, rec)
    if not fresh:
        raise HTTPException(status_code=404, detail="No Telnyx recording found for this call")
    await _persist_recording_to_disk(call_control_id, fresh)
    update_call(call_control_id, recording_url=fresh)
    return {"ok": True, "recording_url": fresh}


# ── Recording disk cache ─────────────────────────────────────
# Local persistent store for MP3 audio so playback never breaks when Telnyx
# pre-signed S3 URLs expire (~10 min). Files keyed by sanitized cc_id.
_RECORDINGS_DIR = Path(__file__).parent / "data" / "recordings"
_RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)


def _recording_disk_path(call_control_id: str) -> Path:
    # cc_ids contain ':' and '/' — sanitize for FS safety
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in call_control_id)
    return _RECORDINGS_DIR / f"{safe}.mp3"


async def _persist_recording_to_disk(call_control_id: str, url: str) -> None:
    """Fire-and-forget: download recording bytes to data/recordings/{cc}.mp3."""
    import httpx
    dest = _recording_disk_path(call_control_id)
    if dest.exists() and dest.stat().st_size > 0:
        return  # already saved
    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as ac:
            r = await ac.get(url)
            r.raise_for_status()
            dest.write_bytes(r.content)
            logger.info("Persisted recording to disk (%s bytes) for %s", len(r.content), call_control_id[:20])
            # Mark in DB so future proxy calls know to serve from disk
            update_call(call_control_id, recording_local=True)
            if call_control_id in active_calls:
                active_calls[call_control_id]["recording_local"] = True
    except Exception as e:
        logger.warning("Failed to persist recording for %s: %s", call_control_id[:20], e)


@app.get("/api/recordings/{call_control_id}")
async def proxy_recording(call_control_id: str):
    """
    Stream a call recording. Priority:
    1. Serve from local disk if we've cached it (permanent)
    2. Otherwise stream from Telnyx pre-signed URL (expires in ~10 min)
    3. If expired, try to refresh; if that fails, try to download it now
       and serve the fresh copy.
    """
    from fastapi.responses import StreamingResponse, FileResponse

    rec = get_call_by_control_id(call_control_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Call not found")

    # 1) Disk cache wins — serve directly from file.
    disk = _recording_disk_path(call_control_id)
    if disk.exists() and disk.stat().st_size > 0:
        return FileResponse(
            disk,
            media_type="audio/mpeg",
            headers={"Cache-Control": "public, max-age=86400", "Accept-Ranges": "bytes"},
        )

    url = rec.get("recording_url")
    if not url:
        raise HTTPException(status_code=404, detail="No recording available yet")

    # Telnyx recording URLs are typically pre-signed (query string carries
    # auth) — sending a Bearer header on top can cause S3 to 403. Only add
    # auth when the URL has NO query string AND is on api.telnyx.com.
    headers: dict[str, str] = {}
    needs_auth = ("?" not in url) and ("api.telnyx.com" in url) and bool(config.TELNYX_API_KEY)
    if needs_auth:
        headers["Authorization"] = f"Bearer {config.TELNYX_API_KEY}"

    # 2) Pre-flight HEAD — if URL expired, refresh from Telnyx recordings API.
    try:
        async with _httpx.AsyncClient(timeout=10.0) as probe:
            h = await probe.head(url, headers=headers, follow_redirects=True)
            if h.status_code in (401, 403, 404, 410):
                logger.info("Recording URL expired for %s (HEAD %s) — refreshing from Telnyx", call_control_id[:20], h.status_code)
                fresh = await _refresh_telnyx_recording_url(call_control_id, rec)
                if fresh:
                    url = fresh
                    rec["recording_url"] = url
                    update_call(call_control_id, recording_url=url)
                else:
                    raise HTTPException(status_code=410, detail="Recording URL expired and could not be refreshed")
    except HTTPException:
        raise
    except Exception as e:
        logger.debug("Recording HEAD probe failed (%s) — attempting direct GET", e)

    # 3) Download to disk FIRST (so future plays are instant), then stream it back.
    try:
        async with _httpx.AsyncClient(timeout=60.0, follow_redirects=True) as ac:
            r = await ac.get(url, headers=headers)
            r.raise_for_status()
            disk.write_bytes(r.content)
            update_call(call_control_id, recording_local=True)
            logger.info("Cached recording to disk on first play (%s bytes) for %s", len(r.content), call_control_id[:20])
        return FileResponse(
            disk,
            media_type="audio/mpeg",
            headers={"Cache-Control": "public, max-age=86400", "Accept-Ranges": "bytes"},
        )
    except Exception as e:
        logger.warning("Direct download failed for %s: %s — falling back to streaming proxy", call_control_id[:20], e)

    # Streaming fallback (if disk write failed for some reason)
    async def _iter() -> Any:
        async with _httpx.AsyncClient(timeout=60.0) as ac:
            async with ac.stream("GET", url, headers=headers, follow_redirects=True) as r:
                r.raise_for_status()
                total = 0
                async for chunk in r.aiter_bytes():
                    if chunk:
                        total += len(chunk)
                        yield chunk
                logger.info("Recording streamed: %s bytes for %s", total, call_control_id[:20])

    media = "audio/mpeg" if url.lower().endswith(".mp3") else "audio/wav" if url.lower().endswith(".wav") else "audio/mpeg"
    return StreamingResponse(
        _iter(),
        media_type=media,
        headers={"Cache-Control": "private, max-age=300", "Accept-Ranges": "bytes"},
    )


async def _refresh_telnyx_recording_url(call_control_id: str, rec: dict | None = None) -> str | None:
    """
    When an S3 pre-signed recording URL has expired, fetch a fresh one from
    Telnyx. Tries multiple filter shapes — Telnyx's recordings endpoint
    historically accepts call_leg_id, call_session_id, and (sometimes)
    call_control_id. We also fall back to page-scan when no filter matches.
    """
    import httpx
    api_key = config.TELNYX_API_KEY
    if not api_key:
        return None
    rec = rec or get_call_by_control_id(call_control_id) or {}
    call_leg_id = rec.get("call_leg_id") if isinstance(rec, dict) else None

    headers = {"Authorization": f"Bearer {api_key}"}
    attempts: list[dict] = []
    if call_leg_id:
        attempts.append({"filter[call_leg_id]": call_leg_id})
    attempts.append({"filter[call_control_id]": call_control_id})
    # Last-resort: scan first page and match on call_control_id / call_leg_id
    attempts.append({"page[size]": "50"})

    try:
        async with httpx.AsyncClient(timeout=15.0) as ac:
            for params in attempts:
                try:
                    r = await ac.get(
                        "https://api.telnyx.com/v2/recordings",
                        headers=headers,
                        params=params,
                    )
                    if r.status_code >= 300:
                        continue
                    recs = (r.json() or {}).get("data") or []
                    for item in recs:
                        if "page[size]" in params:
                            # Manual match — only accept items whose cc_id / leg_id agrees
                            if (item.get("call_control_id") != call_control_id and
                                (not call_leg_id or item.get("call_leg_id") != call_leg_id)):
                                continue
                        urls = {**(item.get("recording_urls") or {}), **(item.get("public_recording_urls") or {})}
                        fresh = urls.get("mp3") or urls.get("wav") or item.get("download_url")
                        if fresh:
                            logger.info(
                                "Refreshed Telnyx recording URL for %s (filter=%s)",
                                call_control_id[:20], list(params.keys())[0],
                            )
                            return fresh
                except Exception as e:
                    logger.debug("Refresh attempt %s failed: %s", params, e)
                    continue
    except Exception as e:
        logger.warning("Recording URL refresh failed for %s: %s", call_control_id[:20], e)
    return None


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
_campaign_start_lock: asyncio.Lock = asyncio.Lock()


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


def _mark_campaign_prospect(camp_id: str, phone: str, **fields: Any) -> None:
    """Update a prospect's per-call fields inside a campaign record.
    Matches on normalized phone."""
    target = normalize_phone(phone) or phone
    camps = _load_campaigns()
    changed = False
    for c in camps:
        if c.get("id") != camp_id:
            continue
        for p in c.get("prospects", []):
            pn = normalize_phone(p.get("phone")) or p.get("phone")
            if pn == target:
                p.update(fields)
                changed = True
                break
        break
    if changed:
        _save_campaigns(camps)


# ── DNC (Do-Not-Call) persistence ──
_DNC_FILE = Path(__file__).parent / "data" / "dnc.json"


def _load_dnc() -> set[str]:
    if _DNC_FILE.exists():
        try:
            data = json.loads(_DNC_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return {str(x) for x in data}
        except Exception:
            pass
    return set()


def _save_dnc(nums: set[str]) -> None:
    _DNC_FILE.parent.mkdir(parents=True, exist_ok=True)
    _DNC_FILE.write_text(json.dumps(sorted(nums), indent=2), encoding="utf-8")


_dnc_cache: set[str] = _load_dnc()


def is_dnc(phone: str) -> bool:
    n = normalize_phone(phone) or phone
    return n in _dnc_cache


def add_dnc(phone: str, reason: str = "") -> None:
    n = normalize_phone(phone) or phone
    if not n or n in _dnc_cache:
        return
    _dnc_cache.add(n)
    _save_dnc(_dnc_cache)
    logger.info("DNC ADDED: %s (reason=%s)", n, reason)


@app.get("/api/dnc")
async def get_dnc():
    return {"blocked": sorted(_dnc_cache)}


@app.post("/api/dnc/add")
async def post_dnc_add(body: dict):
    phone = str(body.get("phone") or "").strip()
    if not phone:
        raise HTTPException(400, "phone required")
    add_dnc(phone, reason=body.get("reason") or "manual")
    return {"status": "added", "phone": phone}


@app.post("/api/dnc/remove")
async def post_dnc_remove(body: dict):
    phone = str(body.get("phone") or "").strip()
    n = normalize_phone(phone) or phone
    if n in _dnc_cache:
        _dnc_cache.discard(n)
        _save_dnc(_dnc_cache)
    return {"status": "removed", "phone": n}


@app.get("/api/campaigns/list")
async def list_campaigns():
    camps = _load_campaigns()
    return [_recompute_campaign_outcomes(c) for c in camps if isinstance(c, dict)]


@app.get("/api/campaigns/history")
async def get_campaigns_history():
    """Backwards compat — returns same as list."""
    camps = _load_campaigns()
    return [_recompute_campaign_outcomes(c) for c in camps if isinstance(c, dict)]


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


def _recompute_campaign_outcomes(c: dict) -> dict:
    """Aggregate per-prospect outcomes into the campaign's outcomes dict for UI stats."""
    from collections import Counter
    counts: Counter = Counter()
    dialed = 0
    for p in (c.get("prospects") or []):
        if not isinstance(p, dict):
            continue
        if (p.get("status") or "").lower() in ("dialed", "completed"):
            dialed += 1
        oc = _normalize_outcome(p.get("outcome"))
        if oc and oc != "unknown":
            counts[oc] += 1
    c["outcomes"] = dict(counts)
    if c.get("status") not in ("running", "paused"):
        c["dialed"] = dialed
    return c


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
    _recompute_campaign_outcomes(c)
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
    # Serialize campaign starts — prevents two rapid start clicks from
    # racing on _active_campaign_id and each other's voice config.
    async with _campaign_start_lock:
        return await _start_named_campaign_unlocked(camp_id)


async def _start_named_campaign_unlocked(camp_id: str) -> bool:
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
        # DNC check: skip prospects on the persistent do-not-call list.
        if is_dnc(phone):
            logger.info("DNC: skipping %s — prospect previously marked not-interested", phone)
            _mark_campaign_prospect(camp_id, phone, status="skipped", outcome="dnc")
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
        cc_id = result.get("call_control_id")
        # Mark prospect status on the campaign record so the UI reflects progress.
        _mark_campaign_prospect(
            camp_id, phone,
            status="dialed",
            outcome="",
            call_control_id=cc_id,
            dialed_at=datetime.now().isoformat(),
        )
        return cc_id

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

    # Resume: pick up where we left off if campaign was paused/stopped.
    starting_index = int(c.get("current_index") or c.get("dialed") or 0)
    if starting_index >= len(prospects):
        starting_index = 0  # already done — allow a fresh run from the top
    if starting_index > 0:
        logger.info("Campaign %s: resuming from index %d/%d", camp_id, starting_index, len(prospects))
    ok = start_campaign(prospects, spacing, dial_one, starting_index=starting_index)
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


def _find_campaign_for_call(rec: dict) -> tuple[str, str]:
    """Return (campaign_id, campaign_name) for a call record, or ('','') if none."""
    phone = (rec.get("to") or rec.get("phone") or "").strip()
    if not phone:
        return "", ""
    for camp in _load_campaigns():
        for p in (camp.get("prospects") or []):
            if isinstance(p, dict) and (p.get("phone") or "").strip() == phone:
                return camp.get("id", ""), camp.get("name", "")
    return "", ""


def _ensure_task_for_outcome(call_control_id: str, rec: dict, insights: dict) -> None:
    """
    Create an appropriate follow-up task based on call outcome.
    Called after insights are generated. Idempotent — won't duplicate
    if a task already exists for this call_control_id.
    """
    try:
        outcome = _normalize_outcome(insights.get("outcome"))
        # Skip these — nothing to follow up on
        if outcome in ("no_answer", "voicemail", "not_interested", "do_not_call", "hung_up", "gatekeeper", "unknown", "no_conversation"):
            return
        # meeting_booked is handled by _ensure_task_for_meeting separately
        if outcome == "meeting_booked":
            return

        # Check for duplicate
        existing = load_tasks() or []
        for t in existing:
            if t.get("call_control_id") == call_control_id and t.get("type") in ("callback", "follow_up", "interested"):
                return

        task_type = "callback" if outcome == "callback_scheduled" else "follow_up"
        due = ""
        # Default due date: callback tomorrow, follow_up in 2 days
        from datetime import timedelta
        if outcome == "callback_scheduled":
            due = (datetime.utcnow() + timedelta(days=1)).date().isoformat()
        else:
            due = (datetime.utcnow() + timedelta(days=2)).date().isoformat()

        next_step = (insights.get("next_step") or "").strip()
        summary = (insights.get("summary") or "").strip()
        notes = next_step or f"Follow up on: {summary[:160]}" if summary else f"Follow up — outcome: {outcome.replace('_',' ')}"

        camp_id, camp_name = _find_campaign_for_call(rec)
        task = {
            "id": str(uuid.uuid4())[:8],
            "prospect_name": rec.get("prospect_name", ""),
            "phone": rec.get("to", ""),
            "company": rec.get("company", ""),
            "type": task_type,
            "outcome": outcome,  # carry outcome forward so Tasks page can filter by it
            "campaign_id": camp_id,
            "campaign_name": camp_name,
            "due_date": due,
            "notes": notes,
            "status": "pending",
            "call_control_id": call_control_id,
            "created_at": datetime.utcnow().isoformat(),
        }
        save_task(task)
        logger.info("Auto-task (%s) created from outcome=%s for %s", task_type, outcome, rec.get("prospect_name", "?"))
    except Exception:
        logger.exception("_ensure_task_for_outcome failed for %s", call_control_id[:20])


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
        camp_id, camp_name = _find_campaign_for_call(rec)
        task = {
            "id": str(uuid.uuid4())[:8],
            "prospect_name": prospect_name,
            "phone": phone,
            "company": company,
            "type": "meeting",
            "outcome": "meeting_booked",  # so Tasks page Outcome column + filter work
            "campaign_id": camp_id,
            "campaign_name": camp_name,
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
#  PHASE 3 — CAMPAIGN SCHEDULER + EMAIL FOLLOWUP + CALLBACK AUTO-DIAL
# ════════════════════════════════════════════════════════════
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

# Common TZs surfaced in the UI.
COMMON_TIMEZONES = [
    "UTC",
    "America/New_York",      # EST/EDT
    "America/Chicago",       # CST/CDT
    "America/Denver",        # MST/MDT
    "America/Los_Angeles",   # PST/PDT
    "America/Phoenix",
    "America/Toronto",
    "America/Sao_Paulo",
    "Europe/London",         # GMT/BST
    "Europe/Berlin",
    "Europe/Paris",
    "Europe/Madrid",
    "Europe/Athens",
    "Europe/Moscow",
    "Asia/Dubai",
    "Asia/Karachi",
    "Asia/Kolkata",          # IST
    "Asia/Singapore",
    "Asia/Tokyo",
    "Asia/Shanghai",
    "Australia/Sydney",
    "Pacific/Auckland",
]


@app.get("/api/timezones")
async def list_timezones():
    """Surface a curated TZ list for the campaign-schedule UI."""
    return {"timezones": COMMON_TIMEZONES}


# ──────────────────────────────────────────────────────────────
# Email (SMTP) — used for meeting invites + SDR notifications
# ──────────────────────────────────────────────────────────────
SMTP_CONFIG_FILE = Path(__file__).parent / "data" / "smtp_config.json"


def _load_smtp_json() -> dict:
    """User-saved SMTP config (overrides env when present)."""
    try:
        if SMTP_CONFIG_FILE.exists():
            d = json.loads(SMTP_CONFIG_FILE.read_text(encoding="utf-8"))
            if isinstance(d, dict):
                return d
    except Exception:
        pass
    return {}


def _save_smtp_json(d: dict) -> dict:
    keep = {k: (d.get(k) or "") for k in (
        "host", "port", "user", "pass", "from", "sdr_notify", "use_tls"
    )}
    keep["port"] = int(str(keep["port"]).strip() or 587) if str(keep["port"]).strip() else 587
    keep["use_tls"] = bool(d.get("use_tls", True))
    SMTP_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    SMTP_CONFIG_FILE.write_text(json.dumps(keep, indent=2), encoding="utf-8")
    return keep


def _smtp_config() -> dict:
    """JSON store wins over env vars — lets users configure from the UI."""
    j = _load_smtp_json()
    def pick(k_json, env_keys, default=""):
        v = (j.get(k_json) or "").strip() if isinstance(j.get(k_json), str) else j.get(k_json)
        if v not in (None, ""):
            return v
        for ek in env_keys:
            ev = os.environ.get(ek, "").strip()
            if ev:
                return ev
        return default
    host = pick("host", ["SMTP_HOST"])
    port_raw = j.get("port") if j.get("port") not in (None, "") else os.environ.get("SMTP_PORT", "587") or "587"
    try:
        port = int(port_raw)
    except Exception:
        port = 587
    use_tls_val = j.get("use_tls")
    if use_tls_val is None:
        use_tls = (os.environ.get("SMTP_USE_TLS", "1").strip() != "0")
    else:
        use_tls = bool(use_tls_val)
    return {
        "host": host,
        "port": port,
        "user": pick("user", ["SMTP_USER"]),
        "pass": pick("pass", ["SMTP_PASS"]),
        "from": pick("from", ["EMAIL_FROM", "SMTP_FROM"]),
        "sdr_notify": pick("sdr_notify", ["SDR_NOTIFY_EMAIL"]),
        "use_tls": use_tls,
    }


def _send_email_sync(to_addr: str, subject: str, body_text: str,
                     body_html: str = "", ics_content: str = "",
                     cc: str = "", reply_to: str = "") -> bool:
    """Synchronous SMTP send. Returns True on success.
    Designed to be called from a thread executor so we don't block the loop."""
    cfg = _smtp_config()
    if not (cfg["host"] and cfg["from"] and to_addr):
        logger.info("SMTP not configured or no recipient — skipping email to %r", to_addr)
        return False
    msg = EmailMessage()
    msg["From"] = cfg["from"]
    msg["To"] = to_addr
    if cc:
        msg["Cc"] = cc
    if reply_to:
        msg["Reply-To"] = reply_to
    msg["Subject"] = subject
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid()
    msg.set_content(body_text or " ")
    if body_html:
        msg.add_alternative(body_html, subtype="html")
    if ics_content:
        # Attach as both alternative (for inline parsing) and as a file so
        # most clients (Gmail, Outlook, Apple Mail) auto-detect the invite.
        msg.add_attachment(
            ics_content.encode("utf-8"),
            maintype="text",
            subtype="calendar",
            filename="invite.ics",
        )
    try:
        ctx = ssl.create_default_context()
        recipients = [to_addr] + ([cc] if cc else [])
        if cfg["port"] == 465:
            with smtplib.SMTP_SSL(cfg["host"], cfg["port"], context=ctx, timeout=20) as s:
                if cfg["user"]:
                    s.login(cfg["user"], cfg["pass"])
                s.send_message(msg, from_addr=cfg["from"], to_addrs=recipients)
        else:
            with smtplib.SMTP(cfg["host"], cfg["port"], timeout=20) as s:
                s.ehlo()
                if cfg["use_tls"]:
                    s.starttls(context=ctx)
                    s.ehlo()
                if cfg["user"]:
                    s.login(cfg["user"], cfg["pass"])
                s.send_message(msg, from_addr=cfg["from"], to_addrs=recipients)
        logger.info("Email sent to %s — subject=%r", to_addr, subject[:60])
        return True
    except Exception:
        logger.exception("SMTP send failed to %s", to_addr)
        return False


async def _send_email(to_addr: str, subject: str, body_text: str,
                      body_html: str = "", ics_content: str = "",
                      cc: str = "", reply_to: str = "") -> bool:
    return await asyncio.to_thread(
        _send_email_sync, to_addr, subject, body_text, body_html,
        ics_content, cc, reply_to,
    )


def _build_ics(*, summary: str, description: str, start_utc: datetime,
               duration_minutes: int = 30, organizer_email: str = "",
               attendee_email: str = "", uid: str = "") -> str:
    """Build a minimal RFC5545 VEVENT inline. start_utc must be a naive UTC datetime."""
    end_utc = start_utc + timedelta(minutes=duration_minutes)
    fmt = lambda d: d.strftime("%Y%m%dT%H%M%SZ")
    uid = uid or (str(uuid.uuid4()) + "@knight-ai-sdr")
    org = organizer_email or _smtp_config()["from"] or "noreply@example.com"
    desc = (description or "").replace("\r", "").replace("\n", "\\n")
    summ = (summary or "Meeting").replace("\r", " ").replace("\n", " ")

    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Knight AI SDR//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:REQUEST",
        "BEGIN:VEVENT",
        f"UID:{uid}",
        f"DTSTAMP:{fmt(datetime.utcnow())}",
        f"DTSTART:{fmt(start_utc)}",
        f"DTEND:{fmt(end_utc)}",
        f"SUMMARY:{summ}",
        f"DESCRIPTION:{desc}",
        f"ORGANIZER;CN=Knight AI SDR:mailto:{org}",
    ]
    if attendee_email:
        lines.append(
            f"ATTENDEE;CN={attendee_email};RSVP=TRUE;PARTSTAT=NEEDS-ACTION:mailto:{attendee_email}"
        )
    lines += ["STATUS:CONFIRMED", "END:VEVENT", "END:VCALENDAR"]
    return "\r\n".join(lines) + "\r\n"


# ──────────────────────────────────────────────────────────────
# Time parsing — turn "in 2 hours", "tomorrow at 3pm" into UTC ISO
# ──────────────────────────────────────────────────────────────
_REL_RE = _re.compile(
    r"in\s+(\d+(?:\.\d+)?)\s*(minute|min|hour|hr|day)s?",
    flags=_re.IGNORECASE,
)


def _parse_relative_when(text: str) -> str | None:
    """Best-effort parse of 'in N hours/min/days' phrases. Returns ISO8601 UTC or None."""
    if not text:
        return None
    m = _REL_RE.search(text)
    if not m:
        return None
    qty = float(m.group(1))
    unit = m.group(2).lower()
    if unit.startswith("min"):
        delta = timedelta(minutes=qty)
    elif unit.startswith(("hour", "hr")):
        delta = timedelta(hours=qty)
    else:
        delta = timedelta(days=qty)
    return (datetime.utcnow() + delta).replace(microsecond=0).isoformat() + "Z"


def _resolve_callback_when(insights: dict, transcript_text: str = "") -> str | None:
    """Extract callback time from insights or transcript. Returns ISO8601 UTC or None."""
    if not isinstance(insights, dict):
        return None
    # 1. Explicit field if Claude already produced one.
    for k in ("callback_at_utc", "callback_time_utc", "callback_iso"):
        v = (insights.get(k) or "").strip() if isinstance(insights.get(k), str) else ""
        if v:
            return v
    # 2. Search next_step / summary / short_tag / transcript for "in N hours".
    blobs = []
    for k in ("next_step", "summary", "short_tag"):
        if isinstance(insights.get(k), str):
            blobs.append(insights[k])
    if transcript_text:
        blobs.append(transcript_text)
    for blob in blobs:
        iso = _parse_relative_when(blob)
        if iso:
            return iso
    return None


# ──────────────────────────────────────────────────────────────
# Callback queue — auto-dial at scheduled time.
# Stored alongside tasks (type=callback, auto_dial=True, due_at_utc=...).
# ──────────────────────────────────────────────────────────────
def _queue_auto_callback(rec: dict, due_at_utc_iso: str, reason: str = "",
                         call_control_id: str = "") -> None:
    """Persist a callback task that the scheduler will auto-dial when due."""
    try:
        existing = load_tasks() or []
        # Prevent duplicate auto-dial for the same source call.
        for t in existing:
            if (t.get("call_control_id") == call_control_id
                    and t.get("auto_dial")
                    and (t.get("status") or "").lower() == "pending"):
                return
        camp_id, camp_name = _find_campaign_for_call(rec)
        task = {
            "id": str(uuid.uuid4())[:8],
            "prospect_name": rec.get("prospect_name", ""),
            "phone": rec.get("to", "") or rec.get("phone", ""),
            "company": rec.get("company", ""),
            "type": "callback",
            "outcome": "callback_scheduled",
            "campaign_id": camp_id,
            "campaign_name": camp_name,
            "due_date": due_at_utc_iso[:10],
            "due_at_utc": due_at_utc_iso,
            "auto_dial": True,
            "auto_dial_status": "scheduled",
            "notes": reason or "Auto-callback scheduled from call.",
            "status": "pending",
            "call_control_id": call_control_id,
            "prospect_email": rec.get("prospect_email", ""),
            "created_at": datetime.utcnow().isoformat(),
        }
        save_task(task)
        logger.info("Queued auto-callback to %s at %s", task["phone"], due_at_utc_iso)
    except Exception:
        logger.exception("Failed to queue auto-callback")


def _list_pending_auto_callbacks() -> list[dict]:
    out: list[dict] = []
    for t in (load_tasks() or []):
        if (t.get("auto_dial")
                and (t.get("status") or "").lower() == "pending"
                and (t.get("auto_dial_status") or "scheduled") in ("scheduled",)):
            out.append(t)
    return out


def _mark_task_field(task_id: str, **fields) -> None:
    try:
        ts = load_tasks() or []
        changed = False
        for t in ts:
            if t.get("id") == task_id:
                t.update(fields)
                changed = True
                break
        if changed:
            from json import dumps as _dumps
            (Path(__file__).parent / "data" / "tasks.json").write_text(
                _dumps(ts, indent=2, default=str), encoding="utf-8")
    except Exception:
        logger.exception("_mark_task_field failed for %s", task_id)


# ──────────────────────────────────────────────────────────────
# Post-call: send invite + notify + queue auto-callback
# ──────────────────────────────────────────────────────────────
async def _post_call_email_actions(rec: dict, insights: dict,
                                    call_control_id: str) -> None:
    """Fire emails and auto-callback queueing based on outcome."""
    if not isinstance(insights, dict):
        return
    outcome = _normalize_outcome(insights.get("outcome"))
    cfg = _smtp_config()
    sdr_notify = cfg["sdr_notify"]
    prospect_email = (rec.get("prospect_email") or rec.get("email") or "").strip()
    prospect_name = rec.get("prospect_name") or "there"
    company = rec.get("company") or ""
    summary = (insights.get("summary") or "").strip()
    next_step = (insights.get("next_step") or "").strip()
    short_tag = (insights.get("short_tag") or "").strip()
    phone = rec.get("to") or rec.get("phone") or ""

    # ─── 0. Email Agent: queue an AI-drafted follow-up email if eligible ───
    try:
        ea = _load_email_agent()
        eligible_outcomes = set(ea.get("outcomes_to_email") or [])
        skip_outcomes = set(ea.get("skip_outcomes") or [])
        if (ea.get("enabled") and ea.get("auto_send_after_call")
                and outcome in eligible_outcomes and outcome not in skip_outcomes
                and prospect_email):
            transcript_text = _gather_call_transcript_text(rec)
            _queue_email_send(
                rec=rec, insights=insights,
                call_control_id=call_control_id,
                transcript_text=transcript_text,
                delay_minutes=int(ea.get("send_delay_minutes") or 5),
            )
    except Exception:
        logger.exception("Email Agent auto-queue failed for %s", call_control_id[:20])

    # ─── 1. Meeting booked → send .ics invite + notify SDR ───
    if outcome == "meeting_booked":
        mt = (insights.get("meeting_time_utc") or insights.get("meeting_time") or "").strip()
        start_utc: datetime | None = None
        if mt:
            try:
                start_utc = datetime.fromisoformat(mt.replace("Z", "")).replace(microsecond=0)
            except Exception:
                pass
        if start_utc is None:
            # Fallback: tomorrow at 10:00 UTC if AI didn't extract a time.
            start_utc = (datetime.utcnow() + timedelta(days=1)).replace(
                hour=10, minute=0, second=0, microsecond=0)
        ics = _build_ics(
            summary=f"Meeting with {prospect_name}" + (f" — {company}" if company else ""),
            description=f"Scheduled via call. {summary}\n\n{next_step}",
            start_utc=start_utc,
            duration_minutes=30,
            organizer_email=cfg["from"],
            attendee_email=prospect_email,
        )
        body = (f"Hi {prospect_name},\n\n"
                f"Thanks for the call! I've blocked 30 minutes on the calendar "
                f"as discussed. Calendar invite is attached — accept it and you're set.\n\n"
                f"{summary}\n\n"
                f"Talk soon!\n")
        html = body.replace("\n", "<br>")
        if prospect_email:
            await _send_email(
                to_addr=prospect_email,
                subject=f"Meeting confirmation — {start_utc.strftime('%b %d, %H:%M UTC')}",
                body_text=body, body_html=html,
                ics_content=ics,
                cc=sdr_notify,  # cc the SDR so they have it on their calendar too
                reply_to=cfg["from"],
            )
        # Always notify SDR, even if prospect email is missing
        if sdr_notify and not prospect_email:
            await _send_email(
                to_addr=sdr_notify,
                subject=f"[Meeting Booked] {prospect_name} — {short_tag or 'see notes'}",
                body_text=(f"Meeting booked with {prospect_name} ({phone}).\n"
                          f"Time (UTC): {start_utc.isoformat()}\n\n{summary}\n\n"
                          f"Next step: {next_step}\n"),
                ics_content=ics,
            )
        return

    # ─── 2. Callback scheduled → queue auto-dial + email confirm + notify SDR ───
    if outcome == "callback_scheduled":
        # Try to resolve a precise callback time (e.g. "in 2 hours").
        cb_iso = _resolve_callback_when(insights)
        if cb_iso:
            _queue_auto_callback(rec, cb_iso, reason=next_step or summary,
                                 call_control_id=call_control_id)
            try:
                cb_dt = datetime.fromisoformat(cb_iso.replace("Z", ""))
                when_h = cb_dt.strftime("%b %d, %H:%M UTC")
            except Exception:
                when_h = cb_iso
            if prospect_email:
                await _send_email(
                    to_addr=prospect_email,
                    subject=f"Quick follow-up at {when_h}",
                    body_text=(f"Hi {prospect_name},\n\n"
                              f"Thanks for the chat — as agreed I'll give you a quick "
                              f"call back around {when_h}.\n\n"
                              f"{summary}\n\nTalk soon!\n"),
                )
            if sdr_notify:
                await _send_email(
                    to_addr=sdr_notify,
                    subject=f"[Callback queued] {prospect_name} — {when_h}",
                    body_text=(f"Auto-callback scheduled for {prospect_name} ({phone}).\n"
                              f"Due (UTC): {cb_iso}\n\n{summary}\n\nReason: {next_step}\n"),
                )
        else:
            # No specific time → just notify the SDR.
            if sdr_notify:
                await _send_email(
                    to_addr=sdr_notify,
                    subject=f"[Callback requested] {prospect_name}",
                    body_text=(f"Callback requested by {prospect_name} ({phone}) "
                              f"but no specific time captured.\n\n{summary}\n"
                              f"Next step: {next_step}\n"),
                )
        return

    # ─── 3. Interested → polite follow-up email ───
    if outcome == "interested":
        if prospect_email:
            await _send_email(
                to_addr=prospect_email,
                subject=f"Following up on our call",
                body_text=(f"Hi {prospect_name},\n\n"
                          f"Thanks for taking the time today. Quick recap:\n"
                          f"{summary}\n\n"
                          f"Reply to this email and we can lock in next steps.\n\n"
                          f"Best,\n"),
                cc=sdr_notify,
                reply_to=cfg["from"],
            )
        if sdr_notify:
            await _send_email(
                to_addr=sdr_notify,
                subject=f"[Interested] {prospect_name} — followup sent",
                body_text=(f"{prospect_name} ({phone}) showed interest.\n\n"
                          f"{summary}\n\nNext: {next_step}\n"),
            )
        return


# ──────────────────────────────────────────────────────────────
# Campaign scheduler — windows per day, dial caps, timezone-aware
# ──────────────────────────────────────────────────────────────
def _validate_schedule(sched: dict) -> tuple[bool, str]:
    """Validate a campaign schedule dict. Returns (ok, error_msg)."""
    if not isinstance(sched, dict):
        return False, "schedule must be an object"
    tz = (sched.get("timezone") or "UTC").strip()
    if ZoneInfo is not None:
        try:
            ZoneInfo(tz)
        except Exception:
            return False, f"unknown timezone {tz!r}"
    days = sched.get("days_of_week") or [0, 1, 2, 3, 4]
    if not (isinstance(days, list) and all(isinstance(d, int) and 0 <= d <= 6 for d in days)):
        return False, "days_of_week must be a list of ints 0..6 (Mon=0)"
    windows = sched.get("windows") or []
    if not isinstance(windows, list):
        return False, "windows must be a list"
    parsed: list[tuple[int, int, int]] = []  # (start_min, end_min, cap)
    for i, w in enumerate(windows):
        if not isinstance(w, dict):
            return False, f"window {i} must be an object"
        try:
            sh, sm = (w.get("start") or "").split(":")
            eh, em = (w.get("end") or "").split(":")
            sm_int = int(sh) * 60 + int(sm)
            em_int = int(eh) * 60 + int(em)
            cap = int(w.get("dial_cap", 0))
        except Exception:
            return False, f"window {i}: invalid HH:MM or dial_cap"
        if em_int <= sm_int:
            return False, f"window {i}: end must be after start"
        if cap < 0:
            return False, f"window {i}: dial_cap must be >= 0"
        parsed.append((sm_int, em_int, cap))
    parsed.sort()
    for i in range(1, len(parsed)):
        if parsed[i][0] < parsed[i - 1][1]:
            return False, f"windows overlap at index {i}"
    return True, ""


def _campaign_now_local(sched: dict) -> datetime:
    tz = (sched.get("timezone") or "UTC").strip()
    if ZoneInfo is not None:
        try:
            return datetime.now(ZoneInfo(tz))
        except Exception:
            pass
    return datetime.utcnow()


def _campaign_active_window_idx(sched: dict, now_local: datetime) -> int | None:
    """Return current active window index, or None."""
    days = sched.get("days_of_week") or [0, 1, 2, 3, 4]
    if now_local.weekday() not in days:
        return None
    cur_min = now_local.hour * 60 + now_local.minute
    windows = sched.get("windows") or []
    for i, w in enumerate(windows):
        try:
            sh, sm = (w.get("start") or "").split(":")
            eh, em = (w.get("end") or "").split(":")
            s = int(sh) * 60 + int(sm)
            e = int(eh) * 60 + int(em)
            if s <= cur_min < e:
                return i
        except Exception:
            continue
    return None


async def _scheduler_dial_one_for_campaign(c: dict) -> bool:
    """Place a single call for the campaign if any prospect is eligible."""
    prospects = c.get("prospects") or []
    for p in prospects:
        if not isinstance(p, dict):
            continue
        status = (p.get("status") or "queued").lower()
        if status in ("dialed", "completed", "skipped"):
            continue
        phone = normalize_phone(p.get("phone")) or p.get("phone") or ""
        if not phone:
            continue
        if is_dnc(phone):
            _mark_campaign_prospect(c["id"], phone, status="skipped", outcome="dnc")
            continue
        # Place the call.
        try:
            req = CallRequest(
                to_number=phone,
                prospect_name=prospect_display_name(p),
                company=p.get("company") or "",
                notes=p.get("notes") or "",
                prospect_email=str(p.get("email") or "").strip(),
                voice_id=c.get("voice_id") or "",
            )
            result = await place_outbound_call(req)
            cc_id = result.get("call_control_id")
            _mark_campaign_prospect(
                c["id"], phone,
                status="dialed", outcome="",
                call_control_id=cc_id,
                dialed_at=datetime.utcnow().isoformat(),
            )
            return True
        except Exception:
            logger.exception("Scheduler dial failed for %s", phone)
            _mark_campaign_prospect(c["id"], phone, status="error")
            continue
    return False


async def _scheduler_tick() -> None:
    """One iteration of the unified scheduler (campaigns + auto-callbacks + email_send)."""
    # ── 0. Email-Agent auto-sends (drain everything that's due — emails are cheap) ──
    try:
        now_utc = datetime.utcnow()
        for t in _list_pending_email_sends():
            due = (t.get("due_at_utc") or "").replace("Z", "")
            try:
                due_dt = datetime.fromisoformat(due)
            except Exception:
                continue
            if due_dt > now_utc:
                continue
            try:
                await _dispatch_email_send_task(t)
            except Exception:
                logger.exception("Email send dispatch failed for task %s", t.get("id"))
                _mark_task_field(t["id"], auto_send_status="error")
    except Exception:
        logger.exception("email_send tick failed")

    # ── A. Auto-callbacks (highest priority — always honor user-promised time) ──
    try:
        now_utc = datetime.utcnow()
        for t in _list_pending_auto_callbacks():
            due = (t.get("due_at_utc") or "").replace("Z", "")
            try:
                due_dt = datetime.fromisoformat(due)
            except Exception:
                continue
            if due_dt > now_utc:
                continue
            phone = t.get("phone") or ""
            if not phone or is_dnc(phone):
                _mark_task_field(t["id"], auto_dial_status="skipped",
                                 status="done", completed=True)
                continue
            try:
                req = CallRequest(
                    to_number=phone,
                    prospect_name=t.get("prospect_name") or "there",
                    company=t.get("company") or "",
                    notes=f"Auto-callback follow-up. {t.get('notes') or ''}",
                    prospect_email=t.get("prospect_email") or "",
                )
                result = await place_outbound_call(req)
                _mark_task_field(t["id"],
                                 auto_dial_status="dialed",
                                 status="done", completed=True,
                                 dialed_call_control_id=result.get("call_control_id"),
                                 dialed_at=datetime.utcnow().isoformat())
                logger.info("Auto-callback dialed task=%s phone=%s", t.get("id"), phone)
            except Exception:
                logger.exception("Auto-callback dial failed for %s", phone)
                _mark_task_field(t["id"], auto_dial_status="error")
    except Exception:
        logger.exception("Auto-callback tick failed")

    # ── B. Campaign window scheduler ──
    try:
        # Skip if a manual campaign run is already in progress (avoid double-dial).
        if _active_campaign_id:
            return
        camps = _load_campaigns()
        for c in camps:
            if not isinstance(c, dict):
                continue
            sched = c.get("schedule") or {}
            if not sched.get("enabled"):
                continue
            if (c.get("status") or "").lower() in ("completed", "stopped"):
                continue
            now_local = _campaign_now_local(sched)
            widx = _campaign_active_window_idx(sched, now_local)
            if widx is None:
                continue
            window = (sched.get("windows") or [])[widx]
            cap = int(window.get("dial_cap", 0))
            if cap <= 0:
                continue
            today = now_local.date().isoformat()
            counters = c.setdefault("dials_by_date", {})
            day_counters = counters.setdefault(today, {})
            already = int(day_counters.get(str(widx), 0))
            if already >= cap:
                continue
            ok = await _scheduler_dial_one_for_campaign(c)
            if ok:
                day_counters[str(widx)] = already + 1
                # Persist counter increment.
                _update_campaign(c["id"], {
                    "dials_by_date": counters,
                    "status": "scheduled",
                })
                # One dial per tick across all campaigns to avoid bursts.
                return
    except Exception:
        logger.exception("Campaign scheduler tick failed")


_scheduler_task: asyncio.Task | None = None


async def _scheduler_loop() -> None:
    logger.info("Phase 3 scheduler loop started.")
    while True:
        try:
            await _scheduler_tick()
        except Exception:
            logger.exception("Scheduler loop iteration failed")
        # 60s cadence — fine-grained enough for hourly/15-min dial caps.
        await asyncio.sleep(60)


@app.on_event("startup")
async def _start_phase3_scheduler():
    global _scheduler_task
    if _scheduler_task is None or _scheduler_task.done():
        _scheduler_task = asyncio.create_task(_scheduler_loop())


# ──────────────────────────────────────────────────────────────
# Phase 3 API endpoints
# ──────────────────────────────────────────────────────────────
@app.patch("/api/campaigns/{campaign_id}/schedule")
async def patch_campaign_schedule(campaign_id: str, request: Request):
    body = await request.json()
    sched = body.get("schedule") if isinstance(body, dict) else None
    if not isinstance(sched, dict):
        sched = body  # allow PATCH with raw schedule body
    sched.setdefault("enabled", True)
    sched.setdefault("timezone", "UTC")
    sched.setdefault("days_of_week", [0, 1, 2, 3, 4])
    sched.setdefault("windows", [])
    ok, err = _validate_schedule(sched)
    if not ok:
        raise HTTPException(400, err)
    updated = _update_campaign(campaign_id, {"schedule": sched})
    if not updated:
        raise HTTPException(404, "Campaign not found")
    return {"ok": True, "schedule": sched}


@app.post("/api/callbacks/schedule")
async def post_schedule_callback(request: Request):
    body = await request.json()
    phone = (body.get("phone") or "").strip()
    when = (body.get("when_utc") or body.get("due_at_utc") or "").strip()
    if not phone or not when:
        raise HTTPException(400, "phone and when_utc are required")
    rec = {
        "to": phone,
        "prospect_name": body.get("prospect_name", ""),
        "company": body.get("company", ""),
        "prospect_email": body.get("prospect_email", ""),
    }
    _queue_auto_callback(rec, when, reason=body.get("notes", ""),
                         call_control_id=body.get("call_control_id", ""))
    return {"ok": True}


@app.get("/api/callbacks/pending")
async def list_pending_callbacks():
    return {"callbacks": _list_pending_auto_callbacks()}


@app.post("/api/admin/test-email")
async def admin_test_email(request: Request):
    body = await request.json()
    to = (body.get("to") or _smtp_config()["sdr_notify"] or "").strip()
    if not to:
        raise HTTPException(400, "Recipient 'to' missing and SDR_NOTIFY_EMAIL not set")
    ok = await _send_email(
        to_addr=to,
        subject="Knight AI SDR — SMTP test",
        body_text="If you're reading this, SMTP is configured correctly.\n",
    )
    return {"ok": ok, "to": to, "smtp_configured": bool(_smtp_config()["host"])}


@app.get("/api/email/status")
async def email_status():
    """Lightweight probe used by the dashboard banner."""
    cfg = _smtp_config()
    return {
        "configured": bool(cfg["host"] and cfg["from"]),
        "host": cfg["host"],
        "from_addr": cfg["from"],
        "sdr_notify": cfg["sdr_notify"],
    }


@app.get("/api/smtp/config")
async def get_smtp_config():
    """Return the current SMTP config — masks the password but tells you if it's set."""
    cfg = _smtp_config()
    j = _load_smtp_json()
    return {
        "host":       cfg["host"],
        "port":       cfg["port"],
        "user":       cfg["user"],
        "from":       cfg["from"],
        "sdr_notify": cfg["sdr_notify"],
        "use_tls":    cfg["use_tls"],
        "has_password": bool(cfg["pass"]),
        "source":     "json" if j.get("host") else ("env" if cfg["host"] else "none"),
    }


@app.post("/api/smtp/config")
async def save_smtp_config(request: Request):
    """Save user-supplied SMTP credentials. Pass empty 'pass' to keep existing one."""
    body = await request.json() or {}
    existing = _load_smtp_json()
    new_pass = (body.get("pass") or "").strip()
    if not new_pass and existing.get("pass"):
        new_pass = existing["pass"]
    saved = _save_smtp_json({
        "host":       (body.get("host") or "").strip(),
        "port":       body.get("port") or 587,
        "user":       (body.get("user") or "").strip(),
        "pass":       new_pass,
        "from":       (body.get("from") or "").strip(),
        "sdr_notify": (body.get("sdr_notify") or "").strip(),
        "use_tls":    bool(body.get("use_tls", True)),
    })
    cfg = _smtp_config()
    return {
        "ok": True,
        "configured": bool(cfg["host"] and cfg["from"]),
        "host": saved["host"], "port": saved["port"], "from": saved["from"],
        "user": saved["user"], "sdr_notify": saved["sdr_notify"], "use_tls": saved["use_tls"],
        "has_password": bool(saved["pass"]),
    }


@app.post("/api/smtp/test")
async def smtp_test(request: Request):
    """Save (if body has creds) then send a one-line test email."""
    body = await request.json() or {}
    cfg = _smtp_config()
    to = (body.get("to") or cfg["sdr_notify"] or cfg["from"] or "").strip()
    if not to:
        raise HTTPException(400, "Provide a recipient ('to') — or save SDR Notify Email first.")
    if not (cfg["host"] and cfg["from"]):
        raise HTTPException(400, "SMTP not configured. Save host/port/user/pass/from first.")
    ok = await _send_email(
        to_addr=to,
        subject="Knight AI SDR — SMTP test",
        body_text=f"If you can read this, SMTP is working.\nHost: {cfg['host']}:{cfg['port']}\nFrom: {cfg['from']}\n",
        body_html=f"<p>If you can read this, <b>SMTP is working</b>.</p><p><code>{cfg['host']}:{cfg['port']}</code> · From <code>{cfg['from']}</code></p>",
    )
    return {"ok": ok, "to": to}


# ════════════════════════════════════════════════════════════
#  EMAIL AGENT — persona + Claude drafter + auto-send pipeline
# ════════════════════════════════════════════════════════════
EMAIL_AGENT_FILE = Path(__file__).parent / "data" / "email_agent.json"


_EMAIL_AGENT_DEFAULTS = {
    "enabled": True,
    "auto_send_after_call": True,
    "send_delay_minutes": 5,         # how long after the call ends before drafting+sending
    "sender_name": "Alex from Knight",
    "from_address": "",               # if empty, uses EMAIL_FROM env
    "default_cc": "",                 # comma-separated additional CCs
    "reply_to": "",
    "tone": "Friendly, concise, professional. Match the prospect's energy.",
    "length": "short",                # short | medium | long
    "language": "English",
    "signature": "Best,\n{sender_name}\nKnight AI SDR",
    "signature_html": "",             # optional HTML signature
    "draft_instructions": (
        "You are an SDR following up on a sales call. Reference one specific thing "
        "they said. Keep it under 120 words unless asked otherwise. End with a clear "
        "next step (calendar link, reply prompt, or specific question)."
    ),
    "subject_template": "Following up on our call",
    "include_call_summary": True,
    "send_meeting_invite_with_followup": True,
    "outcomes_to_email": ["meeting_booked", "callback_scheduled", "interested"],
    "skip_outcomes": ["do_not_call", "not_interested"],
    "updated_at": "",
}


def _load_email_agent() -> dict:
    if EMAIL_AGENT_FILE.exists():
        try:
            data = json.loads(EMAIL_AGENT_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                merged = dict(_EMAIL_AGENT_DEFAULTS)
                merged.update(data)
                return merged
        except Exception:
            pass
    return dict(_EMAIL_AGENT_DEFAULTS)


def _save_email_agent(data: dict) -> dict:
    merged = dict(_EMAIL_AGENT_DEFAULTS)
    merged.update({k: v for k, v in (data or {}).items() if k in _EMAIL_AGENT_DEFAULTS})
    merged["updated_at"] = datetime.utcnow().isoformat()
    EMAIL_AGENT_FILE.parent.mkdir(parents=True, exist_ok=True)
    EMAIL_AGENT_FILE.write_text(json.dumps(merged, indent=2, default=str), encoding="utf-8")
    return merged


@app.get("/api/email-agent")
async def get_email_agent():
    return _load_email_agent()


@app.post("/api/email-agent")
async def save_email_agent(request: Request):
    body = await request.json()
    return _save_email_agent(body or {})


# ── Claude drafter ─────────────────────────────────────────
async def _claude_draft_email(*, ea: dict, rec: dict, insights: dict,
                              transcript_text: str = "",
                              user_brief: str = "",
                              extra_context: str = "") -> dict:
    """Use Claude to write a follow-up email. Returns {subject, body_text, body_html}."""
    sender_name = (ea.get("sender_name") or "").strip() or "Alex"
    tone = ea.get("tone") or "Friendly, concise, professional."
    length = ea.get("length") or "short"
    language = ea.get("language") or "English"
    instr = ea.get("draft_instructions") or ""
    sig = (ea.get("signature") or "").replace("{sender_name}", sender_name)
    subj_tpl = ea.get("subject_template") or "Following up on our call"
    include_summary = ea.get("include_call_summary", True)
    prospect_name = rec.get("prospect_name") or "there"
    company = rec.get("company") or ""
    outcome = (insights or {}).get("outcome") or ""
    summary = (insights or {}).get("summary") or ""
    next_step = (insights or {}).get("next_step") or ""
    pain = (insights or {}).get("prospect_pain_points") or []
    signals = (insights or {}).get("buying_signals") or []
    objections = (insights or {}).get("objections") or []

    transcript_snippet = ""
    if transcript_text:
        # Trim transcript to last ~3000 chars to keep prompt focused.
        transcript_snippet = transcript_text[-3000:]

    prompt = f"""You are an SDR named {sender_name} drafting a personal follow-up email to a prospect.

# Persona / Style guide
Tone: {tone}
Length: {length}
Language: {language}

# Custom instructions
{instr}

# Prospect
Name: {prospect_name}
Company: {company}

# Call outcome
Outcome: {outcome}
Summary: {summary}
Recommended next step: {next_step}
Pain points: {", ".join(map(str, pain)) or "—"}
Buying signals: {", ".join(map(str, signals)) or "—"}
Objections raised: {", ".join(map(str, objections)) or "—"}

# Recent transcript (verbatim — reference one specific thing they said)
{transcript_snippet or "(no transcript captured)"}

# Extra context from user (if any)
{user_brief or extra_context or "(none)"}

Now produce JSON ONLY in this shape — no preamble, no fences:
{{
  "subject": "concise subject line (no clickbait, max 60 chars). Default base: \\"{subj_tpl}\\".",
  "body_text": "the full email body in plain text. {'' if include_summary else 'Do NOT include the call summary.'} End with the signature exactly:\\n\\n{sig}",
  "body_html": "the same email but rendered as simple HTML (use <p> tags, <br> for line breaks, do NOT include <html>/<body> wrappers)."
}}"""

    try:
        client = AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
        resp = await client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=900,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        if raw.startswith("json"):
            raw = raw[4:]
        data = json.loads(raw.strip())
        if not isinstance(data, dict):
            raise ValueError("draft not a JSON object")
        return {
            "subject": (data.get("subject") or subj_tpl).strip(),
            "body_text": (data.get("body_text") or "").strip(),
            "body_html": (data.get("body_html") or "").strip(),
        }
    except Exception as e:
        logger.exception("Claude email draft failed")
        # Fallback: basic template so we never silently fail.
        body = (f"Hi {prospect_name},\n\n"
                f"Thanks for taking the time today. Quick recap:\n{summary or '—'}\n\n"
                f"{next_step or 'Reply to this email and we can pick the right next step.'}\n\n{sig}")
        return {"subject": subj_tpl, "body_text": body,
                "body_html": body.replace("\n", "<br>")}


def _gather_call_transcript_text(rec: dict) -> str:
    turns = (rec or {}).get("transcript") or (rec or {}).get("turns") or []
    if not isinstance(turns, list):
        return ""
    out: list[str] = []
    for t in turns:
        if not isinstance(t, dict):
            continue
        role = (t.get("role") or t.get("speaker") or "").lower()
        text = (t.get("text") or t.get("content") or "").strip()
        if not text:
            continue
        spk = "Prospect" if role in ("user", "human", "prospect") else "AI"
        out.append(f"{spk}: {text}")
    return "\n".join(out)


@app.post("/api/email-agent/draft-preview")
async def post_email_agent_draft_preview(request: Request):
    """Manual preview — feed sample fields, see what Claude will write."""
    body = await request.json()
    ea = _load_email_agent()
    rec = body.get("rec") or {"prospect_name": body.get("prospect_name", "Sample"),
                              "company": body.get("company", "Acme")}
    insights = body.get("insights") or {"outcome": "interested",
                                        "summary": body.get("summary") or "Discussed integration timeline.",
                                        "next_step": "Send pricing one-pager and book technical demo."}
    transcript = body.get("transcript") or ""
    extra = body.get("extra_context") or ""
    return await _claude_draft_email(ea=ea, rec=rec, insights=insights,
                                     transcript_text=transcript, extra_context=extra)


@app.post("/api/email-agent/test-send")
async def post_email_agent_test_send(request: Request):
    body = await request.json()
    to = (body.get("to") or "").strip()
    if not to:
        raise HTTPException(400, "to is required")
    ea = _load_email_agent()
    draft = await _claude_draft_email(
        ea=ea,
        rec={"prospect_name": body.get("prospect_name", "Sample"),
             "company": body.get("company", "Acme")},
        insights={"outcome": "interested",
                  "summary": "Sample call summary for SMTP test.",
                  "next_step": "Confirm receipt of this test email."},
        transcript_text="Prospect: This is a sample transcript.",
        extra_context=body.get("extra_context") or "",
    )
    ok = await _send_email(
        to_addr=to,
        subject=draft["subject"],
        body_text=draft["body_text"],
        body_html=draft["body_html"],
        cc=ea.get("default_cc") or "",
        reply_to=ea.get("reply_to") or "",
    )
    return {"ok": ok, "draft": draft}


# ── Auto-send pipeline ─────────────────────────────────────
def _queue_email_send(*, rec: dict, insights: dict, call_control_id: str,
                     transcript_text: str = "", delay_minutes: int = 5) -> str | None:
    """Persist a pending email_send task. Returns task id or None."""
    try:
        existing = load_tasks() or []
        # Idempotent — one auto-email per call.
        for t in existing:
            if (t.get("call_control_id") == call_control_id
                    and (t.get("type") or "") == "email_send"):
                return None
        camp_id, camp_name = _find_campaign_for_call(rec)
        due_iso = (datetime.utcnow() + timedelta(minutes=max(0, int(delay_minutes))))\
            .replace(microsecond=0).isoformat() + "Z"
        # Capture the transcript NOW so we don't lose it later.
        task = {
            "id": str(uuid.uuid4())[:8],
            "type": "email_send",
            "auto_send": True,
            "auto_send_status": "scheduled",
            "due_at_utc": due_iso,
            "due_date": due_iso[:10],
            "status": "pending",
            "prospect_name": rec.get("prospect_name", ""),
            "phone": rec.get("to", "") or rec.get("phone", ""),
            "company": rec.get("company", ""),
            "email_to": (rec.get("prospect_email") or rec.get("email") or "").strip(),
            "outcome": (insights or {}).get("outcome") or "",
            "campaign_id": camp_id,
            "campaign_name": camp_name,
            "call_control_id": call_control_id,
            "snapshot_insights": insights or {},
            "snapshot_rec": {k: rec.get(k) for k in
                             ("prospect_name", "company", "to", "phone",
                              "prospect_email", "email")},
            "snapshot_transcript": (transcript_text or "")[-6000:],
            "notes": f"Auto-followup for outcome={(insights or {}).get('outcome','?')}",
            "created_at": datetime.utcnow().isoformat(),
        }
        save_task(task)
        logger.info("Queued email_send task=%s due=%s to=%s",
                    task["id"], due_iso, task["email_to"] or "(no addr)")
        return task["id"]
    except Exception:
        logger.exception("_queue_email_send failed")
        return None


def _list_pending_email_sends() -> list[dict]:
    out: list[dict] = []
    for t in (load_tasks() or []):
        if ((t.get("type") or "") == "email_send"
                and (t.get("status") or "").lower() == "pending"
                and (t.get("auto_send_status") or "scheduled") == "scheduled"):
            out.append(t)
    return out


async def _dispatch_email_send_task(t: dict) -> None:
    """Draft via Claude and send via SMTP. Marks the task done either way."""
    ea = _load_email_agent()
    if not ea.get("enabled"):
        _mark_task_field(t["id"], auto_send_status="disabled",
                         status="done", completed=True)
        return
    to = (t.get("email_to") or "").strip()
    if not to:
        _mark_task_field(t["id"], auto_send_status="no_address",
                         status="done", completed=True,
                         notes=(t.get("notes") or "") + " — no email address on prospect")
        return
    rec_snap = dict(t.get("snapshot_rec") or {})
    rec_snap.setdefault("prospect_name", t.get("prospect_name", ""))
    rec_snap.setdefault("company", t.get("company", ""))
    rec_snap.setdefault("prospect_email", to)
    insights = dict(t.get("snapshot_insights") or {})
    transcript = t.get("snapshot_transcript") or ""

    # Respect skip-outcomes guard.
    outcome = (insights.get("outcome") or t.get("outcome") or "").lower()
    if outcome and outcome in (ea.get("skip_outcomes") or []):
        _mark_task_field(t["id"], auto_send_status="skipped_outcome",
                         status="done", completed=True)
        return

    draft = await _claude_draft_email(
        ea=ea, rec=rec_snap, insights=insights, transcript_text=transcript,
    )
    ok = await _send_email(
        to_addr=to,
        subject=draft["subject"],
        body_text=draft["body_text"],
        body_html=draft["body_html"],
        cc=ea.get("default_cc") or "",
        reply_to=ea.get("reply_to") or "",
    )
    _mark_task_field(
        t["id"],
        auto_send_status="sent" if ok else "error",
        status="done" if ok else "pending",
        completed=bool(ok),
        sent_at=(datetime.utcnow().isoformat() if ok else ""),
        sent_subject=draft.get("subject", ""),
        sent_body_text=draft.get("body_text", ""),
    )


@app.post("/api/email/send-from-call/{cc_id}")
async def post_send_email_from_call(cc_id: str, request: Request):
    """Manual trigger — draft and send right now using the call's transcript."""
    body = (await request.json()) if request.headers.get("content-type", "").startswith("application/json") else {}
    rec = active_calls.get(cc_id) or get_call_by_control_id(cc_id)
    if not rec:
        raise HTTPException(404, f"Call {cc_id} not found")
    to = (body.get("to") or rec.get("prospect_email") or rec.get("email") or "").strip()
    if not to:
        raise HTTPException(400, "No email address — pass 'to' in body or set prospect_email on the call")
    ea = _load_email_agent()
    transcript = _gather_call_transcript_text(rec)
    draft = await _claude_draft_email(
        ea=ea, rec=rec, insights=rec.get("insights") or {},
        transcript_text=transcript, extra_context=body.get("extra_context") or "",
    )
    if body.get("preview_only"):
        return {"ok": True, "draft": draft, "to": to}
    ok = await _send_email(
        to_addr=to,
        subject=draft["subject"],
        body_text=draft["body_text"],
        body_html=draft["body_html"],
        cc=ea.get("default_cc") or "",
        reply_to=ea.get("reply_to") or "",
    )
    return {"ok": ok, "draft": draft, "to": to}


@app.get("/api/email/sent")
async def list_sent_emails():
    """Email-send history derived from completed email_send tasks."""
    out = []
    for t in (load_tasks() or []):
        if (t.get("type") or "") != "email_send":
            continue
        out.append({
            "id": t.get("id"),
            "status": t.get("auto_send_status") or t.get("status"),
            "to": t.get("email_to"),
            "prospect_name": t.get("prospect_name"),
            "company": t.get("company"),
            "campaign_name": t.get("campaign_name"),
            "subject": t.get("sent_subject") or "",
            "due_at_utc": t.get("due_at_utc"),
            "sent_at": t.get("sent_at"),
            "outcome": t.get("outcome"),
            "call_control_id": t.get("call_control_id"),
        })
    out.sort(key=lambda x: (x.get("sent_at") or x.get("due_at_utc") or ""), reverse=True)
    return {"emails": out}


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
    # Voice override: stored on the call record rather than mutating the
    # global config — global mutation races between concurrent outbound
    # calls (two dials requesting different voices would stomp each other).
    # The assistant itself is tuned via sync_assistant_to_script once at
    # startup, so per-call voice swaps are tracked but not applied to the
    # shared AI Assistant config.
    voice_id = (req.voice_id or "").strip()
    if voice_id:
        logger.info("Voice override requested for call: %s (call-local only)", voice_id)
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

    # ── Build message_history: ONLY per-prospect briefing (not knowledge) ──
    # Product knowledge is already in the assistant's synced `instructions` —
    # duplicating it here bloats the API payload and LLM context, adding
    # seconds to warmup latency. Send only prospect-specific research.
    msg_history: list[dict] = []
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

    # ── NO manual recording start: Telnyx AI Assistant already records via
    # its built-in recording_settings (dual/mp3). Running a second recorder
    # on the same call caused audio-pipeline interference (echo/choppiness
    # from the very start of the call) and saved duplicate files. ──
    try:
        t0 = time.monotonic()
        await loop.run_in_executor(None, lambda: tx.calls.actions.start_ai_assistant(**ai_kwargs))
        latency_ms = (time.monotonic() - t0) * 1000
        active_calls.setdefault(cc_id, {})["ai_assistant"] = True
        _ai_assistant_started.add(cc_id)
        logger.info("AI Assistant started in %.0fms — greeting: %s", latency_ms, greeting[:60])
    except Exception as e:
        err_str = str(e)
        # 422 "already in progress" means Telnyx already auto-started it — treat as success
        if "90061" in err_str or "already in progress" in err_str.lower():
            logger.info("AI Assistant already running (auto-started by Telnyx) — continuing normally")
            active_calls.setdefault(cc_id, {})["ai_assistant"] = True
            _ai_assistant_started.add(cc_id)
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
            # setdefault: don't wipe history if a duplicate call.answered
            # webhook sneaks past the opened_calls guard (rare race).
            conversations.setdefault(cc_id, [])

            # Start silence watchdog — auto-hangup if no speech for 30s
            _last_speech_time[cc_id] = time.time()
            if cc_id in _silence_watchdog_tasks:
                _silence_watchdog_tasks[cc_id].cancel()
            _silence_watchdog_tasks[cc_id] = asyncio.create_task(_silence_watchdog(cc_id))

            # Voicemail detection is delegated to Telnyx (see sync_assistant_to_script):
            # telephony_settings.voicemail_detection.action = "hangup"
            _ai_user_turn_times[cc_id] = []
            _ai_agent_turn_times[cc_id] = []

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
                        _ai_user_turn_times.setdefault(cc_id, []).append(time.time())
                        # ── Hard stop from prospect → end call quickly ──
                        if _is_hard_stop(ai_text):
                            logger.info("HARD STOP detected from prospect — scheduling hangup in 4s")
                            if cc_id in _auto_hangup_tasks:
                                _auto_hangup_tasks[cc_id].cancel()
                            _auto_hangup_tasks[cc_id] = asyncio.create_task(_auto_hangup_after_goodbye(cc_id))
                        elif _is_booking_confirmed(ai_text):
                            logger.info("BOOKING confirmed by prospect — scheduling hangup in 4s")
                            if cc_id in _auto_hangup_tasks:
                                _auto_hangup_tasks[cc_id].cancel()
                            _auto_hangup_tasks[cc_id] = asyncio.create_task(_auto_hangup_after_goodbye(cc_id))
                        elif _is_goodbye(ai_text):
                            logger.info("Prospect said goodbye — keeping auto-hangup active")
                        elif cc_id in _auto_hangup_tasks:
                            # Prospect said something that's NOT an end-signal — cancel pending hangup
                            _auto_hangup_tasks[cc_id].cancel()
                            _auto_hangup_tasks.pop(cc_id, None)
                    else:
                        logger.info(f"AI-ASST SAID: \"{ai_text}\"")
                        _stop_filler_if_playing(cc_id)
                        _ai_agent_turn_times.setdefault(cc_id, []).append(time.time())
                        # Detect end-of-call signals from AI → schedule auto-hangup
                        if _is_booking_confirmed(ai_text):
                            logger.info("BOOKING confirmed by AI — scheduling hangup in 4s")
                            if cc_id in _auto_hangup_tasks:
                                _auto_hangup_tasks[cc_id].cancel()
                            _auto_hangup_tasks[cc_id] = asyncio.create_task(_auto_hangup_after_goodbye(cc_id))
                        elif _is_goodbye(ai_text):
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
                # Try to RESTART the AI Assistant rather than permanently
                # killing it. A single transient LLM/STT hiccup used to flip
                # ai_assistant=False forever → the call stayed alive but the
                # bot went silent for the rest of the conversation. One
                # automatic restart attempt recovers from ~all transient
                # errors; if it fails again, THEN we fall back to TTS.
                rec_active = active_calls.get(cc_id, {})
                name = rec_active.get("prospect_name") or "there"
                title = rec_active.get("prospect_title") or ""
                company = rec_active.get("company") or ""
                already_restarted = rec_active.get("_ai_restart_attempted", False)
                if not already_restarted:
                    rec_active["_ai_restart_attempted"] = True
                    logger.info("Attempting one-shot AI Assistant restart for %s", cc_id)
                    asyncio.create_task(_start_ai_assistant_fast(cc_id, name, title, company, background_tasks))
                else:
                    logger.warning("AI Assistant error after restart already tried — falling back to TTS for %s", cc_id)
                    rec_active["ai_assistant"] = False
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

            # Persist prospect turn to transcript (only on final to avoid dup interim noise)
            if is_final:
                rec = active_calls.get(cc_id)
                if rec:
                    rec.setdefault("transcript", []).append({"role": "prospect", "text": text})
                    save_call(rec)

            if not should_emit_transcription_reply(cc_id, text, is_final):
                return JSONResponse(content={"status": "ok"})
            if cc_id in speaking_calls:
                return JSONResponse(content={"status": "ok"})

            speaking_calls.add(cc_id)
            asyncio.create_task(_main_bg_transcription_reply(cc_id, text))
            return JSONResponse(content={"status": "ok"})

        # ── CONVERSATION CREATED → remember conversation_id so we can
        #    pull the full transcript from Telnyx at hangup. Telnyx does
        #    NOT always fire call.ai_assistant.transcription webhooks to
        #    us (seen in prod: assistant speaks, prospect replies, call
        #    ends with transcript_len=0). Polling the conversation API at
        #    hangup gives us a reliable transcript regardless. ──
        elif etype == "call.conversation.created":
            conv_id = (
                pl.get("conversation_id")
                or pl.get("id")
                or (pl.get("conversation") or {}).get("id")
            )
            if cc_id and conv_id:
                if cc_id in active_calls:
                    active_calls[cc_id]["conversation_id"] = conv_id
                update_call(cc_id, conversation_id=conv_id)
                logger.info("Captured conversation_id=%s for %s", conv_id, cc_id[:20])

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
                # Backfill transcript from Telnyx payload when local transcript is empty
                tx_src = (
                    pl.get("transcript")
                    or (insights_data.get("transcript") if isinstance(insights_data, dict) else None)
                    or pl.get("conversation")
                )
                backfilled: list[dict] = []
                if isinstance(tx_src, list):
                    for t in tx_src:
                        if not isinstance(t, dict):
                            continue
                        txt = (t.get("text") or t.get("content") or t.get("message") or "").strip()
                        if not txt:
                            continue
                        r = (t.get("role") or t.get("speaker") or "").lower()
                        role = "agent" if r in ("assistant", "agent", "ai", "bot") else "prospect"
                        backfilled.append({"role": role, "text": txt})
                elif isinstance(tx_src, str) and tx_src.strip():
                    backfilled.append({"role": "prospect", "text": tx_src.strip()})
                if rec:
                    rec["telnyx_insights"] = insights_data
                    if backfilled and not (rec.get("transcript") or []):
                        rec["transcript"] = backfilled
                        save_call(rec)
                        logger.info("Backfilled %d transcript turns from Telnyx insights for %s", len(backfilled), cc_id)
                update_call(cc_id, telnyx_insights=insights_data)
                if backfilled:
                    # Also persist to calls.json row if rec wasn't in active_calls
                    existing = get_call_by_control_id(cc_id) or {}
                    if not (existing.get("transcript") or []):
                        finalize_call_end(cc_id, transcript=backfilled)
                # Re-run Claude insights using Telnyx summary when local transcript is empty.
                asyncio.create_task(_generate_call_insights(cc_id))

        # ── RECORDING SAVED ────────────────────────────
        elif etype == "call.recording.saved":
            raw = event.get("raw") or pl or {}
            rec_urls = raw.get("recording_urls") or pl.get("recording_urls") or {}
            pub_urls = raw.get("public_recording_urls") or pl.get("public_recording_urls") or {}
            url = (
                rec_urls.get("mp3") or rec_urls.get("wav")
                or pub_urls.get("mp3") or pub_urls.get("wav")
                or raw.get("download_url") or pl.get("download_url")
            )
            if url:
                logger.info(f"Recording saved: {url}")
                if cc_id in active_calls:
                    active_calls[cc_id]["recording_url"] = url
                update_call(cc_id, recording_url=url)
                # Download & persist to disk IMMEDIATELY — the S3 pre-signed URL
                # expires in ~10 minutes so by the time the user clicks play
                # in the UI it would be dead. Saving bytes locally means the
                # recording stays playable forever.
                asyncio.create_task(_persist_recording_to_disk(cc_id, url))
            else:
                logger.warning("call.recording.saved had no URL — payload keys: %s", list(raw.keys())[:10])

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
            vm_task = _voicemail_watchdog_tasks.pop(hang_cc, None)
            if vm_task:
                vm_task.cancel()
            _ai_user_turn_times.pop(hang_cc, None)
            _ai_agent_turn_times.pop(hang_cc, None)
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

            # ── Backfill transcript from Telnyx (if webhooks were skipped) THEN run insights ──
            async def _backfill_then_insights(cc: str) -> None:
                try:
                    existing = (active_calls.get(cc) or get_call_by_control_id(cc) or {}).get("transcript") or []
                    if len(existing) < 2:
                        turns = await _fetch_telnyx_conversation_transcript(cc)
                        if turns:
                            r2 = active_calls.get(cc) or {}
                            r2["transcript"] = turns
                            active_calls[cc] = r2
                            finalize_call_end(cc, transcript=turns)
                except Exception:
                    logger.exception("Backfill Telnyx transcript failed for %s", cc)
                await _generate_call_insights(cc)

            asyncio.create_task(_backfill_then_insights(hang_cc))
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


async def _fetch_telnyx_conversation_transcript(cc_id: str) -> list[dict]:
    """
    Pull the full AI-Assistant conversation from Telnyx after hangup.

    Telnyx sometimes fails to deliver `call.ai_assistant.transcription`
    webhooks in real time (observed in prod — calls end with empty
    transcripts even though the audio clearly had a conversation). As a
    safety net we fetch the conversation via REST when the call ends,
    so the dashboard always shows the dialogue.

    Tries two endpoint shapes that Telnyx has used historically:
      1. GET /v2/ai/assistants/{id}/conversations/{conv}
      2. GET /v2/ai/conversations/{conv}

    Returns: list of {"role": "agent"|"prospect", "text": str}
    """
    import httpx

    rec = active_calls.get(cc_id) or get_call_by_control_id(cc_id) or {}
    conv_id = rec.get("conversation_id") if isinstance(rec, dict) else None
    if not conv_id:
        logger.info("No conversation_id stored for %s — skipping Telnyx transcript fetch", cc_id[:20])
        return []
    api_key = config.TELNYX_API_KEY
    if not api_key:
        return []
    headers = {"Authorization": f"Bearer {api_key}"}
    candidate_urls = [
        f"https://api.telnyx.com/v2/ai/assistants/{ASSISTANT_ID}/conversations/{conv_id}",
        f"https://api.telnyx.com/v2/ai/conversations/{conv_id}",
        f"https://api.telnyx.com/v2/ai/assistants/{ASSISTANT_ID}/conversations/{conv_id}/messages",
        f"https://api.telnyx.com/v2/ai/conversations/{conv_id}/messages",
    ]
    async with httpx.AsyncClient(timeout=15.0) as ac:
        for url in candidate_urls:
            try:
                r = await ac.get(url, headers=headers)
                if r.status_code >= 300:
                    continue
                data = r.json()
                # Telnyx wraps in {"data": ...} usually
                body = data.get("data") if isinstance(data.get("data"), (dict, list)) else data
                msgs = []
                if isinstance(body, dict):
                    msgs = (
                        body.get("messages")
                        or body.get("turns")
                        or body.get("conversation")
                        or body.get("transcript")
                        or []
                    )
                elif isinstance(body, list):
                    msgs = body
                turns: list[dict] = []
                for m in msgs or []:
                    if not isinstance(m, dict):
                        continue
                    txt = (m.get("content") or m.get("text") or m.get("message") or "").strip()
                    if not txt:
                        continue
                    # Skip internal priming messages used to brief the AI:
                    #  1. Our [BRIEFING] user-turn that seeds prospect context
                    #  2. The synthetic "Got it." acknowledgement we pair with it
                    #  3. Any system/tool/function message Telnyx echoes back
                    role_raw = (m.get("role") or m.get("speaker") or m.get("sender") or "").lower()
                    if role_raw in ("system", "tool", "function"):
                        continue
                    upper = txt.upper()
                    if "[BRIEFING]" in upper or upper.startswith("PROSPECT INFO") or upper.startswith("RESEARCH SUMMARY"):
                        continue
                    if txt.strip().lower() in ("got it.", "got it", "okay.", "okay", "understood.", "understood"):
                        continue
                    role = "agent" if role_raw in ("assistant", "agent", "ai", "bot") else "prospect"
                    turns.append({"role": role, "text": txt})
                if turns:
                    logger.info(
                        "Fetched %d transcript turns from Telnyx conversation API for %s (url=%s)",
                        len(turns), cc_id[:20], url.split("/v2/")[-1][:60],
                    )
                    return turns
            except Exception as e:
                logger.debug("Telnyx conv fetch attempt failed (%s): %s", url, e)
                continue
    logger.info("Telnyx conversation API returned no transcript for %s", cc_id[:20])
    return []


# Canonical outcome vocabulary — keep in sync with Tasks filter chips & Campaign filters.
CANONICAL_OUTCOMES = {
    "meeting_booked", "callback_scheduled", "interested",
    "not_interested", "no_answer", "voicemail",
    "gatekeeper", "hung_up", "do_not_call", "unknown", "no_conversation",
}


def _normalize_outcome(raw: Any) -> str:
    """Map free-form model outputs to the canonical outcome set."""
    if not raw:
        return "unknown"
    s = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    if s in CANONICAL_OUTCOMES:
        return s
    # Fuzzy aliases
    aliases = {
        "booked": "meeting_booked", "meeting": "meeting_booked", "demo_booked": "meeting_booked",
        "call_back": "callback_scheduled", "callback": "callback_scheduled", "call_later": "callback_scheduled",
        "positive": "interested", "qualified": "interested",
        "rejected": "not_interested", "declined": "not_interested", "dnc": "do_not_call",
        "vm": "voicemail", "machine": "voicemail",
        "no_one": "no_answer", "no_pickup": "no_answer", "missed": "no_answer",
        "hang_up": "hung_up", "hangup": "hung_up",
    }
    return aliases.get(s, "unknown")


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
"callback_at_utc": "ISO8601 UTC timestamp if prospect requested a callback at a specific time (e.g. 'call me back in 2 hours' or 'tomorrow at 3pm'), else empty string. Compute relative times from the current UTC time.",
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
            insights.setdefault("callback_at_utc", "")
            # Force outcome into the canonical vocabulary so filters & tasks match.
            insights["outcome"] = _normalize_outcome(insights.get("outcome"))
        # Persist normalized outcome at BOTH insights.outcome AND the top-level
        # `outcome` column so call-logs filters, Tasks grouping, and campaign
        # propagation all agree on a single value.
        top_outcome = insights.get("outcome") if isinstance(insights, dict) else "unknown"
        update_call(cc_id, insights=insights, outcome=top_outcome)
        if cc_id in active_calls:
            active_calls[cc_id]["insights"] = insights
            active_calls[cc_id]["outcome"] = top_outcome
        logger.info("Call insights generated for %s: outcome=%s, interest=%s%%",
                     cc_id[:20], insights.get("outcome"), insights.get("interest_level"))
        if isinstance(insights, dict) and insights.get("outcome") == "meeting_booked":
            _ensure_task_for_meeting(cc_id, rec if isinstance(rec, dict) else {}, insights)
        # ── Auto-create task based on outcome for the new Tasks sidebar ──
        if isinstance(insights, dict):
            _ensure_task_for_outcome(cc_id, rec if isinstance(rec, dict) else {}, insights)
        # ── Phase 3: email follow-ups + auto-callback queueing ──
        if isinstance(insights, dict):
            asyncio.create_task(_post_call_email_actions(
                rec if isinstance(rec, dict) else {}, insights, cc_id))
        # ── Propagate outcome back to campaign prospect + DNC ──
        try:
            r = rec if isinstance(rec, dict) else (active_calls.get(cc_id) or {})
            phone = r.get("to") or ""
            outcome = _normalize_outcome((insights or {}).get("outcome"))
            if phone:
                # Update EVERY campaign that contains this phone (rare to be in
                # multiple, but safe). Outcome flags are: meeting_booked,
                # interested, not_interested, callback, voicemail, no_answer, unknown.
                for camp in _load_campaigns():
                    _mark_campaign_prospect(
                        camp.get("id", ""), phone,
                        outcome=outcome,
                        status=("completed" if outcome != "unknown" else "dialed"),
                        ended_at=datetime.now().isoformat(),
                    )
                # Auto-add to DNC on clear rejection signals.
                if outcome in ("not_interested", "do_not_call"):
                    add_dnc(phone, reason=outcome)
        except Exception:
            logger.exception("Failed to propagate outcome to campaign/DNC for %s", cc_id)
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
            _persist_uploaded_docs()  # survive Railway restarts
            logger.info("Knowledge doc uploaded: %s (%d chars, persisted)", fname, len(text))
            # Re-sync the Telnyx AI Assistant so new KB is available on the next call
            try:
                _rebuild_hot_cache()
                sync_assistant_to_script()
            except Exception as e:
                logger.warning("Post-upload assistant sync failed (non-fatal): %s", e)
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
    _persist_uploaded_docs()
    try:
        _rebuild_hot_cache()
        sync_assistant_to_script()
    except Exception as e:
        logger.warning("Post-clear assistant sync failed (non-fatal): %s", e)
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
