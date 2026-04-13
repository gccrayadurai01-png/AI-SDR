"""
Telnyx WebSocket (μ-law) → Deepgram live STT → Claude → Telnyx speak.
Uses raw websockets to connect to Deepgram (no SDK dependency issues).
"""
from __future__ import annotations

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any

import websockets

import config
from sdr_agent import join_streamed_reply_parts, next_sdr_reply, stream_sdr_reply_sentences
from telnyx_handler import speak_on_call

logger = logging.getLogger(__name__)

# call_control_id -> session
_sessions: dict[str, CallAudioSession] = {}
_stt_missing_logged = False

DEEPGRAM_WS_URL = (
    "wss://api.deepgram.com/v1/listen"
    "?encoding=mulaw"
    "&sample_rate=8000"
    "&channels=1"
    "&model=nova-2"
    "&punctuate=true"
    "&smart_format=true"
    "&interim_results=false"
    "&endpointing=400"
    "&utterance_end_ms=1500"
)


def get_session(call_control_id: str) -> CallAudioSession | None:
    return _sessions.get(call_control_id)


class CallAudioSession:
    """Manages real-time STT → LLM → TTS for one phone call."""

    def __init__(
        self,
        call_control_id: str,
        prospect_name: str,
        active_calls_ref: dict[str, Any],
    ) -> None:
        self.call_control_id = call_control_id
        self.prospect_name = prospect_name
        self._active_calls = active_calls_ref
        self._ws: Any = None
        self._lock = asyncio.Lock()
        self._listener_task: asyncio.Task | None = None
        self.conversation: list[dict[str, Any]] = []
        self._closed = False

    async def start(self) -> bool:
        """Open WebSocket to Deepgram and start listening for transcripts."""
        try:
            extra_headers = {"Authorization": f"Token {config.DEEPGRAM_API_KEY}"}
            self._ws = await websockets.connect(
                DEEPGRAM_WS_URL,
                additional_headers=extra_headers,
                ping_interval=20,
                ping_timeout=10,
            )
            logger.info(f"🎙️ Deepgram WS connected for {self.call_control_id}")

            # Start background task to listen for transcript results
            self._listener_task = asyncio.create_task(self._listen_loop())
            return True

        except Exception as e:
            logger.error(f"❌ Deepgram connect failed: {e}")
            return False

    async def _listen_loop(self) -> None:
        """Background loop: read transcript results from Deepgram."""
        try:
            async for message in self._ws:
                if self._closed:
                    break
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    continue

                # Only process final transcripts (not interim)
                if not data.get("is_final", False):
                    continue

                # Extract transcript text
                text = self._extract_transcript(data)
                if not text:
                    continue

                logger.info(f"👂 Heard: \"{text}\"")
                await self._on_final_transcript(text)

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"🔌 Deepgram WS closed for {self.call_control_id}")
        except Exception as e:
            if not self._closed:
                logger.error(f"❌ Deepgram listener error: {e}")

    def _extract_transcript(self, data: dict) -> str:
        """Pull transcript text from Deepgram JSON response."""
        try:
            channel = data.get("channel", {})
            alternatives = channel.get("alternatives", [])
            if alternatives:
                return (alternatives[0].get("transcript") or "").strip()
        except Exception:
            pass
        return ""

    async def _on_final_transcript(self, text: str) -> None:
        """Prospect said something → send to Claude → speak response."""
        async with self._lock:
            rec = self._active_calls.get(self.call_control_id)
            if rec is None or rec.get("state") == "ended":
                return

            # Log prospect utterance
            tlist = rec.setdefault("transcript", [])
            tlist.append({"role": "prospect", "text": text})
            self.conversation.append({"role": "user", "content": text})

            # Get AI response from Claude
            try:
                logger.info(f"🧠 Claude thinking... ({len(self.conversation)} turns)")
                if config.STREAM_SPEECH_PIPELINE:
                    parts: list[str] = []
                    async for sent in stream_sdr_reply_sentences(self.conversation):
                        parts.append(sent)
                        await speak_on_call(self.call_control_id, sent)
                    reply = join_streamed_reply_parts(parts)
                    if not reply:
                        reply = (
                            "Sorry, I'm having a rough connection. Mind if I try you again tomorrow?"
                        )
                else:
                    reply = await next_sdr_reply(self.conversation)
                logger.info(f"🤖 AI says: \"{reply}\"")
            except Exception as e:
                logger.exception(f"SDR reply failed: {e}")
                try:
                    await speak_on_call(
                        self.call_control_id,
                        "Sorry, I'm having a rough connection. Mind if I try you again tomorrow?",
                    )
                except Exception:
                    pass
                return

            tlist.append({"role": "agent", "text": reply})
            self.conversation.append({"role": "assistant", "content": reply})

            if not config.STREAM_SPEECH_PIPELINE:
                try:
                    await speak_on_call(self.call_control_id, reply)
                except Exception as e:
                    logger.exception(f"speak_on_call failed: {e}")

    async def send_audio(self, chunk: bytes) -> None:
        """Send raw audio bytes to Deepgram for transcription."""
        if self._ws and not self._closed and chunk:
            try:
                await self._ws.send(chunk)
            except Exception as e:
                logger.warning(f"Audio send failed: {e}")

    async def close(self) -> None:
        """Clean up: close WebSocket and cancel listener."""
        self._closed = True
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
        try:
            if self._ws:
                await self._ws.close()
        except Exception as e:
            logger.warning(f"Deepgram close: {e}")
        self._ws = None
        logger.info(f"🔇 Session closed: {self.call_control_id}")


async def ensure_session(
    call_control_id: str,
    active_calls: dict[str, Any],
) -> CallAudioSession | None:
    global _stt_missing_logged
    if not config.DEEPGRAM_API_KEY:
        if not _stt_missing_logged:
            logger.warning("DEEPGRAM_API_KEY missing — skipping live STT/LLM")
            _stt_missing_logged = True
        return None

    if call_control_id in _sessions:
        return _sessions[call_control_id]

    meta = active_calls.get(call_control_id) or {}
    name = meta.get("prospect_name") or "there"
    session = CallAudioSession(call_control_id, name, active_calls)
    started = await session.start()
    if started:
        _sessions[call_control_id] = session
        logger.info(f"✅ Voice pipeline ready for {call_control_id}")
        return session
    return None


async def end_session(call_control_id: str) -> None:
    s = _sessions.pop(call_control_id, None)
    if s:
        await s.close()


def parse_telnyx_media_payload(data: dict) -> tuple[bytes | None, str | None]:
    """Decode Telnyx WebSocket media event → (audio_bytes, track)."""
    media = data.get("media") or {}
    if isinstance(media, dict):
        b64 = media.get("payload")
        track = media.get("track")
    else:
        b64 = data.get("payload")
        track = data.get("track")

    if not b64:
        return None, track
    try:
        return base64.b64decode(b64), track
    except Exception:
        logger.warning("Invalid base64 audio chunk")
        return None, track


def parse_start_call_control_id(data: dict) -> str | None:
    st = data.get("start") or {}
    if isinstance(st, dict):
        return st.get("call_control_id")
    return data.get("call_control_id")
