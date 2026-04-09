"""
LLM brain for live outbound SDR — short, conversational turns (2–3 sentences).
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator
from typing import Any

from anthropic import AsyncAnthropic

import config
from knowledge_base import get_full_knowledge
from storage import load_script

logger = logging.getLogger(__name__)

# Cap injected KB so phone turns stay within latency/token budget
_KB_IN_PROMPT_MAX_CHARS = 48000
# Cap playbook block (topics + discovery + objections + booking) so phone prompt stays bounded
_SCRIPT_PLAYBOOK_MAX_CHARS = 8000


def script_playbook_block(s: dict[str, Any]) -> str:
    """Format call_topics, discovery_questions, objections, booking for prompts."""
    parts: list[str] = []
    topics = s.get("call_topics")
    if isinstance(topics, list):
        lines = [t.strip() for t in topics if isinstance(t, str) and t.strip()]
        if lines:
            parts.append(
                "CALL TOPICS (work these in naturally across the conversation; never read as a checklist):\n"
                + "\n".join(f"• {line}" for line in lines)
            )
    qs = s.get("discovery_questions")
    if isinstance(qs, list):
        ql = [q.strip() for q in qs if isinstance(q, str) and q.strip()]
        if ql:
            parts.append(
                "DISCOVERY QUESTION BANK (at most one per turn):\n" + "\n".join(f"• {q}" for q in ql)
            )
    obj = s.get("objections")
    if isinstance(obj, dict):
        olines = []
        for key, label in (
            ("not_interested", "not interested"),
            ("send_email", "send email"),
            ("call_back", "call back later"),
            ("have_solution", "already have a solution"),
            ("no_budget", "no budget"),
            ("manage_fine", "all set on Manage / no SaaS-mgmt need — pivot to Migrate"),
        ):
            v = obj.get(key)
            if isinstance(v, str) and v.strip():
                olines.append(f"• [{label}] {v.strip()}")
        if olines:
            parts.append("OBJECTION GIST:\n" + "\n".join(olines))
    bp = s.get("booking_phrase")
    if isinstance(bp, str) and bp.strip():
        parts.append(f"WHEN ASKING FOR THE MEETING (adapt wording):\n{bp.strip()}")
    return "\n\n".join(parts)


def script_playbook_compact(s: dict[str, Any], limit: int = 1400) -> str:
    """Shorter playbook for Telnyx assistant instructions (token budget)."""
    b = script_playbook_block(s)
    if len(b) <= limit:
        return b
    return b[: limit - 1] + "…"


def build_system_prompt() -> str:
    """Build the Claude system prompt from script config (kept compact for lower TTFT on phone)."""
    s = load_script()
    sdr_name    = s.get("sdr_name", "Alex")
    company     = s.get("company_name", "Your Company")
    objective   = s.get("call_objective", "Book a 15-minute discovery call")
    persona     = s.get("target_persona", "B2B decision makers")
    value_prop  = s.get("value_proposition", "")
    opening     = s.get("opening_line", "")
    extra       = s.get("additional_instructions", "")

    prompt = f"""You are {sdr_name}, SDR at {company} on a LIVE call. Objective: {objective}. Audience: {persona}.
Context: {value_prop}
Opening already used: {opening}

CRITICAL — TALK LESS: Your turns are usually ONE short sentence OR ONE question (not both unless the whole thing is under ~12 seconds spoken). Do NOT deliver product training, stacked features, or multiple ideas in one turn. One insight → stop → question OR meeting ask.

QUESTION DISCIPLINE: Ask exactly ONE discovery question at a time; wait for their answer. Prefer questions that move toward a 15-minute meeting (migrations, scale, AI launch, who decides).

SPEED: As soon as there is a credible hook, propose a specific short meeting time; do not keep explaining to "earn" the meeting.

STYLE: Plain English, human, curious — never say you are an AI. Brief fillers OK ("Got it—", "Makes sense—"). If busy → callback. If not interested → one light probe then exit.

Reply with ONLY your next spoken words. No bullets or lists in speech.
"""
    pb = script_playbook_block(s)
    if pb:
        if len(pb) > _SCRIPT_PLAYBOOK_MAX_CHARS:
            pb = pb[:_SCRIPT_PLAYBOOK_MAX_CHARS] + "\n...[playbook truncated]..."
        prompt += f"\n{pb}\n"
    if extra:
        prompt += f"\nExtra: {extra}\n"
    kb = get_full_knowledge()
    if kb:
        if len(kb) > _KB_IN_PROMPT_MAX_CHARS:
            kb = kb[:_KB_IN_PROMPT_MAX_CHARS] + "\n...[knowledge truncated]..."
        prompt += (
            "\nINTERNAL KNOWLEDGE (answer factual questions from this; speak naturally; "
            "never read bullet lists verbatim):\n"
            f"{kb}\n"
        )
    prompt += "Output spoken lines only — no bullets, labels, or surrounding quotes."
    return prompt

_client: AsyncAnthropic | None = None
_cached_anthropic_key: str | None = None


def _get_client() -> AsyncAnthropic:
    global _client, _cached_anthropic_key
    key = config.ANTHROPIC_API_KEY
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    if _client is None or _cached_anthropic_key != key:
        _client = AsyncAnthropic(api_key=key)
        _cached_anthropic_key = key
    return _client


def opening_line(prospect_name: str) -> str:
    s    = load_script()
    name = (prospect_name or "there").strip() or "there"
    sdr  = s.get("sdr_name", "Alex")
    co   = s.get("company_name", "our company")
    tmpl = s.get("opening_line", "Hi {name}, this is {sdr_name} from {company} — quick question: did I catch you at an okay time?")
    return tmpl.replace("{name}", name).replace("{sdr_name}", sdr).replace("{company}", co)


def strip_wrapping_quotes(text: str) -> str:
    t = text.strip()
    if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
        t = t[1:-1].strip()
    return t


def sanitize_reply(text: str) -> str:
    t = strip_wrapping_quotes(text)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > 320:
        t = t[:317] + "..."
    return t


def join_streamed_reply_parts(parts: list[str]) -> str:
    reply = " ".join(p.strip() for p in parts if p and p.strip()).strip()
    if len(reply) >= 2 and reply[0] in '"\'':
        if reply[-1] == reply[0]:
            reply = reply[1:-1].strip()
    return reply


def pop_first_speakable_chunk(buffer: str) -> tuple[str | None, str]:
    """
    Split streaming LLM text into the next chunk safe to send to TTS.
    Prefers sentence boundaries; otherwise breaks at word boundary after ~96 chars.
    """
    if not buffer:
        return None, buffer
    for sep in (". ", "? ", "! ", "\n"):
        i = buffer.find(sep)
        if i != -1:
            chunk = buffer[: i + len(sep)].strip()
            rest = buffer[i + len(sep) :]
            return (chunk if chunk else None), rest
    if len(buffer) >= 96:
        cut = buffer[:96].rfind(" ")
        if cut > 12:
            chunk = buffer[:cut].strip()
            rest = buffer[cut + 1 :]
            return chunk, rest
    return None, buffer


async def stream_sdr_reply_sentences(
    conversation: list[dict[str, Any]],
) -> AsyncIterator[str]:
    """
    Stream Claude tokens, yielding speakable chunks as soon as boundaries allow.
    Caller must already have appended the latest user message to `conversation`.
    Does not append the assistant message — caller joins yielded parts and appends.
    """
    client = _get_client()
    sys_text = build_system_prompt()
    system_blocks: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": sys_text,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    buffer = ""

    async def _consume_stream(sys: str | list) -> AsyncIterator[str]:
        nonlocal buffer
        async with client.messages.stream(
            model=config.phone_reply_model(),
            max_tokens=config.ANTHROPIC_MAX_TOKENS_REPLY,
            system=sys,
            messages=conversation,
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
        async for part in _consume_stream(system_blocks):
            yield part
    except Exception as e:
        logger.warning("stream_sdr_reply_sentences without prompt cache (%s)", e)
        buffer = ""
        async for part in _consume_stream(sys_text):
            yield part

    if buffer.strip():
        yield buffer.strip()


async def next_sdr_reply(conversation: list[dict[str, Any]]) -> str:
    """Return the next agent utterance from transcript history (streaming for lower time-to-complete)."""
    client = _get_client()
    sys_text = build_system_prompt()
    system = [
        {
            "type": "text",
            "text": sys_text,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    full_text = ""
    try:
        async with client.messages.stream(
            model=config.phone_reply_model(),
            max_tokens=config.ANTHROPIC_MAX_TOKENS_REPLY,
            system=system,
            messages=conversation,
        ) as stream:
            async for chunk in stream.text_stream:
                full_text += chunk
    except Exception as e:
        logger.warning("Live reply stream without prompt cache (%s)", e)
        full_text = ""
        async with client.messages.stream(
            model=config.phone_reply_model(),
            max_tokens=config.ANTHROPIC_MAX_TOKENS_REPLY,
            system=sys_text,
            messages=conversation,
        ) as stream:
            async for chunk in stream.text_stream:
                full_text += chunk
    if not full_text.strip():
        return "Thanks for taking the call — I'll keep this brief. What's the biggest headache on your plate this quarter?"
    return sanitize_reply(full_text)


def transcript_from_deepgram(result: Any) -> str:
    """Extract final transcript text from Deepgram LiveResultResponse."""
    ch = getattr(result, "channel", None)
    if ch is None:
        return ""
    if isinstance(ch, list):
        ch = ch[0] if ch else None
    if ch is None:
        return ""
    alts = getattr(ch, "alternatives", None) or []
    if not alts:
        return ""
    t = getattr(alts[0], "transcript", "") or ""
    return t.strip()
