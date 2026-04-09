"""
After a call ends: wait (default 5 min), draft a Sandler-style recap email with Claude, send via SMTP.

Requires: prospect email on the call (or in notes as `Email: x@y.com`, or matching Contacts phone),
ANTHROPIC_API_KEY, and a configured email provider (SMTP, SendGrid, Resend, or Mailgun) + EMAIL_FROM.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from anthropic import AsyncAnthropic

import config
import contacts_store
from email_sequences import email_delivery_ready, send_email_async
from storage import get_call_by_control_id, load_script, update_call

logger = logging.getLogger(__name__)


def _extract_email_from_notes(notes: str) -> str | None:
    m = re.search(r"Email:\s*([^\s|]+@[^\s|]+)", notes or "", re.I)
    if m:
        return m.group(1).strip().rstrip(">,)")
    return None


def resolve_prospect_email(rec: dict[str, Any]) -> str | None:
    for k in ("prospect_email", "email"):
        v = (rec.get(k) or "").strip()
        if v and "@" in v:
            return v
    from_notes = _extract_email_from_notes(str(rec.get("notes") or ""))
    if from_notes:
        return from_notes
    return contacts_store.find_email_by_phone_e164(str(rec.get("to") or ""))


def _transcript_text(transcript: list[Any]) -> str:
    lines: list[str] = []
    for t in transcript or []:
        if not isinstance(t, dict):
            continue
        role = (t.get("role") or "").lower()
        who = "Prospect" if role in ("prospect", "user") else "SDR"
        txt = (t.get("text") or "").strip()
        if txt:
            lines.append(f"{who}: {txt}")
    return "\n".join(lines)


def _should_skip_low_signal(rec: dict[str, Any]) -> bool:
    """Skip when there was effectively no live conversation."""
    tr = rec.get("transcript") or []
    prospect_lines = sum(
        1
        for t in tr
        if isinstance(t, dict)
        and (t.get("role") or "").lower() in ("prospect", "user")
        and len((t.get("text") or "").strip()) > 2
    )
    ins = rec.get("insights") if isinstance(rec.get("insights"), dict) else {}
    outcome = (ins.get("outcome") or rec.get("outcome") or "").lower()
    if outcome in ("voicemail", "no_answer", "no_conversation"):
        return True
    if prospect_lines == 0 and len(_transcript_text(tr)) < 30:
        return True
    return False


def _strip_json_fence(raw: str) -> str:
    t = (raw or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
    if t.endswith("```"):
        t = t.rsplit("```", 1)[0]
    t = t.strip()
    if t.lower().startswith("json"):
        t = t[4:].lstrip()
    return t.strip()


async def draft_sandler_followup_email(
    *,
    prospect_name: str,
    company: str,
    transcript_text: str,
    insights_summary: str,
    insights_outcome: str,
    sdr_name: str,
    sdr_company: str,
) -> tuple[str, str]:
    config.reload_secrets()
    key = config.ANTHROPIC_API_KEY
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    prompt = f"""You write ONE follow-up email after a live sales phone call. Use the **Sandler Selling Method** mindset:
- Reinforce rapport; sound like a human who listened — not a pitch deck.
- Briefly mirror what they shared (pain, context, or constraints) without being creepy.
- **No hard sell**, no feature dumps, no fake urgency. No "just following up" fluff.
- Tone: collaborative, adult-to-adult. Optional soft "up-front contract" line: happy to stay brief / they stay in control.
- If a next step was discussed (callback, meeting), reflect it naturally — do not invent meetings that are not supported below.

Prospect first name or name: {prospect_name}
Their company: {company}
Call outcome tag (if any): {insights_outcome or "unknown"}

AI / platform summary of the call (may be empty):
{insights_summary[:2000] if insights_summary else "(none)"}

Call transcript (may be partial):
{transcript_text[:8000] if transcript_text else "(no transcript)"}

You are {sdr_name} at {sdr_company}.

Return **only** valid JSON with exactly two keys (no markdown):
{{"subject": "short conversational subject under 65 chars", "body": "plain text email body, 3-7 short paragraphs or flowing lines, sign off with first name only"}}"""

    client = AsyncAnthropic(api_key=key)
    resp = await client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=900,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip()
    data = json.loads(_strip_json_fence(raw))
    if not isinstance(data, dict):
        raise ValueError("model did not return an object")
    subj = str(data.get("subject") or "Thanks for the conversation").strip()
    body = str(data.get("body") or "").strip()
    if not body:
        raise ValueError("empty body")
    return subj, body


async def run_post_call_followup_email(call_control_id: str) -> None:
    try:
        delay = max(60, int(config.POST_CALL_FOLLOWUP_DELAY_SEC or 300))
        await asyncio.sleep(float(delay))
        config.reload_secrets()
        if not config.POST_CALL_FOLLOWUP_EMAIL_ENABLED:
            return
        if not email_delivery_ready():
            logger.warning(
                "Post-call follow-up email skipped (cc_id=%s): email delivery not configured",
                call_control_id[:16],
            )
            return

        rec = get_call_by_control_id(call_control_id)
        if not rec:
            logger.warning("Post-call email: no record for %s", call_control_id[:16])
            return
        if rec.get("post_call_followup_email_sent"):
            return

        to_email = resolve_prospect_email(rec)
        if not to_email:
            logger.info(
                "Post-call email skipped (cc_id=%s): no prospect email (set on dial, notes, or Contacts)",
                call_control_id[:16],
            )
            return

        # Re-fetch so insights from parallel task are usually present by now
        await asyncio.sleep(2.0)
        rec = get_call_by_control_id(call_control_id) or rec

        if _should_skip_low_signal(rec):
            logger.info("Post-call email skipped (cc_id=%s): low-signal / voicemail / no conversation", call_control_id[:16])
            update_call(
                call_control_id,
                post_call_followup_email_skipped="voicemail_or_no_conversation",
            )
            return

        script = load_script()
        sdr = script.get("sdr_name") or config.SDR_NAME or "Alex"
        co = script.get("company_name") or config.COMPANY_NAME or "our team"
        name = (rec.get("prospect_name") or "there").strip() or "there"
        company = (rec.get("company") or "").strip()
        ins = rec.get("insights") if isinstance(rec.get("insights"), dict) else {}
        summary = (ins.get("summary") or "").strip()
        outcome = (ins.get("outcome") or rec.get("outcome") or "").strip()
        tr_text = _transcript_text(rec.get("transcript") or [])

        subject, body = await draft_sandler_followup_email(
            prospect_name=name,
            company=company,
            transcript_text=tr_text,
            insights_summary=summary,
            insights_outcome=outcome,
            sdr_name=sdr,
            sdr_company=co,
        )

        await send_email_async(to_email, subject, body)
        now = datetime.now(timezone.utc).isoformat()
        update_call(
            call_control_id,
            post_call_followup_email_sent=True,
            post_call_followup_email_at=now,
            post_call_followup_email_to=to_email,
            post_call_followup_email_subject=subject,
        )
        logger.info("Post-call Sandler follow-up email sent to %s (cc_id=%s)", to_email, call_control_id[:16])
    except Exception:
        logger.exception("Post-call follow-up email failed for %s", call_control_id[:16])
        try:
            update_call(call_control_id, post_call_followup_email_error="send_or_draft_failed")
        except Exception:
            pass
