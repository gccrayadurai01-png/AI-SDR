"""
Minimal test: make a call, handle webhooks, test transcription.
Run this INSTEAD of main.py to debug the voice issue.
"""
import asyncio
import json
import logging
import os
import telnyx
from telnyx import omit
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

app = FastAPI()

API_KEY = os.getenv("TELNYX_API_KEY", "")
CONNECTION_ID = os.getenv("TELNYX_CONNECTION_ID", "")
FROM_NUMBER = os.getenv("TELNYX_PHONE_NUMBER", "")
BASE_URL = os.getenv("APP_BASE_URL", "")

client = telnyx.Telnyx(api_key=API_KEY)

# Track state
answered = set()
spoke_opening = set()


@app.post("/webhooks/telnyx")
async def webhook(request: Request):
    body = await request.json()
    data = body.get("data", {})
    etype = data.get("event_type", "?")
    payload = data.get("payload", {})
    cc_id = payload.get("call_control_id", "?")

    log.info(f"=== EVENT: {etype} === call: {cc_id[:20]}...")

    # ── CALL ANSWERED ──
    if etype == "call.answered":
        if cc_id in answered:
            log.info("SKIP: duplicate call.answered")
            return JSONResponse({"status": "ok"})
        answered.add(cc_id)

        log.info("Speaking opening line...")
        try:
            client.calls.actions.speak(
                call_control_id=cc_id,
                payload="Hi there! This is a test. Please say something after I finish speaking.",
                voice="female",
                language="en-US",
            )
            spoke_opening.add(cc_id)
            log.info("Speak command sent OK")
        except Exception as e:
            log.error(f"Speak failed: {e}")

    # ── SPEAK FINISHED ──
    elif etype == "call.speak.ended":
        log.info("Speak finished.")

        # Only start transcription AFTER opening line finishes
        if cc_id in spoke_opening:
            spoke_opening.discard(cc_id)
            log.info("Starting transcription...")
            try:
                client.calls.actions.start_transcription(
                    call_control_id=cc_id,
                    transcription_tracks="inbound",
                )
                log.info("Transcription started OK!")
            except Exception as e:
                log.error(f"Transcription failed: {e}")
                # Try without params
                try:
                    client.calls.actions.start_transcription(
                        call_control_id=cc_id,
                    )
                    log.info("Transcription started (no params) OK!")
                except Exception as e2:
                    log.error(f"Transcription fallback failed: {e2}")

    # ── TRANSCRIPTION EVENT ──
    elif etype == "call.transcription":
        td = payload.get("transcription_data", {})
        text = td.get("transcript", "")
        is_final = td.get("is_final", False)
        log.info(f"TRANSCRIPTION ({'FINAL' if is_final else 'interim'}): \"{text}\"")

        if is_final and text.strip():
            log.info(f"*** HEARD FINAL: \"{text}\" ***")
            # Respond with confirmation
            try:
                client.calls.actions.speak(
                    call_control_id=cc_id,
                    payload=f"I heard you say: {text}. That's great! The transcription is working.",
                    voice="AWS.Polly.Matthew-Neural",
                    language=omit,
                    service_level="premium",
                )
                log.info("Replied OK!")
            except Exception as e:
                log.error(f"Reply speak failed: {e}")

    # ── HANGUP ──
    elif etype == "call.hangup":
        log.info("Call ended.")
        answered.discard(cc_id)
        spoke_opening.discard(cc_id)

    else:
        log.info(f"(unhandled event: {etype})")

    return JSONResponse({"status": "ok"})


@app.get("/")
async def home():
    return {"status": "test server running"}


@app.post("/call")
async def make_call(request: Request):
    body = await request.json()
    to = body.get("to", "+12524015699")
    log.info(f"Dialing {to}...")
    result = client.calls.dial(
        connection_id=CONNECTION_ID,
        to=to,
        from_=FROM_NUMBER,
        webhook_url=f"{BASE_URL}/webhooks/telnyx",
    )
    cc_id = result.data.call_control_id
    log.info(f"Call initiated: {cc_id}")
    return {"call_control_id": cc_id}


if __name__ == "__main__":
    log.info("TEST SERVER starting on port 8000...")
    log.info(f"Webhook URL: {BASE_URL}/webhooks/telnyx")
    uvicorn.run(app, host="0.0.0.0", port=8000)
