"""
AWS Polly TTS handler — drop-in alternative to Telnyx native speak() and
ElevenLabs TTS. ~95% cheaper than ElevenLabs Pro and ~100% cheaper than
Telnyx TTS at scale.

Pricing (us-east-1, 2026):
  - Standard voices:  $0.000004 / character  (= $4.00 per 1M chars)
  - Neural voices:    $0.000016 / character  (= $16.00 per 1M chars)
  - Long-form voices: $0.000100 / character  (= $100.00 per 1M chars)

Compare:
  - ElevenLabs Pro:   $0.198 / 1k chars  = $198 per 1M chars  (12× more than neural)
  - Telnyx TTS:       ~$0.005 / min      ≈ $0.50 per 1k call-mins

Setup:
  pip install boto3
  Set env vars:
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_REGION (default: us-east-1)
    AWS_POLLY_VOICE (default: Joanna)
    AWS_POLLY_ENGINE (default: neural)  # one of: standard | neural | long-form
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_polly_client: Any = None


def _get_client():
    """Lazy-init Polly client. Raises clear error if boto3 missing."""
    global _polly_client
    if _polly_client is not None:
        return _polly_client
    try:
        import boto3
    except ImportError as e:
        raise RuntimeError(
            "boto3 is not installed. Run: pip install boto3"
        ) from e
    region = os.environ.get("AWS_REGION", "us-east-1")
    _polly_client = boto3.client(
        "polly",
        region_name=region,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID") or None,
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY") or None,
    )
    return _polly_client


def is_configured() -> bool:
    """True if AWS Polly env vars are set so we can actually call it."""
    return bool(
        os.environ.get("AWS_ACCESS_KEY_ID")
        and os.environ.get("AWS_SECRET_ACCESS_KEY")
    )


def estimate_cost_per_1k_chars(engine: str | None = None) -> float:
    """USD per 1,000 characters."""
    eng = (engine or os.environ.get("AWS_POLLY_ENGINE", "neural")).lower()
    return {
        "standard":  0.004,
        "neural":    0.016,
        "long-form": 0.100,
    }.get(eng, 0.016)


def synthesize_speech(
    text: str,
    voice_id: str | None = None,
    engine: str | None = None,
    output_format: str = "mp3",
) -> bytes:
    """Synthesize speech via Polly. Returns raw audio bytes."""
    if not text or not text.strip():
        return b""
    voice = voice_id or os.environ.get("AWS_POLLY_VOICE", "Joanna")
    eng   = engine   or os.environ.get("AWS_POLLY_ENGINE", "neural")
    try:
        client = _get_client()
        resp = client.synthesize_speech(
            Text=text,
            OutputFormat=output_format,
            VoiceId=voice,
            Engine=eng,
        )
        audio = resp.get("AudioStream")
        if audio is None:
            return b""
        data = audio.read()
        logger.info(
            "Polly synth: voice=%s engine=%s chars=%d bytes=%d cost=$%.6f",
            voice, eng, len(text), len(data),
            (len(text) / 1000.0) * estimate_cost_per_1k_chars(eng),
        )
        return data
    except Exception:
        logger.exception("Polly synthesize_speech failed")
        raise


def list_voices(language_code: str | None = None) -> list[dict[str, Any]]:
    """Return Polly voice catalog. Useful for tenant voice picker."""
    try:
        client = _get_client()
        kwargs: dict[str, Any] = {}
        if language_code:
            kwargs["LanguageCode"] = language_code
        resp = client.describe_voices(**kwargs)
        return resp.get("Voices") or []
    except Exception:
        logger.exception("Polly list_voices failed")
        return []
