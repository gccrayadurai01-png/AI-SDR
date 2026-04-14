from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Optional

from dotenv import dotenv_values, load_dotenv

# Always load .env from this project folder (not only cwd -- fixes "keys not working" when uvicorn/IDE starts elsewhere)
_ENV_FILE = Path(__file__).resolve().parent / ".env"
_ENV_ENCODING = "utf-8-sig"  # strip BOM if present (Windows editors)

load_dotenv(_ENV_FILE, encoding=_ENV_ENCODING)


def _env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read env var; strip whitespace and optional surrounding quotes (common .env mistakes)."""
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip()
    if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
        v = v[1:-1].strip()
    return v if v else default


def _normalize_e164_phone(raw: Optional[str]) -> Optional[str]:
    """Normalize Telnyx caller ID to +E.164 (digits only after +)."""
    if not raw:
        return None
    s = str(raw).strip()
    digits = re.sub(r"\D", "", s)
    if not digits:
        return None
    if s.startswith("+"):
        return "+" + digits
    if len(digits) == 10:
        return "+1" + digits
    if len(digits) == 11 and digits[0] == "1":
        return "+" + digits
    if len(digits) >= 10:
        return "+" + digits
    return None


def _env_int(name: str, default: int) -> int:
    v = _env_str(name)
    if v is None:
        return default
    try:
        return max(32, min(1024, int(v.strip())))
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    v = _env_str(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


# Telnyx
TELNYX_API_KEY = _env_str("TELNYX_API_KEY")
TELNYX_PUBLIC_KEY = _env_str("TELNYX_PUBLIC_KEY")
TELNYX_PHONE_NUMBER = _normalize_e164_phone(_env_str("TELNYX_PHONE_NUMBER"))
_cid = (_env_str("TELNYX_CONNECTION_ID") or "").strip()
TELNYX_CONNECTION_ID = _cid if _cid else None

# Cartesia
CARTESIA_API_KEY     = _env_str("CARTESIA_API_KEY")

# Apollo.io
APOLLO_API_KEY       = _env_str("APOLLO_API_KEY")

# Anthropic
ANTHROPIC_API_KEY    = _env_str("ANTHROPIC_API_KEY")
# Default: Claude Sonnet 4 snapshot (3.5 Sonnet 20241022 is retired)
ANTHROPIC_MODEL      = _env_str("ANTHROPIC_MODEL") or "claude-sonnet-4-20250514"
# Live phone turns: set ANTHROPIC_LIVE_MODEL to override. If unset, uses Haiku.
ANTHROPIC_LIVE_MODEL = _env_str("ANTHROPIC_LIVE_MODEL")
# Fallback when ANTHROPIC_LIVE_MODEL is empty (Haiku is fastest)
ANTHROPIC_PHONE_MODEL_DEFAULT = _env_str("ANTHROPIC_PHONE_MODEL_DEFAULT") or "claude-haiku-4-5"
# Shorter replies = lower latency & cost on phone (tune via ANTHROPIC_MAX_TOKENS_REPLY)
ANTHROPIC_MAX_TOKENS_REPLY = _env_int("ANTHROPIC_MAX_TOKENS_REPLY", 88)

# Telnyx speak
TELNYX_SPEAK_VOICE = _env_str("TELNYX_SPEAK_VOICE") or "AWS.Polly.Joanna-Neural"
_sl = (_env_str("TELNYX_SPEAK_SERVICE_LEVEL") or "premium").strip().lower()
TELNYX_SPEAK_SERVICE_LEVEL: str = "basic" if _sl == "basic" else "premium"
TELNYX_SPEAK_LANGUAGE = _env_str("TELNYX_SPEAK_LANGUAGE") or "en-US"

# ElevenLabs
ELEVENLABS_VOICE_ID = _env_str("ELEVENLABS_VOICE_ID")
ELEVENLABS_MODEL_ID = _env_str("ELEVENLABS_MODEL_ID") or "eleven_multilingual_v2"
ELEVENLABS_API_KEY_REF = _env_str("ELEVENLABS_API_KEY_REF")
ELEVENLABS_API_KEY = _env_str("ELEVENLABS_API_KEY")
ELEVENLABS_DIRECT_FIRST = _env_bool("ELEVENLABS_DIRECT_FIRST", True)
ELEVENLABS_HTTP_STREAM = _env_bool("ELEVENLABS_HTTP_STREAM", False)
STREAM_SPEECH_PIPELINE = _env_bool("STREAM_SPEECH_PIPELINE", False)
TRANSCRIPTION_REPLY_ON_INTERIM = _env_bool("TRANSCRIPTION_REPLY_ON_INTERIM", True)
PHONE_THINK_FILLER = _env_bool("PHONE_THINK_FILLER", False)

# SDR persona
COMPANY_NAME         = os.getenv("COMPANY_NAME", "Your Company")
SDR_NAME             = os.getenv("SDR_NAME", "Alex")

# Call-learned Q/A KB (persists Q/A from calls)
QA_KB_ENABLED = _env_bool("QA_KB_ENABLED", True)
_qa_min = _env_str("QA_KB_MIN_SCORE")
try:
    QA_KB_MIN_SCORE = float(_qa_min.strip()) if _qa_min else 0.82
    if not (0.0 <= QA_KB_MIN_SCORE <= 1.0):
        QA_KB_MIN_SCORE = 0.82
except (ValueError, AttributeError):
    QA_KB_MIN_SCORE = 0.82

# Email outbound: smtp | sendgrid | resend | mailgun | gmail_oauth | outlook_oauth
_ep_raw = (_env_str("EMAIL_PROVIDER") or "smtp").strip().lower()
EMAIL_PROVIDER = (
    _ep_raw
    if _ep_raw in ("smtp", "sendgrid", "resend", "mailgun", "gmail_oauth", "outlook_oauth")
    else "smtp"
)
GOOGLE_OAUTH_CLIENT_ID = _env_str("GOOGLE_OAUTH_CLIENT_ID")
GOOGLE_OAUTH_CLIENT_SECRET = _env_str("GOOGLE_OAUTH_CLIENT_SECRET")
MICROSOFT_OAUTH_CLIENT_ID = _env_str("MICROSOFT_OAUTH_CLIENT_ID")
MICROSOFT_OAUTH_CLIENT_SECRET = _env_str("MICROSOFT_OAUTH_CLIENT_SECRET")
MICROSOFT_OAUTH_TENANT = _env_str("MICROSOFT_OAUTH_TENANT") or "common"
SENDGRID_API_KEY = _env_str("SENDGRID_API_KEY")
RESEND_API_KEY = _env_str("RESEND_API_KEY")
MAILGUN_API_KEY = _env_str("MAILGUN_API_KEY")
MAILGUN_DOMAIN = _env_str("MAILGUN_DOMAIN")
_mgb = (_env_str("MAILGUN_API_BASE") or "https://api.mailgun.net").strip().rstrip("/")
MAILGUN_API_BASE = _mgb if _mgb.startswith("http") else "https://api.mailgun.net"

# Email sequences (SMTP — when EMAIL_PROVIDER=smtp)
SMTP_HOST = _env_str("SMTP_HOST")
_smtp_port_raw = _env_str("SMTP_PORT")
try:
    SMTP_PORT = int(_smtp_port_raw.strip()) if _smtp_port_raw else 587
    if not (1 <= SMTP_PORT <= 65535):
        SMTP_PORT = 587
except (ValueError, AttributeError):
    SMTP_PORT = 587
SMTP_USER = _env_str("SMTP_USER")
SMTP_PASSWORD = _env_str("SMTP_PASSWORD")
EMAIL_FROM = _env_str("EMAIL_FROM")
SMTP_USE_TLS = _env_bool("SMTP_USE_TLS", True)
EMAIL_AUTOMATION_ENABLED = _env_bool("EMAIL_AUTOMATION_ENABLED", False)
_tick_raw = _env_str("EMAIL_SEQUENCE_TICK_SEC")
# After call ends: draft + send a Sandler-style recap email via SMTP (requires prospect email + Anthropic)
POST_CALL_FOLLOWUP_EMAIL_ENABLED = _env_bool("POST_CALL_FOLLOWUP_EMAIL_ENABLED", True)
_pcf_delay = _env_str("POST_CALL_FOLLOWUP_DELAY_SEC")
try:
    POST_CALL_FOLLOWUP_DELAY_SEC = max(60, int(_pcf_delay.strip())) if _pcf_delay else 300
except (ValueError, AttributeError):
    POST_CALL_FOLLOWUP_DELAY_SEC = 300
try:
    EMAIL_SEQUENCE_TICK_SEC = max(15, int(_tick_raw.strip())) if _tick_raw else 60
except ValueError:
    EMAIL_SEQUENCE_TICK_SEC = 60

# App — auto-detect Railway URL if APP_BASE_URL not explicitly set in env
_bu_explicit = os.environ.get("APP_BASE_URL", "").strip()
_railway_domain = os.environ.get("RAILWAY_PUBLIC_DOMAIN", "").strip()
if _bu_explicit:
    _bu = _bu_explicit
elif _railway_domain:
    _bu = f"https://{_railway_domain}"
else:
    _bu = _env_str("APP_BASE_URL") or "http://localhost:8000"
APP_BASE_URL = _bu.strip().rstrip("/")
PORT = int(os.getenv("PORT", "8000"))


def phone_reply_model() -> str:
    """Model for each live call turn."""
    return ANTHROPIC_LIVE_MODEL or ANTHROPIC_PHONE_MODEL_DEFAULT


# Short fillers - 1-2 words, plays instantly from cache
PHONE_FILLER_UTTERANCES: list[str] = [
    "Mm-hmm.",
    "Yeah.",
    "Right.",
    "Sure.",
    "Oh yeah.",
    "Hmm.",
    "Gotcha.",
    "Yep.",
    "Totally.",
    "For sure.",
    "Okay.",
    "Oh nice.",
    "Interesting.",
]


def phone_think_filler_phrase() -> str:
    return random.choice(PHONE_FILLER_UTTERANCES)


def should_play_think_filler(prospect_text: str) -> bool:
    """Play a brief filler on most substantial prospect utterances to fill AI thinking gap."""
    if not PHONE_THINK_FILLER:
        return False
    t = (prospect_text or "").strip()
    # Skip very short utterances like "yes", "no", "ok"
    if len(t) < 5:
        return False
    # Skip common short acknowledgements that don't need a thoughtful reply
    tl = t.lower().rstrip(".!?, ")
    skip = {"yes", "no", "yeah", "yep", "nope", "ok", "okay", "sure", "bye", "hi", "hello", "hey", "thanks", "thank you", "goodbye", "good", "great", "fine", "right", "hmm", "uh huh"}
    if tl in skip:
        return False
    # Play filler on anything else — the AI needs time to think
    return True


def telnyx_speak_voice_effective() -> str:
    """Describes active TTS path (logging)."""
    if ELEVENLABS_DIRECT_FIRST and ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID:
        return f"ElevenLabs-direct:{ELEVENLABS_VOICE_ID}"
    if ELEVENLABS_VOICE_ID and ELEVENLABS_API_KEY_REF:
        return f"ElevenLabs.{ELEVENLABS_MODEL_ID}.{ELEVENLABS_VOICE_ID} (secret={ELEVENLABS_API_KEY_REF})"
    return TELNYX_SPEAK_VOICE or "AWS.Polly.Joanna-Neural"


def tts_mode_description() -> str:
    """Human-readable active TTS path."""
    if ELEVENLABS_DIRECT_FIRST and ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID:
        return "ElevenLabs direct API -> Telnyx playback"
    if ELEVENLABS_VOICE_ID and ELEVENLABS_API_KEY_REF:
        return "ElevenLabs via Telnyx (integration secret)"
    return "Telnyx speak -- Polly/Azure fallback"


def telnyx_speak_voice_for_api() -> str:
    """Voice ID for Telnyx speak()."""
    if ELEVENLABS_VOICE_ID and ELEVENLABS_API_KEY_REF:
        return f"ElevenLabs.{ELEVENLABS_MODEL_ID}.{ELEVENLABS_VOICE_ID}"
    return TELNYX_SPEAK_VOICE or "AWS.Polly.Joanna-Neural"


def elevenlabs_voice_settings() -> dict[str, str] | None:
    """Extra speak() payload for ElevenLabs via Telnyx Mission Control secret."""
    if ELEVENLABS_VOICE_ID and ELEVENLABS_API_KEY_REF:
        return {"type": "elevenlabs", "api_key_ref": ELEVENLABS_API_KEY_REF}
    return None


def reload_secrets() -> None:
    """Reload .env into os.environ and refresh module-level settings."""
    load_dotenv(_ENV_FILE, override=True, encoding=_ENV_ENCODING)
    raw = dotenv_values(_ENV_FILE, encoding=_ENV_ENCODING) or {}
    for k, v in raw.items():
        if v is None:
            continue
        s = str(v).strip()
        if s:
            os.environ[k] = s
    global TELNYX_API_KEY, TELNYX_PUBLIC_KEY, TELNYX_PHONE_NUMBER, TELNYX_CONNECTION_ID
    global CARTESIA_API_KEY, APOLLO_API_KEY, ANTHROPIC_API_KEY, ANTHROPIC_MODEL
    global ANTHROPIC_LIVE_MODEL, ANTHROPIC_PHONE_MODEL_DEFAULT, ANTHROPIC_MAX_TOKENS_REPLY
    global TELNYX_SPEAK_VOICE, TELNYX_SPEAK_SERVICE_LEVEL, TELNYX_SPEAK_LANGUAGE
    global ELEVENLABS_VOICE_ID, ELEVENLABS_MODEL_ID, ELEVENLABS_API_KEY_REF, ELEVENLABS_API_KEY, ELEVENLABS_DIRECT_FIRST, ELEVENLABS_HTTP_STREAM, STREAM_SPEECH_PIPELINE, TRANSCRIPTION_REPLY_ON_INTERIM, PHONE_THINK_FILLER
    global APP_BASE_URL, PORT, COMPANY_NAME, SDR_NAME
    global QA_KB_ENABLED, QA_KB_MIN_SCORE
    global EMAIL_PROVIDER, SENDGRID_API_KEY, RESEND_API_KEY, MAILGUN_API_KEY, MAILGUN_DOMAIN, MAILGUN_API_BASE
    global GOOGLE_OAUTH_CLIENT_ID, GOOGLE_OAUTH_CLIENT_SECRET
    global MICROSOFT_OAUTH_CLIENT_ID, MICROSOFT_OAUTH_CLIENT_SECRET, MICROSOFT_OAUTH_TENANT
    global SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, EMAIL_FROM, SMTP_USE_TLS
    global EMAIL_AUTOMATION_ENABLED, EMAIL_SEQUENCE_TICK_SEC
    global POST_CALL_FOLLOWUP_EMAIL_ENABLED, POST_CALL_FOLLOWUP_DELAY_SEC
    TELNYX_API_KEY = _env_str("TELNYX_API_KEY")
    TELNYX_PUBLIC_KEY = _env_str("TELNYX_PUBLIC_KEY")
    TELNYX_PHONE_NUMBER = _normalize_e164_phone(_env_str("TELNYX_PHONE_NUMBER"))
    _cid2 = (_env_str("TELNYX_CONNECTION_ID") or "").strip()
    TELNYX_CONNECTION_ID = _cid2 if _cid2 else None
    CARTESIA_API_KEY     = _env_str("CARTESIA_API_KEY")
    APOLLO_API_KEY       = _env_str("APOLLO_API_KEY")
    ANTHROPIC_API_KEY    = _env_str("ANTHROPIC_API_KEY")
    ANTHROPIC_MODEL      = _env_str("ANTHROPIC_MODEL") or "claude-sonnet-4-20250514"
    ANTHROPIC_LIVE_MODEL = _env_str("ANTHROPIC_LIVE_MODEL")
    ANTHROPIC_PHONE_MODEL_DEFAULT = _env_str("ANTHROPIC_PHONE_MODEL_DEFAULT") or "claude-haiku-4-5"
    ANTHROPIC_MAX_TOKENS_REPLY = _env_int("ANTHROPIC_MAX_TOKENS_REPLY", 88)
    TELNYX_SPEAK_VOICE = _env_str("TELNYX_SPEAK_VOICE") or "AWS.Polly.Joanna-Neural"
    _sl2 = (_env_str("TELNYX_SPEAK_SERVICE_LEVEL") or "premium").strip().lower()
    TELNYX_SPEAK_SERVICE_LEVEL = "basic" if _sl2 == "basic" else "premium"
    TELNYX_SPEAK_LANGUAGE = _env_str("TELNYX_SPEAK_LANGUAGE") or "en-US"
    ELEVENLABS_VOICE_ID = _env_str("ELEVENLABS_VOICE_ID")
    ELEVENLABS_MODEL_ID = _env_str("ELEVENLABS_MODEL_ID") or "eleven_multilingual_v2"
    ELEVENLABS_API_KEY_REF = _env_str("ELEVENLABS_API_KEY_REF")
    ELEVENLABS_API_KEY = _env_str("ELEVENLABS_API_KEY")
    ELEVENLABS_DIRECT_FIRST = _env_bool("ELEVENLABS_DIRECT_FIRST", True)
    ELEVENLABS_HTTP_STREAM = _env_bool("ELEVENLABS_HTTP_STREAM", False)
    STREAM_SPEECH_PIPELINE = _env_bool("STREAM_SPEECH_PIPELINE", False)
    TRANSCRIPTION_REPLY_ON_INTERIM = _env_bool("TRANSCRIPTION_REPLY_ON_INTERIM", True)
    PHONE_THINK_FILLER = _env_bool("PHONE_THINK_FILLER", False)
    QA_KB_ENABLED = _env_bool("QA_KB_ENABLED", True)
    _qa_min2 = _env_str("QA_KB_MIN_SCORE")
    try:
        QA_KB_MIN_SCORE = float(_qa_min2.strip()) if _qa_min2 else 0.82
        if not (0.0 <= QA_KB_MIN_SCORE <= 1.0):
            QA_KB_MIN_SCORE = 0.82
    except (ValueError, AttributeError):
        QA_KB_MIN_SCORE = 0.82
    _bu2_explicit = os.environ.get("APP_BASE_URL", "").strip()
    _rd2 = os.environ.get("RAILWAY_PUBLIC_DOMAIN", "").strip()
    if _bu2_explicit:
        APP_BASE_URL = _bu2_explicit.rstrip("/")
    elif _rd2:
        APP_BASE_URL = f"https://{_rd2}"
    else:
        _bu2 = _env_str("APP_BASE_URL") or "http://localhost:8000"
        APP_BASE_URL = _bu2.strip().rstrip("/")
    PORT = int(os.getenv("PORT", "8000"))
    COMPANY_NAME         = os.getenv("COMPANY_NAME", "Your Company")
    SDR_NAME             = os.getenv("SDR_NAME", "Alex")
    _ep2 = (_env_str("EMAIL_PROVIDER") or "smtp").strip().lower()
    EMAIL_PROVIDER = (
        _ep2
        if _ep2 in ("smtp", "sendgrid", "resend", "mailgun", "gmail_oauth", "outlook_oauth")
        else "smtp"
    )
    GOOGLE_OAUTH_CLIENT_ID = _env_str("GOOGLE_OAUTH_CLIENT_ID")
    GOOGLE_OAUTH_CLIENT_SECRET = _env_str("GOOGLE_OAUTH_CLIENT_SECRET")
    MICROSOFT_OAUTH_CLIENT_ID = _env_str("MICROSOFT_OAUTH_CLIENT_ID")
    MICROSOFT_OAUTH_CLIENT_SECRET = _env_str("MICROSOFT_OAUTH_CLIENT_SECRET")
    MICROSOFT_OAUTH_TENANT = _env_str("MICROSOFT_OAUTH_TENANT") or "common"
    SENDGRID_API_KEY = _env_str("SENDGRID_API_KEY")
    RESEND_API_KEY = _env_str("RESEND_API_KEY")
    MAILGUN_API_KEY = _env_str("MAILGUN_API_KEY")
    MAILGUN_DOMAIN = _env_str("MAILGUN_DOMAIN")
    _mgb2 = (_env_str("MAILGUN_API_BASE") or "https://api.mailgun.net").strip().rstrip("/")
    MAILGUN_API_BASE = _mgb2 if _mgb2.startswith("http") else "https://api.mailgun.net"
    SMTP_HOST = _env_str("SMTP_HOST")
    _spr = _env_str("SMTP_PORT")
    try:
        SMTP_PORT = int(_spr.strip()) if _spr else 587
        if not (1 <= SMTP_PORT <= 65535):
            SMTP_PORT = 587
    except (ValueError, AttributeError):
        SMTP_PORT = 587
    SMTP_USER = _env_str("SMTP_USER")
    SMTP_PASSWORD = _env_str("SMTP_PASSWORD")
    EMAIL_FROM = _env_str("EMAIL_FROM")
    SMTP_USE_TLS = _env_bool("SMTP_USE_TLS", True)
    EMAIL_AUTOMATION_ENABLED = _env_bool("EMAIL_AUTOMATION_ENABLED", False)
    _tr = _env_str("EMAIL_SEQUENCE_TICK_SEC")
    try:
        EMAIL_SEQUENCE_TICK_SEC = max(15, int(_tr.strip())) if _tr else 60
    except ValueError:
        EMAIL_SEQUENCE_TICK_SEC = 60
    POST_CALL_FOLLOWUP_EMAIL_ENABLED = _env_bool("POST_CALL_FOLLOWUP_EMAIL_ENABLED", True)
    _pcf2 = _env_str("POST_CALL_FOLLOWUP_DELAY_SEC")
    try:
        POST_CALL_FOLLOWUP_DELAY_SEC = max(60, int(_pcf2.strip())) if _pcf2 else 300
    except (ValueError, AttributeError):
        POST_CALL_FOLLOWUP_DELAY_SEC = 300


def env_file_nonempty(key: str) -> bool:
    """True if key exists in .env OR os.environ with a non-empty value.
    On Railway, secrets are injected as real env vars (no .env file), so we
    must check os.environ too."""
    raw = dotenv_values(_ENV_FILE, encoding=_ENV_ENCODING) or {}
    v = raw.get(key)
    if v is None:
        # Fall back to real environment (Railway / Docker / shell exports)
        v = os.environ.get(key)
    if v is None:
        return False
    s = str(v).strip()
    if len(s) >= 2 and s[0] in "\"'" and s[-1] == s[0]:
        s = s[1:-1].strip()
    return bool(s)


def dashboard_connection_flags() -> dict[str, bool]:
    return {
        "telnyx": env_file_nonempty("TELNYX_API_KEY")
        and env_file_nonempty("TELNYX_CONNECTION_ID")
        and env_file_nonempty("TELNYX_PHONE_NUMBER"),
        "deepgram": env_file_nonempty("DEEPGRAM_API_KEY"),
        "anthropic": env_file_nonempty("ANTHROPIC_API_KEY"),
        "apollo": env_file_nonempty("APOLLO_API_KEY"),
        "email": _email_outbound_env_ready(),
    }


def _email_outbound_env_ready() -> bool:
    """True when .env (or os.environ) has enough for the selected EMAIL_PROVIDER to send."""
    raw = dotenv_values(_ENV_FILE, encoding=_ENV_ENCODING) or {}
    p = (str(raw.get("EMAIL_PROVIDER") or os.environ.get("EMAIL_PROVIDER") or "smtp").strip().lower() or "smtp")
    if p not in ("smtp", "sendgrid", "resend", "mailgun", "gmail_oauth", "outlook_oauth"):
        p = "smtp"
    if p == "smtp":
        return env_file_nonempty("SMTP_HOST") and env_file_nonempty("EMAIL_FROM")
    if p == "sendgrid":
        return env_file_nonempty("SENDGRID_API_KEY") and env_file_nonempty("EMAIL_FROM")
    if p == "resend":
        return env_file_nonempty("RESEND_API_KEY") and env_file_nonempty("EMAIL_FROM")
    if p == "mailgun":
        return (
            env_file_nonempty("MAILGUN_API_KEY")
            and env_file_nonempty("MAILGUN_DOMAIN")
            and env_file_nonempty("EMAIL_FROM")
        )
    if p == "gmail_oauth":
        try:
            from email_oauth import oauth_account_ready

            return oauth_account_ready("google") and env_file_nonempty("EMAIL_FROM")
        except Exception:
            return False
    if p == "outlook_oauth":
        try:
            from email_oauth import oauth_account_ready

            return oauth_account_ready("microsoft") and env_file_nonempty("EMAIL_FROM")
        except Exception:
            return False
    return False
