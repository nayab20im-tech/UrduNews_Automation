"""
distribution/config.py
=========================
Configuration for Section 4 — Distribution & Automation (4.1-4.5).

Mirrors the project's existing config pattern (``pipeline/config.py``,
``video_production/config.py``): every setting overridable via
environment variables, sensible defaults otherwise.  No secrets are
hard-coded -- all API credentials come from the environment.
"""

from __future__ import annotations

import os
from pathlib import Path

from data_acquisition import config as acq_cfg

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
DISTRIBUTION_OUTPUT_DIR = Path(os.getenv(
    "DISTRIBUTION_OUTPUT_DIR", str(acq_cfg.DATA_DIR / "media" / "distribution")))
REPORTS_DIR = Path(os.getenv(
    "DISTRIBUTION_REPORTS_DIR", str(DISTRIBUTION_OUTPUT_DIR / "reports")))

# ---------------------------------------------------------------------------
# Database (reuses the same SQLite file as the rest of the pipeline)
# ---------------------------------------------------------------------------
DISTRIBUTION_DB_PATH = Path(os.getenv(
    "DISTRIBUTION_DB_PATH", str(acq_cfg.DATA_DIR / "raw_news.db")))

# ---------------------------------------------------------------------------
# 4.1  YouTube
# ---------------------------------------------------------------------------
YOUTUBE_CLIENT_SECRETS_FILE = os.getenv("YOUTUBE_CLIENT_SECRETS_FILE", "")
YOUTUBE_TOKEN_FILE = os.getenv(
    "YOUTUBE_TOKEN_FILE", str(DISTRIBUTION_OUTPUT_DIR / "youtube_token.json"))
YOUTUBE_SCOPES = os.getenv(
    "YOUTUBE_SCOPES",
    "https://www.googleapis.com/auth/youtube.upload "
    "https://www.googleapis.com/auth/youtube.readonly "
    "https://www.googleapis.com/auth/youtube.force-ssl",
).split()
YOUTUBE_PRIVACY_STATUS = os.getenv("YOUTUBE_PRIVACY_STATUS", "public")
YOUTUBE_CATEGORY_ID = os.getenv("YOUTUBE_CATEGORY_ID", "25")   # 25 = News & Politics
YOUTUBE_SCHEDULE_TIME = os.getenv("YOUTUBE_SCHEDULE_TIME", "")  # ISO-8601 for scheduled publish
YOUTUBE_TITLE_TEMPLATE = os.getenv(
    "YOUTUBE_TITLE_TEMPLATE", "{headline} | {channel_name}")
YOUTUBE_DESCRIPTION_TEMPLATE = os.getenv("YOUTUBE_DESCRIPTION_TEMPLATE", "")
CHANNEL_NAME = os.getenv("CHANNEL_NAME", "اردو نیوز AI")

# ---------------------------------------------------------------------------
# 4.2  Social Media — Facebook / Meta
# ---------------------------------------------------------------------------
FACEBOOK_PAGE_ID = os.getenv("FACEBOOK_PAGE_ID", "")
FACEBOOK_ACCESS_TOKEN = os.getenv("FACEBOOK_ACCESS_TOKEN", "")
FACEBOOK_API_VERSION = os.getenv("FACEBOOK_API_VERSION", "v19.0")

# ---------------------------------------------------------------------------
# 4.2  Social Media — X / Twitter
# ---------------------------------------------------------------------------
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY", "")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET", "")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN", "")
TWITTER_ACCESS_TOKEN_SECRET = os.getenv("TWITTER_ACCESS_TOKEN_SECRET", "")

# ---------------------------------------------------------------------------
# 4.2  Social Media — Telegram
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ---------------------------------------------------------------------------
# 4.2  Social Media — WhatsApp Business API
# ---------------------------------------------------------------------------
WHATSAPP_API_URL = os.getenv("WHATSAPP_API_URL", "")
WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN", "")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")

# ---------------------------------------------------------------------------
# 4.3  Playlist management
# ---------------------------------------------------------------------------
PLAYLIST_NAME_PREFIX = os.getenv("PLAYLIST_NAME_PREFIX", "")  # e.g. "Urdu News AI"
PLAYLIST_PRIVACY = os.getenv("PLAYLIST_PRIVACY", "public")

# ---------------------------------------------------------------------------
# 4.5  Scheduler
# ---------------------------------------------------------------------------
SCHEDULER_INTERVAL_HOURS = float(os.getenv("SCHEDULER_INTERVAL_HOURS", "6"))
SCHEDULER_TIMEZONE = os.getenv("SCHEDULER_TIMEZONE", "UTC")
SCHEDULER_DAILY_TIME = os.getenv("SCHEDULER_DAILY_TIME", "")  # HH:MM for daily run
SCHEDULER_MAX_RETRIES = int(os.getenv("SCHEDULER_MAX_RETRIES", "3"))
SCHEDULER_RETRY_DELAY_MINUTES = int(os.getenv("SCHEDULER_RETRY_DELAY_MINUTES", "30"))

# ---------------------------------------------------------------------------
# Mock / dry-run mode
# ---------------------------------------------------------------------------
# When True, API calls are simulated and results are clearly marked as mock.
# Automatically enabled when required credentials are missing.
DISTRIBUTION_MOCK_MODE = os.getenv(
    "DISTRIBUTION_MOCK_MODE", "false").lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def is_youtube_configured() -> bool:
    """Return True when YouTube OAuth client secrets file is available."""
    return bool(YOUTUBE_CLIENT_SECRETS_FILE) and Path(YOUTUBE_CLIENT_SECRETS_FILE).exists()


def is_facebook_configured() -> bool:
    return bool(FACEBOOK_PAGE_ID and FACEBOOK_ACCESS_TOKEN)


def is_twitter_configured() -> bool:
    return bool(TWITTER_API_KEY and TWITTER_ACCESS_TOKEN)


def is_telegram_configured() -> bool:
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


def is_whatsapp_configured() -> bool:
    return bool(WHATSAPP_API_URL and WHATSAPP_ACCESS_TOKEN and WHATSAPP_PHONE_NUMBER_ID)
