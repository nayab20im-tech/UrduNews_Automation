"""
config.py
=========
Central configuration for the Data Acquisition layer.

Loads secrets (API keys) from a `.env` file via python-dotenv, and loads
source definitions (RSS feeds, website selector templates, API settings,
Google Trends settings, social media settings, manual input folder) from
`data/sources_config.yaml`.

Nothing in this file talks to the network -- it only resolves paths and
settings so every other module has one single source of truth.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
MANUAL_INPUT_DIR = DATA_DIR / "manual_input"
SOURCES_CONFIG_PATH = DATA_DIR / "sources_config.yaml"
DEFAULT_DB_PATH = DATA_DIR / "raw_news.db"

LOG_DIR.mkdir(parents=True, exist_ok=True)
MANUAL_INPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Environment variables (.env)
# ---------------------------------------------------------------------------
load_dotenv(BASE_DIR / ".env", override=False)

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY", "")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")

# ---------------------------------------------------------------------------
# HTTP / scraping engine defaults
# ---------------------------------------------------------------------------
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "15"))          # seconds
REQUEST_RETRIES = int(os.getenv("REQUEST_RETRIES", "3"))
REQUEST_BACKOFF_FACTOR = float(os.getenv("REQUEST_BACKOFF_FACTOR", "0.6"))
MIN_DELAY_BETWEEN_REQUESTS = float(os.getenv("MIN_DELAY_BETWEEN_REQUESTS", "1.0"))
RESPECT_ROBOTS_TXT = os.getenv("RESPECT_ROBOTS_TXT", "true").lower() == "true"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]

MAX_ARTICLES_PER_WEBSITE = int(os.getenv("MAX_ARTICLES_PER_WEBSITE", "20"))
MAX_TWEETS_PER_QUERY = int(os.getenv("MAX_TWEETS_PER_QUERY", "50"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = LOG_DIR / "data_acquisition.log"


def load_sources_config(path: "str | Path" = SOURCES_CONFIG_PATH) -> Dict[str, Any]:
    """Load the YAML file describing every configured news source.

    Returns an empty, safely-defaulted structure if the file is missing,
    so the rest of the pipeline can still run (e.g. manual input only).
    """
    path = Path(path)
    if not path.exists():
        return {
            "rss_feeds": [],
            "websites": [],
            "news_api": {"enabled": False},
            "google_trends": {"enabled": False},
            "social_media": {"enabled": False},
            "manual_input": {"enabled": True, "folder": str(MANUAL_INPUT_DIR)},
        }

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cfg.setdefault("rss_feeds", [])
    cfg.setdefault("websites", [])
    cfg.setdefault("news_api", {"enabled": False})
    cfg.setdefault("google_trends", {"enabled": False})
    cfg.setdefault("social_media", {"enabled": False})
    cfg.setdefault("manual_input", {"enabled": True, "folder": str(MANUAL_INPUT_DIR)})
    return cfg
