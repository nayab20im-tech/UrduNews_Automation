"""
orchestration/config.py
=======================
Configuration for Section 5 — Orchestration & Pipeline Control.

Mirrors the project's existing config pattern (``pipeline/config.py``,
``distribution/config.py``): every setting overridable via environment
variables, sensible defaults otherwise.  No secrets hard-coded -- SMTP
and Telegram credentials come from the environment only.
"""

from __future__ import annotations

import os
from pathlib import Path

from data_acquisition import config as acq_cfg

# ---------------------------------------------------------------------------
# Orchestration database (run history + automation job queue; same
# single-artifact convention as the rest of the project)
# ---------------------------------------------------------------------------
ORCH_DB_PATH = Path(os.getenv("ORCH_DB_PATH", str(acq_cfg.DEFAULT_DB_PATH)))

# ---------------------------------------------------------------------------
# Pipeline Orchestrator — error handling & retry
# ---------------------------------------------------------------------------
ORCH_MAX_RETRIES = int(os.getenv("ORCH_MAX_RETRIES", "2"))
ORCH_RETRY_DELAY_SECONDS = float(os.getenv("ORCH_RETRY_DELAY_SECONDS", "5"))

# ---------------------------------------------------------------------------
# Automation Engine — job queue polling
# ---------------------------------------------------------------------------
JOB_POLL_INTERVAL_SECONDS = float(os.getenv("JOB_POLL_INTERVAL_SECONDS", "60"))
JOB_MAX_ATTEMPTS = int(os.getenv("JOB_MAX_ATTEMPTS", "3"))
JOB_RETRY_DELAY_SECONDS = float(os.getenv("JOB_RETRY_DELAY_SECONDS", "30"))

# ---------------------------------------------------------------------------
# Notification System — email / Telegram alerts
# ---------------------------------------------------------------------------
NOTIFY_ENABLED = os.getenv("NOTIFY_ENABLED", "true").lower() in ("1", "true", "yes")
NOTIFY_ON_ERROR = os.getenv("NOTIFY_ON_ERROR", "true").lower() in ("1", "true", "yes")
NOTIFY_ON_COMPLETE = os.getenv("NOTIFY_ON_COMPLETE", "false").lower() in ("1", "true", "yes")

# Email (optional; disabled when host or recipients are missing)
NOTIFY_EMAIL_HOST = os.getenv("NOTIFY_EMAIL_HOST", "")
NOTIFY_EMAIL_PORT = int(os.getenv("NOTIFY_EMAIL_PORT", "587"))
NOTIFY_EMAIL_USER = os.getenv("NOTIFY_EMAIL_USER", "")
NOTIFY_EMAIL_PASSWORD = os.getenv("NOTIFY_EMAIL_PASSWORD", "")
NOTIFY_EMAIL_FROM = os.getenv("NOTIFY_EMAIL_FROM", "urdu-news-ai@localhost")
NOTIFY_EMAIL_TO = [t.strip() for t in os.getenv("NOTIFY_EMAIL_TO", "").split(",") if t.strip()]

# Telegram reuses the Section 4 bot credentials (distribution/config.py)
# so operators only configure one bot for sharing + alerts.
