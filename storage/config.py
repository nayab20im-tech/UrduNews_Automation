"""
storage/config.py
=================
Configuration for Section 6 — Data Storage Layer (all local).

Mirrors the project's existing config pattern (``pipeline/config.py``,
``distribution/config.py``): every setting overridable via environment
variables, sensible defaults otherwise.  All paths hang off the
acquisition ``DATA_DIR`` so nothing is hard-coded to a machine.

By convention the registry tables (media assets, model registry, config
store, automation jobs) live in the *same* SQLite file as Sections 1-4
(``raw_news.db``) so the whole project stays one artifact; the chatty
system logs get their own file so they can never bloat the news data.
"""

from __future__ import annotations

import os
from pathlib import Path

from data_acquisition import config as acq_cfg

# ---------------------------------------------------------------------------
# SQLite artifacts
# ---------------------------------------------------------------------------
# Shared primary artifact (raw + processed + distribution + registries).
STORAGE_DB_PATH = Path(os.getenv("STORAGE_DB_PATH", str(acq_cfg.DEFAULT_DB_PATH)))

# Separate file for system logs (Section 6 "Logs DB" box).
LOGS_DB_PATH = Path(os.getenv("LOGS_DB_PATH", str(acq_cfg.DATA_DIR / "system_logs.db")))

# ---------------------------------------------------------------------------
# Media storage (Section 6 "Media Storage" box: audio, video, images)
# ---------------------------------------------------------------------------
MEDIA_ROOT = Path(os.getenv("MEDIA_ROOT", str(acq_cfg.DATA_DIR / "media")))

# ---------------------------------------------------------------------------
# Config & model storage (Section 6 "Config & Model Storage" box)
# ---------------------------------------------------------------------------
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(acq_cfg.DATA_DIR / "models")))
CONFIG_SNAPSHOTS_DIR = Path(os.getenv(
    "CONFIG_SNAPSHOTS_DIR", str(acq_cfg.DATA_DIR / "config")))

# ---------------------------------------------------------------------------
# Log retention
# ---------------------------------------------------------------------------
LOG_RETENTION_DAYS = int(os.getenv("LOG_RETENTION_DAYS", "30"))
