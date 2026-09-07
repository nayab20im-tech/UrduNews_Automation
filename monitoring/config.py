"""
monitoring/config.py
====================
Configuration for Section 7 — Monitoring & Dashboard.

Mirrors the project's existing config pattern (``pipeline/config.py``,
``orchestration/config.py``): every setting overridable via environment
variables, sensible defaults otherwise.

The dashboard backend follows the project's "auto with offline
fallback" convention: ``streamlit`` when installed, otherwise the
zero-dependency stdlib web server in ``monitoring/server.py``.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Web dashboard
# ---------------------------------------------------------------------------
DASHBOARD_BACKEND = os.getenv("DASHBOARD_BACKEND", "auto")  # auto | stdlib | streamlit
DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "127.0.0.1")
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8601"))
DASHBOARD_REFRESH_SECONDS = int(os.getenv("DASHBOARD_REFRESH_SECONDS", "15"))

# ---------------------------------------------------------------------------
# Content limits
# ---------------------------------------------------------------------------
DASHBOARD_NEWS_LIMIT = int(os.getenv("DASHBOARD_NEWS_LIMIT", "10"))
DASHBOARD_LOG_LIMIT = int(os.getenv("DASHBOARD_LOG_LIMIT", "20"))
