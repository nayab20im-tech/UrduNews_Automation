"""
monitoring/__init__.py
======================
Section 7 — Monitoring & Dashboard (+ Section 8 system check).

Public API:

* :class:`monitoring.dashboard_data.DashboardData` -- backend-agnostic
  aggregation of pipeline status, latest-news preview, upload status and
  analytics overview on top of the Section 6 storage layer.
* :mod:`monitoring.server` -- zero-dependency stdlib web dashboard with
  a JSON API (offline fallback for Streamlit/FastAPI).
* :func:`monitoring.system_check.collect_system_check` -- compares the
  host against the Section 8 suggested specification.

CLI::

    python -m monitoring.run_dashboard                 # serve dashboard
    python -m monitoring.run_dashboard --report        # JSON report dump
    python -m monitoring.run_dashboard --syscheck      # requirements check
"""

from monitoring.dashboard_data import DashboardData
from monitoring.system_check import collect_system_check, summary

__all__ = ["DashboardData", "collect_system_check", "summary"]
