"""
monitoring/run_dashboard.py
===========================
CLI for Section 7 — Monitoring & Dashboard.

Usage::

    python -m monitoring.run_dashboard                  # serve web dashboard
    python -m monitoring.run_dashboard --port 9000      # custom port
    python -m monitoring.run_dashboard --report         # dump JSON report
    python -m monitoring.run_dashboard --syscheck       # Section 8 check
    python -m monitoring.run_dashboard --backend streamlit

Backend selection (``DASHBOARD_BACKEND``):

* ``auto`` (default) -- Streamlit if installed, else stdlib server
* ``stdlib``         -- force the zero-dependency server
* ``streamlit``      -- Streamlit only (clear error if not installed)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys

from data_acquisition.utils.logger import get_logger

from monitoring import config as mcfg
from monitoring.dashboard_data import DashboardData
from monitoring.system_check import collect_system_check, summary

logger = get_logger("monitoring.run_dashboard")


def _streamlit_available() -> bool:
    return importlib.util.find_spec("streamlit") is not None


def _run_streamlit() -> int:
    app_path = __file__.replace("run_dashboard.py", "app_streamlit.py")
    cmd = [sys.executable, "-m", "streamlit", "run", app_path,
           "--server.port", str(mcfg.DASHBOARD_PORT),
           "--server.address", mcfg.DASHBOARD_HOST]
    logger.info("Launching Streamlit: %s", " ".join(cmd))
    try:
        return subprocess.run(cmd).returncode
    except KeyboardInterrupt:
        return 0


def _print_syscheck() -> None:
    checks = collect_system_check()
    print(f"{'REQUIREMENT':<16} {'STATUS':<9} {'DETECTED':<46} RECOMMENDED")
    print("-" * 100)
    for check in checks:
        print(f"{check['name']:<16} {check['status']:<9} "
              f"{check['detail'][:45]:<46} {check['recommended']}")
    counts = summary(checks)
    print(f"\nok: {counts['ok']}  warn: {counts['warn']}  "
          f"missing: {counts['missing']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="monitoring.run_dashboard",
        description="UrduNewsAI Section 7 — monitoring & operations dashboard",
    )
    parser.add_argument("--host", default=mcfg.DASHBOARD_HOST)
    parser.add_argument("--port", type=int, default=mcfg.DASHBOARD_PORT)
    parser.add_argument("--refresh", type=int,
                        default=mcfg.DASHBOARD_REFRESH_SECONDS,
                        help="dashboard auto-refresh seconds")
    parser.add_argument("--backend", default=mcfg.DASHBOARD_BACKEND,
                        choices=["auto", "stdlib", "streamlit"])
    parser.add_argument("--report", action="store_true",
                        help="print the full JSON report and exit")
    parser.add_argument("--syscheck", action="store_true",
                        help="run the Section 8 system-requirements check")
    parser.add_argument("--export", metavar="PATH",
                        help="write the JSON report to a file and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.syscheck:
        _print_syscheck()
        return 0

    with DashboardData() as data:
        if args.report or args.export:
            report = data.full_report()
            text = json.dumps(report, ensure_ascii=False, indent=2,
                              default=str)
            if args.export:
                with open(args.export, "w", encoding="utf-8") as fh:
                    fh.write(text)
                logger.info("Dashboard report written to %s", args.export)
                print(f"Report exported to {args.export}")
            else:
                print(text)
            return 0

        if args.backend == "streamlit" or (
                args.backend == "auto" and _streamlit_available()):
            if not _streamlit_available():
                logger.error("Streamlit requested but not installed")
                return 1
            return _run_streamlit()

        # stdlib fallback (also used by --backend auto without Streamlit)
        from monitoring.server import serve

        serve(data, host=args.host, port=args.port,
              refresh_seconds=args.refresh)
        return 0


if __name__ == "__main__":
    sys.exit(main())
