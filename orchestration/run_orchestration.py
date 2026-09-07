#!/usr/bin/env python3
"""
orchestration/run_orchestration.py
==================================
CLI entry point for Section 5 — Orchestration & Pipeline Control.

Examples
--------
    python -m orchestration.run_orchestration                      # full 1 -> 4 run
    python -m orchestration.run_orchestration --stages process,video
    python -m orchestration.run_orchestration --limit 2
    python -m orchestration.run_orchestration --retry              # re-run failed stages
    python -m orchestration.run_orchestration --status             # run history
    python -m orchestration.run_orchestration --health             # Section 6 storage health
    python -m orchestration.run_orchestration --snapshot           # persist effective config
    python -m orchestration.run_orchestration --submit process     # enqueue an automation job
    python -m orchestration.run_orchestration --run-due            # execute due queued jobs
    python -m orchestration.run_orchestration --notify-test        # test the alert channels
"""

from __future__ import annotations

import argparse
import json
import sys

from data_acquisition.utils.logger import get_logger

from .automation_engine import AutomationEngine
from .config_manager import ConfigManager
from .notifications import NotificationManager
from .orchestrator import DEFAULT_STAGES, EndToEndOrchestrator

logger = get_logger("orchestration.cli")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Section 5 — Orchestration & Pipeline Control "
                    "(end-to-end workflow, automation queue, config manager, notifications)."
    )
    parser.add_argument(
        "--stages", type=str, default="all",
        help="Comma-separated stages: collect,process,video,distribute (default: all)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit articles per stage")
    parser.add_argument("--retry", action="store_true", help="Re-run stages that failed on the latest run")
    parser.add_argument("--status", action="store_true", help="Print orchestration run history and exit")
    parser.add_argument("--health", action="store_true", help="Print Section 6 storage health and exit")
    parser.add_argument("--snapshot", action="store_true", help="Persist the effective config snapshot and exit")
    parser.add_argument("--submit", type=str, default="", help="Enqueue an automation job by handler name")
    parser.add_argument("--payload", type=str, default="{}", help="JSON payload for --submit")
    parser.add_argument("--run-due", action="store_true", help="Execute all due automation jobs once and exit")
    parser.add_argument("--queue", action="store_true", help="List automation jobs and exit")
    parser.add_argument("--notify-test", action="store_true", help="Send a test notification and exit")
    parser.add_argument("--export", type=str, default="", help="Also write the run report to this JSON path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # --- read-only modes -------------------------------------------------
    if args.health:
        from storage.storage_manager import StorageManager

        with StorageManager() as sm:
            print(json.dumps(sm.health(), indent=2, ensure_ascii=False, default=str))
        return 0

    if args.status:
        with EndToEndOrchestrator() as orch:
            print(json.dumps(orch.status(), indent=2, ensure_ascii=False))
        return 0

    if args.snapshot:
        print(json.dumps(ConfigManager().snapshot(), indent=2, ensure_ascii=False, default=str))
        return 0

    if args.notify_test:
        result = NotificationManager(enabled=True).send(
            "[UrduNewsAI] Test notification",
            "Notification system is wired up (log always; email/Telegram when configured).",
        )
        print(json.dumps(result, indent=2))
        return 0

    # --- automation queue modes ---------------------------------------------
    if args.submit or args.run_due or args.queue:
        with AutomationEngine() as engine:
            engine.register_default_handlers()
            if args.submit:
                job_id = engine.submit(args.submit, payload=json.loads(args.payload))
                print(json.dumps({"submitted": job_id, "name": args.submit}))
            if args.run_due:
                results = engine.run_due()
                print(json.dumps(results, indent=2, ensure_ascii=False, default=str))
            if args.queue:
                print(json.dumps(engine.list_jobs(), indent=2, ensure_ascii=False, default=str))
        return 0

    # --- pipeline runs -------------------------------------------------
    from storage.logs_db import LogsDatabase, attach_sqlite_handler

    logs_db = LogsDatabase()
    attach_sqlite_handler(logs_db)  # mirror every module logger into the Logs DB

    try:
        with EndToEndOrchestrator(logs_db=logs_db) as orch:
            if args.retry:
                report = orch.retry_failed(limit=args.limit)
            else:
                stages = (
                    list(DEFAULT_STAGES) if args.stages == "all"
                    else [s.strip() for s in args.stages.split(",") if s.strip()]
                )
                report = orch.run(stages=stages, limit=args.limit)
        output = report.as_dict()
        print(json.dumps(output, indent=2, ensure_ascii=False))
        if args.export:
            from pathlib import Path

            Path(args.export).write_text(
                json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"\nReport exported to: {args.export}")
    finally:
        logs_db.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
