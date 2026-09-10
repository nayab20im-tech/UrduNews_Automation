#!/usr/bin/env python3
"""
distribution/run_distribution.py
====================================
CLI entry point for Section 4 — Distribution & Automation.

Examples
--------
    python -m distribution.run_distribution                     # distribute all eligible
    python -m distribution.run_distribution --limit 5           # first 5 only
    python -m distribution.run_distribution --retry             # retry failed jobs
    python -m distribution.run_distribution --analytics         # collect stats + daily report
    python -m distribution.run_distribution --stats             # print distribution stats
    python -m distribution.run_distribution --export report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from data_acquisition import config as acquisition_config
from data_acquisition.database.raw_news_db import RawNewsDatabase
from data_acquisition.utils.logger import get_logger

from . import config as dc
from .distribution_db import DistributionDatabase
from .pipeline import run_distribution, retry_failed_jobs
from . import analytics

logger = get_logger("distribution.cli")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Section 4 Distribution & Automation pipeline."
    )
    parser.add_argument("--raw-db-path", type=str, default=str(acquisition_config.DEFAULT_DB_PATH))
    parser.add_argument(
        "--processed-db-path", type=str,
        default=str(acquisition_config.DATA_DIR / "processed_news.db"),
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit articles processed")
    parser.add_argument("--retry", action="store_true", help="Retry failed distribution jobs")
    parser.add_argument("--analytics", action="store_true", help="Collect stats and generate daily report")
    parser.add_argument("--stats", action="store_true", help="Print distribution DB stats and exit")
    parser.add_argument("--export", type=str, default="", help="Export report to JSON path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    db = DistributionDatabase()

    try:
        # --- Stats mode ---
        if args.stats:
            jobs = db.get_all_jobs()
            published = sum(1 for j in jobs if j.get("youtube_status") == "published")
            failed = sum(1 for j in jobs if j.get("status") == "failed")
            pending = sum(1 for j in jobs if j.get("status") in ("pending", "video_ready"))
            print(json.dumps({
                "total_jobs": len(jobs),
                "published": published,
                "failed": failed,
                "pending": pending,
                "playlists": len(db.get_all_playlists()),
            }, indent=2, ensure_ascii=False))
            return 0

        # --- Retry mode ---
        if args.retry:
            report = retry_failed_jobs(max_retries=dc.SCHEDULER_MAX_RETRIES, db=db)
            output = report.as_dict()
            print(json.dumps(output, indent=2, ensure_ascii=False))
            if args.export:
                Path(args.export).write_text(
                    json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            return 0

        # --- Analytics mode ---
        if args.analytics:
            count = analytics.collect_all_stats(db)
            report = analytics.generate_daily_report(db)
            print(json.dumps({
                "collected": count,
                "report": report.get("summary", {}),
            }, indent=2, ensure_ascii=False))
            return 0

        # --- Distribution mode (default) ---
        # Load articles from the processed news DB via build_final_dataset
        try:
            from pipeline.processed_news_db import ProcessedNewsDatabase
            processed_db = ProcessedNewsDatabase(db_path=args.processed_db_path)
            dataset = processed_db.build_final_dataset()
            processed_db.close()
        except Exception as exc:
            logger.warning("Could not load processed DB: %s; using empty dataset", exc)
            dataset = []

        report = run_distribution(dataset, limit=args.limit, db=db)
        output = report.as_dict()
        print(json.dumps(output, indent=2, ensure_ascii=False))

        if args.export:
            Path(args.export).write_text(
                json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"\nReport exported to: {args.export}")

    finally:
        db.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
