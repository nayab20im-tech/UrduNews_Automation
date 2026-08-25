#!/usr/bin/env python3
"""
main.py
=======
CLI entry point for Section 1 (Data Collection Layer).

Examples
--------
    python -m data_acquisition.main                     # run every enabled source
    python -m data_acquisition.main --sources rss,manual # run only these sources
    python -m data_acquisition.main --export json        # also dump raw_news.json
    python -m data_acquisition.main --stats              # just print DB stats and exit
"""

from __future__ import annotations

import argparse
import json
import sys

from . import config
from .database.raw_news_db import RawNewsDatabase
from .orchestrator.data_acquisition_orchestrator import DataAcquisitionOrchestrator
from .utils.logger import get_logger

logger = get_logger(__name__)

_TYPE_ALIASES = {
    "rss": "rss",
    "website": "website",
    "websites": "website",
    "api": "api",
    "newsapi": "api",
    "trends": "trends",
    "social": "social",
    "manual": "manual",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the UrduNewsAI Data Acquisition layer.")
    parser.add_argument(
        "--sources",
        type=str,
        default="all",
        help="Comma-separated source types to run: rss,website,api,trends,social,manual "
        "(default: all enabled sources)",
    )
    parser.add_argument("--db-path", type=str, default=str(config.DEFAULT_DB_PATH), help="SQLite DB path")
    parser.add_argument(
        "--sources-config", type=str, default=str(config.SOURCES_CONFIG_PATH), help="Path to sources_config.yaml"
    )
    parser.add_argument("--export", choices=["json"], help="Also export the full raw DB to JSON after the run")
    parser.add_argument("--stats", action="store_true", help="Print DB stats and exit (no scraping)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    db = RawNewsDatabase(db_path=args.db_path)

    if args.stats:
        print(json.dumps(db.get_stats(), indent=2, ensure_ascii=False))
        db.close()
        return 0

    sources_cfg = config.load_sources_config(path=args.sources_config)

    if args.sources != "all":
        requested = {_TYPE_ALIASES.get(s.strip().lower(), s.strip().lower()) for s in args.sources.split(",")}
        if "rss" not in requested:
            sources_cfg["rss_feeds"] = []
        if "website" not in requested:
            sources_cfg["websites"] = []
        if "api" not in requested:
            sources_cfg["news_api"] = {"enabled": False}
        if "trends" not in requested:
            sources_cfg["google_trends"] = {"enabled": False}
        if "social" not in requested:
            sources_cfg["social_media"] = {"enabled": False}
        if "manual" not in requested:
            sources_cfg["manual_input"] = {"enabled": False}

    orchestrator = DataAcquisitionOrchestrator(sources_config=sources_cfg, db=db)
    try:
        report = orchestrator.run_all()
    finally:
        orchestrator.close()

    print(json.dumps(report.as_dict(), indent=2, ensure_ascii=False))

    if args.export == "json":
        export_path = config.DATA_DIR / "raw_news_export.json"
        db.export_to_json(export_path)
        print(f"\nExported full raw database to: {export_path}")

    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
