#!/usr/bin/env python3
"""
video_production/run_video.py
================================
CLI entry point for Section 3 — Video Production Pipeline.

Examples
--------
    # Run video production on all processed articles
    python -m video_production.run_video

    # Limit to first 3 articles
    python -m video_production.run_video --limit 3

    # Use a custom processed-DB path
    python -m video_production.run_video --processed-db-path data/raw_news.db

    # Print stats only
    python -m video_production.run_video --stats
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from data_acquisition import config as acquisition_config
from data_acquisition.utils.logger import get_logger

from pipeline import config as pcfg
from pipeline.processed_news_db import ProcessedNewsDatabase

from .pipeline import run_video_production

logger = get_logger("video_production.cli")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Section 3 Video Production Pipeline (3.1-3.6) "
                    "on processed Urdu news articles.",
    )
    parser.add_argument(
        "--processed-db-path", type=str,
        default=str(pcfg.PROCESSED_DB_PATH),
        help="Path to the processed news SQLite database.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N articles.",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Print video production statistics and exit.",
    )
    parser.add_argument(
        "--export", action="store_true",
        help="Also export video production results to JSON.",
    )
    parser.add_argument(
        "--export-path", type=str,
        default=str(acquisition_config.DATA_DIR / "video_production_export.json"),
        help="Where to write --export output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    processed_db = ProcessedNewsDatabase(db_path=args.processed_db_path)

    try:
        if args.stats:
            dataset = processed_db.build_final_dataset()
            audio_count = sum(
                1 for r in dataset
                if r.get("processed_audio_path") or r.get("tts_audio_path")
            )
            print(json.dumps({
                "total_articles": len(dataset),
                "articles_with_audio": audio_count,
                "ready_for_video_production": audio_count,
            }, indent=2, ensure_ascii=False))
            return 0

        # Build final dataset and run video production
        dataset = processed_db.build_final_dataset()
        if not dataset:
            logger.warning("No articles in processed database; nothing to produce.")
            print(json.dumps({"message": "No articles available"}, indent=2))
            return 0

        report = run_video_production(dataset, limit=args.limit)
        print(json.dumps(report.as_dict(), indent=2, ensure_ascii=False))

        if args.export:
            with open(args.export_path, "w", encoding="utf-8") as f:
                json.dump(report.as_dict(), f, ensure_ascii=False, indent=2)
            print(f"\nExported video production report to: {args.export_path}")

    finally:
        processed_db.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
