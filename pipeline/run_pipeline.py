#!/usr/bin/env python3
"""
pipeline/run_pipeline.py
===========================
CLI entry point for Stages 2.1-2.12 + Section 3 + Section 4 (Data Preprocessing ->
News Classification -> Duplicate Detection -> Key Information/Claim
Extraction -> English Summarization -> Translation & Urdu Refinement ->
Bias & Sensational Detection -> Fact Verification -> Urdu Script
Generation -> Script Structuring -> Text-to-Speech -> Audio
Post-Processing -> Video Production 3.1-3.6 -> Distribution 4.1-4.4).

Examples
--------
    python -m pipeline.run_pipeline                    # run every stage
    python -m pipeline.run_pipeline --stage preprocess  # just Stage 2.1
    python -m pipeline.run_pipeline --export            # also write processed_news_export.json
    python -m pipeline.run_pipeline --stats             # print processed-DB stats and exit
"""

from __future__ import annotations

import argparse
import json
import sys

from data_acquisition import config as acquisition_config
from data_acquisition.database.raw_news_db import RawNewsDatabase
from data_acquisition.utils.logger import get_logger

from . import config as pcfg
from .pipeline_orchestrator import NewsProcessingPipeline
from .processed_news_db import ProcessedNewsDatabase

logger = get_logger("pipeline.cli")

_STAGE_METHODS = {
    "preprocess": "run_preprocessing",
    "classify": "run_classification",
    "dedup": "run_duplicate_detection",
    "extract": "run_extraction",
    "summarize": "run_summarization",
    "translate": "run_translation",
    "bias": "run_bias_detection",
    "verify": "run_fact_verification",
    "generate": "run_script_generation",
    "structure": "run_script_structuring",
    "tts": "run_tts",
    "audio": "run_audio_postprocessing",
    "video": "run_video_production",
    "distribute": "run_distribution",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stages 2.1-2.12 + Section 3 + Section 4 of the UrduNewsAI processing pipeline."
    )
    parser.add_argument(
        "--stage",
        choices=["all", *_STAGE_METHODS.keys()],
        default="all",
        help="Run a single stage, or the full pipeline in sequence (default: all)",
    )
    parser.add_argument("--raw-db-path", type=str, default=str(acquisition_config.DEFAULT_DB_PATH))
    parser.add_argument("--processed-db-path", type=str, default=str(pcfg.PROCESSED_DB_PATH))
    parser.add_argument("--limit", type=int, default=None, help="Limit how many raw articles Stage 2.1 reads")
    parser.add_argument("--export", action="store_true", help="Also export the final joined dataset to JSON")
    parser.add_argument(
        "--export-path", type=str, default=str(pcfg.FINAL_DATASET_EXPORT_PATH), help="Where to write --export output"
    )
    parser.add_argument("--stats", action="store_true", help="Print processed-DB stage counts and exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    raw_db = RawNewsDatabase(db_path=args.raw_db_path)
    processed_db = ProcessedNewsDatabase(db_path=args.processed_db_path)
    pipeline = NewsProcessingPipeline(raw_db=raw_db, processed_db=processed_db)

    try:
        if args.stats:
            dataset = processed_db.build_final_dataset()
            tts_map = processed_db.get_tts_map()
            audio_ok = sum(1 for v in tts_map.values() if v.get("status") == "ok")
            print(json.dumps(
                {
                    "processed_articles": len(dataset),
                    "tts_audio_files": audio_ok,
                    "processed_audio_files": sum(
                        1 for r in dataset if r.get("processed_audio_path")
                    ),
                },
                indent=2, ensure_ascii=False,
            ))
            return 0

        if args.stage == "all":
            report = pipeline.run_all(limit=args.limit)
            print(json.dumps(report.as_dict(), indent=2, ensure_ascii=False))
        else:
            method_name = _STAGE_METHODS[args.stage]
            method = getattr(pipeline, method_name)
            # Stages that accept limit
            if args.stage in ("preprocess", "video", "distribute"):
                stage_report = method(limit=args.limit)
            else:
                stage_report = method()
            print(json.dumps(
                {
                    "stage": stage_report.stage,
                    "processed": stage_report.processed,
                    "ok": stage_report.ok,
                    "skipped_or_error": stage_report.skipped_or_error,
                    "details": stage_report.details,
                },
                indent=2, ensure_ascii=False,
            ))

        if args.export:
            n = pipeline.export_final_dataset(args.export_path)
            print(f"\nExported {n} processed articles to: {args.export_path}")
    finally:
        raw_db.close()
        processed_db.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
