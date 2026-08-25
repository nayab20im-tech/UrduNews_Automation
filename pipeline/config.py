"""
pipeline/config.py
=====================
Configurable settings for Stages 2.1-2.5 (Requirement 12: "make
thresholds, model names, paths, and other important settings
configurable rather than hard-coded"). Mirrors the style of
`data_acquisition/config.py` -- everything overridable via environment
variables, sensible defaults otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path

from data_acquisition import config as acquisition_config

# By default the pipeline reads/writes alongside the existing raw DB, in
# the same `data/` folder the acquisition layer already uses.
PROCESSED_DB_PATH = Path(
    os.getenv("PROCESSED_DB_PATH", str(acquisition_config.DATA_DIR / "raw_news.db"))
)
FINAL_DATASET_EXPORT_PATH = Path(
    os.getenv("FINAL_DATASET_EXPORT_PATH", str(acquisition_config.DATA_DIR / "processed_news_export.json"))
)

# Stage 2.1
MIN_CONTENT_CHARS = int(os.getenv("MIN_CONTENT_CHARS", "5"))

# Stage 2.2
CLASSIFICATION_CONFIDENCE_THRESHOLD = float(os.getenv("CLASSIFICATION_CONFIDENCE_THRESHOLD", "0.08"))

# Stage 2.3
DUPLICATE_SIMILARITY_THRESHOLD = float(os.getenv("DUPLICATE_SIMILARITY_THRESHOLD", "0.65"))

# Stage 2.4
KEYPOINT_COUNT = int(os.getenv("KEYPOINT_COUNT", "3"))
MAX_CLAIMS = int(os.getenv("MAX_CLAIMS", "5"))

# Stage 2.5
SUMMARY_SENTENCE_COUNT = int(os.getenv("SUMMARY_SENTENCE_COUNT", "2"))
SUMMARY_KEYPOINT_COUNT = int(os.getenv("SUMMARY_KEYPOINT_COUNT", "4"))
