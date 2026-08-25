"""
pipeline/pipeline_orchestrator.py
====================================
Wires Stages 2.1 -> 2.5 together, each stage's output becoming the next
stage's input:

    RawNewsDatabase (status="raw")
        -> 2.1 Preprocessor            -> preprocessed_news
        -> 2.2 NewsClassifier          -> classified_news
        -> 2.3 DuplicateDetector       -> duplicate_status
        -> 2.4 InfoExtractor           -> extracted_info      (primary articles only)
        -> 2.5 Summarizer              -> summaries           (primary articles only)

Every stage is independently callable (`run_preprocessing()`, etc.) as
well as chainable via `run_all()`, and every stage tolerates individual
article failures without stopping the batch (each `*.run()` already
does this -- see the per-stage modules).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from data_acquisition.database.raw_news_db import RawNewsDatabase
from data_acquisition.utils.logger import get_logger
from data_acquisition.utils.text_utils import safe_iso_now

from classification.classifier import NewsClassifier
from duplicate_detection.duplicate_detector import DuplicateDetector
from extraction.extractor import InfoExtractor
from preprocessing.preprocessor import Preprocessor
from summarization.summarizer import Summarizer

from . import config as pcfg
from .processed_news_db import ProcessedNewsDatabase

logger = get_logger("pipeline.orchestrator")


@dataclass
class StageReport:
    stage: str
    processed: int = 0
    ok: int = 0
    skipped_or_error: int = 0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineReport:
    started_at: str
    finished_at: str = ""
    stages: List[StageReport] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "stages": [
                {
                    "stage": s.stage,
                    "processed": s.processed,
                    "ok": s.ok,
                    "skipped_or_error": s.skipped_or_error,
                    "details": s.details,
                }
                for s in self.stages
            ],
        }


class NewsProcessingPipeline:
    def __init__(
        self,
        raw_db: RawNewsDatabase,
        processed_db: ProcessedNewsDatabase,
        min_content_chars: int = pcfg.MIN_CONTENT_CHARS,
        classification_confidence_threshold: float = pcfg.CLASSIFICATION_CONFIDENCE_THRESHOLD,
        duplicate_similarity_threshold: float = pcfg.DUPLICATE_SIMILARITY_THRESHOLD,
        keypoint_count: int = pcfg.KEYPOINT_COUNT,
        max_claims: int = pcfg.MAX_CLAIMS,
        summary_sentence_count: int = pcfg.SUMMARY_SENTENCE_COUNT,
        summary_keypoint_count: int = pcfg.SUMMARY_KEYPOINT_COUNT,
    ):
        self.raw_db = raw_db
        self.processed_db = processed_db

        self.preprocessor = Preprocessor(min_content_chars=min_content_chars)
        self.classifier = NewsClassifier(confidence_threshold=classification_confidence_threshold)
        self.duplicate_detector = DuplicateDetector(similarity_threshold=duplicate_similarity_threshold)
        self.extractor = InfoExtractor()
        self.summarizer = Summarizer(
            summary_sentence_count=summary_sentence_count, keypoint_count=summary_keypoint_count
        )

    # -- Stage 2.1 ------------------------------------------------------
    def run_preprocessing(self, limit: Optional[int] = None) -> StageReport:
        raw_rows = self.raw_db.get_all(status="raw", limit=limit)
        results = self.preprocessor.run(raw_rows)

        now = safe_iso_now()
        status_updates: Dict[str, str] = {
            "ok": "preprocessed", "empty_content": "preprocessed_empty",
            "duplicate": "duplicate_raw", "error": "preprocess_error",
        }
        for rec in results:
            self.processed_db.save_preprocessed(rec, now)
            self.raw_db.update_status(rec.article_id, status_updates.get(rec.preprocessing_status, "preprocess_error"))

        ok = sum(1 for r in results if r.preprocessing_status == "ok")
        return StageReport(
            stage="2.1_preprocessing", processed=len(results), ok=ok, skipped_or_error=len(results) - ok,
            details={"by_status": _count_by(results, "preprocessing_status")},
        )

    # -- Stage 2.2 --------------------------------------------------------
    def run_classification(self) -> StageReport:
        rows = self.processed_db.get_all_preprocessed(status="ok")
        articles = [{"article_id": r["article_id"], "text": r["processed_text"] or r["cleaned_text"]} for r in rows]
        results = self.classifier.run(articles)

        now = safe_iso_now()
        for rec in results:
            self.processed_db.save_classification(rec, now)

        ok = sum(1 for r in results if r.classification_status == "ok")
        return StageReport(
            stage="2.2_classification", processed=len(results), ok=ok, skipped_or_error=len(results) - ok,
            details={"by_category": _count_by(results, "category")},
        )

    # -- Stage 2.3 ----------------------------------------------------------
    def run_duplicate_detection(self) -> StageReport:
        rows = self.processed_db.get_all_preprocessed(status="ok")
        articles = [
            {"article_id": r["article_id"], "title": r["title"], "text": r["cleaned_text"]} for r in rows
        ]
        results = self.duplicate_detector.detect(articles)

        now = safe_iso_now()
        for rec in results:
            self.processed_db.save_duplicate_status(rec, now)

        dup_count = sum(1 for r in results if r.is_duplicate)
        return StageReport(
            stage="2.3_duplicate_detection", processed=len(results), ok=len(results) - dup_count,
            skipped_or_error=dup_count,
            details={"groups": len({r.duplicate_group_id for r in results if r.duplicate_group_id})},
        )

    # -- Stage 2.4 -------------------------------------------------------------
    def run_extraction(self) -> StageReport:
        rows = self._valid_primary_rows()
        articles = [
            {"article_id": r["article_id"], "text": r["cleaned_text"], "sentences": _load_sentences(r)}
            for r in rows
        ]
        results = self.extractor.run(articles)

        now = safe_iso_now()
        for rec in results:
            self.processed_db.save_extraction(rec, now)

        ok = sum(1 for r in results if r.status == "ok")
        return StageReport(
            stage="2.4_extraction", processed=len(results), ok=ok, skipped_or_error=len(results) - ok,
        )

    # -- Stage 2.5 ------------------------------------------------------------
    def run_summarization(self) -> StageReport:
        rows = self._valid_primary_rows()
        articles = [
            {"article_id": r["article_id"], "sentences": _load_sentences(r), "language": r["language"]}
            for r in rows
        ]
        results = self.summarizer.run(articles)

        now = safe_iso_now()
        for rec in results:
            self.processed_db.save_summary(rec, now)

        ok = sum(1 for r in results if r.status == "ok")
        return StageReport(
            stage="2.5_summarization", processed=len(results), ok=ok, skipped_or_error=len(results) - ok,
            details={"by_status": _count_by(results, "status")},
        )

    def _valid_primary_rows(self) -> List[Dict[str, Any]]:
        """Rows that are preprocessed OK and not a Stage-2.3 duplicate."""
        rows = self.processed_db.get_all_preprocessed(status="ok")
        dup_map = self.processed_db.get_duplicate_map()
        return [r for r in rows if not dup_map.get(r["article_id"], {}).get("is_duplicate")]

    # -- full pipeline ------------------------------------------------------
    def run_all(self, limit: Optional[int] = None) -> PipelineReport:
        report = PipelineReport(started_at=safe_iso_now())
        report.stages.append(self.run_preprocessing(limit=limit))
        report.stages.append(self.run_classification())
        report.stages.append(self.run_duplicate_detection())
        report.stages.append(self.run_extraction())
        report.stages.append(self.run_summarization())
        report.finished_at = safe_iso_now()
        logger.info("Pipeline run complete.")
        return report

    def export_final_dataset(self, path) -> int:
        import json

        dataset = self.processed_db.build_final_dataset()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        logger.info("Exported final processed dataset (%d articles) to %s", len(dataset), path)
        return len(dataset)


def _count_by(records: List[Any], attr: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in records:
        key = str(getattr(r, attr))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _load_sentences(row: Dict[str, Any]) -> List[str]:
    import json

    raw = row.get("sentences_json") or row.get("sentences")
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return []
