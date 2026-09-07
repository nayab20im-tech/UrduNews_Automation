"""
orchestration/orchestrator.py
=============================
Implements Section 5's "Pipeline Orchestrator" box: workflow management,
step-wise execution, error handling & retry for the *entire* channel --

    collect    -> Section 1  (DataAcquisitionOrchestrator)
    process    -> Section 2  (NewsProcessingPipeline stages 2.1-2.12)
    video      -> Section 3  (video_production.pipeline)
    distribute -> Section 4  (distribution.pipeline)

Design rules (same spirit as every other layer in this project):
- step-wise: each stage is independently callable and chainable;
- failure isolation: one failing stage never stops the next;
- retry: each stage gets ``max_retries`` extra attempts with a delay;
- every stage attempt is persisted to the ``orchestration_runs`` table
  (Section 6 storage) so ``--status`` / dashboards can audit history;
- alerts flow through the Notification System (error alerts always,
  completion alerts when enabled).
"""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from data_acquisition import config as acq_cfg
from data_acquisition.database.raw_news_db import RawNewsDatabase
from data_acquisition.utils.logger import get_logger
from data_acquisition.utils.text_utils import safe_iso_now

from . import config as oc
from .notifications import NotificationManager

logger = get_logger("orchestration.orchestrator")

DEFAULT_STAGES = ["collect", "process", "video", "distribute"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS orchestration_runs (
    run_id       TEXT NOT NULL,
    stage        TEXT NOT NULL,
    status       TEXT NOT NULL,
    attempts     INTEGER NOT NULL DEFAULT 1,
    started_at   TEXT,
    finished_at  TEXT,
    summary_json TEXT,
    error        TEXT,
    PRIMARY KEY (run_id, stage)
);
CREATE INDEX IF NOT EXISTS idx_orchestration_runs_started ON orchestration_runs(started_at);
"""


@dataclass
class StageRun:
    stage: str
    status: str = "pending"          # ok | error | skipped
    attempts: int = 0
    started_at: str = ""
    finished_at: str = ""
    summary: Dict[str, Any] = field(default_factory=dict)
    error: str = ""


@dataclass
class OrchestrationReport:
    run_id: str
    started_at: str
    finished_at: str = ""
    stages: List[StageRun] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "stages": [
                {
                    "stage": s.stage,
                    "status": s.status,
                    "attempts": s.attempts,
                    "started_at": s.started_at,
                    "finished_at": s.finished_at,
                    "summary": s.summary,
                    "error": s.error,
                }
                for s in self.stages
            ],
        }


class EndToEndOrchestrator:
    """Drives Sections 1 -> 4 with per-stage retry and failure isolation."""

    def __init__(
        self,
        raw_db_path: Path | str = acq_cfg.DEFAULT_DB_PATH,
        processed_db_path: Optional[Path | str] = None,
        max_retries: int = oc.ORCH_MAX_RETRIES,
        retry_delay: float = oc.ORCH_RETRY_DELAY_SECONDS,
        notifier: Optional[NotificationManager] = None,
        logs_db=None,
        db_path: Path | str = oc.ORCH_DB_PATH,
    ):
        from pipeline import config as pcfg

        self.raw_db_path = Path(raw_db_path)
        self.processed_db_path = Path(processed_db_path or pcfg.PROCESSED_DB_PATH)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.notifier = notifier or NotificationManager()
        self.logs_db = logs_db

        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)

    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            finally:
                cur.close()

    def _log(self, level: str, run_id: str, stage: str, message: str, *args) -> None:
        getattr(logger, level)(message, *args)
        if self.logs_db is not None:
            self.logs_db.log(level.upper(), message % args if args else message,
                             logger_name="orchestration.orchestrator",
                             stage=stage, run_id=run_id)

    # -- workflow ------------------------------------------------------------
    def run(self, stages: Optional[List[str]] = None, limit: Optional[int] = None) -> OrchestrationReport:
        """Execute the requested stages in order (step-wise, isolated, retried)."""
        stages = list(stages) if stages else list(DEFAULT_STAGES)
        report = OrchestrationReport(run_id=uuid.uuid4().hex[:12], started_at=safe_iso_now())
        self._log("info", report.run_id, "", "Orchestration run started: stages=%s", stages)

        raw_db = RawNewsDatabase(db_path=self.raw_db_path)
        try:
            from pipeline.processed_news_db import ProcessedNewsDatabase

            processed_db = ProcessedNewsDatabase(db_path=self.processed_db_path)
        except Exception as exc:
            self._log("error", report.run_id, "", "Cannot open processed DB: %s", exc)
            processed_db = None

        try:
            for stage in stages:
                run = self._run_stage_with_retry(report.run_id, stage, limit, raw_db, processed_db)
                report.stages.append(run)
        finally:
            raw_db.close()
            if processed_db is not None:
                processed_db.close()

        report.finished_at = safe_iso_now()
        failed = [s.stage for s in report.stages if s.status == "error"]
        if failed:
            self.notifier.notify_error("orchestrator", f"run {report.run_id} failed stages: {failed}")
        else:
            self.notifier.notify_run_report(report.as_dict())
        self._log("info", report.run_id, "", "Orchestration run complete (failed=%s)", failed or "none")
        return report

    def _run_stage_with_retry(
        self, run_id: str, stage: str, limit: Optional[int],
        raw_db: RawNewsDatabase, processed_db,
    ) -> StageRun:
        run = StageRun(stage=stage, started_at=safe_iso_now())
        last_error = ""
        for attempt in range(1, self.max_retries + 2):
            run.attempts = attempt
            try:
                self._log("info", run_id, stage, "Stage '%s' starting (attempt %d/%d)",
                          stage, attempt, self.max_retries + 1)
                run.summary = self._dispatch(stage, limit, raw_db, processed_db)
                run.status = "ok"
                run.error = ""
                break
            except Exception as exc:
                last_error = str(exc)
                self._log("error", run_id, stage, "Stage '%s' attempt %d failed: %s",
                          stage, attempt, exc)
                if attempt <= self.max_retries:
                    time.sleep(self.retry_delay)
        else:
            run.status = "error"
            run.error = last_error
            self.notifier.notify_error(f"stage {stage}", last_error)

        run.finished_at = safe_iso_now()
        self._persist_run(run_id, run)
        return run

    def _dispatch(
        self, stage: str, limit: Optional[int],
        raw_db: RawNewsDatabase, processed_db,
    ) -> Dict[str, Any]:
        if stage == "collect":
            return self._stage_collect(raw_db)
        if stage == "process":
            return self._stage_process(raw_db, processed_db, limit)
        if stage == "video":
            return self._stage_video(processed_db, limit)
        if stage == "distribute":
            return self._stage_distribute(processed_db, limit)
        raise ValueError(f"unknown stage: {stage}")

    # -- Section 1 -----------------------------------------------------------
    def _stage_collect(self, raw_db: RawNewsDatabase) -> Dict[str, Any]:
        from data_acquisition.orchestrator.data_acquisition_orchestrator import (
            DataAcquisitionOrchestrator,
        )

        orchestrator = DataAcquisitionOrchestrator(
            sources_config=acq_cfg.load_sources_config(), db=raw_db
        )
        try:
            report = orchestrator.run_all()
        finally:
            orchestrator.close()
        return {"fetched": report.total_fetched, "inserted": report.total_inserted}

    # -- Section 2 -----------------------------------------------------------
    def _stage_process(self, raw_db: RawNewsDatabase, processed_db, limit: Optional[int]) -> Dict[str, Any]:
        from pipeline.pipeline_orchestrator import NewsProcessingPipeline

        pipeline = NewsProcessingPipeline(raw_db=raw_db, processed_db=processed_db)
        summary: Dict[str, Any] = {"stages": {}}
        steps = [
            ("2.1_preprocessing", lambda: pipeline.run_preprocessing(limit=limit)),
            ("2.2_classification", pipeline.run_classification),
            ("2.3_duplicate_detection", pipeline.run_duplicate_detection),
            ("2.4_extraction", pipeline.run_extraction),
            ("2.5_summarization", pipeline.run_summarization),
            ("2.6_translation", pipeline.run_translation),
            ("2.7_bias_detection", pipeline.run_bias_detection),
            ("2.8_fact_verification", pipeline.run_fact_verification),
            ("2.9_script_generation", pipeline.run_script_generation),
            ("2.10_script_structuring", pipeline.run_script_structuring),
            ("2.11_tts", pipeline.run_tts),
            ("2.12_audio_postprocessing", pipeline.run_audio_postprocessing),
        ]
        for name, step in steps:
            try:
                rep = step()
                summary["stages"][name] = {
                    "processed": rep.processed, "ok": rep.ok,
                    "skipped_or_error": rep.skipped_or_error,
                }
            except Exception as exc:  # per-step isolation, like run_all()
                logger.error("Step %s failed: %s", name, exc)
                summary["stages"][name] = {"error": str(exc)}
        return summary

    # -- Section 3 -----------------------------------------------------------
    def _stage_video(self, processed_db, limit: Optional[int]) -> Dict[str, Any]:
        from video_production.pipeline import run_video_production

        dataset = processed_db.build_final_dataset()
        report = run_video_production(dataset, limit=limit)
        return {
            "processed": len(report.outputs),
            "ok": sum(1 for o in report.outputs if o.get("subtitled_video")),
        }

    # -- Section 4 -----------------------------------------------------------
    def _stage_distribute(self, processed_db, limit: Optional[int]) -> Dict[str, Any]:
        from distribution.distribution_db import DistributionDatabase
        from distribution.pipeline import run_distribution

        dataset = processed_db.build_final_dataset()
        db = DistributionDatabase()
        try:
            report = run_distribution(dataset, limit=limit, db=db)
        finally:
            db.close()
        return {
            "processed": len(report.outputs),
            "ok": sum(
                1 for o in report.outputs
                if o.get("youtube_status") in ("published", "mock_published")
            ),
        }

    # -- run history -----------------------------------------------------------
    def _persist_run(self, run_id: str, run: StageRun) -> None:
        import json

        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO orchestration_runs
                    (run_id, stage, status, attempts, started_at, finished_at, summary_json, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, stage) DO UPDATE SET
                    status = excluded.status, attempts = excluded.attempts,
                    finished_at = excluded.finished_at,
                    summary_json = excluded.summary_json, error = excluded.error
                """,
                (run_id, run.stage, run.status, run.attempts, run.started_at,
                 run.finished_at, json.dumps(run.summary, ensure_ascii=False, default=str),
                 run.error),
            )

    def retry_failed(self, limit: Optional[int] = None) -> OrchestrationReport:
        """Re-run every stage that ended in error on the most recent run."""
        with self._cursor() as cur:
            cur.execute("SELECT MAX(started_at) AS latest FROM orchestration_runs")
            latest = cur.fetchone()["latest"]
            if not latest:
                return OrchestrationReport(run_id=uuid.uuid4().hex[:12], started_at=safe_iso_now())
            cur.execute(
                "SELECT stage FROM orchestration_runs WHERE started_at = ? AND status = 'error' "
                "ORDER BY rowid",
                (latest,),
            )
            failed = [r["stage"] for r in cur.fetchall()]
        if not failed:
            logger.info("No failed stages to retry")
            return OrchestrationReport(run_id=uuid.uuid4().hex[:12], started_at=safe_iso_now())
        logger.info("Retrying failed stages: %s", failed)
        return self.run(stages=failed, limit=limit)

    def status(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Recent run history, newest first."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM orchestration_runs ORDER BY started_at DESC, stage LIMIT ?",
                (limit,),
            )
            return [dict(r) for r in cur.fetchall()]

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "EndToEndOrchestrator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
