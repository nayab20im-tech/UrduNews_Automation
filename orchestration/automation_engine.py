"""
orchestration/automation_engine.py
==================================
Implements Section 5's "Automation Engine" box: task scheduling, job
queue and logging.

- ``automation_jobs`` -- durable SQLite job queue (survives restarts).
- ``register_handler``/``submit``/``run_due`` -- pull-style execution
  with per-job attempt counting and automatic re-queue on failure.
- ``start()/stop()`` -- ``sched``-based polling loop (same pattern as
  ``distribution/scheduler.py``: stdlib only, no external scheduler).
- every state transition is logged through the shared logger and, when
  a ``LogsDatabase`` is supplied, mirrored into Section 6's Logs DB.

Default handlers map queue names onto the end-to-end orchestrator so a
cron-less host can drive the whole channel from one queue.
"""

from __future__ import annotations

import json
import sched
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

from data_acquisition.utils.logger import get_logger
from data_acquisition.utils.text_utils import safe_iso_now

from . import config as oc

logger = get_logger("orchestration.automation_engine")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS automation_jobs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    payload_json  TEXT,
    status        TEXT NOT NULL DEFAULT 'pending',
    attempts      INTEGER NOT NULL DEFAULT 0,
    max_attempts  INTEGER NOT NULL DEFAULT 3,
    scheduled_at  TEXT NOT NULL,
    started_at    TEXT,
    finished_at   TEXT,
    last_error    TEXT,
    result_json   TEXT,
    created_at    TEXT,
    updated_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_automation_jobs_status ON automation_jobs(status);
CREATE INDEX IF NOT EXISTS idx_automation_jobs_scheduled ON automation_jobs(scheduled_at);
"""


class AutomationEngine:
    """Durable job queue + polling scheduler for pipeline tasks."""

    def __init__(
        self,
        db_path: Path | str = oc.ORCH_DB_PATH,
        logs_db=None,
        poll_interval: float = oc.JOB_POLL_INTERVAL_SECONDS,
        retry_delay: float = oc.JOB_RETRY_DELAY_SECONDS,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.logs_db = logs_db
        self.poll_interval = poll_interval
        self.retry_delay = retry_delay
        self._handlers: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
        self._stop_event = threading.Event()
        self._scheduler = sched.scheduler(time.time, time.sleep)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)
        logger.info("Automation engine connected at %s", self.db_path)

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

    def _log(self, level: str, message: str, *args) -> None:
        getattr(logger, level)(message, *args)
        if self.logs_db is not None:
            self.logs_db.log(level.upper(), message % args if args else message,
                             logger_name="orchestration.automation_engine", stage="automation")

    # -- handlers -----------------------------------------------------------
    def register_handler(self, name: str, fn: Callable[[Dict[str, Any]], Any]) -> None:
        self._handlers[name] = fn

    def register_default_handlers(self) -> None:
        """Wire queue names onto the end-to-end orchestrator + analytics."""
        from .orchestrator import EndToEndOrchestrator

        orchestrator = EndToEndOrchestrator(logs_db=self.logs_db)

        def _run(stages: List[str]) -> Callable[[Dict[str, Any]], Any]:
            def handler(payload: Dict[str, Any]) -> Any:
                report = orchestrator.run(stages=stages, limit=payload.get("limit"))
                return report.as_dict()
            return handler

        self.register_handler("collect", _run(["collect"]))
        self.register_handler("process", _run(["process"]))
        self.register_handler("video", _run(["video"]))
        self.register_handler("distribute", _run(["distribute"]))
        self.register_handler("full_pipeline", _run(["collect", "process", "video", "distribute"]))

        def _analytics(payload: Dict[str, Any]) -> Any:
            from distribution import analytics
            from distribution.distribution_db import DistributionDatabase

            db = DistributionDatabase()
            try:
                count = analytics.collect_all_stats(db)
                report = analytics.generate_daily_report(db)
            finally:
                db.close()
            return {"collected": count, "report": report.get("summary", {})}

        self.register_handler("analytics", _analytics)

    # -- queue writes -----------------------------------------------------------
    def submit(
        self,
        name: str,
        payload: Optional[Dict[str, Any]] = None,
        delay_seconds: float = 0,
        max_attempts: int = oc.JOB_MAX_ATTEMPTS,
    ) -> int:
        """Enqueue a job; returns its id. ``delay_seconds`` defers execution."""
        from datetime import datetime, timedelta, timezone

        scheduled = datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)
        now = safe_iso_now()
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO automation_jobs
                    (name, payload_json, status, attempts, max_attempts,
                     scheduled_at, created_at, updated_at)
                VALUES (?, ?, 'pending', 0, ?, ?, ?, ?)
                """,
                (name, json.dumps(payload or {}, ensure_ascii=False), max_attempts,
                 scheduled.isoformat(timespec="seconds"), now, now),
            )
            job_id = cur.lastrowid
        self._log("info", "Job #%d '%s' submitted (max_attempts=%d)", job_id, name, max_attempts)
        return job_id

    # -- queue reads -----------------------------------------------------------
    def get_job(self, job_id: int) -> Optional[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM automation_jobs WHERE id = ?", (job_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def list_jobs(self, status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM automation_jobs"
        params: List[Any] = []
        if status:
            sql += " WHERE status = ?"
            params.append(status)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._cursor() as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    # -- execution -----------------------------------------------------------
    def run_due(self) -> List[Dict[str, Any]]:
        """Execute every pending job whose scheduled_at has passed.

        Returns one result dict per executed job. A failed job is
        re-queued (status back to pending, scheduled_at deferred) until
        its attempts are exhausted, then marked failed.
        """
        results: List[Dict[str, Any]] = []
        for job in self._claim_due():
            results.append(self._execute(job))
        return results

    def _claim_due(self) -> List[Dict[str, Any]]:
        now = safe_iso_now()
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM automation_jobs WHERE status = 'pending' AND scheduled_at <= ? "
                "ORDER BY id",
                (now,),
            )
            jobs = [dict(r) for r in cur.fetchall()]
            for job in jobs:
                cur.execute(
                    "UPDATE automation_jobs SET status = 'running', started_at = ?, updated_at = ? "
                    "WHERE id = ?",
                    (now, now, job["id"]),
                )
        return jobs

    def _execute(self, job: Dict[str, Any]) -> Dict[str, Any]:
        name = job["name"]
        handler = self._handlers.get(name)
        now = safe_iso_now()
        attempts = job["attempts"] + 1

        if handler is None:
            self._finish(job, "failed", attempts, now,
                         last_error=f"no handler registered for '{name}'")
            return {"job_id": job["id"], "name": name, "status": "failed",
                    "error": f"no handler registered for '{name}'"}

        try:
            payload = json.loads(job["payload_json"] or "{}")
            result = handler(payload)
            self._finish(job, "done", attempts, now,
                         result_json=json.dumps(result, ensure_ascii=False, default=str))
            self._log("info", "Job #%d '%s' done (attempt %d)", job["id"], name, attempts)
            return {"job_id": job["id"], "name": name, "status": "done", "result": result}
        except Exception as exc:
            if attempts < job["max_attempts"]:
                from datetime import datetime, timedelta, timezone

                retry_at = (datetime.now(timezone.utc)
                            + timedelta(seconds=self.retry_delay)).isoformat(timespec="seconds")
                with self._cursor() as cur:
                    cur.execute(
                        "UPDATE automation_jobs SET status = 'pending', attempts = ?, "
                        "scheduled_at = ?, last_error = ?, updated_at = ? WHERE id = ?",
                        (attempts, retry_at, str(exc), safe_iso_now(), job["id"]),
                    )
                self._log("warning", "Job #%d '%s' failed (attempt %d/%d), re-queued: %s",
                          job["id"], name, attempts, job["max_attempts"], exc)
                return {"job_id": job["id"], "name": name, "status": "pending",
                        "attempts": attempts, "error": str(exc)}
            self._finish(job, "failed", attempts, now, last_error=str(exc))
            self._log("error", "Job #%d '%s' failed permanently: %s", job["id"], name, exc)
            return {"job_id": job["id"], "name": name, "status": "failed", "error": str(exc)}

    def _finish(self, job: Dict[str, Any], status: str, attempts: int, now: str,
                last_error: str = "", result_json: str = "") -> None:
        with self._cursor() as cur:
            cur.execute(
                "UPDATE automation_jobs SET status = ?, attempts = ?, finished_at = ?, "
                "last_error = ?, result_json = ?, updated_at = ? WHERE id = ?",
                (status, attempts, now, last_error, result_json, now, job["id"]),
            )

    # -- scheduling loop -----------------------------------------------------------
    def _tick(self) -> None:
        if self._stop_event.is_set():
            return
        try:
            self.run_due()
        except Exception as exc:  # keep the loop alive no matter what
            self._log("error", "Automation tick failed: %s", exc)
        if not self._stop_event.is_set():
            self._scheduler.enter(self.poll_interval, 1, self._tick)

    def start(self, blocking: bool = True) -> None:
        """Poll loop; identical usage pattern to distribution.scheduler."""
        if blocking:
            logger.info("Automation engine starting (blocking, poll=%.0fs)", self.poll_interval)
            self._tick()
            self._scheduler.run()
        else:
            t = threading.Thread(target=self._run_thread, daemon=True)
            t.start()
            logger.info("Automation engine started in background thread")

    def _run_thread(self) -> None:
        self._tick()
        self._scheduler.run()

    def stop(self) -> None:
        self._stop_event.set()
        logger.info("Automation engine stop signal sent")

    def close(self) -> None:
        with self._lock:
            self._conn.close()
        logger.info("Automation engine connection closed (%s)", self.db_path)

    def __enter__(self) -> "AutomationEngine":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
