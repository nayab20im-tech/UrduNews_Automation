"""
storage/logs_db.py
==================
Implements Section 6's "Logs DB (System Logs)" box.

SQLite-backed system log store plus a ``logging.Handler`` that mirrors
any logger's records into it, so Section 5 (Orchestration) can query,
report on and purge log history the same way it queries news data.

The handler is wired in through ``data_acquisition.utils.logger.
add_global_handler`` -- one call mirrors every module logger (console +
rotating file + SQLite) without touching a single call site.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from data_acquisition.utils.logger import add_global_handler, get_logger
from data_acquisition.utils.text_utils import safe_iso_now

from . import config as sc

logger = get_logger("storage.logs_db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS system_logs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at  TEXT NOT NULL,
    level      TEXT NOT NULL,
    logger     TEXT,
    stage      TEXT,
    run_id     TEXT,
    message    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_system_logs_level ON system_logs(level);
CREATE INDEX IF NOT EXISTS idx_system_logs_logged_at ON system_logs(logged_at);
CREATE INDEX IF NOT EXISTS idx_system_logs_run_id ON system_logs(run_id);
"""


class LogsDatabase:
    """Thread-safe SQLite wrapper for the system logs store."""

    def __init__(self, db_path: Path | str = sc.LOGS_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._closed = False
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)
        logger.info("Connected to logs database at %s", self.db_path)

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

    # -- writes -----------------------------------------------------------
    def log(
        self,
        level: str,
        message: str,
        logger_name: str = "",
        stage: str = "",
        run_id: str = "",
    ) -> None:
        """Insert one log row. Never raises -- a logging sink must not
        break the pipeline it observes."""
        if self._closed:
            return
        try:
            with self._cursor() as cur:
                cur.execute(
                    "INSERT INTO system_logs (logged_at, level, logger, stage, run_id, message) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (safe_iso_now(), str(level).upper(), logger_name, stage, run_id, message),
                )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to persist log row: %s", exc)

    # -- reads --------------------------------------------------------------
    def get_logs(
        self,
        level: Optional[str] = None,
        stage: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM system_logs"
        clauses: List[str] = []
        params: List[Any] = []
        if level:
            clauses.append("level = ?")
            params.append(str(level).upper())
        if stage:
            clauses.append("stage = ?")
            params.append(stage)
        if run_id:
            clauses.append("run_id = ?")
            params.append(run_id)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._cursor() as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    def stats(self) -> Dict[str, Any]:
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) AS c FROM system_logs")
            total = cur.fetchone()["c"]
            cur.execute("SELECT level, COUNT(*) AS c FROM system_logs GROUP BY level")
            by_level = {r["level"]: r["c"] for r in cur.fetchall()}
            cur.execute("SELECT MIN(logged_at) AS o, MAX(logged_at) AS n FROM system_logs")
            span = cur.fetchone()
        return {
            "total_entries": total,
            "by_level": by_level,
            "oldest": span["o"],
            "newest": span["n"],
        }

    # -- maintenance ---------------------------------------------------------
    def purge(self, older_than_days: int = sc.LOG_RETENTION_DAYS) -> int:
        """Delete log rows older than the retention window; returns count."""
        from datetime import datetime, timedelta, timezone

        cutoff = (datetime.now(timezone.utc) - timedelta(days=older_than_days)).isoformat(
            timespec="seconds"
        )
        with self._cursor() as cur:
            cur.execute("DELETE FROM system_logs WHERE logged_at < ?", (cutoff,))
            deleted = cur.rowcount
        logger.info("Purged %d log rows older than %s", deleted, cutoff)
        return deleted

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._conn.close()
        logger.info("Logs database connection closed (%s)", self.db_path)

    def __enter__(self) -> "LogsDatabase":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class SQLiteLogHandler(logging.Handler):
    """``logging.Handler`` that mirrors records into a ``LogsDatabase``."""

    def __init__(self, db: LogsDatabase, stage: str = "", run_id: str = ""):
        super().__init__()
        self.db = db
        self.stage = stage
        self.run_id = run_id

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.db.log(
                record.levelname,
                record.getMessage(),
                logger_name=record.name,
                stage=self.stage,
                run_id=self.run_id,
            )
        except Exception:  # pragma: no cover - defensive
            self.handleError(record)


def attach_sqlite_handler(
    db: LogsDatabase,
    logger_name: Optional[str] = None,
    stage: str = "",
    run_id: str = "",
) -> SQLiteLogHandler:
    """Mirror logs into SQLite.

    With ``logger_name`` only that logger is mirrored; without it the
    handler is registered globally so every module logger (and every
    logger created later via ``get_logger``) is mirrored too.
    """
    handler = SQLiteLogHandler(db, stage=stage, run_id=run_id)
    if logger_name:
        logging.getLogger(logger_name).addHandler(handler)
    else:
        add_global_handler(handler)
    return handler
