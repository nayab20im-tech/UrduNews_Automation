"""
distribution/distribution_db.py
==================================
SQLite storage for Section 4 — distribution jobs, analytics snapshots,
and playlist mappings.

Reuses the project's existing SQLite pattern from
``pipeline/processed_news_db.py``: thread-safe connection, UPSERT via
``ON CONFLICT … DO UPDATE``, ``row_factory = sqlite3.Row``.

By default the tables live in the **same** database file as the rest
of the pipeline (``raw_news.db``) so the whole project stays one
artifact.  Override ``DISTRIBUTION_DB_PATH`` to use a separate file.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from data_acquisition.utils.logger import get_logger

from . import config as dc

logger = get_logger("distribution.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS distribution_jobs (
    article_id          INTEGER PRIMARY KEY,
    job_id              TEXT,
    status              TEXT,
    video_path          TEXT,
    thumbnail_path      TEXT,
    headline            TEXT,
    category            TEXT,
    youtube_video_id    TEXT,
    youtube_url         TEXT,
    youtube_status      TEXT,
    facebook_post_id    TEXT,
    facebook_status     TEXT,
    twitter_post_id     TEXT,
    twitter_status      TEXT,
    telegram_message_id TEXT,
    telegram_status     TEXT,
    whatsapp_status     TEXT,
    retry_count         INTEGER DEFAULT 0,
    created_at          TEXT,
    updated_at          TEXT
);

CREATE TABLE IF NOT EXISTS analytics_snapshots (
    video_id        TEXT NOT NULL,
    platform        TEXT NOT NULL,
    snapshot_date   TEXT NOT NULL,
    views           INTEGER DEFAULT 0,
    likes           INTEGER DEFAULT 0,
    comments        INTEGER DEFAULT 0,
    shares          INTEGER DEFAULT 0,
    watch_time      REAL DEFAULT 0.0,
    engagement_rate REAL DEFAULT 0.0,
    raw_json        TEXT,
    PRIMARY KEY (video_id, platform, snapshot_date)
);

CREATE TABLE IF NOT EXISTS playlist_map (
    category      TEXT PRIMARY KEY,
    playlist_id   TEXT,
    playlist_name TEXT,
    updated_at    TEXT
);
"""


class DistributionDatabase:
    """Thread-safe SQLite wrapper for Section 4 distribution tables."""

    def __init__(self, db_path: Path | str = dc.DISTRIBUTION_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)
        logger.info("Distribution database connected at %s", self.db_path)

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

    # -- generic upsert ------------------------------------------------------
    def _upsert(self, table: str, row: Dict[str, Any], pk: str = "article_id") -> None:
        columns = list(row.keys())
        placeholders = ", ".join(f":{c}" for c in columns)
        col_list = ", ".join(columns)
        update_clause = ", ".join(
            f"{c}=excluded.{c}" for c in columns if c != pk
        )
        sql = (
            f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "
            f"ON CONFLICT({pk}) DO UPDATE SET {update_clause}"
        )
        with self._cursor() as cur:
            cur.execute(sql, row)

    # -- distribution_jobs ---------------------------------------------------
    def save_job(self, row: Dict[str, Any]) -> None:
        self._upsert("distribution_jobs", row)

    def get_job(self, article_id: int) -> Optional[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM distribution_jobs WHERE article_id = ?",
                (article_id,),
            )
            r = cur.fetchone()
            return dict(r) if r else None

    def get_jobs_by_status(self, status: str) -> List[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM distribution_jobs WHERE status = ?",
                (status,),
            )
            return [dict(r) for r in cur.fetchall()]

    def get_all_jobs(self) -> List[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM distribution_jobs ORDER BY article_id")
            return [dict(r) for r in cur.fetchall()]

    def get_pending_articles(self) -> List[Dict[str, Any]]:
        """Articles that have video_ready but are not yet fully distributed."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM distribution_jobs "
                "WHERE status IN ('pending', 'video_ready', 'retry_pending') "
                "ORDER BY article_id"
            )
            return [dict(r) for r in cur.fetchall()]

    def get_failed_jobs(self, max_retries: int = 3) -> List[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM distribution_jobs "
                "WHERE status IN ('failed', 'retry_pending') "
                "AND retry_count < ? "
                "ORDER BY article_id",
                (max_retries,),
            )
            return [dict(r) for r in cur.fetchall()]

    # -- analytics_snapshots -------------------------------------------------
    def save_snapshot(self, row: Dict[str, Any]) -> None:
        self._upsert(
            "analytics_snapshots", row,
            pk="video_id, platform, snapshot_date",
        )

    def _upsert_analytics(self, row: Dict[str, Any]) -> None:
        columns = list(row.keys())
        placeholders = ", ".join(f":{c}" for c in columns)
        col_list = ", ".join(columns)
        update_clause = ", ".join(
            f"{c}=excluded.{c}"
            for c in columns
            if c not in ("video_id", "platform", "snapshot_date")
        )
        sql = (
            f"INSERT INTO analytics_snapshots ({col_list}) VALUES ({placeholders}) "
            f"ON CONFLICT(video_id, platform, snapshot_date) DO UPDATE SET {update_clause}"
        )
        with self._cursor() as cur:
            cur.execute(sql, row)

    def save_snapshot(self, row: Dict[str, Any]) -> None:
        self._upsert_analytics(row)

    def get_snapshots(
        self, video_id: Optional[str] = None, platform: Optional[str] = None,
        start_date: str = "", end_date: str = "",
    ) -> List[Dict[str, Any]]:
        clauses, params = [], []
        if video_id:
            clauses.append("video_id = ?"); params.append(video_id)
        if platform:
            clauses.append("platform = ?"); params.append(platform)
        if start_date:
            clauses.append("snapshot_date >= ?"); params.append(start_date)
        if end_date:
            clauses.append("snapshot_date <= ?"); params.append(end_date)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._cursor() as cur:
            cur.execute(
                f"SELECT * FROM analytics_snapshots{where} ORDER BY snapshot_date",
                params,
            )
            return [dict(r) for r in cur.fetchall()]

    # -- playlist_map --------------------------------------------------------
    def save_playlist(self, category: str, playlist_id: str,
                      playlist_name: str, updated_at: str) -> None:
        self._upsert(
            "playlist_map",
            {
                "category": category,
                "playlist_id": playlist_id,
                "playlist_name": playlist_name,
                "updated_at": updated_at,
            },
            pk="category",
        )

    def get_playlist(self, category: str) -> Optional[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM playlist_map WHERE category = ?", (category,),
            )
            r = cur.fetchone()
            return dict(r) if r else None

    def get_all_playlists(self) -> Dict[str, Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM playlist_map")
            return {r["category"]: dict(r) for r in cur.fetchall()}

    # -- lifecycle -----------------------------------------------------------
    def close(self) -> None:
        with self._lock:
            self._conn.close()
        logger.info("Distribution database closed (%s)", self.db_path)

    def __enter__(self) -> "DistributionDatabase":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
