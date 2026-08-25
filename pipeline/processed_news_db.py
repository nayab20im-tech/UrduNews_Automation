"""
pipeline/processed_news_db.py
================================
Storage for Stages 2.1-2.5 output.

Reuses the project's existing storage format (SQLite, same convention as
`data_acquisition/database/raw_news_db.py`) and, by default, the *same*
database file as the raw layer -- new tables, not a new file -- so the
whole pipeline stays one artifact. Each table is keyed by `article_id`,
a foreign key back to `raw_news.id`. The raw table itself is only ever
touched for its `status` column (see `RawNewsDatabase`); every stage's
output lives in its own table here, keeping raw data fully intact
(Requirement 11).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from data_acquisition.utils.logger import get_logger

logger = get_logger("pipeline.processed_news_db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS preprocessed_news (
    article_id          INTEGER PRIMARY KEY,
    title               TEXT,
    source              TEXT,
    url                 TEXT,
    published_date      TEXT,
    language            TEXT,
    cleaned_text        TEXT,
    processed_text      TEXT,
    sentences_json       TEXT,
    preprocessing_status TEXT,
    error_message       TEXT,
    updated_at          TEXT
);

CREATE TABLE IF NOT EXISTS classified_news (
    article_id             INTEGER PRIMARY KEY,
    category               TEXT,
    confidence              REAL,
    classification_status  TEXT,
    updated_at             TEXT
);

CREATE TABLE IF NOT EXISTS duplicate_status (
    article_id          INTEGER PRIMARY KEY,
    duplicate_group_id  INTEGER,
    similarity_score    REAL,
    is_duplicate        INTEGER,
    is_primary          INTEGER,
    status              TEXT,
    updated_at          TEXT
);

CREATE TABLE IF NOT EXISTS extracted_info (
    article_id      INTEGER PRIMARY KEY,
    entities_json   TEXT,
    keypoints_json  TEXT,
    claims_json     TEXT,
    status          TEXT,
    updated_at      TEXT
);

CREATE TABLE IF NOT EXISTS summaries (
    article_id           INTEGER PRIMARY KEY,
    summary              TEXT,
    key_points_json      TEXT,
    bullet_summary_json  TEXT,
    status               TEXT,
    updated_at           TEXT
);
"""


class ProcessedNewsDatabase:
    """Thread-safe SQLite wrapper for all Stage 2.1-2.5 output tables."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)
        logger.info("Connected to processed news database at %s", self.db_path)

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

    def _upsert(self, table: str, row: Dict[str, Any]) -> None:
        columns = list(row.keys())
        placeholders = ", ".join(f":{c}" for c in columns)
        col_list = ", ".join(columns)
        update_clause = ", ".join(f"{c}=excluded.{c}" for c in columns if c != "article_id")
        sql = (
            f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "
            f"ON CONFLICT(article_id) DO UPDATE SET {update_clause}"
        )
        with self._cursor() as cur:
            cur.execute(sql, row)

    # -- writes, one per stage --------------------------------------------
    def save_preprocessed(self, rec, updated_at: str) -> None:
        self._upsert(
            "preprocessed_news",
            {
                "article_id": rec.article_id,
                "title": rec.title,
                "source": rec.source,
                "url": rec.url,
                "published_date": rec.published_date,
                "language": rec.language,
                "cleaned_text": rec.cleaned_text,
                "processed_text": rec.processed_text,
                "sentences_json": json.dumps(rec.sentences, ensure_ascii=False),
                "preprocessing_status": rec.preprocessing_status,
                "error_message": rec.error_message,
                "updated_at": updated_at,
            },
        )

    def save_classification(self, rec, updated_at: str) -> None:
        self._upsert(
            "classified_news",
            {
                "article_id": rec.article_id,
                "category": rec.category,
                "confidence": rec.confidence,
                "classification_status": rec.classification_status,
                "updated_at": updated_at,
            },
        )

    def save_duplicate_status(self, rec, updated_at: str) -> None:
        self._upsert(
            "duplicate_status",
            {
                "article_id": rec.article_id,
                "duplicate_group_id": rec.duplicate_group_id,
                "similarity_score": rec.similarity_score,
                "is_duplicate": int(rec.is_duplicate),
                "is_primary": int(rec.is_primary),
                "status": rec.status,
                "updated_at": updated_at,
            },
        )

    def save_extraction(self, rec, updated_at: str) -> None:
        self._upsert(
            "extracted_info",
            {
                "article_id": rec.article_id,
                "entities_json": json.dumps(rec.entities, ensure_ascii=False),
                "keypoints_json": json.dumps(rec.keypoints, ensure_ascii=False),
                "claims_json": json.dumps(rec.claims, ensure_ascii=False),
                "status": rec.status,
                "updated_at": updated_at,
            },
        )

    def save_summary(self, rec, updated_at: str) -> None:
        self._upsert(
            "summaries",
            {
                "article_id": rec.article_id,
                "summary": rec.summary,
                "key_points_json": json.dumps(rec.key_points, ensure_ascii=False),
                "bullet_summary_json": json.dumps(rec.bullet_summary, ensure_ascii=False),
                "status": rec.status,
                "updated_at": updated_at,
            },
        )

    # -- reads --------------------------------------------------------------
    def get_all_preprocessed(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM preprocessed_news"
        params: List[Any] = []
        if status:
            sql += " WHERE preprocessing_status = ?"
            params.append(status)
        with self._cursor() as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    def get_duplicate_map(self) -> Dict[int, Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM duplicate_status")
            return {r["article_id"]: dict(r) for r in cur.fetchall()}

    def build_final_dataset(self) -> List[Dict[str, Any]]:
        """Join every stage's table into one record per article, for the
        final "Processed News Dataset" export."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT
                    p.article_id, p.title, p.source, p.url, p.published_date,
                    p.language, p.cleaned_text, p.preprocessing_status,
                    c.category, c.confidence AS classification_confidence,
                    d.duplicate_group_id, d.similarity_score, d.is_duplicate, d.is_primary,
                    e.entities_json, e.keypoints_json, e.claims_json,
                    s.summary, s.key_points_json, s.bullet_summary_json
                FROM preprocessed_news p
                LEFT JOIN classified_news c ON c.article_id = p.article_id
                LEFT JOIN duplicate_status d ON d.article_id = p.article_id
                LEFT JOIN extracted_info e ON e.article_id = p.article_id
                LEFT JOIN summaries s ON s.article_id = p.article_id
                ORDER BY p.article_id
                """
            )
            rows = [dict(r) for r in cur.fetchall()]

        for row in rows:
            for key in ("entities_json", "keypoints_json", "claims_json", "key_points_json", "bullet_summary_json"):
                new_key = key.replace("_json", "")
                raw = row.pop(key, None)
                row[new_key] = json.loads(raw) if raw else ([] if new_key != "entities" else {})
        return rows

    def close(self) -> None:
        with self._lock:
            self._conn.close()
        logger.info("Processed news database connection closed (%s)", self.db_path)

    def __enter__(self) -> "ProcessedNewsDatabase":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
