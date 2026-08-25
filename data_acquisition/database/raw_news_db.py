"""
database/raw_news_db.py
========================
Implements Section 1's "Raw News Database (Local JSON / SQLite)" box.

SQLite is the primary store (transactional, queryable, safe for a single
local pipeline). `export_to_json()` produces the JSON snapshot the diagram
also calls for, e.g. for handing off to Section 2 (Data Preprocessing) or
for quick inspection.

Every insert is de-duplicated on a content hash (computed from the URL,
or from title+content when no URL exists) so re-running collection never
creates duplicate rows -- this is intentionally separate from Section 2.3
"Duplicate Detection" (which does semantic/near-duplicate detection later
in the pipeline); this is just exact/near-exact raw-level dedup.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from ..utils.logger import get_logger
from ..utils.text_utils import clean_text, compute_content_hash, safe_iso_now

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_news (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    title           TEXT NOT NULL,
    content         TEXT,
    url             TEXT,
    source          TEXT NOT NULL,
    source_type     TEXT NOT NULL,
    language        TEXT DEFAULT 'unknown',
    category        TEXT DEFAULT 'uncategorized',
    published_date  TEXT,
    scraped_at      TEXT NOT NULL,
    raw_html        TEXT,
    extra_json      TEXT,
    content_hash    TEXT NOT NULL UNIQUE,
    status          TEXT NOT NULL DEFAULT 'raw'
);
CREATE INDEX IF NOT EXISTS idx_raw_news_source_type ON raw_news(source_type);
CREATE INDEX IF NOT EXISTS idx_raw_news_scraped_at ON raw_news(scraped_at);
CREATE INDEX IF NOT EXISTS idx_raw_news_status ON raw_news(status);
"""


@dataclass
class Article:
    """Canonical article record used by every scraper in this layer."""

    title: str
    content: str = ""
    url: str = ""
    source: str = "unknown"
    source_type: str = "unknown"   # website | rss | api | trends | social | manual
    language: str = "unknown"
    category: str = "uncategorized"
    published_date: Optional[str] = None
    raw_html: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    scraped_at: str = field(default_factory=safe_iso_now)

    def to_row(self) -> Dict[str, Any]:
        title = clean_text(self.title)
        content = clean_text(self.content)
        # Dedup key: if a URL is present it is authoritative (the same
        # article re-scraped with a slightly reworded/updated headline
        # must still collapse to one row). Only fall back to a
        # title+content hash for sources with no URL (e.g. manual notes,
        # Google Trends terms).
        if self.url:
            content_hash = compute_content_hash(self.url)
        else:
            content_hash = compute_content_hash(title, content[:300])
        return {
            "title": title,
            "content": content,
            "url": self.url or "",
            "source": self.source,
            "source_type": self.source_type,
            "language": self.language,
            "category": self.category or "uncategorized",
            "published_date": self.published_date,
            "scraped_at": self.scraped_at,
            "raw_html": self.raw_html,
            "extra_json": json.dumps(self.extra, ensure_ascii=False) if self.extra else None,
            "content_hash": content_hash,
            "status": "raw",
        }


class RawNewsDatabase:
    """Thread-safe SQLite wrapper for the raw news store."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        logger.info("Connected to raw news database at %s", self.db_path)

    # -- setup ---------------------------------------------------------
    def _init_schema(self) -> None:
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

    # -- writes ----------------------------------------------------------
    def insert_article(self, article: Article) -> bool:
        """Insert one article. Returns True if inserted, False if it was
        a duplicate (already present) or invalid (e.g. empty title)."""
        if not article.title or not article.title.strip():
            logger.warning("Skipping article with empty title (source=%s)", article.source)
            return False

        row = article.to_row()
        with self._cursor() as cur:
            try:
                cur.execute(
                    """
                    INSERT INTO raw_news
                        (title, content, url, source, source_type, language,
                         category, published_date, scraped_at, raw_html,
                         extra_json, content_hash, status)
                    VALUES
                        (:title, :content, :url, :source, :source_type, :language,
                         :category, :published_date, :scraped_at, :raw_html,
                         :extra_json, :content_hash, :status)
                    """,
                    row,
                )
                return True
            except sqlite3.IntegrityError:
                # duplicate content_hash -> already collected
                return False

    def bulk_insert(self, articles: Iterable[Article]) -> Dict[str, int]:
        """Insert many articles, tolerating per-item failures.

        Returns a stats dict: {"inserted": n, "duplicates": n, "errors": n}
        """
        stats = {"inserted": 0, "duplicates": 0, "errors": 0}
        for article in articles:
            try:
                if self.insert_article(article):
                    stats["inserted"] += 1
                else:
                    stats["duplicates"] += 1
            except Exception as exc:  # keep the batch going no matter what
                stats["errors"] += 1
                logger.error("Failed to insert article %r: %s", getattr(article, "url", ""), exc)
        return stats

    # -- reads -----------------------------------------------------------
    def get_all(self, limit: Optional[int] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
        query = "SELECT * FROM raw_news"
        params: List[Any] = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY scraped_at DESC"
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        with self._cursor() as cur:
            cur.execute(query, params)
            return [dict(r) for r in cur.fetchall()]

    def update_status(self, article_id: int, status: str) -> None:
        """Update just the `status` column for one row.

        This is the intended handoff mechanism described in the README's
        "What to wire up next (Section 2)" section: Section 2 stages read
        `status="raw"` rows and advance them (e.g. to "preprocessed") so
        this layer and the next never reprocess the same row twice. Only
        `status` is touched -- every other column (the raw scraped data)
        is left exactly as collected.
        """
        with self._cursor() as cur:
            cur.execute("UPDATE raw_news SET status = ? WHERE id = ?", (status, article_id))

    def get_by_source_type(self, source_type: str) -> List[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM raw_news WHERE source_type = ? ORDER BY scraped_at DESC", (source_type,))
            return [dict(r) for r in cur.fetchall()]

    def count(self) -> int:
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) AS c FROM raw_news")
            return cur.fetchone()["c"]

    def get_stats(self) -> Dict[str, Any]:
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) AS c FROM raw_news")
            total = cur.fetchone()["c"]

            cur.execute("SELECT source_type, COUNT(*) AS c FROM raw_news GROUP BY source_type")
            by_type = {r["source_type"]: r["c"] for r in cur.fetchall()}

            cur.execute("SELECT language, COUNT(*) AS c FROM raw_news GROUP BY language")
            by_language = {r["language"]: r["c"] for r in cur.fetchall()}

            cur.execute("SELECT status, COUNT(*) AS c FROM raw_news GROUP BY status")
            by_status = {r["status"]: r["c"] for r in cur.fetchall()}

        return {
            "total_articles": total,
            "by_source_type": by_type,
            "by_language": by_language,
            "by_status": by_status,
        }

    # -- export ------------------------------------------------------------
    def export_to_json(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.get_all()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        logger.info("Exported %d articles to %s", len(rows), path)
        return path

    def close(self) -> None:
        with self._lock:
            self._conn.close()
        logger.info("Database connection closed (%s)", self.db_path)

    def __enter__(self) -> "RawNewsDatabase":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
