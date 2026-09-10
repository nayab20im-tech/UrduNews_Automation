"""
storage/media_storage.py
========================
Implements Section 6's "Media Storage (Audio, Video, Images)" box.

Sections 2.11-3.6 already write their media files under
``DATA_DIR/media/`` (see ``pipeline/config.py`` and
``video_production/config.py``); this module adds the missing management
layer on top: a SQLite registry of every media asset, directory
discovery (``scan()``), per-kind usage stats and cleanup of stale
registry rows -- so the orchestration layer always knows what media
exists for which article without walking the filesystem.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from data_acquisition.utils.logger import get_logger
from data_acquisition.utils.text_utils import safe_iso_now

from . import config as sc

logger = get_logger("storage.media_storage")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS media_assets (
    path          TEXT PRIMARY KEY,
    article_id    INTEGER,
    kind          TEXT NOT NULL,
    size_bytes    INTEGER,
    registered_at TEXT,
    updated_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_media_assets_kind ON media_assets(kind);
CREATE INDEX IF NOT EXISTS idx_media_assets_article ON media_assets(article_id);
"""

# kind -> directory relative to MEDIA_ROOT.  Longest prefix wins when
# inferring a file's kind (e.g. video/thumbnails > video).
KIND_DIRS = {
    "audio": "audio",
    "audio_processed": "audio_processed",
    "images": "images",
    "thumbnails": "video/thumbnails",
    "subtitles": "video/subtitles",
    "video": "video",
    "distribution": "distribution",
}


class MediaStorage:
    """Registry + directory manager for every media file the pipeline writes."""

    def __init__(self, root: Path | str = sc.MEDIA_ROOT, db_path: Path | str = sc.STORAGE_DB_PATH):
        self.root = Path(root)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)
        logger.info("Media storage rooted at %s", self.root)

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

    # -- directories -------------------------------------------------------
    def dir_for(self, kind: str) -> Path:
        return self.root / KIND_DIRS.get(kind, kind)

    def ensure_dirs(self) -> None:
        for kind in KIND_DIRS:
            self.dir_for(kind).mkdir(parents=True, exist_ok=True)

    def infer_kind(self, path: Path) -> str:
        """Most-specific directory prefix wins (video/thumbnails > video)."""
        try:
            rel = path.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return "other"
        best_kind, best_len = "other", -1
        for kind, sub in KIND_DIRS.items():
            if (rel == sub or rel.startswith(sub + "/")) and len(sub) > best_len:
                best_kind, best_len = kind, len(sub)
        return best_kind

    # -- registry writes ------------------------------------------------------
    def register(
        self,
        path: Path | str,
        kind: Optional[str] = None,
        article_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Register one media file (idempotent). Returns the registry row."""
        path = Path(path)
        kind = kind or self.infer_kind(path)
        size = path.stat().st_size if path.exists() else None
        now = safe_iso_now()
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO media_assets (path, article_id, kind, size_bytes, registered_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    article_id = COALESCE(excluded.article_id, media_assets.article_id),
                    kind = excluded.kind,
                    size_bytes = excluded.size_bytes,
                    updated_at = excluded.updated_at
                """,
                (str(path), article_id, kind, size, now, now),
            )
        return {"path": str(path), "kind": kind, "article_id": article_id, "size_bytes": size}

    def scan(self, kind: Optional[str] = None) -> int:
        """Walk the media tree and register every file found.

        Returns the number of files registered. With ``kind`` only that
        sub-directory is walked.
        """
        roots = [self.dir_for(kind)] if kind else [self.root]
        count = 0
        for root in roots:
            if not root.exists():
                continue
            for file in sorted(root.rglob("*")):
                if file.is_file():
                    self.register(file)
                    count += 1
        logger.info("Media scan registered %d files under %s", count, roots)
        return count

    # -- registry reads ---------------------------------------------------------
    def list_assets(
        self,
        kind: Optional[str] = None,
        article_id: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM media_assets"
        clauses: List[str] = []
        params: List[Any] = []
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        if article_id is not None:
            clauses.append("article_id = ?")
            params.append(article_id)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY path"
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        with self._cursor() as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    def stats(self) -> Dict[str, Any]:
        """Per-kind file counts and byte totals from the registry."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT kind, COUNT(*) AS files, COALESCE(SUM(size_bytes), 0) AS bytes "
                "FROM media_assets GROUP BY kind ORDER BY kind"
            )
            by_kind = {r["kind"]: {"files": r["files"], "bytes": r["bytes"]} for r in cur.fetchall()}
            cur.execute("SELECT COUNT(*) AS c FROM media_assets")
            total = cur.fetchone()["c"]
        return {"total_assets": total, "by_kind": by_kind, "root": str(self.root)}

    # -- maintenance ---------------------------------------------------------
    def cleanup_missing(self) -> int:
        """Drop registry rows whose file no longer exists; returns count."""
        rows = self.list_assets()
        missing = [r["path"] for r in rows if not Path(r["path"]).exists()]
        if missing:
            with self._cursor() as cur:
                cur.executemany(
                    "DELETE FROM media_assets WHERE path = ?",
                    [(p,) for p in missing],
                )
            logger.info("Removed %d stale media registry rows", len(missing))
        return len(missing)

    def close(self) -> None:
        with self._lock:
            self._conn.close()
        logger.info("Media storage connection closed (%s)", self.db_path)

    def __enter__(self) -> "MediaStorage":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
