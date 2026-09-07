"""
pipeline/processed_news_db.py
================================
Storage for Stages 2.1-2.12 output.

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

CREATE TABLE IF NOT EXISTS translations (
    article_id            INTEGER PRIMARY KEY,
    urdu_title            TEXT,
    urdu_text             TEXT,
    urdu_sentences_json   TEXT,
    source_language       TEXT,
    translated            INTEGER,
    refinement_notes_json TEXT,
    status                TEXT,
    updated_at            TEXT
);

CREATE TABLE IF NOT EXISTS bias_analysis (
    article_id        INTEGER PRIMARY KEY,
    clickbait         INTEGER,
    sensational_score REAL,
    sentiment_score   REAL,
    bias_score        REAL,
    label             TEXT,
    flags_json        TEXT,
    status            TEXT,
    updated_at        TEXT
);

CREATE TABLE IF NOT EXISTS verification (
    article_id         INTEGER PRIMARY KEY,
    source_credibility REAL,
    claim_results_json TEXT,
    evidence_count     INTEGER,
    confidence         REAL,
    verdict            TEXT,
    status             TEXT,
    updated_at         TEXT
);

CREATE TABLE IF NOT EXISTS news_scripts (
    article_id      INTEGER PRIMARY KEY,
    script_text     TEXT,
    word_count      INTEGER,
    backend_used    TEXT,
    verification_note TEXT,
    status          TEXT,
    updated_at      TEXT
);

CREATE TABLE IF NOT EXISTS structured_scripts (
    article_id      INTEGER PRIMARY KEY,
    headline        TEXT,
    intro           TEXT,
    main_story      TEXT,
    key_points_json TEXT,
    ending          TEXT,
    full_script     TEXT,
    status          TEXT,
    updated_at      TEXT
);

CREATE TABLE IF NOT EXISTS tts_audio (
    article_id   INTEGER PRIMARY KEY,
    audio_path   TEXT,
    duration_sec REAL,
    sample_rate  INTEGER,
    backend_used TEXT,
    voice        TEXT,
    status       TEXT,
    updated_at   TEXT
);

CREATE TABLE IF NOT EXISTS processed_audio (
    article_id          INTEGER PRIMARY KEY,
    source_path         TEXT,
    audio_path          TEXT,
    duration_before_sec REAL,
    duration_after_sec  REAL,
    peak_before         REAL,
    peak_after          REAL,
    silence_removed_ms  INTEGER,
    bgm_applied         INTEGER,
    status              TEXT,
    updated_at          TEXT
);

CREATE TABLE IF NOT EXISTS media_relevance (
    article_id          INTEGER PRIMARY KEY,
    image_url           TEXT,
    video_url           TEXT,
    relevance_score     REAL,
    status              TEXT,
    updated_at          TEXT
);
"""


class ProcessedNewsDatabase:
    """Thread-safe SQLite wrapper for all Stage 2.1-2.12 output tables."""

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

    def save_translation(self, rec, updated_at: str) -> None:
        self._upsert(
            "translations",
            {
                "article_id": rec.article_id,
                "urdu_title": rec.urdu_title,
                "urdu_text": rec.urdu_text,
                "urdu_sentences_json": json.dumps(rec.urdu_sentences, ensure_ascii=False),
                "source_language": rec.source_language,
                "translated": int(rec.translated),
                "refinement_notes_json": json.dumps(rec.refinement_notes, ensure_ascii=False),
                "status": rec.status,
                "updated_at": updated_at,
            },
        )

    def save_bias_analysis(self, rec, updated_at: str) -> None:
        self._upsert(
            "bias_analysis",
            {
                "article_id": rec.article_id,
                "clickbait": int(rec.clickbait),
                "sensational_score": rec.sensational_score,
                "sentiment_score": rec.sentiment_score,
                "bias_score": rec.bias_score,
                "label": rec.label,
                "flags_json": json.dumps(rec.flags, ensure_ascii=False),
                "status": rec.status,
                "updated_at": updated_at,
            },
        )

    def save_verification(self, rec, updated_at: str) -> None:
        self._upsert(
            "verification",
            {
                "article_id": rec.article_id,
                "source_credibility": rec.source_credibility,
                "claim_results_json": json.dumps(rec.claim_results, ensure_ascii=False),
                "evidence_count": rec.evidence_count,
                "confidence": rec.confidence,
                "verdict": rec.verdict,
                "status": rec.status,
                "updated_at": updated_at,
            },
        )

    def save_script(self, rec, updated_at: str) -> None:
        self._upsert(
            "news_scripts",
            {
                "article_id": rec.article_id,
                "script_text": rec.script_text,
                "word_count": rec.word_count,
                "backend_used": rec.backend_used,
                "verification_note": rec.verification_note,
                "status": rec.status,
                "updated_at": updated_at,
            },
        )

    def save_structured_script(self, rec, updated_at: str) -> None:
        self._upsert(
            "structured_scripts",
            {
                "article_id": rec.article_id,
                "headline": rec.headline,
                "intro": rec.intro,
                "main_story": rec.main_story,
                "key_points_json": json.dumps(rec.key_points, ensure_ascii=False),
                "ending": rec.ending,
                "full_script": rec.full_script,
                "status": rec.status,
                "updated_at": updated_at,
            },
        )

    def save_tts_audio(self, rec, updated_at: str) -> None:
        self._upsert(
            "tts_audio",
            {
                "article_id": rec.article_id,
                "audio_path": rec.audio_path,
                "duration_sec": rec.duration_sec,
                "sample_rate": rec.sample_rate,
                "backend_used": rec.backend_used,
                "voice": rec.voice,
                "status": rec.status,
                "updated_at": updated_at,
            },
        )

    def save_processed_audio(self, rec, updated_at: str) -> None:
        self._upsert(
            "processed_audio",
            {
                "article_id": rec.article_id,
                "source_path": rec.source_path,
                "audio_path": rec.audio_path,
                "duration_before_sec": rec.duration_before_sec,
                "duration_after_sec": rec.duration_after_sec,
                "peak_before": rec.peak_before,
                "peak_after": rec.peak_after,
                "silence_removed_ms": rec.silence_removed_ms,
                "bgm_applied": int(rec.bgm_applied),
                "status": rec.status,
                "updated_at": updated_at,
            },
        )

    def save_media_relevance(self, rec, updated_at: str) -> None:
        self._upsert(
            "media_relevance",
            {
                "article_id": rec.article_id,
                "image_url": getattr(rec, "image_url", ""),
                "video_url": getattr(rec, "video_url", ""),
                "relevance_score": getattr(rec, "relevance_score", 0.0),
                "status": rec.status,
                "updated_at": updated_at,
            }
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

    def _table_map(self, table: str) -> Dict[int, Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute(f"SELECT * FROM {table}")
            return {r["article_id"]: dict(r) for r in cur.fetchall()}

    def get_extraction_map(self) -> Dict[int, Dict[str, Any]]:
        """article_id -> {"entities", "keypoints", "claims"} (parsed)."""
        out = {}
        for aid, row in self._table_map("extracted_info").items():
            out[aid] = {
                "entities": json.loads(row["entities_json"]) if row["entities_json"] else {},
                "keypoints": json.loads(row["keypoints_json"]) if row["keypoints_json"] else [],
                "claims": json.loads(row["claims_json"]) if row["claims_json"] else [],
            }
        return out

    def get_summary_map(self) -> Dict[int, Dict[str, Any]]:
        """article_id -> {"summary", "key_points", "bullet_summary"} (parsed)."""
        out = {}
        for aid, row in self._table_map("summaries").items():
            out[aid] = {
                "summary": row["summary"] or "",
                "key_points": json.loads(row["key_points_json"]) if row["key_points_json"] else [],
                "bullet_summary": json.loads(row["bullet_summary_json"]) if row["bullet_summary_json"] else [],
            }
        return out

    def get_translation_map(self) -> Dict[int, Dict[str, Any]]:
        out = {}
        for aid, row in self._table_map("translations").items():
            out[aid] = {
                "urdu_title": row["urdu_title"] or "",
                "urdu_text": row["urdu_text"] or "",
                "urdu_sentences": json.loads(row["urdu_sentences_json"]) if row["urdu_sentences_json"] else [],
                "translated": bool(row["translated"]),
            }
        return out

    def get_bias_map(self) -> Dict[int, Dict[str, Any]]:
        return self._table_map("bias_analysis")

    def get_verification_map(self) -> Dict[int, Dict[str, Any]]:
        out = {}
        for aid, row in self._table_map("verification").items():
            out[aid] = {
                "source_credibility": row["source_credibility"],
                "claim_results": json.loads(row["claim_results_json"]) if row["claim_results_json"] else [],
                "evidence_count": row["evidence_count"],
                "confidence": row["confidence"],
                "verdict": row["verdict"],
            }
        return out

    def get_script_map(self) -> Dict[int, Dict[str, Any]]:
        return self._table_map("news_scripts")

    def get_structured_map(self) -> Dict[int, Dict[str, Any]]:
        """article_id -> {"headline", "full_script", ...} from Stage 2.10."""
        out = {}
        for aid, row in self._table_map("structured_scripts").items():
            out[aid] = {
                "headline": row["headline"] or "",
                "full_script": row["full_script"] or "",
                "key_points": json.loads(row["key_points_json"]) if row["key_points_json"] else [],
                "status": row["status"],
            }
        return out

    def get_tts_map(self) -> Dict[int, Dict[str, Any]]:
        return self._table_map("tts_audio")

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
                    s.summary, s.key_points_json, s.bullet_summary_json,
                    t.urdu_title, t.urdu_text, t.urdu_sentences_json, t.translated,
                    b.label AS bias_label, b.bias_score, b.clickbait, b.sentiment_score,
                    v.verdict AS verification_verdict, v.confidence AS verification_confidence,
                    v.source_credibility, v.evidence_count,
                    ns.script_text, ns.word_count AS script_word_count,
                    ss.headline, ss.intro, ss.main_story, ss.ending, ss.full_script,
                    ss.key_points_json AS script_key_points_json,
                    ta.audio_path AS tts_audio_path, ta.duration_sec AS tts_duration_sec,
                    ta.backend_used AS tts_backend,
                    pa.audio_path AS processed_audio_path,
                    pa.duration_after_sec AS audio_duration_sec,
                    pa.peak_after AS audio_peak, pa.bgm_applied AS audio_bgm,
                    mr.image_url, mr.video_url, mr.relevance_score AS media_relevance_score
                FROM preprocessed_news p
                LEFT JOIN classified_news c ON c.article_id = p.article_id
                LEFT JOIN duplicate_status d ON d.article_id = p.article_id
                LEFT JOIN extracted_info e ON e.article_id = p.article_id
                LEFT JOIN summaries s ON s.article_id = p.article_id
                LEFT JOIN translations t ON t.article_id = p.article_id
                LEFT JOIN bias_analysis b ON b.article_id = p.article_id
                LEFT JOIN verification v ON v.article_id = p.article_id
                LEFT JOIN news_scripts ns ON ns.article_id = p.article_id
                LEFT JOIN structured_scripts ss ON ss.article_id = p.article_id
                LEFT JOIN tts_audio ta ON ta.article_id = p.article_id
                LEFT JOIN processed_audio pa ON pa.article_id = p.article_id
                LEFT JOIN media_relevance mr ON mr.article_id = p.article_id
                ORDER BY p.article_id
                """
            )
            rows = [dict(r) for r in cur.fetchall()]

        for row in rows:
            for key in ("entities_json", "keypoints_json", "claims_json", "key_points_json",
                        "bullet_summary_json", "urdu_sentences_json", "script_key_points_json"):
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
