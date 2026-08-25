"""
preprocessing/preprocessor.py
================================
Stage 2.1 -- Data Preprocessing.

Reads rows with status="raw" from the existing `RawNewsDatabase`
(`data_acquisition/database/raw_news_db.py`), runs them through:
  HTML cleaning -> ad removal -> exact-duplicate check -> language
  detection -> text normalization -> stopword removal (features only)
  -> sentence splitting,
and writes one `PreprocessedArticle` record per input row.

The raw table/rows are never modified except for the `status` column
(raw -> preprocessed / duplicate_raw / preprocess_error), per the
"preserve the original acquired data" requirement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from data_acquisition.utils.logger import get_logger
from data_acquisition.utils.text_utils import compute_content_hash, detect_language, safe_iso_now

from .html_cleaner import remove_ads, strip_html
from .normalizer import normalize_text, remove_stopwords, split_sentences

logger = get_logger("pipeline.preprocessing")


@dataclass
class PreprocessedArticle:
    article_id: int
    title: str
    source: str
    url: str
    published_date: Optional[str]
    language: str
    cleaned_text: str      # HTML/ad-stripped, normalized -- kept for summarization
    processed_text: str    # cleaned_text with stopwords removed -- NLP features only
    sentences: List[str] = field(default_factory=list)
    preprocessing_status: str = "ok"   # ok | duplicate | error | empty_content
    error_message: Optional[str] = None


class Preprocessor:
    """Runs Stage 2.1 over raw articles.

    `min_content_chars` guards the "very short articles" edge case: below
    this length there isn't enough text to usefully classify/summarize,
    but the article is still kept (status="empty_content") rather than
    silently dropped -- later stages decide what to do with it.
    """

    def __init__(self, min_content_chars: int = 5):
        self.min_content_chars = min_content_chars
        self._seen_hashes: Dict[str, int] = {}   # exact-dup hash -> article_id kept

    def process_article(self, raw_row: Dict[str, Any]) -> PreprocessedArticle:
        article_id = raw_row["id"]
        title_raw = raw_row.get("title") or ""
        content_raw = raw_row.get("content") or raw_row.get("raw_html") or ""

        try:
            # 1. Clean HTML (title is virtually never HTML, but scraped
            #    content sometimes is -- safe either way).
            title = strip_html(title_raw).strip()
            content = strip_html(content_raw)

            # 2. Remove ads/boilerplate.
            content = remove_ads(content)

            # 5. Text normalization (also applied to the title).
            title = normalize_text(title)
            cleaned_text = normalize_text(content)

            if not title and not cleaned_text:
                return PreprocessedArticle(
                    article_id=article_id, title=title, source=raw_row.get("source", ""),
                    url=raw_row.get("url", ""), published_date=raw_row.get("published_date"),
                    language="unknown", cleaned_text="", processed_text="", sentences=[],
                    preprocessing_status="empty_content",
                    error_message="Both title and content were empty after cleaning.",
                )

            # 3. Exact-duplicate check (content-level, post-cleaning --
            #    catches re-scrapes with cosmetic HTML differences that
            #    the raw layer's URL-based hash wouldn't).
            dup_hash = compute_content_hash(title.lower(), cleaned_text[:500].lower())
            if dup_hash in self._seen_hashes:
                return PreprocessedArticle(
                    article_id=article_id, title=title, source=raw_row.get("source", ""),
                    url=raw_row.get("url", ""), published_date=raw_row.get("published_date"),
                    language=raw_row.get("language") or "unknown",
                    cleaned_text=cleaned_text, processed_text="", sentences=[],
                    preprocessing_status="duplicate",
                    error_message=f"Exact duplicate of article_id={self._seen_hashes[dup_hash]}",
                )
            self._seen_hashes[dup_hash] = article_id

            # 4. Language detection -- trust an explicit source-config
            #    language if the raw row already has one and it isn't the
            #    'unknown' default; otherwise detect from content.
            language = raw_row.get("language") or "unknown"
            if language in (None, "", "unknown"):
                language = detect_language(cleaned_text or title)

            # 6. Sentence splitting (on the untouched cleaned text).
            sentences = split_sentences(cleaned_text)

            # 7. Stopword removal -- ONLY for the NLP-feature copy.
            processed_text = remove_stopwords(cleaned_text, language)

            status = "ok" if len(cleaned_text) >= self.min_content_chars else "empty_content"

            return PreprocessedArticle(
                article_id=article_id,
                title=title,
                source=raw_row.get("source", ""),
                url=raw_row.get("url", ""),
                published_date=raw_row.get("published_date"),
                language=language,
                cleaned_text=cleaned_text,
                processed_text=processed_text,
                sentences=sentences,
                preprocessing_status=status,
            )
        except Exception as exc:  # never let one bad article stop the batch
            logger.error("Preprocessing failed for article_id=%s: %s", article_id, exc)
            return PreprocessedArticle(
                article_id=article_id,
                title=title_raw,
                source=raw_row.get("source", ""),
                url=raw_row.get("url", ""),
                published_date=raw_row.get("published_date"),
                language="unknown",
                cleaned_text="",
                processed_text="",
                sentences=[],
                preprocessing_status="error",
                error_message=str(exc),
            )

    def run(self, raw_rows: List[Dict[str, Any]]) -> List[PreprocessedArticle]:
        results = []
        for row in raw_rows:
            results.append(self.process_article(row))
        logger.info(
            "Preprocessing complete: %d ok, %d duplicate, %d empty, %d error (of %d)",
            sum(1 for r in results if r.preprocessing_status == "ok"),
            sum(1 for r in results if r.preprocessing_status == "duplicate"),
            sum(1 for r in results if r.preprocessing_status == "empty_content"),
            sum(1 for r in results if r.preprocessing_status == "error"),
            len(results),
        )
        return results
