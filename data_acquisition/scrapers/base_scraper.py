"""
scrapers/base_scraper.py
=========================
Abstract base every source-specific collector implements. Standardises:

- a `name` / `source_type` identity for logging + DB tagging
- a single `fetch() -> List[Article]` entry point the orchestrator calls
- a `_build_article(...)` helper so every scraper produces the exact same
  shape of record, with language auto-detected when not supplied.

Keeping `fetch()` as the only required method means the orchestrator can
treat websites, RSS, APIs, trends, social media and manual files 100%
uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..database.raw_news_db import Article
from ..utils.logger import get_logger
from ..utils.text_utils import detect_language


class BaseScraper(ABC):
    #: short machine name, e.g. "rss", "website", "manual" -- must match
    #: one of the source_type values used across the pipeline.
    source_type: str = "unknown"

    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(f"{self.__class__.__name__}[{name}]")

    @abstractmethod
    def fetch(self) -> List[Article]:
        """Collect articles from this source. Must never raise for a
        single bad item -- log and skip it, and only let genuinely fatal
        errors (e.g. missing config) propagate."""
        raise NotImplementedError

    def _build_article(
        self,
        title: str,
        content: str = "",
        url: str = "",
        category: str = "uncategorized",
        published_date: Optional[str] = None,
        language: Optional[str] = None,
        raw_html: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Article:
        lang = language or detect_language(f"{title} {content}")
        return Article(
            title=title,
            content=content,
            url=url,
            source=self.name,
            source_type=self.source_type,
            language=lang,
            category=category,
            published_date=published_date,
            raw_html=raw_html,
            extra=extra or {},
        )
