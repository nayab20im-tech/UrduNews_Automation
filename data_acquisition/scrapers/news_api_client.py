"""
scrapers/news_api_client.py
=============================
Implements the "News APIs (Public / Open)" source, via NewsAPI.org's
REST API (https://newsapi.org/v2). Any similar JSON news API (GNews,
Mediastack, etc.) can be plugged in by adding another thin client with
the same `fetch() -> List[Article]` contract.

Requires a free/paid API key set as NEWSAPI_KEY in `.env`. If no key is
configured, `fetch()` logs a clear warning and returns an empty list
instead of raising, so a missing key never breaks the rest of the
pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests

from .base_scraper import BaseScraper
from ..database.raw_news_db import Article
from ..utils.text_utils import clean_text
from .. import config as app_config

_BASE_URL = "https://newsapi.org/v2"


class NewsAPIClient(BaseScraper):
    source_type = "api"

    def __init__(
        self,
        api_key: Optional[str] = None,
        queries: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        language: str = "en",
        page_size: int = 50,
    ):
        super().__init__(name="newsapi_org")
        self.api_key = api_key or app_config.NEWSAPI_KEY
        self.queries = queries or []
        self.categories = categories or ["general"]
        self.language = language
        self.page_size = min(page_size, 100)  # NewsAPI hard limit

    def fetch(self) -> List[Article]:
        if not self.api_key:
            self.logger.warning(
                "NEWSAPI_KEY is not set (see .env.example); skipping News API collection."
            )
            return []

        articles: List[Article] = []

        for category in self.categories:
            articles.extend(self._fetch_top_headlines(category=category))

        for query in self.queries:
            articles.extend(self._fetch_everything(query=query))

        self.logger.info("Fetched %d articles from NewsAPI.org", len(articles))
        return articles

    # -- endpoint calls, each isolated so one failure doesn't kill the rest --
    def _fetch_top_headlines(self, category: str) -> List[Article]:
        params = {
            "apiKey": self.api_key,
            "category": category,
            "language": self.language,
            "pageSize": self.page_size,
        }
        return self._call("/top-headlines", params, extra={"endpoint": "top-headlines", "category": category})

    def _fetch_everything(self, query: str) -> List[Article]:
        params = {
            "apiKey": self.api_key,
            "q": query,
            "language": self.language,
            "pageSize": self.page_size,
            "sortBy": "publishedAt",
        }
        return self._call("/everything", params, extra={"endpoint": "everything", "query": query})

    def _call(self, path: str, params: Dict[str, Any], extra: Dict[str, Any]) -> List[Article]:
        try:
            response = requests.get(f"{_BASE_URL}{path}", params=params, timeout=app_config.REQUEST_TIMEOUT)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            self.logger.error("NewsAPI request failed for %s (%s): %s", path, extra, exc)
            return []
        except ValueError as exc:
            self.logger.error("NewsAPI returned invalid JSON for %s: %s", path, exc)
            return []

        if payload.get("status") != "ok":
            self.logger.error("NewsAPI error for %s: %s", path, payload.get("message"))
            return []

        return [self._to_article(item, extra) for item in payload.get("articles", []) if item]

    def _to_article(self, item: Dict[str, Any], extra: Dict[str, Any]) -> Article:
        title = clean_text(item.get("title") or "")
        content = clean_text(item.get("description") or item.get("content") or "")
        source_name = (item.get("source") or {}).get("name") or "NewsAPI"
        return self._build_article(
            title=title,
            content=content,
            url=item.get("url") or "",
            category=extra.get("category", "general"),
            published_date=item.get("publishedAt"),
            language=self.language,
            extra={**extra, "source_name": source_name, "author": item.get("author")},
        )
