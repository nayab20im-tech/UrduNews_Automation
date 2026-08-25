"""
scrapers/trends_scraper.py
============================
Implements the "Trending Topics (Google Trends)" source.

NOTE on the fix applied here (see README "Known issues -> Google Trends"):
this used to call `pytrends`, an unofficial Google Trends wrapper. Google
retired/changed the internal endpoints pytrends scraped, and the pytrends
project itself was archived (read-only) by its maintainers on 2025-04-17.
In practice it now fails outright -- typically with a "Google returned a
response with code 404" error, exactly what showed up in this project's
logs.

The fix: use `trendspyg` instead, an actively maintained pytrends
replacement. Its RSS-backed path (`download_google_trends_rss`) reads
Google's own public "Daily Search Trends" RSS feed
(https://trends.google.com/trending/rss?geo=<COUNTRY>) rather than
Google's internal, frequently-changing widget API. That's the same feed
Google exposes for people to subscribe to trending searches in a feed
reader, so it's much less likely to break, needs no browser/Selenium, and
needs no API key.

Trending searches don't have article bodies, so each is stored as a
lightweight Article whose `content` is empty and whose `extra` carries
rank/volume -- downstream (Section 2) this feeds topic selection rather
than being summarised directly.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .base_scraper import BaseScraper
from ..database.raw_news_db import Article
from ..utils.text_utils import safe_iso_now


class GoogleTrendsScraper(BaseScraper):
    source_type = "trends"

    def __init__(self, country: str = "PK", language: str = "ur"):
        """
        country: ISO-3166-1 alpha-2 code (e.g. "PK", "US", "GB") -- this is
            what trendspyg's RSS path expects. (The old pytrends-based
            version of this scraper took a long-form name like "pakistan";
            that's no longer valid input.)
        """
        super().__init__(name="google_trends")
        self.country = country
        self.language = language

    def fetch(self) -> List[Article]:
        try:
            from trendspyg import download_google_trends_rss
        except ImportError:
            self.logger.error(
                "trendspyg is not installed. Run `pip install trendspyg` "
                "(this replaced the unmaintained/archived `pytrends`)."
            )
            return []

        try:
            result = download_google_trends_rss(geo=self.country, normalize=True)
        except Exception as exc:
            self.logger.error("Failed to fetch Google Trends for %s: %s", self.country, exc)
            return []

        trend_entries: List[Dict[str, Any]] = (
            result.get("trends", []) if isinstance(result, dict) else list(result or [])
        )

        articles: List[Article] = []
        for entry in trend_entries:
            title = str(entry.get("keyword") or entry.get("trend") or "").strip()
            if not title:
                continue
            articles.append(
                self._build_article(
                    title=title,
                    content="",
                    url=entry.get("explore_url") or "",
                    category="trending",
                    published_date=safe_iso_now(),
                    language=self.language,
                    extra={
                        "rank": entry.get("rank"),
                        "country": self.country,
                        "volume": entry.get("volume_text") or entry.get("volume_min"),
                    },
                )
            )

        self.logger.info("Fetched %d trending topics for %s", len(articles), self.country)
        return articles
