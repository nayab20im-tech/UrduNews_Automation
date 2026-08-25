"""
scrapers/rss_scraper.py
=========================
Implements the "RSS Feeds" news source. Uses `feedparser`, which is
resilient to the malformed/half-standard XML many news outlets ship.

Design note: `_parse_feed()` accepts either a live URL *or* raw feed
content/a local file path (feedparser supports all three transparently).
That keeps this class unit-testable offline with a fixture file, while
`fetch()` -- the method the orchestrator calls -- always hits real URLs
from config.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import feedparser
import requests

from .base_scraper import BaseScraper
from ..database.raw_news_db import Article
from ..utils.text_utils import clean_text
from .. import config as app_config


class RSSScraper(BaseScraper):
    source_type = "rss"

    def __init__(self, feeds: List[Dict[str, Any]]):
        """
        feeds: list of dicts, each like
            {"name": "BBC Urdu", "url": "https://...", "language": "ur",
             "category": "general"}
        (as loaded from data/sources_config.yaml -> rss_feeds)
        """
        super().__init__(name="rss_feeds")
        self.feeds = feeds or []

    def fetch(self) -> List[Article]:
        articles: List[Article] = []
        if not self.feeds:
            self.logger.info("No RSS feeds configured; skipping.")
            return articles

        for feed_cfg in self.feeds:
            feed_name = feed_cfg.get("name", "unknown_feed")
            feed_url = feed_cfg.get("url")
            if not feed_url:
                self.logger.warning("RSS feed entry %r has no url, skipping.", feed_name)
                continue
            try:
                items = self._parse_feed(
                    source=feed_url,
                    feed_name=feed_name,
                    default_language=feed_cfg.get("language"),
                    default_category=feed_cfg.get("category", "general"),
                )
                self.logger.info("Fetched %d items from RSS feed '%s'", len(items), feed_name)
                articles.extend(items)
            except Exception as exc:
                self.logger.error("Failed to fetch RSS feed '%s' (%s): %s", feed_name, feed_url, exc)
                continue

        return articles

    # -- testable core logic -------------------------------------------------
    def _parse_feed(
        self,
        source: str,
        feed_name: str,
        default_language: Optional[str] = None,
        default_category: str = "general",
    ) -> List[Article]:
        """Parse a feed from a URL, local path, or raw XML/bytes string.

        This separation from `fetch()` is what makes RSS parsing testable
        without any network access -- pass a local fixture file's content.
        """
        # feedparser.parse(url) does its own HTTP fetch under the hood with
        # a generic "python-feedparser/x.y" User-Agent when given a URL.
        # Some publishers (Dawn included) reject that with a 403, which
        # feedparser then reports as an opaque XML "syntax error" instead
        # of the real cause. Fetch with a normal browser User-Agent first
        # and hand feedparser the bytes directly -- this also reuses the
        # feed-content code path (feedparser.parse() on a string/bytes),
        # which is already exercised by the offline fixture tests.
        if source.startswith("http://") or source.startswith("https://"):
            try:
                response = requests.get(
                    source,
                    headers={"User-Agent": random.choice(app_config.USER_AGENTS)},
                    timeout=app_config.REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                parsed = feedparser.parse(response.content)
            except requests.RequestException as exc:
                raise ValueError(f"Could not fetch feed: {exc}") from exc
        else:
            parsed = feedparser.parse(source)

        if parsed.bozo and not parsed.entries:
            raise ValueError(f"Unparseable feed ({parsed.get('bozo_exception')})")

        articles: List[Article] = []
        for entry in parsed.entries:
            title = clean_text(getattr(entry, "title", ""))
            if not title:
                continue

            content = clean_text(
                getattr(entry, "summary", "")
                or getattr(entry, "description", "")
                or (entry.get("content", [{}])[0].get("value", "") if entry.get("content") else "")
            )
            link = getattr(entry, "link", "") or ""
            published = getattr(entry, "published", None) or getattr(entry, "updated", None)

            categories = [t.get("term") for t in getattr(entry, "tags", []) if t.get("term")]
            category = categories[0] if categories else default_category

            articles.append(
                self._build_article(
                    title=title,
                    content=content,
                    url=link,
                    category=category,
                    published_date=published,
                    language=default_language,
                    extra={"feed_name": feed_name, "raw_categories": categories},
                )
            )
        return articles
