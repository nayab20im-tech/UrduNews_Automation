"""
scrapers/social_media_scraper.py
==================================
Implements the "Social Media (Twitter, etc.)" source.

Two collection paths are supported:

1. `snscrape` (default) -- scrapes public tweets without needing API
   keys or hitting Twitter/X's paid API. Good fit for "fully offline
   afterwards / open-source" style pipelines.
2. Official API (fallback) -- if `TWITTER_BEARER_TOKEN` is set in `.env`
   and `tweepy` is installed, that's used instead (more reliable, but
   requires paid API access at meaningful volume).

Either path can be unavailable in a given environment (no internet,
missing package, no key) -- in all cases `fetch()` logs a clear reason
and returns an empty list rather than crashing the orchestrator.
"""

from __future__ import annotations

from typing import List

from .base_scraper import BaseScraper
from ..database.raw_news_db import Article
from ..utils.text_utils import clean_text, safe_iso_now
from .. import config as app_config


class SocialMediaScraper(BaseScraper):
    source_type = "social"

    def __init__(self, queries: List[str], platform: str = "twitter", limit_per_query: int = 50):
        super().__init__(name=f"social_{platform}")
        self.queries = queries or []
        self.platform = platform
        self.limit_per_query = limit_per_query or app_config.MAX_TWEETS_PER_QUERY

    def fetch(self) -> List[Article]:
        if self.platform != "twitter":
            self.logger.warning("Platform %r is not implemented yet, skipping.", self.platform)
            return []
        if not self.queries:
            self.logger.info("No social media queries configured; skipping.")
            return []

        articles = self._fetch_with_snscrape()
        if not articles and app_config.TWITTER_BEARER_TOKEN:
            articles = self._fetch_with_official_api()
        return articles

    # -- path 1: snscrape (no API key) --------------------------------------
    def _fetch_with_snscrape(self) -> List[Article]:
        try:
            import snscrape.modules.twitter as sntwitter
        except ImportError:
            self.logger.warning(
                "snscrape is not installed (`pip install snscrape`); "
                "falling back to official API if TWITTER_BEARER_TOKEN is set."
            )
            return []

        articles: List[Article] = []
        for query in self.queries:
            try:
                scraper = sntwitter.TwitterSearchScraper(query)
                for i, tweet in enumerate(scraper.get_items()):
                    if i >= self.limit_per_query:
                        break
                    articles.append(
                        self._build_article(
                            title=clean_text(tweet.rawContent[:120]),
                            content=clean_text(tweet.rawContent),
                            url=tweet.url,
                            category="social",
                            published_date=tweet.date.isoformat() if tweet.date else None,
                            extra={
                                "query": query,
                                "username": getattr(tweet.user, "username", None),
                                "like_count": getattr(tweet, "likeCount", None),
                                "retweet_count": getattr(tweet, "retweetCount", None),
                            },
                        )
                    )
            except Exception as exc:
                self.logger.error("snscrape failed for query %r: %s", query, exc)
                continue

        self.logger.info("Fetched %d posts via snscrape", len(articles))
        return articles

    # -- path 2: official Twitter API v2 (needs TWITTER_BEARER_TOKEN) -------
    def _fetch_with_official_api(self) -> List[Article]:
        try:
            import tweepy
        except ImportError:
            self.logger.error("tweepy is not installed (`pip install tweepy`); cannot use official API fallback.")
            return []

        try:
            client = tweepy.Client(bearer_token=app_config.TWITTER_BEARER_TOKEN)
        except Exception as exc:
            self.logger.error("Failed to initialise Twitter API client: %s", exc)
            return []

        articles: List[Article] = []
        for query in self.queries:
            try:
                response = client.search_recent_tweets(
                    query=query,
                    max_results=min(self.limit_per_query, 100),
                    tweet_fields=["created_at", "public_metrics", "author_id"],
                )
                for tweet in response.data or []:
                    articles.append(
                        self._build_article(
                            title=clean_text(tweet.text[:120]),
                            content=clean_text(tweet.text),
                            url=f"https://twitter.com/i/web/status/{tweet.id}",
                            category="social",
                            published_date=str(tweet.created_at) if tweet.created_at else safe_iso_now(),
                            extra={"query": query, "metrics": tweet.public_metrics},
                        )
                    )
            except Exception as exc:
                self.logger.error("Official API search failed for query %r: %s", query, exc)
                continue

        self.logger.info("Fetched %d posts via official Twitter API", len(articles))
        return articles
