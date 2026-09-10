"""
orchestrator/data_acquisition_orchestrator.py
================================================
Ties Section 1 together: reads `sources_config.yaml`, instantiates every
enabled scraper, runs each one, and stores the combined results in the
Raw News Database. This is the local, Section-1-scoped counterpart to
the diagram's "5. Orchestration & Pipeline Control" box -- it plugs into
that layer as one step, but is fully usable standalone.

Design goals:
- one failing source (bad selectors, expired API key, network outage)
  never stops the others from running.
- every run returns a structured report (per-source counts + errors) for
  the "4. Notification System" / "7. Monitoring & Dashboard" boxes to
  consume later.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import config as app_config
from ..database.raw_news_db import Article, RawNewsDatabase
from ..engine.scraping_engine import ScrapingEngine
from ..scrapers.base_scraper import BaseScraper
from ..scrapers.manual_input_loader import ManualInputLoader
from ..scrapers.news_api_client import NewsAPIClient
from ..scrapers.news_website_scraper import NewsWebsiteScraper
from ..scrapers.rss_scraper import RSSScraper
from ..scrapers.social_media_scraper import SocialMediaScraper
from ..scrapers.trends_scraper import GoogleTrendsScraper
from ..utils.logger import get_logger
from ..utils.text_utils import safe_iso_now

logger = get_logger(__name__)


@dataclass
class SourceRunResult:
    source_name: str
    source_type: str
    fetched: int = 0
    inserted: int = 0
    duplicates: int = 0
    errors: int = 0
    duration_seconds: float = 0.0
    error_message: Optional[str] = None


@dataclass
class AcquisitionReport:
    started_at: str
    finished_at: str = ""
    results: List[SourceRunResult] = field(default_factory=list)

    @property
    def total_fetched(self) -> int:
        return sum(r.fetched for r in self.results)

    @property
    def total_inserted(self) -> int:
        return sum(r.inserted for r in self.results)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_fetched": self.total_fetched,
            "total_inserted": self.total_inserted,
            "sources": [r.__dict__ for r in self.results],
        }


class DataAcquisitionOrchestrator:
    def __init__(self, sources_config: Dict[str, Any], db: RawNewsDatabase):
        self.sources_config = sources_config
        self.db = db
        self.engine = ScrapingEngine()
        self.scrapers: List[BaseScraper] = self._build_scrapers()

    # -- construction ------------------------------------------------------
    def _build_scrapers(self) -> List[BaseScraper]:
        scrapers: List[BaseScraper] = []
        cfg = self.sources_config

        rss_feeds = cfg.get("rss_feeds", [])
        if rss_feeds:
            scrapers.append(RSSScraper(feeds=rss_feeds))

        for site_cfg in cfg.get("websites", []):
            if site_cfg.get("enabled", True):
                scrapers.append(NewsWebsiteScraper(site_config=site_cfg, engine=self.engine))

        api_cfg = cfg.get("news_api", {})
        if api_cfg.get("enabled"):
            scrapers.append(
                NewsAPIClient(
                    queries=api_cfg.get("queries", []),
                    categories=api_cfg.get("categories", ["general"]),
                    language=api_cfg.get("language", "en"),
                    page_size=api_cfg.get("page_size", 50),
                )
            )

        trends_cfg = cfg.get("google_trends", {})
        if trends_cfg.get("enabled"):
            scrapers.append(
                GoogleTrendsScraper(
                    country=trends_cfg.get("country", "PK"),
                    language=trends_cfg.get("language", "ur"),
                )
            )

        social_cfg = cfg.get("social_media", {})
        if social_cfg.get("enabled"):
            scrapers.append(
                SocialMediaScraper(
                    queries=social_cfg.get("queries", []),
                    platform=social_cfg.get("platform", "twitter"),
                    limit_per_query=social_cfg.get("limit", 50),
                )
            )

        manual_cfg = cfg.get("manual_input", {"enabled": True})
        if manual_cfg.get("enabled", True):
            folder = Path(manual_cfg.get("folder") or app_config.MANUAL_INPUT_DIR)
            # A relative path in the YAML (e.g. "data/manual_input") is
            # relative to the package root, NOT to whatever directory the
            # process happens to be launched from -- resolve it here so
            # `python -m data_acquisition.main` behaves the same no matter
            # where it's invoked from.
            if not folder.is_absolute():
                folder = app_config.BASE_DIR / folder
            scrapers.append(ManualInputLoader(folder=folder))

        logger.info("Built %d scraper(s): %s", len(scrapers), [s.name for s in scrapers])
        return scrapers

    # -- run -----------------------------------------------------------------
    def run_all(self) -> AcquisitionReport:
        report = AcquisitionReport(started_at=safe_iso_now())
        for scraper in self.scrapers:
            report.results.append(self._run_one(scraper))
        report.finished_at = safe_iso_now()
        logger.info(
            "Data acquisition run complete: %d fetched, %d newly inserted",
            report.total_fetched,
            report.total_inserted,
        )
        return report

    def run_single(self, source_name: str) -> Optional[SourceRunResult]:
        for scraper in self.scrapers:
            if scraper.name == source_name:
                return self._run_one(scraper)
        logger.warning("No configured scraper named %r", source_name)
        return None

    def _run_one(self, scraper: BaseScraper) -> SourceRunResult:
        start = time.monotonic()
        result = SourceRunResult(source_name=scraper.name, source_type=scraper.source_type)
        try:
            articles: List[Article] = scraper.fetch()
            result.fetched = len(articles)
            stats = self.db.bulk_insert(articles)
            result.inserted = stats["inserted"]
            result.duplicates = stats["duplicates"]
            result.errors = stats["errors"]
        except Exception as exc:
            result.error_message = str(exc)
            logger.exception("Source %r failed entirely: %s", scraper.name, exc)
        finally:
            result.duration_seconds = round(time.monotonic() - start, 2)
        logger.info(
            "[%s] fetched=%d inserted=%d duplicates=%d errors=%d (%.2fs)",
            scraper.name, result.fetched, result.inserted, result.duplicates,
            result.errors, result.duration_seconds,
        )
        return result

    def close(self) -> None:
        self.engine.close()
