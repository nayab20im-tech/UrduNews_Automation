"""
scrapers/news_website_scraper.py
==================================
Implements the "News Websites (Urdu, English)" source using the shared
Web Scraping Engine (BeautifulSoup, with an automatic Selenium fallback
for JS-rendered pages).

Every site has different HTML, so this class is *config-driven*: each
site entry in `data/sources_config.yaml -> websites` supplies CSS
selectors for the listing page (where to find article links) and for an
article page (title/content/date/category). No selectors are hard-coded
here, which means:
  1. adding a new site never requires touching this file, and
  2. this file can't silently ship a wrong/stale selector for a site
     nobody has verified -- see the README for how to fill in selectors.

Parsing (`_parse_listing`, `_parse_article`) is deliberately separated
from network I/O (`fetch`) so both can be unit-tested against local HTML
fixtures without hitting the internet.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from .base_scraper import BaseScraper
from ..database.raw_news_db import Article
from ..engine.scraping_engine import ScrapingEngine
from ..utils.text_utils import clean_text
from .. import config as app_config


class NewsWebsiteScraper(BaseScraper):
    source_type = "website"

    def __init__(self, site_config: Dict[str, Any], engine: Optional[ScrapingEngine] = None):
        """
        site_config example (see data/sources_config.yaml):
        {
          "name": "Example News",
          "listing_url": "https://example-news.com/latest",
          "language": "ur",
          "category": "general",
          "max_articles": 20,
          "use_selenium": false,
          "selectors": {
            "article_links": "a.story-link",   # CSS selector on listing page
            "title": "h1.headline",            # CSS selectors on article page
            "content": "div.article-body p",
            "date": "time.published",
            "date_attr": "datetime",            # optional: read an attribute instead of text
            "category": "span.section-label"
          }
        }
        """
        self.site_config = site_config
        super().__init__(name=site_config.get("name", "unnamed_site"))
        self.engine = engine or ScrapingEngine()
        self.selectors: Dict[str, str] = site_config.get("selectors", {})
        self.max_articles = int(site_config.get("max_articles", app_config.MAX_ARTICLES_PER_WEBSITE))

    def fetch(self) -> List[Article]:
        listing_url = self.site_config.get("listing_url")
        if not listing_url:
            self.logger.warning("Site %r has no listing_url configured, skipping.", self.name)
            return []
        if not self.selectors.get("article_links"):
            self.logger.warning(
                "Site %r has no 'article_links' selector configured, skipping. "
                "Fill in data/sources_config.yaml -> websites -> selectors.",
                self.name,
            )
            return []

        use_selenium = bool(self.site_config.get("use_selenium", False))
        listing_html = self._fetch_html(listing_url, use_selenium)
        if not listing_html:
            return []

        listing_soup = BeautifulSoup(listing_html, "lxml")
        links = self._parse_listing(listing_soup, base_url=listing_url)
        links = links[: self.max_articles]
        self.logger.info("Found %d article links on %s", len(links), self.name)

        articles: List[Article] = []
        for link in links:
            article_html = self._fetch_html(link, use_selenium)
            if not article_html:
                continue
            try:
                article_soup = BeautifulSoup(article_html, "lxml")
                article = self._parse_article(article_soup, url=link)
                if article:
                    articles.append(article)
            except Exception as exc:
                self.logger.error("Failed to parse article %s: %s", link, exc)
                continue

        self.logger.info("Successfully scraped %d/%d articles from %s", len(articles), len(links), self.name)
        return articles

    # -- I/O -------------------------------------------------------------
    def _fetch_html(self, url: str, use_selenium: bool) -> Optional[str]:
        if use_selenium:
            return self.engine.get_dynamic_html(url)
        response = self.engine.get(url)
        if response is None:
            return None
        response.encoding = response.encoding or "utf-8"
        return response.text

    # -- testable parsing logic -------------------------------------------
    def _parse_listing(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract absolute article URLs from a listing/homepage."""
        seen: List[str] = []
        for tag in soup.select(self.selectors["article_links"]):
            href = tag.get("href")
            if not href:
                continue
            absolute = urljoin(base_url, href)
            if absolute not in seen:
                seen.append(absolute)
        return seen

    def _parse_article(self, soup: BeautifulSoup, url: str) -> Optional[Article]:
        """Extract a single article's fields from its page HTML."""
        title = self._extract_text(soup, self.selectors.get("title"))
        if not title:
            self.logger.debug("No title found for %s, skipping.", url)
            return None

        content_selector = self.selectors.get("content")
        content = ""
        if content_selector:
            paragraphs = [clean_text(p.get_text(" ", strip=True)) for p in soup.select(content_selector)]
            content = "\n\n".join(p for p in paragraphs if p)

        published_date = None
        date_selector = self.selectors.get("date")
        if date_selector:
            date_tag = soup.select_one(date_selector)
            if date_tag is not None:
                date_attr = self.selectors.get("date_attr")
                published_date = (date_tag.get(date_attr) if date_attr else None) or clean_text(
                    date_tag.get_text(" ", strip=True)
                )

        category = self.site_config.get("category", "uncategorized")
        category_selector = self.selectors.get("category")
        if category_selector:
            cat_tag = soup.select_one(category_selector)
            if cat_tag is not None:
                category = clean_text(cat_tag.get_text(" ", strip=True)) or category

        return self._build_article(
            title=title,
            content=content,
            url=url,
            category=category,
            published_date=published_date,
            language=self.site_config.get("language"),
            raw_html=str(soup)[:200_000],  # cap stored raw HTML size
        )

    def _extract_text(self, soup: BeautifulSoup, selector: Optional[str]) -> str:
        if not selector:
            return ""
        tag = soup.select_one(selector)
        return clean_text(tag.get_text(" ", strip=True)) if tag else ""
