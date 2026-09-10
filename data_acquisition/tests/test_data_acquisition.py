"""
tests/test_data_acquisition.py
=================================
Offline test suite for the Data Acquisition layer. None of these tests
touch the network -- RSS/website parsing is exercised against local
fixture files, so the whole suite runs anywhere (including sandboxes
with no internet access) and still proves the collection logic is
correct.

Run with:  pytest -v  (from inside data_acquisition/, or point pytest at
the package: `pytest data_acquisition/tests`)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from ..database.raw_news_db import Article, RawNewsDatabase
from ..scrapers.manual_input_loader import ManualInputLoader
from ..scrapers.news_website_scraper import NewsWebsiteScraper
from ..scrapers.rss_scraper import RSSScraper
from ..utils.text_utils import clean_text, compute_content_hash, detect_language

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# text_utils
# ---------------------------------------------------------------------------
class TestTextUtils:
    def test_clean_text_collapses_whitespace(self):
        assert clean_text("  hello   world  \n\n\n\nagain  ") == "hello world\n\nagain"

    def test_clean_text_handles_none_and_empty(self):
        assert clean_text(None) == ""
        assert clean_text("") == ""

    def test_detect_language_urdu(self):
        assert detect_language("پاکستان میں معاشی اصلاحات کا نیا منصوبہ") == "ur"

    def test_detect_language_english(self):
        assert detect_language("The national cricket team won the match today") == "en"

    def test_detect_language_empty(self):
        assert detect_language("") == "unknown"

    def test_content_hash_is_stable_and_case_insensitive(self):
        h1 = compute_content_hash("https://x.test/a", "Title", "Body")
        h2 = compute_content_hash("HTTPS://X.TEST/A", "title", "body")
        assert h1 == h2

    def test_content_hash_differs_for_different_input(self):
        h1 = compute_content_hash("https://x.test/a")
        h2 = compute_content_hash("https://x.test/b")
        assert h1 != h2


# ---------------------------------------------------------------------------
# RawNewsDatabase
# ---------------------------------------------------------------------------
class TestRawNewsDatabase:
    @pytest.fixture
    def db(self, tmp_path):
        database = RawNewsDatabase(db_path=tmp_path / "test_raw_news.db")
        yield database
        database.close()

    def test_insert_and_count(self, db):
        article = Article(title="Test Title", content="Some body", url="https://x.test/1",
                           source="unit-test", source_type="manual")
        assert db.insert_article(article) is True
        assert db.count() == 1

    def test_duplicate_url_is_ignored(self, db):
        a1 = Article(title="Title A", url="https://x.test/dup", source="s", source_type="manual")
        a2 = Article(title="Title A (slightly different fetch)", url="https://x.test/dup",
                     source="s", source_type="manual")
        assert db.insert_article(a1) is True
        assert db.insert_article(a2) is False
        assert db.count() == 1

    def test_empty_title_is_rejected(self, db):
        assert db.insert_article(Article(title="   ", source="s", source_type="manual")) is False
        assert db.count() == 0

    def test_bulk_insert_stats(self, db):
        articles = [
            Article(title="A", url="https://x.test/a", source="s", source_type="manual"),
            Article(title="B", url="https://x.test/b", source="s", source_type="manual"),
            Article(title="A dup", url="https://x.test/a", source="s", source_type="manual"),
        ]
        stats = db.bulk_insert(articles)
        assert stats == {"inserted": 2, "duplicates": 1, "errors": 0}

    def test_stats_group_by_source_type_and_language(self, db):
        db.insert_article(Article(title="A", url="https://x.test/a", source="s1",
                                   source_type="rss", language="ur"))
        db.insert_article(Article(title="B", url="https://x.test/b", source="s2",
                                   source_type="manual", language="en"))
        stats = db.get_stats()
        assert stats["total_articles"] == 2
        assert stats["by_source_type"] == {"rss": 1, "manual": 1}
        assert stats["by_language"] == {"ur": 1, "en": 1}

    def test_export_to_json(self, db, tmp_path):
        db.insert_article(Article(title="A", url="https://x.test/a", source="s", source_type="manual"))
        out_path = db.export_to_json(tmp_path / "export.json")
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert len(data) == 1
        assert data[0]["title"] == "A"


# ---------------------------------------------------------------------------
# RSSScraper (parsed from a local fixture file -- no network)
# ---------------------------------------------------------------------------
class TestRSSScraper:
    def test_parse_feed_from_local_fixture(self):
        scraper = RSSScraper(feeds=[])
        articles = scraper._parse_feed(
            source=str(FIXTURES / "sample_feed.xml"),
            feed_name="Sample Feed",
            default_category="general",
        )
        # 3 items in the fixture, one has an empty title and must be skipped
        assert len(articles) == 2

        urdu_article = next(a for a in articles if "معاشی" in a.title)
        assert urdu_article.language == "ur"
        assert urdu_article.category == "Business"
        assert urdu_article.url == "https://example-news.test/articles/economic-reforms"

        english_article = next(a for a in articles if "Series" in a.title)
        assert english_article.category == "Sports"
        assert english_article.source_type == "rss"

    def test_fetch_with_no_feeds_returns_empty(self):
        assert RSSScraper(feeds=[]).fetch() == []

    def test_fetch_skips_feed_without_url(self):
        scraper = RSSScraper(feeds=[{"name": "Bad Feed"}])
        assert scraper.fetch() == []


# ---------------------------------------------------------------------------
# NewsWebsiteScraper (parsed from local fixture HTML -- no network)
# ---------------------------------------------------------------------------
class TestNewsWebsiteScraper:
    SITE_CONFIG = {
        "name": "Fixture Site",
        "listing_url": "https://example-news.test/latest",
        "language": "en",
        "category": "general",
        "selectors": {
            "article_links": "a.story-link",
            "title": "h1.headline",
            "content": "div.article-body p",
            "date": "time.published",
            "date_attr": "datetime",
            "category": "span.section-label",
        },
    }

    def test_parse_listing_extracts_only_article_links(self):
        scraper = NewsWebsiteScraper(site_config=self.SITE_CONFIG)
        soup = BeautifulSoup((FIXTURES / "sample_listing.html").read_text(encoding="utf-8"), "lxml")
        links = scraper._parse_listing(soup, base_url="https://example-news.test/latest")
        assert links == [
            "https://example-news.test/articles/story-one",
            "https://example-news.test/articles/story-two",
        ]

    def test_parse_article_extracts_all_fields(self):
        scraper = NewsWebsiteScraper(site_config=self.SITE_CONFIG)
        soup = BeautifulSoup((FIXTURES / "sample_article.html").read_text(encoding="utf-8"), "lxml")
        article = scraper._parse_article(soup, url="https://example-news.test/articles/story-one")

        assert article is not None
        assert article.title == "Local Tech Startups Raise Record Funding"
        assert "record amount of funding" in article.content
        assert article.category == "Technology"
        assert article.published_date == "2026-08-17T10:00:00Z"
        assert article.source_type == "website"

    def test_parse_article_returns_none_without_title(self):
        scraper = NewsWebsiteScraper(site_config=self.SITE_CONFIG)
        soup = BeautifulSoup("<html><body><p>no title here</p></body></html>", "lxml")
        assert scraper._parse_article(soup, url="https://example-news.test/x") is None


# ---------------------------------------------------------------------------
# ManualInputLoader
# ---------------------------------------------------------------------------
class TestManualInputLoader:
    def test_loads_json_list(self, tmp_path):
        payload = [
            {"title": "Manual A", "content": "Body A", "url": "https://x.test/manual-a"},
            {"title": "Manual B", "content": "Body B", "url": "https://x.test/manual-b"},
        ]
        (tmp_path / "batch.json").write_text(json.dumps(payload), encoding="utf-8")

        loader = ManualInputLoader(folder=tmp_path)
        articles = loader.fetch()

        assert len(articles) == 2
        assert {a.title for a in articles} == {"Manual A", "Manual B"}
        # processed files get renamed so re-running doesn't reprocess them
        assert (tmp_path / "batch.json.loaded").exists()
        assert not (tmp_path / "batch.json").exists()

    def test_loads_csv(self, tmp_path):
        csv_path = tmp_path / "batch.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["title", "content", "url", "category"])
            writer.writeheader()
            writer.writerow({"title": "CSV Story", "content": "CSV body", "url": "https://x.test/csv",
                              "category": "local"})

        articles = ManualInputLoader(folder=tmp_path).fetch()
        assert len(articles) == 1
        assert articles[0].title == "CSV Story"
        assert articles[0].category == "local"

    def test_loads_txt_first_line_as_title(self, tmp_path):
        (tmp_path / "note.txt").write_text("Quick Update\nEverything is on schedule.", encoding="utf-8")
        articles = ManualInputLoader(folder=tmp_path).fetch()
        assert len(articles) == 1
        assert articles[0].title == "Quick Update"
        assert articles[0].content == "Everything is on schedule."

    def test_empty_folder_returns_empty_list(self, tmp_path):
        assert ManualInputLoader(folder=tmp_path).fetch() == []
