"""
distribution/tests/test_distribution.py
==========================================
Offline test suite for Section 4 — Distribution & Automation (4.1-4.5).
No network access, API keys, or external services required.

Run with:  pytest distribution/tests -v   (from the project root)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class TestConfig:
    def test_helper_youtube_not_configured(self):
        from distribution import config as dc
        # Without a real secrets file, YouTube is not configured
        original = dc.YOUTUBE_CLIENT_SECRETS_FILE
        try:
            dc.YOUTUBE_CLIENT_SECRETS_FILE = ""
            assert dc.is_youtube_configured() is False
        finally:
            dc.YOUTUBE_CLIENT_SECRETS_FILE = original

    def test_helper_facebook_not_configured(self):
        from distribution import config as dc
        original_page = dc.FACEBOOK_PAGE_ID
        original_token = dc.FACEBOOK_ACCESS_TOKEN
        try:
            dc.FACEBOOK_PAGE_ID = ""
            dc.FACEBOOK_ACCESS_TOKEN = ""
            assert dc.is_facebook_configured() is False
        finally:
            dc.FACEBOOK_PAGE_ID = original_page
            dc.FACEBOOK_ACCESS_TOKEN = original_token

    def test_helper_twitter_not_configured(self):
        from distribution import config as dc
        original_key = dc.TWITTER_API_KEY
        original_token = dc.TWITTER_ACCESS_TOKEN
        try:
            dc.TWITTER_API_KEY = ""
            dc.TWITTER_ACCESS_TOKEN = ""
            assert dc.is_twitter_configured() is False
        finally:
            dc.TWITTER_API_KEY = original_key
            dc.TWITTER_ACCESS_TOKEN = original_token

    def test_helper_telegram_not_configured(self):
        from distribution import config as dc
        original_token = dc.TELEGRAM_BOT_TOKEN
        original_chat = dc.TELEGRAM_CHAT_ID
        try:
            dc.TELEGRAM_BOT_TOKEN = ""
            dc.TELEGRAM_CHAT_ID = ""
            assert dc.is_telegram_configured() is False
        finally:
            dc.TELEGRAM_BOT_TOKEN = original_token
            dc.TELEGRAM_CHAT_ID = original_chat

    def test_helper_whatsapp_not_configured(self):
        from distribution import config as dc
        original_url = dc.WHATSAPP_API_URL
        original_token = dc.WHATSAPP_ACCESS_TOKEN
        original_phone = dc.WHATSAPP_PHONE_NUMBER_ID
        try:
            dc.WHATSAPP_API_URL = ""
            dc.WHATSAPP_ACCESS_TOKEN = ""
            dc.WHATSAPP_PHONE_NUMBER_ID = ""
            assert dc.is_whatsapp_configured() is False
        finally:
            dc.WHATSAPP_API_URL = original_url
            dc.WHATSAPP_ACCESS_TOKEN = original_token
            dc.WHATSAPP_PHONE_NUMBER_ID = original_phone

    def test_default_values(self):
        from distribution import config as dc
        assert dc.YOUTUBE_CATEGORY_ID == "25"
        assert dc.YOUTUBE_PRIVACY_STATUS == "public"
        assert dc.SCHEDULER_MAX_RETRIES >= 1
        assert dc.SCHEDULER_INTERVAL_HOURS > 0


# ---------------------------------------------------------------------------
# Distribution Database
# ---------------------------------------------------------------------------
class TestDistributionDatabase:
    def test_create_and_save_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            from distribution.distribution_db import DistributionDatabase
            db = DistributionDatabase(Path(tmp) / "test.db")
            db.save_job({
                "article_id": 1, "job_id": "j1", "status": "pending",
                "video_path": "/v.mp4", "thumbnail_path": "/t.jpg",
                "headline": "Test", "category": "Politics",
                "youtube_video_id": "", "youtube_url": "", "youtube_status": "",
                "facebook_post_id": "", "facebook_status": "",
                "twitter_post_id": "", "twitter_status": "",
                "telegram_message_id": "", "telegram_status": "",
                "whatsapp_status": "",
                "retry_count": 0, "created_at": "2026-01-01", "updated_at": "2026-01-01",
            })
            job = db.get_job(1)
            assert job is not None
            assert job["article_id"] == 1
            assert job["status"] == "pending"
            db.close()

    def test_get_jobs_by_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            from distribution.distribution_db import DistributionDatabase
            db = DistributionDatabase(Path(tmp) / "test.db")
            for i in range(3):
                db.save_job({
                    "article_id": i, "job_id": f"j{i}",
                    "status": "failed" if i == 1 else "completed",
                    "video_path": "", "thumbnail_path": "",
                    "headline": "", "category": "",
                    "youtube_video_id": "", "youtube_url": "", "youtube_status": "",
                    "facebook_post_id": "", "facebook_status": "",
                    "twitter_post_id": "", "twitter_status": "",
                    "telegram_message_id": "", "telegram_status": "",
                    "whatsapp_status": "",
                    "retry_count": 0, "created_at": "", "updated_at": "",
                })
            failed = db.get_jobs_by_status("failed")
            assert len(failed) == 1
            assert failed[0]["article_id"] == 1
            db.close()

    def test_get_all_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            from distribution.distribution_db import DistributionDatabase
            db = DistributionDatabase(Path(tmp) / "test.db")
            db.save_job({
                "article_id": 10, "job_id": "x", "status": "pending",
                "video_path": "", "thumbnail_path": "",
                "headline": "", "category": "",
                "youtube_video_id": "", "youtube_url": "", "youtube_status": "",
                "facebook_post_id": "", "facebook_status": "",
                "twitter_post_id": "", "twitter_status": "",
                "telegram_message_id": "", "telegram_status": "",
                "whatsapp_status": "",
                "retry_count": 0, "created_at": "", "updated_at": "",
            })
            all_jobs = db.get_all_jobs()
            assert len(all_jobs) == 1
            db.close()

    def test_upsert_updates_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            from distribution.distribution_db import DistributionDatabase
            db = DistributionDatabase(Path(tmp) / "test.db")
            db.save_job({
                "article_id": 5, "job_id": "j5", "status": "pending",
                "video_path": "", "thumbnail_path": "",
                "headline": "Original", "category": "Sports",
                "youtube_video_id": "", "youtube_url": "", "youtube_status": "",
                "facebook_post_id": "", "facebook_status": "",
                "twitter_post_id": "", "twitter_status": "",
                "telegram_message_id": "", "telegram_status": "",
                "whatsapp_status": "",
                "retry_count": 0, "created_at": "", "updated_at": "",
            })
            db.save_job({
                "article_id": 5, "job_id": "j5", "status": "completed",
                "video_path": "", "thumbnail_path": "",
                "headline": "Updated", "category": "Sports",
                "youtube_video_id": "VID123", "youtube_url": "https://yt/VID123",
                "youtube_status": "published",
                "facebook_post_id": "", "facebook_status": "",
                "twitter_post_id": "", "twitter_status": "",
                "telegram_message_id": "", "telegram_status": "",
                "whatsapp_status": "",
                "retry_count": 0, "created_at": "", "updated_at": "",
            })
            job = db.get_job(5)
            assert job["status"] == "completed"
            assert job["headline"] == "Updated"
            assert job["youtube_video_id"] == "VID123"
            db.close()

    def test_save_and_get_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            from distribution.distribution_db import DistributionDatabase
            db = DistributionDatabase(Path(tmp) / "test.db")
            db.save_snapshot({
                "video_id": "V1", "platform": "youtube",
                "snapshot_date": "2026-01-15",
                "views": 100, "likes": 10, "comments": 5,
                "shares": 2, "watch_time": 50.0, "engagement_rate": 0.15,
                "raw_json": "{}",
            })
            snaps = db.get_snapshots(video_id="V1")
            assert len(snaps) == 1
            assert snaps[0]["views"] == 100
            db.close()

    def test_save_and_get_playlist(self):
        with tempfile.TemporaryDirectory() as tmp:
            from distribution.distribution_db import DistributionDatabase
            db = DistributionDatabase(Path(tmp) / "test.db")
            db.save_playlist("Politics", "PL123", "Urdu News AI - Politics", "2026-01-01")
            pl = db.get_playlist("Politics")
            assert pl is not None
            assert pl["playlist_id"] == "PL123"
            assert pl["playlist_name"] == "Urdu News AI - Politics"

            all_pl = db.get_all_playlists()
            assert "Politics" in all_pl
            db.close()

    def test_get_failed_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            from distribution.distribution_db import DistributionDatabase
            db = DistributionDatabase(Path(tmp) / "test.db")
            db.save_job({
                "article_id": 1, "job_id": "j1", "status": "failed",
                "video_path": "", "thumbnail_path": "",
                "headline": "", "category": "",
                "youtube_video_id": "", "youtube_url": "", "youtube_status": "",
                "facebook_post_id": "", "facebook_status": "",
                "twitter_post_id": "", "twitter_status": "",
                "telegram_message_id": "", "telegram_status": "",
                "whatsapp_status": "",
                "retry_count": 0, "created_at": "", "updated_at": "",
            })
            db.save_job({
                "article_id": 2, "job_id": "j2", "status": "failed",
                "video_path": "", "thumbnail_path": "",
                "headline": "", "category": "",
                "youtube_video_id": "", "youtube_url": "", "youtube_status": "",
                "facebook_post_id": "", "facebook_status": "",
                "twitter_post_id": "", "twitter_status": "",
                "telegram_message_id": "", "telegram_status": "",
                "whatsapp_status": "",
                "retry_count": 5, "created_at": "", "updated_at": "",
            })
            # max_retries=3 should only return article 1 (retry_count=0 < 3)
            failed = db.get_failed_jobs(max_retries=3)
            assert len(failed) == 1
            assert failed[0]["article_id"] == 1
            db.close()

    def test_context_manager(self):
        with tempfile.TemporaryDirectory() as tmp:
            from distribution.distribution_db import DistributionDatabase
            with DistributionDatabase(Path(tmp) / "ctx.db") as db:
                db.save_job({
                    "article_id": 99, "job_id": "ctx", "status": "pending",
                    "video_path": "", "thumbnail_path": "",
                    "headline": "", "category": "",
                    "youtube_video_id": "", "youtube_url": "", "youtube_status": "",
                    "facebook_post_id": "", "facebook_status": "",
                    "twitter_post_id": "", "twitter_status": "",
                    "telegram_message_id": "", "telegram_status": "",
                    "whatsapp_status": "",
                    "retry_count": 0, "created_at": "", "updated_at": "",
                })
                assert db.get_job(99) is not None


# ---------------------------------------------------------------------------
# Stage 4.1 — YouTube Uploader
# ---------------------------------------------------------------------------
class TestYouTubeUploader:
    def test_build_title_default_template(self):
        from distribution.youtube_uploader import _build_title
        title = _build_title("Breaking News", "Test Channel")
        assert "Breaking News" in title
        assert "Test Channel" in title

    def test_build_title_truncation(self):
        from distribution.youtube_uploader import _build_title
        long_headline = "A" * 200
        title = _build_title(long_headline)
        assert len(title) <= 100

    def test_build_description_includes_content(self):
        from distribution.youtube_uploader import _build_description
        desc = _build_description(
            "Test Headline", "This is the full script content.", "Politics",
        )
        assert "full script" in desc
        assert "Politics" in desc
        assert "#UrduNews" in desc

    def test_build_description_truncates_long_script(self):
        from distribution.youtube_uploader import _build_description
        long_script = "Word " * 500
        desc = _build_description("Title", long_script)
        assert len(desc) <= 5000

    def test_build_tags_includes_keywords(self):
        from distribution.youtube_uploader import _build_tags
        tags = _build_tags("Government announces new budget", "Business")
        assert "Government" in tags
        assert "Urdu News" in tags
        assert "Business" in tags

    def test_build_tags_deduplicates(self):
        from distribution.youtube_uploader import _build_tags
        tags = _build_tags("test test test news")
        assert tags.count("test") == 1

    def test_mock_upload(self, monkeypatch):
        from distribution import config as dc
        from distribution.youtube_uploader import upload_video
        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setattr(dc, "DISTRIBUTION_MOCK_MODE", True)
            vid = Path(tmp) / "test.mp4"
            vid.write_bytes(b"fake video")
            result = upload_video(1, vid, "Test Headline", "Script", "Politics")
            assert result["status"] == "mock_published"
            assert result["video_id"].startswith("MOCK_")
            assert "youtube.com" in result["url"]

    def test_upload_missing_file(self):
        from distribution.youtube_uploader import upload_video
        result = upload_video(1, "/nonexistent/file.mp4", "Test")
        assert result["status"] == "error"
        assert "not found" in result["error"]

    def test_upload_not_configured(self, monkeypatch):
        from distribution import config as dc
        from distribution.youtube_uploader import upload_video
        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setattr(dc, "DISTRIBUTION_MOCK_MODE", False)
            monkeypatch.setattr(dc, "YOUTUBE_CLIENT_SECRETS_FILE", "")
            vid = Path(tmp) / "test.mp4"
            vid.write_bytes(b"fake video")
            result = upload_video(1, vid, "Test")
            assert result["status"] == "not_configured"

    def test_mock_set_thumbnail(self, monkeypatch):
        from distribution import config as dc
        from distribution.youtube_uploader import set_thumbnail
        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setattr(dc, "DISTRIBUTION_MOCK_MODE", True)
            thumb = Path(tmp) / "thumb.jpg"
            thumb.write_bytes(b"fake image")
            result = set_thumbnail("VID123", thumb)
            assert result["status"] == "mock_ok"

    def test_set_thumbnail_missing_file(self):
        from distribution.youtube_uploader import set_thumbnail
        result = set_thumbnail("VID123", "/nonexistent/thumb.jpg")
        assert result["status"] == "error"

    def test_batch_run_mock(self, monkeypatch):
        from distribution import config as dc
        from distribution.youtube_uploader import run
        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setattr(dc, "DISTRIBUTION_MOCK_MODE", True)
            vid = Path(tmp) / "video.mp4"
            vid.write_bytes(b"fake")
            results = run([
                {"article_id": 1, "subtitled_video": str(vid),
                 "headline": "Test 1", "full_script": "", "category": "News"},
                {"article_id": 2, "subtitled_video": "",
                 "headline": "Test 2", "full_script": "", "category": "News"},
            ])
            assert len(results) == 2
            assert results[0]["status"] == "mock_published"
            assert results[1]["status"] == "skipped"


# ---------------------------------------------------------------------------
# Stage 4.2 — Social Content
# ---------------------------------------------------------------------------
class TestSocialContent:
    def test_facebook_post_includes_headline(self):
        from distribution.social_content import generate_facebook_post
        post = generate_facebook_post("Breaking News", "A summary.", "Politics", "https://yt.be/x")
        assert "Breaking News" in post
        assert "Politics" in post
        assert "https://yt.be/x" in post
        assert "#UrduNews" in post

    def test_twitter_post_max_280(self):
        from distribution.social_content import generate_twitter_post
        post = generate_twitter_post(
            "A" * 200, "B" * 200, "Category", "https://youtube.com/watch?v=123",
        )
        assert len(post) <= 280

    def test_twitter_post_includes_url(self):
        from distribution.social_content import generate_twitter_post
        post = generate_twitter_post("Short headline", "", "", "https://yt.be/x")
        assert "https://yt.be/x" in post

    def test_telegram_caption_has_markdown(self):
        from distribution.social_content import generate_telegram_caption
        caption = generate_telegram_caption("Test Headline", "Summary", "Sports")
        assert "*Test Headline*" in caption
        assert "_Sports_" in caption

    def test_telegram_caption_includes_link(self):
        from distribution.social_content import generate_telegram_caption
        caption = generate_telegram_caption("H", "", "", "https://yt.be/abc")
        assert "https://yt.be/abc" in caption

    def test_whatsapp_message_plain_text(self):
        from distribution.social_content import generate_whatsapp_message
        msg = generate_whatsapp_message("Test", "Summary", "Tech")
        assert "*Test*" not in msg  # no markdown bold
        assert "Test" in msg
        assert "Tech" in msg

    def test_generate_all_returns_all_platforms(self):
        from distribution.social_content import generate_all
        result = generate_all("Headline", "Summary", "Politics", "https://yt.be/x")
        assert "facebook" in result
        assert "twitter" in result
        assert "telegram" in result
        assert "whatsapp" in result

    def test_empty_inputs_dont_crash(self):
        from distribution.social_content import generate_all
        result = generate_all("", "", "", "")
        assert all(isinstance(v, str) for v in result.values())


# ---------------------------------------------------------------------------
# Stage 4.2 — Social Sharer
# ---------------------------------------------------------------------------
class TestSocialSharer:
    def test_share_article_mock(self, monkeypatch):
        from distribution import config as dc
        from distribution.social_sharer import share_article
        monkeypatch.setattr(dc, "DISTRIBUTION_MOCK_MODE", True)
        monkeypatch.setattr(dc, "FACEBOOK_PAGE_ID", "page1")
        monkeypatch.setattr(dc, "FACEBOOK_ACCESS_TOKEN", "token1")
        monkeypatch.setattr(dc, "TWITTER_API_KEY", "key1")
        monkeypatch.setattr(dc, "TWITTER_ACCESS_TOKEN", "tok1")
        monkeypatch.setattr(dc, "TELEGRAM_BOT_TOKEN", "bot1")
        monkeypatch.setattr(dc, "TELEGRAM_CHAT_ID", "chat1")
        monkeypatch.setattr(dc, "WHATSAPP_API_URL", "https://api.wa.test")
        monkeypatch.setattr(dc, "WHATSAPP_ACCESS_TOKEN", "wa_tok")
        monkeypatch.setattr(dc, "WHATSAPP_PHONE_NUMBER_ID", "phone1")

        results = share_article("Test Headline", "Summary", "News", "https://yt.be/x")
        for platform in ("facebook", "twitter", "telegram", "whatsapp"):
            assert platform in results
            assert results[platform]["status"] == "mock_posted"

    def test_share_not_configured(self, monkeypatch):
        from distribution import config as dc
        from distribution.social_sharer import share_article
        monkeypatch.setattr(dc, "DISTRIBUTION_MOCK_MODE", False)
        monkeypatch.setattr(dc, "FACEBOOK_PAGE_ID", "")
        monkeypatch.setattr(dc, "FACEBOOK_ACCESS_TOKEN", "")
        monkeypatch.setattr(dc, "TWITTER_API_KEY", "")
        monkeypatch.setattr(dc, "TWITTER_ACCESS_TOKEN", "")
        monkeypatch.setattr(dc, "TELEGRAM_BOT_TOKEN", "")
        monkeypatch.setattr(dc, "TELEGRAM_CHAT_ID", "")
        monkeypatch.setattr(dc, "WHATSAPP_API_URL", "")
        monkeypatch.setattr(dc, "WHATSAPP_ACCESS_TOKEN", "")
        monkeypatch.setattr(dc, "WHATSAPP_PHONE_NUMBER_ID", "")

        results = share_article("Test")
        for platform in ("facebook", "twitter", "telegram", "whatsapp"):
            assert results[platform]["status"] == "not_configured"

    def test_batch_run_mock(self, monkeypatch):
        from distribution import config as dc
        from distribution.social_sharer import run
        monkeypatch.setattr(dc, "DISTRIBUTION_MOCK_MODE", True)
        monkeypatch.setattr(dc, "FACEBOOK_PAGE_ID", "p")
        monkeypatch.setattr(dc, "FACEBOOK_ACCESS_TOKEN", "t")
        monkeypatch.setattr(dc, "TWITTER_API_KEY", "k")
        monkeypatch.setattr(dc, "TWITTER_ACCESS_TOKEN", "t")
        monkeypatch.setattr(dc, "TELEGRAM_BOT_TOKEN", "b")
        monkeypatch.setattr(dc, "TELEGRAM_CHAT_ID", "c")
        monkeypatch.setattr(dc, "WHATSAPP_API_URL", "u")
        monkeypatch.setattr(dc, "WHATSAPP_ACCESS_TOKEN", "t")
        monkeypatch.setattr(dc, "WHATSAPP_PHONE_NUMBER_ID", "p")

        results = run([
            {"article_id": 1, "headline": "H1", "youtube_url": "https://yt.be/1",
             "full_script": "Script 1", "category": "News"},
        ])
        assert len(results) == 1
        assert results[0]["article_id"] == 1


# ---------------------------------------------------------------------------
# Stage 4.3 — Playlist Manager
# ---------------------------------------------------------------------------
class TestPlaylistManager:
    def test_build_playlist_name(self):
        from distribution.playlist_manager import _build_playlist_name
        from distribution import config as dc
        original = dc.PLAYLIST_NAME_PREFIX
        try:
            dc.PLAYLIST_NAME_PREFIX = "Test News"
            name = _build_playlist_name("Politics")
            assert name == "Test News - Politics"
        finally:
            dc.PLAYLIST_NAME_PREFIX = original

    def test_build_playlist_name_default_prefix(self):
        from distribution.playlist_manager import _build_playlist_name
        from distribution import config as dc
        original = dc.PLAYLIST_NAME_PREFIX
        try:
            dc.PLAYLIST_NAME_PREFIX = ""
            name = _build_playlist_name("Sports")
            assert "Sports" in name
            assert dc.CHANNEL_NAME in name
        finally:
            dc.PLAYLIST_NAME_PREFIX = original

    def test_get_or_create_mock(self, monkeypatch):
        from distribution import config as dc
        from distribution.playlist_manager import get_or_create_playlist
        from distribution.distribution_db import DistributionDatabase
        monkeypatch.setattr(dc, "DISTRIBUTION_MOCK_MODE", True)
        with tempfile.TemporaryDirectory() as tmp:
            db = DistributionDatabase(Path(tmp) / "test.db")
            result = get_or_create_playlist("Politics", db=db)
            assert result["status"] == "mock_created"
            assert result["playlist_id"].startswith("PL_MOCK_")
            # Second call should return cached
            result2 = get_or_create_playlist("Politics", db=db)
            assert result2["status"] == "cached"
            assert result2["playlist_id"] == result["playlist_id"]
            db.close()

    def test_add_video_mock(self, monkeypatch):
        from distribution import config as dc
        from distribution.playlist_manager import add_video_to_playlist
        monkeypatch.setattr(dc, "DISTRIBUTION_MOCK_MODE", True)
        result = add_video_to_playlist("VID123", "PL456")
        assert result["status"] == "mock_ok"

    def test_add_video_missing_ids(self):
        from distribution.playlist_manager import add_video_to_playlist
        result = add_video_to_playlist("", "PL456")
        assert result["status"] == "skipped"

    def test_batch_run_mock(self, monkeypatch):
        from distribution import config as dc
        from distribution.playlist_manager import run
        from distribution.distribution_db import DistributionDatabase
        monkeypatch.setattr(dc, "DISTRIBUTION_MOCK_MODE", True)
        with tempfile.TemporaryDirectory() as tmp:
            db = DistributionDatabase(Path(tmp) / "test.db")
            results = run([
                {"article_id": 1, "category": "Politics", "youtube_video_id": "V1"},
                {"article_id": 2, "category": "Sports", "youtube_video_id": ""},
            ], db=db)
            assert len(results) == 2
            assert results[0]["status"] == "mock_ok"
            assert results[1]["status"] == "skipped"
            db.close()


# ---------------------------------------------------------------------------
# Stage 4.4 — Analytics
# ---------------------------------------------------------------------------
class TestAnalytics:
    def test_mock_fetch_stats(self, monkeypatch):
        from distribution import config as dc
        from distribution.analytics import fetch_youtube_stats
        monkeypatch.setattr(dc, "DISTRIBUTION_MOCK_MODE", True)
        stats = fetch_youtube_stats("VID123")
        assert stats is not None
        assert stats["video_id"] == "VID123"
        assert stats["views"] == 150

    def test_store_and_retrieve_snapshot(self, monkeypatch):
        from distribution import config as dc
        from distribution.analytics import store_snapshot
        from distribution.distribution_db import DistributionDatabase
        monkeypatch.setattr(dc, "DISTRIBUTION_MOCK_MODE", True)
        with tempfile.TemporaryDirectory() as tmp:
            db = DistributionDatabase(Path(tmp) / "test.db")
            store_snapshot(db, "VID1", "youtube", {
                "video_id": "VID1", "views": 200, "likes": 20,
                "comments": 5, "shares": 3, "watch_time": 100.0,
                "engagement_rate": 0.125,
            })
            snaps = db.get_snapshots(video_id="VID1")
            assert len(snaps) == 1
            assert snaps[0]["views"] == 200
            db.close()

    def test_daily_report_generation(self, monkeypatch):
        from distribution import config as dc
        from distribution.analytics import generate_daily_report
        from distribution.distribution_db import DistributionDatabase
        monkeypatch.setattr(dc, "DISTRIBUTION_MOCK_MODE", True)
        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setattr(dc, "REPORTS_DIR", str(Path(tmp) / "reports"))
            db = DistributionDatabase(Path(tmp) / "test.db")
            db.save_job({
                "article_id": 1, "job_id": "j1", "status": "completed",
                "video_path": "", "thumbnail_path": "",
                "headline": "", "category": "",
                "youtube_video_id": "V1", "youtube_url": "",
                "youtube_status": "published",
                "facebook_post_id": "", "facebook_status": "",
                "twitter_post_id": "", "twitter_status": "",
                "telegram_message_id": "", "telegram_status": "",
                "whatsapp_status": "",
                "retry_count": 0, "created_at": "", "updated_at": "",
            })
            report = generate_daily_report(db, "2026-01-15")
            assert report["report_type"] == "daily"
            assert report["date"] == "2026-01-15"
            assert report["summary"]["total_videos"] == 1
            assert report["summary"]["published"] == 1
            # Check files were written
            reports_dir = Path(tmp) / "reports"
            assert (reports_dir / "daily_report_2026-01-15.json").exists()
            assert (reports_dir / "daily_report_2026-01-15.csv").exists()
            db.close()

    def test_weekly_report_generation(self, monkeypatch):
        from distribution import config as dc
        from distribution.analytics import generate_weekly_report
        from distribution.distribution_db import DistributionDatabase
        monkeypatch.setattr(dc, "DISTRIBUTION_MOCK_MODE", True)
        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setattr(dc, "REPORTS_DIR", str(Path(tmp) / "reports"))
            db = DistributionDatabase(Path(tmp) / "test.db")
            report = generate_weekly_report(db, "2026-01-15")
            assert report["report_type"] == "weekly"
            assert report["start_date"] == "2026-01-09"
            assert report["end_date"] == "2026-01-15"
            db.close()

    def test_collect_all_stats(self, monkeypatch):
        from distribution import config as dc
        from distribution.analytics import collect_all_stats
        from distribution.distribution_db import DistributionDatabase
        monkeypatch.setattr(dc, "DISTRIBUTION_MOCK_MODE", True)
        with tempfile.TemporaryDirectory() as tmp:
            db = DistributionDatabase(Path(tmp) / "test.db")
            for i in range(3):
                db.save_job({
                    "article_id": i, "job_id": f"j{i}",
                    "status": "completed",
                    "video_path": "", "thumbnail_path": "",
                    "headline": "", "category": "",
                    "youtube_video_id": f"V{i}" if i < 2 else "",
                    "youtube_url": "",
                    "youtube_status": "published" if i < 2 else "",
                    "facebook_post_id": "", "facebook_status": "",
                    "twitter_post_id": "", "twitter_status": "",
                    "telegram_message_id": "", "telegram_status": "",
                    "whatsapp_status": "",
                    "retry_count": 0, "created_at": "", "updated_at": "",
                })
            count = collect_all_stats(db)
            assert count == 2  # only 2 have published status
            db.close()


# ---------------------------------------------------------------------------
# Distribution Pipeline (orchestrator)
# ---------------------------------------------------------------------------
class TestDistributionPipeline:
    def test_filter_eligible(self):
        from distribution.pipeline import _filter_eligible
        with tempfile.TemporaryDirectory() as tmp:
            vid = Path(tmp) / "video.mp4"
            vid.write_bytes(b"fake")
            articles = [
                {"subtitled_video": str(vid), "headline": "H1"},
                {"subtitled_video": "", "headline": "H2"},
                {"subtitled_video": "/nonexistent.mp4", "headline": "H3"},
            ]
            eligible = _filter_eligible(articles)
            assert len(eligible) == 1
            assert eligible[0]["headline"] == "H1"

    def test_build_work_items(self):
        from distribution.pipeline import _build_work_items
        from distribution.distribution_db import DistributionDatabase
        with tempfile.TemporaryDirectory() as tmp:
            db = DistributionDatabase(Path(tmp) / "test.db")
            articles = [
                {"article_id": 1, "headline": "Test", "subtitled_video": "/v.mp4",
                 "full_script": "Script", "category": "Politics",
                 "thumbnail_path": "/t.jpg"},
            ]
            items = _build_work_items(articles, db)
            assert len(items) == 1
            assert items[0]["headline"] == "Test"
            assert items[0]["job_id"]  # UUID assigned
            # Verify DB row created
            job = db.get_job(1)
            assert job is not None
            assert job["status"] == "pending"
            db.close()

    def test_run_distribution_mock(self, monkeypatch):
        from distribution import config as dc
        from distribution.pipeline import run_distribution
        from distribution.distribution_db import DistributionDatabase
        monkeypatch.setattr(dc, "DISTRIBUTION_MOCK_MODE", True)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            monkeypatch.setattr(dc, "DISTRIBUTION_DB_PATH", Path(tmp) / "dist.db")
            monkeypatch.setattr(dc, "REPORTS_DIR", str(Path(tmp) / "reports"))
            vid = Path(tmp) / "video.mp4"
            vid.write_bytes(b"fake video")
            db = DistributionDatabase(Path(tmp) / "dist.db")
            report = run_distribution(
                [{"article_id": 1, "headline": "Test", "subtitled_video": str(vid),
                  "full_script": "Script text here", "category": "News",
                  "thumbnail_path": ""}],
                db=db,
            )
            assert len(report.stages) == 4  # youtube, playlist, social, analytics
            assert report.stages[0].stage == "4.1_youtube_upload"
            assert len(report.outputs) == 1
            db.close()

    def test_run_distribution_empty(self):
        from distribution.pipeline import run_distribution
        report = run_distribution([])
        assert len(report.outputs) == 0
        assert len(report.stages) == 0

    def test_run_distribution_no_eligible(self):
        from distribution.pipeline import run_distribution
        report = run_distribution([
            {"article_id": 1, "headline": "H", "subtitled_video": ""},
        ])
        assert len(report.outputs) == 0

    def test_retry_failed_jobs_empty(self):
        from distribution.pipeline import retry_failed_jobs
        from distribution.distribution_db import DistributionDatabase
        with tempfile.TemporaryDirectory() as tmp:
            db = DistributionDatabase(Path(tmp) / "test.db")
            report = retry_failed_jobs(max_retries=3, db=db)
            assert len(report.outputs) == 0
            db.close()

    def test_report_as_dict(self):
        from distribution.pipeline import DistributionReport, DistributionStageReport
        report = DistributionReport(
            started_at="2026-01-01T00:00:00",
            finished_at="2026-01-01T00:01:00",
            stages=[DistributionStageReport(stage="4.1_youtube", processed=2, ok=1, skipped_or_error=1)],
            outputs=[{"article_id": 1}],
        )
        d = report.as_dict()
        assert d["started_at"] == "2026-01-01T00:00:00"
        assert len(d["stages"]) == 1
        assert d["stages"][0]["stage"] == "4.1_youtube"


# ---------------------------------------------------------------------------
# Stage 4.5 — Scheduler
# ---------------------------------------------------------------------------
class TestScheduler:
    def test_interval_seconds(self):
        from distribution.scheduler import DistributionScheduler
        s = DistributionScheduler(interval_hours=2)
        assert s._interval_seconds() == 7200.0

    def test_daily_time_calculation(self):
        from distribution.scheduler import DistributionScheduler
        s = DistributionScheduler(daily_time="12:00")
        delay = s._seconds_until_daily()
        assert 0 < delay <= 86400

    def test_daily_time_invalid_falls_back(self):
        from distribution.scheduler import DistributionScheduler
        s = DistributionScheduler(daily_time="invalid", interval_hours=1)
        delay = s._seconds_until_daily()
        assert delay == 3600.0  # falls back to interval

    def test_run_once_doesnt_crash(self, monkeypatch):
        from distribution import config as dc
        from distribution.scheduler import DistributionScheduler
        monkeypatch.setattr(dc, "DISTRIBUTION_MOCK_MODE", True)
        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setattr(dc, "DISTRIBUTION_DB_PATH", Path(tmp) / "test.db")
            s = DistributionScheduler()
            s.run_once()  # should not raise

    def test_stop_sets_event(self):
        from distribution.scheduler import DistributionScheduler
        s = DistributionScheduler()
        assert not s._stop_event.is_set()
        s.stop()
        assert s._stop_event.is_set()


# ---------------------------------------------------------------------------
# Full distribution pipeline (integration)
# ---------------------------------------------------------------------------
class TestDistributionIntegration:
    def test_full_mock_pipeline(self, monkeypatch):
        """End-to-end: 2 articles with videos through all 4 stages."""
        from distribution import config as dc
        from distribution.pipeline import run_distribution
        from distribution.distribution_db import DistributionDatabase

        monkeypatch.setattr(dc, "DISTRIBUTION_MOCK_MODE", True)
        monkeypatch.setattr(dc, "FACEBOOK_PAGE_ID", "p")
        monkeypatch.setattr(dc, "FACEBOOK_ACCESS_TOKEN", "t")
        monkeypatch.setattr(dc, "TWITTER_API_KEY", "k")
        monkeypatch.setattr(dc, "TWITTER_ACCESS_TOKEN", "t")
        monkeypatch.setattr(dc, "TELEGRAM_BOT_TOKEN", "b")
        monkeypatch.setattr(dc, "TELEGRAM_CHAT_ID", "c")
        monkeypatch.setattr(dc, "WHATSAPP_API_URL", "u")
        monkeypatch.setattr(dc, "WHATSAPP_ACCESS_TOKEN", "t")
        monkeypatch.setattr(dc, "WHATSAPP_PHONE_NUMBER_ID", "p")

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            monkeypatch.setattr(dc, "DISTRIBUTION_DB_PATH", Path(tmp) / "dist.db")
            monkeypatch.setattr(dc, "REPORTS_DIR", str(Path(tmp) / "reports"))

            vid1 = Path(tmp) / "v1.mp4"
            vid2 = Path(tmp) / "v2.mp4"
            vid1.write_bytes(b"video1")
            vid2.write_bytes(b"video2")

            db = DistributionDatabase(Path(tmp) / "dist.db")
            report = run_distribution([
                {"article_id": 1, "headline": "First News", "subtitled_video": str(vid1),
                 "full_script": "Script 1", "category": "Politics", "thumbnail_path": ""},
                {"article_id": 2, "headline": "Second News", "subtitled_video": str(vid2),
                 "full_script": "Script 2", "category": "Sports", "thumbnail_path": ""},
            ], db=db)

            assert len(report.outputs) == 2
            assert len(report.stages) == 4
            # YouTube stage should have 2 published
            yt_stage = report.stages[0]
            assert yt_stage.ok == 2

            # Both should be completed
            for item in report.outputs:
                assert item.get("youtube_status") in ("published", "mock_published")

            # Jobs should be in DB
            jobs = db.get_all_jobs()
            assert len(jobs) == 2
            assert all(j["status"] == "completed" for j in jobs)
            db.close()
