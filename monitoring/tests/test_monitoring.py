"""
Tests for monitoring/ (Section 7 dashboard data layer, server, CLI and
the Section 8 system check). Fully offline; every store lives in
``tmp_path`` so the real databases are never touched.
"""

from __future__ import annotations

import importlib.util
import json
import threading
import urllib.request
from types import SimpleNamespace

import pytest

from data_acquisition.database.raw_news_db import Article
from data_acquisition.utils.text_utils import safe_iso_now
from storage.storage_manager import StorageManager

from monitoring.dashboard_data import DashboardData
from monitoring.server import create_dashboard_server
from monitoring.system_check import collect_system_check, summary


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def storage(tmp_path):
    manager = StorageManager(
        db_path=tmp_path / "raw_news.db",
        logs_db_path=tmp_path / "system_logs.db",
        media_root=tmp_path / "media",
    )
    _seed(manager)
    yield manager
    manager.close()


def _seed(manager: StorageManager) -> None:
    manager.raw.insert_article(Article(
        title="Test headline", content="Some body text.",
        url="http://example.com/1", source="test", source_type="website",
        language="en",
    ))
    manager.processed.save_preprocessed(SimpleNamespace(
        article_id=1, title="Test headline", source="test",
        url="http://example.com/1", published_date="2026-08-26",
        language="en", cleaned_text="Some body text.",
        processed_text="some body text.", sentences=["Some body text."],
        preprocessing_status="ok", error_message=None,
    ), updated_at=safe_iso_now())

    manager.distribution.save_job({
        "article_id": 1, "headline": "Test headline",
        "video_path": "/tmp/video.mp4", "status": "pending",
        "youtube_status": "draft", "retry_count": 0,
        "created_at": safe_iso_now(), "updated_at": safe_iso_now(),
    })
    # Two snapshots for the same video: totals must use the latest only.
    manager.distribution.save_snapshot({
        "video_id": "v1", "platform": "youtube", "snapshot_date": "2026-08-25",
        "views": 100, "likes": 5, "comments": 1, "shares": 2,
    })
    manager.distribution.save_snapshot({
        "video_id": "v1", "platform": "youtube", "snapshot_date": "2026-08-26",
        "views": 150, "likes": 8, "comments": 2, "shares": 3,
    })
    manager.logs.log("INFO", "test log entry", logger_name="monitoring",
                     stage="seed")


@pytest.fixture()
def data(storage):
    return DashboardData(storage=storage)


# ---------------------------------------------------------------------------
# DashboardData
# ---------------------------------------------------------------------------
def test_pipeline_status(data):
    status = data.pipeline_status()
    assert status["raw"]["total_articles"] == 1
    assert status["processed"]["articles"] == 1
    assert status["distribution"]["jobs"] == 1
    assert status["media"]["total_assets"] == 0
    assert isinstance(status["recent_runs"], list)


def test_latest_news(data):
    items = data.latest_news(limit=5)
    assert len(items) == 1
    item = items[0]
    assert item["title"] == "Test headline"
    assert item["youtube_status"] == "draft"
    assert item["has_tts_audio"] is False


def test_upload_status(data):
    uploads = data.upload_status()
    assert uploads["total_jobs"] == 1
    assert uploads["by_status"] == {"pending": 1}
    assert uploads["by_youtube_status"] == {"draft": 1}
    assert uploads["recent"][0]["headline"] == "Test headline"


def test_analytics_uses_latest_snapshot_per_video(data):
    analytics = data.analytics_overview()
    assert analytics["totals"]["views"] == 150  # not 250
    assert analytics["totals"]["videos"] == 1
    assert analytics["by_platform"]["youtube"]["likes"] == 8
    assert analytics["top_videos"][0]["video_id"] == "v1"


def test_storage_health_and_logs(data):
    health = data.storage_health()
    assert all(probe.get("ok") for probe in health.values())
    logs = data.recent_logs(limit=5)
    assert any(row.get("message") == "test log entry" for row in logs)


def test_full_report_shape(data):
    report = data.full_report()
    for key in ("generated_at", "pipeline_status", "latest_news",
                "upload_status", "analytics_overview", "storage_health",
                "recent_logs", "system_check"):
        assert key in report
    assert report["generated_at"]


# ---------------------------------------------------------------------------
# system check (Section 8)
# ---------------------------------------------------------------------------
def test_system_check_entries():
    checks = collect_system_check()
    names = {c["name"] for c in checks}
    assert {"python_version", "cpu_cores", "ram", "gpu", "storage",
            "ffmpeg", "urdu_font", "tts_models"} <= names
    for check in checks:
        assert check["status"] in {"ok", "warn", "missing"}
        assert check["recommended"]
    counts = summary(checks)
    assert sum(counts.values()) == len(checks)


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------
def test_server_endpoints(data):
    httpd, port = create_dashboard_server(data, host="127.0.0.1", port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    try:
        html = urllib.request.urlopen(f"{base}/", timeout=5).read().decode()
        assert "UrduNewsAI" in html
        assert "__REFRESH_SECONDS__" not in html

        report = json.loads(
            urllib.request.urlopen(f"{base}/api/report", timeout=5).read())
        assert report["pipeline_status"]["raw"]["total_articles"] == 1

        health = json.loads(
            urllib.request.urlopen(f"{base}/api/health", timeout=5).read())
        assert health["raw_news_db"]["ok"] is True

        try:
            urllib.request.urlopen(f"{base}/api/nope", timeout=5)
            raise AssertionError("expected 404")
        except urllib.error.HTTPError as err:
            assert err.code == 404
    finally:
        httpd.shutdown()
        httpd.server_close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_report_and_export(data, tmp_path, monkeypatch, capsys):
    from monitoring import run_dashboard

    monkeypatch.setattr(run_dashboard, "DashboardData", lambda: data)
    assert run_dashboard.main(["--report"]) == 0
    out = capsys.readouterr().out
    assert '"pipeline_status"' in out

    export_path = tmp_path / "report.json"
    assert run_dashboard.main(["--export", str(export_path)]) == 0
    assert json.loads(export_path.read_text(encoding="utf-8"))["latest_news"]


def test_cli_syscheck(monkeypatch, capsys):
    from monitoring import run_dashboard

    assert run_dashboard.main(["--syscheck"]) == 0
    assert "REQUIREMENT" in capsys.readouterr().out


@pytest.mark.skipif(
    importlib.util.find_spec("streamlit") is not None,
    reason="streamlit installed; the missing-backend error path is skipped",
)
def test_cli_streamlit_backend_missing(monkeypatch):
    from monitoring import run_dashboard

    class _FakeData:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

    monkeypatch.setattr(run_dashboard, "DashboardData", _FakeData)
    assert run_dashboard.main(["--backend", "streamlit"]) == 1
