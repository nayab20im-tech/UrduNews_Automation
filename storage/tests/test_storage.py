"""
storage/tests/test_storage.py
=============================
Offline test suite for Section 6 — Data Storage Layer. Everything runs
against temporary SQLite files and temp directories; no network access
required, mirroring the offline-testability convention of
`data_acquisition/tests/test_data_acquisition.py`.

Run with:  pytest storage/tests -v   (from the project root, alongside
data_acquisition/)
"""

from __future__ import annotations

import logging
from pathlib import Path

from storage.config_model_storage import ConfigModelStorage
from storage.logs_db import LogsDatabase, attach_sqlite_handler
from storage.media_storage import MediaStorage
from storage.storage_manager import StorageManager


# ---------------------------------------------------------------------------
# Logs DB
# ---------------------------------------------------------------------------
class TestLogsDatabase:
    def test_log_get_and_stats(self, tmp_path):
        db = LogsDatabase(db_path=tmp_path / "logs.db")
        db.log("INFO", "hello world", logger_name="probe", stage="2.1", run_id="r1")
        db.log("ERROR", "boom", run_id="r1")

        rows = db.get_logs(limit=10)
        assert len(rows) == 2
        assert rows[0]["level"] == "ERROR"  # newest first

        assert db.get_logs(level="info")[0]["message"] == "hello world"
        assert db.get_logs(stage="2.1")[0]["logger"] == "probe"

        stats = db.stats()
        assert stats["total_entries"] == 2
        assert stats["by_level"]["ERROR"] == 1
        db.close()

    def test_log_after_close_is_silent(self, tmp_path):
        db = LogsDatabase(db_path=tmp_path / "logs.db")
        db.close()
        db.log("INFO", "must not raise")  # no exception, no recursion

    def test_purge(self, tmp_path):
        db = LogsDatabase(db_path=tmp_path / "logs.db")
        db.log("INFO", "fresh entry")
        with db._cursor() as cur:
            cur.execute(
                "INSERT INTO system_logs (logged_at, level, logger, stage, run_id, message) "
                "VALUES ('2000-01-01T00:00:00+00:00', 'INFO', '', '', '', 'ancient entry')"
            )
        assert db.purge(older_than_days=30) == 1
        messages = [r["message"] for r in db.get_logs(limit=10)]
        assert "fresh entry" in messages
        assert "ancient entry" not in messages
        db.close()

    def test_named_handler_mirrors_records(self, tmp_path):
        db = LogsDatabase(db_path=tmp_path / "logs.db")
        handler = attach_sqlite_handler(db, logger_name="storage_test_probe")
        probe = logging.getLogger("storage_test_probe")
        probe.setLevel(logging.INFO)
        try:
            probe.info("mirrored message")
            assert any(r["message"] == "mirrored message" for r in db.get_logs())
        finally:
            probe.removeHandler(handler)
        db.close()

    def test_global_handler_captures_future_loggers(self, tmp_path):
        from data_acquisition.utils.logger import get_logger

        db = LogsDatabase(db_path=tmp_path / "logs.db")
        handler = attach_sqlite_handler(db)
        fresh = get_logger("storage_test_brand_new")
        try:
            fresh.info("global mirror")
            assert any(r["message"] == "global mirror" for r in db.get_logs())
        finally:
            fresh.removeHandler(handler)
        db.close()


# ---------------------------------------------------------------------------
# Media Storage
# ---------------------------------------------------------------------------
class TestMediaStorage:
    def _make_tree(self, root: Path) -> None:
        (root / "audio").mkdir(parents=True)
        (root / "video" / "final").mkdir(parents=True)
        (root / "video" / "thumbnails").mkdir(parents=True)
        (root / "misc").mkdir(parents=True)
        (root / "audio" / "a1.wav").write_bytes(b"x" * 10)
        (root / "video" / "final" / "v1.mp4").write_bytes(b"y" * 20)
        (root / "video" / "thumbnails" / "t1.jpg").write_bytes(b"z" * 5)
        (root / "misc" / "m.bin").write_bytes(b"q")

    def test_scan_infers_kinds(self, tmp_path):
        root = tmp_path / "media"
        self._make_tree(root)
        ms = MediaStorage(root=root, db_path=tmp_path / "reg.db")
        assert ms.scan() == 4

        kinds = {a["kind"] for a in ms.list_assets()}
        assert kinds == {"audio", "video", "thumbnails", "other"}

        stats = ms.stats()
        assert stats["total_assets"] == 4
        assert stats["by_kind"]["audio"]["bytes"] == 10
        assert stats["by_kind"]["video"]["files"] == 1  # thumbnails not double counted
        ms.close()

    def test_register_and_cleanup_missing(self, tmp_path):
        root = tmp_path / "media"
        (root / "audio").mkdir(parents=True)
        f = root / "audio" / "a.wav"
        f.write_bytes(b"12345")

        ms = MediaStorage(root=root, db_path=tmp_path / "reg.db")
        row = ms.register(f, article_id=7)
        assert row["kind"] == "audio"
        assert row["size_bytes"] == 5
        assert ms.list_assets(article_id=7)[0]["article_id"] == 7

        f.unlink()
        assert ms.cleanup_missing() == 1
        assert ms.list_assets() == []
        ms.close()


# ---------------------------------------------------------------------------
# Config & Model Storage
# ---------------------------------------------------------------------------
class TestConfigModelStorage:
    def test_config_store_roundtrip(self, tmp_path):
        cms = ConfigModelStorage(db_path=tmp_path / "cms.db")
        cms.save_config("effective_config", {"a": 1, "ur": "خبر"})
        assert cms.get_config("effective_config")["ur"] == "خبر"
        assert cms.list_configs()[0]["name"] == "effective_config"
        cms.save_config("effective_config", {"a": 2})  # upsert
        assert cms.get_config("effective_config")["a"] == 2
        cms.delete_config("effective_config")
        assert cms.get_config("effective_config") is None
        cms.close()

    def test_model_registry(self, tmp_path):
        cms = ConfigModelStorage(db_path=tmp_path / "cms.db")
        cms.register_model("piper_ur", tmp_path / "model.onnx", kind="tts", meta={"voice": "ur"})
        model = cms.get_model("piper_ur")
        assert model["kind"] == "tts"
        assert model["meta"]["voice"] == "ur"
        assert len(cms.list_models(kind="tts")) == 1
        assert cms.list_models(kind="avatar") == []
        cms.close()

    def test_snapshot_to_file(self, tmp_path, monkeypatch):
        from storage import config as sc

        monkeypatch.setattr(sc, "CONFIG_SNAPSHOTS_DIR", tmp_path / "snap")
        cms = ConfigModelStorage(db_path=tmp_path / "cms.db")
        path = cms.snapshot_to_file("effective_config", {"x": 1})
        assert path.exists()
        assert path.parent == tmp_path / "snap"
        cms.close()


# ---------------------------------------------------------------------------
# StorageManager facade
# ---------------------------------------------------------------------------
class TestStorageManager:
    def test_health_reports_every_store(self, tmp_path):
        sm = StorageManager(
            db_path=tmp_path / "main.db",
            logs_db_path=tmp_path / "logs.db",
            media_root=tmp_path / "media",
        )
        health = sm.health()
        for key in (
            "raw_news_db", "processed_news_db", "distribution_db",
            "media_storage", "logs_db", "config_model_storage",
        ):
            assert health[key].get("ok") is True, health[key]
        assert health["raw_news_db"]["articles"] == 0
        sm.close()
