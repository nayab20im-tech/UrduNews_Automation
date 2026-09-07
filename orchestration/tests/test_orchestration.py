"""
orchestration/tests/test_orchestration.py
=========================================
Offline test suite for Section 5 — Orchestration & Pipeline Control.
Runs against temporary SQLite files; no network access required (the
notification channels are unconfigured by default, so sends degrade to
log-only).

Run with:  pytest orchestration/tests -v   (from the project root,
alongside data_acquisition/)
"""

from __future__ import annotations

from orchestration.automation_engine import AutomationEngine
from orchestration.config_manager import ConfigManager
from orchestration.notifications import NotificationManager
from orchestration.orchestrator import EndToEndOrchestrator


# ---------------------------------------------------------------------------
# Config Manager
# ---------------------------------------------------------------------------
class TestConfigManager:
    def test_sources_config_shape(self):
        cfg = ConfigManager().sources_config()
        for key in ("rss_feeds", "websites", "manual_input_enabled"):
            assert key in cfg

    def test_model_config_shape(self):
        models = ConfigManager().model_config()
        assert models["tts_backend"] in ("auto", "xtts", "piper", "espeak", "placeholder")
        assert "ffmpeg_binary" in models

    def test_category_config_shape(self):
        cats = ConfigManager().category_config()
        assert len(cats["categories"]) > 0
        assert cats["keyword_counts"]

    def test_effective_merges_all_families_without_secrets(self):
        eff = ConfigManager().effective()
        assert set(eff) == {"sources", "models", "categories", "distribution"}
        assert "youtube_configured" in eff["distribution"]
        # secrets must never appear in the auditable view
        blob = str(eff)
        assert "NEWSAPI_KEY" not in blob
        assert "TELEGRAM_BOT_TOKEN" not in blob


# ---------------------------------------------------------------------------
# Notification System
# ---------------------------------------------------------------------------
class TestNotificationManager:
    def test_send_degrades_to_log_only(self):
        nm = NotificationManager(enabled=True)
        assert not nm.email_configured()
        assert not nm.telegram_configured()
        assert nm.send("subject", "message") == {
            "logged": True, "email": False, "telegram": False,
        }

    def test_notify_error_and_run_report(self):
        nm = NotificationManager(enabled=True, on_error=True, on_complete=False)
        assert nm.notify_error("stage x", "boom")["logged"] is True
        res = nm.notify_run_report({"run_id": "r1"})
        assert res["email"] is False and res["telegram"] is False

    def test_notify_upload_status(self):
        nm = NotificationManager(enabled=True)
        res = nm.notify_upload_status(
            {"article_id": 3, "headline": "خبر", "youtube_status": "mock_published"}
        )
        assert res["logged"] is True


# ---------------------------------------------------------------------------
# Automation Engine (job queue)
# ---------------------------------------------------------------------------
class TestAutomationEngine:
    def test_submit_and_run_due(self, tmp_path):
        eng = AutomationEngine(db_path=tmp_path / "jobs.db", retry_delay=0)
        eng.register_handler("double", lambda p: {"doubled": p["x"] * 2})
        job_id = eng.submit("double", {"x": 21})

        results = eng.run_due()
        assert results[0]["status"] == "done"
        assert results[0]["result"]["doubled"] == 42
        assert eng.get_job(job_id)["status"] == "done"
        eng.close()

    def test_failure_requeues_then_fails(self, tmp_path):
        eng = AutomationEngine(db_path=tmp_path / "jobs.db", retry_delay=0)

        def boom(_payload):
            raise RuntimeError("nope")

        eng.register_handler("boom", boom)
        job_id = eng.submit("boom", max_attempts=2)

        first = eng.run_due()[0]
        assert first["status"] == "pending"      # re-queued for retry
        assert eng.get_job(job_id)["attempts"] == 1

        # force the deferred retry to be due
        with eng._cursor() as cur:
            cur.execute(
                "UPDATE automation_jobs SET scheduled_at = '2000-01-01T00:00:00' WHERE id = ?",
                (job_id,),
            )
        second = eng.run_due()[0]
        assert second["status"] == "failed"
        assert eng.get_job(job_id)["attempts"] == 2
        eng.close()

    def test_unknown_handler_fails_job(self, tmp_path):
        eng = AutomationEngine(db_path=tmp_path / "jobs.db")
        eng.submit("ghost")
        res = eng.run_due()[0]
        assert res["status"] == "failed"
        assert "no handler" in res["error"]
        eng.close()


# ---------------------------------------------------------------------------
# End-to-end orchestrator
# ---------------------------------------------------------------------------
class TestEndToEndOrchestrator:
    def _orchestrator(self, tmp_path, **kw):
        return EndToEndOrchestrator(
            raw_db_path=tmp_path / "raw.db",
            processed_db_path=tmp_path / "proc.db",
            db_path=tmp_path / "orch.db",
            max_retries=kw.get("max_retries", 0),
            retry_delay=0,
        )

    def test_process_stage_on_empty_dbs(self, tmp_path):
        orch = self._orchestrator(tmp_path)
        report = orch.run(stages=["process"])

        run = report.stages[0]
        assert run.stage == "process"
        assert run.status == "ok"
        assert run.summary["stages"]["2.1_preprocessing"]["processed"] == 0

        history = orch.status()
        assert any(r["stage"] == "process" and r["status"] == "ok" for r in history)
        orch.close()

    def test_unknown_stage_retried_then_isolated(self, tmp_path):
        orch = self._orchestrator(tmp_path, max_retries=1)
        report = orch.run(stages=["bogus", "process"])

        bad, good = report.stages
        assert bad.status == "error"
        assert bad.attempts == 2                 # 1 try + 1 retry
        assert good.status == "ok"               # failure isolation

        # retry_failed re-runs exactly the failed stage
        rep2 = orch.retry_failed()
        assert [s.stage for s in rep2.stages] == ["bogus"]
        orch.close()

    def test_retry_failed_with_no_failures(self, tmp_path):
        orch = self._orchestrator(tmp_path)
        assert orch.retry_failed().stages == []
        orch.close()
