"""
monitoring/dashboard_data.py
============================
Backend-agnostic data layer for Section 7 — Monitoring & Dashboard.

Aggregates the four dashboard views from the architecture diagram on top
of the Section 6 storage layer:

* **Pipeline status**    -- stage counters + recent orchestration runs
* **Latest news preview**-- most recent processed articles (with Urdu titles)
* **Upload status**      -- distribution jobs (YouTube/social progress)
* **Analytics overview** -- aggregated ``analytics_snapshots`` per platform

Every section is failure-isolated (a broken table returns an ``error``
entry instead of taking the dashboard down), matching the pipeline's
per-stage error-handling convention.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from data_acquisition.utils.logger import get_logger
from data_acquisition.utils.text_utils import safe_iso_now

from storage.storage_manager import StorageManager

logger = get_logger("monitoring.dashboard_data")


class DashboardData:
    """Read-only aggregation facade over :class:`StorageManager`."""

    def __init__(self, storage: Optional[StorageManager] = None):
        self.storage = storage or StorageManager()
        self._owns_storage = storage is None

    # ------------------------------------------------------------------ #
    # Pipeline status
    # ------------------------------------------------------------------ #
    def pipeline_status(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        try:
            out["raw"] = self.storage.raw.get_stats()
        except Exception as exc:  # noqa: BLE001 - dashboard must survive
            out["raw"] = {"error": str(exc)}

        try:
            dataset = self.storage.processed.build_final_dataset()
            out["processed"] = {
                "articles": len(dataset),
                "translated": sum(1 for r in dataset if r.get("translated")),
                "classified": sum(1 for r in dataset if r.get("category")),
                "with_tts_audio": sum(1 for r in dataset if r.get("tts_audio_path")),
                "with_processed_audio": sum(
                    1 for r in dataset if r.get("processed_audio_path")),
                "by_category": _count(dataset, "category"),
            }
        except Exception as exc:  # noqa: BLE001
            out["processed"] = {"error": str(exc)}

        try:
            jobs = self.storage.distribution.get_all_jobs()
            out["distribution"] = {
                "jobs": len(jobs),
                "by_status": _count(jobs, "status"),
                "published": sum(
                    1 for j in jobs if j.get("youtube_status") == "published"),
            }
        except Exception as exc:  # noqa: BLE001
            out["distribution"] = {"error": str(exc)}

        try:
            out["media"] = self.storage.media.stats()
        except Exception as exc:  # noqa: BLE001
            out["media"] = {"error": str(exc)}

        out["recent_runs"] = self.recent_runs()
        return out

    def recent_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Recent orchestration history (Part 5) -- empty when none yet."""
        try:
            from orchestration.orchestrator import EndToEndOrchestrator

            db_path = getattr(self.storage, "db_path", None)
            orch = EndToEndOrchestrator(db_path=db_path) if db_path \
                else EndToEndOrchestrator()
            try:
                return orch.status(limit=limit)
            finally:
                orch.close()
        except Exception as exc:  # noqa: BLE001
            return [{"error": str(exc)}]

    # ------------------------------------------------------------------ #
    # Latest news preview
    # ------------------------------------------------------------------ #
    def latest_news(self, limit: int = 10) -> List[Dict[str, Any]]:
        try:
            dataset = self.storage.processed.build_final_dataset()
            video_by_article: Dict[int, Dict[str, Any]] = {}
            try:
                for job in self.storage.distribution.get_all_jobs():
                    video_by_article[job.get("article_id")] = job
            except Exception:  # noqa: BLE001 - jobs are optional enrichment
                pass

            items: List[Dict[str, Any]] = []
            for row in dataset[-limit:][::-1]:
                article_id = row.get("article_id")
                job = video_by_article.get(article_id, {})
                items.append({
                    "article_id": article_id,
                    "title": row.get("title"),
                    "urdu_title": row.get("urdu_title"),
                    "category": row.get("category"),
                    "language": row.get("language"),
                    "published_date": row.get("published_date"),
                    "bias_label": row.get("bias_label"),
                    "verification_verdict": row.get("verification_verdict"),
                    "has_tts_audio": bool(row.get("tts_audio_path")),
                    "video_path": job.get("video_path"),
                    "youtube_status": job.get("youtube_status"),
                })
            return items
        except Exception as exc:  # noqa: BLE001
            return [{"error": str(exc)}]

    # ------------------------------------------------------------------ #
    # Upload status
    # ------------------------------------------------------------------ #
    def upload_status(self) -> Dict[str, Any]:
        try:
            jobs = self.storage.distribution.get_all_jobs()
            recent = sorted(
                jobs,
                key=lambda j: (j.get("updated_at") or "", j.get("article_id") or 0),
                reverse=True,
            )[:10]
            return {
                "total_jobs": len(jobs),
                "by_status": _count(jobs, "status"),
                "by_youtube_status": _count(jobs, "youtube_status"),
                "recent": [
                    {k: j.get(k) for k in (
                        "article_id", "headline", "video_path", "status",
                        "youtube_status", "retry_count", "updated_at")}
                    for j in recent
                ],
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    # ------------------------------------------------------------------ #
    # Analytics overview
    # ------------------------------------------------------------------ #
    def analytics_overview(self) -> Dict[str, Any]:
        try:
            snapshots = self.storage.distribution.get_snapshots()
            if not snapshots:
                return {"totals": {}, "by_platform": {}, "top_videos": []}

            # Latest snapshot per (video, platform) so totals aren't
            # inflated by repeated daily snapshots of the same video.
            latest: Dict[tuple, Dict[str, Any]] = {}
            for snap in snapshots:
                key = (snap.get("video_id"), snap.get("platform"))
                cur = latest.get(key)
                if cur is None or (snap.get("snapshot_date") or "") >= \
                        (cur.get("snapshot_date") or ""):
                    latest[key] = snap

            by_platform: Dict[str, Dict[str, int]] = {}
            totals = {"videos": 0, "views": 0, "likes": 0, "comments": 0,
                      "shares": 0}
            for snap in latest.values():
                platform = snap.get("platform") or "unknown"
                agg = by_platform.setdefault(
                    platform, {"videos": 0, "views": 0, "likes": 0,
                               "comments": 0, "shares": 0})
                for field in ("views", "likes", "comments", "shares"):
                    value = int(snap.get(field) or 0)
                    agg[field] += value
                    totals[field] += value
                agg["videos"] += 1
                totals["videos"] += 1

            top = sorted(
                latest.values(),
                key=lambda s: int(s.get("views") or 0),
                reverse=True,
            )[:5]
            return {
                "totals": totals,
                "by_platform": by_platform,
                "top_videos": [
                    {k: s.get(k) for k in (
                        "video_id", "platform", "views", "likes",
                        "comments", "shares", "snapshot_date")}
                    for s in top
                ],
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    # ------------------------------------------------------------------ #
    # Storage health + logs + system check
    # ------------------------------------------------------------------ #
    def storage_health(self) -> Dict[str, Any]:
        try:
            return self.storage.health()
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    def recent_logs(self, limit: int = 20) -> List[Dict[str, Any]]:
        try:
            return self.storage.logs.get_logs(limit=limit)
        except Exception as exc:  # noqa: BLE001
            return [{"error": str(exc)}]

    def system_check(self) -> List[Dict[str, Any]]:
        from monitoring.system_check import collect_system_check

        return collect_system_check()

    # ------------------------------------------------------------------ #
    # Full report (what the dashboard renders / CLI exports)
    # ------------------------------------------------------------------ #
    def full_report(self, news_limit: int = 10,
                    log_limit: int = 20) -> Dict[str, Any]:
        return {
            "generated_at": safe_iso_now(),
            "pipeline_status": self.pipeline_status(),
            "latest_news": self.latest_news(limit=news_limit),
            "upload_status": self.upload_status(),
            "analytics_overview": self.analytics_overview(),
            "storage_health": self.storage_health(),
            "recent_logs": self.recent_logs(limit=log_limit),
            "system_check": self.system_check(),
        }

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        if self._owns_storage:
            self.storage.close()

    def __enter__(self) -> "DashboardData":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _count(rows: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return counts
