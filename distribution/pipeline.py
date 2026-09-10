"""
distribution/pipeline.py
===========================
Orchestrator for Section 4 — chains stages 4.1 → 4.2 → 4.3 → 4.4
for each article, with job tracking and per-platform retry.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from data_acquisition.utils.logger import get_logger
from data_acquisition.utils.text_utils import safe_iso_now

from . import config as dc
from .distribution_db import DistributionDatabase
from . import youtube_uploader
from . import social_sharer
from . import playlist_manager
from . import analytics

logger = get_logger("distribution.pipeline")


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class DistributionStageReport:
    stage: str
    processed: int = 0
    ok: int = 0
    skipped_or_error: int = 0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DistributionReport:
    started_at: str = ""
    finished_at: str = ""
    stages: List[DistributionStageReport] = field(default_factory=list)
    outputs: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "stages": [
                {"stage": s.stage, "processed": s.processed,
                 "ok": s.ok, "skipped_or_error": s.skipped_or_error,
                 "details": s.details}
                for s in self.stages
            ],
            "outputs": self.outputs,
        }


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------

def _filter_eligible(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep articles that have a valid video file ready for distribution."""
    eligible = []
    for art in articles:
        video = art.get("subtitled_video") or art.get("video_path", "")
        if video and Path(video).exists():
            eligible.append(art)
    return eligible


def _build_work_items(
    articles: List[Dict[str, Any]],
    db: DistributionDatabase,
) -> List[Dict[str, Any]]:
    """Normalize article fields and create/reuse distribution_jobs rows."""
    items: List[Dict[str, Any]] = []
    for art in articles:
        aid = art.get("article_id", 0)
        existing = db.get_job(aid)
        if existing and existing.get("status") == "completed":
            continue  # already fully distributed

        item = {
            "article_id": aid,
            "headline": art.get("headline", "") or art.get("urdu_title", ""),
            "full_script": art.get("full_script", "") or art.get("urdu_text", ""),
            "category": art.get("category", "General"),
            "video_path": art.get("subtitled_video", "") or art.get("video_path", ""),
            "thumbnail_path": art.get("thumbnail_path", ""),
            "summary": art.get("summary", ""),
            "srt_path": art.get("srt_path", ""),
            "audio_path": art.get("processed_audio_path", "") or art.get("audio_path", ""),
        }

        # Create or reuse job
        if existing:
            item["job_id"] = existing.get("job_id", "")
            item["youtube_video_id"] = existing.get("youtube_video_id", "")
            item["youtube_url"] = existing.get("youtube_url", "")
            item["youtube_status"] = existing.get("youtube_status", "")
        else:
            item["job_id"] = str(uuid.uuid4())
            item["youtube_video_id"] = ""
            item["youtube_url"] = ""
            item["youtube_status"] = ""
            db.save_job({
                "article_id": aid,
                "job_id": item["job_id"],
                "status": "pending",
                "video_path": item["video_path"],
                "thumbnail_path": item["thumbnail_path"],
                "headline": item["headline"],
                "category": item["category"],
                "youtube_video_id": "",
                "youtube_url": "",
                "youtube_status": "",
                "facebook_post_id": "",
                "facebook_status": "",
                "twitter_post_id": "",
                "twitter_status": "",
                "telegram_message_id": "",
                "telegram_status": "",
                "whatsapp_status": "",
                "retry_count": 0,
                "created_at": safe_iso_now(),
                "updated_at": safe_iso_now(),
            })
        items.append(item)
    return items


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_distribution(
    articles: List[Dict[str, Any]],
    limit: Optional[int] = None,
    db: Optional[DistributionDatabase] = None,
) -> DistributionReport:
    """Run the full Section 4 distribution pipeline.

    1. Filter eligible articles (must have video file)
    2. For each article:
       a. Stage 4.1 — Upload to YouTube + set thumbnail
       b. Stage 4.3 — Assign to category playlist
       c. Stage 4.2 — Share to social media
       d. Stage 4.4 — Collect analytics snapshot
    3. Return DistributionReport
    """
    report = DistributionReport(started_at=safe_iso_now())
    _db = db or DistributionDatabase()
    _owned = db is None

    try:
        eligible = _filter_eligible(articles)
        if limit:
            eligible = eligible[:limit]

        logger.info("Distribution pipeline: %d eligible articles (limit=%s)",
                     len(eligible), limit)

        if not eligible:
            report.finished_at = safe_iso_now()
            return report

        items = _build_work_items(eligible, _db)

        # --- Stage 4.1: YouTube Upload ---
        yt_articles = []
        for item in items:
            if item.get("youtube_status") in ("published", "mock_published"):
                yt_articles.append({**item, "status": "already_published"})
                continue

            result = youtube_uploader.upload_video(
                article_id=item["article_id"],
                video_path=item["video_path"],
                headline=item["headline"],
                full_script=item["full_script"],
                category=item["category"],
                schedule_time=dc.YOUTUBE_SCHEDULE_TIME,
            )

            # Set thumbnail if upload succeeded
            if result["status"] in ("published", "mock_published") and item["thumbnail_path"]:
                youtube_uploader.set_thumbnail(result["video_id"], item["thumbnail_path"])

            item["youtube_video_id"] = result.get("video_id", "")
            item["youtube_url"] = result.get("url", "")
            item["youtube_status"] = result["status"]
            yt_articles.append(item)

            # Update job
            _db.save_job({
                "article_id": item["article_id"],
                "job_id": item["job_id"],
                "status": "youtube_done" if result["status"] in ("published", "mock_published") else "youtube_failed",
                "youtube_video_id": item["youtube_video_id"],
                "youtube_url": item["youtube_url"],
                "youtube_status": item["youtube_status"],
                "headline": item["headline"],
                "category": item["category"],
                "video_path": item["video_path"],
                "thumbnail_path": item["thumbnail_path"],
                "updated_at": safe_iso_now(),
            })

        yt_ok = sum(1 for a in yt_articles if a.get("youtube_status") in ("published", "mock_published", "already_published"))
        report.stages.append(DistributionStageReport(
            stage="4.1_youtube_upload",
            processed=len(yt_articles), ok=yt_ok,
            skipped_or_error=len(yt_articles) - yt_ok,
        ))

        # --- Stage 4.3: Playlist Management ---
        pl_items = [i for i in items if i.get("youtube_video_id")]
        pl_results = playlist_manager.run(pl_items, db=_db)
        pl_ok = sum(1 for r in pl_results if r["status"] in ("ok", "mock_ok"))
        report.stages.append(DistributionStageReport(
            stage="4.3_playlist_management",
            processed=len(pl_results), ok=pl_ok,
            skipped_or_error=len(pl_results) - pl_ok,
        ))

        # --- Stage 4.2: Social Media Sharing ---
        social_items = []
        for item in items:
            social_items.append({
                "article_id": item["article_id"],
                "headline": item["headline"],
                "full_script": item["full_script"],
                "summary": item.get("summary", ""),
                "category": item["category"],
                "youtube_url": item.get("youtube_url", ""),
            })
        social_results = social_sharer.run(social_items)

        # Merge social results back into items and update DB
        # DB column mapping: telegram -> telegram_message_id (not telegram_post_id)
        _plat_col_map = {
            "facebook": ("facebook_post_id", "facebook_status"),
            "twitter":  ("twitter_post_id", "twitter_status"),
            "telegram": ("telegram_message_id", "telegram_status"),
            "whatsapp": (None, "whatsapp_status"),
        }
        for i, sr in enumerate(social_results):
            item = items[i] if i < len(items) else None
            if item:
                job_update = {
                    "article_id": item["article_id"],
                    "job_id": item["job_id"],
                    "status": "social_done",
                    "headline": item["headline"],
                    "category": item["category"],
                    "video_path": item["video_path"],
                    "thumbnail_path": item["thumbnail_path"],
                    "youtube_video_id": item.get("youtube_video_id", ""),
                    "youtube_url": item.get("youtube_url", ""),
                    "youtube_status": item.get("youtube_status", ""),
                    "updated_at": safe_iso_now(),
                }
                for plat, (id_col, status_col) in _plat_col_map.items():
                    if id_col:
                        job_update[id_col] = sr.get(f"{plat}_post_id", "")
                    job_update[status_col] = sr.get(f"{plat}_status", "")
                _db.save_job(job_update)

        social_ok = sum(
            1 for sr in social_results
            if any(sr.get(f"{p}_status", "") in ("posted", "mock_posted")
                   for p in ("facebook", "twitter", "telegram", "whatsapp"))
        )
        report.stages.append(DistributionStageReport(
            stage="4.2_social_sharing",
            processed=len(social_results), ok=social_ok,
            skipped_or_error=len(social_results) - social_ok,
        ))

        # --- Stage 4.4: Analytics ---
        analytics_items = [
            {"article_id": i["article_id"], "youtube_video_id": i.get("youtube_video_id", "")}
            for i in items if i.get("youtube_video_id")
        ]
        analytics_result = analytics.run(analytics_items, db=_db)
        report.stages.append(DistributionStageReport(
            stage="4.4_analytics",
            processed=analytics_result.get("collected", 0),
            ok=analytics_result.get("collected", 0),
            skipped_or_error=0,
            details={"total_views": analytics_result.get("total_views", 0)},
        ))

        # Mark completed
        for item in items:
            _db.save_job({
                "article_id": item["article_id"],
                "job_id": item["job_id"],
                "status": "completed",
                "headline": item["headline"],
                "category": item["category"],
                "video_path": item["video_path"],
                "thumbnail_path": item["thumbnail_path"],
                "youtube_video_id": item.get("youtube_video_id", ""),
                "youtube_url": item.get("youtube_url", ""),
                "youtube_status": item.get("youtube_status", ""),
                "updated_at": safe_iso_now(),
            })

        report.outputs = items
        report.finished_at = safe_iso_now()
        logger.info("Distribution pipeline complete: %d articles processed", len(items))
    finally:
        if _owned:
            _db.close()

    return report


# ---------------------------------------------------------------------------
# Retry failed jobs
# ---------------------------------------------------------------------------

def retry_failed_jobs(
    max_retries: int = 0,
    db: Optional[DistributionDatabase] = None,
) -> DistributionReport:
    """Re-attempt distribution for failed jobs."""
    max_retries = max_retries or dc.SCHEDULER_MAX_RETRIES
    _db = db or DistributionDatabase()
    _owned = db is None

    try:
        failed = _db.get_failed_jobs(max_retries=max_retries)
        if not failed:
            logger.info("No failed jobs to retry (max_retries=%d)", max_retries)
            return DistributionReport(started_at=safe_iso_now(), finished_at=safe_iso_now())

        # Build article-like dicts from failed jobs
        articles = []
        for job in failed:
            job["retry_count"] = job.get("retry_count", 0) + 1
            _db.save_job({**job, "status": "retry_pending", "updated_at": safe_iso_now()})
            articles.append({
                "article_id": job["article_id"],
                "headline": job.get("headline", ""),
                "full_script": "",
                "category": job.get("category", ""),
                "subtitled_video": job.get("video_path", ""),
                "thumbnail_path": job.get("thumbnail_path", ""),
            })

        logger.info("Retrying %d failed distribution jobs", len(articles))
        return run_distribution(articles, db=_db)
    finally:
        if _owned:
            _db.close()
