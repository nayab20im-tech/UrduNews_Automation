"""
distribution/analytics.py
============================
Stage 4.4 — fetch YouTube video statistics, store periodic snapshots,
and generate daily / weekly JSON+CSV reports.

Reports are written to ``DISTRIBUTION_REPORTS_DIR``.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from data_acquisition.utils.logger import get_logger
from data_acquisition.utils.text_utils import safe_iso_now

from . import config as dc
from .distribution_db import DistributionDatabase

logger = get_logger("distribution.analytics")


# ---------------------------------------------------------------------------
# YouTube stats fetch
# ---------------------------------------------------------------------------

def fetch_youtube_stats(video_id: str) -> Optional[Dict[str, Any]]:
    """Fetch view/like/comment counts for a YouTube video."""
    if dc.DISTRIBUTION_MOCK_MODE:
        return {
            "video_id": video_id,
            "views": 150, "likes": 12, "comments": 3,
            "shares": 0, "watch_time": 0.0,
            "engagement_rate": 0.10,
        }

    from .youtube_uploader import _get_credentials, _build_youtube_service

    creds = _get_credentials()
    if creds is None:
        return None
    service = _build_youtube_service(creds)
    if service is None:
        return None

    try:
        resp = service.videos().list(
            part="statistics,snippet",
            id=video_id,
        ).execute()
        items = resp.get("items", [])
        if not items:
            return None
        stats = items[0].get("statistics", {})
        views = int(stats.get("viewCount", 0))
        likes = int(stats.get("likeCount", 0))
        comments = int(stats.get("commentCount", 0))
        engagement = (likes + comments) / max(views, 1)
        return {
            "video_id": video_id,
            "views": views,
            "likes": likes,
            "comments": comments,
            "shares": 0,
            "watch_time": 0.0,
            "engagement_rate": round(engagement, 4),
        }
    except Exception as exc:
        logger.warning("Failed to fetch stats for %s: %s", video_id, exc)
        return None


def store_snapshot(
    db: DistributionDatabase,
    video_id: str,
    platform: str = "youtube",
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """Fetch stats (if not provided) and store a snapshot in the DB."""
    if metrics is None:
        metrics = fetch_youtube_stats(video_id)
    if metrics is None:
        logger.info("No metrics available for %s; skipping snapshot", video_id)
        return

    row = {
        "video_id": video_id,
        "platform": platform,
        "snapshot_date": datetime.utcnow().strftime("%Y-%m-%d"),
        "views": metrics.get("views", 0),
        "likes": metrics.get("likes", 0),
        "comments": metrics.get("comments", 0),
        "shares": metrics.get("shares", 0),
        "watch_time": metrics.get("watch_time", 0.0),
        "engagement_rate": metrics.get("engagement_rate", 0.0),
        "raw_json": json.dumps(metrics, ensure_ascii=False),
    }
    db.save_snapshot(row)
    logger.info("Stored analytics snapshot: %s/%s on %s", video_id, platform, row["snapshot_date"])


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _ensure_reports_dir() -> Path:
    path = Path(dc.REPORTS_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def generate_daily_report(
    db: DistributionDatabase,
    date: str = "",
) -> Dict[str, Any]:
    """Generate a daily distribution report as JSON + CSV.

    Returns the report dict; also writes files to the reports directory.
    """
    if not date:
        date = datetime.utcnow().strftime("%Y-%m-%d")

    jobs = db.get_all_jobs()
    snapshots = db.get_snapshots(start_date=date, end_date=date)

    # Aggregate
    total_videos = len(jobs)
    published = sum(1 for j in jobs if j.get("youtube_status") == "published")
    failed = sum(1 for j in jobs if j.get("status") == "failed")
    total_views = sum(s.get("views", 0) for s in snapshots)
    total_likes = sum(s.get("likes", 0) for s in snapshots)
    total_comments = sum(s.get("comments", 0) for s in snapshots)

    # Best performer
    best = max(snapshots, key=lambda s: s.get("views", 0)) if snapshots else {}

    report = {
        "report_type": "daily",
        "date": date,
        "generated_at": safe_iso_now(),
        "summary": {
            "total_videos": total_videos,
            "published": published,
            "failed": failed,
            "total_views": total_views,
            "total_likes": total_likes,
            "total_comments": total_comments,
        },
        "best_performer": {
            "video_id": best.get("video_id", ""),
            "views": best.get("views", 0),
        },
        "snapshots": snapshots,
    }

    # Write JSON
    reports_dir = _ensure_reports_dir()
    json_path = reports_dir / f"daily_report_{date}.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Write CSV
    csv_path = reports_dir / f"daily_report_{date}.csv"
    _write_snapshots_csv(snapshots, csv_path)

    logger.info("Daily report generated: %s (%d snapshots)", json_path.name, len(snapshots))
    return report


def generate_weekly_report(
    db: DistributionDatabase,
    end_date: str = "",
) -> Dict[str, Any]:
    """Generate a weekly report covering the 7 days before end_date."""
    if not end_date:
        end_date = datetime.utcnow().strftime("%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=6)
    start_date = start_dt.strftime("%Y-%m-%d")

    snapshots = db.get_snapshots(start_date=start_date, end_date=end_date)
    jobs = db.get_all_jobs()

    total_views = sum(s.get("views", 0) for s in snapshots)
    total_likes = sum(s.get("likes", 0) for s in snapshots)
    total_comments = sum(s.get("comments", 0) for s in snapshots)

    # Per-day breakdown
    daily: Dict[str, Dict[str, int]] = {}
    for s in snapshots:
        d = s.get("snapshot_date", "")
        if d not in daily:
            daily[d] = {"views": 0, "likes": 0, "comments": 0, "videos": 0}
        daily[d]["views"] += s.get("views", 0)
        daily[d]["likes"] += s.get("likes", 0)
        daily[d]["comments"] += s.get("comments", 0)
        daily[d]["videos"] += 1

    report = {
        "report_type": "weekly",
        "start_date": start_date,
        "end_date": end_date,
        "generated_at": safe_iso_now(),
        "summary": {
            "total_snapshots": len(snapshots),
            "total_views": total_views,
            "total_likes": total_likes,
            "total_comments": total_comments,
        },
        "daily_breakdown": daily,
    }

    reports_dir = _ensure_reports_dir()
    json_path = reports_dir / f"weekly_report_{end_date}.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Weekly report generated: %s", json_path.name)
    return report


def _write_snapshots_csv(snapshots: List[Dict[str, Any]], path: Path) -> None:
    """Write a list of snapshot dicts to a CSV file."""
    if not snapshots:
        path.write_text("video_id,platform,snapshot_date,views,likes,comments,shares,engagement_rate\n",
                        encoding="utf-8")
        return

    fieldnames = ["video_id", "platform", "snapshot_date", "views", "likes",
                  "comments", "shares", "engagement_rate"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(snapshots)


# ---------------------------------------------------------------------------
# Batch: collect stats for all published videos
# ---------------------------------------------------------------------------

def collect_all_stats(db: DistributionDatabase) -> int:
    """Fetch and store snapshots for every published video. Returns count."""
    jobs = db.get_all_jobs()
    count = 0
    for job in jobs:
        video_id = job.get("youtube_video_id", "")
        if video_id and job.get("youtube_status") == "published":
            store_snapshot(db, video_id)
            count += 1
    logger.info("Collected analytics for %d published videos", count)
    return count


def run(
    articles: List[Dict[str, Any]],
    db: Optional[DistributionDatabase] = None,
) -> Dict[str, Any]:
    """Run analytics collection and generate a daily report.

    ``articles`` is used to identify which videos to collect stats for.
    Returns a summary dict.
    """
    _db = db or DistributionDatabase()
    _owned = db is None

    try:
        # Collect stats for articles with YouTube video IDs
        collected = 0
        for art in articles:
            video_id = art.get("youtube_video_id", "")
            if video_id:
                store_snapshot(_db, video_id)
                collected += 1

        # Generate daily report
        report = generate_daily_report(_db)

        return {
            "collected": collected,
            "report_date": report.get("date", ""),
            "total_views": report.get("summary", {}).get("total_views", 0),
            "status": "ok",
        }
    finally:
        if _owned:
            _db.close()
