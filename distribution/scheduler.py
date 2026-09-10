"""
distribution/scheduler.py
============================
Stage 4.5 — lightweight continuous scheduler that periodically runs
the full Section 2 → 3 → 4 pipeline (or just Section 4 distribution).

Uses Python's built-in ``sched`` + ``threading`` modules — no external
scheduler dependency required.

Usage
-----
    python -m distribution.scheduler                  # run continuously
    python -m distribution.scheduler --once           # run once and exit
    python -m distribution.scheduler --interval 4     # every 4 hours
"""

from __future__ import annotations

import argparse
import sched
import sys
import threading
import time
from datetime import datetime
from typing import Optional

from data_acquisition.utils.logger import get_logger

from . import config as dc
from .pipeline import run_distribution, retry_failed_jobs
from .distribution_db import DistributionDatabase

logger = get_logger("distribution.scheduler")


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class DistributionScheduler:
    """Periodically runs distribution for pending articles."""

    def __init__(
        self,
        interval_hours: float = 0,
        daily_time: str = "",
        max_retries: int = 0,
    ):
        self.interval_hours = interval_hours or dc.SCHEDULER_INTERVAL_HOURS
        self.daily_time = daily_time or dc.SCHEDULER_DAILY_TIME
        self.max_retries = max_retries or dc.SCHEDULER_MAX_RETRIES
        self._stop_event = threading.Event()
        self._scheduler = sched.scheduler(time.time, time.sleep)

    def _interval_seconds(self) -> float:
        return self.interval_hours * 3600

    def _seconds_until_daily(self) -> float:
        """Calculate seconds until the next daily run time (HH:MM)."""
        if not self.daily_time:
            return self._interval_seconds()
        try:
            hour, minute = map(int, self.daily_time.split(":"))
        except (ValueError, AttributeError):
            return self._interval_seconds()

        now = datetime.utcnow()
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            # Next day
            from datetime import timedelta
            target += timedelta(days=1)
        return (target - now).total_seconds()

    def run_once(self) -> None:
        """Execute one distribution cycle."""
        logger.info("Scheduler: starting distribution cycle at %s",
                     datetime.utcnow().isoformat())
        try:
            db = DistributionDatabase()
            # Get pending articles
            pending = db.get_pending_articles()
            if pending:
                # Convert DB rows to article-like dicts
                articles = [
                    {
                        "article_id": j["article_id"],
                        "headline": j.get("headline", ""),
                        "full_script": "",
                        "category": j.get("category", ""),
                        "subtitled_video": j.get("video_path", ""),
                        "thumbnail_path": j.get("thumbnail_path", ""),
                    }
                    for j in pending
                ]
                logger.info("Found %d pending articles for distribution", len(articles))
                report = run_distribution(articles, db=db)
                logger.info(
                    "Distribution cycle complete: %d stages, %d outputs",
                    len(report.stages), len(report.outputs),
                )
            else:
                logger.info("No pending articles for distribution")

            # Retry failed jobs
            failed = db.get_failed_jobs(max_retries=self.max_retries)
            if failed:
                logger.info("Retrying %d failed jobs", len(failed))
                retry_failed_jobs(max_retries=self.max_retries, db=db)

            db.close()
        except Exception as exc:
            logger.error("Scheduler cycle failed: %s", exc, exc_info=True)

    def _schedule_next(self) -> None:
        """Schedule the next run."""
        if self.daily_time:
            delay = self._seconds_until_daily()
        else:
            delay = self._interval_seconds()

        logger.info("Next distribution run in %.1f hours (%.0f seconds)",
                     delay / 3600, delay)

        if not self._stop_event.is_set():
            self._scheduler.enter(delay, 1, self._tick)

    def _tick(self) -> None:
        """Single tick: run + schedule next."""
        if self._stop_event.is_set():
            return
        self.run_once()
        self._schedule_next()

    def start(self, blocking: bool = True) -> None:
        """Start the scheduler loop.

        If ``blocking=True``, runs in the current thread until stopped.
        If ``blocking=False``, starts in a daemon thread.
        """
        if blocking:
            logger.info("Scheduler starting (blocking, interval=%.1fh)", self.interval_hours)
            self._tick()
            self._scheduler.run()
        else:
            t = threading.Thread(target=self._run_thread, daemon=True)
            t.start()
            logger.info("Scheduler started in background thread")

    def _run_thread(self) -> None:
        self._tick()
        self._scheduler.run()

    def stop(self) -> None:
        """Signal the scheduler to stop after the current cycle."""
        self._stop_event.set()
        logger.info("Scheduler stop signal sent")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Section 4 Distribution Scheduler — runs distribution "
                    "cycles at configurable intervals."
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run one distribution cycle and exit",
    )
    parser.add_argument(
        "--interval", type=float, default=dc.SCHEDULER_INTERVAL_HOURS,
        help=f"Hours between cycles (default: {dc.SCHEDULER_INTERVAL_HOURS})",
    )
    parser.add_argument(
        "--daily-time", type=str, default=dc.SCHEDULER_DAILY_TIME,
        help="Run daily at HH:MM UTC (overrides --interval)",
    )
    parser.add_argument(
        "--max-retries", type=int, default=dc.SCHEDULER_MAX_RETRIES,
        help=f"Max retry attempts for failed jobs (default: {dc.SCHEDULER_MAX_RETRIES})",
    )
    args = parser.parse_args()

    scheduler = DistributionScheduler(
        interval_hours=args.interval,
        daily_time=args.daily_time,
        max_retries=args.max_retries,
    )

    if args.once:
        scheduler.run_once()
        return 0

    try:
        scheduler.start(blocking=True)
    except KeyboardInterrupt:
        scheduler.stop()
        logger.info("Scheduler interrupted by user")

    return 0


if __name__ == "__main__":
    sys.exit(main())
