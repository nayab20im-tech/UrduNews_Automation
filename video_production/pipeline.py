"""
video_production/pipeline.py
===============================
Section 3 orchestrator — chains Stages 3.1 → 3.6:

    3.1  Avatar Generation
    3.2  Lip Synchronization
    3.3  Background & Visuals
    3.4  Video Composition
    3.5  Subtitle Generation + Burn-in
    3.6  Thumbnail Generation

Accepts a list of article dicts (produced by
``ProcessedNewsDatabase.build_final_dataset()``) and runs each stage
sequentially.  Each stage's outputs are collected and forwarded as
inputs to the next stage.

Usage
-----
    from video_production.pipeline import run_video_production

    articles = processed_db.build_final_dataset()
    report = run_video_production(articles)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from data_acquisition.utils.logger import get_logger
from data_acquisition.utils.text_utils import safe_iso_now

from . import config as vc
from .lip_sync import LipSyncEngine
from .visual_generator import VisualGenerator
from .video_composer import VideoComposer
from .subtitle_generator import SubtitleGenerator
from .thumbnail_generator import ThumbnailGenerator

logger = get_logger("video_production.pipeline")


@dataclass
class VideoStageReport:
    stage: str
    processed: int = 0
    ok: int = 0
    skipped_or_error: int = 0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoProductionReport:
    started_at: str = ""
    finished_at: str = ""
    stages: List[VideoStageReport] = field(default_factory=list)
    outputs: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "stages": [
                {
                    "stage": s.stage,
                    "processed": s.processed,
                    "ok": s.ok,
                    "skipped_or_error": s.skipped_or_error,
                    "details": s.details,
                }
                for s in self.stages
            ],
            "outputs": self.outputs,
        }


def run_video_production(
    articles: List[Dict[str, Any]],
    limit: Optional[int] = None,
) -> VideoProductionReport:
    """Run the full Section 3 video production pipeline.

    Parameters
    ----------
    articles : list of dict
        Each dict should contain at minimum:
        - ``article_id`` (int)
        - ``headline`` (str, Urdu)
        - ``full_script`` or ``urdu_text`` (str, Urdu)
        - ``category`` (str)
        - ``processed_audio_path`` or ``tts_audio_path`` (str)
        - ``audio_duration_sec`` or ``tts_duration_sec`` (float)

        Optional:
        - ``news_image`` — path to a news-related image

    limit : int, optional
        Process only the first *limit* articles.

    Returns
    -------
    VideoProductionReport with per-stage stats and per-article outputs.
    """
    report = VideoProductionReport(started_at=safe_iso_now())

    if limit and limit > 0:
        articles = articles[:limit]

    # Filter to articles that have audio (required for video production)
    eligible = _filter_eligible(articles)
    if not eligible:
        logger.warning("No eligible articles for video production (missing audio)")
        report.finished_at = safe_iso_now()
        return report

    logger.info("Video production: %d eligible articles", len(eligible))

    # Build working data with normalized field names
    work = _build_work_items(eligible)

    # ---- 3.1 Avatar Generation (Gemini) ----------------------------------
    logger.info("=== Stage 3.1: AI Avatar Generation (Gemini) ===")
    from .gemini_generator import GeminiVideoGenerator
    avatar_gen = GeminiVideoGenerator()
    avatar_results = avatar_gen.run([
        {"article_id": w["article_id"],
         "audio_path": w["audio_path"],
         "audio_duration": w["audio_duration"]}
        for w in work
    ])
    avatar_map = {r.article_id: r for r in avatar_results}
    for w in work:
        ar = avatar_map.get(w["article_id"])
        if ar and ar.status == "ok":
            w["avatar_video"] = ar.video_path

    ok_count = sum(1 for r in avatar_results if r.status == "ok")
    report.stages.append(VideoStageReport(
        stage="3.1_avatar_generation",
        processed=len(avatar_results), ok=ok_count,
        skipped_or_error=len(avatar_results) - ok_count,
        details={"by_backend": _count_by(avatar_results, "backend_used")},
    ))

    # ---- 3.2 Lip Synchronization ------------------------------------------
    logger.info("=== Stage 3.2: Lip Synchronization ===")
    lip_sync = LipSyncEngine()
    lip_results = lip_sync.run([
        {"article_id": w["article_id"],
         "avatar_video": w.get("avatar_video", ""),
         "audio_path": w["audio_path"],
         "audio_duration": w["audio_duration"]}
        for w in work
    ])
    lip_map = {r.article_id: r for r in lip_results}
    for w in work:
        lr = lip_map.get(w["article_id"])
        if lr and lr.status == "ok":
            w["lip_sync_video"] = lr.video_path

    ok_count = sum(1 for r in lip_results if r.status == "ok")
    report.stages.append(VideoStageReport(
        stage="3.2_lip_sync",
        processed=len(lip_results), ok=ok_count,
        skipped_or_error=len(lip_results) - ok_count,
        details={"by_backend": _count_by(lip_results, "backend_used")},
    ))

    # ---- 3.3 Background & Visuals ----------------------------------------
    logger.info("=== Stage 3.3: Background & Visuals ===")
    visual_gen = VisualGenerator()
    visual_results = visual_gen.run([
        {"article_id": w["article_id"],
         "headline": w["headline"],
         "category": w["category"],
         "news_image": w.get("news_image", "")}
        for w in work
    ])
    vis_map = {r.article_id: r for r in visual_results}
    for w in work:
        vr = vis_map.get(w["article_id"])
        if vr and vr.status == "ok":
            w["background_image"] = vr.background_path
            w["lower_third_image"] = vr.lower_third_path
            w["ticker_image"] = vr.ticker_path
            if vr.news_image_path:
                w["news_image"] = vr.news_image_path

    ok_count = sum(1 for r in visual_results if r.status == "ok")
    report.stages.append(VideoStageReport(
        stage="3.3_visuals",
        processed=len(visual_results), ok=ok_count,
        skipped_or_error=len(visual_results) - ok_count,
    ))

    # ---- 3.4 Video Composition -------------------------------------------
    logger.info("=== Stage 3.4: Video Composition ===")
    composer = VideoComposer()
    comp_results = composer.run([
        {"article_id": w["article_id"],
         "lip_sync_video": w.get("lip_sync_video", ""),
         "audio_path": w["audio_path"],
         "audio_duration": w["audio_duration"],
         "background_image": w.get("background_image", ""),
         "lower_third_image": w.get("lower_third_image", ""),
         "ticker_image": w.get("ticker_image", ""),
         "news_image": w.get("news_image", ""),
         "timeline_segments": w.get("timeline_segments")}
        for w in work
    ])
    comp_map = {r.article_id: r for r in comp_results}
    for w in work:
        cr = comp_map.get(w["article_id"])
        if cr and cr.status == "ok":
            w["composed_video"] = cr.video_path

    ok_count = sum(1 for r in comp_results if r.status == "ok")
    report.stages.append(VideoStageReport(
        stage="3.4_composition",
        processed=len(comp_results), ok=ok_count,
        skipped_or_error=len(comp_results) - ok_count,
    ))

    # ---- 3.5 Subtitle Generation + Burn-in --------------------------------
    logger.info("=== Stage 3.5: Subtitle Generation ===")
    subtitle_gen = SubtitleGenerator()
    sub_results = subtitle_gen.run([
        {"article_id": w["article_id"],
         "urdu_script": w["urdu_script"],
         "video_path": w.get("composed_video", ""),
         "audio_duration": w["audio_duration"]}
        for w in work
    ])
    sub_map = {r.article_id: r for r in sub_results}
    for w in work:
        sr = sub_map.get(w["article_id"])
        if sr and sr.status == "ok":
            w["srt_path"] = sr.srt_path
            w["subtitled_video"] = sr.subtitled_video_path

    ok_count = sum(1 for r in sub_results if r.status == "ok")
    report.stages.append(VideoStageReport(
        stage="3.5_subtitles",
        processed=len(sub_results), ok=ok_count,
        skipped_or_error=len(sub_results) - ok_count,
        details={"total_subtitles": sum(r.subtitle_count for r in sub_results)},
    ))

    # ---- 3.6 Thumbnail Generation ----------------------------------------
    logger.info("=== Stage 3.6: Thumbnail Generation ===")
    thumb_gen = ThumbnailGenerator()
    thumb_results = thumb_gen.run([
        {"article_id": w["article_id"],
         "headline": w["headline"],
         "category": w["category"],
         "presenter_video": w.get("lip_sync_video", "") or w.get("avatar_video", ""),
         "news_image": w.get("news_image", "")}
        for w in work
    ])
    thumb_map = {r.article_id: r for r in thumb_results}
    for w in work:
        tr = thumb_map.get(w["article_id"])
        if tr and tr.status == "ok":
            w["thumbnail_path"] = tr.thumbnail_path

    ok_count = sum(1 for r in thumb_results if r.status == "ok")
    report.stages.append(VideoStageReport(
        stage="3.6_thumbnail",
        processed=len(thumb_results), ok=ok_count,
        skipped_or_error=len(thumb_results) - ok_count,
    ))

    # ---- Collect final outputs -------------------------------------------
    for w in work:
        final_video = w.get("subtitled_video", "") or w.get("composed_video", "")
        report.outputs.append({
            "article_id": w["article_id"],
            "headline": w["headline"],
            "category": w["category"],
            "avatar_video": w.get("avatar_video", ""),
            "lip_sync_video": w.get("lip_sync_video", ""),
            "composed_video": w.get("composed_video", ""),
            "subtitled_video": final_video,
            "srt_path": w.get("srt_path", ""),
            "thumbnail_path": w.get("thumbnail_path", ""),
            "audio_path": w["audio_path"],
        })

    report.finished_at = safe_iso_now()
    logger.info(
        "Video production complete: %d articles processed, %d with final video",
        len(work),
        sum(1 for o in report.outputs if o.get("subtitled_video")),
    )
    return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filter_eligible(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep articles that have an audio file (required for video).

    Also attempts to remap stored paths from other machines (e.g. Linux)
    to the local data directory so the pipeline works portably.
    """
    eligible = []
    for a in articles:
        audio = (
            a.get("processed_audio_path")
            or a.get("tts_audio_path")
            or a.get("audio_path")
            or ""
        )
        # Try direct path first
        if audio and Path(audio).exists():
            eligible.append(a)
        else:
            # Attempt local path remapping
            remapped = _remap_audio_path(audio)
            if remapped:
                # Update the article dict with the remapped path
                if a.get("processed_audio_path"):
                    a["processed_audio_path"] = remapped
                elif a.get("tts_audio_path"):
                    a["tts_audio_path"] = remapped
                elif a.get("audio_path"):
                    a["audio_path"] = remapped
                eligible.append(a)
            else:
                logger.debug(
                    "Skipping article %d: no audio file found (path: %s)",
                    a.get("article_id", "?"), audio[:80] if audio else "empty",
                )
    return eligible


def _remap_audio_path(old_path: str) -> str:
    """Attempt to remap a stored audio path to the local filesystem.

    Handles paths from other machines (e.g. ``/home/user/.../data_acquisition/data/...``)
    by extracting the relative portion and resolving against the local DATA_DIR.
    """
    import re as _re

    if not old_path:
        return ""
    # Normalize separators
    normalized = old_path.replace("\\", "/")
    # Look for the data_acquisition/data/ boundary
    match = _re.search(r"data_acquisition/data/(.+)$", normalized)
    if match:
        from data_acquisition import config as acq_cfg
        rel = match.group(1)
        local = acq_cfg.DATA_DIR / rel
        if local.exists():
            return str(local)
    # Also try just the filename in the standard audio directories
    from data_acquisition import config as acq_cfg
    filename = Path(old_path).name
    for subdir in ("audio_processed", "audio"):
        candidate = acq_cfg.DATA_DIR / "media" / subdir / filename
        if candidate.exists():
            return str(candidate)
    return ""


def _build_work_items(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize field names into a consistent working dict."""
    work = []
    for a in articles:
        audio = (
            a.get("processed_audio_path")
            or a.get("tts_audio_path")
            or a.get("audio_path")
            or ""
        )
        duration = (
            a.get("audio_duration_sec")
            or a.get("tts_duration_sec")
            or a.get("audio_duration")
            or 0.0
        )
        # Ensure float
        try:
            duration = float(duration)
        except (TypeError, ValueError):
            duration = 0.0

        # Urdu script: prefer full_script (Stage 2.10), fall back to urdu_text
        urdu_script = (
            a.get("full_script")
            or a.get("urdu_text")
            or ""
        )

        # Headline: prefer structured headline, fall back to urdu_title or title
        headline = (
            a.get("headline")
            or a.get("urdu_title")
            or a.get("title")
            or ""
        )

        work.append({
            "article_id": a.get("article_id", 0),
            "headline": headline,
            "urdu_script": urdu_script,
            "category": a.get("category", ""),
            "audio_path": audio,
            "audio_duration": duration,
            "news_image": a.get("news_image", ""),
            "timeline_segments": a.get("timeline_segments"),
        })
    return work


def _count_by(records: List[Any], attr: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in records:
        key = str(getattr(r, attr, ""))
        counts[key] = counts.get(key, 0) + 1
    return counts
