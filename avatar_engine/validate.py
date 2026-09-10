"""
avatar_engine/validate.py
============================
Automated output validator (Phase 11).

Runs FFprobe/FFmpeg checks on the final MP4 and returns a structured
result; ``validate_video(..., strict=True)`` raises ``ValidationError``
when any check fails so the pipeline never reports success on a broken
artefact.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from data_acquisition.utils.logger import get_logger

from . import config as cfg
from .errors import ValidationError

logger = get_logger("avatar_engine.validate")

_DURATION_TOLERANCE_SEC = 0.6


def _probe(path: Path) -> dict:
    proc = subprocess.run(
        [cfg.FFPROBE_BINARY, "-v", "quiet", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, timeout=120,
    )
    if proc.returncode != 0:
        raise ValidationError(f"FFprobe cannot read {path}")
    return json.loads(proc.stdout.decode("utf-8", "replace"))


def validate_video(
    path: Path | str,
    config: cfg.AvatarConfig | None = None,
    strict: bool = False,
) -> dict:
    """Return the structured QC report for one generated MP4."""
    config = config or cfg.AvatarConfig()
    path = Path(path)
    errors: list[str] = []

    result = {
        "success": False,
        "video": str(path),
        "duration": 0.0,
        "fps": 0,
        "width": 0,
        "height": 0,
        "audio": False,
        "sync_ok": False,
        "errors": errors,
    }

    # 1. exists / readable
    if not path.exists() or path.stat().st_size == 0:
        errors.append("output file missing or empty")
        if strict:
            raise ValidationError("; ".join(errors))
        return result

    try:
        info = _probe(path)
    except ValidationError as exc:
        errors.append(str(exc))
        if strict:
            raise
        return result

    streams = info.get("streams", [])
    vstream = next((s for s in streams if s.get("codec_type") == "video"), None)
    astream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    # 2-3. stream presence
    if vstream is None:
        errors.append("no video stream")
    if astream is None:
        errors.append("no audio stream")
    else:
        result["audio"] = True

    # 8. codecs
    if vstream and vstream.get("codec_name") not in ("h264", "libx264"):
        errors.append(f"unexpected video codec {vstream.get('codec_name')}")
    if astream and astream.get("codec_name") not in ("aac",):
        errors.append(f"unexpected audio codec {astream.get('codec_name')}")

    # 4-5. durations + A/V sync
    v_dur = float(vstream.get("duration", 0) or 0) if vstream else 0.0
    a_dur = float(astream.get("duration", 0) or 0) if astream else 0.0
    fmt_dur = float(info.get("format", {}).get("duration", 0) or 0)
    duration = fmt_dur or max(v_dur, a_dur)
    result["duration"] = round(duration, 2)
    if duration <= 0:
        errors.append("duration <= 0")
    if v_dur <= 0 and vstream:
        errors.append("zero-frame video")
    if a_dur <= 0 and astream:
        errors.append("empty audio")
    if v_dur and a_dur and abs(v_dur - a_dur) > _DURATION_TOLERANCE_SEC:
        errors.append(
            f"A/V duration mismatch: video={v_dur:.2f}s audio={a_dur:.2f}s")
    else:
        result["sync_ok"] = bool(vstream and astream)

    # 6. fps
    fps = 0.0
    if vstream:
        rate = vstream.get("avg_frame_rate") or vstream.get("r_frame_rate") or "0/1"
        try:
            num, den = rate.split("/")
            fps = float(num) / float(den) if float(den) else 0.0
        except (ValueError, ZeroDivisionError):
            fps = 0.0
    result["fps"] = round(fps, 2)
    if fps <= 0:
        errors.append("invalid fps")

    # 7. resolution
    width = int(vstream.get("width", 0)) if vstream else 0
    height = int(vstream.get("height", 0)) if vstream else 0
    result["width"], result["height"] = width, height
    if (width, height) != (config.video_width, config.video_height):
        errors.append(f"resolution {width}x{height} != "
                      f"{config.video_width}x{config.video_height}")

    # 9-10. full decode pass (catches corruption FFprobe misses)
    proc = subprocess.run(
        [cfg.FFMPEG_BINARY, "-v", "error", "-i", str(path),
         "-f", "null", "-"],
        capture_output=True, timeout=600,
    )
    if proc.returncode != 0:
        errors.append("FFmpeg decode pass failed")
    elif proc.stderr.strip():
        errors.append(f"decode warnings: {proc.stderr.decode(errors='replace')[:200]}")

    result["success"] = not errors
    for e in errors:
        logger.warning("[VALIDATE] %s: %s", path.name, e)
    logger.info(
        "[VALIDATE] %s | %.2fs | %s fps | %dx%d | audio=%s sync=%s",
        path.name, result["duration"], result["fps"], width, height,
        "OK" if result["audio"] else "MISSING",
        "OK" if result["sync_ok"] else "FAIL",
    )
    if strict and errors:
        raise ValidationError(f"{path}: " + "; ".join(errors))
    return result
