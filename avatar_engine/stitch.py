"""
avatar_engine/stitch.py
==========================
FFmpeg stitching of per-chunk avatar videos into the final MP4.

Chunks are concatenated with the concat demuxer and then passed
through ONE final controlled encode stage (H.264 + AAC, yuv420p,
configured fps/resolution).  We never blindly stream-copy: even when
chunk codecs match, the final encode normalizes timestamps, SAR and
metadata so the output is a single clean broadcast file.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from data_acquisition.utils.logger import get_logger

from . import config as cfg
from .errors import FFmpegError

logger = get_logger("avatar_engine.stitch")


def stitch_chunks(
    chunk_videos: list[Path | str],
    output_path: Path | str,
    config: cfg.AvatarConfig | None = None,
) -> Path:
    """Concatenate chunk MP4s into ``output_path`` (final encode)."""
    config = config or cfg.AvatarConfig()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not chunk_videos:
        raise FFmpegError("No chunk videos to stitch")
    for video in chunk_videos:
        if not Path(video).exists() or Path(video).stat().st_size == 0:
            raise FFmpegError(f"Missing/empty chunk video: {video}")

    # Single chunk: still run the controlled final encode so metadata
    # and codecs are guaranteed identical to multi-chunk outputs.
    list_file = Path(tempfile.mkstemp(suffix=".txt", prefix="concat_")[1])
    try:
        list_file.write_text(
            "".join(f"file '{Path(v).resolve()}'\n" for v in chunk_videos),
            encoding="utf-8",
        )
        cmd = [
            cfg.FFMPEG_BINARY, "-y",
            "-f", "concat", "-safe", "0", "-i", str(list_file),
            "-vf", f"scale={config.video_width}:{config.video_height},"
                   f"fps={config.video_fps}",
            "-c:v", cfg.VIDEO_CODEC,
            "-crf", str(cfg.VIDEO_CRF),
            "-preset", "medium",
            "-pix_fmt", "yuv420p",
            "-c:a", cfg.AUDIO_CODEC,
            "-b:a", cfg.AUDIO_BITRATE,
            "-ar", str(cfg.TARGET_SAMPLE_RATE),
            "-movflags", "+faststart",
            str(output_path),
        ]
        logger.info("[STITCH] Encoding %d chunk(s) -> %s",
                    len(chunk_videos), output_path.name)
        proc = subprocess.run(cmd, capture_output=True, timeout=1800)
        if proc.returncode != 0:
            raise FFmpegError(
                "Final FFmpeg encode failed: "
                + proc.stderr.decode(errors="replace")[-500:])
    finally:
        list_file.unlink(missing_ok=True)

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise FFmpegError(f"Stitched output missing: {output_path}")
    logger.info("[STITCH] Complete: %s", output_path)
    return output_path
