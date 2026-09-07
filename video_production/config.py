"""
video_production/config.py
=============================
Configuration for Section 3 — Video Production Pipeline (3.1-3.6).

Mirrors the project's existing config pattern (`pipeline/config.py`):
every setting is overridable via environment variables with sensible
defaults.  All paths are relative to the acquisition DATA_DIR so
nothing is hard-coded to a specific machine.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

from data_acquisition import config as acq_cfg

# ---------------------------------------------------------------------------
# Root output directories
# ---------------------------------------------------------------------------
VIDEO_OUTPUT_DIR = Path(os.getenv(
    "VIDEO_OUTPUT_DIR", str(acq_cfg.DATA_DIR / "media" / "video")))
AVATAR_DIR = Path(os.getenv(
    "AVATAR_DIR", str(VIDEO_OUTPUT_DIR / "avatar")))
LIPSYNC_DIR = Path(os.getenv(
    "LIPSYNC_DIR", str(VIDEO_OUTPUT_DIR / "lip_sync")))
VISUALS_DIR = Path(os.getenv(
    "VISUALS_DIR", str(VIDEO_OUTPUT_DIR / "visuals")))
COMPOSITION_DIR = Path(os.getenv(
    "COMPOSITION_DIR", str(VIDEO_OUTPUT_DIR / "composition")))
SUBTITLES_DIR = Path(os.getenv(
    "SUBTITLES_DIR", str(VIDEO_OUTPUT_DIR / "subtitles")))
THUMBNAILS_DIR = Path(os.getenv(
    "THUMBNAILS_DIR", str(VIDEO_OUTPUT_DIR / "thumbnails")))
FINAL_OUTPUT_DIR = Path(os.getenv(
    "FINAL_OUTPUT_DIR", str(VIDEO_OUTPUT_DIR / "final")))

# ---------------------------------------------------------------------------
# Video specifications
# ---------------------------------------------------------------------------
VIDEO_WIDTH = int(os.getenv("VIDEO_WIDTH", "1920"))
VIDEO_HEIGHT = int(os.getenv("VIDEO_HEIGHT", "1080"))
VIDEO_FPS = int(os.getenv("VIDEO_FPS", "30"))
VIDEO_CODEC = os.getenv("VIDEO_CODEC", "libx264")
VIDEO_CRF = int(os.getenv("VIDEO_CRF", "18"))          # broadcast quality (lower=better)
AUDIO_CODEC = os.getenv("AUDIO_CODEC", "aac")
AUDIO_BITRATE = os.getenv("AUDIO_BITRATE", "192k")

# ---------------------------------------------------------------------------
# 3.1  Avatar generation
# ---------------------------------------------------------------------------
# AVATAR_BACKEND: auto | did | heygen | sadtalker | placeholder
AVATAR_BACKEND = os.getenv("AVATAR_BACKEND", "auto")
SADTALKER_CHECKPOINT_DIR = os.getenv("SADTALKER_CHECKPOINT_DIR", "")

# ---------------------------------------------------------------------------
# Cloud AI API keys (D-ID and HeyGen) — best quality, require paid accounts
# ---------------------------------------------------------------------------
DID_API_KEY        = os.getenv("DID_API_KEY", "")
DID_AVATAR_ID      = os.getenv("DID_AVATAR_ID", "")      # Pre-configured D-ID presenter ID
HEYGEN_API_KEY     = os.getenv("HEYGEN_API_KEY", "")
HEYGEN_AVATAR_ID   = os.getenv("HEYGEN_AVATAR_ID", "")   # HeyGen avatar_id for your presenter

_MODULE_DIR = Path(__file__).resolve().parent


def _detect_avatar_image() -> str:
    """Resolve the presenter/avatar source image.

    Priority: ``AVATAR_SOURCE_IMAGE`` env var, then any bundled image
    dropped into the ``video_production/`` directory (e.g. the anchor
    photo ``images.jpeg``).
    """
    env_val = os.getenv("AVATAR_SOURCE_IMAGE", "")
    if env_val and Path(env_val).exists():
        return env_val
    for name in (
        "actual_anchor.png", "actual_anchor.jpg", "actual_anchor.jpeg",
        "avatar.png", "avatar.jpg", "avatar.jpeg",
        "presenter.png", "presenter.jpg", "presenter.jpeg",
        "images.jpeg", "images.jpg", "images.png",
    ):
        candidate = _MODULE_DIR / name
        if candidate.exists():
            return str(candidate)
    return ""


AVATAR_SOURCE_IMAGE = _detect_avatar_image()   # path to anchor face image

# ---------------------------------------------------------------------------
# 3.2  Lip synchronization
# ---------------------------------------------------------------------------
# LIPSYNC_BACKEND: auto | did | heygen | musetalk | latentsync | wav2lip | sadtalker | animated | placeholder
LIPSYNC_BACKEND = os.getenv("LIPSYNC_BACKEND", "auto")
WAV2LIP_CHECKPOINT = os.getenv("WAV2LIP_CHECKPOINT", "")
WAV2LIP_FACE_IMAGE = os.getenv("WAV2LIP_FACE_IMAGE", "")

# GPU open-source lip-sync / talking-head models
MUSETALK_CHECKPOINT_DIR   = os.getenv("MUSETALK_CHECKPOINT_DIR", "")
LATENTSYNC_CHECKPOINT_DIR = os.getenv("LATENTSYNC_CHECKPOINT_DIR", "")

# Mouth region of the avatar image (fractions of the presenter frame),
# used by the offline "animated" lip-sync backend to open/close the
# mouth in sync with the voice-over energy.
AVATAR_MOUTH_CX = float(os.getenv("AVATAR_MOUTH_CX", "0.5"))
AVATAR_MOUTH_CY = float(os.getenv("AVATAR_MOUTH_CY", "0.72"))
AVATAR_MOUTH_W = float(os.getenv("AVATAR_MOUTH_W", "0.30"))
AVATAR_MOUTH_H = float(os.getenv("AVATAR_MOUTH_H", "0.08"))
# Jaw drop as fraction of face height per unit of open_amount (0-1).
# 0.14 → max ~25 px on a 184 px face, clearly visible without distortion.
MOUTH_OPEN_SCALE = float(os.getenv("MOUTH_OPEN_SCALE", "0.14"))

# ---------------------------------------------------------------------------
# 3.3  Background & visuals
# ---------------------------------------------------------------------------
STUDIO_BG_IMAGE = os.getenv("STUDIO_BG_IMAGE", "")     # custom studio background
NEWS_IMAGES_DIR = Path(os.getenv(
    "NEWS_IMAGES_DIR", str(acq_cfg.DATA_DIR / "media" / "images")))

# Lower-third / ticker appearance
LOWER_THIRD_BG_COLOR = os.getenv("LOWER_THIRD_BG_COLOR", "#1a1a6e")
LOWER_THIRD_TEXT_COLOR = os.getenv("LOWER_THIRD_TEXT_COLOR", "#ffffff")
TICKER_BG_COLOR = os.getenv("TICKER_BG_COLOR", "#cc0000")
TICKER_TEXT_COLOR = os.getenv("TICKER_TEXT_COLOR", "#ffffff")
CHANNEL_NAME = os.getenv("CHANNEL_NAME", "اردو نیوز AI")

# ---------------------------------------------------------------------------
# 3.4  Composition
# ---------------------------------------------------------------------------
FFMPEG_BINARY = os.getenv("FFMPEG_BINARY", "")

# Auto-detect FFmpeg: env var → system PATH → static_ffmpeg bundle
if not FFMPEG_BINARY or not Path(FFMPEG_BINARY).exists():
    _detected = shutil.which("ffmpeg")
    if _detected:
        FFMPEG_BINARY = _detected
    else:
        try:
            import static_ffmpeg
            _ffmpeg_path, _ = static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()
            FFMPEG_BINARY = _ffmpeg_path
        except Exception:
            FFMPEG_BINARY = "ffmpeg"
PRESENTER_WIDTH = int(os.getenv("PRESENTER_WIDTH", "750"))
PRESENTER_HEIGHT = int(os.getenv("PRESENTER_HEIGHT", "1020"))
PRESENTER_X = int(os.getenv("PRESENTER_X", "80"))    # left-aligned for broadcast layout
PRESENTER_Y = int(os.getenv("PRESENTER_Y", "30"))

# Optional channel logo / watermark (PNG with transparency, top-left corner)
CHANNEL_LOGO_PATH = os.getenv("CHANNEL_LOGO_PATH", "")

# ---------------------------------------------------------------------------
# 3.5  Subtitles
# ---------------------------------------------------------------------------
SUBTITLE_FONT_SIZE = int(os.getenv("SUBTITLE_FONT_SIZE", "48"))
SUBTITLE_FONT_PATH = os.getenv("SUBTITLE_FONT_PATH", "")
SUBTITLE_POSITION_Y = int(os.getenv("SUBTITLE_POSITION_Y", "850"))
SUBTITLE_TEXT_COLOR = os.getenv("SUBTITLE_TEXT_COLOR", "#ffffff")
SUBTITLE_BG_COLOR = os.getenv("SUBTITLE_BG_COLOR", "#000000")
SUBTITLE_BG_OPACITY = float(os.getenv("SUBTITLE_BG_OPACITY", "0.6"))
SUBTITLE_MAX_CHARS = int(os.getenv("SUBTITLE_MAX_CHARS", "60"))

# ---------------------------------------------------------------------------
# 3.6  Thumbnail
# ---------------------------------------------------------------------------
THUMBNAIL_WIDTH = int(os.getenv("THUMBNAIL_WIDTH", "1280"))
THUMBNAIL_HEIGHT = int(os.getenv("THUMBNAIL_HEIGHT", "720"))
THUMBNAIL_FONT_PATH = os.getenv("THUMBNAIL_FONT_PATH", "")

# ---------------------------------------------------------------------------
# Urdu font — used by visual_generator, subtitle_generator, thumbnail_generator.
# Tries a sensible default chain; set URDU_FONT_PATH to override.
# ---------------------------------------------------------------------------
_URDU_FONT_CANDIDATES = [
    "C:/Windows/Fonts/arialuni.ttf",         # Windows: Arial Unicode MS
    "C:/Windows/Fonts/segoeui.ttf",           # Windows: Segoe UI (has Arabic)
    "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoNaskhArabic-Regular.ttf",
    "/System/Library/Fonts/GeezaPro.ttc",     # macOS
]


def resolve_urdu_font() -> str:
    """Return the first existing font path from the candidate list, or ''."""
    custom = os.getenv("URDU_FONT_PATH", "")
    if custom and Path(custom).exists():
        return custom
    for candidate in _URDU_FONT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return ""


URDU_FONT_PATH = resolve_urdu_font()

# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------
HAS_CUDA = False
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    pass
