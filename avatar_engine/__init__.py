"""
avatar_engine
================
Audio-driven talking-anchor subsystem for the Automated Urdu News
Channel (Alibaba Cloud AI Hackathon).

One call turns an Urdu broadcast script into a lip-synced, blinking,
professionally animated 1080p presenter video:

    from avatar_engine import generate_anchor_video

    generate_anchor_video(
        script="آج کی اہم خبر...",
        anchor_id="female",          # or "male"
        output_path="output/news_001.mp4",
    )

Everything runs locally with open-source models; no SaaS dependency.
See ``avatar_engine/README.md`` for architecture, licenses and how to
swap TTS / talking-head backends.
"""

from __future__ import annotations

from .errors import (
    AvatarEngineError,
    AvatarGenerationError,
    FFmpegError,
    GPUUnavailableError,
    InvalidAnchorError,
    LipSyncError,
    ModelNotFoundError,
    TTSGenerationError,
    ValidationError,
)
from .pipeline import generate_anchor_video

__version__ = "1.0.0"

__all__ = [
    "generate_anchor_video",
    "AvatarEngineError",
    "TTSGenerationError",
    "AvatarGenerationError",
    "LipSyncError",
    "InvalidAnchorError",
    "ModelNotFoundError",
    "GPUUnavailableError",
    "FFmpegError",
    "ValidationError",
    "__version__",
]
