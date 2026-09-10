"""
avatar_engine/errors.py
==========================
Exception hierarchy for the avatar generation subsystem.

Every stage raises a dedicated, clearly-named error so the caller can
tell exactly which stage failed.  All errors derive from
``AvatarEngineError`` so a single ``except`` can catch subsystem
failures when desired.
"""

from __future__ import annotations


class AvatarEngineError(Exception):
    """Base class for every avatar_engine failure."""

    stage: str = "engine"


class TTSGenerationError(AvatarEngineError):
    """Speech synthesis failed for a script chunk."""

    stage = "tts"


class AvatarGenerationError(AvatarEngineError):
    """Talking-head video generation failed for a chunk."""

    stage = "avatar"


class LipSyncError(AvatarEngineError):
    """Audio/mouth synchronization failed."""

    stage = "lipsync"


class InvalidAnchorError(AvatarEngineError):
    """Unknown anchor_id or unusable anchor asset."""

    stage = "anchor"


class ModelNotFoundError(AvatarEngineError):
    """A configured model checkpoint / binary is missing."""

    stage = "model"


class GPUUnavailableError(AvatarEngineError):
    """A GPU-only backend was requested but no CUDA device exists."""

    stage = "gpu"


class FFmpegError(AvatarEngineError):
    """An FFmpeg encode/concat/probe command failed."""

    stage = "ffmpeg"


class ValidationError(AvatarEngineError):
    """Final output failed the automated quality-control checks."""

    stage = "validate"
