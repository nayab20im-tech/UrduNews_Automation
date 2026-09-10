"""
avatar_engine/backends/base.py
=================================
Abstract talking-head backend.

The rest of the project never knows which model renders the anchor —
it only calls ``generate(anchor_image, audio, output_video, ...)``.
A backend raises (never returns False silently) so the pipeline can
report the exact failing stage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from .. import config as cfg


class TalkingHeadBackend(ABC):
    """Base class for all audio-driven talking-head renderers."""

    name: str = "base"
    requires_gpu: bool = False

    def __init__(self, config: cfg.AvatarConfig | None = None):
        self.config = config or cfg.AvatarConfig()

    @abstractmethod
    def available(self) -> bool:
        """True when this backend can run on the current host."""

    @abstractmethod
    def generate(
        self,
        anchor_image: Path | str,
        audio_path: Path | str,
        output_video: Path | str,
        duration_sec: float,
    ) -> None:
        """Render ``output_video`` (with audio) or raise a typed error."""

    def generate_dual(
        self,
        canvas,
        faces: dict,
        speaker_id: str,
        audio_path: Path | str,
        output_video: Path | str,
        duration_sec: float,
    ) -> None:
        """Render a two-anchor scene (speaker + listening co-anchor).

        Backends that cannot animate a shared scene raise
        ``NotImplementedError``; the pipeline then falls back to the
        default CPU renderer for dual scenes.
        """
        raise NotImplementedError(
            f"{self.name} does not implement dual-anchor scenes")

    def version(self) -> str:
        return self.name
