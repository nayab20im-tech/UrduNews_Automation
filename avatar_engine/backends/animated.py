"""
avatar_engine/backends/animated.py
=====================================
Default talking-head backend: deterministic audio-driven presenter
animation on CPU (no model weights, no GPU).

LIP-SYNC NOTE (language independence)
-------------------------------------
Audio-driven lip sync operates on *acoustic* features — per-frame RMS
energy and spectral bands (bass ≈ syllable nuclei, mid/high ≈
consonant shaping) — not on language semantics.  The same renderer
therefore synchronizes Urdu, English or any other speech without a
language-specific model.  The jaw-open curve is built from
syllable-nucleus peak detection (see
``video_production/lip_sync.LipSyncEngine._analyse_fft_energy``), which
this backend reuses verbatim so there is exactly ONE implementation of
the animation math in the repository.

The backend renders, per frame: mouth opening/closing on the detected
lip seam, subtle eyebrow lift, natural eye blinking, speech-tied head
sway and breathing — calm, professional presenter motion (no dancing,
no exaggerated smiles).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from data_acquisition.utils.logger import get_logger

from .. import config as cfg
from ..errors import AvatarGenerationError, LipSyncError
from .base import TalkingHeadBackend

logger = get_logger("avatar_engine.backends.animated")


class AnimatedBackend(TalkingHeadBackend):
    """CPU presenter renderer reusing the proven lip-sync engine."""

    name = "animated"
    requires_gpu = False

    def available(self) -> bool:
        try:
            import cv2  # noqa: F401
            import scipy  # noqa: F401
        except ImportError:
            return False
        return bool(shutil.which(cfg.FFMPEG_BINARY) or Path(cfg.FFMPEG_BINARY).exists())

    def version(self) -> str:
        # The trailing revision tag is part of the chunk cache key:
        # bump it whenever the shared renderer's geometry changes so
        # stale cached chunks are invalidated.
        try:
            import cv2
            return f"animated-cv{cv2.__version__}-r4"
        except ImportError:
            return "animated-r4"

    def generate(
        self,
        anchor_image: Path | str,
        audio_path: Path | str,
        output_video: Path | str,
        duration_sec: float,
    ) -> None:
        if not Path(anchor_image).exists():
            raise AvatarGenerationError(f"Anchor image missing: {anchor_image}")
        if not Path(audio_path).exists():
            raise LipSyncError(f"Driving audio missing: {audio_path}")

        # Reuse the single proven implementation of the animation math
        # (video_production/lip_sync.py).  Temporarily apply this
        # subsystem's layout (centred presenter, configured fps/size)
        # and restore the host pipeline's values afterwards.
        from video_production import config as vc
        from video_production.lip_sync import LipSyncEngine

        saved = {
            "PRESENTER_WIDTH": vc.PRESENTER_WIDTH,
            "PRESENTER_HEIGHT": vc.PRESENTER_HEIGHT,
            "PRESENTER_X": vc.PRESENTER_X,
            "PRESENTER_Y": vc.PRESENTER_Y,
            "VIDEO_FPS": vc.VIDEO_FPS,
            "VIDEO_WIDTH": vc.VIDEO_WIDTH,
            "VIDEO_HEIGHT": vc.VIDEO_HEIGHT,
            "VIDEO_CODEC": vc.VIDEO_CODEC,
            "VIDEO_CRF": vc.VIDEO_CRF,
            "AUDIO_CODEC": vc.AUDIO_CODEC,
            "AUDIO_BITRATE": vc.AUDIO_BITRATE,
        }
        try:
            vc.PRESENTER_WIDTH = cfg.PRESENTER_WIDTH
            vc.PRESENTER_HEIGHT = cfg.PRESENTER_HEIGHT
            vc.PRESENTER_X = (cfg.VIDEO_WIDTH - cfg.PRESENTER_WIDTH) // 2
            vc.PRESENTER_Y = max(0, (cfg.VIDEO_HEIGHT - cfg.PRESENTER_HEIGHT) // 2)
            vc.VIDEO_FPS = cfg.VIDEO_FPS
            vc.VIDEO_WIDTH = cfg.VIDEO_WIDTH
            vc.VIDEO_HEIGHT = cfg.VIDEO_HEIGHT
            vc.VIDEO_CODEC = cfg.VIDEO_CODEC
            vc.VIDEO_CRF = cfg.VIDEO_CRF
            vc.AUDIO_CODEC = cfg.AUDIO_CODEC
            vc.AUDIO_BITRATE = cfg.AUDIO_BITRATE

            engine = LipSyncEngine(
                output_dir=Path(output_video).parent,
                backend="animated",
                source_image=str(anchor_image),
                ffmpeg=cfg.FFMPEG_BINARY,
            )
            ok = engine._run_animated(
                str(audio_path), Path(output_video), duration_sec,
            )
        except (AvatarGenerationError, LipSyncError):
            raise
        except Exception as exc:
            raise AvatarGenerationError(f"Animated backend crashed: {exc}") from exc
        finally:
            for key, value in saved.items():
                setattr(vc, key, value)

        if not ok or not Path(output_video).exists():
            raise AvatarGenerationError(
                "Animated backend produced no output "
                f"(audio={audio_path}, image={anchor_image})")
        logger.info("[AVATAR] animated chunk -> %s", output_video)

    def generate_dual(
        self,
        canvas,
        faces: dict,
        speaker_id: str,
        audio_path: Path | str,
        output_video: Path | str,
        duration_sec: float,
    ) -> None:
        """Two-anchor scene: speaker talks, co-anchor listens."""
        if not Path(audio_path).exists():
            raise LipSyncError(f"Driving audio missing: {audio_path}")

        from video_production import config as vc
        from video_production.lip_sync import LipSyncEngine

        saved = {
            "VIDEO_FPS": vc.VIDEO_FPS,
            "VIDEO_WIDTH": vc.VIDEO_WIDTH,
            "VIDEO_HEIGHT": vc.VIDEO_HEIGHT,
            "VIDEO_CODEC": vc.VIDEO_CODEC,
            "VIDEO_CRF": vc.VIDEO_CRF,
            "AUDIO_CODEC": vc.AUDIO_CODEC,
            "AUDIO_BITRATE": vc.AUDIO_BITRATE,
        }
        try:
            vc.VIDEO_FPS = cfg.VIDEO_FPS
            vc.VIDEO_WIDTH = cfg.VIDEO_WIDTH
            vc.VIDEO_HEIGHT = cfg.VIDEO_HEIGHT
            vc.VIDEO_CODEC = cfg.VIDEO_CODEC
            vc.VIDEO_CRF = cfg.VIDEO_CRF
            vc.AUDIO_CODEC = cfg.AUDIO_CODEC
            vc.AUDIO_BITRATE = cfg.AUDIO_BITRATE

            engine = LipSyncEngine(
                output_dir=Path(output_video).parent,
                backend="animated",
                source_image=None,
                ffmpeg=cfg.FFMPEG_BINARY,
            )
            ok = engine._run_dual_animated(
                canvas, faces, speaker_id,
                str(audio_path), Path(output_video), duration_sec,
            )
        except (AvatarGenerationError, LipSyncError):
            raise
        except Exception as exc:
            raise AvatarGenerationError(f"Dual renderer crashed: {exc}") from exc
        finally:
            for key, value in saved.items():
                setattr(vc, key, value)

        if not ok or not Path(output_video).exists():
            raise AvatarGenerationError(
                f"Dual renderer produced no output (speaker={speaker_id})")
        logger.info("[AVATAR] animated dual chunk (speaker=%s) -> %s",
                    speaker_id, output_video)
