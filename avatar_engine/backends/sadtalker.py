"""
avatar_engine/backends/sadtalker.py
======================================
Optional GPU talking-head backend: SadTalker (CVPR 2023).

License: Apache-2.0 (non-commercial restriction removed upstream) —
suitable for a monetized channel.  Produces audio-driven head pose,
blink and lip motion from a single still image.

Requirements (verified against the official repository):
  * cloned repo (``AVATAR_SADTALKER_REPO``) with ``inference.py``
  * checkpoints dir (``AVATAR_SADTALKER_CHECKPOINT_DIR``)
  * CUDA GPU + torch

On CPU-only hosts ``available()`` is False and the registry falls back
to the ``animated`` backend.  This backend is never silently swapped:
if explicitly requested and unavailable, the pipeline raises.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from data_acquisition.utils.logger import get_logger

from .. import config as cfg
from ..errors import AvatarGenerationError, GPUUnavailableError, ModelNotFoundError
from ..gpu import detect_gpu, release_gpu_memory
from .base import TalkingHeadBackend

logger = get_logger("avatar_engine.backends.sadtalker")


class SadTalkerBackend(TalkingHeadBackend):
    name = "sadtalker"
    requires_gpu = True

    def _repo(self) -> Path | None:
        repo = self.config.extra.get("sadtalker_repo", cfg.SADTALKER_REPO)
        if repo and Path(repo).exists() and (Path(repo) / "inference.py").exists():
            return Path(repo)
        return None

    def available(self) -> bool:
        return (
            detect_gpu().available
            and self._repo() is not None
            and bool(self.config.sadtalker_checkpoint_dir)
            and Path(self.config.sadtalker_checkpoint_dir).exists()
        )

    def version(self) -> str:
        return "sadtalker-apache2.0"

    def generate(
        self,
        anchor_image: Path | str,
        audio_path: Path | str,
        output_video: Path | str,
        duration_sec: float,
    ) -> None:
        if not detect_gpu().available:
            raise GPUUnavailableError("SadTalker requested but no CUDA device")
        repo = self._repo()
        if repo is None:
            raise ModelNotFoundError(
                "SadTalker repo/inference.py not found (set AVATAR_SADTALKER_REPO)")
        ckpt = self.config.sadtalker_checkpoint_dir
        if not ckpt or not Path(ckpt).exists():
            raise ModelNotFoundError(
                f"SadTalker checkpoint dir missing: {ckpt!r}")

        out_dir = Path(output_video).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "python", str(repo / "inference.py"),
            "--driven_audio", str(audio_path),
            "--source_image", str(anchor_image),
            "--checkpoint_dir", ckpt,
            "--result_dir", str(out_dir),
            "--still",                       # calm presenter motion
        ]
        if self.config.face_enhancer:
            cmd += ["--enhancer", self.config.face_enhancer]
        proc = subprocess.run(cmd, capture_output=True, timeout=3600)
        if proc.returncode != 0:
            raise AvatarGenerationError(
                "SadTalker inference failed: "
                + proc.stderr.decode(errors="replace")[-500:])

        # SadTalker writes <result_dir>/<timestamp>.mp4 — pick the newest.
        candidates = sorted(out_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise AvatarGenerationError("SadTalker wrote no output video")
        shutil.move(str(candidates[-1]), str(output_video))
        release_gpu_memory()
        logger.info("[AVATAR] sadtalker chunk -> %s", output_video)
