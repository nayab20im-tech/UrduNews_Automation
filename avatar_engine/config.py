"""
avatar_engine/config.py
==========================
Configuration for the avatar generation subsystem.

Follows the project-wide pattern (``pipeline/config.py``,
``video_production/config.py``): every setting is overridable through
environment variables, with a YAML file (``--config``) layered on top
for per-run overrides.  Nothing is hard-coded to a machine path.

Model-selection rationale (see README.md for the full research table):

* **TTS primary**  — Piper ``ur_PK-fasih-medium`` (MIT license, neural,
  local).  Falls back to espeak-ng ``-v ur`` (bundled Urdu voice, zero
  download) when the Piper model is not installed, which is the
  verified default on CPU-only hosts.
* **TTS fallback** — espeak-ng Urdu, then the deterministic tone-burst
  placeholder from ``tts/wav_synth`` so the pipeline never dead-ends.
* **Talking head** — ``animated`` (audio-driven CPU presenter renderer,
  proven in ``video_production/lip_sync.py``) as the verified default;
  ``sadtalker`` (Apache-2.0) available behind config on CUDA hosts.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODULE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
ASSETS_DIR = Path(os.getenv(
    "AVATAR_ASSETS_DIR", str(_MODULE_DIR / "assets")))
CACHE_DIR = Path(os.getenv(
    "AVATAR_CACHE_DIR", str(_MODULE_DIR / ".cache")))
OUTPUT_DIR = Path(os.getenv(
    "AVATAR_OUTPUT_DIR", str(_PROJECT_ROOT / "output" / "avatar_engine")))

# Canonical two-anchor source image (male + female behind the Pakistan
# News desk).  ``prepare_anchors.py`` derives assets/male.png and
# assets/female.png from it.
ANCHOR_SOURCE_IMAGE = os.getenv("ANCHOR_SOURCE_IMAGE", "")
if not ANCHOR_SOURCE_IMAGE:
    for _cand in (_PROJECT_ROOT / "anchors.jpeg",
                  _PROJECT_ROOT / "anchors.jpg",
                  _PROJECT_ROOT / "anchors.png"):
        if _cand.exists():
            ANCHOR_SOURCE_IMAGE = str(_cand)
            break

# ---------------------------------------------------------------------------
# FFmpeg detection (same priority chain as video_production/config.py)
# ---------------------------------------------------------------------------
FFMPEG_BINARY = os.getenv("AVATAR_FFMPEG_BINARY", "")
if not FFMPEG_BINARY or not Path(FFMPEG_BINARY).exists():
    _detected = shutil.which("ffmpeg")
    if _detected:
        FFMPEG_BINARY = _detected
    else:
        try:
            import static_ffmpeg
            _p, _ = static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()
            FFMPEG_BINARY = _p
        except Exception:
            FFMPEG_BINARY = "ffmpeg"
FFPROBE_BINARY = shutil.which("ffprobe") or "ffprobe"

# ---------------------------------------------------------------------------
# Video specification (broadcast target)
# ---------------------------------------------------------------------------
VIDEO_WIDTH = int(os.getenv("AVATAR_VIDEO_WIDTH", "1920"))
VIDEO_HEIGHT = int(os.getenv("AVATAR_VIDEO_HEIGHT", "1080"))
VIDEO_FPS = int(os.getenv("AVATAR_VIDEO_FPS", "25"))
VIDEO_CODEC = os.getenv("AVATAR_VIDEO_CODEC", "libx264")
VIDEO_CRF = int(os.getenv("AVATAR_VIDEO_CRF", "18"))
AUDIO_CODEC = os.getenv("AVATAR_AUDIO_CODEC", "aac")
AUDIO_BITRATE = os.getenv("AVATAR_AUDIO_BITRATE", "192k")

# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------
TTS_BACKEND = os.getenv("AVATAR_TTS_BACKEND", "auto")   # auto|piper|espeak|xtts|placeholder
PIPER_BINARY = os.getenv("PIPER_BINARY", "piper")
PIPER_MODEL_MALE = os.getenv(
    "PIPER_MODEL_MALE",
    str(ASSETS_DIR / "models" / "ur_PK-fasih-medium.onnx"))
ESPEAK_SPEED = int(os.getenv("AVATAR_ESPEAK_SPEED", "160"))   # words/min
ESPEAK_PITCH_MALE = int(os.getenv("AVATAR_ESPEAK_PITCH_MALE", "40"))
ESPEAK_PITCH_FEMALE = int(os.getenv("AVATAR_ESPEAK_PITCH_FEMALE", "72"))
TARGET_SAMPLE_RATE = int(os.getenv("AVATAR_TARGET_SAMPLE_RATE", "22050"))
CHUNK_SILENCE_PAD_MS = int(os.getenv("AVATAR_CHUNK_SILENCE_PAD_MS", "300"))

# Chunking: target seconds of speech per animation chunk.
CHUNK_MIN_SEC = float(os.getenv("AVATAR_CHUNK_MIN_SEC", "8"))
CHUNK_MAX_SEC = float(os.getenv("AVATAR_CHUNK_MAX_SEC", "20"))

# ---------------------------------------------------------------------------
# Talking-head backend
# ---------------------------------------------------------------------------
AVATAR_BACKEND = os.getenv("AVATAR_HEAD_BACKEND", "auto")  # auto|animated|sadtalker
SADTALKER_CHECKPOINT_DIR = os.getenv("AVATAR_SADTALKER_CHECKPOINT_DIR", "")
SADTALKER_REPO = os.getenv("AVATAR_SADTALKER_REPO", "")
FACE_ENHANCER = os.getenv("AVATAR_FACE_ENHANCER", "")       # e.g. "gfpgan" (optional)
GESTURE_ENABLED = os.getenv("AVATAR_GESTURE_ENABLED", "false").lower() in (
    "1", "true", "yes")

# Presenter placement inside the 1080p frame (centred single anchor).
PRESENTER_WIDTH = int(os.getenv("AVATAR_PRESENTER_WIDTH", "900"))
PRESENTER_HEIGHT = int(os.getenv("AVATAR_PRESENTER_HEIGHT", "1050"))

# ---------------------------------------------------------------------------
# Scene layout
# ---------------------------------------------------------------------------
# "dual"  — both anchors from the canonical two-anchor studio photo share
#           the frame and alternate reading the news (real-channel style).
# "single"— one centred anchor (legacy behaviour).
SCENE_MODE = os.getenv("AVATAR_SCENE_MODE", "dual")
FIRST_SPEAKER = os.getenv("AVATAR_FIRST_SPEAKER", "female")


@dataclass
class AvatarConfig:
    """Runtime configuration object passed through the pipeline.

    Built from defaults + env vars; a YAML file and/or explicit kwargs
    override individual fields.  Only settings that affect generated
    artefacts are part of the cache key (see ``cache.py``).
    """

    anchor_id: str = "female"
    tts_backend: str = TTS_BACKEND
    avatar_backend: str = AVATAR_BACKEND
    video_width: int = VIDEO_WIDTH
    video_height: int = VIDEO_HEIGHT
    video_fps: int = VIDEO_FPS
    espeak_speed: int = ESPEAK_SPEED
    chunk_max_sec: float = CHUNK_MAX_SEC
    chunk_min_sec: float = CHUNK_MIN_SEC
    gesture_enabled: bool = GESTURE_ENABLED
    face_enhancer: str = FACE_ENHANCER
    sadtalker_checkpoint_dir: str = SADTALKER_CHECKPOINT_DIR
    piper_model_male: str = PIPER_MODEL_MALE
    scene_mode: str = SCENE_MODE          # dual|single
    first_speaker: str = FIRST_SPEAKER    # female|male (dual mode)
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> "AvatarConfig":
        import yaml
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(data, **overrides)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], **overrides: Any) -> "AvatarConfig":
        merged: Dict[str, Any] = {}
        valid = set(asdict(cls()))
        for key, value in {**data, **overrides}.items():
            if key in valid:
                merged[key] = value
            else:
                merged.setdefault("extra", {})[key] = value
        return cls(**merged)

    def cache_relevant_dict(self) -> Dict[str, Any]:
        """Settings that change the generated artefacts (cache key)."""
        d = asdict(self)
        d.pop("extra", None)
        return d
