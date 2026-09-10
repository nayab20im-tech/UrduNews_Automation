"""
avatar_engine/cache.py
=========================
Content-hash caching so identical chunks are never regenerated.

Cache key = SHA256 over:

    normalized script chunk
  + anchor id
  + TTS backend / version
  + avatar backend / version
  + cache-relevant configuration

Each entry lives in ``CACHE_DIR/<key>/`` containing ``audio.wav`` and
``video.mp4``.  A hit is only trusted when both artefacts exist and are
non-empty (a crashed run never poisons the cache).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from data_acquisition.utils.logger import get_logger

from . import config as cfg

logger = get_logger("avatar_engine.cache")


def chunk_cache_key(
    chunk_text: str,
    anchor_id: str,
    tts_backend: str,
    avatar_backend: str,
    config_dict: dict,
) -> str:
    """Deterministic SHA256 key for one (chunk, anchor, stack) tuple."""
    payload = json.dumps(
        {
            "text": chunk_text,
            "anchor": anchor_id,
            "tts": tts_backend,
            "avatar": avatar_backend,
            "config": config_dict,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ChunkCache:
    """Filesystem cache of per-chunk TTS audio + avatar video."""

    def __init__(self, cache_dir: Path | str = cfg.CACHE_DIR):
        self.cache_dir = Path(cache_dir)

    def _entry_dir(self, key: str) -> Path:
        return self.cache_dir / key[:2] / key

    def lookup(self, key: str) -> dict | None:
        """Return {'audio': Path, 'video': Path} when a valid entry exists."""
        d = self._entry_dir(key)
        audio, video = d / "audio.wav", d / "video.mp4"
        if (
            audio.exists() and video.exists()
            and audio.stat().st_size > 0 and video.stat().st_size > 0
        ):
            return {"audio": audio, "video": video}
        return None

    def store(self, key: str, audio_path: Path | str, video_path: Path | str) -> dict:
        import shutil
        d = self._entry_dir(key)
        d.mkdir(parents=True, exist_ok=True)
        dst_audio, dst_video = d / "audio.wav", d / "video.mp4"
        shutil.copy2(str(audio_path), dst_audio)
        shutil.copy2(str(video_path), dst_video)
        logger.info("[CACHE] Stored %s", key[:12])
        return {"audio": dst_audio, "video": dst_video}
