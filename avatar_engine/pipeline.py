"""
avatar_engine/pipeline.py
============================
Orchestration + the ONE public API of the subsystem:

    generate_anchor_video(
        script_path=None, script=None,
        anchor_id="female", output_path=None, config=None,
    )

Flow (every chunk cached by content hash, never regenerated):

    script → sentence segmentation → semantic chunking
           → TTS chunk → avatar chunk → validation → stitching
           → final controlled FFmpeg encode → QC validation

Callers (the existing UrduNewsAI pipeline, the CLI, the optional
FastAPI server) never see TTS/PyTorch/FFmpeg details — they get a
structured result dict or a typed ``AvatarEngineError`` naming the
failed stage.
"""

from __future__ import annotations

import time
from pathlib import Path

from data_acquisition.utils.logger import get_logger

from . import config as cfg
from .anchors import anchor_image_path
from .backends import resolve_backend
from .cache import ChunkCache, chunk_cache_key
from .errors import AvatarEngineError, ValidationError
from .gpu import detect_gpu
from .stitch import stitch_chunks
from .text_processing import chunk_script
from .tts import AvatarTTS
from .validate import validate_video

logger = get_logger("avatar_engine.pipeline")


def _as_config(config) -> cfg.AvatarConfig:
    if config is None or isinstance(config, cfg.AvatarConfig):
        return config or cfg.AvatarConfig()
    if isinstance(config, dict):
        return cfg.AvatarConfig.from_dict(config)
    return cfg.AvatarConfig.from_yaml(config)


def generate_anchor_video(
    script_path: str | Path | None = None,
    script: str | None = None,
    anchor_id: str = "female",
    output_path: str | Path | None = None,
    config=None,
) -> dict:
    """Generate the final talking-anchor MP4 for one Urdu script.

    Returns the structured validation report (see ``validate.py``),
    extended with ``chunks`` and ``backends``.  Raises typed errors
    (``TTSGenerationError``, ``AvatarGenerationError``, ...) on stage
    failure.
    """
    started = time.time()
    conf = _as_config(config)
    conf.anchor_id = anchor_id

    if script is None and script_path is not None:
        script = Path(script_path).read_text(encoding="utf-8")
    if not script or not script.strip():
        raise AvatarEngineError("No script provided (script or script_path)")

    output_path = Path(output_path) if output_path else (
        cfg.OUTPUT_DIR / f"news_{anchor_id}.mp4")

    detect_gpu()  # log device before any inference
    backend = resolve_backend(conf)
    tts = AvatarTTS(conf)
    cache = ChunkCache()

    dual = conf.scene_mode == "dual"
    scene = None
    if dual:
        from .scene import build_scene
        scene = build_scene(conf.video_width, conf.video_height)
        if not hasattr(backend, "generate_dual"):
            from .backends.animated import AnimatedBackend
            backend = AnimatedBackend(conf)
        anchor_image = None
    else:
        anchor_image = anchor_image_path(anchor_id)

    chunks = chunk_script(
        script, min_sec=conf.chunk_min_sec, max_sec=conf.chunk_max_sec,
        words_per_min=conf.espeak_speed,
    )
    if not chunks:
        raise AvatarEngineError("Script produced no speech chunks")

    if dual:
        from .scene import speaker_schedule
        schedule = speaker_schedule(len(chunks), conf.first_speaker)
    else:
        schedule = [anchor_id] * len(chunks)
    logger.info("[PIPELINE] %d chunk(s) scene=%s schedule=%s backend=%s",
                len(chunks), "dual" if dual else "single",
                "->".join(schedule), backend.name)

    config_key = conf.cache_relevant_dict()
    chunk_videos: list[Path] = []

    for i, chunk in enumerate(chunks, start=1):
        speaker = schedule[i - 1]
        cache_anchor = f"dual:{speaker}" if dual else anchor_id
        tts_backend = tts.resolve_backend(speaker)
        key = chunk_cache_key(
            chunk.text, cache_anchor,
            f"{tts_backend}:{tts.backend_version(tts_backend)}",
            backend.version(), config_key,
        )
        hit = cache.lookup(key)
        if hit:
            logger.info("[CACHE] Hit %s for %s", key[:12], chunk.chunk_id)
            chunk_videos.append(hit["video"])
            continue

        work = cfg.OUTPUT_DIR / "work" / key[:12]
        work.mkdir(parents=True, exist_ok=True)

        # ---- TTS stage (voice matches the speaking anchor) -------------
        t0 = time.time()
        logger.info("[TTS] Generating chunk %d/%d (speaker=%s)",
                    i, len(chunks), speaker)
        tts_chunk = tts.generate_speech(
            chunk.text, speaker, work / "audio.wav", conf)
        logger.info("[TTS] Complete: %.2f sec", time.time() - t0)

        # ---- avatar / lip-sync stage ------------------------------------
        t0 = time.time()
        logger.info("[AVATAR] Generating chunk %d/%d (%s, speaker=%s)",
                    i, len(chunks), backend.name, speaker)
        if dual:
            backend.generate_dual(
                scene.canvas, scene.faces, speaker,
                tts_chunk.audio_path, work / "video.mp4",
                tts_chunk.duration_sec,
            )
        else:
            backend.generate(
                anchor_image, tts_chunk.audio_path,
                work / "video.mp4", tts_chunk.duration_sec,
            )
        logger.info("[AVATAR] Complete: %.1f sec", time.time() - t0)

        stored = cache.store(key, tts_chunk.audio_path, work / "video.mp4")
        chunk_videos.append(stored["video"])

    # ---- stitch + final encode ------------------------------------------
    stitched = stitch_chunks(chunk_videos, output_path, conf)

    # ---- validation -------------------------------------------------------
    report = validate_video(stitched, conf, strict=True)
    report["chunks"] = len(chunks)
    report["anchor_id"] = "dual" if dual else anchor_id
    report["scene"] = "dual" if dual else "single"
    if dual:
        report["speakers"] = schedule
    report["backends"] = {
        "tts": tts.resolve_backend(schedule[0]),
        "avatar": backend.name,
    }
    report["elapsed_sec"] = round(time.time() - started, 1)
    logger.info("[VALIDATE] A/V sync: %s", "OK" if report["sync_ok"] else "FAIL")
    logger.info("[PIPELINE] Done in %.1fs -> %s",
                report["elapsed_sec"], output_path)
    return report


__all__ = ["generate_anchor_video"]
