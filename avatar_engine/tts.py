"""
avatar_engine/tts.py
=======================
Modular Urdu TTS layer for the avatar subsystem.

Public contract (the rest of the pipeline only sees this):

    generate_speech(text, voice_id, output_path, config) -> TTSChunk

``voice_id`` is ``male`` or ``female``.  The backend is replaceable
without touching the avatar pipeline (``tts_backend`` in config):

* **piper** — Piper neural TTS with the ``ur_PK-fasih-medium`` Urdu
  voice (rhasspy/piper-voices, MIT).  Highest quality; requires the
  onnx model + piper binary.  Only a male Urdu Piper voice exists
  today, so female requests fall through to espeak with a raised pitch.
* **espeak** — espeak-ng ``-v ur`` (verified on this host, zero
  download).  Male/female are differentiated by pitch/speed.
* **xtts** — Coqui XTTS-v2 (CPML — non-commercial; experimental only,
  never the auto default).
* **placeholder** — deterministic tone synth from ``tts/wav_synth``.

Every generated WAV is normalized: mono, target sample rate, peak
loudness 0.85, trailing silence pad so the mouth visibly closes
between stitched chunks.
"""

from __future__ import annotations

import array
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from data_acquisition.utils.logger import get_logger

from audio_processing.wav_io import read_wav, write_wav, clamp16
from tts import wav_synth

from . import config as cfg
from .errors import TTSGenerationError

logger = get_logger("avatar_engine.tts")

_TARGET_PEAK = 0.85


@dataclass
class TTSChunk:
    audio_path: str
    duration_sec: float
    sample_rate: int
    backend_used: str
    voice_id: str


class AvatarTTS:
    """Urdu speech synthesis with per-gender voices and normalization."""

    def __init__(self, config: cfg.AvatarConfig | None = None):
        self.config = config or cfg.AvatarConfig()
        self._version_cache: dict[str, str] = {}

    # -- backend resolution --------------------------------------------------
    def resolve_backend(self, voice_id: str) -> str:
        want = self.config.tts_backend
        if want not in ("auto", ""):
            return want
        # Piper only serves a male Urdu voice (fasih) today.
        if voice_id == "male" and self._piper_available():
            return "piper"
        if shutil.which("espeak-ng") or shutil.which("espeak"):
            return "espeak"
        return "placeholder"

    def _piper_available(self) -> bool:
        return (
            shutil.which(cfg.PIPER_BINARY) is not None
            and Path(self.config.piper_model_male).exists()
        )

    def backend_version(self, backend: str) -> str:
        if backend in self._version_cache:
            return self._version_cache[backend]
        version = backend
        try:
            if backend == "espeak":
                bin_ = shutil.which("espeak-ng") or shutil.which("espeak")
                out = subprocess.run(
                    [bin_, "--version"], capture_output=True, timeout=30,
                ).stdout.decode(errors="replace").splitlines()[0].strip()
                version = out or "espeak"
            elif backend == "piper":
                out = subprocess.run(
                    [cfg.PIPER_BINARY, "--version"], capture_output=True,
                    timeout=30,
                ).stdout.decode(errors="replace").strip()
                version = out or "piper"
        except Exception:
            pass
        self._version_cache[backend] = version
        return version

    # -- public API ------------------------------------------------------------
    def generate_speech(
        self,
        text: str,
        voice_id: str,
        output_path: Path | str,
        config: cfg.AvatarConfig | None = None,
    ) -> TTSChunk:
        """Synthesize one Urdu chunk to a normalized WAV.

        Raises ``TTSGenerationError`` on failure — the pipeline must
        never silently continue with missing audio.
        """
        config = config or self.config
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not text or not text.strip():
            raise TTSGenerationError("Empty text passed to TTS")

        backend = self.resolve_backend(voice_id)
        raw_path = output_path.with_suffix(".raw.wav")
        ok = False
        if backend == "piper":
            ok = self._synth_piper(text, raw_path)
        if not ok and backend in ("piper", "espeak", "auto"):
            ok = self._synth_espeak(text, raw_path, voice_id, config)
            if ok:
                backend = "espeak"
        if not ok and backend == "xtts":
            ok = self._synth_xtts(text, raw_path, voice_id)
        if not ok:
            if backend not in ("placeholder", "auto"):
                logger.warning(
                    "[TTS] backend %r unavailable; using placeholder", backend)
            wav_synth.synthesize_script(text, raw_path)
            backend = "placeholder"

        if not raw_path.exists():
            raise TTSGenerationError(
                f"TTS backend {backend!r} produced no audio for voice {voice_id!r}")

        normalized = self._normalize(raw_path, output_path)
        if raw_path.exists() and raw_path != output_path:
            raw_path.unlink(missing_ok=True)

        wav = read_wav(output_path)
        logger.info(
            "[TTS] %s voice=%s backend=%s duration=%.2fs",
            output_path.name, voice_id, backend, wav.duration_sec,
        )
        return TTSChunk(
            audio_path=str(output_path),
            duration_sec=round(wav.duration_sec, 3),
            sample_rate=wav.rate,
            backend_used=backend,
            voice_id=voice_id,
        )

    # -- backends ------------------------------------------------------------
    def _synth_espeak(
        self, text: str, out_path: Path, voice_id: str,
        config: cfg.AvatarConfig,
    ) -> bool:
        binary = shutil.which("espeak-ng") or shutil.which("espeak")
        if not binary:
            return False
        pitch = (
            cfg.ESPEAK_PITCH_FEMALE if voice_id == "female"
            else cfg.ESPEAK_PITCH_MALE
        )
        cmd = [
            binary, "--stdin", "-v", "ur",
            "-s", str(config.espeak_speed), "-p", str(pitch),
            "-w", str(out_path),
        ]
        proc = subprocess.run(
            cmd, input=text.encode("utf-8"), capture_output=True, timeout=300,
        )
        return proc.returncode == 0 and out_path.exists()

    def _synth_piper(self, text: str, out_path: Path) -> bool:
        if not self._piper_available():
            return False
        cmd = [
            cfg.PIPER_BINARY, "--model", self.config.piper_model_male,
            "--output_file", str(out_path),
        ]
        proc = subprocess.run(
            cmd, input=text.encode("utf-8"), capture_output=True, timeout=600,
        )
        return proc.returncode == 0 and out_path.exists()

    def _synth_xtts(self, text: str, out_path: Path, voice_id: str) -> bool:
        """Experimental Coqui XTTS-v2 — CPML license (non-commercial)."""
        model_path = self.config.extra.get("xtts_model_path", "")
        if not model_path or not Path(model_path).exists():
            return False
        try:
            from TTS.api import TTS
            tts = TTS(model_path=model_path).to("cpu")
            tts.tts_to_file(text=text, file_path=str(out_path), language="ur")
            return out_path.exists()
        except Exception as exc:
            logger.warning("[TTS] XTTS failed: %s", exc)
            return False

    # -- normalization ---------------------------------------------------------
    def _normalize(self, raw_path: Path, out_path: Path) -> Path:
        """Mono + target rate (FFmpeg) → peak normalize + silence pad."""
        resampled = out_path.with_suffix(".rs.wav")
        cmd = [
            cfg.FFMPEG_BINARY, "-y", "-i", str(raw_path),
            "-ac", "1", "-ar", str(cfg.TARGET_SAMPLE_RATE),
            str(resampled),
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=120)
        if proc.returncode != 0 or not resampled.exists():
            raise TTSGenerationError(
                f"FFmpeg sample-rate normalization failed: "
                f"{proc.stderr.decode(errors='replace')[:300]}")

        wav = read_wav(resampled)
        samples = array.array("h", wav.samples)
        peak = max((abs(s) for s in samples), default=0)
        if peak > 0:
            gain = _TARGET_PEAK * 32767.0 / peak
            samples = array.array(
                "h", (clamp16(s * gain) for s in samples))

        # Trailing silence pad so stitched chunks show a closed mouth
        # and a breath between sentences.
        pad_frames = int(cfg.TARGET_SAMPLE_RATE * cfg.CHUNK_SILENCE_PAD_MS / 1000)
        samples.extend(array.array("h", [0] * pad_frames))

        write_wav(out_path, samples, cfg.TARGET_SAMPLE_RATE, 1)
        resampled.unlink(missing_ok=True)
        return out_path
