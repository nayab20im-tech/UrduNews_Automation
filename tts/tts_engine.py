"""
tts/tts_engine.py
====================
Stage 2.11 -- Text-to-Speech (Urdu Voice Generation).

Reads each article's structured broadcast script (Stage 2.10
`full_script`) and writes one WAV file per article into the media
storage area (architecture box 6: "Media Storage (Audio, ...)").

Backends, resolved by `TTS_BACKEND` (`auto` tries them in this order):

1. **xtts**      -- Coqui TTS / XTTS-v2 (natural multilingual voice),
                    needs a local model dir + speaker sample configured
                    (`XTTS_MODEL_PATH`, `XTTS_SPEAKER_WAV`).
2. **piper**     -- Piper neural TTS binary + local model
                    (`PIPER_MODEL_PATH`), text on stdin.
3. **edge_tts**  -- Microsoft Edge Neural TTS (online); default female
                    Urdu voice ``ur-PK-UzmaNeural``.  Requires internet.
4. **espeak**    -- espeak-ng system binary; ships an Urdu voice
                    (`-v ur`) and needs no downloads -- the preferred
                    real-voice offline option.
5. **placeholder** -- deterministic tone-burst synth (`wav_synth`) so
                    the pipeline always produces a valid WAV artifact
                    even on a bare host (this environment).

Emotion/tone control maps to each backend's prosody knobs: espeak
`-s` (speed wpm) / `-p` (pitch 0-99) / `-a` (amplitude), configurable
via `TTS_SPEED` / `TTS_PITCH`; XTTS receives the speaker sample (which
carries the voice style). Every result records `backend_used` so the
audio's provenance is always auditable.

Before synthesis, Urdu text is preprocessed: broadcast section labels
(`ہیڈلائن:`, `تعارف:`, etc.) that are visual-only in the script are
stripped so they are not spoken aloud, Unicode is NFKC-normalized, and
whitespace is collapsed.
"""

from __future__ import annotations

import asyncio
import re
import shutil
import subprocess
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import List

from data_acquisition.utils.logger import get_logger

from . import wav_synth
from audio_processing.wav_io import read_wav

logger = get_logger("pipeline.tts")

# Urdu broadcast section labels emitted by Stage 2.10's ScriptStructurer.
# These are visual-only markers in `full_script` and must not be spoken.
_URDU_LABEL_RE = re.compile(
    r"^(?:آغاز|ہیڈلائن|تعارف|مرکزی خبر|اہم نکات|اختتامیہ)\s*:?\s*",
    re.MULTILINE,
)
# Strip bullet markers from key-point lines (e.g. "- نکتہ")
_BULLET_RE = re.compile(r"^-\s+", re.MULTILINE)


def clean_text_for_tts(text: str) -> str:
    """Prepare Urdu script text for speech synthesis.

    Strips broadcast section labels (``ہیڈلائن:``, ``تعارف:``, etc.)
    emitted by Stage 2.10 that should not be spoken aloud, applies
    Unicode NFKC normalization for consistent Urdu rendering, and
    collapses whitespace.
    """
    if not text or not text.strip():
        return ""
    # Unicode NFKC: fold Arabic presentation forms into canonical code
    # points so TTS engines see a single consistent representation.
    text = unicodedata.normalize("NFKC", text)
    # Strip broadcast section labels (visual-only, not spoken).
    text = _URDU_LABEL_RE.sub("", text)
    # Strip bullet markers (- نکتہ → نکتہ)
    text = _BULLET_RE.sub("", text)
    # Collapse whitespace, drop blank lines.
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


@dataclass
class TTSResult:
    article_id: int
    audio_path: str = ""
    duration_sec: float = 0.0
    sample_rate: int = 0
    backend_used: str = ""
    voice: str = "ur"
    status: str = "ok"                # ok | empty_text | error


class TTSEngine:
    def __init__(
        self,
        audio_dir: Path | str,
        backend: str = "auto",        # auto | xtts | piper | espeak | placeholder
        voice: str = "ur",
        speed: int = 160,             # espeak words-per-minute
        pitch: int = 50,              # espeak pitch 0-99
        piper_model_path: str = "",
        piper_binary: str = "piper",
        xtts_model_path: str = "",
        xtts_speaker_wav: str = "",
    ):
        self.audio_dir = Path(audio_dir)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.voice = voice
        self.speed = speed
        self.pitch = pitch
        self.piper_model_path = piper_model_path
        self.piper_binary = piper_binary
        self.xtts_model_path = xtts_model_path
        self.xtts_speaker_wav = xtts_speaker_wav
        self.edge_voice = "ur-PK-UzmaNeural"   # Female Urdu neural voice
        self._xtts = None             # lazy-loaded Coqui model, if any
        import os
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY", "")
        self.elevenlabs_voice_id = os.getenv("ELEVENLABS_VOICE_ID", "")

    # -- backend resolution ----------------------------------------------------
    def resolve_backend(self) -> str:
        if self.backend != "auto":
            return self.backend
        if self.elevenlabs_api_key:
            return "elevenlabs"
        if self.xtts_model_path:
            return "xtts"
        if self.piper_model_path and shutil.which(self.piper_binary):
            return "piper"
        # Try edge-tts (online Microsoft Neural TTS — best natural voice)
        try:
            import edge_tts as _et  # noqa: F401
            return "edge_tts"
        except ImportError:
            pass
        if shutil.which("espeak-ng") or shutil.which("espeak"):
            return "espeak"
        return "placeholder"

    # -- public API -----------------------------------------------------------
    def synthesize(self, article_id: int, text: str) -> TTSResult:
        try:
            cleaned = clean_text_for_tts(text)
            if not cleaned:
                return TTSResult(article_id, status="empty_text")

            out_path = self.audio_dir / f"article_{article_id}.wav"
            backend = self.resolve_backend()

            if backend == "elevenlabs" and self._synth_elevenlabs(cleaned, out_path):
                used = "elevenlabs"
            elif backend == "xtts" and self._synth_xtts(cleaned, out_path):
                used = "xtts"
            elif backend == "piper" and self._synth_piper(cleaned, out_path):
                used = "piper"
            elif backend == "edge_tts" and self._synth_edge_tts(cleaned, out_path):
                used = "edge_tts"
            elif backend == "espeak" and self._synth_espeak(cleaned, out_path):
                used = "espeak"
            else:
                if backend not in ("placeholder", "auto"):
                    logger.warning(
                        "TTS backend %r unavailable for article_id=%s; using placeholder",
                        backend, article_id,
                    )
                wav_synth.synthesize_script(cleaned, out_path)
                used = "placeholder"

            wav = read_wav(out_path)
            return TTSResult(
                article_id, str(out_path), round(wav.duration_sec, 3),
                wav.rate, used, self.voice, "ok",
            )
        except Exception as exc:
            logger.error("TTS failed for article_id=%s: %s", article_id, exc)
            return TTSResult(article_id, status="error")

    def run(self, articles: List[dict]) -> List[TTSResult]:
        """`articles`: list of {"article_id", "text"} (the 2.10 full_script)."""
        results = [self.synthesize(a["article_id"], a.get("text") or "") for a in articles]
        by_backend: dict = {}
        for r in results:
            by_backend[r.backend_used or "none"] = by_backend.get(r.backend_used or "none", 0) + 1
        logger.info("TTS complete for %d articles %s", len(results), by_backend)
        return results

    # -- backend implementations -------------------------------------------------
    def _synth_espeak(self, text: str, out_path: Path) -> bool:
        binary = shutil.which("espeak-ng") or shutil.which("espeak")
        if not binary:
            return False
        cmd = [
            binary, "--stdin", "-v", self.voice,
            "-s", str(self.speed), "-p", str(self.pitch), "-w", str(out_path),
        ]
        proc = subprocess.run(cmd, input=text.encode("utf-8"),
                              capture_output=True, timeout=300)
        return proc.returncode == 0 and out_path.exists()

    def _synth_edge_tts(self, text: str, out_path: Path) -> bool:
        """Synthesize using Microsoft Edge Neural TTS (online).

        Uses the ``ur-PK-UzmaNeural`` female Urdu voice by default.
        Converts the output MP3 to WAV using ffmpeg so the rest of the
        pipeline receives consistent WAV files.
        """
        try:
            import edge_tts
            import tempfile
            import os

            mp3_tmp = Path(tempfile.mktemp(suffix=".mp3"))
            voice = self.edge_voice

            async def _do_synth() -> None:
                communicate = edge_tts.Communicate(text, voice)
                await communicate.save(str(mp3_tmp))

            asyncio.run(_do_synth())

            if not mp3_tmp.exists():
                return False

            # Convert MP3 → WAV with ffmpeg for consistency
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg:
                proc = subprocess.run(
                    [ffmpeg, "-y", "-i", str(mp3_tmp),
                     "-ar", "22050", "-ac", "1", str(out_path)],
                    capture_output=True, timeout=60,
                )
                mp3_tmp.unlink(missing_ok=True)
                return proc.returncode == 0 and out_path.exists()
            else:
                # No ffmpeg: just move the mp3 (pipeline may handle it)
                mp3_tmp.rename(out_path)
                return out_path.exists()
        except Exception as exc:
            logger.warning("edge-tts backend failed: %s", exc)
            return False

    def _synth_piper(self, text: str, out_path: Path) -> bool:
        if not (self.piper_model_path and shutil.which(self.piper_binary)):
            return False
        cmd = [self.piper_binary, "--model", self.piper_model_path,
               "--output_file", str(out_path)]
        proc = subprocess.run(cmd, input=text.encode("utf-8"),
                              capture_output=True, timeout=600)
        return proc.returncode == 0 and out_path.exists()

    def _synth_xtts(self, text: str, out_path: Path) -> bool:
        """Optional Coqui/XTTS backend -- only if installed AND configured."""
        if not self.xtts_model_path:
            return False
        try:
            if self._xtts is None:
                try:
                    from TTS.api import TTS  # local install, never downloaded here
                except ImportError:
                    logger.warning(
                        "Coqui TTS package not installed; install with "
                        "'pip install TTS' for XTTS backend support"
                    )
                    return False
                model_path = Path(self.xtts_model_path)
                if not model_path.exists():
                    logger.warning(
                        "XTTS model path does not exist: %s", model_path
                    )
                    return False
                logger.info("Loading XTTS model from %s", model_path)
                self._xtts = TTS(model_path=str(model_path)).to("cpu")
            speaker_wav = self.xtts_speaker_wav or None
            if speaker_wav and not Path(speaker_wav).exists():
                logger.warning(
                    "XTTS speaker WAV not found: %s; synthesizing without it",
                    speaker_wav,
                )
                speaker_wav = None
            self._xtts.tts_to_file(
                text=text, file_path=str(out_path), language=self.voice,
                speaker_wav=speaker_wav,
            )
            return out_path.exists()
        except Exception as exc:
            logger.warning("XTTS backend failed (%s); falling back", exc)
            return False

    def _synth_elevenlabs(self, text: str, out_path: Path) -> bool:
        """Synthesize using ElevenLabs."""
        if not self.elevenlabs_api_key:
            return False
        
        try:
            from elevenlabs import ElevenLabs
            import tempfile
            import os

            client = ElevenLabs(api_key=self.elevenlabs_api_key)
            voice_id = self.elevenlabs_voice_id or "EXAVITQu4vr4xnSDxMaL"
            audio_gen = client.text_to_speech.convert(
                voice_id=voice_id,
                output_format="mp3_44100_128",
                text=text,
                model_id="eleven_multilingual_v2",
            )
            mp3_tmp = Path(tempfile.mktemp(suffix=".mp3"))
            with open(mp3_tmp, "wb") as f:
                for chunk in audio_gen:
                    if chunk:
                        f.write(chunk)
            
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg:
                proc = subprocess.run(
                    [ffmpeg, "-y", "-i", str(mp3_tmp),
                     "-ar", "22050", "-ac", "1", str(out_path)],
                    capture_output=True, timeout=60,
                )
                mp3_tmp.unlink(missing_ok=True)
                return proc.returncode == 0 and out_path.exists()
            else:
                mp3_tmp.rename(out_path)
                return out_path.exists()
        except Exception as exc:
            logger.warning("ElevenLabs backend failed: %s", exc)
            return False
