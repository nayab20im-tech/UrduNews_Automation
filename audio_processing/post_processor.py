"""
audio_processing/post_processor.py
=====================================
Stage 2.12 -- Audio Post-Processing.

Takes each article's TTS WAV (Stage 2.11) and applies the four boxes
from the architecture, in order:

1. **Noise reduction**     -- envelope-following noise gate with smooth
                              attack/release to avoid zipper noise:
                              samples below a quiet floor are attenuated
                              rather than hard-zeroed, with gain ramps
                              at speech/silence transitions.
2. **Pause/silence adjustment** -- runs of silence longer than a cap are
                              shortened to it (with fade in/out at splice
                              points), leading/trailing silence is trimmed
                              with a preserved speech margin.
3. **Volume normalization** -- peak-normalized to a broadcast-safe
                              target so every article plays at the same
                              loudness regardless of the TTS backend.
4. **Background music (optional)** -- if `BACKGROUND_MUSIC_PATH` points
                              at a WAV of the same sample rate, it is
                              looped under the voice at a low gain.

Pure-stdlib DSP over `array('h')` (no numpy/audioop): frame-wise,
channel-aware, clamps to 16-bit range. Output goes to a separate
`audio_processed/` media folder, leaving the raw TTS files intact.
"""

from __future__ import annotations

import array
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from data_acquisition.utils.logger import get_logger

from .wav_io import WavData, clamp16, read_wav, write_wav

logger = get_logger("pipeline.audio")

_MAX16 = 32767


@dataclass
class ProcessedAudioResult:
    article_id: int
    source_path: str = ""
    audio_path: str = ""
    duration_before_sec: float = 0.0
    duration_after_sec: float = 0.0
    peak_before: float = 0.0
    peak_after: float = 0.0
    silence_removed_ms: int = 0
    bgm_applied: bool = False
    status: str = "ok"                # ok | empty_audio | missing_file | error


class AudioPostProcessor:
    def __init__(
        self,
        out_dir: Path | str,
        target_peak: float = 0.85,       # broadcast-safe peak (0..1)
        noise_gate: float = 0.01,        # absolute level floor (0..1)
        max_silence_ms: int = 700,       # longer pauses get shortened
        trim_silence: bool = True,
        bgm_path: str = "",
        bgm_gain: float = 0.08,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.target_peak = target_peak
        self.noise_gate = noise_gate
        self.max_silence_ms = max_silence_ms
        self.trim_silence = trim_silence
        self.bgm_path = bgm_path
        self.bgm_gain = bgm_gain

    # -- public API -----------------------------------------------------------
    def process(self, article_id: int, audio_path: str | Path) -> ProcessedAudioResult:
        try:
            src = Path(audio_path)
            if not src.exists():
                return ProcessedAudioResult(article_id, str(src), status="missing_file")

            wav = read_wav(src)
            if len(wav.samples) == 0:
                return ProcessedAudioResult(article_id, str(src), status="empty_audio")

            gate = int(self.noise_gate * _MAX16)
            peak_before = max(abs(s) for s in wav.samples) / _MAX16
            duration_before = wav.duration_sec

            samples = _smooth_gate(wav.samples, gate, wav.rate)
            samples, removed_ms = _collapse_silence(
                samples, wav.channels, wav.rate, gate, self.max_silence_ms,
                margin_frames=int(wav.rate * 0.01),  # 10 ms margin
            )
            trimmed_frames = 0
            if self.trim_silence:
                samples, trimmed_frames = _trim_silence(
                    samples, wav.channels, wav.rate, gate, keep_ms=120)
            removed_ms += int(trimmed_frames * 1000 / max(wav.rate, 1))
            samples = _normalize(samples, self.target_peak)

            bgm_applied = False
            if self.bgm_path:
                samples, bgm_applied = _mix_background_music(
                    samples, wav.channels, wav.rate, self.bgm_path, self.bgm_gain)

            out_path = self.out_dir / f"article_{article_id}_processed.wav"
            write_wav(out_path, samples, wav.rate, wav.channels)
            peak_after = (max(abs(s) for s in samples) / _MAX16) if samples else 0.0
            nch = max(wav.channels, 1)
            duration_after = len(samples) / (wav.rate * nch)

            logger.debug(
                "Article %d: %.2fs -> %.2fs, peak %.3f -> %.3f, "
                "silence -%dms, bgm=%s",
                article_id, duration_before, duration_after,
                peak_before, peak_after, removed_ms, bgm_applied,
            )

            return ProcessedAudioResult(
                article_id, str(src), str(out_path),
                round(duration_before, 3), round(duration_after, 3),
                round(peak_before, 4), round(peak_after, 4),
                removed_ms, bgm_applied, "ok",
            )
        except Exception as exc:
            logger.error("Audio post-processing failed for article_id=%s: %s", article_id, exc)
            return ProcessedAudioResult(article_id, str(audio_path), status="error")

    def run(self, audios: List[dict]) -> List[ProcessedAudioResult]:
        """`audios`: list of {"article_id", "audio_path"} from Stage 2.11."""
        results = [self.process(a["article_id"], a.get("audio_path") or "") for a in audios]
        ok = sum(1 for r in results if r.status == "ok")
        logger.info("Audio post-processing: %d/%d ok", ok, len(results))
        return results


# -- DSP primitives (frame-wise, channel-aware) ---------------------------------

def _frame_levels(samples: array.array, nch: int) -> List[int]:
    """Max absolute level of each frame across all channels."""
    if nch <= 1:
        return [abs(s) for s in samples]
    levels = []
    for i in range(0, len(samples) - nch + 1, nch):
        levels.append(max(abs(s) for s in samples[i:i + nch]))
    return levels


def _smooth_gate(
    samples: array.array, gate: int, rate: int,
    attack_ms: float = 5.0, release_ms: float = 30.0,
) -> array.array:
    """Envelope-following noise gate with smooth attack/release.

    Instead of hard-zeroing sub-gate samples (which creates zipper noise
    at speech boundaries), a gain envelope ramps up/down over short
    attack/release windows.  Silence is still heavily attenuated but
    transitions are click-free.
    """
    n = len(samples)
    if n == 0:
        return array.array("h")

    attack_frames = max(1, int(rate * attack_ms / 1000))
    release_frames = max(1, int(rate * release_ms / 1000))
    attack_step = 1.0 / attack_frames
    release_step = 1.0 / release_frames

    out = array.array("h", [0] * n)
    gain = 0.0
    for i in range(n):
        above = abs(samples[i]) >= gate
        if above:
            gain = min(1.0, gain + attack_step)
        else:
            gain = max(0.0, gain - release_step)
        out[i] = clamp16(samples[i] * gain)
    return out


def _fade_region(
    samples: array.array, nch: int, start_frame: int, end_frame: int,
    fade_ms: int = 10,
) -> array.array:
    """Apply a short linear fade in/out at region boundaries to avoid
    clicks when splicing audio segments together."""
    fade_frames = min(fade_ms, (end_frame - start_frame) // 2)
    if fade_frames <= 0:
        return samples
    out = array.array("h", samples)
    for j in range(fade_frames):
        g = j / fade_frames
        for c in range(nch):
            idx = (start_frame + j) * nch + c
            if idx < len(out):
                out[idx] = clamp16(out[idx] * g)
    for j in range(fade_frames):
        g = j / fade_frames
        for c in range(nch):
            idx = (end_frame - 1 - j) * nch + c
            if 0 <= idx < len(out):
                out[idx] = clamp16(out[idx] * g)
    return out


def _apply_noise_gate(samples: array.array, gate: int) -> array.array:
    return array.array("h", (0 if abs(s) < gate else s for s in samples))


def _collapse_silence(
    samples: array.array, nch: int, rate: int, gate: int, max_silence_ms: int,
    margin_frames: int = 0,
) -> Tuple[array.array, int]:
    """Shorten any silent run longer than `max_silence_ms` down to it.

    `margin_frames` preserves a small number of samples around each
    speech region boundary, preventing abrupt cuts at the start/end of
    utterances.
    """
    levels = _frame_levels(samples, nch)
    max_frames = int(rate * max_silence_ms / 1000)
    keep: List[Tuple[int, int]] = []          # (start_frame, end_frame)
    i, n, removed = 0, len(levels), 0
    while i < n:
        if levels[i] >= gate:
            j = i
            while j < n and levels[j] >= gate:
                j += 1
            keep.append((i, j))
            i = j
        else:
            j = i
            while j < n and levels[j] < gate:
                j += 1
            run = j - i
            if run > max_frames:
                removed += run - max_frames
                keep.append((i, i + max_frames))
            else:
                keep.append((i, j))
            i = j
    # Apply margin: preserve a few frames around each speech region.
    if margin_frames > 0 and keep:
        expanded: List[Tuple[int, int]] = []
        for a, b in keep:
            expanded.append((max(0, a - margin_frames), min(n, b + margin_frames)))
        # Merge any overlapping regions.
        merged: List[Tuple[int, int]] = [expanded[0]]
        for a, b in expanded[1:]:
            if a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        keep = merged
    out = array.array("h")
    fade_frames = min(int(rate * 0.008), 128)   # 8 ms fade at splice
    prev_end = 0
    for a, b in keep:
        # Fade in at region start if it follows a gap.
        if a > prev_end and fade_frames > 0:
            out.extend(_fade_region(
                samples[a * nch:b * nch], nch, 0,
                min(b - a, fade_frames), fade_ms=int(fade_frames * 1000 / max(rate, 1)),
            ))
            out.extend(samples[(a + min(b - a, fade_frames)) * nch:b * nch])
        else:
            out.extend(samples[a * nch:b * nch])
        prev_end = b
    removed_ms = int(removed * 1000 / max(rate, 1))
    return out, removed_ms


def _trim_silence(
    samples: array.array, nch: int, rate: int, gate: int, keep_ms: int = 120,
) -> Tuple[array.array, int]:
    """Remove leading/trailing silence, keeping a small margin (frames removed)."""
    levels = _frame_levels(samples, nch)
    n = len(levels)
    start = next((i for i in range(n) if levels[i] >= gate), n)
    end = next((i for i in range(n - 1, -1, -1) if levels[i] >= gate), -1) + 1
    margin = int(rate * keep_ms / 1000)
    start = max(0, start - margin)
    end = min(n, end + margin)
    if end <= start:
        return array.array("h"), n
    return samples[start * nch:end * nch], n - (end - start)


def _normalize(samples: array.array, target_peak: float) -> array.array:
    peak = max((abs(s) for s in samples), default=0)
    if peak == 0:
        return samples
    factor = (target_peak * _MAX16) / peak
    if abs(factor - 1.0) < 0.001:
        return samples
    return array.array("h", (clamp16(s * factor) for s in samples))


def _mix_background_music(
    voice: array.array, nch: int, rate: int, bgm_path: str, bgm_gain: float,
) -> Tuple[array.array, bool]:
    """Loop same-rate BGM under the voice at low gain. Skips on any mismatch."""
    try:
        music = read_wav(bgm_path)
        if music.rate != rate or len(music.samples) == 0:
            logger.warning("BGM sample-rate mismatch (%s vs %s); skipping music",
                           music.rate, rate)
            return voice, False
        # downmix BGM to mono once, then tile to voice channel layout
        if music.channels > 1:
            mono = array.array("h", (
                int(sum(music.samples[i:i + music.channels]) / music.channels)
                for i in range(0, len(music.samples), music.channels)))
        else:
            mono = music.samples
        out = array.array("h", (0 for _ in range(len(voice))))
        mlen = len(mono)
        for i in range(len(voice) // nch):
            m = mono[i % mlen] * bgm_gain
            for c in range(nch):
                out[i * nch + c] = clamp16(voice[i * nch + c] + m)
        return out, True
    except Exception as exc:
        logger.warning("BGM mix failed (%s); skipping music", exc)
        return voice, False
