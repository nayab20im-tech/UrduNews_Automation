"""
tts/wav_synth.py
===================
Deterministic offline stand-in voice for Stage 2.11.

When no real TTS engine is available on the host (no espeak-ng binary,
no Piper model, no Coqui/XTTS install -- the normal state of this
environment), the pipeline still has to produce a *real, well-formed
WAV artifact* per article so the downstream Video Production section
(3.4 Video Composition) has audio to consume and the whole chain stays
testable offline. This module synthesizes that placeholder: one
speech-like tone burst per script sentence (length proportional to the
sentence's word count, pitch varying per sentence to mimic prosody),
separated by inter-sentence pauses.

It is deliberately honest: `TTSEngine` reports `backend_used=
"placeholder"` in the DB so placeholder audio is never mistaken for a
natural voice, and pointing the config at any real backend (espeak /
Piper / XTTS) replaces it with zero other code changes -- the same
swap-out philosophy used for translation (2.6) and script writing (2.9).
"""

from __future__ import annotations

import array
import math
from pathlib import Path

from audio_processing.wav_io import write_wav
from preprocessing.normalizer import split_sentences

DEFAULT_RATE = 16000
DEFAULT_SEC_PER_WORD = 0.22
DEFAULT_PAUSE_MS = 350
_AMPLITUDE = 0.6 * 32767
_FADE_MS = 40


def synthesize_script(
    text: str,
    out_path: Path | str,
    rate: int = DEFAULT_RATE,
    sec_per_word: float = DEFAULT_SEC_PER_WORD,
    pause_ms: int = DEFAULT_PAUSE_MS,
) -> float:
    """Write a placeholder WAV for `text`; returns its duration in seconds."""
    sentences = [s for s in split_sentences(text) if s.strip()] or ([text] if text.strip() else [])
    pause_frames = int(rate * pause_ms / 1000)
    fade_frames = int(rate * _FADE_MS / 1000)

    chunks: list[array.array] = []
    for idx, sentence in enumerate(sentences):
        n_words = max(len(sentence.split()), 1)
        n_frames = int(rate * max(0.4, n_words * sec_per_word))
        # per-sentence pitch varies like prosody; Urdu news reads low-ish.
        freq = 170.0 + (idx % 5) * 25.0
        chunks.append(_tone(n_frames, freq, rate, fade_frames))
        chunks.append(array.array("h", bytes(pause_frames * 2)))

    if not chunks:
        chunks.append(array.array("h", bytes(int(rate * 0.5) * 2)))

    samples = array.array("h")
    for chunk in chunks:
        samples.extend(chunk)

    write_wav(out_path, samples, rate, channels=1)
    return len(samples) / rate


def _tone(n_frames: int, freq: float, rate: int, fade_frames: int) -> array.array:
    """One speech-like burst: sine + 2nd harmonic, with fade in/out."""
    # Build a single period and tile it (much faster than per-sample sin).
    period = max(int(rate / freq), 8)
    one = array.array(
        "h",
        (
            int(
                _AMPLITUDE
                * 0.7
                * (math.sin(2 * math.pi * freq * i / rate)
                   + 0.35 * math.sin(4 * math.pi * freq * i / rate))
            )
            for i in range(period)
        ),
    )
    repeats, rem = divmod(n_frames, period)
    buf = array.array("h")
    if repeats:
        buf.extend(one * repeats)
    buf.extend(one[:rem])

    # fade edges so bursts don't click
    for i in range(min(fade_frames, len(buf))):
        gain = i / fade_frames
        buf[i] = int(buf[i] * gain)
        buf[len(buf) - 1 - i] = int(buf[len(buf) - 1 - i] * gain)
    return buf
