"""
audio_processing/wav_io.py
=============================
Minimal, dependency-free WAV read/write helpers shared by Stage 2.11
(placeholder synthesis) and Stage 2.12 (post-processing).

Pure stdlib (`wave` + `array`) -- no numpy/audioop, so the audio stages
run in the same bare environment as every other stage and survive the
Python 3.13 removal of `audioop`. Samples are always returned as a
flat 16-bit signed (`array('h')`) buffer, channel-interleaved; 8/24/32-bit
input files are converted to 16-bit on read.
"""

from __future__ import annotations

import array
import wave
from pathlib import Path
from typing import NamedTuple

_MAX16 = 32767


class WavData(NamedTuple):
    samples: array.array     # flat 'h' buffer, channel-interleaved
    rate: int
    channels: int

    @property
    def frames(self) -> int:
        return len(self.samples) // max(self.channels, 1)

    @property
    def duration_sec(self) -> float:
        return self.frames / max(self.rate, 1)


def read_wav(path: Path | str) -> WavData:
    """Read any common PCM WAV into a flat 16-bit sample buffer."""
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    if width == 2:
        samples = array.array("h")
        samples.frombytes(raw)
    elif width == 1:  # unsigned 8-bit -> signed 16-bit
        samples = array.array("h", ((b - 128) << 8 for b in raw))
    elif width == 3:  # packed 24-bit -> 16-bit (drop the low byte)
        samples = array.array("h", (
            int.from_bytes(raw[i:i + 3], "little", signed=True) >> 8
            for i in range(0, len(raw), 3)
        ))
    elif width == 4:  # 32-bit -> 16-bit (truncate high bits)
        buf = array.array("i")
        buf.frombytes(raw)
        samples = array.array("h", (max(-_MAX16 - 1, min(_MAX16, v >> 16)) for v in buf))
    else:
        raise ValueError(f"Unsupported WAV sample width: {width}")

    # byte order: wave module returns native-endian raw frames on most
    # platforms; if the file was big-endian the wave module already
    # exposes raw bytes as stored -- assume little-endian (standard for
    # the TTS tools this pipeline shells out to).
    if not _is_little_endian():
        samples.byteswap()
    return WavData(samples, rate, channels)


def write_wav(path: Path | str, samples: array.array, rate: int, channels: int = 1) -> None:
    """Write a flat 16-bit sample buffer as a PCM WAV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(samples.tobytes())


def _is_little_endian() -> bool:
    import sys

    return sys.byteorder == "little"


def clamp16(value: float) -> int:
    return int(max(-_MAX16 - 1, min(_MAX16, value)))
