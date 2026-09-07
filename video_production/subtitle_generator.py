"""
video_production/subtitle_generator.py
========================================
Stage 3.5 — Urdu Subtitle Generation + Burn-in.

Generates time-synchronised Urdu subtitles from the existing Urdu
script (Stage 2.10 ``full_script`` or Stage 2.6 ``urdu_text``) and
burns them into the composed video using FFmpeg's ASS-subtitle filter.

Workflow
--------
1. Split the Urdu script into sentences (reuses the project's own
   ``preprocessing.normalizer.split_sentences``).
2. Distribute timing evenly across the audio duration (or use word-
   count-proportional timing for a better fit).
3. Write an ``.srt`` file (kept for debugging / reuse).
4. Convert the SRT to an ASS overlay with configurable font, size,
   colour, position and background box.
5. Burn the subtitles into the video via ``ffmpeg -vf subtitles=…``.

Outputs
-------
* ``subtitles_<id>.srt``   — standalone SRT file
* ``<video>_subtitled.mp4`` — video with hardcoded subtitles
"""

from __future__ import annotations

import os
import re
import subprocess
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from data_acquisition.utils.logger import get_logger

from . import config as vc

logger = get_logger("video_production.subtitles")


@dataclass
class SubtitleResult:
    article_id: int
    srt_path: str = ""
    subtitled_video_path: str = ""
    subtitle_count: int = 0
    status: str = "ok"          # ok | no_input | error


class SubtitleGenerator:
    """Generate Urdu subtitles and burn them into the news video."""

    def __init__(
        self,
        output_dir: Path | str = vc.SUBTITLES_DIR,
        font_path: str = vc.SUBTITLE_FONT_PATH or vc.URDU_FONT_PATH,
        font_size: int = vc.SUBTITLE_FONT_SIZE,
        position_y: int = vc.SUBTITLE_POSITION_Y,
        text_color: str = vc.SUBTITLE_TEXT_COLOR,
        bg_color: str = vc.SUBTITLE_BG_COLOR,
        bg_opacity: float = vc.SUBTITLE_BG_OPACITY,
        max_chars: int = vc.SUBTITLE_MAX_CHARS,
        ffmpeg: str = vc.FFMPEG_BINARY,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.font_path = font_path
        self.font_size = font_size
        self.position_y = position_y
        self.text_color = text_color
        self.bg_color = bg_color
        self.bg_opacity = bg_opacity
        self.max_chars = max_chars
        self.ffmpeg = ffmpeg

    # -- public API ----------------------------------------------------------
    def generate(
        self,
        article_id: int,
        urdu_script: str,
        video_path: str,
        audio_duration: float,
    ) -> SubtitleResult:
        try:
            if not urdu_script or not urdu_script.strip():
                return SubtitleResult(article_id, status="no_input")
            if not video_path or not Path(video_path).exists():
                return SubtitleResult(article_id, status="no_input")

            # Normalise Urdu text
            script = unicodedata.normalize("NFKC", urdu_script).strip()

            # Strip broadcast section labels (same as TTS does)
            script = _strip_broadcast_labels(script)

            # Split into sentences then break into broadcast-style 3-5 word chunks
            sentences = self._split_sentences(script)
            if not sentences:
                return SubtitleResult(article_id, status="no_input")
            chunks = self._chunk_sentences(sentences, max_words=5)

            # Build timed subtitle entries
            entries = self._time_sentences(chunks, audio_duration)

            # Write SRT (kept for debugging / reuse)
            srt_path = self.output_dir / f"subtitles_{article_id}.srt"
            self._write_srt(srt_path, entries)

            # Write ASS with explicit resolution + styling (reliable burn-in)
            ass_path = self.output_dir / f"subtitles_{article_id}.ass"
            self._write_ass(ass_path, entries)

            # Burn into video
            subtitled_path = self._burn_subtitles(article_id, video_path, ass_path)

            return SubtitleResult(
                article_id,
                srt_path=str(srt_path),
                subtitled_video_path=subtitled_path,
                subtitle_count=len(entries),
                status="ok",
            )
        except Exception as exc:
            logger.error("Subtitle generation failed for article %d: %s", article_id, exc)
            return SubtitleResult(article_id, status="error")

    def run(self, articles: List[dict]) -> List[SubtitleResult]:
        """articles: [{"article_id", "urdu_script", "video_path", "audio_duration"}]"""
        results = []
        for a in articles:
            r = self.generate(
                a["article_id"],
                a.get("urdu_script", ""),
                a.get("video_path", ""),
                a.get("audio_duration", 0.0),
            )
            results.append(r)
        ok = sum(1 for r in results if r.status == "ok")
        logger.info("Subtitle generation: %d/%d ok", ok, len(results))
        return results

    # -- sentence splitting --------------------------------------------------
    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split Urdu text into sentences using the project's own regex."""
        # Re-use the same sentence-enders as preprocessing.normalizer
        parts = re.split(r"(?<=[.!?۔؟])\s+", text.strip())
        sentences = [s.strip() for s in parts if s and s.strip()]
        return sentences

    @staticmethod
    def _chunk_sentences(sentences: List[str], max_words: int = 5) -> List[str]:
        """Break sentences into short broadcast-style display chunks.

        Each sentence is split into groups of at most ``max_words`` words
        so that subtitles change every 2–4 seconds rather than lingering
        for the full sentence duration (8–15 s).
        """
        chunks: List[str] = []
        for sent in sentences:
            words = sent.split()
            if len(words) <= max_words:
                chunks.append(sent)
            else:
                for i in range(0, len(words), max_words):
                    group = " ".join(words[i : i + max_words])
                    if group.strip():
                        chunks.append(group)
        return chunks

    # -- timing --------------------------------------------------------------
    def _time_sentences(
        self, sentences: List[str], total_duration: float,
    ) -> List[dict]:
        """Assign start/end times proportional to word count."""
        if total_duration <= 0:
            total_duration = max(len(sentences) * 3.0, 3.0)

        # Compute word weights for proportional timing
        weights = [max(len(s.split()), 1) for s in sentences]
        total_words = sum(weights)
        if total_words == 0:
            total_words = len(sentences)

        entries: List[dict] = []
        cursor = 0.0
        for i, (sent, w) in enumerate(zip(sentences, weights)):
            frac = w / total_words
            seg_dur = frac * total_duration
            # Minimum 0.5s per subtitle
            seg_dur = max(seg_dur, 0.5)
            start = cursor
            end = cursor + seg_dur
            # Clamp to total duration
            if end > total_duration:
                end = total_duration
            entries.append({
                "index": i + 1,
                "start": start,
                "end": end,
                "text": self._wrap_line(sent),
            })
            cursor = end
        return entries

    def _wrap_line(self, text: str) -> str:
        """Wrap long Urdu lines at ``max_chars`` boundary."""
        if len(text) <= self.max_chars:
            return text
        # Try to break at a word boundary
        mid = len(text) // 2
        # Search outward from mid for a space
        best = mid
        for offset in range(min(20, mid)):
            for pos in (mid + offset, mid - offset):
                if 0 < pos < len(text) and text[pos] == " ":
                    best = pos
                    break
            else:
                continue
            break
        return text[:best].rstrip() + "\n" + text[best:].lstrip()

    # -- SRT writer ----------------------------------------------------------
    @staticmethod
    def _write_srt(path: Path, entries: List[dict]) -> None:
        lines: List[str] = []
        for e in entries:
            lines.append(str(e["index"]))
            lines.append(
                f"{_fmt_srt_time(e['start'])} --> {_fmt_srt_time(e['end'])}"
            )
            lines.append(e["text"])
            lines.append("")  # blank separator
        path.write_text("\n".join(lines), encoding="utf-8-sig")

    # -- ASS writer ----------------------------------------------------------
    def _write_ass(self, path: Path, entries: List[dict]) -> None:
        """Write an ASS subtitle file with explicit resolution and styling.

        ASS gives exact control over PlayResX/PlayResY (matching the video),
        font size in real pixels, and a fixed bottom-centre position —
        avoiding the SRT→ASS conversion scaling problems in libass.
        """
        # Resolve font name: prefer the real Urdu-capable font's family name
        font_name = "Segoe UI"
        if self.font_path and Path(self.font_path).exists():
            font_name = _font_family_name(self.font_path) or "Segoe UI"

        primary = _hex_to_ass(self.text_color)
        outline_color = _hex_to_ass(self.bg_color)
        back_color = _hex_to_ass_opacity(self.bg_color, self.bg_opacity)

        # Fixed bottom-centre: Alignment=2, MarginV = distance from bottom
        margin_v = 80
        margin_lr = 60

        header = (
            "[Script Info]\n"
            "; Generated by UrduNews_AI video_production\n"
            "ScriptType: v4.00+\n"
            f"PlayResX: {vc.VIDEO_WIDTH}\n"
            f"PlayResY: {vc.VIDEO_HEIGHT}\n"
            "WrapStyle: 0\n"
            "ScaledBorderAndShadow: yes\n"
            "\n"
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding\n"
            f"Style: Default,{font_name},{self.font_size},{primary},&H000000FF,"
            f"{outline_color},{back_color},1,0,0,0,100,100,0,0,3,3,0,"
            f"2,{margin_lr},{margin_lr},{margin_v},1\n"
            "\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        )

        lines = [header]
        for e in entries:
            # Replace newlines in wrapped text with ASS line-break marker
            text = e["text"].replace("\n", "\\N")
            lines.append(
                f"Dialogue: 0,{_fmt_ass_time(e['start'])},{_fmt_ass_time(e['end'])},"
                f"Default,,0,0,0,,{text}\n"
            )
        path.write_text("".join(lines), encoding="utf-8-sig")

    # -- burn-in via FFmpeg --------------------------------------------------
    def _burn_subtitles(
        self, article_id: int, video_path: str, sub_path: Path,
    ) -> str:
        """Burn ASS subtitles into video using FFmpeg subtitles filter.

        The ASS file embeds all styling (font, size, colour, fixed
        bottom-centre position with PlayRes matching the video), so no
        force_style override is needed.
        """
        out_path = str(
            Path(video_path).parent
            / f"{Path(video_path).stem}_subtitled.mp4"
        )

        # FFmpeg's subtitles filter cannot parse absolute Windows paths.
        # Work around by cd-ing into the subtitle's directory and using
        # just the filename, with absolute paths for input/output video.
        sub_dir = str(sub_path.parent)
        sub_filename = sub_path.name

        vf_filter = f"subtitles={sub_filename}"

        cmd = [
            self.ffmpeg, "-y",
            "-i", video_path,
            "-vf", vf_filter,
            "-map", "0:v",
            "-map", "0:a?",
            "-c:v", vc.VIDEO_CODEC,
            "-preset", "medium",
            "-crf", str(vc.VIDEO_CRF),
            "-c:a", "copy",
            "-pix_fmt", "yuv420p",
            out_path,
        ]

        logger.debug("Subtitle burn cmd: %s", " ".join(cmd))
        try:
            old_cwd = os.getcwd()
            os.chdir(sub_dir)
            proc = subprocess.run(cmd, capture_output=True, timeout=600)
            os.chdir(old_cwd)
        except FileNotFoundError:
            os.chdir(old_cwd)
            logger.warning(
                "FFmpeg not found (%s); returning original video without subtitles",
                self.ffmpeg,
            )
            return video_path
        except Exception as exc:
            os.chdir(old_cwd)
            logger.warning("FFmpeg subtitle burn raised: %s", exc)
            return video_path

        if proc.returncode != 0 or not Path(out_path).exists():
            stderr = proc.stderr.decode(errors="replace")[:1000]
            logger.warning(
                "FFmpeg subtitle burn failed for article %d: %s", article_id, stderr,
            )
            # Final fallback: copy input without subtitles
            logger.info("Returning original video without subtitle burn for article %d", article_id)
            return video_path

        return out_path


# -- module-level helpers ----------------------------------------------------

# Urdu broadcast section labels (same pattern as TTS engine)
_BROADCAST_LABEL_RE = re.compile(
    r"(?:ہیڈلائن|تعارف|مرکزی خبر|اہم نکات|اختتامیہ)\s*:?\s*",
)


def _strip_broadcast_labels(text: str) -> str:
    """Remove broadcast section markers from subtitle text."""
    text = _BROADCAST_LABEL_RE.sub("", text)
    # Remove bullet markers for key points
    text = re.sub(r"^-\s*", "", text, flags=re.MULTILINE)
    # Collapse extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _fmt_srt_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS,mmm for SRT."""
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _fmt_ass_time(seconds: float) -> str:
    """Format seconds as H:MM:SS.cc for ASS."""
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def _font_family_name(font_path: str) -> str:
    """Try to read the real font family name from a TTF file.

    Falls back to an empty string (caller uses a default) if the font
    cannot be parsed.
    """
    try:
        from PIL import ImageFont
        font = ImageFont.truetype(font_path, 16)
        family, style = font.getname()
        return family or ""
    except Exception:
        return ""


def _hex_to_ass(hex_color: str) -> str:
    """Convert #RRGGBB to ASS &H00BBGGRR."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        h = "ffffff"
    rr, gg, bb = h[0:2], h[2:4], h[4:6]
    return f"&H00{bb}{gg}{rr}".upper()


def _hex_to_ass_opacity(hex_color: str, opacity: float) -> str:
    """Convert #RRGGBB + opacity (0-1) to ASS &HAABBGGRR."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        h = "000000"
    rr, gg, bb = h[0:2], h[2:4], h[4:6]
    # ASS alpha: 00=fully opaque, FF=fully transparent
    alpha = int((1.0 - opacity) * 255)
    aa = f"{alpha:02X}"
    return f"&H{aa}{bb}{gg}{rr}".upper()
