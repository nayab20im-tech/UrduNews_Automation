"""
video_production/video_composer.py
======================================
Stage 3.4 — Video Composition.

Combines all visual layers into the final composed news video using
FFmpeg.  The composition order (bottom to top):

1. **Studio background** (full-frame, static image looped)
2. **News image / B-roll** (scaled, positioned right-of-centre)
3. **Presenter / lip-synced video** (scaled, positioned left)
4. **Lower-third overlay** (headline + category)
5. **Ticker bar** (bottom of screen)
6. **Audio** (Urdu voice-over from Stage 2.12)

Output: 1920×1080 @ 30fps MP4 (H.264 + AAC).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from data_acquisition.utils.logger import get_logger

from . import config as vc

logger = get_logger("video_production.composer")


@dataclass
class CompositionResult:
    article_id: int
    video_path: str = ""
    duration_sec: float = 0.0
    status: str = "ok"          # ok | no_input | error


class VideoComposer:
    """Compose final news video from all visual layers."""

    def __init__(
        self,
        output_dir: Path | str = vc.COMPOSITION_DIR,
        width: int = vc.VIDEO_WIDTH,
        height: int = vc.VIDEO_HEIGHT,
        fps: int = vc.VIDEO_FPS,
        ffmpeg: str = vc.FFMPEG_BINARY,
        presenter_w: int = vc.PRESENTER_WIDTH,
        presenter_h: int = vc.PRESENTER_HEIGHT,
        presenter_x: int = vc.PRESENTER_X,
        presenter_y: int = vc.PRESENTER_Y,
        logo_path: str = vc.CHANNEL_LOGO_PATH,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.fps = fps
        self.ffmpeg = ffmpeg
        self.presenter_w = presenter_w
        self.presenter_h = presenter_h
        self.presenter_x = presenter_x
        self.presenter_y = presenter_y
        self.logo_path = logo_path

    # -- public API ----------------------------------------------------------
    def compose(
        self,
        article_id: int,
        lip_sync_video: str,
        audio_path: str,
        audio_duration: float,
        background_image: str = "",
        lower_third_image: str = "",
        ticker_image: str = "",
        news_image: str = "",
        timeline_segments: Optional[List[Dict[str, Any]]] = None,
        urdu_script: str = "",
    ) -> CompositionResult:
        try:
            if not lip_sync_video or not Path(lip_sync_video).exists():
                return CompositionResult(article_id, status="no_input")
            if not audio_path or not Path(audio_path).exists():
                return CompositionResult(article_id, status="no_input")

            out_path = self.output_dir / f"composed_{article_id}.mp4"
            dur = max(audio_duration, 1.0)

            self._compose_ffmpeg(
                out_path, lip_sync_video, audio_path, dur,
                background_image, lower_third_image, ticker_image, news_image,
                timeline_segments, urdu_script
            )

            return CompositionResult(
                article_id, str(out_path), round(dur, 2), "ok",
            )
        except Exception as exc:
            logger.error("Video composition failed for article %d: %s", article_id, exc)
            return CompositionResult(article_id, status="error")

    def run(self, articles: List[dict]) -> List[CompositionResult]:
        """articles: [{"article_id", "lip_sync_video", "audio_path",
        "audio_duration", "background_image", "lower_third_image",
        "ticker_image", "news_image", "timeline_segments", "urdu_script"}]"""
        results = []
        for a in articles:
            r = self.compose(
                a["article_id"],
                a.get("lip_sync_video", ""),
                a.get("audio_path", ""),
                a.get("audio_duration", 0.0),
                a.get("background_image", ""),
                a.get("lower_third_image", ""),
                a.get("ticker_image", ""),
                a.get("news_image", ""),
                a.get("timeline_segments"),
                a.get("urdu_script", "")
            )
            results.append(r)
        ok = sum(1 for r in results if r.status == "ok")
        logger.info("Video composition: %d/%d ok", ok, len(results))
        return results

    # -- FFmpeg composition --------------------------------------------------
    def _compose_ffmpeg(
        self,
        out_path: Path,
        presenter_video: str,
        audio_path: str,
        duration: float,
        bg_image: str,
        lt_image: str,
        ticker_image: str,
        news_image: str,
        timeline_segments: Optional[List[Dict[str, Any]]] = None,
        urdu_script: str = "",
    ) -> None:
        """Build and execute a multi-layer FFmpeg filter_complex command.

        The lip-synced presenter video already contains the full studio
        background (rendered by the animation engine), so we use it as
        the base layer and overlay only the lower-third and ticker.

        Composition order (bottom to top):
        1. Presenter / lip-synced video (full 1920x1080 frame)
        2. News image panel overlay (right half)
        3. Lower-third overlay (headline + category)
        4. Ticker bar (bottom, scrolled)
        5. Channel logo watermark (top-left corner, optional)
        6. Audio (Urdu voice-over from Stage 2.12)
        """
        inputs: List[str] = []
        filter_parts: List[str] = []
        overlay_chain = "[base]"
        input_idx = 0

        # --- Input 0: presenter video (full-frame with studio background) ---
        inputs.extend(["-i", presenter_video])
        filter_parts.append(
            f"[{input_idx}:v]scale={self.width}:{self.height},"
            f"setsar=1,format=yuva420p[base]"
        )
        input_idx += 1

        # --- Input 1: B-roll Video / Image (Full screen crossfade) ---
        if timeline_segments and len(timeline_segments) > 0 and False: 
            # We disable the old PIP slideshow logic to force full-screen B-roll 
            pass
            
        elif news_image and Path(news_image).exists():
            is_video = news_image.lower().endswith(('.mp4', '.avi', '.mov'))
            if is_video:
                inputs.extend(["-stream_loop", "-1", "-i", news_image])
            else:
                inputs.extend(["-loop", "1", "-i", news_image])
                
            # Scale to full 1920x1080, crop to fit
            # Fade in from alpha 0 to 1 at t=3
            fade_start = min(3.0, duration * 0.4) # Fade at 3s or 40% of duration, whichever is earlier
            filter_parts.append(
                f"[{input_idx}:v]"
                f"scale={self.width}:{self.height}:force_original_aspect_ratio=increase,"
                f"crop={self.width}:{self.height},"
                f"format=yuva420p,"
                f"fade=t=in:st={fade_start}:d=1:alpha=1[broll]"
            )
            input_idx += 1
            filter_parts.append(
                f"{overlay_chain}[broll]overlay=0:0:shortest=1[broll_out]"
            )
            overlay_chain = "[broll_out]"

        # --- Lower third overlay ---
        if lt_image and Path(lt_image).exists():
            inputs.extend(["-loop", "1", "-i", lt_image])
            filter_parts.append(
                f"[{input_idx}:v]scale={self.width}:{self.height},"
                f"format=yuva420p[lt]"
            )
            input_idx += 1
            # Delay the lower third slightly so it fades in over the presenter
            filter_parts.append(f"{overlay_chain}[lt]overlay=0:0:shortest=1[lt_out]")
            overlay_chain = "[lt_out]"

        # --- Ticker overlay (animated scroll for Urdu RTL text) ---
        if ticker_image and Path(ticker_image).exists():
            inputs.extend(["-loop", "1", "-i", ticker_image])
            ticker_y = self.height - 55
            ticker_w_img = self.width * 3  # ticker image is 3× wide for scroll room
            filter_parts.append(
                f"[{input_idx}:v]"
                f"scale={ticker_w_img}:-1,format=yuva420p,"
                # Scroll from right to left over the duration (RTL for Urdu)
                f"crop={self.width}:55:"
                f"'mod(t*120,iw-{self.width})'"  # px/s scroll speed
                f":0[tk]"
            )
            input_idx += 1
            filter_parts.append(
                f"{overlay_chain}[tk]overlay=0:{ticker_y}:shortest=1[tk_out]"
            )
            overlay_chain = "[tk_out]"

        # --- Channel logo watermark (top-left, semi-transparent) ---
        if self.logo_path and Path(self.logo_path).exists():
            inputs.extend(["-loop", "1", "-i", self.logo_path])
            filter_parts.append(
                f"[{input_idx}:v]"
                f"scale=160:-1,format=yuva420p,"
                f"colorchannelmixer=aa=0.75[logo]"
            )
            input_idx += 1
            filter_parts.append(
                f"{overlay_chain}[logo]overlay=20:12:shortest=1[logo_out]"
            )
            overlay_chain = "[logo_out]"

        # Audio input
        inputs.extend(["-i", audio_path])
        audio_idx = input_idx

        # Final output label
        final_label = overlay_chain.strip("[]")
        filter_complex = ";".join(filter_parts)

        cmd = [
            self.ffmpeg, "-y",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", f"[{final_label}]",
            "-map", f"{audio_idx}:a",
            "-c:v", vc.VIDEO_CODEC,
            "-preset", "slow",
            "-crf", str(vc.VIDEO_CRF),
            "-c:a", vc.AUDIO_CODEC,
            "-b:a", vc.AUDIO_BITRATE,
            "-pix_fmt", "yuv420p",
            "-t", str(duration),
            "-shortest",
            "-r", str(self.fps),
            str(out_path),
        ]

        logger.debug("FFmpeg cmd: %s", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, timeout=600)
        if proc.returncode != 0:
            stderr = proc.stderr.decode(errors="replace")
            logger.warning("FFmpeg composition failed: %s", stderr[:1000])
            # Fallback: simple merge without overlays
            self._compose_simple(out_path, presenter_video, audio_path, duration)
        elif not out_path.exists():
            self._compose_simple(out_path, presenter_video, audio_path, duration)

    def _compose_simple(
        self, out_path: Path, video: str, audio: str, duration: float,
    ) -> None:
        """Fallback: simple video + audio merge without overlays."""
        cmd = [
            self.ffmpeg, "-y",
            "-i", video,
            "-i", audio,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", vc.VIDEO_CODEC,
            "-c:a", vc.AUDIO_CODEC,
            "-b:a", vc.AUDIO_BITRATE,
            "-pix_fmt", "yuv420p",
            "-t", str(duration),
            "-shortest",
            str(out_path),
        ]
        subprocess.run(cmd, capture_output=True, timeout=300, check=True)
