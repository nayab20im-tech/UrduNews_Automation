"""
video_production/thumbnail_generator.py
==========================================
Stage 3.6 — Automatic Thumbnail Generation.

Generates a professional news thumbnail for each article using:

* **Category-based design** — colour scheme, accent style and label
  change automatically per category (Politics, Sports, Technology …).
* **Dynamic Urdu headline** — the actual generated headline from
  Stage 2.10 is rendered in large readable Urdu text.
* **Presenter or news image** — a frame extracted from the avatar /
  lip-synced video, or the article's news image, composited onto
  the thumbnail.
* **Channel branding** — channel name logo in the corner.

Outputs
-------
``thumbnail_<id>.jpg``  (1280×720, JPEG quality 92)
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from data_acquisition.utils.logger import get_logger

from . import config as vc

logger = get_logger("video_production.thumbnail")

# ---- Category design system --------------------------------------------------
# Each category gets its own background gradient, accent colour, label style.
CATEGORY_THEMES: Dict[str, Dict] = {
    "Politics": {
        "bg_top": "#0a1628", "bg_bottom": "#1a3a6e",
        "accent": "#3a7aee", "label": "سیاست",
    },
    "Sports": {
        "bg_top": "#0a2810", "bg_bottom": "#0d6e1a",
        "accent": "#22cc44", "label": "کھیل",
    },
    "Business": {
        "bg_top": "#1a1408", "bg_bottom": "#6e4a1a",
        "accent": "#ee9933", "label": "کاروبار",
    },
    "Technology": {
        "bg_top": "#081a28", "bg_bottom": "#1a5a6e",
        "accent": "#33ccee", "label": "ٹیکنالوجی",
    },
    "International": {
        "bg_top": "#140a28", "bg_bottom": "#4a1a6e",
        "accent": "#9933ee", "label": "بین الاقوامی",
    },
    "Entertainment": {
        "bg_top": "#280a1a", "bg_bottom": "#6e1a5a",
        "accent": "#ee33aa", "label": "تفریح",
    },
    "Health": {
        "bg_top": "#0a2818", "bg_bottom": "#1a6e4a",
        "accent": "#33ee88", "label": "صحت",
    },
    "Science": {
        "bg_top": "#140a28", "bg_bottom": "#3a1a6e",
        "accent": "#7744ee", "label": "سائنس",
    },
    "Breaking News": {
        "bg_top": "#280808", "bg_bottom": "#cc0000",
        "accent": "#ff3333", "label": "بریکنگ نیوز",
    },
    "Local": {
        "bg_top": "#141820", "bg_bottom": "#3a4a5a",
        "accent": "#7799aa", "label": "مقامی",
    },
}

_DEFAULT_THEME = {
    "bg_top": "#141428", "bg_bottom": "#3a3a5a",
    "accent": "#6677aa", "label": "خبر",
}


@dataclass
class ThumbnailResult:
    article_id: int
    thumbnail_path: str = ""
    status: str = "ok"          # ok | no_input | error


class ThumbnailGenerator:
    """Generate news thumbnail for each article."""

    def __init__(
        self,
        output_dir: Path | str = vc.THUMBNAILS_DIR,
        width: int = vc.THUMBNAIL_WIDTH,
        height: int = vc.THUMBNAIL_HEIGHT,
        font_path: str = vc.THUMBNAIL_FONT_PATH or vc.URDU_FONT_PATH,
        ffmpeg: str = vc.FFMPEG_BINARY,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.font_path = font_path
        self.ffmpeg = ffmpeg

    # -- public API ----------------------------------------------------------
    def generate(
        self,
        article_id: int,
        headline: str,
        category: str = "",
        presenter_video: str = "",
        news_image: str = "",
    ) -> ThumbnailResult:
        try:
            if not headline or not headline.strip():
                return ThumbnailResult(article_id, status="no_input")

            out_path = self.output_dir / f"thumbnail_{article_id}.jpg"
            theme = CATEGORY_THEMES.get(category, _DEFAULT_THEME)

            # Get presenter frame or news image
            subject_image = self._extract_subject(
                article_id, presenter_video, news_image,
            )

            self._render_thumbnail(out_path, headline, theme, subject_image)

            return ThumbnailResult(article_id, str(out_path), "ok")
        except Exception as exc:
            logger.error("Thumbnail generation failed for article %d: %s", article_id, exc)
            return ThumbnailResult(article_id, status="error")

    def run(self, articles: List[dict]) -> List[ThumbnailResult]:
        """articles: [{"article_id", "headline", "category",
        "presenter_video", "news_image"}]"""
        results = []
        for a in articles:
            r = self.generate(
                a["article_id"],
                a.get("headline", ""),
                a.get("category", ""),
                a.get("presenter_video", ""),
                a.get("news_image", ""),
            )
            results.append(r)
        ok = sum(1 for r in results if r.status == "ok")
        logger.info("Thumbnail generation: %d/%d ok", ok, len(results))
        return results

    # -- subject extraction --------------------------------------------------
    def _extract_subject(
        self, article_id: int, presenter_video: str, news_image: str,
    ) -> Optional[str]:
        """Get a subject image: news image first, else a frame from video."""
        # Prefer explicit news image
        if news_image and Path(news_image).exists():
            return news_image

        # Extract a frame from the presenter/lip-synced video
        if presenter_video and Path(presenter_video).exists():
            frame_path = self.output_dir / f"_frame_{article_id}.jpg"
            try:
                cmd = [
                    self.ffmpeg, "-y",
                    "-i", presenter_video,
                    "-ss", "1",           # grab frame at 1s
                    "-vframes", "1",
                    "-q:v", "2",
                    str(frame_path),
                ]
                proc = subprocess.run(cmd, capture_output=True, timeout=30)
                if proc.returncode == 0 and frame_path.exists():
                    return str(frame_path)
            except Exception:
                pass
        return None

    # -- thumbnail rendering -------------------------------------------------
    def _render_thumbnail(
        self,
        out_path: Path,
        headline: str,
        theme: Dict,
        subject_image: Optional[str],
    ) -> None:
        from PIL import Image, ImageDraw, ImageFont

        w, h = self.width, self.height
        img = Image.new("RGB", (w, h))
        draw = ImageDraw.Draw(img)

        # Gradient background
        top = self._hex_to_rgb(theme["bg_top"])
        bottom = self._hex_to_rgb(theme["bg_bottom"])
        for y in range(h):
            ratio = y / h
            r = int(top[0] + (bottom[0] - top[0]) * ratio)
            g = int(top[1] + (bottom[1] - top[1]) * ratio)
            b = int(top[2] + (bottom[2] - top[2]) * ratio)
            draw.line([(0, y), (w, y)], fill=(r, g, b))

        # Subject image (right side for Urdu RTL layout)
        if subject_image and Path(subject_image).exists():
            try:
                subj = Image.open(subject_image).convert("RGB")
                # Scale to fill right ~55% of thumbnail
                subj_w = int(w * 0.55)
                subj_h = h
                subj = subj.resize((subj_w, subj_h), Image.LANCZOS)
                # Paste on right side
                img.paste(subj, (w - subj_w, 0))

                # Gradient overlay on the left side of the subject to blend
                overlay = Image.new("RGBA", (subj_w, subj_h), (0, 0, 0, 0))
                ov_draw = ImageDraw.Draw(overlay)
                for x in range(subj_w):
                    alpha = int(200 * (1 - x / subj_w))
                    ov_draw.line(
                        [(x, 0), (x, subj_h)],
                        fill=(top[0], top[1], top[2], alpha),
                    )
                img.paste(
                    Image.alpha_composite(
                        Image.new("RGBA", (subj_w, subj_h), (0, 0, 0, 0)),
                        overlay,
                    ).convert("RGB"),
                    (w - subj_w, 0),
                    overlay.split()[3],
                )
            except Exception:
                pass

        # Accent bar at top
        accent = self._hex_to_rgb(theme["accent"])
        draw.rectangle([0, 0, w, 6], fill=accent)

        # Category label badge (top-left)
        font_path = self.font_path
        try:
            badge_font = ImageFont.truetype(font_path, 28) if font_path else ImageFont.load_default()
        except Exception:
            badge_font = ImageFont.load_default()

        cat_label = theme.get("label", "خبر")
        cat_bbox = draw.textbbox((0, 0), cat_label, font=badge_font)
        cat_tw = cat_bbox[2] - cat_bbox[0]
        cat_th = cat_bbox[3] - cat_bbox[1]
        badge_pad = 12
        badge_w = cat_tw + badge_pad * 2
        badge_h = cat_th + badge_pad * 2
        badge_y = 20

        draw.rounded_rectangle(
            [20, badge_y, 20 + badge_w, badge_y + badge_h],
            radius=8, fill=accent,
        )
        draw.text(
            (20 + badge_pad, badge_y + badge_pad - 2),
            cat_label, fill="#ffffff", font=badge_font,
        )

        # Headline — large Urdu text, bottom-left area
        try:
            head_font = ImageFont.truetype(font_path, 52) if font_path else ImageFont.load_default()
        except Exception:
            head_font = ImageFont.load_default()

        # Wrap headline to fit
        display = self._wrap_headline(headline, max_chars=40)

        # Text shadow for readability
        text_x = 30
        text_y = h - 200

        # Draw shadow
        for dx, dy in [(2, 2), (-1, -1), (1, -1), (-1, 1)]:
            draw.text(
                (text_x + dx, text_y + dy),
                display, fill="#000000", font=head_font,
            )
        # Draw main text
        draw.text((text_x, text_y), display, fill="#ffffff", font=head_font)

        # Channel name (bottom-right corner)
        try:
            ch_font = ImageFont.truetype(font_path, 22) if font_path else ImageFont.load_default()
        except Exception:
            ch_font = ImageFont.load_default()

        ch_text = vc.CHANNEL_NAME
        ch_bbox = draw.textbbox((0, 0), ch_text, font=ch_font)
        ch_tw = ch_bbox[2] - ch_bbox[0]
        draw.text(
            (w - ch_tw - 20, h - 40),
            ch_text, fill="#cccccc", font=ch_font,
        )

        img.save(str(out_path), "JPEG", quality=92)

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _wrap_headline(text: str, max_chars: int = 40) -> str:
        if len(text) <= max_chars:
            return text
        # Split into max two lines
        mid = len(text) // 2
        best = mid
        for offset in range(min(15, mid)):
            for pos in (mid + offset, mid - offset):
                if 0 < pos < len(text) and text[pos] == " ":
                    best = pos
                    break
            else:
                continue
            break
        line1 = text[:best].rstrip()
        line2 = text[best:].lstrip()
        if len(line2) > max_chars:
            line2 = line2[:max_chars - 3] + "..."
        return line1 + "\n" + line2

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> tuple:
        h = hex_color.lstrip("#")
        if len(h) != 6:
            return (0, 0, 0)
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
