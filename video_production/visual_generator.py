"""
video_production/visual_generator.py
========================================
Stage 3.3 — Background & Visuals.

Generates all visual layers for the news video composition:

* **Studio background** — a 1920×1080 professional news-studio style
  background (gradient + decorative elements via Pillow).  If a custom
  ``STUDIO_BG_IMAGE`` is configured, that image is loaded and resized
  instead.

* **Lower-third** — headline + category label with RTL Urdu text,
  drawn on a semi-transparent bar.

* **Breaking-news ticker** — scrolling-style ticker bar at the bottom
  with the headline text.

* **News images** — if the article has associated images in
  ``NEWS_IMAGES_DIR``, the first one is loaded and resized for the
  B-roll layer.

All outputs are PNG frames that the composer (Stage 3.4) will encode
into video.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import os
import time
import google.genai as genai
from google.genai import types

from data_acquisition.utils.logger import get_logger

from . import config as vc

logger = get_logger("video_production.visuals")

# Category -> accent colour mapping for lower-third / ticker styling
CATEGORY_COLORS: Dict[str, str] = {
    "Politics": "#1a3a6e",
    "Sports": "#0d6e1a",
    "Business": "#6e4a1a",
    "Technology": "#1a5a6e",
    "International": "#4a1a6e",
    "Entertainment": "#6e1a5a",
    "Health": "#1a6e4a",
    "Science": "#3a1a6e",
    "Breaking News": "#cc0000",
    "Local": "#3a4a5a",
    "Others": "#3a3a5a",
}


@dataclass
class VisualAssets:
    article_id: int
    background_path: str = ""
    lower_third_path: str = ""
    ticker_path: str = ""
    news_image_path: str = ""  # Will contain the path to the Veo-generated B-roll video, or static image as fallback
    status: str = "ok"


class VisualGenerator:
    """Generate visual layers for each news article."""

    def __init__(
        self,
        output_dir: Path | str = vc.VISUALS_DIR,
        width: int = vc.VIDEO_WIDTH,
        height: int = vc.VIDEO_HEIGHT,
        custom_bg: str = vc.STUDIO_BG_IMAGE,
        news_images_dir: str | Path = vc.NEWS_IMAGES_DIR,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.custom_bg = custom_bg
        self.news_images_dir = Path(news_images_dir)
        
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None

    # -- public API ----------------------------------------------------------
    def generate(
        self,
        article_id: int,
        headline: str,
        category: str = "",
        news_image: str = "",
    ) -> VisualAssets:
        try:
            bg_path = self._make_background(article_id, category)
            lt_path = self._make_lower_third(article_id, headline, category)
            ticker_path = self._make_ticker(article_id, headline)
            img_path = self._find_news_image(article_id, news_image)
            broll_path = self._make_broll_video(article_id, img_path)
            return VisualAssets(
                article_id, str(bg_path), str(lt_path),
                str(ticker_path), broll_path, "ok",
            )
        except Exception as exc:
            logger.error("Visual generation failed for article %d: %s", article_id, exc)
            return VisualAssets(article_id, status="error")

    def run(self, articles: List[dict]) -> List[VisualAssets]:
        """articles: [{"article_id", "headline", "category", "news_image"}]"""
        results = []
        for a in articles:
            r = self.generate(
                a["article_id"],
                a.get("headline", ""),
                a.get("category", ""),
                a.get("news_image", ""),
            )
            results.append(r)
        ok = sum(1 for r in results if r.status == "ok")
        logger.info("Visual generation: %d/%d ok", ok, len(results))
        return results

    # -- background ----------------------------------------------------------
    def _make_background(self, article_id: int, category: str) -> Path:
        from PIL import Image, ImageDraw, ImageFilter

        out = self.output_dir / f"bg_{article_id}.png"

        # Use custom background if configured
        if self.custom_bg and Path(self.custom_bg).exists():
            img = Image.open(self.custom_bg).convert("RGB")
            img = img.resize((self.width, self.height), Image.LANCZOS)
            img.save(str(out))
            return out

        # ----------------------------------------------------------------
        # Cinematic ARY-style professional studio background
        # ----------------------------------------------------------------
        accent = CATEGORY_COLORS.get(category, "#1a1a4a")
        accent_rgb = self._hex_to_rgb(accent)
        img = Image.new("RGB", (self.width, self.height))
        draw = ImageDraw.Draw(img)

        # 1. Base gradient: deep navy top → near-black with subtle teal tinge at bottom
        for y in range(self.height):
            ratio = y / self.height
            r = int(6  + 16 * ratio)
            g = int(8  + 18 * ratio)
            b = int(40 + 28 * ratio)
            draw.line([(0, y), (self.width, y)], fill=(r, g, b))

        # 2. Radial spotlight centred on presenter area (left half)
        #    We draw concentric ellipses with increasing alpha towards centre.
        spotlight_cx = self.width // 4 + 60
        spotlight_cy = self.height // 2 - 30
        spotlight_rx = 420
        spotlight_ry = 380
        spot_layer = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        spot_draw = ImageDraw.Draw(spot_layer)
        steps = 40
        for i in range(steps, 0, -1):
            alpha = int(32 * (1 - i / steps))   # brightest at centre
            rx = int(spotlight_rx * i / steps)
            ry = int(spotlight_ry * i / steps)
            spot_draw.ellipse(
                [spotlight_cx - rx, spotlight_cy - ry,
                 spotlight_cx + rx, spotlight_cy + ry],
                fill=(255, 255, 240, alpha),
            )
        try:
            spot_layer = spot_layer.filter(ImageFilter.GaussianBlur(radius=18))
        except Exception:
            pass
        img = Image.alpha_composite(img.convert("RGBA"), spot_layer).convert("RGB")
        draw = ImageDraw.Draw(img)

        # 3. Subtle category accent sweep in top-right corner
        sweep_layer = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        sw_draw = ImageDraw.Draw(sweep_layer)
        for i in range(8):
            offset = i * 40
            sw_draw.line(
                [(self.width - 280 + offset, 0), (self.width, 280 - offset)],
                fill=(*accent_rgb, 28), width=2,
            )
        img = Image.alpha_composite(img.convert("RGBA"), sweep_layer).convert("RGB")
        draw = ImageDraw.Draw(img)

        # 4. Left accent stripe (ARY red brand bar)
        draw.rectangle([0, 0, 6, self.height], fill=(210, 20, 20))

        # 5. Right news-image panel — well-lit with gradient fill
        panel_x = self.width // 2 + 60
        panel_w = self.width - panel_x - 45
        panel_y = 75
        panel_h = self.height - 275
        # Inner fill — slightly lighter than BG so it reads as a distinct zone
        for py in range(panel_y, panel_y + panel_h):
            ratio = (py - panel_y) / panel_h
            r = int(18 + 12 * ratio)
            g = int(20 + 14 * ratio)
            b = int(60 + 20 * ratio)
            draw.line([(panel_x, py), (panel_x + panel_w, py)], fill=(r, g, b))
        # Golden border (2px outer + 1px inner highlight)
        draw.rectangle(
            [panel_x - 3, panel_y - 3,
             panel_x + panel_w + 3, panel_y + panel_h + 3],
            outline=(180, 158, 70), width=3,
        )
        draw.rectangle(
            [panel_x - 1, panel_y - 1,
             panel_x + panel_w + 1, panel_y + panel_h + 1],
            outline=(220, 200, 110), width=1,
        )

        # 6. Presenter desk prop — subtle perspective trapezoid at the bottom
        desk_y_top = self.height - 290
        desk_y_bot = self.height - 240
        margin = 40
        desk_inset = 120
        desk_pts = [
            (margin, desk_y_bot),
            (self.width // 2 + 30, desk_y_bot),
            (self.width // 2 + 30 - desk_inset // 2, desk_y_top),
            (margin + desk_inset // 2, desk_y_top),
        ]
        desk_layer = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        desk_draw = ImageDraw.Draw(desk_layer)
        desk_draw.polygon(desk_pts, fill=(28, 32, 72, 210))
        # Desk highlight edge
        desk_draw.line(
            [(desk_pts[3][0], desk_pts[3][1]), (desk_pts[2][0], desk_pts[2][1])],
            fill=(140, 130, 80, 180), width=2,
        )
        img = Image.alpha_composite(img.convert("RGBA"), desk_layer).convert("RGB")
        draw = ImageDraw.Draw(img)

        # 7. Top channel branding bar
        draw.rectangle([0, 0, self.width, 5], fill=(210, 20, 20))

        # 8. Bottom ticker-area darkening band
        for y in range(self.height - 110, self.height):
            progress = (y - (self.height - 110)) / 110
            r = int(6 * (1 - progress))
            g = int(8 * (1 - progress))
            b = int(30 * (1 - progress))
            draw.line([(0, y), (self.width, y)], fill=(r, g, b))

        img.save(str(out))
        return out

    # -- lower third ---------------------------------------------------------
    def _make_lower_third(
        self, article_id: int, headline: str, category: str,
    ) -> Path:
        from PIL import Image, ImageDraw, ImageFont, ImageFilter

        out = self.output_dir / f"lower_third_{article_id}.png"
        img = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        if not headline:
            img.save(str(out))
            return out

        accent = CATEGORY_COLORS.get(category, vc.LOWER_THIRD_BG_COLOR)
        accent_rgb = self._hex_to_rgb(accent)
        text_rgb = self._hex_to_rgb(vc.LOWER_THIRD_TEXT_COLOR)

        # ----------------------------------------------------------------
        # Broadcast-quality lower third (positioned clear of subtitle zone)
        # Layout:
        #   ┌──────────────────────────────────────────────────────────┐
        #   │ [CATEGORY BOX] ║ Headline text in frosted dark glass bar │
        #   └──────────────────────────────────────────────────────────┘
        # Frosted glass effect = draw fill on a separate layer, blur, merge
        # ----------------------------------------------------------------
        bar_y = self.height - 248
        bar_h = 92
        cat_w = 250
        sep_w = 5   # Red vertical separator bar width

        # Layer A: frosted glass backing (draw, blur, composite)
        glass = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glass)
        # Category box — solid accent colour
        gd.rectangle([0, bar_y, cat_w, bar_y + bar_h], fill=(*accent_rgb, 248))
        # Headline bar — semi-transparent dark navy
        gd.rectangle([cat_w, bar_y, self.width, bar_y + bar_h], fill=(14, 16, 50, 215))
        try:
            glass = glass.filter(ImageFilter.GaussianBlur(radius=1))
        except Exception:
            pass
        img = Image.alpha_composite(img, glass)
        draw = ImageDraw.Draw(img)

        # Red top separator line
        draw.rectangle(
            [0, bar_y - 4, self.width, bar_y],
            fill=(210, 20, 20, 255),
        )

        # Red vertical separator between category box and headline text
        draw.rectangle(
            [cat_w, bar_y, cat_w + sep_w, bar_y + bar_h],
            fill=(210, 20, 20, 255),
        )

        # Bottom separator line
        draw.rectangle(
            [0, bar_y + bar_h, self.width, bar_y + bar_h + 4],
            fill=(210, 20, 20, 255),
        )

        # Load fonts
        font_path = vc.URDU_FONT_PATH
        try:
            cat_font = ImageFont.truetype(font_path, 32) if font_path else ImageFont.load_default()
            head_font = ImageFont.truetype(font_path, 36) if font_path else ImageFont.load_default()
        except Exception:
            cat_font = ImageFont.load_default()
            head_font = ImageFont.load_default()

        # Category label — bold white (centred in box)
        cat_label = category or "خبر"
        cat_bbox = draw.textbbox((0, 0), cat_label, font=cat_font)
        cat_tw = cat_bbox[2] - cat_bbox[0]
        cat_th = cat_bbox[3] - cat_bbox[1]
        cat_tx = cat_w // 2 - cat_tw // 2
        cat_ty = bar_y + bar_h // 2 - cat_th // 2
        # Drop shadow
        draw.text((cat_tx + 2, cat_ty + 2), cat_label, fill=(0, 0, 0, 140), font=cat_font)
        draw.text((cat_tx, cat_ty), cat_label, fill=(255, 255, 255, 255), font=cat_font)

        # Headline text — truncated, right-aligned (RTL), with strong shadow
        display_headline = headline[:90] if len(headline) > 90 else headline
        head_x = cat_w + sep_w + 25
        head_y = bar_y + bar_h // 2 - 22
        # Multi-pixel shadow for readability on any background
        for dx, dy in ((2, 2), (-1, 1), (1, -1)):
            draw.text(
                (head_x + dx, head_y + dy),
                display_headline,
                fill=(0, 0, 0, 160),
                font=head_font,
            )
        draw.text(
            (head_x, head_y),
            display_headline,
            fill=(*text_rgb, 255),
            font=head_font,
        )

        img.save(str(out))
        return out

    # -- ticker --------------------------------------------------------------
    def _make_ticker(self, article_id: int, headline: str) -> Path:
        from PIL import Image, ImageDraw, ImageFont

        out = self.output_dir / f"ticker_{article_id}.png"
        ticker_h = 55
        img = Image.new("RGBA", (self.width, ticker_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # ARY-style ticker: bold red background with white text
        draw.rectangle(
            [0, 0, self.width, ticker_h],
            fill=(190, 20, 20, 240),
        )
        # Top highlight line
        draw.rectangle(
            [0, 0, self.width, 2],
            fill=(255, 80, 80, 200),
        )

        # Ticker text
        font_path = vc.URDU_FONT_PATH
        try:
            font = ImageFont.truetype(font_path, 28) if font_path else ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        # Breaking/news label + headline
        text = f"  {vc.CHANNEL_NAME}  ●  {headline[:120]}  "
        text_rgb = (255, 255, 255, 255)

        # Right-aligned for Urdu RTL
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        x = self.width - tw - 20
        # Shadow for readability
        draw.text((x + 1, 14), text, fill=(80, 0, 0, 180), font=font)
        draw.text((x, 13), text, fill=text_rgb, font=font)

        img.save(str(out))
        return out

    # -- news image lookup ---------------------------------------------------
    def _find_news_image(self, article_id: int, news_image: str) -> str:
        if news_image and Path(news_image).exists():
            return news_image
        # Check news images directory for article-specific images
        if self.news_images_dir.exists():
            for ext in ("jpg", "jpeg", "png"):
                candidate = self.news_images_dir / f"article_{article_id}.{ext}"
                if candidate.exists():
                    return str(candidate)
        return ""

    def _make_broll_video(self, article_id: int, image_path: str) -> str:
        if not image_path or not Path(image_path).exists():
            return ""
            
        if image_path.lower().endswith((".mp4", ".avi", ".mov")):
            return image_path
            
        if not self.client:
            logger.warning(f"No GEMINI_API_KEY, falling back to static B-roll for {article_id}")
            return image_path
            
        output_path = self.output_dir / f"broll_{article_id}.mp4"
        if output_path.exists():
            return str(output_path)
            
        logger.info(f"Generating dynamic B-roll via Veo for {article_id}...")
        try:
            import mimetypes
            with open(image_path, "rb") as f:
                img_bytes = f.read()
            mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
            
            operation = self.client.models.generate_videos(
                model="veo-3.1-fast-generate-preview",
                source=types.GenerateVideosSource(
                    image=types.Image(
                        image_bytes=img_bytes,
                        mime_type=mime_type
                    ),
                    prompt="Cinematic slow push-in, natural motion, realistic broadcast news footage.",
                ),
            )
            while not operation.done:
                time.sleep(10)
                logger.info("Waiting for Veo B-roll generation...")
                operation = self.client.operations.get(operation)
                
            if not operation.result or not getattr(operation.result, "generated_videos", None):
                return image_path
                
            video_uri = operation.result.generated_videos[0].video.uri
            try:
                file_name = "files/" + video_uri.split('/')[-1]
                video_bytes = self.client.files.download(name=file_name)
                with open(output_path, 'wb') as f:
                    f.write(video_bytes)
            except Exception:
                import requests
                response = requests.get(video_uri, stream=True)
                if response.status_code == 200:
                    with open(output_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                else:
                    return image_path
            return str(output_path)
        except Exception as e:
            logger.error(f"Veo B-roll generation failed: {e}")
            return image_path

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _hex_to_rgb(hex_color: str) -> tuple:
        h = hex_color.lstrip("#")
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
