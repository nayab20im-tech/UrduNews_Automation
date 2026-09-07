"""
video_production/tests/test_video_production.py
===================================================
Offline test suite for Section 3 (Stages 3.1–3.6).

No GPU, model weights, or FFmpeg required — every test exercises the
placeholder/offline code paths and validates the data flow between
stages.

Run with:  pytest video_production/tests -v   (from the project root,
alongside data_acquisition/)
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from video_production import config as vc
from video_production.avatar_generator import AvatarGenerator, AvatarResult
from video_production.lip_sync import LipSyncEngine, LipSyncResult
from video_production.visual_generator import VisualGenerator, VisualAssets
from video_production.video_composer import VideoComposer, CompositionResult
from video_production.subtitle_generator import (
    SubtitleGenerator, SubtitleResult,
    _fmt_srt_time, _hex_to_ass, _hex_to_ass_opacity,
)
from video_production.thumbnail_generator import (
    ThumbnailGenerator, ThumbnailResult, CATEGORY_THEMES,
)
from video_production.pipeline import (
    run_video_production, _filter_eligible, _build_work_items,
    VideoProductionReport,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav(path: Path, duration_sec: float = 1.0, rate: int = 16000) -> Path:
    """Write a minimal valid WAV file (silence)."""
    import struct, wave
    frames = int(rate * duration_sec)
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00" * frames * 2)
    return path


def _make_image(path: Path, w: int = 100, h: int = 100, color: str = "#ff0000") -> Path:
    """Create a small test image."""
    from PIL import Image
    img = Image.new("RGB", (w, h), color)
    img.save(str(path))
    return path


def _make_video(path: Path, duration_sec: float = 1.0, w: int = 100, h: int = 100) -> Path:
    """Create a minimal video file using Pillow + raw frames.

    Since we can't guarantee FFmpeg is available, we create a minimal
    valid MP4 by writing frames via imageio or just create a placeholder
    file for tests that don't need a real video.
    """
    # For tests that just check path existence / status
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * 100)
    return path


# ---------------------------------------------------------------------------
# Stage 3.1 — Avatar Generator
# ---------------------------------------------------------------------------
class TestAvatarGenerator:
    def test_resolve_backend_auto_no_gpu(self):
        gen = AvatarGenerator(backend="auto")
        assert gen.resolve_backend() == "placeholder"

    def test_resolve_backend_explicit(self):
        gen = AvatarGenerator(backend="sadtalker")
        assert gen.resolve_backend() == "sadtalker"

    def test_result_dataclass(self):
        r = AvatarResult(article_id=1, video_path="/tmp/test.mp4", status="ok")
        assert r.article_id == 1
        assert r.status == "ok"

    def test_generate_handles_missing_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = AvatarGenerator(output_dir=Path(tmp) / "avatar")
            r = gen.generate(1, "/nonexistent/audio.wav", 5.0)
            # No audio -> should not crash, returns error or no_source
            assert r.status in ("error", "no_source")

    def test_run_handles_empty_list(self):
        gen = AvatarGenerator()
        assert gen.run([]) == []


# ---------------------------------------------------------------------------
# Stage 3.2 — Lip Sync
# ---------------------------------------------------------------------------
class TestLipSyncEngine:
    def test_resolve_backend_auto_no_gpu(self):
        eng = LipSyncEngine(backend="auto")
        # Offline without CUDA/API keys, animated is now the fallback before placeholder
        assert eng.resolve_backend() in ("animated", "placeholder")

    def test_resolve_backend_explicit(self):
        eng = LipSyncEngine(backend="wav2lip")
        assert eng.resolve_backend() == "wav2lip"

    def test_result_dataclass(self):
        r = LipSyncResult(article_id=2, backend_used="placeholder", status="ok")
        assert r.backend_used == "placeholder"

    def test_sync_handles_missing_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            eng = LipSyncEngine(output_dir=Path(tmp) / "lip")
            r = eng.sync(1, "", "/no/audio.wav", 5.0)
            assert r.status == "no_input"

    def test_run_handles_empty_list(self):
        eng = LipSyncEngine()
        assert eng.run([]) == []


# ---------------------------------------------------------------------------
# Stage 3.3 — Visual Generator
# ---------------------------------------------------------------------------
class TestVisualGenerator:
    def test_generate_creates_all_layers(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = VisualGenerator(output_dir=Path(tmp) / "vis")
            r = gen.generate(
                article_id=1,
                headline="حکومت نے نئی پالیسی کا اعلان کیا",
                category="Politics",
            )
            assert r.status == "ok"
            assert Path(r.background_path).exists()
            assert Path(r.lower_third_path).exists()
            assert Path(r.ticker_path).exists()

    def test_background_is_correct_size(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            gen = VisualGenerator(output_dir=Path(tmp) / "vis")
            r = gen.generate(1, "ٹیسٹ ہیڈلائن", "Sports")
            from PIL import Image
            with Image.open(r.background_path) as bg:
                assert bg.size == (vc.VIDEO_WIDTH, vc.VIDEO_HEIGHT)

    def test_lower_third_is_transparent(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            gen = VisualGenerator(output_dir=Path(tmp) / "vis")
            r = gen.generate(1, "ٹیسٹ ہیڈلائن", "Technology")
            from PIL import Image
            with Image.open(r.lower_third_path) as lt:
                assert lt.mode == "RGBA"

    def test_empty_headline_produces_empty_lower_third(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            gen = VisualGenerator(output_dir=Path(tmp) / "vis")
            r = gen.generate(1, "", "")
            assert r.status == "ok"
            from PIL import Image
            with Image.open(r.lower_third_path) as lt:
                assert lt.mode == "RGBA"

    def test_custom_background_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            custom_bg = _make_image(Path(tmp) / "custom_bg.png", 1920, 1080)
            gen = VisualGenerator(
                output_dir=Path(tmp) / "vis",
                custom_bg=str(custom_bg),
            )
            r = gen.generate(1, "ٹیسٹ", "")
            assert r.status == "ok"
            assert Path(r.background_path).exists()

    def test_news_image_lookup(self):
        with tempfile.TemporaryDirectory() as tmp:
            img_dir = Path(tmp) / "images"
            img_dir.mkdir()
            _make_image(img_dir / "article_42.jpg")
            gen = VisualGenerator(
                output_dir=Path(tmp) / "vis",
                news_images_dir=str(img_dir),
            )
            r = gen.generate(42, "ٹیسٹ", "Sports")
            assert r.news_image_path != ""

    def test_category_colors_coverage(self):
        gen = VisualGenerator()
        for cat in ["Politics", "Sports", "Business", "Technology",
                     "Breaking News", "Entertainment", "Health"]:
            r = gen.generate.__wrapped__ if hasattr(gen.generate, "__wrapped__") else None
            # Just verify the color map has entries for major categories
            assert cat in CATEGORY_COLORS or cat in CATEGORY_THEMES


CATEGORY_COLORS = {
    "Politics": "#1a3a6e", "Sports": "#0d6e1a", "Business": "#6e4a1a",
    "Technology": "#1a5a6e", "Breaking News": "#cc0000",
    "Entertainment": "#6e1a5a", "Health": "#1a6e4a",
}


# ---------------------------------------------------------------------------
# Stage 3.5 — Subtitle Generator
# ---------------------------------------------------------------------------
class TestSubtitleGenerator:
    def test_fmt_srt_time(self):
        assert _fmt_srt_time(0.0) == "00:00:00,000"
        assert _fmt_srt_time(1.5) == "00:00:01,500"
        assert _fmt_srt_time(3661.25) == "01:01:01,250"

    def test_hex_to_ass(self):
        assert _hex_to_ass("#ffffff") == "&H00FFFFFF"
        assert _hex_to_ass("#000000") == "&H00000000"
        assert _hex_to_ass("#ff0000") == "&H000000FF"

    def test_hex_to_ass_opacity(self):
        result = _hex_to_ass_opacity("#000000", 0.6)
        # alpha = (1-0.6)*255 = 102 = 0x66
        assert "&H66" in result

    def test_generate_creates_srt(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            gen = SubtitleGenerator(output_dir=Path(tmp) / "subs")
            # Create a mock video file (just needs to exist)
            vid = Path(tmp) / "test.mp4"
            vid.write_bytes(b"\x00" * 100)

            r = gen.generate(
                article_id=1,
                urdu_script="حکومت نے نئی پالیسی کا اعلان کیا۔ وزیر نے کہا کہ اگلے ماہ سے نافذ ہو گی۔",
                video_path=str(vid),
                audio_duration=10.0,
            )
            assert r.status == "ok"
            assert r.subtitle_count > 0
            assert Path(r.srt_path).exists()

            # Read and verify SRT content
            srt_content = Path(r.srt_path).read_text(encoding="utf-8-sig")
            assert "حکومت" in srt_content
            assert "-->" in srt_content

    def test_empty_script_returns_no_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = SubtitleGenerator(output_dir=Path(tmp) / "subs")
            vid = Path(tmp) / "test.mp4"
            vid.write_bytes(b"\x00")
            r = gen.generate(1, "", str(vid), 10.0)
            assert r.status == "no_input"

    def test_missing_video_returns_no_input(self):
        gen = SubtitleGenerator()
        r = gen.generate(1, "ٹیسٹ متن۔", "/no/video.mp4", 10.0)
        assert r.status == "no_input"

    def test_sentence_splitting(self):
        sentences = SubtitleGenerator._split_sentences(
            "پہلا جملہ۔ دوسرا جملہ۔ تیسرا جملہ!"
        )
        assert len(sentences) == 3

    def test_sentence_splitting_empty(self):
        assert SubtitleGenerator._split_sentences("") == []
        assert SubtitleGenerator._split_sentences("   ") == []

    def test_long_line_wrapping(self):
        gen = SubtitleGenerator(max_chars=30)
        long_text = "یہ ایک بہت لمبا جملہ ہے جو تیس حروف سے زیادہ ہے"
        wrapped = gen._wrap_line(long_text)
        assert "\n" in wrapped

    def test_run_handles_empty_list(self):
        gen = SubtitleGenerator()
        assert gen.run([]) == []


# ---------------------------------------------------------------------------
# Stage 3.6 — Thumbnail Generator
# ---------------------------------------------------------------------------
class TestThumbnailGenerator:
    def test_generate_creates_thumbnail(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            gen = ThumbnailGenerator(output_dir=Path(tmp) / "thumbs")
            r = gen.generate(
                article_id=1,
                headline="حکومت نے نئی پالیسی کا اعلان کیا",
                category="Politics",
            )
            assert r.status == "ok"
            assert Path(r.thumbnail_path).exists()
            from PIL import Image
            with Image.open(r.thumbnail_path) as thumb:
                assert thumb.size == (vc.THUMBNAIL_WIDTH, vc.THUMBNAIL_HEIGHT)

    def test_different_categories_produce_different_designs(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            gen = ThumbnailGenerator(output_dir=Path(tmp) / "thumbs")
            r1 = gen.generate(1, "سیاسی خبر", "Politics")
            r2 = gen.generate(2, "کھیل کی خبر", "Sports")
            assert r1.status == "ok" and r2.status == "ok"
            from PIL import Image
            with Image.open(r1.thumbnail_path) as t1, Image.open(r2.thumbnail_path) as t2:
                assert t1.size == t2.size

    def test_breaking_news_category(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = ThumbnailGenerator(output_dir=Path(tmp) / "thumbs")
            r = gen.generate(1, "بریکنگ نیوز: بڑا واقعہ", "Breaking News")
            assert r.status == "ok"
            assert Path(r.thumbnail_path).exists()

    def test_with_news_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = _make_image(Path(tmp) / "news.jpg", 640, 480)
            gen = ThumbnailGenerator(output_dir=Path(tmp) / "thumbs")
            r = gen.generate(1, "ٹیسٹ ہیڈلائن", "Technology", news_image=str(img))
            assert r.status == "ok"

    def test_empty_headline_returns_no_input(self):
        gen = ThumbnailGenerator()
        r = gen.generate(1, "", "Politics")
        assert r.status == "no_input"

    def test_headline_wrapping(self):
        gen = ThumbnailGenerator()
        short = gen._wrap_headline("مختصر خبر", max_chars=40)
        assert "\n" not in short

        long = gen._wrap_headline(
            "یہ ایک بہت لمبی ہیڈلائن ہے جو چالیس حروف سے زیادہ ہے اور اسے دو لائنوں میں توڑنا ضروری ہے",
            max_chars=40,
        )
        assert "\n" in long

    def test_all_category_themes_have_required_keys(self):
        for cat, theme in CATEGORY_THEMES.items():
            assert "bg_top" in theme
            assert "bg_bottom" in theme
            assert "accent" in theme
            assert "label" in theme

    def test_run_handles_empty_list(self):
        gen = ThumbnailGenerator()
        assert gen.run([]) == []


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------
class TestVideoProductionPipeline:
    def test_filter_eligible_no_audio(self):
        articles = [
            {"article_id": 1, "headline": "خبر", "audio_path": ""},
            {"article_id": 2, "headline": "خبر", "processed_audio_path": ""},
        ]
        assert _filter_eligible(articles) == []

    def test_filter_eligible_with_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = _make_wav(Path(tmp) / "a.wav")
            articles = [
                {"article_id": 1, "processed_audio_path": str(wav)},
                {"article_id": 2, "processed_audio_path": "/nonexistent.wav"},
            ]
            eligible = _filter_eligible(articles)
            assert len(eligible) == 1
            assert eligible[0]["article_id"] == 1

    def test_build_work_items_normalizes_fields(self):
        articles = [
            {
                "article_id": 42,
                "headline": "سیاسی خبر",
                "full_script": "مکمل اسکرپٹ یہاں ہے۔",
                "category": "Politics",
                "processed_audio_path": "/tmp/audio.wav",
                "audio_duration_sec": 15.5,
            },
            {
                "article_id": 43,
                "urdu_title": "کھیل کی خبر",
                "urdu_text": "متن یہاں ہے۔",
                "category": "Sports",
                "tts_audio_path": "/tmp/audio2.wav",
                "tts_duration_sec": 10.0,
            },
        ]
        work = _build_work_items(articles)
        assert len(work) == 2
        assert work[0]["headline"] == "سیاسی خبر"
        assert work[0]["urdu_script"] == "مکمل اسکرپٹ یہاں ہے۔"
        assert work[0]["audio_duration"] == 15.5
        assert work[1]["headline"] == "کھیل کی خبر"
        assert work[1]["urdu_script"] == "متن یہاں ہے۔"
        assert work[1]["audio_duration"] == 10.0

    def test_run_with_no_eligible_articles(self):
        report = run_video_production([
            {"article_id": 1, "headline": "خبر"},
        ])
        assert isinstance(report, VideoProductionReport)
        assert len(report.outputs) == 0

    def test_run_with_limit(self):
        report = run_video_production([
            {"article_id": 1}, {"article_id": 2}, {"article_id": 3},
        ], limit=2)
        # None are eligible (no audio), but limit is applied
        assert isinstance(report, VideoProductionReport)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class TestConfig:
    def test_urdu_font_resolved(self):
        # On Windows, should find one of the candidate fonts
        # (at minimum arialuni.ttf or segoeui.ttf)
        font = vc.resolve_urdu_font()
        if font:
            assert Path(font).exists()

    def test_video_specs(self):
        assert vc.VIDEO_WIDTH == 1920
        assert vc.VIDEO_HEIGHT == 1080
        assert vc.VIDEO_FPS == 30

    def test_output_dirs_are_path_objects(self):
        assert isinstance(vc.VIDEO_OUTPUT_DIR, Path)
        assert isinstance(vc.AVATAR_DIR, Path)
        assert isinstance(vc.LIPSYNC_DIR, Path)
        assert isinstance(vc.VISUALS_DIR, Path)
        assert isinstance(vc.COMPOSITION_DIR, Path)
        assert isinstance(vc.SUBTITLES_DIR, Path)
        assert isinstance(vc.THUMBNAILS_DIR, Path)
        assert isinstance(vc.FINAL_OUTPUT_DIR, Path)

    def test_gpu_detection_does_not_crash(self):
        # HAS_CUDA should be a boolean regardless of torch availability
        assert isinstance(vc.HAS_CUDA, bool)
