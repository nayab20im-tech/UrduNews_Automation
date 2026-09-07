"""
run_news_video.py
=================
End-to-end ARY-style news video generator.

What it does:
  1. Reads the latest processed news article from the database
     (structured_scripts + classified_news tables).
  2. Generates fresh Urdu female voice audio via Microsoft Edge Neural TTS
     (ur-PK-UzmaNeural).
  3. Sends the audio + avatar image to D-ID API (stitch=True for full-frame).
  4. Composites the final 1920x1080 ARY-style broadcast video.

Usage:
    source venv/bin/activate
    python run_news_video.py [--article-id N]

Output:
    FinalVideos/news_<article_id>.mp4
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, ".")

import video_production.config as vc
from video_production.avatar_generator import AvatarGenerator
from video_production.lip_sync import LipSyncEngine
from video_production.visual_generator import VisualGenerator
from video_production.video_composer import VideoComposer
from tts.tts_engine import TTSEngine
from data_acquisition.utils.logger import get_logger

logger = get_logger("run_news_video")

DB_PATH = Path("data_acquisition/data/raw_news.db")
OUTPUT_DIR = Path("FinalVideos")
OUTPUT_DIR.mkdir(exist_ok=True)


def fetch_article(article_id: int | None = None) -> dict:
    """Fetch one article from the database with headline, script and category."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    if article_id:
        cur.execute("""
            SELECT ss.article_id, ss.headline, ss.full_script,
                   COALESCE(cn.category, 'Breaking News') AS category
            FROM structured_scripts ss
            LEFT JOIN classified_news cn ON ss.article_id = cn.article_id
            WHERE ss.status = 'ok' AND ss.article_id = ?
        """, (article_id,))
    else:
        cur.execute("""
            SELECT ss.article_id, ss.headline, ss.full_script,
                   COALESCE(cn.category, 'Breaking News') AS category
            FROM structured_scripts ss
            LEFT JOIN classified_news cn ON ss.article_id = cn.article_id
            WHERE ss.status = 'ok' AND ss.full_script IS NOT NULL
              AND length(ss.full_script) > 50
            ORDER BY ss.article_id DESC
            LIMIT 1
        """)

    row = cur.fetchone()
    conn.close()
    if not row:
        raise RuntimeError("No valid articles found in DB")
    return dict(row)


def generate_female_audio(article_id: int, script: str, out_dir: Path) -> tuple[str, float]:
    """Generate female Urdu voice audio using edge-tts (ur-PK-UzmaNeural)."""
    import edge_tts
    import tempfile
    import shutil
    import subprocess
    import wave
    import contextlib

    voice = "ur-PK-UzmaNeural"
    mp3_tmp = Path(tempfile.mktemp(suffix=".mp3"))
    wav_path = out_dir / f"audio_{article_id}.wav"

    logger.info("Generating female Urdu audio with voice: %s", voice)

    async def _synth():
        communicate = edge_tts.Communicate(script, voice)
        await communicate.save(str(mp3_tmp))

    asyncio.run(_synth())

    if not mp3_tmp.exists():
        raise RuntimeError("edge-tts failed to produce audio")

    ffmpeg = shutil.which("ffmpeg")
    subprocess.run(
        [ffmpeg, "-y", "-i", str(mp3_tmp), "-ar", "22050", "-ac", "1", str(wav_path)],
        capture_output=True, check=True,
    )
    mp3_tmp.unlink(missing_ok=True)

    with contextlib.closing(wave.open(str(wav_path), 'r')) as f:
        duration = f.getnframes() / float(f.getframerate())

    logger.info("Audio written: %s (%.2fs)", wav_path, duration)
    return str(wav_path), duration


def main():
    parser = argparse.ArgumentParser(description="Generate ARY-style news video")
    parser.add_argument("--article-id", type=int, default=None,
                        help="Specific article ID to use (default: latest)")
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Step 1: Fetch real news article from database
    # ------------------------------------------------------------------ #
    logger.info("=== Step 1: Fetching article from database ===")
    article = fetch_article(args.article_id)
    article_id = article["article_id"]
    headline   = article["headline"]
    script     = article["full_script"]
    category   = article["category"]

    logger.info("Article #%d | Category: %s", article_id, category)
    logger.info("Headline: %s", headline[:80])

    # ------------------------------------------------------------------ #
    # Step 2: Generate female Urdu voice audio
    # ------------------------------------------------------------------ #
    logger.info("=== Step 2: Generating female Urdu TTS audio ===")
    audio_path, duration = generate_female_audio(article_id, script, OUTPUT_DIR)

    # ------------------------------------------------------------------ #
    # Step 3: Generate D-ID talking avatar (full-frame, stitched)
    # ------------------------------------------------------------------ #
    logger.info("=== Step 3: Generating D-ID avatar (stitched full-frame) ===")
    av = AvatarGenerator(output_dir=OUTPUT_DIR, backend="auto")
    r_av = av.generate(article_id, audio_path, duration)
    if r_av.status != "ok" or not r_av.video_path:
        logger.error("Avatar generation failed! Status: %s", r_av.status)
        sys.exit(1)
    logger.info("Avatar ready: %s (backend: %s)", r_av.video_path, r_av.backend_used)

    # ------------------------------------------------------------------ #
    # Step 4: Lip-sync (D-ID bypass \u2014 video is already synced)
    # ------------------------------------------------------------------ #
    logger.info("=== Step 4: Lip-sync (D-ID pass-through) ===")
    eng = LipSyncEngine(output_dir=OUTPUT_DIR, backend="auto")
    r_lip = eng.sync(article_id, r_av.video_path, audio_path, duration)
    if not r_lip.video_path:
        logger.error("Lip-sync step produced no output")
        sys.exit(1)
    logger.info("Lip-sync ready: %s (backend: %s)", r_lip.video_path, r_lip.backend_used)

    # ------------------------------------------------------------------ #
    # Step 5: Generate broadcast visuals (background, lower-third, ticker)
    # ------------------------------------------------------------------ #
    logger.info("=== Step 5: Generating ARY-style broadcast visuals ===")
    vis = VisualGenerator(output_dir=OUTPUT_DIR)
    r_vis = vis.generate(article_id, headline, category)

    # ------------------------------------------------------------------ #
    # Step 6: Compose final video with FFmpeg
    # ------------------------------------------------------------------ #
    logger.info("=== Step 6: Compositing final broadcast video ===")
    final_dir = OUTPUT_DIR
    comp = VideoComposer(output_dir=final_dir)
    r_comp = comp.compose(
        article_id=article_id,
        lip_sync_video=r_lip.video_path,
        audio_path=audio_path,
        audio_duration=duration,
        background_image=r_vis.background_path,
        lower_third_image=r_vis.lower_third_path,
        ticker_image=r_vis.ticker_path,
    )

    # ------------------------------------------------------------------ #
    # Done!
    # ------------------------------------------------------------------ #
    final_path = r_comp.video_path
    logger.info("=" * 60)
    logger.info("Video generation complete!")
    logger.info("Article  : #%d \u2014 %s", article_id, headline[:60])
    logger.info("Category : %s", category)
    logger.info("Duration : %.2f seconds", duration)
    logger.info("Output   : %s", final_path)
    logger.info("=" * 60)
    print(f"\nFINAL VIDEO PATH: {final_path}\n")


if __name__ == "__main__":
    main()
