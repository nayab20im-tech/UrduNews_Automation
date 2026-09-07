import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, ".")

import video_production.config as vc
from video_production.lip_sync import LipSyncEngine
from video_production.visual_generator import VisualGenerator
from video_production.video_composer import VideoComposer
from data_acquisition.utils.logger import get_logger
import wave
import contextlib

logger = get_logger("demo")

out_dir = vc._MODULE_DIR.parent / "DemoOutput"
out_dir.mkdir(exist_ok=True)

# 1. Use real TTS audio from previous runs
audio_src = "output/avatar_engine/work/4c71bc86b4e3/audio.wav"
if not os.path.exists(audio_src):
    logger.error("Audio not found: " + audio_src)
    sys.exit(1)

with contextlib.closing(wave.open(audio_src, 'r')) as f:
    frames = f.getnframes()
    rate = f.getframerate()
    duration = frames / float(rate)

logger.info(f"Using audio: {audio_src} (duration: {duration:.2f}s)")

# 2. Visual Layer Generation (Background, Lower-Thirds, Tickers)
logger.info("Generating visuals (ARY style)...")
vis = VisualGenerator(output_dir=out_dir)
r_vis = vis.generate(42, "بجلی کی قیمتوں میں ہوشربا اضافہ، عوام پریشان", "Breaking News")

# 3. Avatar Generation (D-ID via API)
logger.info("Generating avatar (D-ID API)...")

# 4. Lip Sync Layer (Skips processing if D-ID was used)
logger.info("Generating lip-sync layer...")
eng = LipSyncEngine(output_dir=out_dir, backend="auto")
r_lip = eng.sync(42, "DemoOutput/avatar_42.mp4", audio_src, duration)

# 5. Composite everything in FFmpeg (Watermark, Ticker Scroll, News Panel)
logger.info("Composing final ARY-style broadcast video...")
comp = VideoComposer(output_dir=out_dir)
r_comp = comp.compose(
    article_id=42,
    lip_sync_video=r_lip.video_path,
    audio_path=audio_src,
    audio_duration=duration,
    background_image=r_vis.background_path,
    lower_third_image=r_vis.lower_third_path,
    ticker_image=r_vis.ticker_path
)

logger.info(f"DONE! Final broadcast-quality video is located at: {r_comp.video_path}")
print(f"OUTPUT_VIDEO_PATH={r_comp.video_path}")
