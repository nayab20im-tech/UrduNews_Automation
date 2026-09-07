import wave, tempfile, pathlib, sys
sys.path.insert(0, ".")
from video_production.lip_sync import LipSyncEngine
from video_production.visual_generator import VisualGenerator
from video_production.video_composer import VideoComposer
from data_acquisition.utils.logger import get_logger

logger = get_logger("demo")

out_dir = pathlib.Path("DemoOutput")
out_dir.mkdir(exist_ok=True)

# 1. Create a 5-second empty WAV file (silent audio)
wav = out_dir / "demo_audio.wav"
with wave.open(str(wav), "w") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(16000)
    w.writeframes(b"\x00" * 16000*5*2)

# 2. Generate visuals
logger.info("Generating visuals...")
vis = VisualGenerator(output_dir=out_dir)
r_vis = vis.generate(999, "بجلی کی قیمتوں میں ہوشربا اضافہ، عوام پریشان", "Breaking News")

# 3. Generate CPU Animated Lip-sync (fallback since we have no API keys)
logger.info("Generating lip-sync animation (CPU fallback)...")
eng = LipSyncEngine(output_dir=out_dir, backend="animated")
r_lip = eng.sync(999, "", str(wav), 5.0)

# 4. Compose final video
logger.info("Composing final video...")
comp = VideoComposer(output_dir=out_dir)
r_comp = comp.compose(
    article_id=999,
    lip_sync_video=r_lip.video_path,
    audio_path=str(wav),
    audio_duration=5.0,
    background_image=r_vis.background_path,
    lower_third_image=r_vis.lower_third_path,
    ticker_image=r_vis.ticker_path
)

logger.info(f"Done! Final video is at: {r_comp.video_path}")
print(f"OUTPUT_VIDEO_PATH={r_comp.video_path}")
