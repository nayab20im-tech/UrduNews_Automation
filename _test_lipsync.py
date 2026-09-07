"""Quick test: run the lip-sync animation engine on one article."""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, ".")

from video_production.lip_sync import LipSyncEngine
from video_production import config as vc

print("=== Configuration ===")
print(f"FFmpeg: {vc.FFMPEG_BINARY}")
print(f"Source image: {vc.AVATAR_SOURCE_IMAGE}")
print(f"Presenter: {vc.PRESENTER_WIDTH}x{vc.PRESENTER_HEIGHT} at ({vc.PRESENTER_X},{vc.PRESENTER_Y})")
print(f"Video: {vc.VIDEO_WIDTH}x{vc.VIDEO_HEIGHT} @ {vc.VIDEO_FPS}fps, CRF={vc.VIDEO_CRF}")
print()

# Find an article with audio
from video_production.pipeline import _filter_eligible, _build_work_items
from pipeline.processed_news_db import ProcessedNewsDatabase
from pipeline import config as pcfg

db = ProcessedNewsDatabase(db_path=pcfg.PROCESSED_DB_PATH)
articles = db.build_final_dataset()
print(f"Total articles in DB: {len(articles)}")

eligible = _filter_eligible(articles)
print(f"Eligible (with audio): {len(eligible)}")

if not eligible:
    print("No eligible articles!")
    sys.exit(1)

work = _build_work_items(eligible)
w = work[0]
print(f"\nTest article: id={w['article_id']}")
print(f"  Audio: {w['audio_path']}")
print(f"  Duration: {w['audio_duration']:.1f}s")
print(f"  Headline: {w['headline'][:60]}...")
print()

# Run just the lip-sync engine
engine = LipSyncEngine()
print(f"Backend: {engine.resolve_backend()}")
print("Running lip-sync animation (this may take a few minutes)...")

import time
start = time.time()

result = engine.sync(
    w["article_id"],
    "",  # no avatar video needed for animated backend
    w["audio_path"],
    w["audio_duration"],
)

elapsed = time.time() - start
print(f"\nResult: status={result.status}, backend={result.backend_used}")
print(f"  Video: {result.video_path}")
print(f"  Duration: {result.duration_sec}s")
print(f"  Elapsed: {elapsed:.1f}s")

if result.status == "ok":
    # Verify the output
    import subprocess
    cmd = [vc.FFMPEG_BINARY, "-i", result.video_path, "-hide_banner"]
    proc = subprocess.run(cmd, capture_output=True, timeout=10)
    info = proc.stderr.decode(errors="replace")
    for line in info.split("\n"):
        if "Duration" in line or "Stream" in line:
            print(f"  {line.strip()}")
    print("\nSUCCESS!")
else:
    print("FAILED!")
