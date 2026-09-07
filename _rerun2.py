"""Re-run stages 3.2, 3.4 and 3.5 with the new painted mouth engine."""
import sys, os, glob as _glob
sys.path.insert(0, os.path.dirname(__file__))
from pathlib import Path

# Audio
audio_path = ""
for pattern in [
    r"data_acquisition\data\media\audio\article_1.wav",
]:
    if Path(pattern).exists():
        audio_path = pattern
        break
if not audio_path:
    found = _glob.glob(r"data_acquisition\data\media\audio\article_1*", recursive=False)
    audio_path = found[0] if found else ""
if not audio_path:
    print("ERROR: audio not found"); sys.exit(1)

from audio_processing.wav_io import read_wav
wav = read_wav(audio_path)
duration = len(wav.samples) / wav.rate / max(wav.channels, 1)
print(f"Audio: {audio_path}  duration={duration:.2f}s")

# Script text from SRT
srt = Path(r"data_acquisition\data\media\video\subtitles\subtitles_1.srt")
raw = srt.read_text(encoding="utf-8-sig", errors="replace")
import re
lines = [l.strip() for l in raw.splitlines()
         if l.strip() and not l.strip().isdigit() and "-->" not in l]
script_text = " ".join(lines)
print(f"Script from SRT: {len(script_text)} chars")
article_id = 1

# Stage 3.2 — Lip-sync
print("\n=== Stage 3.2: Lip-sync (painted mouth) ===")
from video_production.lip_sync import LipSyncEngine
engine = LipSyncEngine()
result = engine.sync(article_id, "", audio_path, duration)
print(f"  status={result.status}  backend={result.backend_used}")
if result.status not in ("ok", "success"):
    print("ABORT"); sys.exit(1)
lipsync_video = result.video_path

# Stage 3.4 — Composition
print("\n=== Stage 3.4: Composition ===")
from video_production.video_composer import VideoComposer
from video_production import config as vc
lt_dir = vc.VISUALS_DIR
lt_img = str(lt_dir / "lower_third_1.png") if (lt_dir / "lower_third_1.png").exists() else ""
tk_img = str(lt_dir / "ticker_1.png") if (lt_dir / "ticker_1.png").exists() else ""
composer = VideoComposer()
comp = composer.compose(article_id, lipsync_video, audio_path, duration,
                        lower_third_image=lt_img, ticker_image=tk_img)
print(f"  status={comp.status}  path={comp.video_path}")
if comp.status != "ok":
    print("ABORT"); sys.exit(1)

# Stage 3.5 — Subtitles
print("\n=== Stage 3.5: Subtitles ===")
from video_production.subtitle_generator import SubtitleGenerator
gen = SubtitleGenerator()
sub = gen.generate(article_id, script_text, comp.video_path, duration)
print(f"  status={sub.status}  count={sub.subtitle_count}  path={sub.subtitled_video_path}")

final = sub.subtitled_video_path or comp.video_path
print(f"\nFinal: {final}")
print("Done.")
