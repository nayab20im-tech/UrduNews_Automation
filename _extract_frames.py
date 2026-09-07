"""Extract frames from the final video to verify visual quality."""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, ".")

import subprocess
from video_production import config as vc

video = r"C:\Users\User\Downloads\Project\UrduNews_AI\data_acquisition\data\media\video\composition\composed_1_subtitled.mp4"

# Extract frames at different timestamps: 2s, 10s, 20s, 30s, 39s
timestamps = [2, 10, 20, 30, 39]
out_dir = r"C:\Users\User\Downloads\Project\UrduNews_AI\_frames"
import os
os.makedirs(out_dir, exist_ok=True)

for t in timestamps:
    out = os.path.join(out_dir, f"frame_{t}s.jpg")
    cmd = [
        vc.FFMPEG_BINARY, "-y",
        "-ss", str(t),
        "-i", video,
        "-frames:v", "1",
        "-q:v", "2",
        out,
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=30)
    if os.path.exists(out):
        size = os.path.getsize(out)
        print(f"Frame at {t}s: {out} ({size/1024:.0f} KB)")
    else:
        print(f"Frame at {t}s: FAILED")

# Also get video info
cmd = [vc.FFMPEG_BINARY, "-i", video, "-hide_banner"]
proc = subprocess.run(cmd, capture_output=True, timeout=10)
info = proc.stderr.decode(errors="replace")
print("\n=== Video Info ===")
for line in info.split("\n"):
    if "Duration" in line or "Stream" in line:
        print(line.strip())
