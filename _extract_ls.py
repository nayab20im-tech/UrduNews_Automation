"""Extract frames from the lip-sync video (before composition) to check avatar rendering."""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, ".")

import subprocess
import os
from video_production import config as vc

video = r"C:\Users\User\Downloads\Project\UrduNews_AI\data_acquisition\data\media\video\lip_sync\lipsync_1.mp4"
out_dir = r"C:\Users\User\Downloads\Project\UrduNews_AI\_frames_lipsync"
os.makedirs(out_dir, exist_ok=True)

# Extract frames at different timestamps to see mouth animation states
timestamps = [2, 5, 10, 15, 20, 25]
for t in timestamps:
    out = os.path.join(out_dir, f"ls_{t}s.jpg")
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
        print(f"Frame at {t}s: OK ({os.path.getsize(out)/1024:.0f} KB)")
    else:
        print(f"Frame at {t}s: FAILED")
