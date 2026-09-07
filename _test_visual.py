"""Quick visual test: render a few frames of the animation to check quality."""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, ".")

import cv2
import numpy as np
from video_production.lip_sync import LipSyncEngine
from video_production import config as vc

engine = LipSyncEngine()

# Load and prepare source image (same as _run_animated)
src = cv2.imread(vc.AVATAR_SOURCE_IMAGE)
print(f"Source: {src.shape[1]}x{src.shape[0]}")

pres_w, pres_h = vc.PRESENTER_WIDTH, vc.PRESENTER_HEIGHT
raw_face = engine._detect_face_raw(src)
print(f"Raw face: {raw_face}")

# Face-centred crop
src_h, src_w = src.shape[:2]
target_ratio = pres_w / pres_h
current_ratio = src_w / src_h
if current_ratio > target_ratio:
    new_w = int(src_h * target_ratio)
    face_cx = raw_face[0] + raw_face[2] // 2 if raw_face is not None else src_w // 2
    crop_x = max(0, min(src_w - new_w, face_cx - new_w // 2))
    src = src[:, crop_x:crop_x + new_w]
    print(f"Crop: width {new_w}, x={crop_x} to {crop_x+new_w}")
else:
    new_h = int(src_w / target_ratio)
    face_cy = raw_face[1] + raw_face[3] // 2 if raw_face is not None else src_h // 2
    crop_y = max(0, min(src_h - new_h, face_cy - int(new_h * 0.42)))
    src = src[crop_y:crop_y + new_h, :]
    print(f"Crop: height {new_h}, y={crop_y} to {crop_y+new_h}")

presenter = cv2.resize(src, (pres_w, pres_h), interpolation=cv2.INTER_LANCZOS4)
face_info = engine._detect_face(presenter)
print(f"Face on presenter: {face_info['face']}")
print(f"Left eye: {face_info['left_eye']}")
print(f"Right eye: {face_info['right_eye']}")
print(f"Mouth: {face_info['mouth']}")
print(f"Skin color: {face_info['skin_color']}")

# Render studio background
studio_bg = engine._render_studio_background()
print(f"Studio bg: {studio_bg.shape}")

# Render test frames with different mouth openings
import os
out_dir = "_test_frames"
os.makedirs(out_dir, exist_ok=True)

for open_amt, blink in [(0.0, 0.0), (0.3, 0.0), (0.7, 0.0), (1.0, 0.0), (0.5, 1.0)]:
    frame = presenter.copy()
    frame = engine._animate_mouth(frame, face_info, open_amt)
    if blink > 0:
        frame = engine._animate_blink(frame, face_info, blink)
    
    # Composite onto studio
    canvas = studio_bg.copy()
    px, py = vc.PRESENTER_X, vc.PRESENTER_Y
    canvas[py:py+pres_h, px:px+pres_w] = frame
    
    out = os.path.join(out_dir, f"test_open{open_amt:.1f}_blink{blink:.1f}.jpg")
    cv2.imwrite(out, canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"Saved: {out}")

print("\nDone! Check _test_frames/ directory")
