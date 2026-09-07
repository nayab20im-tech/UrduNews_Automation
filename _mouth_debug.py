import sys, cv2, numpy as np
sys.path.insert(0, '.')
import importlib, video_production.lip_sync as ls
importlib.reload(ls)
from video_production.lip_sync import LipSyncEngine
from video_production import config as vc

eng = LipSyncEngine()
src = cv2.imread(vc.AVATAR_SOURCE_IMAGE)
pres_w, pres_h = vc.PRESENTER_WIDTH, vc.PRESENTER_HEIGHT
src = cv2.resize(src, (pres_w, pres_h))
face_info = eng._detect_face(src)
fx, fy, fw, fh = [int(x) for x in face_info['face']]
m = face_info['mouth']
print(f'Face: ({fx},{fy},{fw},{fh})')
print(f'Mouth: cx={m["cx"]} cy={m["cy"]} w={m["w"]} h={m["h"]}')

# Manual geometry calculation to understand what's happening
import math
open_amount = 1.0
max_gap = max(3, int(fh * 0.14))
gap_h = max(1, int(max_gap * open_amount))
mouth_hw = max(6, int(m["w"] * (0.48 + 0.06 * open_amount)))
print(f'max_gap={max_gap} gap_h={gap_h} mouth_hw={mouth_hw}')

# lip zone
mcy = m["cy"]
lip_zone_y1 = max(0, mcy - int(fh * 0.07))
lip_zone_y2 = min(pres_h, mcy + int(fh * 0.07))
print(f'lip_zone: y1={lip_zone_y1} y2={lip_zone_y2}')

# Crop the mouth region to see what's there
mx = m["cx"]
crop_y1 = max(0, mcy - 80)
crop_y2 = min(pres_h, mcy + 80)
crop_x1 = max(0, mx - 100)
crop_x2 = min(pres_w, mx + 100)
mouth_crop = src[crop_y1:crop_y2, crop_x1:crop_x2]
big_mc = cv2.resize(mouth_crop, (mouth_crop.shape[1]*3, mouth_crop.shape[0]*3), interpolation=cv2.INTER_CUBIC)
cv2.imwrite('_mouth_raw.png', big_mc)
print(f'Mouth region saved: {crop_y1}:{crop_y2} x {crop_x1}:{crop_x2}')

# Also save with roi applied
fr_test = src.copy()
fr_test = eng._animate_mouth(fr_test, face_info, 1.0)
mouth_after = fr_test[crop_y1:crop_y2, crop_x1:crop_x2]
big_ma = cv2.resize(mouth_after, (mouth_after.shape[1]*3, mouth_after.shape[0]*3), interpolation=cv2.INTER_CUBIC)
cv2.imwrite('_mouth_after_open1.png', big_ma)
print('Saved _mouth_after_open1.png')

diff = np.abs(mouth_after.astype(np.float32) - mouth_crop.astype(np.float32))
print(f'Diff stats: mean={diff.mean():.2f} max={diff.max():.2f} nonzero={np.count_nonzero(diff)}/{diff.size}')
