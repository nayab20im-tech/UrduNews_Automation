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
print(f'Face: ({fx},{fy},{fw},{fh})  fh={fh}px')
print(f'Mouth: cx={m["cx"]} cy={m["cy"]} w={m["w"]} h={m["h"]}')

pad = 50
c1 = max(0, fx - pad); c2 = min(pres_w, fx + fw + pad)
r1 = max(0, fy - pad); r2 = min(pres_h, fy + fh + pad)

states = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
crops = []
for oa in states:
    fr = src.copy()
    fr = eng._animate_mouth(fr, face_info, oa)
    crop = fr[r1:r2, c1:c2]
    big = cv2.resize(crop, (crop.shape[1] * 3, crop.shape[0] * 3), interpolation=cv2.INTER_CUBIC)
    cv2.putText(big, f'open={oa:.1f}', (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    crops.append(big)

row = np.hstack(crops)
cv2.imwrite('_mtest3.png', row)
print('Saved _mtest3.png')
base = crops[0].astype(np.float32)
for i, oa in enumerate(states):
    d = float(np.abs(crops[i].astype(np.float32) - base).mean())
    print(f'open={oa:.1f}  diff={d:.2f}')
