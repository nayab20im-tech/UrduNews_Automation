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
if face_info:
    fx, fy, fw, fh = [int(x) for x in face_info['face']]
    m = face_info['mouth']
    print(f'Face: ({fx},{fy},{fw},{fh})  h%={fh/pres_h*100:.1f}%')
    print(f'Mouth cx={m["cx"]} cy={m["cy"]} w={m["w"]} h={m["h"]}')
    print(f'Skin: {face_info["skin_color"]}')
    debug = src.copy()
    cv2.rectangle(debug, (fx,fy), (fx+fw, fy+fh), (0,255,0), 3)
    cv2.circle(debug, (m['cx'], m['cy']), 8, (0,0,255), -1)
    cv2.rectangle(debug, (m['x'], m['y']), (m['x']+m['w'], m['y']+m['h']), (0,0,255), 2)
    small = cv2.resize(debug, (325, 440))
    cv2.imwrite('_face_debug3.png', small)
    print('Saved _face_debug3.png')
else:
    print('None returned')
