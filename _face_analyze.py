"""Visualize the source image to understand face position."""
import sys, cv2, numpy as np
sys.path.insert(0, '.')
from video_production import config as vc

src = cv2.imread(vc.AVATAR_SOURCE_IMAGE)
pres_w, pres_h = vc.PRESENTER_WIDTH, vc.PRESENTER_HEIGHT
src = cv2.resize(src, (pres_w, pres_h))
h, w = src.shape[:2]
print(f'Image size: {w}x{h}')

# Try to find face by looking for the densest skin region in the upper portion
# Skin in BGR for Asian-tone: B=130-180, G=130-185, R=150-210
# Let's sample pixels at candidate face positions
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# The face should be in the upper-middle area
# Check y=50 to y=500, x=150 to x=500 for this 650x880 image
# Find region with most face-like edge density (faces have more edges than backgrounds)
face_region = src[50:550, 150:500]
print('Face candidate region brightness stats:')
print('  mean:', face_region.mean(axis=(0,1)))
print('  (B, G, R)')

# Try finding where horizontal skin gradients are highest (eyes/lips/nose area)
# by checking column brightness profiles
gray_face = gray[50:550, 150:500]
# Vertical profile - find "face row" by looking for uniform-brightness horizontal band
col_profile = gray_face.mean(axis=1)
print('Vertical brightness profile (50-550):')
for y in range(0, 500, 50):
    print(f'  y={y+50}: {col_profile[y]:.1f}')
