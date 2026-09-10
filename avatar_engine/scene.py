"""
avatar_engine/scene.py
=========================
Dual-anchor broadcast scene.

Builds the 1080p studio frame from the canonical two-anchor photo
(``anchors.jpeg`` — male + female behind the Pakistan News desk) and
locates both faces so the talking-head backends can animate them
in-place: the speaking anchor is audio-driven, the listening anchor
keeps natural idle motion (blinks, micro-sway).

The side margins of the 16:9 frame are filled with a blurred mirror
extension of the photo edges so the original composition stays
untouched and centred.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data_acquisition.utils.logger import get_logger

from . import config as cfg
from .errors import AvatarGenerationError

logger = get_logger("avatar_engine.scene")

# Face boxes verified on the canonical 1402x1122 source image
# (prepare_anchors.py uses the same geometry).  Used only as a
# deterministic fallback when the cascade detects fewer than 2 faces.
_CANONICAL_FACES = {
    "male": (260, 205, 201, 201),
    "female": (946, 231, 179, 179),
}


@dataclass
class Scene:
    """Prepared broadcast frame + per-anchor face rectangles."""

    canvas: np.ndarray                      # (H, W, 3) uint8 BGR
    faces: dict                             # {"male": (x,y,w,h), "female": ...}
    source_size: tuple                      # (w, h) of the scaled photo
    photo_offset: int                       # x offset of the photo in canvas


def build_scene(width: int | None = None, height: int | None = None) -> Scene:
    """Compose the 1080p dual-anchor canvas and locate both faces."""
    import cv2

    W = width or cfg.VIDEO_WIDTH
    H = height or cfg.VIDEO_HEIGHT

    src_path = cfg.ANCHOR_SOURCE_IMAGE
    if not src_path or not Path(src_path).exists():
        raise AvatarGenerationError(
            "Dual scene needs the two-anchor source image "
            f"(ANCHOR_SOURCE_IMAGE={src_path!r})")
    src = cv2.imread(str(src_path))
    if src is None:
        raise AvatarGenerationError(f"Could not read anchor image {src_path}")

    sh, sw = src.shape[:2]
    scale = H / sh
    new_w = int(sw * scale)
    scaled = cv2.resize(src, (new_w, H), interpolation=cv2.INTER_LANCZOS4)

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    ox = (W - new_w) // 2

    # Side fill: mirrored, heavily blurred edge strips of the photo so
    # the 16:9 frame looks like one continuous studio shot.
    if ox > 0:
        edge = max(24, new_w // 12)
        left_src = cv2.flip(scaled[:, :edge], 1)
        right_src = cv2.flip(scaled[:, -edge:], 1)
        left_fill = cv2.resize(left_src, (ox + 2, H),
                               interpolation=cv2.INTER_LINEAR)
        right_fill = cv2.resize(right_src, (W - (ox + new_w) + 2, H),
                                interpolation=cv2.INTER_LINEAR)
        k = (ox // 2) | 1
        k = max(k, 31)
        left_fill = cv2.GaussianBlur(left_fill, (k, k), 0)
        right_fill = cv2.GaussianBlur(right_fill, (k, k), 0)
        canvas[:, :ox] = left_fill[:, :ox]
        canvas[:, ox + new_w:] = right_fill[:, -(W - ox - new_w):]

    canvas[:, ox:ox + new_w] = scaled

    faces = _detect_two_faces(canvas, scale, ox)
    logger.info("[SCENE] dual canvas %dx%d | faces=%s", W, H,
                {k: tuple(int(v) for v in r) for k, r in faces.items()})
    return Scene(canvas=canvas, faces=faces,
                 source_size=(new_w, H), photo_offset=ox)


def _detect_two_faces(canvas: np.ndarray, scale: float, ox: int) -> dict:
    """Locate male (left) and female (right) faces in the canvas."""
    import cv2

    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = []
    for sf, mn, ms in [(1.05, 5, 40), (1.03, 3, 30)]:
        det = cascade.detectMultiScale(
            gray, scaleFactor=sf, minNeighbors=mn, minSize=(ms, ms))
        det = [f for f in det if f[3] >= canvas.shape[0] * 0.12]
        if len(det) >= 2:
            faces = sorted(det, key=lambda f: f[2] * f[3], reverse=True)[:2]
            break
    if len(faces) == 2:
        faces = sorted(faces, key=lambda f: f[0])   # left → right
        return {"male": tuple(int(v) for v in faces[0]),
                "female": tuple(int(v) for v in faces[1])}

    # Deterministic fallback: scale the verified canonical boxes.
    logger.warning("[SCENE] cascade found %d face(s) — using canonical "
                   "geometry", len(faces))
    return {
        name: (int(ox + x * scale), int(y * scale),
               int(w * scale), int(h * scale))
        for name, (x, y, w, h) in _CANONICAL_FACES.items()
    }


def speaker_schedule(n_chunks: int, first_speaker: str) -> list[str]:
    """Alternate speakers per chunk, like real co-anchor broadcasts."""
    other = "male" if first_speaker == "female" else "female"
    return [first_speaker if i % 2 == 0 else other for i in range(n_chunks)]
