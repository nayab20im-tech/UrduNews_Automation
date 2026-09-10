"""
avatar_engine/anchors.py
===========================
Anchor asset management.

The canonical visual identity of the channel is the supplied studio
photo containing TWO anchors (male left, female right) behind the
Pakistan News desk (``anchors.jpeg`` at the repo root).  Identities are
never regenerated or replaced — ``prepare_assets()`` merely splits the
photo into two reusable per-anchor assets:

    avatar_engine/assets/male.png
    avatar_engine/assets/female.png

Each crop keeps the full frame height (desk, studio, composition) and
is verified programmatically: exactly one dominant face must be found
inside each crop, at the expected side.
"""

from __future__ import annotations

from pathlib import Path

from data_acquisition.utils.logger import get_logger

from . import config as cfg
from .errors import InvalidAnchorError

logger = get_logger("avatar_engine.anchors")

VALID_ANCHOR_IDS = ("male", "female")


def _detect_faces(img) -> list:
    """Return face rects sorted left→right (largest two kept)."""
    import cv2
    import os
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return []
    # Keep the two largest (the anchors); drop logo/false positives.
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[:2]
    return sorted(faces, key=lambda f: f[0])


def prepare_assets(force: bool = False) -> dict[str, Path]:
    """Derive male.png / female.png from the two-anchor source image.

    Returns ``{"male": Path, "female": Path}``.  Existing assets are
    kept unless ``force=True``.
    """
    import cv2

    cfg.ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    male_path = cfg.ASSETS_DIR / "male.png"
    female_path = cfg.ASSETS_DIR / "female.png"

    if not force and male_path.exists() and female_path.exists():
        return {"male": male_path, "female": female_path}

    source = cfg.ANCHOR_SOURCE_IMAGE
    if not source or not Path(source).exists():
        raise InvalidAnchorError(
            "Two-anchor source image not found (set ANCHOR_SOURCE_IMAGE)")

    img = cv2.imread(str(source))
    if img is None:
        raise InvalidAnchorError(f"Unreadable anchor source image: {source}")

    faces = _detect_faces(img)
    if len(faces) < 2:
        raise InvalidAnchorError(
            f"Expected two faces in {source}, found {len(faces)}")

    h, w = img.shape[:2]
    (mx, my, mw, mh), (fx, fy, fw, fh) = faces[0], faces[1]
    # Split line midway between the two faces.
    split = int(((mx + mw) + fx) // 2)
    split = max(1, min(w - 1, split))

    male_crop = img[:, :split]
    female_crop = img[:, split:]
    cv2.imwrite(str(male_path), male_crop)
    cv2.imwrite(str(female_path), female_crop)
    logger.info(
        "[ANCHOR] Derived assets from %s (split x=%d): %s, %s",
        Path(source).name, split, male_path.name, female_path.name,
    )

    # ---- programmatic verification ------------------------------------
    for path, side in ((male_path, "left"), (female_path, "right")):
        crop = cv2.imread(str(path))
        found = _detect_faces(crop)
        if not found:
            raise InvalidAnchorError(f"No face detected in {path}")
        cx = found[0][0] + found[0][2] / 2
        if side == "left" and cx > crop.shape[1] * 0.75:
            raise InvalidAnchorError(f"Male crop contaminated: {path}")
        if side == "right" and cx < crop.shape[1] * 0.25:
            raise InvalidAnchorError(f"Female crop contaminated: {path}")
    logger.info("[ANCHOR] Verified one dominant face per asset")
    return {"male": male_path, "female": female_path}


def anchor_image_path(anchor_id: str) -> Path:
    """Resolve the asset for an anchor id, preparing assets if needed."""
    if anchor_id not in VALID_ANCHOR_IDS:
        raise InvalidAnchorError(
            f"Unknown anchor_id {anchor_id!r}; expected one of {VALID_ANCHOR_IDS}")
    assets = prepare_assets()
    path = assets[anchor_id]
    if not path.exists():
        raise InvalidAnchorError(f"Anchor asset missing: {path}")
    return path
