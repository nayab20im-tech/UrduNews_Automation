"""
avatar_engine/backends/__init__.py
=====================================
Talking-head backend registry.

``resolve_backend()`` implements the ``auto`` policy: the most reliable
*verified* backend on this host wins (SadTalker only when its repo,
checkpoints and a CUDA GPU are all present; otherwise the deterministic
CPU ``animated`` renderer).

Backends considered but intentionally NOT implemented (feasibility /
license audit, see README.md):

* **MuseTalk**      — real-time latent inpainting; requires CUDA + a
  specific checkpoint pairing that changed between releases; unstable
  documentation → experimental, not production.
* **LivePortrait**  — expression/pose retargeting, not audio-driven lip
  sync by itself; needs CUDA and a separate audio→motion front end.
* **EchoMimicV2/3** — CUDA-only, heavy deps, gesture models trained on
  English/Chinese audio; unreliable Urdu performance → keep
  experimental behind ``gesture_enabled`` once ported.
"""

from __future__ import annotations

from data_acquisition.utils.logger import get_logger

from .. import config as cfg
from ..errors import ModelNotFoundError
from .animated import AnimatedBackend
from .base import TalkingHeadBackend
from .sadtalker import SadTalkerBackend

logger = get_logger("avatar_engine.backends")

_REGISTRY = {
    "animated": AnimatedBackend,
    "sadtalker": SadTalkerBackend,
}


def get_backend(name: str, config: cfg.AvatarConfig | None = None) -> TalkingHeadBackend:
    if name not in _REGISTRY:
        raise ModelNotFoundError(
            f"Unknown talking-head backend {name!r}; "
            f"available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](config)


def resolve_backend(config: cfg.AvatarConfig | None = None) -> TalkingHeadBackend:
    config = config or cfg.AvatarConfig()
    want = config.avatar_backend
    if want not in ("auto", ""):
        backend = get_backend(want, config)
        if not backend.available():
            raise ModelNotFoundError(
                f"Backend {want!r} explicitly requested but unavailable "
                "on this host")
        return backend
    for name in ("sadtalker", "animated"):
        backend = get_backend(name, config)
        if backend.available():
            logger.info("[AVATAR] backend resolved: %s", name)
            return backend
    raise ModelNotFoundError("No talking-head backend available")
