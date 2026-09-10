"""
avatar_engine/gpu.py
=======================
GPU / device detection and management.

Logs CUDA + PyTorch versions and available VRAM before any inference,
selects the device automatically, and exposes helpers to release VRAM
between backend stages.  On CPU-only hosts (like the hackathon demo
machine) everything degrades gracefully to the CPU backends.
"""

from __future__ import annotations

from dataclasses import dataclass

from data_acquisition.utils.logger import get_logger

logger = get_logger("avatar_engine.gpu")


@dataclass
class GpuInfo:
    available: bool = False
    name: str = ""
    total_vram_mb: int = 0
    free_vram_mb: int = 0
    cuda_version: str = ""
    torch_version: str = ""


def detect_gpu() -> GpuInfo:
    """Probe NVIDIA CUDA availability without importing torch eagerly."""
    info = GpuInfo()
    try:
        import torch
        info.torch_version = torch.__version__
        if torch.cuda.is_available():
            info.available = True
            info.name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info.total_vram_mb = int(props.total_memory / 1024 / 1024)
            info.free_vram_mb = int(
                (props.total_memory - torch.cuda.memory_reserved(0))
                / 1024 / 1024)
            info.cuda_version = torch.version.cuda or ""
    except ImportError:
        pass
    if info.available:
        logger.info(
            "[GPU] %s | %d MB VRAM | torch %s | CUDA %s",
            info.name, info.total_vram_mb, info.torch_version,
            info.cuda_version,
        )
    else:
        logger.info("[GPU] No CUDA device — using CPU backends")
    return info


def select_device(prefer_gpu: bool = True) -> str:
    """Return 'cuda' when a GPU exists (and is wanted), else 'cpu'."""
    if not prefer_gpu:
        return "cpu"
    return "cuda" if detect_gpu().available else "cpu"


def release_gpu_memory() -> None:
    """Free cached VRAM between backend stages (no-op on CPU hosts)."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("[GPU] torch.cuda.empty_cache() called")
    except ImportError:
        pass
