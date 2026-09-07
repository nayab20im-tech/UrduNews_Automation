"""
monitoring/system_check.py
==========================
Operationalises Section 8's "System Requirements (Suggested)" box:
probes the current host and compares it against the recommended
specification so operators know which backends will actually run
(GPU avatar vs placeholder, real FFmpeg composition vs fallback, ...).

Every check returns ``ok`` (meets the suggestion), ``warn`` (works but
below the suggested spec / optional component missing -- the pipeline
degrades gracefully) or ``missing`` (a hard requirement is absent).
Pure stdlib + optional torch probe; never raises.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

from data_acquisition import config as acq_cfg
from data_acquisition.utils.logger import get_logger

logger = get_logger("monitoring.system_check")

# Section 8 suggested specification (from the architecture diagram).
RECOMMENDED = {
    "cpu_cores": 8,
    "ram_gb": 32,
    "gpu_vram_gb": 6,
    "storage_total_gb": 1000,
    "python_version": (3, 10),
}


def _ram_total_gb() -> float:
    """Total RAM in GB; Linux /proc first, then best-effort fallbacks."""
    try:
        with open("/proc/meminfo", "r", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 1024 / 1024, 1)
    except OSError:
        pass
    try:  # macOS / generic
        import subprocess

        out = subprocess.run(["sysctl", "-n", "hw.memsize"],
                             capture_output=True, text=True, timeout=5)
        return round(int(out.stdout.strip()) / 1024 ** 3, 1)
    except Exception:
        return -1.0


def _gpu_info() -> Dict[str, Any]:
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {"available": True, "name": props.name,
                    "vram_gb": round(props.total_memory / 1024 ** 3, 1)}
        return {"available": False, "name": "", "vram_gb": 0.0}
    except ImportError:
        return {"available": False, "name": "", "vram_gb": 0.0}


def collect_system_check() -> List[Dict[str, Any]]:
    """One entry per Section 8 requirement: status + detail + suggestion."""
    checks: List[Dict[str, Any]] = []

    # Software: Python 3.10+
    checks.append({
        "name": "python_version",
        "status": "ok" if sys.version_info >= RECOMMENDED["python_version"] else "missing",
        "detail": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "recommended": "Python 3.10+",
    })

    # Hardware: CPU 8+ cores
    cores = os.cpu_count() or 0
    checks.append({
        "name": "cpu_cores",
        "status": "ok" if cores >= RECOMMENDED["cpu_cores"] else "warn",
        "detail": f"{cores} cores",
        "recommended": f"{RECOMMENDED['cpu_cores']}+ cores",
    })

    # Hardware: RAM 32GB+
    ram = _ram_total_gb()
    checks.append({
        "name": "ram",
        "status": "ok" if ram >= RECOMMENDED["ram_gb"] else ("warn" if ram > 0 else "warn"),
        "detail": f"{ram} GB" if ram > 0 else "unknown",
        "recommended": f"{RECOMMENDED['ram_gb']}GB or more",
    })

    # Hardware: GPU NVIDIA 6GB+ (recommended, not required)
    gpu = _gpu_info()
    checks.append({
        "name": "gpu",
        "status": "ok" if gpu["available"] and gpu["vram_gb"] >= RECOMMENDED["gpu_vram_gb"]
        else "warn",
        "detail": f"{gpu['name']} ({gpu['vram_gb']} GB)" if gpu["available"]
        else "not detected -- CPU-only; placeholder avatar/lip-sync backends used",
        "recommended": f"NVIDIA {RECOMMENDED['gpu_vram_gb']}GB+ (recommended)",
    })

    # Hardware: storage 1TB+ SSD
    try:
        usage = shutil.disk_usage(Path(acq_cfg.DATA_DIR).anchor or "/")
        total_gb = round(usage.total / 1024 ** 3, 1)
        free_gb = round(usage.free / 1024 ** 3, 1)
        checks.append({
            "name": "storage",
            "status": "ok" if total_gb >= RECOMMENDED["storage_total_gb"] and free_gb >= 100
            else "warn",
            "detail": f"{free_gb} GB free of {total_gb} GB",
            "recommended": f"{RECOMMENDED['storage_total_gb']}GB+ SSD",
        })
    except OSError:
        checks.append({"name": "storage", "status": "warn",
                       "detail": "unknown", "recommended": "1TB+ SSD"})

    # Software: FFmpeg (video composition)
    from video_production import config as vcfg

    ffmpeg = shutil.which(vcfg.FFMPEG_BINARY)
    checks.append({
        "name": "ffmpeg",
        "status": "ok" if ffmpeg else "warn",
        "detail": ffmpeg or "not on PATH -- composition stage degrades gracefully",
        "recommended": "FFmpeg installed",
    })

    # Software: Urdu font (subtitles / thumbnails / visuals)
    checks.append({
        "name": "urdu_font",
        "status": "ok" if vcfg.URDU_FONT_PATH else "warn",
        "detail": vcfg.URDU_FONT_PATH or "no Urdu-capable font found",
        "recommended": "Noto Nastaliq/Naskh font (URDU_FONT_PATH)",
    })

    # Software: offline TTS models (optional; placeholder synth fallback)
    from pipeline import config as pcfg

    checks.append({
        "name": "tts_models",
        "status": "ok" if (pcfg.PIPER_MODEL_PATH or pcfg.XTTS_MODEL_PATH) else "warn",
        "detail": (pcfg.XTTS_MODEL_PATH or pcfg.PIPER_MODEL_PATH
                   or "no local TTS model -- espeak/placeholder synth used"),
        "recommended": "Piper/XTTS Urdu model for natural voice",
    })

    return checks


def summary(checks: List[Dict[str, Any]]) -> Dict[str, int]:
    out = {"ok": 0, "warn": 0, "missing": 0}
    for c in checks:
        out[c["status"]] = out.get(c["status"], 0) + 1
    return out
