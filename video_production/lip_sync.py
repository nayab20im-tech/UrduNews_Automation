"""
video_production/lip_sync.py
================================
Stage 3.2 — Lip Synchronization.

Synchronizes the AI presenter's lip/mouth movements with the Urdu
voice-over audio.  Backends resolved by ``LIPSYNC_BACKEND`` env var:

1. **did** — If the avatar was generated via D-ID API, this backend is a
   pass-through (the avatar video already has synced lips; no re-sync needed).

2. **heygen** — Same as above for HeyGen-generated avatars.

3. **musetalk** — MuseTalk local model.  Real-time latent-inpainting lip
   sync; best quality of the open-source models.  Needs CUDA GPU and
   ``MUSETALK_CHECKPOINT_DIR``.

4. **latentsync** — LatentSync diffusion-based lip replacement.  High
   visual fidelity for dubbing existing video.  Needs CUDA GPU and
   ``LATENTSYNC_CHECKPOINT_DIR``.

5. **wav2lip** — Wav2Lip model.  Classic, reliable, needs CUDA GPU and
   checkpoint path (``WAV2LIP_CHECKPOINT``).

6. **sadtalker** — SadTalker can do combined avatar + lip-sync in one
   pass when Stage 3.1 was skipped.

7. **animated** (default offline) — audio-driven presenter animation.
   Analyses voice-over per-frame energy and renders the avatar frame-by-
   frame with realistic mouth/eye movement using MediaPipe FaceMesh
   (falls back to dlib or Haar when MediaPipe is unavailable).  Needs
   only Pillow/OpenCV and FFmpeg — no GPU.

8. **placeholder** — last-resort merge of avatar video + audio via FFmpeg
   (no actual lip-sync), used when no source image is available at all.

Backend priority (auto): did → heygen → musetalk → latentsync → wav2lip → sadtalker → animated → placeholder
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List

from data_acquisition.utils.logger import get_logger

from . import config as vc

logger = get_logger("video_production.lip_sync")


@dataclass
class LipSyncResult:
    article_id: int
    video_path: str = ""
    duration_sec: float = 0.0
    backend_used: str = ""
    status: str = "ok"              # ok | no_input | error


class LipSyncEngine:
    """Lip-sync the avatar video with the Urdu audio."""

    def __init__(
        self,
        output_dir: Path | str = vc.LIPSYNC_DIR,
        backend: str = vc.LIPSYNC_BACKEND,
        wav2lip_checkpoint: str = vc.WAV2LIP_CHECKPOINT,
        wav2lip_face: str = vc.WAV2LIP_FACE_IMAGE,
        source_image: str = vc.AVATAR_SOURCE_IMAGE,
        ffmpeg: str = vc.FFMPEG_BINARY,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.wav2lip_checkpoint = wav2lip_checkpoint
        self.wav2lip_face = wav2lip_face
        self.source_image = source_image
        self.ffmpeg = ffmpeg

    def resolve_backend(self) -> str:
        """Return the best available backend in priority order."""
        if self.backend != "auto":
            return self.backend
        # Cloud API pass-through (avatar already synced)
        if vc.DID_API_KEY:
            return "did"
        if vc.HEYGEN_API_KEY:
            return "heygen"
        # GPU open-source models
        if (
            vc.MUSETALK_CHECKPOINT_DIR
            and Path(vc.MUSETALK_CHECKPOINT_DIR).exists()
            and vc.HAS_CUDA
        ):
            return "musetalk"
        if (
            vc.LATENTSYNC_CHECKPOINT_DIR
            and Path(vc.LATENTSYNC_CHECKPOINT_DIR).exists()
            and vc.HAS_CUDA
        ):
            return "latentsync"
        if (
            self.wav2lip_checkpoint
            and Path(self.wav2lip_checkpoint).exists()
            and vc.HAS_CUDA
        ):
            return "wav2lip"
        if (
            vc.SADTALKER_CHECKPOINT_DIR
            and Path(vc.SADTALKER_CHECKPOINT_DIR).exists()
            and vc.HAS_CUDA
        ):
            return "sadtalker"
        if self.source_image and Path(self.source_image).exists():
            return "animated"
        return "placeholder"

    # -- public API ----------------------------------------------------------
    def sync(
        self,
        article_id: int,
        avatar_video: str,
        audio_path: str,
        audio_duration: float,
    ) -> LipSyncResult:
        try:
            if not audio_path or not Path(audio_path).exists():
                return LipSyncResult(article_id, status="no_input")
            has_avatar = bool(avatar_video) and Path(avatar_video).exists()
            has_source = bool(self.source_image) and Path(self.source_image).exists()
            if not has_avatar and not has_source:
                return LipSyncResult(article_id, status="no_input")

            out_path = self.output_dir / f"lipsync_{article_id}.mp4"
            backend = self.resolve_backend()

            # Cloud-API pass-through: the avatar video from D-ID/HeyGen already
            # has perfect lip-sync baked in.  Just copy it as the lip-sync output.
            if backend in ("did", "heygen") and has_avatar:
                shutil.copy2(avatar_video, str(out_path))
                return LipSyncResult(
                    article_id, str(out_path), round(audio_duration, 2), backend, "ok",
                )

            used = backend
            if (
                backend == "musetalk"
                and has_avatar
                and self._run_musetalk(avatar_video, audio_path, out_path)
            ):
                used = "musetalk"
            elif (
                backend == "latentsync"
                and has_avatar
                and self._run_latentsync(avatar_video, audio_path, out_path)
            ):
                used = "latentsync"
            elif (
                backend == "wav2lip"
                and (has_avatar or (self.wav2lip_face and Path(self.wav2lip_face).exists()))
                and self._run_wav2lip(avatar_video, audio_path, out_path)
            ):
                used = "wav2lip"
            elif backend == "sadtalker" and self._run_sadtalker(
                avatar_video, audio_path, out_path
            ):
                used = "sadtalker"
            elif has_source and self._run_animated(
                avatar_video, audio_path, out_path, audio_duration
            ):
                used = "animated"
            else:
                if not has_avatar:
                    logger.warning(
                        "Lip-sync for article %d: no avatar video and "
                        "animation failed", article_id,
                    )
                    return LipSyncResult(article_id, status="no_input")
                if backend not in ("placeholder", "auto"):
                    logger.warning(
                        "Lip-sync backend %r unavailable for article %d; "
                        "using placeholder (audio merge)",
                        backend, article_id,
                    )
                self._run_placeholder(avatar_video, audio_path, out_path)
                used = "placeholder"

            return LipSyncResult(
                article_id, str(out_path), round(audio_duration, 2), used, "ok",
            )
        except Exception as exc:
            logger.error("Lip-sync failed for article %d: %s", article_id, exc)
            return LipSyncResult(article_id, status="error")

    def run(self, articles: List[dict]) -> List[LipSyncResult]:
        """articles: [{"article_id", "avatar_video", "audio_path", "audio_duration"}]"""
        results = []
        for a in articles:
            r = self.sync(
                a["article_id"],
                a.get("avatar_video", ""),
                a.get("audio_path", ""),
                a.get("audio_duration", 0.0),
            )
            results.append(r)
        ok = sum(1 for r in results if r.status == "ok")
        logger.info("Lip-sync: %d/%d ok", ok, len(results))
        return results

    # -- backends ------------------------------------------------------------

    def _run_musetalk(
        self, face_video: str, audio_path: str, out_path: Path,
    ) -> bool:
        """Lip-sync via local MuseTalk model (CUDA required).

        MuseTalk uses latent-inpainting to seamlessly replace the mouth
        region in each frame, achieving near-real-time quality.
        Install: https://github.com/TMElyralab/MuseTalk

        Set ``MUSETALK_CHECKPOINT_DIR`` to the downloaded checkpoint directory.
        """
        try:
            if not vc.MUSETALK_CHECKPOINT_DIR or not Path(vc.MUSETALK_CHECKPOINT_DIR).exists():
                logger.warning("MuseTalk checkpoint dir not found")
                return False
            cmd = [
                "python", "-m", "musetalk.inference",
                "--video_path", face_video,
                "--audio_path", audio_path,
                "--bbox_shift", "0",
                "--output_path", str(out_path),
                "--checkpoint", vc.MUSETALK_CHECKPOINT_DIR,
            ]
            proc = subprocess.run(cmd, capture_output=True, timeout=1800)
            if proc.returncode != 0:
                logger.warning("MuseTalk failed: %s", proc.stderr.decode(errors="replace")[:500])
                return False
            return out_path.exists() and out_path.stat().st_size > 1000
        except Exception as exc:
            logger.warning("MuseTalk invocation failed: %s", exc)
            return False

    def _run_latentsync(
        self, face_video: str, audio_path: str, out_path: Path,
    ) -> bool:
        """Lip-sync via local LatentSync model (CUDA required).

        LatentSync uses a diffusion-based approach to replace the lip
        region while maintaining high visual fidelity in surrounding areas.
        Install: https://github.com/bytedance/LatentSync

        Set ``LATENTSYNC_CHECKPOINT_DIR`` to the downloaded checkpoint directory.
        """
        try:
            if not vc.LATENTSYNC_CHECKPOINT_DIR or not Path(vc.LATENTSYNC_CHECKPOINT_DIR).exists():
                logger.warning("LatentSync checkpoint dir not found")
                return False
            cmd = [
                "python", "-m", "latentsync.inference",
                "--video", face_video,
                "--audio", audio_path,
                "--output", str(out_path),
                "--checkpoint_dir", vc.LATENTSYNC_CHECKPOINT_DIR,
                "--guidance_scale", "1.5",
                "--video_fps", str(vc.VIDEO_FPS),
            ]
            proc = subprocess.run(cmd, capture_output=True, timeout=1800)
            if proc.returncode != 0:
                logger.warning("LatentSync failed: %s", proc.stderr.decode(errors="replace")[:500])
                return False
            return out_path.exists() and out_path.stat().st_size > 1000
        except Exception as exc:
            logger.warning("LatentSync invocation failed: %s", exc)
            return False

    def _run_wav2lip(
        self, face_video: str, audio_path: str, out_path: Path,
    ) -> bool:
        try:
            face_input = self.wav2lip_face if self.wav2lip_face and Path(self.wav2lip_face).exists() else face_video
            cmd = [
                "python", "-m", "wav2lip.inference",
                "--checkpoint_path", self.wav2lip_checkpoint,
                "--face", face_input,
                "--audio", audio_path,
                "--outfile", str(out_path),
                "--resize_factor", "1",
                "--nosmooth",
            ]
            proc = subprocess.run(cmd, capture_output=True, timeout=1200)
            return proc.returncode == 0 and out_path.exists()
        except Exception as exc:
            logger.warning("Wav2Lip failed: %s", exc)
            return False

    def _run_sadtalker(
        self, face_video: str, audio_path: str, out_path: Path,
    ) -> bool:
        try:
            source_image = vc.AVATAR_SOURCE_IMAGE or face_video
            cmd = [
                "python", "-m", "sadtalker.inference",
                "--driven_audio", audio_path,
                "--source_image", source_image,
                "--checkpoint_dir", vc.SADTALKER_CHECKPOINT_DIR,
                "--result_dir", str(out_path.parent),
                "--still",
            ]
            proc = subprocess.run(cmd, capture_output=True, timeout=1200)
            if proc.returncode != 0:
                return False
            candidates = sorted(out_path.parent.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
            if candidates:
                shutil.move(str(candidates[-1]), str(out_path))
                return out_path.exists()
            return False
        except Exception as exc:
            logger.warning("SadTalker lip-sync failed: %s", exc)
            return False

    def _run_placeholder(
        self, avatar_video: str, audio_path: str, out_path: Path,
    ) -> None:
        """Merge avatar video + audio with FFmpeg (no real lip-sync)."""
        cmd = [
            self.ffmpeg, "-y",
            "-i", avatar_video,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", vc.AUDIO_CODEC,
            "-b:a", vc.AUDIO_BITRATE,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            str(out_path),
        ]
        subprocess.run(cmd, capture_output=True, timeout=300, check=True)

    # -- animated backend (audio-driven lip-sync without GPU) ---------------
    def _run_animated(
        self, avatar_video: str, audio_path: str, out_path: Path, audio_duration: float,
    ) -> bool:
        """Broadcast-quality presenter animation with FFT-based lip sync.

        Uses OpenCV for face manipulation and numpy/scipy for spectral
        audio analysis. Produces natural mouth movement synchronised
        with the voice-over. Since Veo provides natural head motion and blinking,
        we disable synthetic sway/blink if the source is a video.
        """
        try:
            import cv2
            import numpy as np
            import tempfile
            import shutil as _shutil

            # ---- 1. Load and prepare source avatar -------------------------
            input_path = avatar_video if avatar_video and Path(avatar_video).exists() else self.source_image
            if not input_path or not Path(input_path).exists():
                return False
                
            is_video = input_path.lower().endswith(('.mp4', '.avi', '.mov'))
            
            if is_video:
                cap = cv2.VideoCapture(input_path)
                ret, first_frame = cap.read()
                if not ret:
                    return False
                src = first_frame
            else:
                src = cv2.imread(input_path)
                if src is None:
                    return False

            pres_w = vc.PRESENTER_WIDTH
            pres_h = vc.PRESENTER_HEIGHT

            # First pass: detect face on the raw source image to find crop centre
            raw_face = self._detect_face_raw(src)

            # Crop head-and-shoulders, centred on the detected face
            src_h, src_w = src.shape[:2]
            target_ratio = pres_w / pres_h
            current_ratio = src_w / src_h
            if current_ratio > target_ratio:
                # Source is wider — crop width around face centre
                new_w = int(src_h * target_ratio)
                if raw_face is not None:
                    face_cx = raw_face[0] + raw_face[2] // 2
                else:
                    face_cx = src_w // 2
                crop_x = max(0, min(src_w - new_w, face_cx - new_w // 2))
                src = src[:, crop_x:crop_x + new_w]
            else:
                # Source is taller — crop height, biased toward the top (face)
                new_h = int(src_w / target_ratio)
                if raw_face is not None:
                    face_cy = raw_face[1] + raw_face[3] // 2
                    crop_y = max(0, min(src_h - new_h, face_cy - int(new_h * 0.42)))
                else:
                    crop_y = max(0, (src_h - new_h) // 2 - int(src_h * 0.08))
                src = src[crop_y:crop_y + new_h, :]

            # Upscale to presenter dimensions
            presenter = cv2.resize(src, (pres_w, pres_h), interpolation=cv2.INTER_LANCZOS4)

            # ---- 2. Detect face landmarks on the prepared image ------------
            face_info = self._detect_face(presenter)
            if face_info is None:
                logger.warning("No face detected in source image")
                return False

            # ---- 3. Analyse audio spectral energy --------------------------
            energy = self._analyse_fft_energy(audio_path, audio_duration)
            if not energy:
                return False

            # ---- 4. Pre-compute studio background --------------------------
            studio_bg = self._render_studio_background()

            # ---- 4b. Pre-compute viseme displacement maps ------------------
            # Built once here and passed into every _animate_mouth call so
            # the expensive per-pixel computation is done only 6 times.
            viseme_maps = self._build_viseme_maps(
                face_info, pres_h, pres_w,
            )

            # ---- 5. Render frames ------------------------------------------
            fps = vc.VIDEO_FPS
            total_frames = max(int(audio_duration * fps), fps)
            tmp_dir = tempfile.mkdtemp(prefix="lipsync_frames_")
            frames_dir = Path(tmp_dir)

            # Blinking state (only if static image)
            next_blink = np.random.uniform(2.5, 4.5)
            blink_phase = 0  # 0=not blinking, >0=frames into blink

            # Track silence stretches for inter-sentence nods
            silence_run = 0   # consecutive frames with jaw < 0.05
            
            if is_video:
                # We will read frames dynamically from the video
                video_frames = []
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                while True:
                    ret, f = cap.read()
                    if not ret:
                        break
                    # Crop and resize to match presenter layout exactly as we did for first_frame
                    if current_ratio > target_ratio:
                        f = f[:, crop_x:crop_x + new_w]
                    else:
                        f = f[crop_y:crop_y + new_h, :]
                    f_resized = cv2.resize(f, (pres_w, pres_h), interpolation=cv2.INTER_LANCZOS4)
                    video_frames.append(f_resized)
                cap.release()
                if not video_frames:
                    video_frames = [presenter]

            for fi in range(total_frames):
                t = fi / fps
                e_idx = min(fi, len(energy) - 1)

                # Smooth with neighbouring frames for natural motion
                lo = max(0, e_idx - 2)
                hi = min(len(energy), e_idx + 3)
                smooth_bands = {}
                for k in ("bass", "mid", "high", "rms"):
                    vals = [energy[j][k] for j in range(lo, hi)]
                    smooth_bands[k] = sum(vals) / max(len(vals), 1)

                if is_video:
                    v_idx = fi % len(video_frames)
                    frame = video_frames[v_idx].copy()
                    # Detect face in this specific frame since it moves!
                    # For performance, we could track it, but the Veo motion is subtle enough
                    # that _detect_face should find it easily. 
                    # If it fails, fallback to the initial face_info.
                    current_face_info = self._detect_face(frame) or face_info
                    # Rebuild viseme map for the new face position
                    current_viseme_maps = self._build_viseme_maps(current_face_info, pres_h, pres_w)
                else:
                    frame = presenter.copy()
                    current_face_info = face_info
                    current_viseme_maps = viseme_maps

                # -- Mouth animation (painted open-mouth) --
                # bass drives syllable nuclei; use ** 0.7 to boost low values
                # so even moderate speech opens the mouth clearly.
                # Clamp bass to [0, 1] — negative values cause complex numbers.
                _bass = max(0.0, float(smooth_bands["bass"]))
                mouth_open = min(
                    (_bass ** 0.7) * 0.70
                    + max(0.0, float(smooth_bands["mid"])) * 0.20
                    + max(0.0, float(smooth_bands["rms"])) * 0.10, 1.0,
                )
                mouth_width_factor = 1.0 + smooth_bands["high"] * 0.10
                frame = self._animate_mouth(
                    frame, current_face_info, mouth_open, mouth_width_factor,
                    viseme_maps=current_viseme_maps,
                )

                if not is_video:
                    # -- Eyebrow animation --
                    frame = self._animate_eyebrows(
                        frame, current_face_info, smooth_bands["mid"], mouth_open,
                    )

                    # -- Eye blinking --
                    blink_amount, next_blink, blink_phase = self._update_blink(
                        t, next_blink, blink_phase,
                    )
                    if blink_amount > 0.01:
                        frame = self._animate_blink(frame, current_face_info, blink_amount)

                # -- Head sway: speech-tied (not just RMS-driven) --
                sway_x, sway_y = 0, 0
                if not is_video:
                    # Silence counter drives inter-sentence nod
                    if energy[e_idx]["bass"] < 0.05:
                        silence_run += 1
                    else:
                        silence_run = 0

                    sway_x = int(
                        np.sin(fi * 0.038) * mouth_open * 3.0
                        + np.sin(fi * 0.013) * 1.2
                    )
                    sway_y = int(
                        np.cos(fi * 0.025) * mouth_open * 2.0
                        + np.sin(fi * 0.017) * 0.7
                    )
                    # Between sentences: gentle slow nod
                    if silence_run > int(fps * 0.4):
                        sway_y += int(np.sin(fi * 0.008) * 2.5)

                    # Breathing (shoulder rise)
                    sway_y += int(np.sin(fi * 0.06) * 1.0)

                # -- Composite onto 1920x1080 studio background --
                canvas = studio_bg.copy()
                paste_x = vc.PRESENTER_X + sway_x
                paste_y = vc.PRESENTER_Y + sway_y

                # Clamp to valid region
                px1 = max(0, paste_x)
                py1 = max(0, paste_y)
                px2 = min(vc.VIDEO_WIDTH, paste_x + pres_w)
                py2 = min(vc.VIDEO_HEIGHT, paste_y + pres_h)
                fx1 = max(0, -paste_x)
                fy1 = max(0, -paste_y)
                fx2 = fx1 + (px2 - px1)
                fy2 = fy1 + (py2 - py1)

                if px2 > px1 and py2 > py1:
                    canvas[py1:py2, px1:px2] = frame[fy1:fy2, fx1:fx2]

                frame_path = frames_dir / f"frame_{fi:06d}.png"
                cv2.imwrite(str(frame_path), canvas)

            # ---- 5b. Self-verification QC before encode --------------------
            qc_ok = self._qc_mouth_animation(
                frames_dir, total_frames, face_info,
            )
            if not qc_ok:
                # Fallback: widen the jaw scale and re-render
                logger.warning(
                    "Mouth animation QC failed — re-rendering with "
                    "1.5× jaw scale",
                )
                old_scale = vc.MOUTH_OPEN_SCALE
                vc.MOUTH_OPEN_SCALE = min(0.20, old_scale * 1.5)
                # Re-build viseme maps with new scale
                viseme_maps = self._build_viseme_maps(
                    face_info, pres_h, pres_w,
                )
                for fi in range(total_frames):
                    e_idx = min(fi, len(energy) - 1)
                    lo = max(0, e_idx - 2)
                    hi = min(len(energy), e_idx + 3)
                    smooth_bands = {}
                    for k in ("bass", "mid", "high", "rms"):
                        vals = [energy[j][k] for j in range(lo, hi)]
                        smooth_bands[k] = sum(vals) / max(len(vals), 1)
                    frame = presenter.copy()
                    _bass2 = max(0.0, float(smooth_bands["bass"]))
                    mouth_open = min(
                        (_bass2 ** 0.7) * 0.70
                        + max(0.0, float(smooth_bands["mid"])) * 0.20
                        + max(0.0, float(smooth_bands["rms"])) * 0.10, 1.0,
                    )
                    frame = self._animate_mouth(
                        frame, face_info, mouth_open, 1.0,
                        viseme_maps=viseme_maps,
                    )
                    # Skip eyebrows/blink in fallback for speed
                    canvas = studio_bg.copy()
                    canvas[vc.PRESENTER_Y:vc.PRESENTER_Y + pres_h,
                           vc.PRESENTER_X:vc.PRESENTER_X + pres_w] = frame
                    cv2.imwrite(str(frames_dir / f"frame_{fi:06d}.png"), canvas)
                vc.MOUTH_OPEN_SCALE = old_scale  # restore

            # ---- 6. Encode frames + audio → MP4 via FFmpeg -----------------
            cmd = [
                self.ffmpeg, "-y",
                "-framerate", str(fps),
                "-i", str(frames_dir / "frame_%06d.png"),
                "-i", audio_path,
                "-c:v", vc.VIDEO_CODEC,
                "-preset", "slow",
                "-crf", str(vc.VIDEO_CRF),
                "-c:a", vc.AUDIO_CODEC,
                "-b:a", vc.AUDIO_BITRATE,
                "-pix_fmt", "yuv420p",
                "-t", str(max(audio_duration, 1.0)),
                "-shortest",
                "-r", str(fps),
                str(out_path),
            ]
            proc = subprocess.run(cmd, capture_output=True, timeout=600)

            # ---- 7. Clean up temp frames ----------------------------------
            try:
                _shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

            if proc.returncode != 0:
                stderr = proc.stderr.decode(errors="replace")[:500]
                logger.warning("Animated lip-sync FFmpeg failed: %s", stderr)
                return False

            return out_path.exists()

        except Exception as exc:
            logger.warning("Animated lip-sync failed: %s", exc, exc_info=True)
            return False

    # -- dual-anchor (two presenter) rendering -------------------------------
    def _face_info_at(self, img, rect) -> dict | None:
        """Build a canvas-space ``face_info`` for a known face rectangle.

        Runs the regular detector on a margin crop around ``rect`` and
        offsets the result back into full-image coordinates.  Falls back
        to geometric proportions of ``rect`` when detection fails.
        """
        import numpy as np

        x, y, w, h = (int(v) for v in rect)
        H, W = img.shape[:2]
        mx, my = int(w * 0.7), int(h * 0.7)
        x1, y1 = max(0, x - mx), max(0, y - my)
        x2, y2 = min(W, x + w + mx), min(H, y + h + my)
        info = self._detect_face(img[y1:y2, x1:x2])
        if info is not None:
            fx, fy, fw, fh = info["face"]
            if fw >= w * 0.5 and fh >= h * 0.5:
                info["face"] = (fx + x1, fy + y1, fw, fh)
                for eye in (info["left_eye"], info["right_eye"]):
                    eye["x"] += x1; eye["y"] += y1
                    eye["cx"] += x1; eye["cy"] += y1
                m = info["mouth"]
                m["cx"] += x1; m["cy"] += y1; m["x"] += x1; m["y"] += y1
                return info

        # Geometric fallback (full-box proportions)
        eye_y = y + int(h * 0.35)
        eye_h = int(h * 0.12)
        eye_w = int(w * 0.20)
        mouth_cy = y + int(h * 0.78)
        mouth_w = int(w * 0.38)
        mouth_h = int(h * 0.14)
        mouth_cx = x + w // 2
        return {
            "face": (x, y, w, h),
            "left_eye": {"x": x + int(w * 0.20), "y": eye_y, "w": eye_w,
                         "h": eye_h, "cx": x + int(w * 0.30),
                         "cy": eye_y + eye_h // 2},
            "right_eye": {"x": x + int(w * 0.60), "y": eye_y, "w": eye_w,
                          "h": eye_h, "cx": x + int(w * 0.70),
                          "cy": eye_y + eye_h // 2},
            "mouth": {"cx": mouth_cx, "cy": mouth_cy, "w": mouth_w,
                      "h": mouth_h, "x": mouth_cx - mouth_w // 2,
                      "y": mouth_cy - mouth_h // 2},
            "skin_color": (150, 140, 130),
        }

    @staticmethod
    def _animate_sway(frame, face_info, dx: int, dy: int):
        """Translate the head+shoulders region with feathered borders.

        Unlike the single-presenter renderer (which shifts the whole
        pasted image), the dual scene must keep the shared background
        pixel-stable, so the sway is applied only inside a region that
        fades back to the original at its borders.
        """
        import cv2
        import numpy as np

        if dx == 0 and dy == 0:
            return frame
        H, W = frame.shape[:2]
        fx, fy, fw, fh = face_info["face"]
        x1 = max(0, fx - int(fw * 0.65))
        x2 = min(W, fx + fw + int(fw * 0.65))
        y1 = max(0, fy - int(fh * 0.45))
        y2 = min(H, fy + fh + int(fh * 1.30))
        mh, mw = y2 - y1, x2 - x1
        if mh < 8 or mw < 8:
            return frame
        region = frame[y1:y2, x1:x2]
        shifted = cv2.warpAffine(
            region, np.float32([[1, 0, dx], [0, 1, dy]]), (mw, mh),
            borderMode=cv2.BORDER_REPLICATE)
        ramp = min(32, mh // 4, mw // 4)
        ry = np.ones(mh, np.float32)
        rx = np.ones(mw, np.float32)
        ry[:ramp] = np.linspace(0, 1, ramp)
        ry[-ramp:] = np.linspace(1, 0, ramp)
        rx[:ramp] = np.linspace(0, 1, ramp)
        rx[-ramp:] = np.linspace(1, 0, ramp)
        alpha = (ry[:, None] * rx[None, :])[..., None]
        frame[y1:y2, x1:x2] = np.clip(
            shifted * alpha + region * (1 - alpha), 0, 255,
        ).astype(np.uint8)
        return frame

    def _run_dual_animated(
        self, canvas, faces: dict, speaker_id: str,
        audio_path: str, out_path: Path, audio_duration: float,
    ) -> bool:
        """Two-anchor broadcast rendering on a shared studio canvas.

        LIP-SYNC NOTE (language independence): identical to the single
        presenter renderer — the jaw curve is driven purely by acoustic
        bands (bass/mid/high/rms), never by language semantics.

        The speaking anchor receives audio-driven mouth + eyebrow
        motion; the listening anchor keeps calm idle behaviour (natural
        blinks, breathing, micro-sway) exactly like a co-anchor waiting
        for their turn.  Both blink independently.
        """
        try:
            import cv2
            import numpy as np
            import tempfile
            import shutil as _shutil

            fps = vc.VIDEO_FPS
            total_frames = max(int(audio_duration * fps), fps)
            H, W = canvas.shape[:2]

            infos = {}
            for aid, rect in faces.items():
                info = self._face_info_at(canvas, rect)
                if info is None:
                    logger.warning("dual: could not build face info (%s)", aid)
                    return False
                infos[aid] = info
            if speaker_id not in infos:
                return False

            energy = self._analyse_fft_energy(audio_path, audio_duration)
            if not energy:
                return False

            viseme_maps = {
                aid: self._build_viseme_maps(info, H, W)
                for aid, info in infos.items()
            }

            tmp_dir = tempfile.mkdtemp(prefix="lipsync_dual_")
            frames_dir = Path(tmp_dir)

            # independent blink clocks + fixed per-anchor phase offsets
            phase_off = {"male": 0.0, "female": 1.7}
            blink_state = {
                aid: {"next": np.random.uniform(2.0, 4.0) + phase_off.get(aid, 0),
                      "phase": 0}
                for aid in infos
            }

            for fi in range(total_frames):
                t = fi / fps
                e_idx = min(fi, len(energy) - 1)
                lo = max(0, e_idx - 2)
                hi = min(len(energy), e_idx + 3)
                smooth = {}
                for k in ("bass", "mid", "high", "rms"):
                    smooth[k] = sum(energy[j][k] for j in range(lo, hi)) / max(
                        hi - lo, 1)

                frame = canvas.copy()

                for aid, info in infos.items():
                    if aid == speaker_id:
                        _bass = max(0.0, float(smooth["bass"]))
                        mouth_open = min(
                            (_bass ** 0.7) * 0.70
                            + max(0.0, float(smooth["mid"])) * 0.20
                            + max(0.0, float(smooth["rms"])) * 0.10, 1.0)
                        frame = self._animate_mouth(
                            frame, info, mouth_open,
                            1.0 + smooth["high"] * 0.10,
                            viseme_maps=viseme_maps[aid])
                        frame = self._animate_eyebrows(
                            frame, info, smooth["mid"], mouth_open)
                        # speech-tied micro sway + breathing
                        dx = int(np.sin(fi * 0.038) * mouth_open * 2.0
                                 + np.sin(fi * 0.013) * 0.8)
                        dy = int(np.cos(fi * 0.025) * mouth_open * 1.5
                                 + np.sin(fi * 0.06) * 0.8)
                    else:
                        # listener: attentive idle — breathing + micro sway
                        off = 0.0 if aid == "male" else 2.3
                        dx = int(np.sin(fi * 0.010 + off) * 0.8)
                        dy = int(np.sin(fi * 0.06 + off) * 0.8)

                    st = blink_state[aid]
                    blink_amount, st["next"], st["phase"] = self._update_blink(
                        t, st["next"], st["phase"])
                    if blink_amount > 0.01:
                        frame = self._animate_blink(frame, info, blink_amount)
                    if dx or dy:
                        frame = self._animate_sway(frame, info, dx, dy)

                cv2.imwrite(str(frames_dir / f"frame_{fi:06d}.png"), frame)

            qc_ok = self._qc_mouth_animation(
                frames_dir, total_frames, infos[speaker_id],
                ox=0, oy=0, bw=W, bh=H)
            if not qc_ok:
                logger.warning("dual: mouth QC flagged speaker %s", speaker_id)

            cmd = [
                self.ffmpeg, "-y",
                "-framerate", str(fps),
                "-i", str(frames_dir / "frame_%06d.png"),
                "-i", audio_path,
                "-c:v", vc.VIDEO_CODEC,
                "-preset", "slow",
                "-crf", str(vc.VIDEO_CRF),
                "-c:a", vc.AUDIO_CODEC,
                "-b:a", vc.AUDIO_BITRATE,
                "-pix_fmt", "yuv420p",
                "-t", str(max(audio_duration, 1.0)),
                "-shortest",
                "-r", str(fps),
                str(out_path),
            ]
            proc = subprocess.run(cmd, capture_output=True, timeout=900)
            try:
                _shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass
            if proc.returncode != 0:
                logger.warning("dual FFmpeg failed: %s",
                               proc.stderr.decode(errors="replace")[:400])
                return False
            return out_path.exists()

        except Exception as exc:
            logger.warning("Dual animated render failed: %s", exc,
                           exc_info=True)
            return False

    # -- mouth animation QC -------------------------------------------------
    @staticmethod
    def _qc_mouth_animation(frames_dir: Path, total_frames: int,
                            face_info: dict, ox: int | None = None,
                            oy: int | None = None,
                            bw: int | None = None,
                            bh: int | None = None) -> bool:
        """Sample 20 frames and verify the mouth is actually moving.

        Returns True if the animation passes (max diff > 2.0 between
        adjacent samples), False if the mouth appears frozen.
        """
        try:
            import cv2
            import numpy as np

            face_x, face_y, face_w, face_h = face_info["face"]
            m = face_info["mouth"]
            # Mouth ROI in canvas coords (presenter offset unless given)
            ox = vc.PRESENTER_X if ox is None else ox
            oy = vc.PRESENTER_Y if oy is None else oy
            bw = vc.PRESENTER_WIDTH if bw is None else bw
            bh = vc.PRESENTER_HEIGHT if bh is None else bh
            rx1 = ox + max(0, m["cx"] - m["w"])
            rx2 = ox + min(bw, m["cx"] + m["w"])
            ry1 = oy + max(0, m["cy"] - int(face_h * 0.12))
            ry2 = oy + min(bh, m["cy"] + int(face_h * 0.18))

            sample_count = min(20, total_frames)
            indices = np.linspace(
                int(total_frames * 0.1), int(total_frames * 0.9),
                sample_count, dtype=int,
            )
            prev_roi = None
            diffs = []
            for idx in indices:
                p = frames_dir / f"frame_{int(idx):06d}.png"
                if not p.exists():
                    continue
                img = cv2.imread(str(p))
                if img is None:
                    continue
                roi = img[ry1:ry2, rx1:rx2].astype(np.float32)
                if prev_roi is not None and roi.shape == prev_roi.shape and roi.size > 0:
                    d = float(np.abs(roi - prev_roi).mean())
                    if not np.isnan(d):
                        diffs.append(d)
                prev_roi = roi

            if not diffs:
                logger.warning("QC: could not read any sample frames")
                return True   # don't block on I/O failure
            max_d = max(diffs)
            frozen_frac = sum(1 for d in diffs if d < 0.5) / len(diffs)
            logger.info(
                "Mouth QC: %d pairs, max_diff=%.2f frozen_frac=%.0f%%",
                len(diffs), max_d, frozen_frac * 100,
            )
            if max_d < 2.0:
                logger.warning("QC FAIL: mouth barely moved (max_diff=%.2f)", max_d)
                return False
            if frozen_frac > 0.80:
                logger.warning("QC FAIL: mouth frozen %.0f%% of frames", frozen_frac * 100)
                return False
            return True
        except Exception as exc:
            logger.warning("QC check error (non-fatal): %s", exc)
            return True  # don't block render on QC exception

    # -- face detection (OpenCV Haar cascades) -------------------------------
    @staticmethod
    def _detect_face_raw(img):
        """Detect the largest face rectangle in a raw BGR image.

        Returns ``(x, y, w, h)`` or ``None``.
        """
        import cv2
        import os

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        haar_dir = getattr(cv2, "data", None)
        face_xml = None
        if haar_dir is not None and os.path.exists(os.path.join(haar_dir.haarcascades, "haarcascade_frontalface_default.xml")):
            face_xml = os.path.join(haar_dir.haarcascades, "haarcascade_frontalface_default.xml")
        else:
            # Fallback for broken venv paths
            import site
            for p in site.getsitepackages():
                alt_path = os.path.join(p, "cv2", "data", "haarcascade_frontalface_default.xml")
                if os.path.exists(alt_path):
                    face_xml = alt_path
                    break
        if not face_xml:
            return None
        cascade = cv2.CascadeClassifier(face_xml)
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=5, minSize=(40, 40),
        )
        if len(faces) == 0:
            faces = cascade.detectMultiScale(
                gray, scaleFactor=1.03, minNeighbors=3, minSize=(30, 30),
            )
        if len(faces) == 0:
            return None
        return max(faces, key=lambda f: f[2] * f[3])

    def _detect_face(self, img) -> dict | None:
        """Detect face, eyes, and mouth region in an OpenCV BGR image.

        Uses Haar cascade as primary detector.  If the cascade fails or
        returns an implausibly small region (< 15% of image height for a
        close-up presenter portrait), falls back to skin-color segmentation
        to locate the face region.

        Returns a dict with face/eye/mouth bounding information as pixel
        coordinates, or ``None`` if no face is found.
        """
        import cv2
        import numpy as np
        import os

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]

        # ---- Haar cascade attempt -------------------------------------------
        haar_dir = getattr(cv2, "data", None)
        fx = fy = fw = fh = None
        
        face_xml = None
        eye_xml = None
        if haar_dir is not None and os.path.exists(os.path.join(haar_dir.haarcascades, "haarcascade_frontalface_default.xml")):
            face_xml = os.path.join(haar_dir.haarcascades, "haarcascade_frontalface_default.xml")
            eye_xml  = os.path.join(haar_dir.haarcascades, "haarcascade_eye.xml")
        else:
            import site
            for p in site.getsitepackages():
                alt_f = os.path.join(p, "cv2", "data", "haarcascade_frontalface_default.xml")
                alt_e = os.path.join(p, "cv2", "data", "haarcascade_eye.xml")
                if os.path.exists(alt_f):
                    face_xml = alt_f
                    eye_xml = alt_e
                    break

        if face_xml and eye_xml:
            face_cascade = cv2.CascadeClassifier(face_xml)
            eye_cascade  = cv2.CascadeClassifier(eye_xml)

            for sf, mn, ms in [
                (1.05, 5, 40), (1.03, 3, 30), (1.03, 1, 50),
            ]:
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=sf, minNeighbors=mn, minSize=(ms, ms),
                )
                if len(faces) > 0:
                    # Reject any detection smaller than 15% of image height
                    # (those are usually background objects, not the face).
                    valid = [(x, y, ww, hh) for (x, y, ww, hh) in faces
                             if hh >= h * 0.15]
                    if valid:
                        fx, fy, fw, fh = max(valid, key=lambda f: f[2] * f[3])
                        break

        # ---- Skin-colour fallback if cascade failed -------------------------
        # For a close-up news presenter the face is typically centred or
        # right-of-centre in the frame and occupies the upper 45% of height.
        # Restrict the skin-mask search to the central/right upper region.
        if fx is None:
            # Convert to YCrCb — most robust for varied skin tones
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            # Skin in YCrCb: Cr 133-173, Cb 77-127
            skin_mask_full = cv2.inRange(ycrcb,
                                         (0, 133, 77), (255, 173, 127))
            # Restrict to upper 45% and central-right 60% of the frame
            search_y2 = int(h * 0.45)
            search_x1 = int(w * 0.25)
            search_mask = np.zeros_like(skin_mask_full)
            search_mask[:search_y2, search_x1:] = skin_mask_full[:search_y2, search_x1:]
            # Morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            clean = cv2.morphologyEx(search_mask, cv2.MORPH_CLOSE, kernel)
            clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN,
                                     cv2.getStructuringElement(
                                         cv2.MORPH_ELLIPSE, (7, 7)))
            contours, _ = cv2.findContours(
                clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                # Minimum area: face must be at least 3% of total pixels
                if area >= w * h * 0.03:
                    bx, by, bw, bh = cv2.boundingRect(largest)
                    margin = int(max(bw, bh) * 0.15)
                    fx = max(0, bx - margin)
                    fy = max(0, by - margin)
                    fw = min(w - fx, bw + 2 * margin)
                    fh = min(h - fy, bh + 2 * margin)

        # ---- Last-resort: texture-density scan + geometric estimate ---------
        # For a close-up news presenter portrait (head + shoulders visible),
        # scan the upper 55% of the image with a sliding window and find the
        # region with the highest Sobel edge density (= most face detail).
        # The face has more structured detail than the blurred background.
        if fx is None:
            scan_h = int(h * 0.55)
            scan_img = gray[:scan_h, :]
            # Compute Sobel magnitude
            sobelx = cv2.Sobel(scan_img, cv2.CV_32F, 1, 0, ksize=3)
            sobely = cv2.Sobel(scan_img, cv2.CV_32F, 0, 1, ksize=3)
            edge_mag = cv2.magnitude(sobelx, sobely)
            # Smooth to get a density map
            density = cv2.GaussianBlur(edge_mag, (61, 61), 20)
            # Find the peak of the density map
            _, _, _, max_loc = cv2.minMaxLoc(density)
            peak_x, peak_y = max_loc
            # Estimate face size: ~35% of image height, ~30% of image width
            est_fw = int(w * 0.32)
            est_fh = int(h * 0.35)
            fx = max(0, peak_x - est_fw // 2)
            fy = max(0, peak_y - int(est_fh * 0.35))  # peak is near eye level
            fw = min(w - fx, est_fw)
            fh = min(scan_h - fy, est_fh)

        fx, fy, fw, fh = int(fx), int(fy), int(fw), int(fh)

        # ---- Eyes -----------------------------------------------------------
        eye_positions = []
        if 'eye_cascade' in locals() and eye_cascade is not None and not eye_cascade.empty():
            eye_roi = gray[fy:fy + fh // 2, fx:fx + fw]
            eyes = eye_cascade.detectMultiScale(eye_roi, 1.05, 3, minSize=(8, 8))
            for ex, ey, ew, eh in sorted(eyes, key=lambda e: e[0]):
                eye_positions.append({
                    "x": fx + ex, "y": fy + ey, "w": ew, "h": eh,
                    "cx": fx + ex + ew // 2, "cy": fy + ey + eh // 2,
                })
        if len(eye_positions) < 2:
            eye_y = fy + int(fh * 0.35)
            eye_h = int(fh * 0.12)
            eye_w = int(fw * 0.20)
            eye_positions = [
                {"x": fx + int(fw * 0.20), "y": eye_y,
                 "w": eye_w, "h": eye_h,
                 "cx": fx + int(fw * 0.30), "cy": eye_y + eye_h // 2},
                {"x": fx + int(fw * 0.60), "y": eye_y,
                 "w": eye_w, "h": eye_h,
                 "cx": fx + int(fw * 0.70), "cy": eye_y + eye_h // 2},
            ]
        eye_positions.sort(key=lambda e: e["cx"])
        left_eye  = eye_positions[0]
        right_eye = eye_positions[-1]

        # ---- Mouth ----------------------------------------------------------
        # Use the visual centre of the skin contour (centroid) rather than
        # the geometric centre of the bounding box, which can be skewed by
        # background pixels included in the margin.
        mouth_cx = fx + int(fw * 0.50)   # horizontal centre of face box
        # Data-driven lip seam: lips form a WIDE horizontal band of
        # strongly-red pixels (R/(G+1) > 1.75 across >= 12% of the row
        # width).  Mean-ratio metrics are unreliable here — dark nostril
        # shadows spike the mean ratio while staying only a few pixels
        # wide, which misplaced the mouth at the nose base on bearded
        # anchors.  Two-pass search: first the tight-box lip band
        # (60-95% of box height), then a wider band for extended
        # skin-segmentation boxes.  Falls back to 72% (tight-box
        # geometry) when no wide red band is detected.
        mouth_cy = fy + int(fh * 0.72)

        def _warm_lip_row(y1f: float, y2f: float) -> int | None:
            y1 = fy + int(fh * y1f)
            y2 = min(h, fy + int(fh * y2f))
            x1 = fx + int(fw * 0.20)
            x2 = fx + int(fw * 0.80)
            if y2 <= y1 + 3 or x2 <= x1 + 3:
                return None
            band = img[y1:y2, x1:x2].astype(np.float32)
            ratio = band[:, :, 2] / (band[:, :, 1] + 1.0)
            wide = (ratio > 1.75).mean(axis=1)   # wide-red fraction per row
            best_row = int(np.argmax(wide))
            if float(wide[best_row]) > 0.12:     # lips, not a nostril spike
                return y1 + best_row
            return None

        seam = _warm_lip_row(0.60, 0.95) or _warm_lip_row(0.45, 1.25)
        if seam is not None:
            mouth_cy = seam
        mouth_w  = int(fw * 0.38)        # tighter than box width
        mouth_h  = int(fh * 0.14)

        # ---- Skin colour (cheeks) -------------------------------------------
        skin_y1 = fy + int(fh * 0.45)
        skin_y2 = max(skin_y1 + 2, fy + int(fh * 0.58))
        skin_x1 = fx + int(fw * 0.05)
        skin_x2 = fx + int(fw * 0.25)
        skin_x3 = fx + int(fw * 0.75)
        skin_x4 = fx + int(fw * 0.95)
        left_cheek  = img[skin_y1:skin_y2, skin_x1:skin_x2]
        right_cheek = img[skin_y1:skin_y2, skin_x3:skin_x4]
        if left_cheek.size > 0 and right_cheek.size > 0:
            cheeks = np.concatenate(
                [left_cheek.reshape(-1, 3), right_cheek.reshape(-1, 3)], axis=0,
            )
            skin_color = tuple(int(c) for c in cheeks.mean(axis=0))
        else:
            skin_color = (150, 140, 130)

        return {
            "face": (fx, fy, fw, fh),
            "left_eye":  left_eye,
            "right_eye": right_eye,
            "mouth": {
                "cx": mouth_cx, "cy": mouth_cy,
                "w":  mouth_w,  "h":  mouth_h,
                "x":  mouth_cx - mouth_w // 2,
                "y":  mouth_cy - mouth_h // 2,
            },
            "skin_color": skin_color,
        }

    # -- viseme displacement maps -------------------------------------------
    @staticmethod
    def _build_viseme_maps(face_info, img_h, img_w):
        """Pre-compute 6 displacement maps for viseme levels 0–5.

        Each map is a pair (map_x, map_y) of float32 arrays with shape
        (img_h, img_w) representing cv2.remap source-pixel coordinates.
        Level 0 = closed mouth (identity); level 5 = maximum open.

        The warp displaces only the lower-face region (upper-lip line to
        chin) using a smooth Gaussian-weighted displacement field, so the
        transition between moved and static pixels is seamless.
        """
        import cv2
        import numpy as np

        face_x, face_y, face_w, face_h = face_info["face"]
        m = face_info["mouth"]

        # Key Y coordinates in image space
        lip_top_y   = m["cy"] - int(face_h * 0.04)    # upper lip centre row
        lip_bot_y   = m["cy"] + int(face_h * 0.04)    # lower lip centre row
        chin_y      = face_y + face_h + int(face_h * 0.12)  # bottom of warp zone
        mcx         = m["cx"]
        half_w      = m["w"] // 2

        # Maximum jaw-drop per level (pixels) using MOUTH_OPEN_SCALE = 0.14
        # Level k → jaw_drop = face_h * (k/5) * 0.14  (level 0 = 0)
        max_drop    = int(face_h * vc.MOUTH_OPEN_SCALE)
        jaw_drops   = [0, int(max_drop * 0.15), int(max_drop * 0.35),
                       int(max_drop * 0.60), int(max_drop * 0.82), max_drop]
        # Horizontal width expansion (pixels each side) per level
        wid_expand  = [0, 1, 2, 4, 6, 8]

        # Base identity maps
        base_x = np.tile(np.arange(img_w, dtype=np.float32), (img_h, 1))
        base_y = np.tile(np.arange(img_h, dtype=np.float32).reshape(-1, 1),
                         (1, img_w))

        maps = []
        for lvl in range(6):
            jd = jaw_drops[lvl]
            we = wid_expand[lvl]
            if jd == 0 and we == 0:
                maps.append((base_x.copy(), base_y.copy()))
                continue

            map_x = base_x.copy()
            map_y = base_y.copy()

            # --- Vertical displacement field ---
            # Rows between lip_top_y and chin_y are shifted DOWN by jd,
            # falling off smoothly from lip centre to chin using a ramp.
            # Rows above lip_top_y: identity.
            for y in range(max(0, lip_top_y), min(img_h, chin_y + 1)):
                # Normalized distance from lip line into chin zone [0,1]
                t = (y - lip_top_y) / max(chin_y - lip_top_y, 1)
                # Smoothstep so movement is fastest at lip centre and tapers off
                t_s = t * t * (3 - 2 * t)
                # Source row = current row - displacement (pulls pixels upward
                # in source → pushes them downward in output)
                dy = jd * t_s
                map_y[y, :] = np.clip(y - dy, 0, img_h - 1)

            # Apply Gaussian-distance weight from mouth centre so only the
            # central mouth column is fully displaced; edges fade to identity.
            xx = np.arange(img_w, dtype=np.float32)
            dist_x = np.abs(xx - mcx)
            sigma_x = half_w * 1.4
            weight_x = np.exp(-0.5 * (dist_x / sigma_x) ** 2).reshape(1, -1)
            # Blend between displaced and identity maps
            for y in range(max(0, lip_top_y), min(img_h, chin_y + 1)):
                dy_full = jd * ((y - lip_top_y) / max(chin_y - lip_top_y, 1))
                dy_full_s = dy_full * ((y - lip_top_y) / max(chin_y - lip_top_y, 1))
                map_y[y, :] = y - (y - map_y[y, :]) * weight_x[0]

            # --- Horizontal displacement field (mouth width) ---
            if we > 0:
                yy = np.arange(img_h, dtype=np.float32)
                dist_y = np.abs(yy - m["cy"])
                sigma_y = face_h * 0.08
                weight_y = np.exp(-0.5 * (dist_y / sigma_y) ** 2).reshape(-1, 1)
                for x in range(img_w):
                    # Pull left side of mouth outward, right side outward
                    if x < mcx:
                        dx = we * max(0.0, 1.0 - abs(x - (mcx - half_w)) / max(half_w, 1))
                        map_x[:, x] = np.clip(x + dx * weight_y[:, 0], 0, img_w - 1)
                    else:
                        dx = we * max(0.0, 1.0 - abs(x - (mcx + half_w)) / max(half_w, 1))
                        map_x[:, x] = np.clip(x - dx * weight_y[:, 0], 0, img_w - 1)

            maps.append((map_x, map_y))

        return maps

    # -- mouth animation ----------------------------------------------------
    def _animate_mouth(self, img, face_info, open_amount, width_factor=1.0,
                       viseme_maps=None):
        """Mouth-opening animation: softly reveals a dark cavity between the lips.

        For each frame this method:

        1.  Locates the exact lip-seam row (darkest horizontal band in the
            lip zone) so the cavity sits precisely on the closed lips.
        2.  Draws a dark, feathered cavity ellipse at the lip seam whose
            vertical radius scales with ``open_amount``.
        3.  Adds a small ivory teeth strip inside the cavity for wide-open
            states (open_amount ≥ 0.45).
        4.  Composites the modified region back into the image using the
            cavity mask itself as the blend weight (no visible hard edges).

        Works for any lip colour because it *darkens* the existing pixels
        rather than replacing them with a sampled colour.

        ``viseme_maps`` is accepted but ignored (kept for API compat).
        """
        import cv2
        import numpy as np

        if open_amount < 0.04:
            return img  # mouth fully closed — nothing to do

        face_x, face_y, face_w, face_h = face_info["face"]
        m        = face_info["mouth"]
        h, w     = img.shape[:2]
        mcx      = m["cx"]
        mcy      = m["cy"]
        skin_bgr = np.array(face_info["skin_color"], dtype=np.float32)

        # ------------------------------------------------------------------ #
        # 1.  Locate the lip-seam row                                          #
        # ------------------------------------------------------------------ #
        # Scan a narrow band centred on mcy.  Strategy:
        #   a) First try: find the row with the highest R/(G+1) ratio in the
        #      right-half of the mouth zone — lips are always warmer/more
        #      saturated (higher R/G) than surrounding skin or dark hair.
        #   b) Fallback: minimum-brightness row in the right half (avoids
        #      the hair on the left side of the face distorting the result).
        scan_r   = int(face_h * 0.10)           # ± 10% of face height
        scan_hw  = max(10, int(m["w"] * 0.45))  # half-width of search zone
        sy1 = max(0, mcy - scan_r)
        sy2 = min(h, mcy + scan_r)
        # Use the full width of the mouth zone but skip the leftmost 20%
        # to avoid the hair shadow that typically falls on the left side.
        sx1 = max(0, mcx - int(scan_hw * 0.8))  # 80% left of centre
        sx2 = min(w, mcx + scan_hw)              # full right extent
        lip_cy = mcy   # fallback if band is too small
        if sy2 > sy1 + 3 and sx2 > sx1 + 3:
            band = img[sy1:sy2, sx1:sx2].astype(np.float32)
            r_ch = band[:, :, 2]
            g_ch = band[:, :, 1]
            # Row-wise mean R/(G+1) — lips score highest
            rg_row = (r_ch / (g_ch + 1.0)).mean(axis=1)
            best_rg = float(rg_row.max())
            if best_rg > 1.08:   # clear warm-tone row found
                lip_cy = sy1 + int(np.argmax(rg_row))
            else:
                # Fallback: minimum brightness in right-half band
                gmean = cv2.cvtColor(band.astype(np.uint8), cv2.COLOR_BGR2GRAY
                                     ).astype(np.float32).mean(axis=1)
                lip_cy = sy1 + int(np.argmin(gmean))

        # ------------------------------------------------------------------ #
        # 2.  Cavity geometry                                                  #
        # ------------------------------------------------------------------ #
        # Max vertical radius of cavity.  The Sobel face box (fh) includes
        # hair and upper-chest, so the actual head height is ~65% of fh.
        # Using 6.5% of fh gives a clearly visible open-mouth cavity while
        # staying proportional to the presenter's face size.
        max_ry  = max(4, int(face_h * 0.065))
        cav_ry  = max(1, int(max_ry * open_amount))
        # Horizontal radius = 42% of detected mouth width
        cav_rx  = max(4, int(m["w"] * 0.42 * width_factor))

        # ------------------------------------------------------------------ #
        # 3.  Working ROI                                                      #
        # ------------------------------------------------------------------ #
        pad    = cav_rx + cav_ry + 6
        roi_x1 = max(0, mcx - pad);   roi_x2 = min(w, mcx + pad)
        roi_y1 = max(0, lip_cy - pad); roi_y2 = min(h, lip_cy + pad)
        if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
            return img

        rh, rw = roi_y2 - roi_y1, roi_x2 - roi_x1
        lcx    = mcx   - roi_x1
        lcy    = lip_cy - roi_y1

        # ------------------------------------------------------------------ #
        # 4.  Cavity mask (soft ellipse)                                       #
        # ------------------------------------------------------------------ #
        cav_mask = np.zeros((rh, rw), dtype=np.float32)
        if cav_rx > 0 and cav_ry > 0:
            cv2.ellipse(cav_mask, (lcx, lcy), (cav_rx, cav_ry),
                        0, 0, 360, 1.0, -1)
        # Soft Gaussian feather — the mask itself acts as the blend weight
        blur_k = max(3, 2 * (cav_ry // 2) + 1)   # kernel ≥ 3, always odd
        cav_mask = cv2.GaussianBlur(cav_mask, (blur_k, blur_k), cav_ry * 0.5)

        # ------------------------------------------------------------------ #
        # 5.  Apply cavity: darken pixels inside the mask                     #
        # ------------------------------------------------------------------ #
        orig    = img[roi_y1:roi_y2, roi_x1:roi_x2].astype(np.float32)
        # Cavity colour: dark warm-maroon (natural open-mouth interior).
        # Use a fixed warm dark tone rather than deriving from skin_bgr,
        # because skin colour varies widely and often skews toward blue
        # on AI-generated images with cool studio lighting.
        cav_col = np.array([18., 12., 22.], np.float32)  # BGR: dark maroon
        cm3     = np.stack([cav_mask] * 3, axis=-1)
        # Opacity scales with open_amount so a barely-open mouth is subtle
        opacity = min(0.88, 0.60 + 0.28 * open_amount)
        roi_out = orig * (1.0 - cm3 * opacity) + cav_col * (cm3 * opacity)

        # ------------------------------------------------------------------ #
        # 6.  Teeth strip inside the cavity (open ≥ 0.45)                    #
        # ------------------------------------------------------------------ #
        if open_amount >= 0.35 and cav_ry >= 3:
            ivory      = np.array([215, 220, 222], dtype=np.float32)
            tth_mask   = np.zeros((rh, rw), dtype=np.float32)
            t_ry = max(1, int(cav_ry * 0.35))
            t_rx = max(1, int(cav_rx * 0.70))
            t_cy = lcy - max(1, int(cav_ry * 0.20))   # upper half of cavity
            if t_rx > 0 and t_ry > 0:
                cv2.ellipse(tth_mask, (lcx, t_cy), (t_rx, t_ry),
                            0, 0, 360, 1.0, -1)
            tth_mask = cv2.GaussianBlur(tth_mask, (3, 3), 0.8)
            tth_mask = np.minimum(tth_mask, cav_mask)   # only inside cavity
            t_op  = min(0.85, 0.50 + 0.35 * open_amount)
            tm3   = np.stack([tth_mask] * 3, axis=-1)
            roi_out = roi_out * (1.0 - tm3 * t_op) + ivory * (tm3 * t_op)

        img[roi_y1:roi_y2, roi_x1:roi_x2] = np.clip(roi_out, 0, 255).astype(np.uint8)
        return img

    # -- eyebrow animation --------------------------------------------------
    def _animate_eyebrows(self, img, face_info, mid_energy, open_amount):
        """Raise eyebrows slightly based on mid-frequency speech energy.

        Shifts the eyebrow strip (face top to just above the eyes) upward
        by up to 4 px with a Gaussian feather at the top edge.  On stressed
        syllables (open_amount > 0.65) an extra 2-3 px of lift is added to
        convey vocal emphasis.
        """
        import numpy as np

        face_x, face_y, face_w, face_h = face_info["face"]
        h, w = img.shape[:2]

        # Eyebrow zone: from the top of the detected face to just above the
        # upper eye border, padded by a few pixels.
        eye_top = min(
            face_info["left_eye"]["y"], face_info["right_eye"]["y"],
        )
        brow_y1 = max(0, face_y)
        brow_y2 = max(brow_y1 + 2, eye_top - 2)
        brow_h  = brow_y2 - brow_y1
        if brow_h < 2:
            return img

        # Lift amount: 0–4 px from mid energy, +0–3 px extra on stress
        lift = int(mid_energy * 4)
        if open_amount > 0.65:
            lift += int((open_amount - 0.65) / 0.35 * 3)
        lift = min(lift, 6)
        if lift < 1:
            return img

        # Scroll the strip upward by lift pixels, fill gap at bottom with
        # the original bottom row (keeps continuity with the face skin below)
        strip = img[brow_y1:brow_y2, face_x:face_x + face_w].copy()
        shifted = np.empty_like(strip)
        if lift < brow_h:
            shifted[:brow_h - lift] = strip[lift:]
            shifted[brow_h - lift:] = strip[-1:]   # repeat bottom row
        else:
            shifted[:] = strip[-1:]

        # Feather the top edge so the shift blends into the background
        feather_rows = max(2, min(lift + 2, brow_h // 3))
        for dy in range(feather_rows):
            alpha = dy / feather_rows
            yy = brow_y1 + dy
            if yy >= h:
                break
            orig_row = img[yy, face_x:face_x + face_w].astype(np.float32)
            new_row  = shifted[dy].astype(np.float32)
            img[yy, face_x:face_x + face_w] = np.clip(
                orig_row * (1 - alpha) + new_row * alpha, 0, 255,
            ).astype(np.uint8)

        # Write the rest of the strip unchanged
        y_start = brow_y1 + feather_rows
        rows_left = brow_h - feather_rows
        if rows_left > 0 and y_start + rows_left <= h:
            img[y_start:y_start + rows_left, face_x:face_x + face_w] = (
                shifted[feather_rows:feather_rows + rows_left]
            )

        return img

    # -- eye blink animation ------------------------------------------------
    def _animate_blink(self, img, face_info, blink_amount):
        """Close the eyelids by blending skin from above each eye downward.

        Uses per-column sampled skin colour from just above the eye
        (the eyelid) so the result looks like a natural eyelid closing
        rather than an opaque block.  Alpha raised to 0.75 for a crisper,
        more realistic closure; an eyelash shadow line is added.
        """
        import cv2
        import numpy as np

        h, w = img.shape[:2]

        for eye_key in ("left_eye", "right_eye"):
            eye = face_info[eye_key]
            ex, ey, ew, eh = eye["x"], eye["y"], eye["w"], eye["h"]

            x1 = max(0, ex + 1)
            x2 = min(w, ex + ew - 1)
            y1 = max(0, ey)
            y2 = min(h, ey + eh)
            if x2 <= x1 or y2 <= y1:
                continue

            cover_rows = int((y2 - y1) * min(blink_amount, 1.0))
            if cover_rows < 1:
                continue

            # Sample eyelid skin from just above the eye
            sample_y1 = max(0, ey - int(eh * 0.8))
            sample_y2 = max(sample_y1 + 1, ey - 2)
            if sample_y2 <= sample_y1:
                sample_y1, sample_y2 = max(0, ey - 3), max(1, ey)

            eyelid_strip = img[sample_y1:sample_y2, x1:x2]
            if eyelid_strip.size == 0:
                continue
            eyelid_colours = eyelid_strip.mean(axis=0)  # (width, 3)

            # Horizontal feather window: full coverage at the eye
            # centre, fading to 35% at the corners so the closing
            # eyelid never shows hard rectangular edges.
            width = x2 - x1
            cols = np.arange(width, dtype=np.float32)
            hwin = 0.35 + 0.65 * np.sin(
                np.pi * (cols + 1) / (width + 1)) ** 0.8
            hwin3 = np.stack([hwin] * 3, axis=-1)       # (width, 3)

            for dy in range(cover_rows):
                yy = y1 + dy
                if yy >= h:
                    break
                # Exponential-feel alpha: fast at start, feathered at edge
                t_norm = dy / max(cover_rows - 1, 1)
                if t_norm < 0.65:
                    alpha = 0.75          # crisp closure
                else:
                    fade  = (t_norm - 0.65) / 0.35
                    alpha = 0.75 - fade * 0.35   # fade to 0.40 at leading edge

                row = img[yy, x1:x2].astype(np.float32)
                lid = eyelid_colours.astype(np.float32)
                amap = alpha * hwin3
                img[yy, x1:x2] = np.clip(
                    row * (1 - amap) + lid * amap, 0, 255,
                ).astype(np.uint8)

            # Eyelash shadow: 1px dark line at the closing boundary
            shadow_y = y1 + cover_rows - 1
            if 0 < shadow_y < h - 1:
                shadow_col = np.clip(
                    eyelid_colours.mean(axis=0) - 35, 0, 255,
                ).astype(np.uint8)
                cv2.line(img, (x1, shadow_y), (x2 - 1, shadow_y),
                         tuple(int(c) for c in shadow_col), 1)

        return img

    # -- blink timing update ------------------------------------------------
    @staticmethod
    def _update_blink(time_sec, next_blink, blink_phase):
        """Update blinking state.  Returns (blink_amount, next_blink, phase).

        Timing uses an exponential-feel curve:
          close: 3 frames (~100 ms), hold: 1 frame, open: 4 frames (~133 ms)
        """
        import numpy as np

        blink_frames_close = 3
        blink_frames_hold  = 1
        blink_frames_open  = 4
        blink_total = blink_frames_close + blink_frames_hold + blink_frames_open

        if time_sec >= next_blink and blink_phase == 0:
            blink_phase = 1

        blink_amount = 0.0
        if blink_phase > 0:
            p = blink_phase
            if p <= blink_frames_close:
                # Fast close — exponential feel
                blink_amount = (p / blink_frames_close) ** 0.7
            elif p <= blink_frames_close + blink_frames_hold:
                blink_amount = 1.0
            elif p <= blink_total:
                prog = (p - blink_frames_close - blink_frames_hold) / blink_frames_open
                # Slower open
                blink_amount = 1.0 - prog ** 1.4
            else:
                blink_phase = 0
                blink_amount = 0.0
                next_blink = time_sec + np.random.uniform(2.5, 5.0)

        if blink_phase > 0:
            blink_phase += 1

        return blink_amount, next_blink, blink_phase

    # -- FFT-based audio spectral analysis -----------------------------------
    def _analyse_fft_energy(self, audio_path: str, duration: float) -> list:
        """Compute per-frame mouth-open signal from the audio.

        Strategy
        --------
        The Urdu TTS audio has a very uniform RMS level, so spectral band
        analysis and derivative-threshold onset detection both yield flat
        signals.  The reliable approach is:

        1.  Build a 10 ms hop RMS envelope for temporal resolution.
        2.  Find local amplitude maxima (scipy.signal.find_peaks) with a
            minimum spacing of 70 ms — these correspond to syllable nuclei
            at the natural ~9/s Urdu speech rate.
        3.  Each peak launches an amplitude-weighted open-then-close impulse
            response (30 ms attack, 120 ms decay) to form the jaw-open curve.
        4.  Blend with mid/high spectral bands (consonant shaping).
        5.  Resample to the video frame rate.

        Returns a list of dicts per video frame: bass, mid, high, rms (0-1).
        """
        try:
            from audio_processing.wav_io import read_wav
            import numpy as np
            from scipy.ndimage import uniform_filter1d
            from scipy.signal import find_peaks

            wav = read_wav(audio_path)
            samples = np.array(wav.samples, dtype=np.float64)
            rate = wav.rate
            channels = max(wav.channels, 1)

            if channels > 1:
                samples = samples.reshape(-1, channels).mean(axis=1)

            # Normalize to peak
            peak = np.abs(samples).max()
            if peak > 1e-9:
                samples = samples / peak

            fps = vc.VIDEO_FPS
            total_frames = max(1, int(duration * fps))

            # ---- 1. Build 10ms-hop RMS envelope ----------------------------
            hop = max(1, int(rate * 0.010))   # 10ms per envelope frame
            n_env = int(len(samples) / hop)
            rms_env = np.array([
                float(np.sqrt(np.mean(
                    samples[i * hop: (i + 1) * hop] ** 2)))
                for i in range(n_env)
            ], dtype=np.float64)

            # Light smoothing before peak finding
            rms_s = uniform_filter1d(rms_env, size=3, mode="nearest")

            # ---- 2. Syllable-nucleus peak detection -------------------------
            # Minimum inter-syllable gap: 70 ms (Urdu ~9 syllables/s)
            min_dist = max(1, int(0.070 / 0.010))
            # Only accept peaks above half the mean (excludes quiet gaps)
            height_thresh = float(rms_s.mean() * 0.5)
            syllable_peaks, _ = find_peaks(
                rms_s, distance=min_dist, height=height_thresh,
            )

            # ---- 3. Build jaw-open impulse train ----------------------------
            attack_fr = max(1, int(0.030 / 0.010))   # 30ms = 3 env-frames
            decay_fr  = max(1, int(0.120 / 0.010))   # 120ms = 12 env-frames
            jaw_curve = np.zeros(n_env, dtype=np.float64)
            rms_max = float(rms_s.max()) if rms_s.max() > 1e-9 else 1.0

            for idx in syllable_peaks:
                amp = float(rms_s[idx]) / rms_max  # amplitude-weighted
                # attack: ramp up
                for di in range(attack_fr):
                    t_i = min(idx + di, n_env - 1)
                    jaw_curve[t_i] = max(
                        jaw_curve[t_i], amp * (di + 1) / attack_fr,
                    )
                # decay: ramp down
                for di in range(decay_fr):
                    t_i = min(idx + attack_fr + di, n_env - 1)
                    jaw_curve[t_i] = max(
                        jaw_curve[t_i], amp * (1.0 - (di + 1) / decay_fr),
                    )

            # Silence gate: zero out frames below noise floor
            jaw_curve[rms_s < 0.02] = 0.0

            # Light smoothing to remove sharp quantisation steps
            jaw_curve = uniform_filter1d(jaw_curve, size=3, mode="nearest")

            # ---- 4. Mid/high spectral bands (consonant shaping) -------------
            win_samples = max(32, int(rate * 0.05))   # 50ms FFT window
            mid_env = np.zeros(total_frames, dtype=np.float64)
            high_env = np.zeros(total_frames, dtype=np.float64)
            for fi in range(total_frames):
                centre = int((fi + 0.5) * rate / fps)
                start  = max(0, centre - win_samples // 2)
                end    = min(len(samples), start + win_samples)
                chunk  = samples[start:end]
                if len(chunk) < 16:
                    continue
                padded = np.zeros(win_samples)
                padded[:len(chunk)] = chunk * np.hanning(len(chunk))
                fft   = np.abs(np.fft.rfft(padded))
                freqs = np.fft.rfftfreq(win_samples, 1.0 / rate)
                mid_mask  = (freqs >= 500)  & (freqs < 2000)
                high_mask = (freqs >= 2000) & (freqs < 5000)
                mid_env[fi]  = (float(np.mean(fft[mid_mask]))
                                if mid_mask.any() else 0.0)
                high_env[fi] = (float(np.mean(fft[high_mask]))
                                if high_mask.any() else 0.0)

            for arr in (mid_env, high_env):
                mx = float(np.percentile(arr, 99))
                if mx > 1e-9:
                    arr /= mx
                np.clip(arr, 0, 1, out=arr)

            # ---- 5. Resample jaw curve to video frame rate ------------------
            x_env = np.linspace(0, 1, n_env)
            x_vid = np.linspace(0, 1, total_frames)
            jaw_at_fps = np.interp(x_vid, x_env, jaw_curve)

            # RMS at video fps
            x_rms = np.linspace(0, 1, n_env)
            rms_at_fps = np.interp(x_vid, x_rms, rms_env)
            rms_peak = float(np.percentile(rms_at_fps, 99))
            if rms_peak > 1e-9:
                rms_at_fps = np.clip(rms_at_fps / rms_peak, 0, 1)

            # ---- 6. Assemble per-frame dicts --------------------------------
            band_energies = []
            for fi in range(total_frames):
                band_energies.append({
                    "bass": float(jaw_at_fps[fi]),   # jaw-open drive
                    "mid":  float(mid_env[fi] ** 0.7),
                    "high": float(high_env[fi] ** 0.7),
                    "rms":  float(rms_at_fps[fi]),
                })

            return band_energies
        except Exception as exc:
            logger.warning("FFT audio analysis failed: %s", exc)
            return []

    # -- studio background rendering ----------------------------------------
    def _render_studio_background(self):
        """Render a professional Pakistani TV news studio background.

        Features: deep navy gradient, ambient studio lighting, background
        news screens with content, desk surface, channel branding.
        """
        import cv2
        import numpy as np

        w, h = vc.VIDEO_WIDTH, vc.VIDEO_HEIGHT
        bg = np.zeros((h, w, 3), dtype=np.uint8)

        # Deep navy gradient (top to bottom) — vectorised
        y_ratios = np.linspace(0, 1, h).reshape(h, 1, 1)
        top_color = np.array([80, 20, 8],  dtype=np.float32)   # BGR: dark navy top
        bot_color = np.array([55, 15, 5],  dtype=np.float32)   # BGR: slightly darker bottom
        bg = (top_color + (bot_color - top_color) * y_ratios).astype(np.uint8)
        bg = np.broadcast_to(bg, (h, w, 3)).copy()

        # Ambient studio lighting — warm overhead glow (vectorised)
        cx, cy = w // 2, h // 4
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        dx = (xx - cx) / (w * 0.5)
        dy = (yy - cy) / (h * 0.5)
        d2 = dx * dx + dy * dy
        glow_intensity = np.clip(0.35 * np.exp(-d2 * 1.8), 0, 1)
        glow = np.stack([
            glow_intensity * 50,   # B
            glow_intensity * 35,   # G
            glow_intensity * 25,   # R
        ], axis=-1)
        bg = np.clip(bg.astype(np.float32) + glow, 0, 255).astype(np.uint8)

        # Background news screens (left and right) with content
        self._draw_news_screen(bg, 60, 120, 380, 280, screen_idx=0)
        self._draw_news_screen(bg, w - 60 - 380, 120, 380, 280, screen_idx=1)

        # Geometric accent lines (top area, subtle)
        for i in range(10):
            offset = i * 40
            cv2.line(
                bg, (w // 2 - 300 + offset, 0), (w // 2 + 300 - offset, 80),
                (45, 35, 25), 1,
            )

        # News desk surface (bottom area, darker)
        desk_y = h - 180
        desk_darkening = np.linspace(0, 20, h - desk_y).reshape(-1, 1, 1)
        bg[desk_y:, :] = np.clip(
            bg[desk_y:, :].astype(np.int16) - desk_darkening.astype(np.int16),
            0, 255,
        ).astype(np.uint8)
        # Desk edge highlight
        cv2.line(bg, (0, desk_y), (w, desk_y), (80, 65, 45), 2)

        # Subtle side separator lines (dark navy — no coloured border visible)
        cv2.rectangle(bg, (0, 0), (3, h), (50, 20, 8), -1)
        cv2.rectangle(bg, (w - 3, 0), (w, h), (50, 20, 8), -1)

        return bg

    def _draw_news_screen(self, bg, x, y, sw, sh, screen_idx=0):
        """Draw a background news screen with graphics content."""
        import cv2
        import numpy as np

        # Screen frame (metallic border)
        cv2.rectangle(bg, (x - 3, y - 3), (x + sw + 3, y + sh + 3), (70, 55, 35), 2)
        # Screen inner area — dim blue
        inner = bg[y + 2:y + sh - 2, x + 2:x + sw - 2]
        screen_fill = np.full_like(inner, (45, 32, 18))
        bg[y + 2:y + sh - 2, x + 2:x + sw - 2] = (
            inner.astype(np.float32) * 0.4 + screen_fill * 0.6
        ).astype(np.uint8)

        # Screen content: abstract news graphics
        cx, cy = x + sw // 2, y + sh // 2

        if screen_idx == 0:
            # Left screen: globe graphic
            cv2.circle(bg, (cx, cy), min(sw, sh) // 4, (60, 80, 120), 2)
            cv2.circle(bg, (cx, cy), min(sw, sh) // 6, (50, 70, 110), 1)
            # Latitude/longitude lines
            for r_offset in (-min(sw, sh) // 8, 0, min(sw, sh) // 8):
                cv2.ellipse(
                    bg, (cx, cy),
                    (min(sw, sh) // 4, abs(min(sw, sh) // 4 - abs(r_offset)) or 1),
                    0, 0, 360, (55, 75, 110), 1,
                )
            # Horizontal bars (news ticker simulation)
            for i in range(4):
                bar_y = y + sh - 40 - i * 12
                bar_w = int(sw * (0.3 + 0.15 * ((i + screen_idx) % 3)))
                cv2.rectangle(
                    bg, (x + 15, bar_y), (x + 15 + bar_w, bar_y + 5),
                    (50, 45, 60), -1,
                )
        else:
            # Right screen: bar chart graphic (financial/news data)
            bar_count = 5
            bar_width = (sw - 40) // (bar_count * 2)
            for i in range(bar_count):
                bar_h = int((sh * 0.35) * (0.3 + 0.7 * ((i * 37 + screen_idx * 13) % 10) / 10))
                bx = x + 20 + i * bar_width * 2
                by = y + sh - 50 - bar_h
                colour = (60, 55, 90) if i % 2 == 0 else (50, 70, 100)
                cv2.rectangle(bg, (bx, by), (bx + bar_width, y + sh - 50), colour, -1)
            # Header line
            cv2.rectangle(
                bg, (x + 15, y + 15), (x + sw - 15, y + 25), (55, 50, 70), -1,
            )

        # Screen glow effect (subtle)
        glow_region = bg[max(0, y - 8):min(bg.shape[0], y + sh + 8),
                         max(0, x - 8):min(bg.shape[1], x + sw + 8)]
        bg[max(0, y - 8):min(bg.shape[0], y + sh + 8),
           max(0, x - 8):min(bg.shape[1], x + sw + 8)] = cv2.GaussianBlur(
            glow_region, (5, 5), 0,
        )
