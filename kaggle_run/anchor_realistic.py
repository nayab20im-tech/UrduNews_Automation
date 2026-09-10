# %% [markdown]
# # Realistic Urdu News Anchor — EchoMimicV2 on Kaggle GPU
# Audio-driven half-body animation (talking + natural hand gestures)
# of a photorealistic anchor image, driven by a neural Urdu (ur-PK)
# voice-over.  Inputs come from the attached private dataset:
#   /kaggle/input/<dataset>/anchor_female.png + bulletin_16k.wav

# %% [code]
# --- 1. Environment: clone repo + install deps -------------------------
import os, subprocess, sys, shutil, glob

def sh(cmd, **kw):
    print(f"\n$ {cmd}")
    return subprocess.run(cmd, shell=True, check=True, **kw)

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

if not os.path.isdir("/kaggle/working/echomimic_v2"):
    sh("git clone --depth 1 https://github.com/antgroup/echomimic_v2 "
       "/kaggle/working/echomimic_v2")
os.chdir("/kaggle/working/echomimic_v2")

# Kaggle's base image already ships torch built for CUDA 12.x — do NOT
# reinstall torch (the repo's requirements.txt would downgrade it).
req = open("requirements.txt").read()
req = "\n".join(
    ln for ln in req.splitlines()
    if ln.strip() and not ln.strip().startswith("#")
    and not ln.lower().startswith(("torch", "xformers", "torchao"))
)
open("/tmp/req_nogpu.txt", "w").write(req)
sh(f"{sys.executable} -m pip install -q -r /tmp/req_nogpu.txt")
sh(f"{sys.executable} -m pip install -q --no-deps facenet_pytorch==2.6.0 "
   "huggingface_hub edge-tts")

print(sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader").check)

# %% [code]
# --- 2. Download pretrained weights (BadToBest/EchoMimicV2 mirror) -----
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="BadToBest/EchoMimicV2",
    local_dir="/kaggle/working/pretrained_weights",
    allow_patterns=[
        "denoising_unet.pth", "reference_unet.pth", "motion_module.pth",
        "pose_encoder.pth", "sd-vae-ft-mse/*", "audio_processor/*",
    ],
)
# sd-image-variations-diffusers base model (SD1.5 UNet config+weights)
snapshot_download(
    repo_id="stabilityai/sd-image-variations-diffusers",
    local_dir="/kaggle/working/pretrained_weights/sd-image-variations-diffusers",
)
sh("du -sh /kaggle/working/pretrained_weights")

# %% [code]
# --- 3. Stage inputs (reference image + Urdu audio) --------------------
DATA_DIR = None
for cand in glob.glob("/kaggle/input/*"):
    if any(os.path.exists(os.path.join(cand, f))
           for f in ("anchor_female.png", "bulletin_16k.wav")):
        DATA_DIR = cand
        break
if DATA_DIR is None:
    raise SystemExit("Attach the anchor dataset (anchor_female.png, "
                     "bulletin_16k.wav) as an input to this notebook.")
print("input dataset:", DATA_DIR)

REF_IMG = os.path.join(DATA_DIR, "anchor_female.png")
AUDIO = os.path.join(DATA_DIR, "bulletin_16k.wav")

os.makedirs("assets/halfbody_demo/refimag/news", exist_ok=True)
os.makedirs("assets/halfbody_demo/audio/urdu", exist_ok=True)
shutil.copy(REF_IMG, "assets/halfbody_demo/refimag/news/anchor.png")
shutil.copy(AUDIO, "assets/halfbody_demo/audio/urdu/bulletin.wav")

from moviepy.editor import AudioFileClip
dur = AudioFileClip(AUDIO).duration
print(f"audio duration: {dur:.1f}s -> ~{int(dur*24)} frames @24fps")

# %% [code]
# --- 4. Pose: use bundled demo pose set (natural anchor gestures) ------
# EchoMimicV2 needs per-frame DWPose .npy files.  The repo ships demo
# pose sequences with natural presenter-style hand movement; we use the
# longest one and cap L accordingly.
POSES = "assets/halfbody_demo/pose"
best_pose, best_n = "01", 0
for p in sorted(os.listdir(POSES)):
    d = os.path.join(POSES, p)
    if os.path.isdir(d):
        n = len(glob.glob(os.path.join(d, "*.npy")))
        print(f"pose set {p}: {n} frames")
        if n > best_n:
            best_pose, best_n = p, n
print("using pose set:", best_pose, f"({best_n} frames)")
L_FRAMES = min(best_n, int(dur * 24))
print("rendering", L_FRAMES, "frames =", round(L_FRAMES / 24, 1), "s")

# %% [code]
# --- 5. Write inference config & run ------------------------------------
os.environ["FFMPEG_PATH"] = "/usr/bin"   # kaggle image has ffmpeg

cfg = """pretrained_base_model_path: "/kaggle/working/pretrained_weights/sd-image-variations-diffusers"
pretrained_vae_path: "/kaggle/working/pretrained_weights/sd-vae-ft-mse"
denoising_unet_path: "/kaggle/working/pretrained_weights/denoising_unet.pth"
reference_unet_path: "/kaggle/working/pretrained_weights/reference_unet.pth"
pose_encoder_path: "/kaggle/working/pretrained_weights/pose_encoder.pth"
motion_module_path: "/kaggle/working/pretrained_weights/motion_module.pth"
audio_model_path: "/kaggle/working/pretrained_weights/audio_processor/tiny.pt"
inference_config: "./configs/inference/inference_v2.yaml"
weight_dtype: "fp16"
"""
open("configs/prompts/infer_news.yaml", "w").write(cfg)

sh("python infer.py --config configs/prompts/infer_news.yaml "
   "-W 768 -H 1024 "
   f"-L {L_FRAMES} "
   "--seed 42 --steps 25 --cfg 2.5 --fps 24 "
   "--ref_images_dir ./assets/halfbody_demo/refimag "
   "--audio_dir ./assets/halfbody_demo/audio "
   f"--refimg_name news/anchor.png "
   "--audio_name urdu/bulletin.wav "
   f"--pose_name {best_pose}")

# %% [code]
# --- 6. Collect final video into /kaggle/working ------------------------
outs = sorted(glob.glob("outputs/**/*_sig.mp4", recursive=True))
print("generated:", outs)
assert outs, "no output video produced!"
shutil.copy(outs[-1], "/kaggle/working/anchor_bulletin.mp4")
# also expose every artefact at the working root for download
for f in outs:
    shutil.copy(f, "/kaggle/working/" + os.path.basename(f))
print("DONE -> /kaggle/working/anchor_bulletin.mp4")
