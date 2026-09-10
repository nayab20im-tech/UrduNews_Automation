#!/usr/bin/env python3
"""Builds the Kaggle ipynb (assets base64-embedded) and pushes it via MCP save_notebook."""
import base64, json, sys
from pathlib import Path

HERE = Path(__file__).parent
IMG = HERE / "assets" / "anchor_female.jpg"
AUD = HERE / "assets" / "bulletin.mp3"

SETUP = r'''
import os, subprocess, sys, shutil, glob

def sh(cmd):
    print(f"\n$ {cmd}")
    subprocess.run(cmd, shell=True, check=True)

if not os.path.isdir("/kaggle/working/echomimic_v2"):
    sh("git clone --depth 1 https://github.com/antgroup/echomimic_v2 /kaggle/working/echomimic_v2")
os.chdir("/kaggle/working/echomimic_v2")

# Kaggle's image already has CUDA-compatible torch; skip torch lines
SKIP = ("torch", "xformers", "torchao", "onnxruntime", "decord", "numpy")
req = open("requirements.txt").read()
req = "\n".join(l for l in req.splitlines() if l.strip() and not l.strip().startswith("#")
                and not l.lower().startswith(SKIP))
open("/tmp/req_nogpu.txt", "w").write(req)
sh(f"{sys.executable} -m pip install -q -r /tmp/req_nogpu.txt")
sh(f"{sys.executable} -m pip install -q --no-deps facenet_pytorch==2.6.0 "
   "huggingface_hub eva-decord")
# decord stub: if the fork installs under a different module name
try:
    import decord  # noqa: F401
    print("decord import OK")
except ImportError:
    try:
        import site, pathlib
        import eva_decord
        stub = pathlib.Path(site.getsitepackages()[0]) / "decord"
        stub.mkdir(exist_ok=True)
        (stub / "__init__.py").write_text("from eva_decord import *  # noqa\n"
                                          "from eva_decord import VideoReader  # noqa\n")
        print("decord stubbed via eva_decord")
    except ImportError:
        print("WARNING: decord unavailable (may not be needed for this path)")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
print("setup OK")
'''

WEIGHTS = r'''
from huggingface_hub import hf_hub_download

PW = "/tmp/pretrained_weights"          # big weights on /tmp (disk quota safe)
os.makedirs(f"{PW}/sd-vae-ft-mse", exist_ok=True)

repo = "BadToBest/EchoMimicV2"
for f in ["denoising_unet_acc.pth", "motion_module_acc.pth",
          "reference_unet.pth", "pose_encoder.pth"]:
    hf_hub_download(repo_id=repo, filename=f, local_dir=PW)

# SD VAE
for f in ["config.json", "diffusion_pytorch_model.safetensors"]:
    hf_hub_download(repo_id="stabilityai/sd-vae-ft-mse", filename=f,
                    local_dir=f"{PW}/sd-vae-ft-mse")

# SD1.5 image-variations base (UNet config + weights for reference net)
iv = f"{PW}/sd-image-variations-diffusers"
os.makedirs(f"{iv}/unet", exist_ok=True)
for f in ["model_index.json"]:
    hf_hub_download(repo_id="stabilityai/sd-image-variations-diffusers",
                    filename=f, local_dir=iv)
for f in ["config.json", "diffusion_pytorch_model.bin"]:
    hf_hub_download(repo_id="stabilityai/sd-image-variations-diffusers",
                    filename=f"unet/{f}", local_dir=f"{iv}/unet")

# whisper tiny audio processor: repo's patched whisper needs the ORIGINAL
# OpenAI checkpoint format (tiny.pt), not the HF safetensors version.
# (hash taken from openai/whisper _MODELS registry)
os.makedirs(f"{PW}/audio_processor", exist_ok=True)
import urllib.request
TINY = ("https://openaipublic.azureedge.net/main/whisper/models/"
        "5d4783c5fdd6a2d00fc0e36ca5f2e34de02c0326d41f0c91f8f0ec11"
        "f8f0ec11/tiny.pt")
urllib.request.urlretrieve(TINY, f"{PW}/audio_processor/tiny.pt")
sh(f"du -sh {PW}")
print("weights OK")
'''

STAGE = r'''
import base64

DATA = "/kaggle/working/assets"
os.makedirs(DATA, exist_ok=True)
with open(f"{DATA}/anchor_female.jpg", "wb") as fh:
    fh.write(base64.b64decode(IMG_B64))
with open(f"{DATA}/bulletin.mp3", "wb") as fh:
    fh.write(base64.b64decode(AUD_B64))
del IMG_B64, AUD_B64

# mp3 -> 16k wav for the audio feature extractor
sh(f"ffmpeg -y -loglevel error -i {DATA}/bulletin.mp3 -ar 16000 -ac 1 {DATA}/bulletin_16k.wav")

os.makedirs("assets/halfbody_demo/refimag/news", exist_ok=True)
os.makedirs("assets/halfbody_demo/audio/urdu", exist_ok=True)
shutil.copy(f"{DATA}/anchor_female.jpg", "assets/halfbody_demo/refimag/news/anchor.jpg")
shutil.copy(f"{DATA}/bulletin_16k.wav", "assets/halfbody_demo/audio/urdu/bulletin.wav")

from moviepy.editor import AudioFileClip
dur = AudioFileClip(f"{DATA}/bulletin_16k.wav").duration
print(f"audio {dur:.1f}s -> up to {int(dur*24)} frames @24fps")
'''

POSE = r'''
POSES = "assets/halfbody_demo/pose"
best_pose, best_n = "01", 0
for p in sorted(os.listdir(POSES)):
    d = os.path.join(POSES, p)
    if os.path.isdir(d):
        n = len(glob.glob(os.path.join(d, "*.npy")))
        print(f"pose set {p}: {n} frames")
        if n > best_n:
            best_pose, best_n = p, n
L_FRAMES = min(best_n, int(dur * 24))
print("using pose set", best_pose, "| rendering", L_FRAMES,
      "frames =", round(L_FRAMES/24, 1), "s")
'''

RUN = r'''
os.environ["FFMPEG_PATH"] = "/usr/bin"          # kaggle has ffmpeg installed
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

PW = "/tmp/pretrained_weights"
REF = "/kaggle/working/echomimic_v2/assets/halfbody_demo/refimag/news/anchor.jpg"
AUD = "/kaggle/working/echomimic_v2/assets/halfbody_demo/audio/urdu/bulletin.wav"
POSE = f"/kaggle/working/echomimic_v2/assets/halfbody_demo/pose/{best_pose}/"

cfg = f"""pretrained_base_model_path: "{PW}/sd-image-variations-diffusers"
pretrained_vae_path: "{PW}/sd-vae-ft-mse"
denoising_unet_path: "{PW}/denoising_unet_acc.pth"
reference_unet_path: "{PW}/reference_unet.pth"
pose_encoder_path: "{PW}/pose_encoder.pth"
motion_module_path: "{PW}/motion_module_acc.pth"
audio_model_path: "{PW}/audio_processor/tiny.pt"
inference_config: "./configs/inference/inference_v2.yaml"
weight_dtype: "fp16"
test_cases:
  "{REF}":
    - "{AUD}"
    - "{POSE}"
"""
open("configs/prompts/infer_news_acc.yaml", "w").write(cfg)

sh("python infer_acc.py --config configs/prompts/infer_news_acc.yaml "
   "-W 768 -H 1024 " + f"-L {L_FRAMES} "
   "--seed 42 --steps 6 --cfg 1.0 --fps 24")
print("inference done")
'''

COLLECT = r'''
outs = sorted(glob.glob("output/**/*_sig.mp4", recursive=True))
print("generated:", outs)
assert outs, "no output video!"
for f in outs:
    shutil.copy(f, "/kaggle/working/" + os.path.basename(f))
shutil.copy(outs[-1], "/kaggle/working/anchor_bulletin.mp4")
print("DONE -> /kaggle/working/anchor_bulletin.mp4")
'''

def chunk_b64(data: bytes, name: str, chunk: int = 900_000) -> list:
    b = base64.b64encode(data).decode()
    cells, parts = [], [b[i:i+chunk] for i in range(0, len(b), chunk)]
    for i, part in enumerate(parts):
        op = "=" if i == 0 else "+="
        cells.append(f'{name} {op} "{part}"\n'
                     + (f'print("chunk {i+1}/{len(parts)} loaded")' if i == len(parts)-1 else ""))
    return cells

cells = [
    ("markdown", "# Realistic Urdu News Anchor — EchoMimicV2 (accelerated) on Kaggle GPU\n"
                 "Audio-driven half-body animation with hand gestures."),
    ("code", SETUP),
    ("code", WEIGHTS),
    *[(c := None, ) for _ in ()],  # placeholder removed
]
cells = [c for c in cells if isinstance(c[1], str)]
cells += [("code", c) for c in chunk_b64(IMG.read_bytes(), "IMG_B64")]
cells += [("code", c) for c in chunk_b64(AUD.read_bytes(), "AUD_B64")]
cells += [("code", STAGE), ("code", POSE), ("code", RUN), ("code", COLLECT)]

nb = {
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                "name": "python3"},
                 "language_info": {"name": "python", "version": "3.10.13"}},
    "nbformat": 4, "nbformat_minor": 4,
    "cells": [{"cell_type": t,
               "source": s.splitlines(keepends=True),
               **({"metadata": {}, "execution_count": None, "outputs": []}
                  if t == "code" else {"metadata": {}})}
              for t, s in cells],
}

text = json.dumps(nb)
print(f"notebook size: {len(text)/1e6:.2f} MB")
Path(HERE / "notebook.ipynb").write_text(text)

# Flat python script (kernel_type=script executes raw python, not ipynb)
script_src = "\n\n".join(s for t, s in cells if t == "code")
Path(HERE / "script.py").write_text(script_src)
print(f"script size: {len(script_src)/1e6:.2f} MB")

if "--push" in sys.argv:
    sys.path.insert(0, str(HERE))
    from kg import call
    resp = call("save_notebook", {
        "text": script_src,
        "newTitle": "urdu-anchor-realistic",
        "slug": "nayabnasir20/urdu-anchor-realistic",
        "language": "python",
        "kernelType": "script",
        "kernelExecutionType": "SaveAndRunAll",
        "isPrivate": True,
        "enableGpu": True,
        "enableInternet": True,
        "machineShape": "gpuT4x2",
        "sessionTimeoutSeconds": 10800,
    })
    print(json.dumps(resp, indent=2))
else:
    print("dry run (use --push to upload & run)")
