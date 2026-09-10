# avatar_engine — Audio-Driven Talking Anchor Subsystem

Turns an Urdu broadcast script into a lip-synced, blinking, naturally
animated 1080p news-anchor video, 100% locally (no SaaS).

```
Urdu script → sentence segmentation → semantic chunking (8–20 s)
     → TTS chunk (WAV) → audio-driven talking head + lip sync
     → per-chunk validation → FFmpeg stitch (H.264/AAC/yuv420p)
     → final QC validation → MP4
```

## 1. Architecture

```
avatar_engine/
├── __init__.py          # public API: generate_anchor_video(...)
├── pipeline.py          # orchestration (chunk → tts → avatar → stitch → validate)
├── config.py            # env-var + YAML configuration
├── errors.py            # typed per-stage exceptions
├── gpu.py               # CUDA/VRAM detection, device selection, VRAM release
├── cache.py             # SHA256 content-hash chunk cache
├── text_processing.py   # Urdu NFKC normalization, sentence split, chunking
├── tts.py               # modular TTS (piper / espeak / xtts / placeholder)
├── anchors.py           # derives + verifies male.png / female.png assets
├── prepare_anchors.py   # CLI: python -m avatar_engine.prepare_anchors
├── stitch.py            # concat demuxer + final controlled FFmpeg encode
├── validate.py          # FFprobe/FFmpeg QC (streams, codecs, A/V sync)
├── generate.py          # CLI entry point
├── server.py            # OPTIONAL FastAPI POST /generate (not a hard dep)
├── assets/              # male.png, female.png (derived from anchors.jpeg)
└── backends/
    ├── base.py          # TalkingHeadBackend ABC
    ├── animated.py      # DEFAULT: deterministic CPU audio-driven renderer
    └── sadtalker.py     # OPTIONAL GPU backend (Apache-2.0)
```

The talking-head renderer reuses the single proven implementation of
the animation math in `video_production/lip_sync.py` (syllable-peak jaw
curve, lip-seam cavity, blink, eyebrow, head sway, breathing) so there
is exactly one implementation in the repository.

## 2. Installation

Uses the existing project `venv` (no duplicate environment):

```bash
./venv/bin/python -m pip install "opencv-python-headless<5" pillow
# system tools already present: ffmpeg, ffprobe, espeak-ng (Urdu voice)
```

## 3. GPU requirements

* **Default stack (this repo's verified default):** CPU-only.
  espeak-ng TTS + `animated` renderer. ~30 s wall time per 6 s clip at
  1080p25 on a laptop CPU.
* **SadTalker backend:** NVIDIA CUDA GPU (≥ 8 GB VRAM recommended),
  torch + cloned repo + checkpoints (see §4).

## 4. Model downloads (optional upgrades)

| Component | ID / URL | License | Notes |
|---|---|---|---|
| Piper Urdu voice | `rhasspy/piper-voices` → `ur/ur_PK/fasih/medium/ur_PK-fasih-medium.onnx` (+ `.onnx.json`) | MIT | male voice; drop into `avatar_engine/assets/models/` and install the `piper` binary |
| SadTalker | github.com/OpenTalker/SadTalker + checkpoints (GoogleDrive/HF per official README) | Apache-2.0 | set `AVATAR_SADTALKER_REPO`, `AVATAR_SADTALKER_CHECKPOINT_DIR` |
| espeak-ng | system package | GPL | bundled `ur` voice — the zero-download default |
| XTTS-v2 | Coqui | CPML (**non-commercial**) | experimental only, never the auto default |

Considered but rejected as production defaults (feasibility/license):
MuseTalk (unstable checkpoint pairing, CUDA-only), LivePortrait (not
audio-driven by itself), EchoMimicV2/3 (CUDA-only, English/Chinese
gesture training), Wav2Lip (already available behind the existing
`video_production` flag; not needed by this subsystem's default stack),
Hallo3 (long-video research model, heavy VRAM).

## 5. Licenses

* espeak-ng: GPL (local use fine; ship source if you distribute).
* Piper + fasih voice: MIT — commercial-friendly.
* SadTalker: Apache-2.0 — commercial-friendly.
* All rendering code here: project license.

## 6. Configuration

Everything via env vars (see `config.py`) or a YAML file:

```yaml
# config/avatar.yaml
anchor_id: female
tts_backend: auto        # auto|piper|espeak|xtts|placeholder
avatar_backend: auto     # auto|animated|sadtalker
video_fps: 25
gesture_enabled: false   # experimental; system works without it
face_enhancer: ""        # e.g. gfpgan with SadTalker only
```

## 7. CLI

```bash
./venv/bin/python -m avatar_engine.generate \
    --script avatar_engine/tests/test_script.txt \
    --anchor female --output output/news.mp4

./venv/bin/python -m avatar_engine.generate --anchor male --text "السلام علیکم۔"
./venv/bin/python -m avatar_engine.generate --config config/avatar.yaml --script s.txt
```

## 8. Python API

```python
from avatar_engine import generate_anchor_video

report = generate_anchor_video(
    script="آج کی اہم خبر...",        # or script_path="generated/script.txt"
    anchor_id="female",               # or "male"
    output_path="output/news_001.mp4",
)
print(report["success"], report["duration"], report["sync_ok"])
```

Integration with the existing UrduNewsAI pipeline: call
`generate_anchor_video(script_path=<Stage 2.10 full_script>,
anchor_id=...)` between script generation and the existing
`video_production` compositor; the returned report doubles as QC.

## 9. Testing

```bash
./venv/bin/python -m avatar_engine.tests.test_e2e
```

Generates `output/avatar_engine/tests/e2e_female.mp4` and
`e2e_male.mp4` and asserts streams/codecs/A-V sync via FFprobe.

## 10. Troubleshooting

* `ERROR [gpu]` — requested SadTalker without CUDA: install GPU stack
  or keep the default `auto`.
* Mouth looks frozen → QC auto-re-renders with wider jaw scale; if
  still frozen, check the anchor asset has one clear frontal face.
* `FFmpegError` — ensure `ffmpeg`/`ffprobe` are on PATH.
* Urdu renders as boxes in *subtitles* (unrelated stage) — set
  `URDU_FONT_PATH`; avatar_engine itself never rasterizes text.

## 11. Model limitations

* espeak-ng Urdu is formant-based (robotic but intelligible); upgrade
  to Piper fasih for neural quality (male voice only today — female
  uses espeak with raised pitch).
* `animated` backend is 2.5-D (image animation), not a neural
  talking-head; on GPU hosts switch `avatar_backend: sadtalker`.
* Gestures are intentionally OFF by default (no broken hands).

## 12–15. Replacing components

* **Replace TTS:** implement `generate_speech(text, voice_id,
  output_path, config)` semantics in a new backend branch in
  `tts.py::AvatarTTS` (or wrap any engine that writes a WAV) — the
  pipeline only consumes the returned `TTSChunk`.
* **Replace talking head:** subclass `backends/base.py::TalkingHeadBackend`
  and register it in `backends/__init__.py::_REGISTRY`.
* **Enable gestures:** `gesture_enabled: true` once a gesture backend
  is registered; the pipeline runs identically with it disabled.
