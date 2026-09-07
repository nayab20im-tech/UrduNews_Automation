"""
pipeline/config.py
=====================
Configurable settings for Stages 2.1-2.10 (Requirement 12: "make
thresholds, model names, paths, and other important settings
configurable rather than hard-coded"). Mirrors the style of
`data_acquisition/config.py` -- everything overridable via environment
variables, sensible defaults otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path

from data_acquisition import config as acquisition_config

# By default the pipeline reads/writes alongside the existing raw DB, in
# the same `data/` folder the acquisition layer already uses.
PROCESSED_DB_PATH = Path(
    os.getenv("PROCESSED_DB_PATH", str(acquisition_config.DATA_DIR / "raw_news.db"))
)
FINAL_DATASET_EXPORT_PATH = Path(
    os.getenv("FINAL_DATASET_EXPORT_PATH", str(acquisition_config.DATA_DIR / "processed_news_export.json"))
)

# Stage 2.1
MIN_CONTENT_CHARS = int(os.getenv("MIN_CONTENT_CHARS", "5"))

# Stage 2.2
CLASSIFICATION_CONFIDENCE_THRESHOLD = float(os.getenv("CLASSIFICATION_CONFIDENCE_THRESHOLD", "0.08"))

# Stage 2.3
DUPLICATE_SIMILARITY_THRESHOLD = float(os.getenv("DUPLICATE_SIMILARITY_THRESHOLD", "0.65"))

# Stage 2.4
KEYPOINT_COUNT = int(os.getenv("KEYPOINT_COUNT", "3"))
MAX_CLAIMS = int(os.getenv("MAX_CLAIMS", "5"))

# Stage 2.5
SUMMARY_SENTENCE_COUNT = int(os.getenv("SUMMARY_SENTENCE_COUNT", "8"))
SUMMARY_KEYPOINT_COUNT = int(os.getenv("SUMMARY_KEYPOINT_COUNT", "6"))

# Stage 2.6 -- translation backend: "dictionary" (offline default) or
# "llm" (a local OpenAI-compatible server, e.g. Ollama / llama.cpp).
TRANSLATION_BACKEND = os.getenv("TRANSLATION_BACKEND", "dictionary")
LOCAL_LLM_ENDPOINT = os.getenv("LOCAL_LLM_ENDPOINT", "")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "")
REFINER_MAX_SENTENCE_WORDS = int(os.getenv("REFINER_MAX_SENTENCE_WORDS", "35"))

# Stage 2.7
BIAS_SENSATIONAL_THRESHOLD = float(os.getenv("BIAS_SENSATIONAL_THRESHOLD", "0.4"))
BIAS_MILD_THRESHOLD = float(os.getenv("BIAS_MILD_THRESHOLD", "0.15"))

# Stage 2.8
EVIDENCE_SIMILARITY_THRESHOLD = float(os.getenv("EVIDENCE_SIMILARITY_THRESHOLD", "0.15"))
MIN_CORROBORATING_SOURCES = int(os.getenv("MIN_CORROBORATING_SOURCES", "2"))
DEFAULT_SOURCE_CREDIBILITY = float(os.getenv("DEFAULT_SOURCE_CREDIBILITY", "0.5"))
TRUSTED_SOURCES_JSON = os.getenv("TRUSTED_SOURCES_JSON", "")

# Stage 2.9
SCRIPT_BACKEND = os.getenv("SCRIPT_BACKEND", "template")

# Stage 2.10
SCRIPT_GREETING_TEXT = os.getenv(
    "SCRIPT_GREETING_TEXT", "السلام علیکم ناظرین، خبروں کے ساتھ ہم حاضر ہیں۔"
)
SCRIPT_CTA_TEXT = os.getenv(
    "SCRIPT_CTA_TEXT", "مزید خبروں اور تازہ ترین اپڈیٹس کے لیے ہمارے چینل سے جڑے رہیں۔"
)

# Stage 2.11 -- TTS backend: "auto" (xtts -> piper -> espeak -> placeholder),
# or pin one explicitly. Placeholder synth keeps the pipeline fully offline
# on hosts without any TTS engine; the backend used is stored per article.
TTS_BACKEND = os.getenv("TTS_BACKEND", "auto")
TTS_VOICE = os.getenv("TTS_VOICE", "ur")
TTS_SPEED = int(os.getenv("TTS_SPEED", "160"))     # espeak words-per-minute
TTS_PITCH = int(os.getenv("TTS_PITCH", "50"))      # espeak pitch 0-99
PIPER_MODEL_PATH = os.getenv("PIPER_MODEL_PATH", "")
XTTS_MODEL_PATH = os.getenv("XTTS_MODEL_PATH", "")
XTTS_SPEAKER_WAV = os.getenv("XTTS_SPEAKER_WAV", "")
AUDIO_DIR = Path(os.getenv("AUDIO_DIR", str(acquisition_config.DATA_DIR / "media" / "audio")))

# Stage 2.12
PROCESSED_AUDIO_DIR = Path(
    os.getenv("PROCESSED_AUDIO_DIR", str(acquisition_config.DATA_DIR / "media" / "audio_processed"))
)
AUDIO_TARGET_PEAK = float(os.getenv("AUDIO_TARGET_PEAK", "0.85"))
AUDIO_NOISE_GATE = float(os.getenv("AUDIO_NOISE_GATE", "0.01"))
AUDIO_MAX_SILENCE_MS = int(os.getenv("AUDIO_MAX_SILENCE_MS", "700"))
AUDIO_TRIM_SILENCE = os.getenv("AUDIO_TRIM_SILENCE", "true").lower() in ("1", "true", "yes")
BACKGROUND_MUSIC_PATH = os.getenv("BACKGROUND_MUSIC_PATH", "")
BACKGROUND_MUSIC_GAIN = float(os.getenv("BACKGROUND_MUSIC_GAIN", "0.08"))
