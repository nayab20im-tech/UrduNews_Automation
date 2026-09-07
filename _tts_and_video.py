"""
_tts_and_video.py
=================
Directly regenerates audio + video for article 1 with exactly 350-char script.
Run: source venv/bin/activate && python _tts_and_video.py
"""
import sys, sqlite3
sys.path.insert(0, ".")

# Load .env manually
import os
env_path = ".env"
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

import pipeline.config as pcfg
import data_acquisition.config as acq_cfg
from data_acquisition.database.raw_news_db import RawNewsDatabase
from pipeline.processed_news_db import ProcessedNewsDatabase
from pipeline.pipeline_orchestrator import NewsProcessingPipeline
from tts.tts_engine import clean_text_for_tts

# Read current DB script
conn = sqlite3.connect(str(pcfg.PROCESSED_DB_PATH))
row = conn.execute("SELECT full_script FROM structured_scripts WHERE article_id=1").fetchone()
conn.close()
full = row[0] if row else ""

clean = clean_text_for_tts(full)
print(f"Clean text: {len(clean)} chars")

# Intelligent truncation — keep greeting + headline + first news sentence only
# Urdu sentences end with ۔
LIMIT = 340
if len(clean) > LIMIT:
    truncated = clean[:LIMIT].rsplit("۔", 1)[0] + "۔"
else:
    truncated = clean
print(f"Using {len(truncated)} chars:")
print(truncated)
print()

# Write to DB
conn2 = sqlite3.connect(str(pcfg.PROCESSED_DB_PATH))
conn2.execute("UPDATE structured_scripts SET full_script=? WHERE article_id=1", (truncated,))
conn2.commit()
conn2.close()

# Run pipeline
raw = RawNewsDatabase(acq_cfg.DEFAULT_DB_PATH)
processed = ProcessedNewsDatabase(pcfg.PROCESSED_DB_PATH)
pipeline = NewsProcessingPipeline(raw, processed)

_orig = pipeline._valid_primary_rows
pipeline._valid_primary_rows = lambda: [r for r in _orig() if r["article_id"] == 1]

print("Stage 2.11: TTS…")
print(pipeline.run_tts())

print("Stage 2.12: Audio PostProcessing…")
print(pipeline.run_audio_postprocessing())

print("Stages 3.1-3.6: Video Production…")
print(pipeline.run_video_production(limit=1))

print("\nDone! Video: data_acquisition/data/media/video/composition/composed_1_subtitled.mp4")
raw.close()
processed.close()
