"""
_regen_broadcast.py
===================
Re-runs stages 2.10 → 3.6 for article_id=1 using the real stored DB text
so the anchor gets a proper greeting + full Urdu news from the database.

Usage:
    source venv/bin/activate && python _regen_broadcast.py
"""

import sys, sqlite3
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

import pipeline.config as pcfg
import data_acquisition.config as acq_cfg
from data_acquisition.database.raw_news_db import RawNewsDatabase
from pipeline.processed_news_db import ProcessedNewsDatabase
from pipeline.pipeline_orchestrator import NewsProcessingPipeline

# ── Load the pipeline ──────────────────────────────────────────────────────
raw = RawNewsDatabase(acq_cfg.DEFAULT_DB_PATH)
processed = ProcessedNewsDatabase(pcfg.PROCESSED_DB_PATH)
pipeline = NewsProcessingPipeline(raw, processed)

# ── Restrict every stage to article_id=1 only ─────────────────────────────
_orig_rows = pipeline._valid_primary_rows
pipeline._valid_primary_rows = lambda: [r for r in _orig_rows() if r["article_id"] == 1]

# ── Show what script text we currently have in the DB ─────────────────────
conn = sqlite3.connect(str(pcfg.PROCESSED_DB_PATH))
cur = conn.cursor()
cur.execute("SELECT full_script FROM structured_scripts WHERE article_id=1")
row = cur.fetchone()
current_script = row[0] if row else "(none)"
print("=== CURRENT STORED SCRIPT ===")
print(current_script[:1000])
print("=" * 60)

# ── Re-run Stage 2.10 (re-structure with greeting) ───────────────────────
print("\nRunning Stage 2.10: Script Structuring (with greeting)…")
r210 = pipeline.run_script_structuring()
print(r210)

# ── Verify the new script ────────────────────────────────────────────────
cur.execute("SELECT full_script FROM structured_scripts WHERE article_id=1")
new_row = cur.fetchone()
new_script = new_row[0] if new_row else "(none)"
conn.close()
print("\n=== NEW STRUCTURED SCRIPT ===")
print(new_script[:1500])
print("=" * 60)

# ── Measure character count for quota check ───────────────────────────────
from tts.tts_engine import clean_text_for_tts
clean = clean_text_for_tts(new_script)
print(f"\nClean TTS text ({len(clean)} chars):")
print(clean[:600])
print()

FREE_TIER_QUOTA = 10000   # total monthly
REMAINING = 493           # credits left
if len(clean) > REMAINING:
    print(f"⚠  Text is {len(clean)} chars — exceeds remaining quota ({REMAINING}).")
    print("   Truncating to first 450 chars to stay within limit…")
    # Update DB with truncated clean text so TTS only sees what fits
    conn2 = sqlite3.connect(str(pcfg.PROCESSED_DB_PATH))
    conn2.execute(
        "UPDATE structured_scripts SET full_script=? WHERE article_id=1",
        (clean[:450],)
    )
    conn2.commit()
    conn2.close()
    print("DB updated with truncated script.")
else:
    print(f"✓  Text fits within remaining quota ({len(clean)}/{REMAINING} chars).")

# ── Stage 2.11: TTS via ElevenLabs ───────────────────────────────────────
print("\nRunning Stage 2.11: TTS (ElevenLabs, Sarah voice)…")
r211 = pipeline.run_tts()
print(r211)

# ── Stage 2.12: Audio post-processing ────────────────────────────────────
print("\nRunning Stage 2.12: Audio Post-Processing…")
r212 = pipeline.run_audio_postprocessing()
print(r212)

# ── Stages 3.1-3.6: Full Video Production ────────────────────────────────
print("\nRunning Stages 3.1-3.6: Video Production…")
r3 = pipeline.run_video_production(limit=1)
print(r3)

print("\n✅ Done — composed video:")
print("  data_acquisition/data/media/video/composition/composed_1_subtitled.mp4")

raw.close()
processed.close()
