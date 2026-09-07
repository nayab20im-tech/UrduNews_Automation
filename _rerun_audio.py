import sys
sys.path.insert(0, ".")
from pipeline.processed_news_db import ProcessedNewsDatabase
from data_acquisition.database.raw_news_db import RawNewsDatabase
from pipeline.pipeline_orchestrator import NewsProcessingPipeline
import data_acquisition.config as acq_cfg
import pipeline.config as pcfg
from dotenv import load_dotenv
import sqlite3

load_dotenv()

conn = sqlite3.connect(pcfg.PROCESSED_DB_PATH)
cursor = conn.cursor()
cursor.execute("UPDATE structured_scripts SET full_script = ? WHERE article_id = 1", ("ناظرین، یہ ایک مختصر خبر ہے جو اردو میں تیار کی گئی ہے۔ شکریہ۔",))
conn.commit()
conn.close()
print("Updated article 1 script to short text to respect Elevenlabs quota")


raw = RawNewsDatabase(acq_cfg.DEFAULT_DB_PATH)
processed = ProcessedNewsDatabase(pcfg.PROCESSED_DB_PATH)
pipeline = NewsProcessingPipeline(raw, processed)

old_func = pipeline._valid_primary_rows
pipeline._valid_primary_rows = lambda: [r for r in old_func() if r["article_id"] == 1]

print("Running TTS...")
report1 = pipeline.run_tts()
print(report1)

print("Running Audio PostProcessing...")
report2 = pipeline.run_audio_postprocessing()
print(report2)

print("Running Video Production...")
report3 = pipeline.run_video_production(limit=1)
print(report3)

print("Done!")
raw.close()
processed.close()
