"""Run the full video production pipeline on 1 article."""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, ".")

import json
from pipeline.processed_news_db import ProcessedNewsDatabase
from pipeline import config as pcfg
from video_production.pipeline import run_video_production
from video_production import config as vc

db = ProcessedNewsDatabase(db_path=pcfg.PROCESSED_DB_PATH)
try:
    dataset = db.build_final_dataset()
    print(f"Articles: {len(dataset)}")
    print("Running full pipeline (stages 3.1-3.6)...")
    print()
    report = run_video_production(dataset, limit=1)
    print()
    print("=== REPORT ===")
    print(json.dumps(report.as_dict(), indent=2, ensure_ascii=False))
finally:
    db.close()
