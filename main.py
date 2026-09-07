import os
import sys
import argparse
import schedule
import time
import sqlite3
from datetime import datetime

# Load environment variables
env_path = ".env"
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
                
from data_acquisition.utils.logger import get_logger
import data_acquisition.config as acq_cfg
from data_acquisition.database.raw_news_db import RawNewsDatabase
from data_acquisition.orchestrator.data_acquisition_orchestrator import DataAcquisitionOrchestrator

import pipeline.config as pcfg
from pipeline.processed_news_db import ProcessedNewsDatabase
from pipeline.pipeline_orchestrator import NewsProcessingPipeline

logger = get_logger("main")

def run_pipeline():
    logger.info("Starting automated news cycle...")
    
    raw_db = None
    processed_db = None
    try:
        raw_db = RawNewsDatabase(acq_cfg.DEFAULT_DB_PATH)
        sources_cfg = acq_cfg.load_sources_config(path=str(acq_cfg.SOURCES_CONFIG_PATH))
        
        # 1. Data Acquisition
        acq_orchestrator = DataAcquisitionOrchestrator(sources_config=sources_cfg, db=raw_db)
        acq_orchestrator.run_all()
        acq_orchestrator.close()
        
        # 2. Content Pipeline (Processing, Video, Distribution)
        processed_db = ProcessedNewsDatabase(pcfg.PROCESSED_DB_PATH)
        pipeline = NewsProcessingPipeline(raw_db, processed_db)
        
        report = pipeline.run_all()
        logger.info(f"News cycle completed at {report.finished_at}")
        
    except Exception as e:
        logger.error(f"Error during automated news cycle: {e}")
    finally:
        if raw_db:
            try:
                raw_db.close()
            except:
                pass
        if processed_db:
            try:
                processed_db.close()
            except:
                pass

def get_status():
    print("=== Pipeline Status ===")
    
    if not os.path.exists(acq_cfg.DEFAULT_DB_PATH):
        print("Database not found. System has not run yet.")
        return
        
    try:
        with sqlite3.connect(acq_cfg.DEFAULT_DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM raw_news")
            print(f"Total raw articles acquired: {cur.fetchone()[0]}")
            
            cur.execute("SELECT COUNT(*) FROM raw_news WHERE status='raw'")
            print(f"Pending raw articles: {cur.fetchone()[0]}")
    except Exception as e:
        print(f"Error reading raw DB: {e}")
        
    try:
        with sqlite3.connect(pcfg.PROCESSED_DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM distribution_jobs")
            print(f"Total distribution jobs: {cur.fetchone()[0]}")
            
            cur.execute("SELECT COUNT(*) FROM distribution_jobs WHERE status='completed'")
            print(f"Completed distributions: {cur.fetchone()[0]}")
    except Exception as e:
        print(f"Error reading processed DB: {e}")

def main():
    parser = argparse.ArgumentParser(description="AI Automated News Channel")
    parser.add_argument("--run-now", action="store_true", help="Run the pipeline immediately")
    parser.add_argument("--scheduler", action="store_true", help="Start the 4-hour schedule loop")
    parser.add_argument("--status", action="store_true", help="Print the current database status")
    
    args = parser.parse_args()
    
    if args.status:
        get_status()
    elif args.run_now:
        run_pipeline()
    elif args.scheduler:
        logger.info("Starting scheduler (every 4 hours)")
        # Schedule every 4 hours
        schedule.every(4).hours.do(run_pipeline)
        
        while True:
            schedule.run_pending()
            time.sleep(60)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
