import sys
import sqlite3
import os
sys.path.insert(0, ".")

# Load .env manually
env_path = ".env"
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

import static_ffmpeg
static_ffmpeg.add_paths()

import pipeline.config as pcfg
import data_acquisition.config as acq_cfg
from data_acquisition.database.raw_news_db import RawNewsDatabase
from pipeline.processed_news_db import ProcessedNewsDatabase
from pipeline.pipeline_orchestrator import NewsProcessingPipeline
from data_acquisition.orchestrator.data_acquisition_orchestrator import DataAcquisitionOrchestrator
from tts.tts_engine import clean_text_for_tts

def run_bulletin():
    print("=== 1. Running Data Acquisition ===")
    raw_db = RawNewsDatabase(acq_cfg.DEFAULT_DB_PATH)
    sources_cfg = acq_cfg.load_sources_config(path=str(acq_cfg.SOURCES_CONFIG_PATH))
    # Run only RSS for speed and reliability of news updates
    for k in list(sources_cfg.keys()):
        if k != "rss_feeds":
            if isinstance(sources_cfg[k], dict):
                sources_cfg[k]["enabled"] = False
            else:
                sources_cfg[k] = []
    
    orchestrator = DataAcquisitionOrchestrator(sources_config=sources_cfg, db=raw_db)
    orchestrator.run_all()
    orchestrator.close()

    print("=== 2. Running Content Pipeline ===")
    processed_db = ProcessedNewsDatabase(pcfg.PROCESSED_DB_PATH)
    pipeline = NewsProcessingPipeline(raw_db, processed_db)
    
    pipeline.run_preprocessing()
    pipeline.run_classification()
    pipeline.run_duplicate_detection()
    pipeline.run_extraction()
    pipeline.run_summarization()
    pipeline.run_translation()
    pipeline.run_bias_detection()
    pipeline.run_fact_verification()
    pipeline.run_script_generation()
    pipeline.run_script_structuring()
    
    # 3. Aggregate 5 News items
    structured_map = processed_db.get_structured_map()
    
    # Get top 5 primary articles
    rows = pipeline._valid_primary_rows()
    if not rows:
        print("No articles found to generate bulletin.")
        return
        
    top_rows = rows[:5]
    
    print(f"=== 3. Building Bulletin with {len(top_rows)} articles ===")
    
    greeting = "آغاز: السلام علیکم ناظرین، میں آپ کی میزبان ثناء ہوں اور آپ دیکھ رہے ہیں آج کی اہم خبریں۔"
    
    headlines = []
    stories = []
    
    for i, r in enumerate(top_rows):
        aid = r["article_id"]
        s_data = structured_map.get(aid, {})
        headline = s_data.get("headline", "")
        if not headline:
            headline = r["title"]
        intro = s_data.get("intro", "")
        main_story = s_data.get("main_story", "")
        
        headlines.append(headline)
        
        # Combine intro and main_story for the detailed section
        story_text = intro
        if main_story:
            story_text += " " + main_story
            
        stories.append(f"خبر نمبر {i+1}: {story_text}")
        
    headlines_text = "ہیڈلائن: " + "۔ ".join(headlines) + "۔"
    details_text = "مرکزی خبر: " + "\n\n".join(stories)
    cta = "اختتامیہ: مزید خبروں اور تازہ ترین اپڈیٹس کے لیے ہمارے چینل سے جڑے رہیں۔"
    
    combined_script = f"{greeting}\n\n{headlines_text}\n\n{details_text}\n\n{cta}"
    
    # Clean text to ensure proper length and TTS
    combined_script_clean = clean_text_for_tts(combined_script)
    
    # Limit D-ID text length if necessary (optional, but keep it just in case)
    # limit = 2000 
    # combined_script_clean = combined_script_clean[:limit]

    print("Combined Script Length:", len(combined_script_clean))
    
    bulletin_id = 9999
    
    # Insert dummy row into preprocessed_news so it appears in final dataset
    with processed_db._cursor() as cur:
        cur.execute(
            """
            INSERT INTO preprocessed_news (article_id, title, preprocessing_status) 
            VALUES (?, ?, ?)
            ON CONFLICT(article_id) DO UPDATE SET title=excluded.title
            """,
            (bulletin_id, "5-News Bulletin", "ok")
        )
        
        # Insert structured script
        cur.execute(
            """
            INSERT INTO structured_scripts (article_id, full_script, status)
            VALUES (?, ?, ?)
            ON CONFLICT(article_id) DO UPDATE SET full_script=excluded.full_script
            """,
            (bulletin_id, combined_script_clean, "ok")
        )
        
    # Hack pipeline to only process bulletin_id for TTS and video
    _orig = pipeline._valid_primary_rows
    pipeline._valid_primary_rows = lambda: [{"article_id": bulletin_id, "title": "5-News Bulletin", "language": "ur"}]
    
    print("=== 4. Generating TTS ===")
    print(pipeline.run_tts())
    
    print("=== 5. Post-Processing Audio ===")
    print(pipeline.run_audio_postprocessing())
    
    print("=== 6. Generating Video ===")
    
    # Pre-download images for each story so they can be mapped in the timeline
    import requests
    from pathlib import Path
    
    media_dir = Path("data_acquisition/data/media/images")
    media_dir.mkdir(parents=True, exist_ok=True)
    
    timeline_segments = []
    
    # We construct the timeline by mapping each story text to its image
    for i, r in enumerate(top_rows):
        s_data = structured_map.get(r["article_id"], {})
        story_text = f"خبر نمبر {i+1}: " + s_data.get("intro", "")
        if s_data.get("main_story", ""):
            story_text += " " + s_data.get("main_story", "")
            
        # Get media relevance image_url
        img_url = ""
        with processed_db._cursor() as cur:
            cur.execute("SELECT image_url FROM media_relevance WHERE article_id=?", (r["article_id"],))
            res = cur.fetchone()
            if res and res["image_url"]:
                img_url = res["image_url"]
                
        local_img = ""
        if img_url:
            local_img = str(media_dir / f"story_{r['article_id']}.jpg")
            try:
                if not Path(local_img).exists():
                    resp = requests.get(img_url, timeout=10)
                    if resp.status_code == 200:
                        with open(local_img, 'wb') as f:
                            f.write(resp.content)
            except Exception:
                local_img = ""
                
        timeline_segments.append({"text": story_text, "image_path": local_img})
        
    # We must also prepend the intro/headline and append the outro to the timeline to keep it perfectly synced
    timeline_segments.insert(0, {"text": greeting + " " + headlines_text, "image_path": ""})
    timeline_segments.append({"text": cta, "image_path": ""})
    
    # Hack for video production
    _orig_build = processed_db.build_final_dataset
    def mock_build():
        ds = _orig_build()
        for d in ds:
            if d["article_id"] == bulletin_id:
                d["timeline_segments"] = timeline_segments
                return [d]
        return []
    processed_db.build_final_dataset = mock_build
    
    print(pipeline.run_video_production(limit=1))
    
    print(f"\nDone! Video: data_acquisition/data/media/video/composition/composed_{bulletin_id}_subtitled.mp4")
    raw_db.close()
    processed_db.close()

if __name__ == "__main__":
    run_bulletin()
