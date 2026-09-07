"""
orchestration/config_manager.py
===============================
Implements Section 5's "Config Manager" box: one place that exposes the
three config families the diagram calls out --

- Sources Config   -> ``data_acquisition`` (sources_config.yaml + .env)
- Model Config     -> ``pipeline`` / ``video_production`` (TTS, LLM,
                      avatar / lip-sync backends and model paths)
- Category Config  -> ``classification`` keywords + pipeline thresholds

``effective()`` merges them into one auditable view (secrets excluded --
only "configured" booleans for credential-backed services) and
``snapshot()`` persists it through Section 6's Config & Model Storage.
"""

from __future__ import annotations

from typing import Any, Dict

from data_acquisition.utils.logger import get_logger

logger = get_logger("orchestration.config_manager")


class ConfigManager:
    """Read-only aggregator over every per-module config module."""

    # -- Sources Config ------------------------------------------------------
    def sources_config(self) -> Dict[str, Any]:
        from data_acquisition import config as acq_cfg

        cfg = acq_cfg.load_sources_config()
        return {
            "rss_feeds": len(cfg.get("rss_feeds", [])),
            "websites": len(cfg.get("websites", [])),
            "news_api_enabled": bool(cfg.get("news_api", {}).get("enabled")),
            "google_trends_enabled": bool(cfg.get("google_trends", {}).get("enabled")),
            "social_media_enabled": bool(cfg.get("social_media", {}).get("enabled")),
            "manual_input_enabled": bool(cfg.get("manual_input", {}).get("enabled")),
            "rss_feed_names": [f.get("name", "") for f in cfg.get("rss_feeds", [])],
            "website_names": [w.get("name", "") for w in cfg.get("websites", [])],
        }

    # -- Model Config -----------------------------------------------------------
    def model_config(self) -> Dict[str, Any]:
        from pipeline import config as pcfg
        from video_production import config as vcfg

        return {
            "translation_backend": pcfg.TRANSLATION_BACKEND,
            "script_backend": pcfg.SCRIPT_BACKEND,
            "local_llm_model": pcfg.LOCAL_LLM_MODEL,
            "local_llm_configured": bool(pcfg.LOCAL_LLM_ENDPOINT),
            "tts_backend": pcfg.TTS_BACKEND,
            "tts_voice": pcfg.TTS_VOICE,
            "piper_model_path": pcfg.PIPER_MODEL_PATH,
            "xtts_model_path": pcfg.XTTS_MODEL_PATH,
            "avatar_backend": vcfg.AVATAR_BACKEND,
            "lipsync_backend": vcfg.LIPSYNC_BACKEND,
            "sadtalker_checkpoint_dir": vcfg.SADTALKER_CHECKPOINT_DIR,
            "wav2lip_checkpoint": vcfg.WAV2LIP_CHECKPOINT,
            "ffmpeg_binary": vcfg.FFMPEG_BINARY,
            "urdu_font_path": vcfg.URDU_FONT_PATH,
            "has_cuda": vcfg.HAS_CUDA,
        }

    # -- Category Config -----------------------------------------------------------
    def category_config(self) -> Dict[str, Any]:
        from classification.category_keywords import CATEGORIES, CATEGORY_KEYWORDS
        from pipeline import config as pcfg

        return {
            "categories": list(CATEGORIES),
            "keyword_counts": {c: len(CATEGORY_KEYWORDS.get(c, [])) for c in CATEGORIES},
            "classification_confidence_threshold": pcfg.CLASSIFICATION_CONFIDENCE_THRESHOLD,
            "duplicate_similarity_threshold": pcfg.DUPLICATE_SIMILARITY_THRESHOLD,
            "bias_sensational_threshold": pcfg.BIAS_SENSATIONAL_THRESHOLD,
        }

    # -- merged view -----------------------------------------------------------
    def effective(self) -> Dict[str, Any]:
        """One auditable view of the whole system. Never contains secrets:
        credential-backed services are exposed as booleans only."""
        from distribution import config as dc

        return {
            "sources": self.sources_config(),
            "models": self.model_config(),
            "categories": self.category_config(),
            "distribution": {
                "youtube_configured": dc.is_youtube_configured(),
                "facebook_configured": dc.is_facebook_configured(),
                "twitter_configured": dc.is_twitter_configured(),
                "telegram_configured": dc.is_telegram_configured(),
                "whatsapp_configured": dc.is_whatsapp_configured(),
                "mock_mode": dc.DISTRIBUTION_MOCK_MODE,
            },
        }

    # -- persistence via Section 6 ---------------------------------------------
    def snapshot(self, storage=None) -> Dict[str, Any]:
        """Persist the effective config into Section 6 (SQLite + JSON file)."""
        payload = self.effective()
        if storage is None:
            from storage.storage_manager import StorageManager

            with StorageManager() as sm:
                sm.config_models.save_config("effective_config", payload)
                path = sm.config_models.snapshot_to_file("effective_config", payload)
        else:
            storage.config_models.save_config("effective_config", payload)
            path = storage.config_models.snapshot_to_file("effective_config", payload)
        logger.info("Effective config snapshot stored (%s)", path)
        return {"stored": True, "snapshot_path": str(path), "config": payload}
