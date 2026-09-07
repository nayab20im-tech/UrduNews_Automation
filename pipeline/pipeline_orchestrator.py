"""
pipeline/pipeline_orchestrator.py
====================================
Wires Stages 2.1 -> 2.12 -> 3.6 -> 4.4 together, each stage's output becoming the next
stage's input:

    RawNewsDatabase (status="raw")
        -> 2.1 Preprocessor            -> preprocessed_news
        -> 2.2 NewsClassifier          -> classified_news
        -> 2.3 DuplicateDetector       -> duplicate_status
        -> 2.4 InfoExtractor           -> extracted_info      (primary articles only)
        -> 2.5 Summarizer              -> summaries           (primary articles only)
        -> 2.6 UrduTranslator          -> translations        (primary articles only)
        -> 2.7 BiasDetector            -> bias_analysis       (primary articles only)
        -> 2.8 FactChecker             -> verification        (primary articles only)
        -> 2.9 ScriptGenerator         -> news_scripts        (primary articles only)
        -> 2.10 ScriptStructurer       -> structured_scripts  (primary articles only)
        -> 2.11 TTSEngine              -> tts_audio           (primary articles only)
        -> 2.12 AudioPostProcessor     -> processed_audio     (primary articles only)
        -> 3.1-3.6 VideoProduction     -> video outputs       (primary articles only)
        -> 4.1-4.4 Distribution        -> distribution jobs   (primary articles only)

Every stage is independently callable (`run_preprocessing()`, etc.) as
well as chainable via `run_all()`, and every stage tolerates individual
article failures without stopping the batch (each `*.run()` already
does this -- see the per-stage modules).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from data_acquisition.database.raw_news_db import RawNewsDatabase
from data_acquisition.utils.logger import get_logger
from data_acquisition.utils.text_utils import safe_iso_now

from classification.classifier import NewsClassifier
from duplicate_detection.duplicate_detector import DuplicateDetector
from extraction.extractor import InfoExtractor
from pipeline.media_matcher import MediaMatcher
from preprocessing.preprocessor import Preprocessor
from summarization.summarizer import Summarizer
from translation.refiner import UrduRefiner
from translation.translator import UrduTranslator
from bias_detection.bias_detector import BiasDetector
from verification.fact_checker import FactChecker
from script_generation.script_generator import ScriptGenerator
from script_generation.script_structurer import ScriptStructurer
from tts.tts_engine import TTSEngine
from audio_processing.post_processor import AudioPostProcessor
from video_production.pipeline import run_video_production
from distribution.pipeline import run_distribution as _run_distribution
from distribution.distribution_db import DistributionDatabase

from . import config as pcfg
from .processed_news_db import ProcessedNewsDatabase

logger = get_logger("pipeline.orchestrator")


@dataclass
class StageReport:
    stage: str
    processed: int = 0
    ok: int = 0
    skipped_or_error: int = 0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineReport:
    started_at: str
    finished_at: str = ""
    stages: List[StageReport] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "stages": [
                {
                    "stage": s.stage,
                    "processed": s.processed,
                    "ok": s.ok,
                    "skipped_or_error": s.skipped_or_error,
                    "details": s.details,
                }
                for s in self.stages
            ],
        }


class NewsProcessingPipeline:
    def __init__(
        self,
        raw_db: RawNewsDatabase,
        processed_db: ProcessedNewsDatabase,
        min_content_chars: int = pcfg.MIN_CONTENT_CHARS,
        classification_confidence_threshold: float = pcfg.CLASSIFICATION_CONFIDENCE_THRESHOLD,
        duplicate_similarity_threshold: float = pcfg.DUPLICATE_SIMILARITY_THRESHOLD,
        keypoint_count: int = pcfg.KEYPOINT_COUNT,
        max_claims: int = pcfg.MAX_CLAIMS,
        summary_sentence_count: int = pcfg.SUMMARY_SENTENCE_COUNT,
        summary_keypoint_count: int = pcfg.SUMMARY_KEYPOINT_COUNT,
    ):
        self.raw_db = raw_db
        self.processed_db = processed_db

        self.preprocessor = Preprocessor(min_content_chars=min_content_chars)
        self.classifier = NewsClassifier(confidence_threshold=classification_confidence_threshold)
        self.duplicate_detector = DuplicateDetector(similarity_threshold=duplicate_similarity_threshold)
        self.media_matcher = MediaMatcher()
        self.extractor = InfoExtractor()
        self.summarizer = Summarizer(
            summary_sentence_count=summary_sentence_count, keypoint_count=summary_keypoint_count
        )

        # Stages 2.6-2.10 share one translator instance (same config).
        self.translator = UrduTranslator(
            backend=pcfg.TRANSLATION_BACKEND,
            llm_endpoint=pcfg.LOCAL_LLM_ENDPOINT,
            llm_model=pcfg.LOCAL_LLM_MODEL,
            refiner=UrduRefiner(max_sentence_words=pcfg.REFINER_MAX_SENTENCE_WORDS),
        )
        self.bias_detector = BiasDetector(
            sensational_threshold=pcfg.BIAS_SENSATIONAL_THRESHOLD,
            mild_threshold=pcfg.BIAS_MILD_THRESHOLD,
        )
        self.fact_checker = FactChecker(
            evidence_similarity_threshold=pcfg.EVIDENCE_SIMILARITY_THRESHOLD,
            min_corroborating_sources=pcfg.MIN_CORROBORATING_SOURCES,
            trusted_sources=_load_trusted_sources(),
            default_credibility=pcfg.DEFAULT_SOURCE_CREDIBILITY,
        )
        self.script_generator = ScriptGenerator(
            translator=self.translator,
            backend=pcfg.SCRIPT_BACKEND,
            llm_endpoint=pcfg.LOCAL_LLM_ENDPOINT,
            llm_model=pcfg.LOCAL_LLM_MODEL,
        )
        self.script_structurer = ScriptStructurer(
            greeting_text=pcfg.SCRIPT_GREETING_TEXT,
            cta_text=pcfg.SCRIPT_CTA_TEXT, 
            translator=self.translator
        )
        self.tts_engine = TTSEngine(
            audio_dir=pcfg.AUDIO_DIR,
            backend=pcfg.TTS_BACKEND,
            voice=pcfg.TTS_VOICE,
            speed=pcfg.TTS_SPEED,
            pitch=pcfg.TTS_PITCH,
            piper_model_path=pcfg.PIPER_MODEL_PATH,
            xtts_model_path=pcfg.XTTS_MODEL_PATH,
            xtts_speaker_wav=pcfg.XTTS_SPEAKER_WAV,
        )
        self.audio_post_processor = AudioPostProcessor(
            out_dir=pcfg.PROCESSED_AUDIO_DIR,
            target_peak=pcfg.AUDIO_TARGET_PEAK,
            noise_gate=pcfg.AUDIO_NOISE_GATE,
            max_silence_ms=pcfg.AUDIO_MAX_SILENCE_MS,
            trim_silence=pcfg.AUDIO_TRIM_SILENCE,
            bgm_path=pcfg.BACKGROUND_MUSIC_PATH,
            bgm_gain=pcfg.BACKGROUND_MUSIC_GAIN,
        )

    # -- Stage 2.1 ------------------------------------------------------
    def run_preprocessing(self, limit: Optional[int] = None) -> StageReport:
        raw_rows = self.raw_db.get_all(status="raw", limit=limit)
        results = self.preprocessor.run(raw_rows)

        now = safe_iso_now()
        status_updates: Dict[str, str] = {
            "ok": "preprocessed", "empty_content": "preprocessed_empty",
            "duplicate": "duplicate_raw", "error": "preprocess_error",
        }
        for rec in results:
            self.processed_db.save_preprocessed(rec, now)
            self.raw_db.update_status(rec.article_id, status_updates.get(rec.preprocessing_status, "preprocess_error"))

        ok = sum(1 for r in results if r.preprocessing_status == "ok")
        return StageReport(
            stage="2.1_preprocessing", processed=len(results), ok=ok, skipped_or_error=len(results) - ok,
            details={"by_status": _count_by(results, "preprocessing_status")},
        )

    # -- Stage 2.2 --------------------------------------------------------
    def run_classification(self) -> StageReport:
        rows = self.processed_db.get_all_preprocessed(status="ok")
        articles = [{"article_id": r["article_id"], "text": r["processed_text"] or r["cleaned_text"]} for r in rows]
        results = self.classifier.run(articles)

        now = safe_iso_now()
        for rec in results:
            self.processed_db.save_classification(rec, now)

        ok = sum(1 for r in results if r.classification_status == "ok")
        return StageReport(
            stage="2.2_classification", processed=len(results), ok=ok, skipped_or_error=len(results) - ok,
            details={"by_category": _count_by(results, "category")},
        )

    # -- Stage 2.3 ----------------------------------------------------------
    def run_duplicate_detection(self) -> StageReport:
        rows = self.processed_db.get_all_preprocessed(status="ok")
        articles = [
            {"article_id": r["article_id"], "title": r["title"], "text": r["cleaned_text"]} for r in rows
        ]
        results = self.duplicate_detector.detect(articles)

        now = safe_iso_now()
        for rec in results:
            self.processed_db.save_duplicate_status(rec, now)

        dup_count = sum(1 for r in results if r.is_duplicate)
        return StageReport(
            stage="2.3_duplicate_detection", processed=len(results), ok=len(results) - dup_count,
            skipped_or_error=dup_count,
            details={"groups": len({r.duplicate_group_id for r in results if r.duplicate_group_id})},
        )

    # -- Stage 2.3.5: Media Matching ----------------------------------------
    def run_media_matching(self) -> StageReport:
        # Match media against all newly scraped valid articles
        rows = self.raw_db.get_all() # Get raw records to access extra_json where URLs are stored
        # Filter for rows that made it past preprocessing ok
        prep_rows = {r["article_id"]: r for r in self.processed_db.get_all_preprocessed(status="ok")}
        
        articles = [{"id": r["id"], **r} for r in rows if r["id"] in prep_rows]
        results = self.media_matcher.run(articles)
        
        now = safe_iso_now()
        for rec in results:
            self.processed_db.save_media_relevance(rec, now)
            
        ok = sum(1 for r in results if r.status == "ok")
        return StageReport(
            stage="2.3.5_media_matching", processed=len(results), ok=ok, skipped_or_error=len(results) - ok,
            details={"by_status": _count_by(results, "status")}
        )

    # -- Stage 2.4 -------------------------------------------------------------
    def run_extraction(self) -> StageReport:
        rows = self._valid_primary_rows()
        articles = [
            {"article_id": r["article_id"], "text": r["cleaned_text"], "sentences": _load_sentences(r)}
            for r in rows
        ]
        results = self.extractor.run(articles)

        now = safe_iso_now()
        for rec in results:
            self.processed_db.save_extraction(rec, now)

        ok = sum(1 for r in results if r.status == "ok")
        return StageReport(
            stage="2.4_extraction", processed=len(results), ok=ok, skipped_or_error=len(results) - ok,
        )

    # -- Stage 2.5 ------------------------------------------------------------
    def run_summarization(self) -> StageReport:
        rows = self._valid_primary_rows()
        articles = [
            {"article_id": r["article_id"], "sentences": _load_sentences(r), "language": r["language"]}
            for r in rows
        ]
        results = self.summarizer.run(articles)

        now = safe_iso_now()
        for rec in results:
            self.processed_db.save_summary(rec, now)

        ok = sum(1 for r in results if r.status == "ok")
        return StageReport(
            stage="2.5_summarization", processed=len(results), ok=ok, skipped_or_error=len(results) - ok,
            details={"by_status": _count_by(results, "status")},
        )

    def _valid_primary_rows(self) -> List[Dict[str, Any]]:
        """Rows that are preprocessed OK and not a Stage-2.3 duplicate."""
        rows = self.processed_db.get_all_preprocessed(status="ok")
        dup_map = self.processed_db.get_duplicate_map()
        return [r for r in rows if not dup_map.get(r["article_id"], {}).get("is_duplicate")]

    # -- Stage 2.6 -------------------------------------------------------------
    def run_translation(self) -> StageReport:
        rows = self._valid_primary_rows()
        articles = [
            {"article_id": r["article_id"], "title": r["title"],
             "text": r["cleaned_text"], "language": r["language"]}
            for r in rows
        ]
        results = self.translator.run(articles)

        now = safe_iso_now()
        for rec in results:
            self.processed_db.save_translation(rec, now)

        ok = sum(1 for r in results if r.status == "ok")
        return StageReport(
            stage="2.6_translation", processed=len(results), ok=ok, skipped_or_error=len(results) - ok,
            details={"translated": sum(1 for r in results if r.translated),
                     "passthrough": sum(1 for r in results if r.status == "ok" and not r.translated)},
        )

    # -- Stage 2.7 -------------------------------------------------------------
    def run_bias_detection(self) -> StageReport:
        rows = self._valid_primary_rows()
        articles = [
            {"article_id": r["article_id"], "title": r["title"], "text": r["cleaned_text"]}
            for r in rows
        ]
        results = self.bias_detector.run(articles)

        now = safe_iso_now()
        for rec in results:
            self.processed_db.save_bias_analysis(rec, now)

        ok = sum(1 for r in results if r.status == "ok")
        return StageReport(
            stage="2.7_bias_detection", processed=len(results), ok=ok, skipped_or_error=len(results) - ok,
            details={"by_label": _count_by(results, "label")},
        )

    # -- Stage 2.8 -------------------------------------------------------------
    def run_fact_verification(self) -> StageReport:
        rows = self._valid_primary_rows()
        extraction_map = self.processed_db.get_extraction_map()
        articles = [
            {"article_id": r["article_id"], "source": r["source"], "title": r["title"],
             "text": r["cleaned_text"],
             "claims": extraction_map.get(r["article_id"], {}).get("claims", [])}
            for r in rows
        ]
        results = self.fact_checker.verify(articles)

        now = safe_iso_now()
        for rec in results:
            self.processed_db.save_verification(rec, now)

        ok = sum(1 for r in results if r.status in ("ok", "no_claims"))
        return StageReport(
            stage="2.8_fact_verification", processed=len(results), ok=ok, skipped_or_error=len(results) - ok,
            details={"by_verdict": _count_by(results, "verdict")},
        )

    # -- Stage 2.9 -------------------------------------------------------------
    def run_script_generation(self) -> StageReport:
        rows = self._valid_primary_rows()
        translation_map = self.processed_db.get_translation_map()
        verification_map = self.processed_db.get_verification_map()
        summary_map = self.processed_db.get_summary_map()
        articles = []
        for r in rows:
            aid = r["article_id"]
            summary = summary_map.get(aid, {})
            articles.append(
                {
                    "article_id": aid,
                    "language": r["language"],
                    "sentences": _load_sentences(r),
                    "urdu_sentences": translation_map.get(aid, {}).get("urdu_sentences", []),
                    "verification": verification_map.get(aid),
                    "summary": summary.get("summary", ""),
                    "key_points": summary.get("key_points", []),
                }
            )
        results = self.script_generator.run(articles)

        now = safe_iso_now()
        for rec in results:
            self.processed_db.save_script(rec, now)

        ok = sum(1 for r in results if r.status == "ok")
        return StageReport(
            stage="2.9_script_generation", processed=len(results), ok=ok, skipped_or_error=len(results) - ok,
            details={"by_backend": _count_by(results, "backend_used")},
        )

    # -- Stage 2.10 ------------------------------------------------------------
    def run_script_structuring(self) -> StageReport:
        rows = self._valid_primary_rows()
        translation_map = self.processed_db.get_translation_map()
        script_map = self.processed_db.get_script_map()
        extraction_map = self.processed_db.get_extraction_map()
        summary_map = self.processed_db.get_summary_map()
        articles = []
        for r in rows:
            aid = r["article_id"]
            keypoints = extraction_map.get(aid, {}).get("keypoints") or \
                summary_map.get(aid, {}).get("key_points", [])
            articles.append(
                {
                    "article_id": aid,
                    "title": r["title"],
                    "urdu_title": translation_map.get(aid, {}).get("urdu_title", ""),
                    "script_text": script_map.get(aid, {}).get("script_text", ""),
                    "key_points": keypoints,
                    "language": r["language"],
                }
            )
        results = self.script_structurer.run(articles)

        now = safe_iso_now()
        for rec in results:
            self.processed_db.save_structured_script(rec, now)

        ok = sum(1 for r in results if r.status == "ok")
        return StageReport(
            stage="2.10_script_structuring", processed=len(results), ok=ok, skipped_or_error=len(results) - ok,
        )

    # -- Stage 2.11 ------------------------------------------------------------
    def run_tts(self) -> StageReport:
        rows = self._valid_primary_rows()
        structured_map = self.processed_db.get_structured_map()
        articles = [
            {"article_id": r["article_id"],
             "text": structured_map.get(r["article_id"], {}).get("full_script", "")}
            for r in rows
        ]
        results = self.tts_engine.run(articles)

        now = safe_iso_now()
        for rec in results:
            self.processed_db.save_tts_audio(rec, now)

        ok = sum(1 for r in results if r.status == "ok")
        return StageReport(
            stage="2.11_tts", processed=len(results), ok=ok, skipped_or_error=len(results) - ok,
            details={"by_backend": _count_by(results, "backend_used")},
        )

    # -- Stage 2.12 ------------------------------------------------------------
    def run_audio_postprocessing(self) -> StageReport:
        rows = self._valid_primary_rows()
        tts_map = self.processed_db.get_tts_map()
        audios = [
            {"article_id": r["article_id"],
             "audio_path": tts_map.get(r["article_id"], {}).get("audio_path", "")}
            for r in rows
            if tts_map.get(r["article_id"], {}).get("status") == "ok"
        ]
        results = self.audio_post_processor.run(audios)

        now = safe_iso_now()
        for rec in results:
            self.processed_db.save_processed_audio(rec, now)

        ok = sum(1 for r in results if r.status == "ok")
        return StageReport(
            stage="2.12_audio_postprocessing", processed=len(results), ok=ok,
            skipped_or_error=len(results) - ok,
            details={"bgm_applied": sum(1 for r in results if r.bgm_applied),
                     "silence_removed_ms": sum(r.silence_removed_ms for r in results)},
        )

    # -- Section 3: Video Production ----------------------------------------
    def run_video_production(self, limit: Optional[int] = None) -> StageReport:
        """Run Section 3 (Stages 3.1-3.6) on processed articles with audio."""
        dataset = self.processed_db.build_final_dataset()
        report = run_video_production(dataset, limit=limit)

        total_ok = sum(
            1 for o in report.outputs if o.get("subtitled_video")
        )
        total_processed = len(report.outputs)

        # Aggregate details from sub-stages
        details = {}
        for s in report.stages:
            details[s.stage] = {"ok": s.ok, "error": s.skipped_or_error}

        return StageReport(
            stage="3_video_production",
            processed=total_processed,
            ok=total_ok,
            skipped_or_error=total_processed - total_ok,
            details=details,
        )

    # -- Section 4: Distribution & Automation ----------------------------
    def run_distribution(self, limit: Optional[int] = None) -> StageReport:
        """Run Section 4 (Stages 4.1-4.4) on articles with video output."""
        dataset = self.processed_db.build_final_dataset()
        report = _run_distribution(dataset, limit=limit)

        total_ok = sum(
            1 for o in report.outputs if o.get("youtube_status") in ("published", "mock_published")
        )
        total_processed = len(report.outputs)

        details = {}
        for s in report.stages:
            details[s.stage] = {"ok": s.ok, "error": s.skipped_or_error}

        return StageReport(
            stage="4_distribution",
            processed=total_processed,
            ok=total_ok,
            skipped_or_error=total_processed - total_ok,
            details=details,
        )

    # -- full pipeline ------------------------------------------------------
    def run_all(self, limit: Optional[int] = None) -> PipelineReport:
        report = PipelineReport(started_at=safe_iso_now())
        report.stages.append(self.run_preprocessing(limit=limit))
        report.stages.append(self.run_classification())
        report.stages.append(self.run_duplicate_detection())
        report.stages.append(self.run_media_matching())
        report.stages.append(self.run_extraction())
        report.stages.append(self.run_summarization())
        report.stages.append(self.run_translation())
        report.stages.append(self.run_bias_detection())
        report.stages.append(self.run_fact_verification())
        report.stages.append(self.run_script_generation())
        report.stages.append(self.run_script_structuring())
        report.stages.append(self.run_tts())
        report.stages.append(self.run_audio_postprocessing())
        report.stages.append(self.run_video_production(limit=limit))
        report.stages.append(self.run_distribution(limit=limit))
        report.finished_at = safe_iso_now()
        logger.info("Pipeline run complete.")
        return report

    def export_final_dataset(self, path) -> int:
        import json

        dataset = self.processed_db.build_final_dataset()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        logger.info("Exported final processed dataset (%d articles) to %s", len(dataset), path)
        return len(dataset)


def _count_by(records: List[Any], attr: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in records:
        key = str(getattr(r, attr))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _load_trusted_sources() -> Optional[Dict[str, float]]:
    """Parse TRUSTED_SOURCES_JSON (e.g. '{"bbc": 0.9}'); None => defaults."""
    import json

    raw = (pcfg.TRUSTED_SOURCES_JSON or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except ValueError:
        logger.warning("TRUSTED_SOURCES_JSON is not valid JSON; using defaults")
        return None


def _load_sentences(row: Dict[str, Any]) -> List[str]:
    import json

    raw = row.get("sentences_json") or row.get("sentences")
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return []
