"""
extraction/extractor.py
==========================
Stage 2.4 orchestrator -- combines NER, keypoint extraction, and claim
identification into one `InfoExtractionResult` per article.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from data_acquisition.utils.logger import get_logger

from .claim_extractor import extract_claims
from .keypoint_extractor import extract_keypoints
from .ner import NERExtractor

logger = get_logger("pipeline.extraction")


@dataclass
class InfoExtractionResult:
    article_id: int
    entities: Dict[str, List[str]] = field(default_factory=dict)
    keypoints: List[str] = field(default_factory=list)
    claims: List[str] = field(default_factory=list)
    status: str = "ok"   # ok | empty_text | error


class InfoExtractor:
    def __init__(self):
        self._ner = NERExtractor()

    def extract(self, article_id: int, text: str, sentences: List[str]) -> InfoExtractionResult:
        try:
            if not text or not text.strip():
                return InfoExtractionResult(article_id, {}, [], [], "empty_text")

            entities = self._ner.extract(text)
            keypoints = extract_keypoints(sentences)
            claims = extract_claims(sentences)
            return InfoExtractionResult(article_id, entities, keypoints, claims, "ok")
        except Exception as exc:
            logger.error("Extraction failed for article_id=%s: %s", article_id, exc)
            return InfoExtractionResult(article_id, {}, [], [], "error")

    def run(self, articles: List[Dict]) -> List[InfoExtractionResult]:
        """`articles`: list of {"article_id", "text", "sentences"}."""
        results = [self.extract(a["article_id"], a["text"], a.get("sentences") or []) for a in articles]
        logger.info("Extraction complete for %d articles", len(results))
        return results
