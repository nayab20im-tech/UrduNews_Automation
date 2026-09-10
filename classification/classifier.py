"""
classification/classifier.py
===============================
Stage 2.2 -- News Classification.

No labeled training data exists in this project yet, so downloading or
training a full ML classifier isn't appropriate (Requirement: "prefer
lightweight and practical approaches if no trained classifier already
exists" / "avoid unnecessarily downloading huge models"). Instead this
uses a TF-IDF-over-keyword-profiles similarity classifier:

  1. Each of the 7 categories gets a "profile document" built from a
     bilingual (English + Urdu) keyword lexicon (`category_keywords.py`).
  2. A single `TfidfVectorizer` is fit across the 7 profiles.
  3. Each article's `processed_text` (stopwords removed -- see Stage 2.1)
     is transformed into the same TF-IDF space and compared to every
     category profile with cosine similarity.
  4. The highest-scoring category wins; below `confidence_threshold` the
     article is assigned "Others" instead of a low-confidence guess.

This is intentionally modular (`NewsClassifier.classify_text()` takes a
plain string in, returns a category + score out) so it's a drop-in
replacement point for a properly trained/fine-tuned classifier later --
nothing else in the pipeline needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data_acquisition.utils.logger import get_logger

from .category_keywords import CATEGORIES, CATEGORY_KEYWORDS

logger = get_logger("pipeline.classification")

_OTHERS = "Others"


@dataclass
class ClassificationResult:
    article_id: int
    category: str
    confidence: float
    classification_status: str = "ok"   # ok | low_confidence | empty_text | error


class NewsClassifier:
    def __init__(self, confidence_threshold: float = 0.08):
        self.confidence_threshold = confidence_threshold
        self._categories: List[str] = list(CATEGORIES)
        profiles = [" ".join(CATEGORY_KEYWORDS[c]) for c in self._categories]
        # word_ngrams over unicode \w tokens covers both English words and
        # Urdu script tokens with the same default token pattern.
        self._vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 1), min_df=1)
        self._profile_matrix = self._vectorizer.fit_transform(profiles)

    def classify_text(self, text: str) -> tuple[str, float]:
        """Return (category, confidence) for a single piece of text."""
        if not text or not text.strip():
            return _OTHERS, 0.0

        vec = self._vectorizer.transform([text])
        if vec.nnz == 0:
            # None of the article's tokens are in any category vocabulary.
            return _OTHERS, 0.0

        sims = cosine_similarity(vec, self._profile_matrix)[0]
        best_idx = int(sims.argmax())
        best_score = float(sims[best_idx])
        category = self._categories[best_idx]

        if category == _OTHERS:
            return _OTHERS, best_score
        if best_score < self.confidence_threshold:
            return _OTHERS, best_score
        return category, best_score

    def classify(self, article_id: int, text: str) -> ClassificationResult:
        try:
            if not text or not text.strip():
                return ClassificationResult(article_id, _OTHERS, 0.0, "empty_text")
            category, score = self.classify_text(text)
            status = "ok" if category != _OTHERS or score > 0 else "low_confidence"
            return ClassificationResult(article_id, category, round(score, 4), status)
        except Exception as exc:
            logger.error("Classification failed for article_id=%s: %s", article_id, exc)
            return ClassificationResult(article_id, _OTHERS, 0.0, "error")

    def run(self, articles: List[Dict]) -> List[ClassificationResult]:
        """`articles`: list of {"article_id": int, "text": str}."""
        results = [self.classify(a["article_id"], a["text"]) for a in articles]
        logger.info("Classification complete for %d articles", len(results))
        return results
