"""
summarization/summarizer.py
==============================
Stage 2.5 -- English Summarization.

Extractive TextRank summarization (same ranking utility as Stage 2.4's
keypoint extraction -- see `extraction/text_rank.py`): selects the
highest-centrality sentences and presents them as a short paragraph, a
key-points list, and a bullet-point summary.

Extractive (verbatim sentences from the article) rather than abstractive
on purpose: it structurally cannot hallucinate facts that aren't in the
source (Requirement: "not hallucinate information", "not introduce facts
that are absent from the article"), and needs no downloaded generation
model to run in this environment.

Only articles whose detected language is English are summarized here, per
the task's Stage 2.5 scope ("English") -- non-English articles are marked
`status="skipped_non_english"` rather than mistranslated or dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from data_acquisition.utils.logger import get_logger

from extraction.text_rank import top_sentences_in_order

logger = get_logger("pipeline.summarization")

DEFAULT_SUMMARY_SENTENCES = 2
DEFAULT_KEYPOINT_COUNT = 4


@dataclass
class SummaryResult:
    article_id: int
    summary: str = ""
    key_points: List[str] = field(default_factory=list)
    bullet_summary: List[str] = field(default_factory=list)
    status: str = "ok"   # ok | skipped_non_english | empty_text | error


class Summarizer:
    def __init__(
        self,
        summary_sentence_count: int = DEFAULT_SUMMARY_SENTENCES,
        keypoint_count: int = DEFAULT_KEYPOINT_COUNT,
    ):
        self.summary_sentence_count = summary_sentence_count
        self.keypoint_count = keypoint_count

    def summarize(self, article_id: int, sentences: List[str], language: str) -> SummaryResult:
        try:
            if language != "en":
                return SummaryResult(article_id, status="skipped_non_english")
            if not sentences:
                return SummaryResult(article_id, status="empty_text")

            # Very short articles (this dataset's RSS teasers are often
            # 1-2 sentences): the "summary" is just the article itself --
            # nothing shorter would remain faithful.
            if len(sentences) <= self.summary_sentence_count:
                summary_sentences = sentences
            else:
                summary_sentences = top_sentences_in_order(sentences, self.summary_sentence_count)

            summary = " ".join(summary_sentences)
            key_points = top_sentences_in_order(sentences, min(self.keypoint_count, len(sentences)))
            bullets = list(key_points)

            return SummaryResult(article_id, summary, key_points, bullets, "ok")
        except Exception as exc:
            logger.error("Summarization failed for article_id=%s: %s", article_id, exc)
            return SummaryResult(article_id, status="error")

    def run(self, articles: List[dict]) -> List[SummaryResult]:
        """`articles`: list of {"article_id", "sentences", "language"}."""
        results = [
            self.summarize(a["article_id"], a.get("sentences") or [], a.get("language", "unknown"))
            for a in articles
        ]
        logger.info("Summarization complete for %d articles", len(results))
        return results
