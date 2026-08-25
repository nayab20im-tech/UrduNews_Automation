"""
pipeline/tests/test_pipeline.py
==================================
Offline test suite for Stages 2.1-2.5. No network access required --
everything runs against inline fixture strings and a temporary SQLite
file, mirroring the offline-testability convention of
`data_acquisition/tests/test_data_acquisition.py`.

Run with:  pytest pipeline/tests -v   (from the project root, alongside
data_acquisition/)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from classification.classifier import NewsClassifier
from duplicate_detection.duplicate_detector import DuplicateDetector
from extraction.claim_extractor import extract_claims
from extraction.extractor import InfoExtractor
from extraction.keypoint_extractor import extract_keypoints
from extraction.ner import NERExtractor
from preprocessing.html_cleaner import remove_ads, strip_html
from preprocessing.normalizer import normalize_text, remove_stopwords, split_sentences
from preprocessing.preprocessor import Preprocessor
from summarization.summarizer import Summarizer


# ---------------------------------------------------------------------------
# Stage 2.1 -- preprocessing
# ---------------------------------------------------------------------------
class TestHtmlCleaner:
    def test_strip_html_removes_tags(self):
        # BeautifulSoup's get_text(separator="\n") inserts a newline between
        # block-level tag boundaries -- normalize whitespace before comparing,
        # since that collapsing is normalizer.py's job, not html_cleaner's.
        result = " ".join(strip_html("<p>Hello <b>world</b></p>").split())
        assert result == "Hello world"

    def test_strip_html_decodes_entities(self):
        assert "&" in strip_html("Tom &amp; Jerry")

    def test_strip_html_handles_none_and_plain_text(self):
        assert strip_html(None) == ""
        assert strip_html("plain text") == "plain text"

    def test_remove_ads_drops_ad_lines_keeps_content(self):
        text = "Real headline sentence.\nAdvertisement\nSubscribe to our newsletter.\nMore real content here."
        cleaned = remove_ads(text)
        assert "Advertisement" not in cleaned
        assert "Subscribe" not in cleaned
        assert "Real headline sentence." in cleaned
        assert "More real content here." in cleaned


class TestNormalizer:
    def test_normalize_collapses_whitespace(self):
        assert normalize_text("hello    world") == "hello world"

    def test_normalize_smart_quotes(self):
        result = normalize_text("\u201cHello\u201d said the \u2018man\u2019")
        assert '"' in result and "'" in result

    def test_normalize_empty(self):
        assert normalize_text("") == ""
        assert normalize_text(None) == ""

    def test_split_sentences_english(self):
        sents = split_sentences("This is one. This is two! Is this three?")
        assert len(sents) == 3

    def test_split_sentences_urdu(self):
        sents = split_sentences("یہ پہلا جملہ ہے۔ یہ دوسرا جملہ ہے۔")
        assert len(sents) == 2

    def test_split_sentences_empty(self):
        assert split_sentences("") == []
        assert split_sentences(None) == []

    def test_remove_stopwords_english(self):
        result = remove_stopwords("this is a test of the system", "en")
        assert "this" not in result.split()
        assert "test" in result.split()

    def test_remove_stopwords_preserves_urdu_content_words(self):
        result = remove_stopwords("یہ ایک ٹیسٹ ہے", "ur")
        assert "ٹیسٹ" in result


class TestPreprocessor:
    def _row(self, **overrides):
        base = {
            "id": 1, "title": "<b>Breaking</b> News", "content": "<p>Something happened today.</p>",
            "source": "test", "url": "http://example.com/1", "published_date": "2026-01-01",
            "language": "unknown", "raw_html": None,
        }
        base.update(overrides)
        return base

    def test_process_article_ok(self):
        pre = Preprocessor()
        result = pre.process_article(self._row())
        assert result.preprocessing_status == "ok"
        assert "<b>" not in result.title
        assert result.language == "en"
        assert len(result.sentences) >= 1

    def test_process_article_missing_title(self):
        pre = Preprocessor()
        result = pre.process_article(self._row(title="", content="Some content."))
        # empty title alone is fine as long as content survives
        assert result.preprocessing_status in ("ok", "empty_content")

    def test_process_article_missing_body(self):
        pre = Preprocessor()
        result = pre.process_article(self._row(content=""))
        assert result.preprocessing_status in ("ok", "empty_content")

    def test_process_article_both_empty(self):
        pre = Preprocessor()
        result = pre.process_article(self._row(title="", content=""))
        assert result.preprocessing_status == "empty_content"

    def test_process_article_exact_duplicate_detected(self):
        pre = Preprocessor()
        first = pre.process_article(self._row(id=1))
        second = pre.process_article(self._row(id=2))
        assert first.preprocessing_status == "ok"
        assert second.preprocessing_status == "duplicate"

    def test_process_article_very_long_content(self):
        pre = Preprocessor()
        long_content = "This is a sentence. " * 500
        result = pre.process_article(self._row(content=long_content))
        assert result.preprocessing_status == "ok"
        assert len(result.sentences) > 100

    def test_process_article_malformed_text_does_not_crash(self):
        pre = Preprocessor()
        weird = "\x00\x01<<<>>>\ufeff   \n\n\n  ###???"
        result = pre.process_article(self._row(title=weird, content=weird))
        assert result.preprocessing_status in ("ok", "empty_content", "error")

    def test_run_tolerates_one_bad_row(self):
        pre = Preprocessor()
        rows = [self._row(id=1), {"id": 2}]  # second row is missing every field
        results = pre.run(rows)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Stage 2.2 -- classification
# ---------------------------------------------------------------------------
class TestClassifier:
    def test_classifies_sports_text(self):
        clf = NewsClassifier()
        category, score = clf.classify_text(
            "The cricket team won the championship final after a thrilling match with the winning wicket."
        )
        assert category == "Sports"

    def test_classifies_business_text(self):
        clf = NewsClassifier()
        category, _ = clf.classify_text(
            "The stock market rallied today as investors reacted to the company's quarterly revenue and profit report."
        )
        assert category == "Business"

    def test_empty_text_is_others(self):
        clf = NewsClassifier()
        category, score = clf.classify_text("")
        assert category == "Others"
        assert score == 0.0

    def test_unrelated_text_falls_back_to_others(self):
        clf = NewsClassifier()
        category, _ = clf.classify_text("xyzzy plugh qwerty foobar")
        assert category == "Others"

    def test_run_never_raises_on_bad_input(self):
        clf = NewsClassifier()
        results = clf.run([{"article_id": 1, "text": None}, {"article_id": 2, "text": "sports match today"}])
        assert len(results) == 2
        assert all(r.classification_status in ("ok", "empty_text", "low_confidence", "error") for r in results)


# ---------------------------------------------------------------------------
# Stage 2.3 -- duplicate detection
# ---------------------------------------------------------------------------
class TestDuplicateDetector:
    def test_near_duplicate_articles_grouped(self):
        dd = DuplicateDetector(similarity_threshold=0.5)
        articles = [
            {"article_id": 1, "title": "PM announces new budget", "text": "The prime minister announced a new national budget today with tax cuts."},
            {"article_id": 2, "title": "Budget announced by PM", "text": "The prime minister today announced a new national budget featuring tax cuts."},
            {"article_id": 3, "title": "Local team wins match", "text": "The local football team won their match yesterday in a thrilling finish."},
        ]
        results = dd.detect(articles)
        by_id = {r.article_id: r for r in results}
        assert by_id[1].duplicate_group_id == by_id[2].duplicate_group_id
        assert by_id[1].duplicate_group_id != by_id[3].duplicate_group_id
        assert sum(1 for r in results if r.is_primary) == 2  # one per group (2 groups)

    def test_different_articles_not_grouped(self):
        dd = DuplicateDetector(similarity_threshold=0.65)
        articles = [
            {"article_id": 1, "title": "Cricket match today", "text": "A cricket team won a big match with an excellent innings."},
            {"article_id": 2, "title": "Stock market update", "text": "The stock market saw gains today as investors bought shares."},
        ]
        results = dd.detect(articles)
        assert all(not r.is_duplicate for r in results)

    def test_empty_and_single_article_lists(self):
        dd = DuplicateDetector()
        assert dd.detect([]) == []
        results = dd.detect([{"article_id": 1, "title": "A", "text": "Some text here."}])
        assert len(results) == 1
        assert results[0].is_primary is True


# ---------------------------------------------------------------------------
# Stage 2.4 -- extraction
# ---------------------------------------------------------------------------
class TestNER:
    def test_extracts_date_and_money(self):
        ner = NERExtractor()
        entities = ner.extract("The deal, worth $5 million, was signed on 12 August 2026.")
        assert entities["MONEY"]
        assert entities["DATE"]

    def test_extracts_person_with_title(self):
        ner = NERExtractor()
        entities = ner.extract("President John Smith met with officials today.")
        assert any("John Smith" in p for p in entities["PERSON"])

    def test_empty_text_returns_empty_lists(self):
        ner = NERExtractor()
        entities = ner.extract("")
        assert all(v == [] for v in entities.values())


class TestKeypointAndClaims:
    def test_keypoints_returns_subset(self):
        sentences = [f"This is sentence number {i} about the ongoing story." for i in range(10)]
        points = extract_keypoints(sentences, max_points=3)
        assert len(points) == 3
        assert all(p in sentences for p in points)

    def test_claims_detects_reporting_verbs_and_numbers(self):
        sentences = [
            "The weather was nice.",
            "The minister said the plan would cost 5 million dollars.",
            "According to officials, the project starts in March.",
        ]
        claims = extract_claims(sentences)
        assert len(claims) == 2

    def test_claims_never_invents_text(self):
        sentences = ["The minister said 100 people attended."]
        claims = extract_claims(sentences)
        assert claims == sentences


class TestInfoExtractor:
    def test_run_tolerates_empty_text(self):
        extractor = InfoExtractor()
        results = extractor.run([{"article_id": 1, "text": "", "sentences": []}])
        assert results[0].status == "empty_text"


# ---------------------------------------------------------------------------
# Stage 2.5 -- summarization
# ---------------------------------------------------------------------------
class TestSummarizer:
    def test_summarizes_long_english_article(self):
        summarizer = Summarizer(summary_sentence_count=2)
        sentences = [
            "The government announced a new policy today.",
            "Officials said the policy would take effect next month.",
            "Critics argue the policy does not go far enough.",
            "Supporters say it is an important first step.",
            "The debate is expected to continue in parliament.",
        ]
        result = summarizer.summarize(1, sentences, "en")
        assert result.status == "ok"
        assert len(result.summary) > 0
        assert len(result.summary) < sum(len(s) for s in sentences)
        assert len(result.bullet_summary) > 0

    def test_short_article_returns_itself(self):
        summarizer = Summarizer(summary_sentence_count=2)
        sentences = ["Just one short sentence."]
        result = summarizer.summarize(1, sentences, "en")
        assert result.summary == "Just one short sentence."

    def test_non_english_is_skipped(self):
        summarizer = Summarizer()
        result = summarizer.summarize(1, ["ایک جملہ۔"], "ur")
        assert result.status == "skipped_non_english"

    def test_empty_sentences(self):
        summarizer = Summarizer()
        result = summarizer.summarize(1, [], "en")
        assert result.status == "empty_text"
