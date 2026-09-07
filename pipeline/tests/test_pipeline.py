"""
pipeline/tests/test_pipeline.py
==================================
Offline test suite for Stages 2.1-2.10. No network access required --
everything runs against inline fixture strings and a temporary SQLite
file, mirroring the offline-testability convention of
`data_acquisition/tests/test_data_acquisition.py`.

Run with:  pytest pipeline/tests -v   (from the project root, alongside
data_acquisition/)
"""

from __future__ import annotations

import array
import math
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
from translation.transliterator import transliterate, transliterate_word
from translation.refiner import UrduRefiner
from translation.translator import UrduTranslator
from bias_detection.bias_detector import BiasDetector
from verification.fact_checker import FactChecker
from script_generation.script_generator import ScriptGenerator
from script_generation.script_structurer import ScriptStructurer
from tts.tts_engine import TTSEngine, clean_text_for_tts
from tts import wav_synth
from audio_processing.post_processor import AudioPostProcessor, _smooth_gate
from audio_processing.wav_io import read_wav, write_wav


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


# ---------------------------------------------------------------------------
# Stage 2.6 -- translation & Urdu refinement
# ---------------------------------------------------------------------------
class TestTransliterator:
    def test_known_name_shapes(self):
        assert transliterate_word("khan") == "خان"
        assert transliterate_word("shah") == "شاہ"

    def test_output_is_urdu_script_and_deterministic(self):
        out1 = transliterate("Bilawal Bhutto")
        out2 = transliterate("Bilawal Bhutto")
        assert out1 == out2
        assert any("؀" <= ch <= "ۿ" for ch in out1)

    def test_empty(self):
        assert transliterate("") == ""


class TestTranslator:
    def test_translates_news_vocabulary(self):
        t = UrduTranslator()
        out = t.translate_sentence("The government announced a new policy today.")
        assert "حکومت" in out
        assert "آج" in out

    def test_said_pattern_becomes_urdu_reporting_syntax(self):
        t = UrduTranslator()
        out = t.translate_sentence("The minister said the economy is growing.")
        assert "نے کہا کہ" in out

    def test_phrase_map_wins_over_word_map(self):
        t = UrduTranslator()
        out = t.translate_sentence("The prime minister visited China.")
        assert "وزیرِ اعظم" in out
        assert "چین" in out

    def test_urdu_passthrough_not_translated(self):
        t = UrduTranslator()
        result = t.translate_article(1, "عنوان", "یہ پہلا جملہ ہے۔ یہ دوسرا جملہ ہے۔", "ur")
        assert result.status == "ok"
        assert result.translated is False
        assert len(result.urdu_sentences) == 2

    def test_empty_input(self):
        t = UrduTranslator()
        result = t.translate_article(1, "", "", "en")
        assert result.status == "empty_text"

    def test_run_tolerates_bad_rows(self):
        t = UrduTranslator()
        results = t.run([{"article_id": 1}, {"article_id": 2, "title": "News today", "text": "The police arrested two people.", "language": "en"}])
        assert len(results) == 2


class TestRefiner:
    def test_terminal_punctuation_added(self):
        refined, notes = UrduRefiner().refine("حکومت نے اعلان کیا")
        assert refined.endswith("۔")
        assert "terminal_punctuation_added" in notes

    def test_duplicate_token_collapsed(self):
        refined, notes = UrduRefiner().refine("کہ کہ حکومت آئی۔")
        assert "کہ کہ" not in refined
        assert "duplicate_token_removed" in notes

    def test_register_lift(self):
        refined, _ = UrduRefiner().refine("اس نے بولا کہ مسئلہ حل ہو گیا۔")
        assert "بولا" not in refined
        assert "کہا" in refined

    def test_empty(self):
        refined, notes = UrduRefiner().refine("")
        assert refined == "" and notes == []


# ---------------------------------------------------------------------------
# Stage 2.7 -- bias & sensational language detection
# ---------------------------------------------------------------------------
class TestBiasDetector:
    def test_clickbait_headline_flagged(self):
        rec = BiasDetector().analyze(1, "You Won't Believe What Happened Next", "a shocking story")
        assert rec.clickbait is True
        assert rec.label == "sensational"
        assert rec.flags

    def test_neutral_article_stays_neutral(self):
        rec = BiasDetector().analyze(
            1, "Government announces new policy", "The government announced a new policy today."
        )
        assert rec.clickbait is False
        assert rec.label == "neutral"
        assert rec.bias_score < 0.15

    def test_urdu_sensational_words_detected(self):
        rec = BiasDetector().analyze(1, "دھماکہ خیز انکشاف", "سنسنی خیز خبر سامنے آ گئی۔")
        assert rec.label == "sensational"

    def test_sentiment_scores(self):
        neg = BiasDetector().analyze(1, "Attack", "Two killed in attack, many injured.")
        pos = BiasDetector().analyze(1, "Victory", "Team won the final, a great victory.")
        assert neg.sentiment_score < 0
        assert pos.sentiment_score > 0

    def test_empty_text(self):
        rec = BiasDetector().analyze(1, "", "")
        assert rec.status == "empty_text"


# ---------------------------------------------------------------------------
# Stage 2.8 -- fact verification & evidence retrieval
# ---------------------------------------------------------------------------
class TestFactChecker:
    _CLAIM = "The prime minister announced a new national budget today with tax cuts."

    def _articles(self):
        return [
            {"article_id": 1, "source": "bbc", "title": "Budget announced",
             "text": self._CLAIM + " Markets reacted calmly.", "claims": [self._CLAIM]},
            {"article_id": 2, "source": "reuters", "title": "New budget",
             "text": self._CLAIM + " Analysts expect relief.", "claims": [self._CLAIM]},
            {"article_id": 3, "source": "dawn", "title": "Budget with tax cuts",
             "text": self._CLAIM + " The opposition criticized it.", "claims": []},
            {"article_id": 4, "source": "other", "title": "Cricket",
             "text": "The local team won the match yesterday.", "claims": ["The local team won the match yesterday."]},
        ]

    def test_corroborated_across_sources(self):
        results = FactChecker(evidence_similarity_threshold=0.1).verify(self._articles())
        by_id = {r.article_id: r for r in results}
        assert by_id[1].verdict == "corroborated"
        assert by_id[1].confidence > 0
        assert by_id[1].evidence_count >= 2

    def test_unique_claim_unverified(self):
        results = FactChecker(evidence_similarity_threshold=0.1).verify(self._articles())
        by_id = {r.article_id: r for r in results}
        assert by_id[4].verdict == "unverified"

    def test_no_claims(self):
        results = FactChecker().verify(self._articles())
        by_id = {r.article_id: r for r in results}
        assert by_id[3].verdict == "no_claims"

    def test_source_credibility_lookup(self):
        fc = FactChecker()
        assert fc.source_credibility("BBC") > fc.source_credibility("unknown-blog")

    def test_empty_corpus(self):
        assert FactChecker().verify([]) == []


# ---------------------------------------------------------------------------
# Stage 2.9 -- final Urdu news script generation
# ---------------------------------------------------------------------------
class TestScriptGenerator:
    _SENTENCES = [
        "The government announced a new policy today.",
        "The minister said the policy would take effect next month.",
        "Officials confirmed the plan in a press conference.",
    ]

    def test_generates_urdu_script_from_english(self):
        gen = ScriptGenerator()
        result = gen.generate(1, "en", self._SENTENCES, [])
        assert result.status == "ok"
        assert "حکومت" in result.script_text
        assert result.word_count > 0

    def test_urdu_source_uses_urdu_sentences(self):
        gen = ScriptGenerator()
        ur = ["حکومت نے نئی پالیسی کا اعلان کیا۔", "وزیر نے کہا کہ اگلے ماہ سے نافذ ہو گی۔"]
        result = gen.generate(1, "ur", [], ur)
        assert result.status == "ok"
        assert ur[0] in result.script_text

    def test_tone_down_replaces_hype_words(self):
        toned = ScriptGenerator._tone_down_en("A shocking attack.")
        assert "shocking" not in toned

    def test_unverified_claims_carry_note(self):
        gen = ScriptGenerator()
        result = gen.generate(1, "en", self._SENTENCES, [], verification={"verdict": "unverified"})
        assert result.verification_note
        assert result.verification_note in result.script_text

    def test_empty(self):
        gen = ScriptGenerator()
        assert gen.generate(1, "en", [], []).status == "empty_text"


# ---------------------------------------------------------------------------
# Stage 2.10 -- script structuring for news
# ---------------------------------------------------------------------------
class TestScriptStructurer:
    def test_structures_all_sections(self):
        structurer = ScriptStructurer()
        script = (
            "حکومت نے نئی پالیسی کا اعلان کیا۔ وزیر نے کہا کہ اگلے ماہ سے نافذ ہو گی۔ "
            "حکام نے پریس کانفرنس میں تصدیق کی۔ عوام میں ملا جلا ردعمل سامنے آیا۔"
        )
        result = structurer.structure(
            1, "New policy", "نئی پالیسی کا اعلان", script, ["حکومت نے نئی پالیسی کا اعلان کیا۔"], "ur",
        )
        assert result.status == "ok"
        assert result.headline == "نئی پالیسی کا اعلان"
        assert result.intro and result.main_story
        assert result.key_points
        assert "ہیڈلائن:" in result.full_script
        assert "تعارف:" in result.full_script
        assert "مرکزی خبر:" in result.full_script
        assert "اہم نکات:" in result.full_script
        assert "اختتامیہ:" in result.full_script
        assert result.ending in result.full_script

    def test_english_keypoints_translated(self):
        structurer = ScriptStructurer()
        result = structurer.structure(
            1, "Title", "عنوان", "جملہ ایک۔ جملہ دو۔", ["The government announced a new policy today."], "en",
        )
        assert any("حکومت" in p for p in result.key_points)

    def test_empty_script(self):
        assert ScriptStructurer().structure(1, "t", "ع", "", [], "ur").status == "empty_text"


# ---------------------------------------------------------------------------
# Stage 2.11 -- TTS
# ---------------------------------------------------------------------------
def _tone_samples(frames: int, freq: float, rate: int, amp: int = 8000) -> array.array:
    return array.array("h", (int(amp * math.sin(2 * math.pi * freq * i / rate)) for i in range(frames)))


class TestWavIO:
    def test_roundtrip_16bit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.wav"
            samples = _tone_samples(800, 440.0, 16000)
            write_wav(path, samples, 16000, channels=1)
            wav = read_wav(path)
            assert wav.rate == 16000 and wav.channels == 1
            assert list(wav.samples) == list(samples)
            assert abs(wav.duration_sec - 800 / 16000) < 1e-6


class TestWavSynth:
    def test_synthesizes_valid_wav(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.wav"
            dur = wav_synth.synthesize_script("جملہ ایک۔ جملہ دو یہاں ہے۔", path)
            wav = read_wav(path)
            assert dur == pytest.approx(wav.duration_sec, abs=0.01)
            assert wav.duration_sec > 0.5

    def test_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.wav", Path(tmp) / "b.wav"
            wav_synth.synthesize_script("ایک ہی متن۔", a)
            wav_synth.synthesize_script("ایک ہی متن۔", b)
            assert a.read_bytes() == b.read_bytes()


class TestTTSEngine:
    def test_placeholder_backend_writes_wav(self):
        with tempfile.TemporaryDirectory() as tmp:
            eng = TTSEngine(audio_dir=Path(tmp) / "tts", backend="placeholder")
            rec = eng.synthesize(7, "حکومت نے نئی پالیسی کا اعلان کیا ہے۔")
            assert rec.status == "ok" and rec.backend_used == "placeholder"
            assert Path(rec.audio_path).exists()
            assert rec.duration_sec > 0 and rec.sample_rate > 0

    def test_empty_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = TTSEngine(audio_dir=Path(tmp) / "tts").synthesize(8, "   ")
            assert rec.status == "empty_text" and rec.audio_path == ""

    def test_auto_resolves_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            # No xtts/piper/espeak configured here -> placeholder.
            assert TTSEngine(audio_dir=tmp).resolve_backend() == "placeholder"

    def test_run_tolerates_missing_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            eng = TTSEngine(audio_dir=Path(tmp) / "tts")
            results = eng.run([{"article_id": 1, "text": "جملہ۔"}, {"article_id": 2}])
            assert results[0].status == "ok" and results[1].status == "empty_text"

    def test_labels_only_text_treated_as_empty(self):
        """full_script containing only section labels produces empty_text."""
        with tempfile.TemporaryDirectory() as tmp:
            eng = TTSEngine(audio_dir=Path(tmp) / "tts", backend="placeholder")
            rec = eng.synthesize(99, "ہیڈلائن:\n\nتعارف:\n\nاختتامیہ:")
            assert rec.status == "empty_text"

    def test_script_with_labels_synthesizes_content(self):
        """full_script with labels + real content: labels stripped, content synthesized."""
        with tempfile.TemporaryDirectory() as tmp:
            eng = TTSEngine(audio_dir=Path(tmp) / "tts", backend="placeholder")
            script = "ہیڈلائن: بڑی خبر\n\nتعارف: اہم واقعہ پیش آیا۔\n\nاختتامیہ: شکریہ۔"
            rec = eng.synthesize(42, script)
            assert rec.status == "ok"
            assert rec.duration_sec > 0


class TestCleanTextForTTS:
    def test_strips_urdu_broadcast_labels(self):
        text = "ہیڈلائن: بڑی خبر\n\nتعارف: حکومت نے اعلان کیا۔\n\nمرکزی خبر: تفصیلات جاری۔"
        result = clean_text_for_tts(text)
        assert "ہیڈلائن" not in result
        assert "تعارف" not in result
        assert "مرکزی خبر" not in result
        assert "حکومت" in result
        assert "تفصیلات" in result

    def test_strips_key_points_label(self):
        text = "اہم نکات:\n- پہلا نکتہ۔\n- دوسرا نکتہ۔"
        result = clean_text_for_tts(text)
        assert "اہم نکات" not in result
        assert "پہلا نکتہ" in result

    def test_strips_ending_label(self):
        text = "اختتامیہ: شکریہ۔"
        result = clean_text_for_tts(text)
        assert "اختتامیہ" not in result
        assert "شکریہ" in result

    def test_unicode_nfkc_normalization(self):
        import unicodedata
        # Arabic presentation form that NFKC folds to standard form.
        text = "\ufefb" + " خبر"  # lam-alef ligature + space + word
        result = clean_text_for_tts(text)
        assert unicodedata.is_normalized("NFKC", result)

    def test_whitespace_collapsed(self):
        result = clean_text_for_tts("hello    world\n\n\n  again")
        assert "  " not in result
        assert "\n\n" not in result

    def test_empty_and_none(self):
        assert clean_text_for_tts("") == ""
        assert clean_text_for_tts(None) == ""
        assert clean_text_for_tts("   ") == ""

    def test_plain_text_passes_through(self):
        text = "حکومت نے نئی پالیسی کا اعلان کیا ہے۔"
        assert clean_text_for_tts(text) == text


# ---------------------------------------------------------------------------
# Stage 2.12 -- Audio post-processing
# ---------------------------------------------------------------------------
class TestAudioPostProcessor:
    def _fixture_wav(self, tmp: Path) -> Path:
        """Silence 0.5s + tone 0.5s + long silence 1.5s + tone 0.5s + silence 0.5s."""
        rate = 16000
        samples = array.array("h")
        samples.extend(array.array("h", bytes(int(rate * 0.5) * 2)))
        samples.extend(_tone_samples(int(rate * 0.5), 440.0, rate))
        samples.extend(array.array("h", bytes(int(rate * 1.5) * 2)))
        samples.extend(_tone_samples(int(rate * 0.5), 440.0, rate))
        samples.extend(array.array("h", bytes(int(rate * 0.5) * 2)))
        path = tmp / "in.wav"
        write_wav(path, samples, rate, channels=1)
        return path

    def test_gate_trim_and_normalize(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = self._fixture_wav(Path(tmp))
            pp = AudioPostProcessor(out_dir=Path(tmp) / "proc")
            rec = pp.process(1, src)
            assert rec.status == "ok"
            assert Path(rec.audio_path).exists()
            assert rec.duration_after_sec < rec.duration_before_sec
            assert rec.silence_removed_ms > 500        # long internal pause collapsed
            assert rec.peak_after == pytest.approx(0.85, abs=0.02)
            assert rec.peak_before < rec.peak_after

    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = AudioPostProcessor(out_dir=Path(tmp) / "p").process(1, Path(tmp) / "nope.wav")
            assert rec.status == "missing_file"

    def test_background_music_mixed(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = self._fixture_wav(Path(tmp))
            music = Path(tmp) / "music.wav"
            write_wav(music, _tone_samples(1600, 220.0, 16000, amp=20000), 16000, channels=1)
            pp = AudioPostProcessor(out_dir=Path(tmp) / "proc", bgm_path=str(music))
            rec = pp.process(1, src)
            assert rec.status == "ok" and rec.bgm_applied

    def test_bgm_rate_mismatch_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = self._fixture_wav(Path(tmp))
            music = Path(tmp) / "music.wav"
            write_wav(music, _tone_samples(2205, 220.0, 22050), 22050, channels=1)
            rec = AudioPostProcessor(out_dir=Path(tmp) / "p", bgm_path=str(music)).process(1, src)
            assert rec.status == "ok" and not rec.bgm_applied

    def test_smooth_gate_attenuates_silence(self):
        """Envelope gate should attenuate (not hard-zero) sub-gate samples."""
        rate = 16000
        samples = array.array("h")
        # 0.1s silence + 0.1s tone + 0.1s silence
        samples.extend(array.array("h", bytes(int(rate * 0.1) * 2)))
        samples.extend(_tone_samples(int(rate * 0.1), 440.0, rate, amp=10000))
        samples.extend(array.array("h", bytes(int(rate * 0.1) * 2)))
        gate = int(0.01 * 32767)
        gated = _smooth_gate(samples, gate, rate)
        # Middle section (tone) should have non-zero samples.
        mid_start = int(rate * 0.1) + 50
        mid_end = int(rate * 0.2) - 50
        assert any(gated[i] != 0 for i in range(mid_start, mid_end))
        # Deep silence at the very start should be attenuated to ~0.
        assert all(abs(gated[i]) < 100 for i in range(20))

    def test_run_batch_processing(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i in range(3):
                p = Path(tmp) / f"a{i}.wav"
                write_wav(p, _tone_samples(8000, 440.0, 16000), 16000, channels=1)
                paths.append(p)
            pp = AudioPostProcessor(out_dir=Path(tmp) / "proc")
            results = pp.run([
                {"article_id": i, "audio_path": str(paths[i])}
                for i in range(3)
            ])
            assert len(results) == 3
            assert all(r.status == "ok" for r in results)
            assert all(Path(r.audio_path).exists() for r in results)


# ---------------------------------------------------------------------------
# Full pipeline 2.1 -> 2.12 (end to end, offline)
# ---------------------------------------------------------------------------
class TestFullPipeline:
    def test_run_all_produces_structured_scripts(self, monkeypatch):
        from data_acquisition.database.raw_news_db import Article, RawNewsDatabase
        from pipeline import config as pcfg
        from pipeline.pipeline_orchestrator import NewsProcessingPipeline
        from pipeline.processed_news_db import ProcessedNewsDatabase

        with tempfile.TemporaryDirectory() as tmp:
            # keep audio artifacts inside the temp dir, not the real data/ folder
            monkeypatch.setattr(pcfg, "AUDIO_DIR", Path(tmp) / "audio")
            monkeypatch.setattr(pcfg, "PROCESSED_AUDIO_DIR", Path(tmp) / "audio_proc")

            # keep video artifacts inside the temp dir
            from video_production import config as vcfg
            vid_dir = Path(tmp) / "video"
            monkeypatch.setattr(vcfg, "VIDEO_OUTPUT_DIR", vid_dir)
            monkeypatch.setattr(vcfg, "AVATAR_DIR", vid_dir / "avatar")
            monkeypatch.setattr(vcfg, "LIPSYNC_DIR", vid_dir / "lip_sync")
            monkeypatch.setattr(vcfg, "VISUALS_DIR", vid_dir / "visuals")
            monkeypatch.setattr(vcfg, "COMPOSITION_DIR", vid_dir / "composition")
            monkeypatch.setattr(vcfg, "SUBTITLES_DIR", vid_dir / "subtitles")
            monkeypatch.setattr(vcfg, "THUMBNAILS_DIR", vid_dir / "thumbnails")
            monkeypatch.setattr(vcfg, "FINAL_OUTPUT_DIR", vid_dir / "final")

            # keep distribution artifacts inside the temp dir
            from distribution import config as dcfg
            monkeypatch.setattr(dcfg, "DISTRIBUTION_OUTPUT_DIR", Path(tmp) / "dist")
            monkeypatch.setattr(dcfg, "REPORTS_DIR", Path(tmp) / "dist" / "reports")
            monkeypatch.setattr(dcfg, "DISTRIBUTION_DB_PATH", Path(tmp) / "dist.db")
            monkeypatch.setattr(dcfg, "DISTRIBUTION_MOCK_MODE", True)

            raw_db = RawNewsDatabase(Path(tmp) / "raw.db")
            processed_db = ProcessedNewsDatabase(Path(tmp) / "processed.db")
            raw_db.bulk_insert([
                Article(
                    title="Government announces new policy",
                    content="The government announced a new policy today with tax cuts for small businesses. "
                            "Markets reacted calmly to the decision. "
                            "Analysts expect the policy to ease inflation over time.",
                    url="http://example.com/en1", source="bbc", source_type="rss", language="en",
                ),
                Article(
                    title="New policy announced",
                    content="The government announced a new policy today with tax cuts for small businesses. "
                            "The opposition criticized the timing of the decision. "
                            "Voters will feel the impact next year, observers noted.",
                    url="http://example.com/en2", source="reuters", source_type="rss", language="en",
                ),
                Article(
                    title="عمران خان کی ہسپتال منتقلی",
                    content="پاکستان تحریک انصاف نے درخواست دائر کر دی ہے۔ "
                            "عمران خان کو ہسپتال منتقل کر دیا گیا ہے۔",
                    url="http://example.com/ur1", source="bbc urdu", source_type="rss", language="ur",
                ),
            ])

            pipeline = NewsProcessingPipeline(raw_db=raw_db, processed_db=processed_db)
            report = pipeline.run_all()

            stage_names = [s.stage for s in report.stages]
            assert stage_names == [
                "2.1_preprocessing", "2.2_classification", "2.3_duplicate_detection",
                "2.4_extraction", "2.5_summarization", "2.6_translation",
                "2.7_bias_detection", "2.8_fact_verification", "2.9_script_generation",
                "2.10_script_structuring", "2.11_tts", "2.12_audio_postprocessing",
                "3_video_production", "4_distribution",
            ]

            structured = processed_db._table_map("structured_scripts")
            assert len(structured) == 3
            for row in structured.values():
                assert row["full_script"]
                assert "ہیڈلائن:" in row["full_script"]
                assert "اختتامیہ:" in row["full_script"]

            tts_rows = processed_db._table_map("tts_audio")
            audio_rows = processed_db._table_map("processed_audio")
            assert len(tts_rows) == 3 and len(audio_rows) == 3
            for row in tts_rows.values():
                assert row["status"] == "ok" and Path(row["audio_path"]).exists()
            for row in audio_rows.values():
                assert row["status"] == "ok" and Path(row["audio_path"]).exists()
                assert row["duration_after_sec"] <= row["duration_before_sec"]

            dataset = processed_db.build_final_dataset()
            assert len(dataset) == 3
            assert all(
                "full_script" in row and "urdu_text" in row
                and row.get("tts_audio_path") and row.get("processed_audio_path")
                for row in dataset
            )

            raw_db.close()
            processed_db.close()
