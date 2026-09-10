"""
preprocessing/normalizer.py
=============================
Stage 2.1 helpers: Unicode-safe text normalization, sentence splitting,
and stopword removal.

Reuses `data_acquisition.utils.text_utils.clean_text` for the base
whitespace/newline collapsing (no need to duplicate that logic) and adds
what that function intentionally leaves out: Unicode normalization,
punctuation normalization, sentence segmentation, and stopword removal.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List

from data_acquisition.utils.text_utils import clean_text

from .stopwords import stopwords_for

# Curly quotes/dashes -> plain ASCII equivalents (keeps downstream
# tokenization/NER regexes simple and consistent).
_PUNCT_MAP = {
    "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-",
    "\u2026": "...",
}

# Sentence enders: Urdu full stop '۔', Urdu/Arabic '؟', plus standard
# Latin punctuation. Keeps the ender attached to the sentence it closes.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?۔؟])\s+")
_MULTI_SPECIAL_RE = re.compile(r"[^\w\s.,!?؟۔،'\"()%\-:/\u0600-\u06FF]", re.UNICODE)


def normalize_text(text: str | None) -> str:
    """Full text normalization pass, safe to run on the text that is kept
    for summarization (does not remove words, only normalizes form)."""
    if not text:
        return ""

    # NFKC folds compatibility characters (e.g. Arabic presentation forms
    # sometimes produced by copy-pasted/scraped Urdu) into their canonical
    # form, which matters for consistent stopword/regex matching later.
    text = unicodedata.normalize("NFKC", text)

    for src, dst in _PUNCT_MAP.items():
        text = text.replace(src, dst)

    # Drop stray control/symbol characters but keep Urdu script, basic
    # Latin, digits, and common punctuation used in news text.
    text = _MULTI_SPECIAL_RE.sub(" ", text)

    # Reuse the project's existing whitespace/newline collapsing.
    text = clean_text(text)

    # Normalize repeated punctuation, e.g. "!!!" -> "!", "...." -> "..."
    text = re.sub(r"([.!?])\1{1,}", r"\1", text)
    return text


def split_sentences(text: str | None) -> List[str]:
    """Split normalized text into sentences.

    Regex-based on purpose (no NLTK punkt download): sentence boundaries
    are marked by the sentence-ender set above. Good enough for news
    prose in English and Urdu; abbreviation edge cases (e.g. "Mr.") may
    occasionally over-split, which is an acceptable trade-off for a
    dependency-free approach.
    """
    if not text or not text.strip():
        return []

    candidates = _SENTENCE_SPLIT_RE.split(text.strip())
    sentences = [s.strip() for s in candidates if s and s.strip()]
    return sentences


def remove_stopwords(text: str | None, language: str = "unknown") -> str:
    """Return `text` with stopwords removed -- for NLP *features* only.

    Callers must keep the original `normalize_text()` output for anything
    shown to a user or fed to summarization; this output is intentionally
    lossy.
    """
    if not text or not text.strip():
        return ""

    stop = stopwords_for(language)
    tokens = re.findall(r"[\w\u0600-\u06FF']+", text, re.UNICODE)
    kept = [t for t in tokens if t.lower() not in stop]
    return " ".join(kept)
