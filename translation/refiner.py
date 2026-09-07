"""
translation/refiner.py
=========================
Stage 2.6 -- Urdu Grammar Correction & Style/Readability Enhancement.

A deterministic rule pass over Urdu text (translated or originally
Urdu). It never rewrites meaning -- every rule is form-level, so the
"do not invent facts" requirement holds by construction:

- Unicode/punctuation normalization (reuses Stage 2.1's `normalize_text`).
- Register lift: colloquial wording -> formal news style
  (`lexicon.STYLE_MAP`, e.g. "بولا" -> "کہا").
- Grammar cleanup: collapsed accidental repeated tokens ("کہ کہ"),
  spaces hugging Urdu punctuation (" ،" -> "،").
- Readability: over-long sentences are split at a comma; a missing
  terminal "۔" is appended so every sentence closes properly in Urdu.

Each applied fix is reported in a notes list so the pipeline stays
auditable (Requirement: traceable processing).
"""

from __future__ import annotations

import re
from typing import List, Tuple

from data_acquisition.utils.logger import get_logger
from preprocessing.normalizer import normalize_text, split_sentences

from .lexicon import STYLE_MAP

logger = get_logger("pipeline.translation.refiner")

_DEFAULT_MAX_SENTENCE_WORDS = 35
_URDU_CHAR_RE = re.compile(r"[\u0600-\u06FF]")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([،۔؛؟!])")
_REPEAT_TOKEN_RE = re.compile(r"(\S+)\s+\1(?=\s|$)")


class UrduRefiner:
    def __init__(self, max_sentence_words: int = _DEFAULT_MAX_SENTENCE_WORDS):
        self.max_sentence_words = max_sentence_words

    def refine(self, text: str) -> Tuple[str, List[str]]:
        """Return (refined_text, notes). Never raises on odd input."""
        if not text or not text.strip():
            return "", []
        notes: List[str] = []
        try:
            refined = normalize_text(text)

            # 1. Register lift (colloquial -> formal news style).
            for colloquial, formal in STYLE_MAP.items():
                if colloquial in refined:
                    refined = refined.replace(colloquial, formal)
                    notes.append(f"register_lift:{colloquial}->{formal}")

            # 2. Collapse accidental repeated tokens ("کہ کہ" -> "کہ").
            collapsed = _REPEAT_TOKEN_RE.sub(r"\1", refined)
            if collapsed != refined:
                notes.append("duplicate_token_removed")
                refined = collapsed

            # 3. No space before Urdu punctuation.
            fixed = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", refined)
            if fixed != refined:
                notes.append("punctuation_spacing_fixed")
                refined = fixed

            # 4. Readability: split over-long sentences at a mid comma.
            sentences = split_sentences(refined)
            split_happened = False
            new_sentences: List[str] = []
            for sentence in sentences:
                words = sentence.split()
                if len(words) > self.max_sentence_words and "،" in sentence:
                    mid = len(words) // 2
                    best = -1
                    pos = 0
                    for ch_idx, ch in enumerate(sentence):
                        if ch == "،":
                            word_pos = len(sentence[:ch_idx].split())
                            if best == -1 or abs(word_pos - mid) < abs(best - mid):
                                best = word_pos
                                pos = ch_idx
                    if best > 0:
                        first = sentence[: pos + 1].rstrip("،").strip()
                        second = sentence[pos + 1:].strip()
                        if first and second:
                            new_sentences.append(first + "۔")
                            new_sentences.append(second)
                            split_happened = True
                            continue
                new_sentences.append(sentence)
            if split_happened:
                notes.append("long_sentence_split")
                refined = " ".join(new_sentences)

            # 5. Every Urdu text must end with a terminal mark.
            refined = refined.strip()
            if refined and _URDU_CHAR_RE.search(refined) and not refined.endswith(("۔", "؟", "!")):
                refined += "۔"
                notes.append("terminal_punctuation_added")

            return refined, notes
        except Exception as exc:
            logger.error("Refinement failed (%s); returning normalized-only text", exc)
            return normalize_text(text), ["refiner_error"]
