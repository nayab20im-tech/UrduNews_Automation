"""
bias_detection/bias_detector.py
===================================
Stage 2.7 -- Bias & Sensational Language Detection.

Three explainable, lexicon/rule-based signals per article (no model to
download, deterministic -- same philosophy as Stages 2.2/2.3):

- **Clickbait detection**   -- headline patterns ("You won't believe...",
  "X ways to...", Urdu equivalents), ALL-CAPS words, exclamation abuse.
- **Sensational-word score** -- density of hype vocabulary (English +
  Urdu) per 100 words, normalized to 0..1.
- **Sentiment / bias analysis** -- small high-precision sentiment
  lexicons give a -1..+1 sentiment score; loaded/bias-indicating words
  ("regime", "so-called", "سازش"...) feed a 0..1 loaded-word score.

The composite `bias_score` = 0.4*clickbait + 0.3*sensational +
0.3*loaded, labelled neutral / mild / sensational against a
configurable threshold. Every hit is recorded in `flags` so a human can
audit *why* an article was flagged (Requirement: traceable decisions).

The stage never modifies the article -- it only scores it. Toning
sensational wording down happens later, in Stage 2.9, using the
neutral-replacement maps from `bias_lexicon.py`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from data_acquisition.utils.logger import get_logger

from . import bias_lexicon as lex

logger = get_logger("pipeline.bias_detection")

_WORD_RE = re.compile(r"[\w\u0600-\u06FF']+", re.UNICODE)
_ALLCAPS_RE = re.compile(r"\b[A-Z]{4,}\b")


@dataclass
class BiasRecord:
    article_id: int
    clickbait: bool = False
    sensational_score: float = 0.0      # 0..1 hype-word density
    sentiment_score: float = 0.0        # -1..+1
    bias_score: float = 0.0             # 0..1 composite
    label: str = "neutral"              # neutral | mild | sensational
    flags: List[str] = field(default_factory=list)
    status: str = "ok"                  # ok | empty_text | error


class BiasDetector:
    def __init__(self, sensational_threshold: float = 0.4, mild_threshold: float = 0.15):
        self.sensational_threshold = sensational_threshold
        self.mild_threshold = mild_threshold

    def analyze(self, article_id: int, title: str, text: str) -> BiasRecord:
        try:
            body = f"{title or ''} {text or ''}".strip()
            if not body:
                return BiasRecord(article_id, status="empty_text")

            words = _WORD_RE.findall(body)
            word_count = max(len(words), 1)
            lowered = {w.lower() for w in words}
            flags: List[str] = []

            # -- clickbait patterns (headline-weighted) ---------------------
            clickbait = False
            for pattern in list(lex.CLICKBAIT_PATTERNS_EN) + list(lex.CLICKBAIT_PATTERNS_UR):
                if pattern.search(title or "") or pattern.search(text or ""):
                    clickbait = True
                    flags.append(f"clickbait_pattern:{pattern.pattern[:30]}")

            caps = _ALLCAPS_RE.findall(title or "")
            if len(caps) >= 2:
                clickbait = True
                flags.append(f"all_caps_words:{','.join(caps[:3])}")

            exclamations = (title or "").count("!") + (text or "").count("!")
            if exclamations >= 3:
                clickbait = True
                flags.append(f"exclamation_abuse:{exclamations}")

            # -- sensational-word density ------------------------------------
            sens_hits = [w for w in lowered
                         if w in lex.SENSATIONAL_WORDS_EN or w in lex.SENSATIONAL_WORDS_UR]
            for hit in sorted(sens_hits)[:5]:
                flags.append(f"sensational_word:{hit}")
            sensational_score = min(1.0, (len(sens_hits) / word_count) * 100 / 5.0)

            # -- loaded / bias words ------------------------------------------
            loaded_hits = [w for w in lowered
                           if w in lex.LOADED_WORDS_EN or w in lex.LOADED_WORDS_UR]
            for hit in sorted(loaded_hits)[:5]:
                flags.append(f"loaded_word:{hit}")
            loaded_score = min(1.0, len(loaded_hits) / 2.0)

            # -- sentiment ------------------------------------------------------
            pos = sum(1 for w in lowered
                      if w in lex.POSITIVE_WORDS_EN or w in lex.POSITIVE_WORDS_UR)
            neg = sum(1 for w in lowered
                      if w in lex.NEGATIVE_WORDS_EN or w in lex.NEGATIVE_WORDS_UR)
            sentiment_score = round((pos - neg) / max(pos + neg, 1), 4) if (pos + neg) else 0.0

            bias_score = round(
                min(1.0, 0.4 * (1.0 if clickbait else 0.0) + 0.3 * sensational_score + 0.3 * loaded_score),
                4,
            )
            if clickbait or bias_score >= self.sensational_threshold:
                label = "sensational"
            elif bias_score >= self.mild_threshold:
                label = "mild"
            else:
                label = "neutral"

            return BiasRecord(
                article_id, clickbait, round(sensational_score, 4),
                sentiment_score, bias_score, label, flags, "ok",
            )
        except Exception as exc:
            logger.error("Bias analysis failed for article_id=%s: %s", article_id, exc)
            return BiasRecord(article_id, status="error")

    def run(self, articles: List[dict]) -> List[BiasRecord]:
        """`articles`: list of {"article_id", "title", "text"}."""
        results = [
            self.analyze(a["article_id"], a.get("title") or "", a.get("text") or "")
            for a in articles
        ]
        logger.info(
            "Bias detection complete: %d sensational, %d mild, %d neutral (of %d)",
            sum(1 for r in results if r.label == "sensational"),
            sum(1 for r in results if r.label == "mild"),
            sum(1 for r in results if r.label == "neutral"),
            len(results),
        )
        return results
