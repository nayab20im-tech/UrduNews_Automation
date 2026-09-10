"""
script_generation/script_structurer.py
==========================================
Stage 2.10 -- Script Structuring for News.

Turns Stage 2.9's flowing Urdu script into the standard broadcast
script layout used by the downstream Video Production pipeline
(Section 3):

    ہیڈلائن   -- the Urdu headline (translated title for EN sources)
    تعارف    -- intro: the lead sentence(s) that set the scene
    مرکزی خبر -- main story: the remaining body paragraphs
    اہم نکات  -- key points as bullets (translated when the source is
                English; already Urdu for Urdu sources)
    اختتامیہ -- ending / CTA (configurable channel call-to-action)

Pure restructuring -- no text is invented or rewritten here (the CTA is
a fixed channel-level line from config, not article content). The
`full_script` field renders all sections in broadcast order for
direct hand-off to TTS (Stage 2.11) and the avatar pipeline (3.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from data_acquisition.utils.logger import get_logger
from preprocessing.normalizer import split_sentences
from translation.translator import UrduTranslator

logger = get_logger("pipeline.script_structuring")

DEFAULT_CTA = "مزید خبروں اور تازہ ترین اپڈیٹس کے لیے ہمارے چینل سے جڑے رہیں۔"
DEFAULT_GREETING = "السلام علیکم ناظرین، خبروں کے ساتھ ہم حاضر ہیں۔"


@dataclass
class StructuredScriptResult:
    article_id: int
    headline: str = ""
    intro: str = ""
    main_story: str = ""
    key_points: List[str] = field(default_factory=list)
    ending: str = ""
    full_script: str = ""
    status: str = "ok"                 # ok | empty_text | error


class ScriptStructurer:
    def __init__(self, cta_text: str = DEFAULT_CTA, greeting_text: str = DEFAULT_GREETING, translator: Optional[UrduTranslator] = None):
        self.cta_text = cta_text or DEFAULT_CTA
        self.greeting_text = greeting_text or DEFAULT_GREETING
        self.translator = translator or UrduTranslator()

    # -- public API ---------------------------------------------------------
    def structure(
        self,
        article_id: int,
        title: str,
        urdu_title: str,
        script_text: str,
        key_points: Optional[List[str]] = None,
        language: str = "unknown",
    ) -> StructuredScriptResult:
        try:
            if not (script_text or "").strip():
                return StructuredScriptResult(article_id, status="empty_text")

            headline = (urdu_title or title or "").strip()
            sentences = split_sentences(script_text)
            if not sentences:
                return StructuredScriptResult(article_id, status="empty_text")

            # Intro: the lead; on longer scripts take two sentences so the
            # anchor has a proper opening; the rest is the main story.
            intro_count = 2 if len(sentences) >= 4 else 1
            intro = " ".join(sentences[:intro_count])
            main_story = " ".join(sentences[intro_count:])

            # Key points: translate verbatim key-point sentences for
            # English sources; Urdu sources pass through unchanged.
            structured_points: List[str] = []
            for point in (key_points or []):
                if not (point or "").strip():
                    continue
                structured_points.append(
                    self.translator.translate_sentence(point) if language == "en" else point
                )

            ending = self.cta_text
            full_script = self._render(self.greeting_text, headline, intro, main_story, structured_points, ending)

            return StructuredScriptResult(
                article_id, headline, intro, main_story, structured_points,
                ending, full_script, "ok",
            )
        except Exception as exc:
            logger.error("Script structuring failed for article_id=%s: %s", article_id, exc)
            return StructuredScriptResult(article_id, status="error")

    def run(self, articles: List[dict]) -> List[StructuredScriptResult]:
        """`articles`: list of {"article_id", "title", "urdu_title",
        "script_text", "key_points", "language"}."""
        results = [
            self.structure(
                a["article_id"], a.get("title") or "", a.get("urdu_title") or "",
                a.get("script_text") or "", a.get("key_points") or [],
                a.get("language", "unknown"),
            )
            for a in articles
        ]
        logger.info("Script structuring complete for %d articles", len(results))
        return results

    # -- rendering -----------------------------------------------------------
    @staticmethod
    def _render(greeting: str, headline: str, intro: str, main_story: str, key_points: List[str], ending: str) -> str:
        blocks = [
            f"آغاز: {greeting}",
            f"ہیڈلائن: {headline}",
            f"تعارف: {intro}",
        ]
        if main_story:
            blocks.append(f"مرکزی خبر: {main_story}")
        if key_points:
            blocks.append("اہم نکات:\n" + "\n".join(f"- {p}" for p in key_points))
        blocks.append(f"اختتامیہ: {ending}")
        return "\n\n".join(blocks)
