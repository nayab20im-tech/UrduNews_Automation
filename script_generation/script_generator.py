"""
script_generation/script_generator.py
=========================================
Stage 2.9 -- Final Urdu News Script Generation.

Produces a broadcast-ready Urdu news script per article. Two backends,
same swap-out philosophy as Stage 2.6:

- **template (default)** -- deterministic, extractive, offline. Every
  script sentence is either a verbatim Urdu source sentence or a
  dictionary-translation of a verbatim English sentence, so the stage
  structurally cannot hallucinate facts. Professional news style is
  achieved by:
    * toning sensational wording down to a neutral register
      (`bias_lexicon.NEUTRAL_REPLACEMENTS_*`, guided by Stage 2.7),
    * formal-register refinement (Stage 2.6 refiner),
    * lead-first paragraph structure (broadcast convention).
- **llm (optional)** -- if `LOCAL_LLM_ENDPOINT` is configured, a local
  Llama-3/Mistral/DeepSeek server (architecture box 2.9) is asked to
  write the script *grounded on the extracted evidence only* (summary,
  key points, verified claims); any failure falls back to the template
  backend so the pipeline never depends on the network.

Evidence grounding (Stage 2.8 output): articles whose claims could not
be corroborated inside the acquired corpus carry an explicit editorial
clarification line in the script instead of silently presenting
unverified claims as fact.
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional

from bias_detection import bias_lexicon as ble
from data_acquisition.utils.logger import get_logger
from preprocessing.normalizer import split_sentences
from translation.translator import UrduTranslator

logger = get_logger("pipeline.script_generation")

_PARAGRAPH_SENTENCES = 3
_UNVERIFIED_NOTE = "وضاحت: اس خبر کے بعض دعوے آزاد ذرائع سے مکمل طور پر تصدیق شدہ نہیں ہیں۔"
# Minimum sentences for a comprehensive news script (ARY-style depth)
_MIN_SCRIPT_SENTENCES = 5


@dataclass
class ScriptResult:
    article_id: int
    script_text: str = ""
    word_count: int = 0
    backend_used: str = "template"     # template | llm
    verification_note: str = ""
    status: str = "ok"                 # ok | empty_text | error


class ScriptGenerator:
    def __init__(
        self,
        translator: Optional[UrduTranslator] = None,
        backend: str = "template",
        llm_endpoint: str = "",
        llm_model: str = "",
        llm_timeout: float = 30.0,
    ):
        self.translator = translator or UrduTranslator()
        self.backend = backend
        self.llm_endpoint = llm_endpoint.rstrip("/")
        self.llm_model = llm_model
        self.llm_timeout = llm_timeout

    # -- public API ---------------------------------------------------------
    def generate(
        self,
        article_id: int,
        language: str,
        sentences: List[str],
        urdu_sentences: List[str],
        verification: Optional[Dict] = None,
        summary: str = "",
        key_points: Optional[List[str]] = None,
    ) -> ScriptResult:
        try:
            if not sentences and not urdu_sentences:
                return ScriptResult(article_id, status="empty_text")

            if self.backend == "llm" and self.llm_endpoint:
                llm_script = self._llm_generate(summary, key_points or [], verification)
                if llm_script:
                    return ScriptResult(
                        article_id, llm_script, len(llm_script.split()), "llm",
                        self._verification_note(verification), "ok",
                    )
                logger.warning("LLM script backend unreachable; falling back to template")

            # Build comprehensive script from ALL available content.
            # For professional news (ARY-style), the script must cover
            # every aspect of the story, not just a brief summary.
            script_sentences = self._compose_sentences(language, sentences, urdu_sentences)

            # Enrich with key points that aren't already in the script
            # to ensure comprehensive coverage of all story aspects.
            if key_points:
                existing_text = " ".join(script_sentences).lower()
                for kp in key_points:
                    kp_clean = kp.strip()
                    if not kp_clean:
                        continue
                    # Only add key points that bring new information
                    kp_translated = (
                        self.translator.translate_sentence(kp_clean)
                        if language == "en" else kp_clean
                    )
                    if kp_translated.lower() not in existing_text:
                        script_sentences.append(kp_translated)

            if not script_sentences:
                return ScriptResult(article_id, status="empty_text")

            # Broadcast structure: lead sentence first, then body paragraphs.
            # For comprehensive news: first 2 sentences as lead/intro,
            # remaining grouped into detailed body paragraphs.
            intro_count = min(2, len(script_sentences))
            paragraphs = [" ".join(script_sentences[:intro_count])]
            rest = script_sentences[intro_count:]
            for i in range(0, len(rest), _PARAGRAPH_SENTENCES):
                paragraphs.append(" ".join(rest[i:i + _PARAGRAPH_SENTENCES]))
            script_text = "\n\n".join(p for p in paragraphs if p.strip())

            note = self._verification_note(verification)
            if note:
                script_text = f"{script_text}\n\n{note}"

            return ScriptResult(
                article_id, script_text, len(script_text.split()), "template", note, "ok",
            )
        except Exception as exc:
            logger.error("Script generation failed for article_id=%s: %s", article_id, exc)
            return ScriptResult(article_id, status="error")

    def run(self, articles: List[dict]) -> List[ScriptResult]:
        """`articles`: list of {"article_id", "language", "sentences",
        "urdu_sentences", "verification", "summary", "key_points"}."""
        results = [
            self.generate(
                a["article_id"], a.get("language", "unknown"), a.get("sentences") or [],
                a.get("urdu_sentences") or [], a.get("verification"),
                a.get("summary") or "", a.get("key_points") or [],
            )
            for a in articles
        ]
        logger.info("Script generation complete for %d articles", len(results))
        return results

    # -- template backend ------------------------------------------------------
    def _compose_sentences(
        self, language: str, sentences: List[str], urdu_sentences: List[str]
    ) -> List[str]:
        if language == "en" and sentences:
            # Tone sensational English wording down *before* translation so
            # the Urdu script comes out in a neutral news register.
            toned = [self._tone_down_en(s) for s in sentences]
            return [self.translator.translate_sentence(s) for s in toned if s.strip()]
        base = urdu_sentences or sentences
        return [self._tone_down_ur(s) for s in base if s.strip()]

    @staticmethod
    def _tone_down_en(sentence: str) -> str:
        # Plain-English stand-ins keep the sentence translatable by the
        # dictionary backend; the Urdu neutral map handles Urdu sources.
        for hype, plain in ble.EN_TONE_DOWN.items():
            sentence = re.sub(rf"\b{hype}\b", plain, sentence, flags=re.IGNORECASE)
        return sentence

    @staticmethod
    def _tone_down_ur(sentence: str) -> str:
        for hype, neutral in ble.NEUTRAL_REPLACEMENTS_UR.items():
            sentence = sentence.replace(hype, neutral)
        return sentence

    @staticmethod
    def _verification_note(verification: Optional[Dict]) -> str:
        if not verification:
            return ""
        if verification.get("verdict") in ("unverified", "partially_corroborated"):
            return _UNVERIFIED_NOTE
        return ""

    # -- optional local-LLM backend -----------------------------------------------
    def _llm_generate(self, summary: str, key_points: List[str], verification: Optional[Dict]) -> str:
        """Ask a local LLM to write the Urdu script from evidence only."""
        evidence = {
            "summary": summary,
            "key_points": key_points,
            "verified_claims": [
                c.get("claim") for c in (verification or {}).get("claim_results", [])
                if c.get("verdict") in ("corroborated", "single_source")
            ],
        }
        payload = json.dumps(
            {
                "model": self.llm_model or "local",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a professional Urdu TV news script writer. "
                                   "Write a short Urdu news script using ONLY the provided "
                                   "evidence. Do not add facts. Reply with only the Urdu script.",
                    },
                    {"role": "user", "content": json.dumps(evidence, ensure_ascii=False)},
                ],
                "temperature": 0.2,
            }
        ).encode("utf-8")
        try:
            req = urllib.request.Request(
                f"{self.llm_endpoint}/v1/chat/completions"
                if "/v1" not in self.llm_endpoint
                else f"{self.llm_endpoint}/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.llm_timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return (body["choices"][0]["message"]["content"] or "").strip()
        except Exception as exc:
            logger.debug("LLM script generation attempt failed: %s", exc)
            return ""
