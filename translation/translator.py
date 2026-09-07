"""
translation/translator.py
============================
Stage 2.6 -- English -> Urdu Translation (offline-first).

Design follows the same swap-out philosophy as the classifier and the
duplicate detector: the default backend is a deterministic, fully
offline dictionary + transliteration translator (no model download, no
API key -- Requirements 13 & 16). If a *local* LLM endpoint is
configured (`TRANSLATION_BACKEND=llm` + `LOCAL_LLM_ENDPOINT`, e.g. an
Ollama / llama.cpp server on localhost), it is tried first and the
dictionary backend remains the automatic fallback whenever the endpoint
is unreachable -- the pipeline never depends on the network.

How the dictionary backend translates one English sentence:

1. A few news-syntax patterns are reordered into natural Urdu syntax
   ("X said Y" -> "X نے کہا کہ Y", "According to X, Y" ->
   "X کے مطابق، Y").
2. Everything else is translated gloss-style: multi-word phrases from
   `lexicon.PHRASE_MAP` (longest first), then single words from
   `lexicon.WORD_MAP` / `lexicon.FUNCTION_MAP`, then unknown tokens
   (mostly proper nouns) are transliterated into Urdu script so no
   entity is silently dropped.

Documented limitation: gloss-style output for pattern-unmatched
sentences keeps English word order, which is approximate Urdu. This is
an honest, deterministic baseline; pointing `LOCAL_LLM_ENDPOINT` at a
local Llama-3/Mistral server upgrades the same stage to fluent NMT
without any other code change.

Urdu-source articles are passed through untouched (translation is a
no-op for them -- they only go through the refiner).
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from data_acquisition.utils.logger import get_logger
from preprocessing.normalizer import split_sentences

from .lexicon import FUNCTION_MAP, PHRASE_MAP, SORTED_PHRASES, WORD_MAP
from .refiner import UrduRefiner
from .transliterator import transliterate_word

logger = get_logger("pipeline.translation")

_PUNCT_MAP = {",": "،", ";": "؛", "?": "؟", ".": "۔", "!": "!"}

# "X said/announced/... (that) Y" -> "X نے کہا کہ Y"
_SAID_RE = re.compile(
    r"^([A-Z][A-Za-z .'\-]+?)\s+(said|says|told|warned|confirmed|stated|claimed|"
    r"announced|reported|declared)(?:\s+that)?[\s,]+(.+)$"
)
# "According to X, Y" -> "X کے مطابق، Y"
_ACCORDING_RE = re.compile(r"^according to\s+([^,]+?)\s*,\s*(.+)$", re.IGNORECASE)

_VERB_UR = {
    "said": "کہا", "says": "کہا", "told": "بتایا", "warned": "خبردار کیا",
    "confirmed": "تصدیق کی", "stated": "بیان دیا", "claimed": "دعویٰ کیا",
    "announced": "اعلان کیا", "reported": "رپورٹ کیا", "declared": "اعلان کیا",
}


@dataclass
class TranslationResult:
    article_id: int
    urdu_title: str = ""
    urdu_text: str = ""
    urdu_sentences: List[str] = field(default_factory=list)
    source_language: str = "unknown"
    translated: bool = False          # False for Urdu-source passthrough
    refinement_notes: List[str] = field(default_factory=list)
    status: str = "ok"                # ok | empty_text | error


class UrduTranslator:
    def __init__(
        self,
        backend: str = "dictionary",
        llm_endpoint: str = "",
        llm_model: str = "",
        llm_timeout: float = 20.0,
        refiner: Optional[UrduRefiner] = None,
    ):
        self.backend = backend
        self.llm_endpoint = llm_endpoint.rstrip("/")
        self.llm_model = llm_model
        self.llm_timeout = llm_timeout
        self.refiner = refiner or UrduRefiner()

    # -- public API ---------------------------------------------------------
    def translate_article(
        self, article_id: int, title: str, text: str, language: str
    ) -> TranslationResult:
        try:
            if not (text or "").strip() and not (title or "").strip():
                return TranslationResult(article_id, status="empty_text")

            if language == "ur":
                # Already Urdu: no translation, only refinement.
                refined, notes = self.refiner.refine(text or "")
                refined_title, _ = self.refiner.refine(title or "")
                return TranslationResult(
                    article_id, refined_title, refined, split_sentences(refined),
                    language, False, notes, "ok",
                )

            urdu_title = self.translate_sentence(title or "")
            sentences = split_sentences(text or "")
            urdu_sentences = [self.translate_sentence(s) for s in sentences]
            raw_urdu = " ".join(urdu_sentences)
            refined, notes = self.refiner.refine(raw_urdu)

            return TranslationResult(
                article_id, urdu_title, refined, split_sentences(refined),
                language, True, notes, "ok",
            )
        except Exception as exc:
            logger.error("Translation failed for article_id=%s: %s", article_id, exc)
            return TranslationResult(article_id, status="error")

    def run(self, articles: List[dict]) -> List[TranslationResult]:
        """`articles`: list of {"article_id", "title", "text", "language"}."""
        results = [
            self.translate_article(
                a["article_id"], a.get("title") or "", a.get("text") or "",
                a.get("language", "unknown"),
            )
            for a in articles
        ]
        logger.info(
            "Translation complete for %d articles (%d translated, %d passthrough)",
            len(results),
            sum(1 for r in results if r.translated),
            sum(1 for r in results if r.status == "ok" and not r.translated),
        )
        return results

    # -- sentence level -------------------------------------------------------
    def translate_sentence(self, sentence: str) -> str:
        if not sentence or not sentence.strip():
            return ""

        if self.backend == "llm" and self.llm_endpoint:
            llm_out = self._llm_translate(sentence)
            if llm_out:
                return llm_out
            logger.warning("LLM backend unreachable; falling back to dictionary translation")

        # Pattern 1: "X said Y" -> "X نے کہا کہ Y"
        m = _SAID_RE.match(sentence.strip())
        if m:
            subject, verb, rest = m.groups()
            return f"{self._translate_gloss(subject)} نے {_VERB_UR[verb.lower()]} کہ {self._translate_gloss(rest)}"

        # Pattern 2: "According to X, Y" -> "X کے مطابق، Y"
        m = _ACCORDING_RE.match(sentence.strip())
        if m:
            source, rest = m.groups()
            return f"{self._translate_gloss(source)} کے مطابق، {self._translate_gloss(rest)}"

        return self._translate_gloss(sentence)

    # -- dictionary backend ----------------------------------------------------
    def _translate_gloss(self, text: str) -> str:
        lower = text.lower()

        # Protect multi-word phrases with placeholder tokens first.
        placeholders = {}
        for idx, phrase in enumerate(SORTED_PHRASES):
            token = f"\x00{idx}\x00"
            if phrase in lower:
                placeholders[token] = PHRASE_MAP[phrase]
                lower = lower.replace(phrase, token)

        tokens = re.findall(r"\x00\d+\x00|\d+(?:[.,]\d+)*|[A-Za-z']+|[^\sA-Za-z0-9]+|\s+", lower)
        out: List[str] = []
        for tok in tokens:
            if tok in placeholders:
                out.append(placeholders[tok])
            elif tok[0].isdigit():
                out.append(tok)
            elif tok[0].isalpha():
                word = tok.strip("'")
                if word in FUNCTION_MAP and FUNCTION_MAP[word] is None:
                    continue  # dropped function word (articles, copula noise)
                ur = WORD_MAP.get(word) or FUNCTION_MAP.get(word)
                out.append(ur if ur else transliterate_word(word))
            elif tok.strip():
                mapped = _PUNCT_MAP.get(tok)
                if mapped:
                    out.append(mapped)
            # whitespace tokens are dropped; pieces re-joined with one space
        return " ".join(p for p in out if p.strip()).replace("  ", " ")

    # -- optional local-LLM backend ----------------------------------------------
    def _llm_translate(self, text: str) -> str:
        """Best-effort call to a local OpenAI-compatible chat endpoint.

        Any failure (connection refused, timeout, bad payload) returns ""
        so the caller falls back to the deterministic dictionary backend.
        """
        payload = json.dumps(
            {
                "model": self.llm_model or "local",
                "messages": [
                    {
                        "role": "system",
                        "content": "Translate the following English news text into "
                                   "formal Urdu news style. Reply with only the Urdu.",
                    },
                    {"role": "user", "content": text},
                ],
                "temperature": 0.1,
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
            logger.debug("LLM translation attempt failed: %s", exc)
            return ""
