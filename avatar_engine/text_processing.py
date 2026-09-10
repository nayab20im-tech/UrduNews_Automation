"""
avatar_engine/text_processing.py
===================================
Urdu-aware script normalization, sentence segmentation and semantic
chunking for the avatar pipeline.

All processing is pure-Unicode (``str`` / ``unicodedata``) — no byte
level or ASCII assumptions, no mid-word truncation.  Long broadcast
scripts are split into sentence-level chunks grouped to land inside
the 8–20 s animation window so no model ever receives an unbounded
single inference call.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Urdu sentence terminators: full stop (۔ U+06D4), question mark
# (؟ U+061F), exclamation, plus their ASCII twins and newlines.
_SENTENCE_END = re.compile(r"([۔؟!.?\n]+)\s*")

# Broadcast section labels (visual-only) inherited from Stage 2.10.
_LABEL_RE = re.compile(
    r"^(?:ہیڈلائن|تعارف|مرکزی خبر|اہم نکات|اختتامیہ)\s*:?\s*",
    re.MULTILINE,
)


def normalize_urdu(text: str) -> str:
    """Normalize Urdu Unicode for TTS without losing any words.

    NFKC folds Arabic presentation forms (U+FB50–U+FDFF, U+FE70–U+FEFF)
    into canonical code points so every engine sees one consistent
    representation; visual-only broadcast labels are stripped and
    whitespace is collapsed.  The text is never truncated.
    """
    if not text or not text.strip():
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _LABEL_RE.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """Split normalized Urdu text into sentences.

    Keeps the terminator attached to its sentence.  A run that contains
    no terminator at all is returned as one sentence (never broken).
    """
    sentences: list[str] = []
    pos = 0
    for m in _SENTENCE_END.finditer(text):
        end = m.end()
        piece = text[pos:end].strip()
        if piece:
            sentences.append(piece)
        pos = end
    tail = text[pos:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def estimate_seconds(text: str, words_per_min: int = 160) -> float:
    """Rough speech-duration estimate used only for chunk grouping.

    Urdu news delivery at ``words_per_min`` ≈ 160 wpm (espeak default).
    """
    words = len(text.split())
    return words / max(words_per_min, 1) * 60.0


@dataclass
class ScriptChunk:
    chunk_id: str        # deterministic: chunk_0001, chunk_0002, ...
    text: str
    est_seconds: float


def chunk_script(
    text: str,
    min_sec: float = 8.0,
    max_sec: float = 20.0,
    words_per_min: int = 160,
) -> list[ScriptChunk]:
    """Group sentences into speech chunks of ~``min_sec``–``max_sec``.

    Sentences are atomic (words are never broken).  A single sentence
    longer than ``max_sec`` becomes its own oversized chunk rather than
    being truncated.
    """
    cleaned = normalize_urdu(text)
    if not cleaned:
        return []

    sentences = split_sentences(cleaned)
    groups: list[str] = []
    current: list[str] = []
    current_sec = 0.0

    for sent in sentences:
        s_sec = estimate_seconds(sent, words_per_min)
        if current and current_sec + s_sec > max_sec:
            groups.append(" ".join(current))
            current, current_sec = [], 0.0
        current.append(sent)
        current_sec += s_sec
        # Close the group once it is comfortably inside the window and
        # the next sentence would push it past max.
        if current_sec >= min_sec:
            groups.append(" ".join(current))
            current, current_sec = [], 0.0
    if current:
        groups.append(" ".join(current))

    chunks = [
        ScriptChunk(
            chunk_id=f"chunk_{i:04d}",
            text=g,
            est_seconds=round(estimate_seconds(g, words_per_min), 2),
        )
        for i, g in enumerate(groups, start=1)
    ]
    return chunks
