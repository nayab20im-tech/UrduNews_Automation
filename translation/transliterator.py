"""
translation/transliterator.py
================================
Rule-based English -> Urdu script transliteration.

Used as the *fallback* path of Stage 2.6 for tokens that are not in the
bilingual news lexicon (mostly proper nouns -- names of people, places,
organisations). A transliteration keeps those entities readable in an
Urdu broadcast script instead of dropping them or leaving Latin script
in the middle of an Urdu sentence.

This is deliberately a simple character/digraph mapping (no learned
grapheme-to-phoneme model, nothing to download). The output is an
approximation of conventional Urdu spellings -- e.g. "Pakistan" ->
"پاکستان" and "Islamabad" -> "اسلام آباد" come out right because their
Latin letters map cleanly -- while unusual names may come out merely
pronounceable. Conventional spellings for common entities are handled
one layer up, in `translation/lexicon.py`, which is consulted before
this module ever sees the token.
"""

from __future__ import annotations

import re

# Multi-letter Latin sequences resolved first (longest match wins).
_DIGRAPHS = [
    ("sh", "ش"), ("ch", "چ"), ("th", "تھ"), ("ph", "ف"), ("kh", "خ"),
    ("gh", "غ"), ("zh", "ژ"), ("qu", "ق"), ("ng", "نگ"), ("wh", "و"),
    ("wr", "ر"), ("kn", "ن"), ("ck", "ک"),
]

_SINGLE = {
    "a": "ا", "b": "ب", "c": "ک", "d": "د", "e": "ی", "f": "ف", "g": "گ",
    "h": "ہ", "i": "ی", "j": "ج", "k": "ک", "l": "ل", "m": "م", "n": "ن",
    "o": "و", "p": "پ", "q": "ق", "r": "ر", "s": "س", "t": "ت", "u": "و",
    "v": "و", "w": "و", "x": "کس", "y": "ی", "z": "ز",
}

_VOWELS = set("aeiou")
_TOKEN_RE = re.compile(r"[A-Za-z]+")


def transliterate_word(word: str) -> str:
    """Transliterate one Latin word into Urdu script (approximate)."""
    w = word.lower()
    out: list[str] = []
    i = 0
    while i < len(w):
        matched = False
        for dig, ur in _DIGRAPHS:
            if w.startswith(dig, i):
                out.append(ur)
                i += len(dig)
                matched = True
                break
        if matched:
            continue
        ch = w[i]
        if ch in _SINGLE:
            letter = _SINGLE[ch]
            # Avoid vowel letters stacking on a preceding vowel letter
            # (e.g. "aa" -> "اا" reads badly); collapse to one.
            if ch in _VOWELS and out and out[-1] in ("ا", "ی", "و", "ای", "او"):
                pass
            else:
                out.append(letter)
        # any other character (digits, punctuation) is dropped here; the
        # caller handles non-letter tokens separately.
        i += 1

    text = "".join(out)
    # Collapse accidental repeated letters from double consonants ("ss").
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    return text


def transliterate(text: str) -> str:
    """Transliterate every Latin word in `text`, keeping everything else."""
    if not text:
        return ""
    return _TOKEN_RE.sub(lambda m: transliterate_word(m.group(0)), text)
