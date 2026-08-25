"""
preprocessing/html_cleaner.py
==============================
Stage 2.1 helpers: strip HTML markup/entities and remove common
advertisement / promotional boilerplate from scraped article text.

Kept separate from `normalizer.py` on purpose -- HTML stripping and ad
removal both operate on the *raw* scraped string, before whitespace/
punctuation normalization happens.
"""

from __future__ import annotations

import html
import re

from bs4 import BeautifulSoup

# Lines/phrases that are almost always ad/boilerplate, not article content.
# Matched case-insensitively, anchored to a whole line (or a short
# standalone sentence) so we don't nuke real sentences that merely mention
# e.g. "subscribe" in passing.
_AD_LINE_PATTERNS = [
    r"^\s*advertisement\s*$",
    r"^\s*sponsored( content)?\s*$",
    r"^\s*read more\s*[:\-]?.*$",
    r"^\s*click here.*$",
    r"^\s*subscribe (to|now).*$",
    r"^\s*sign up (for|to).*(newsletter|updates).*$",
    r"^\s*follow us on\b.*$",
    r"^\s*share (this|on).*(facebook|twitter|whatsapp).*$",
    r"^\s*\(?adsbygoogle.*$",
    r"^\s*continue reading.*$",
    r"^\s*related (articles?|stories?)\s*[:\-]?\s*$",
    r"^\s*this (article|story) (was|is) (originally )?published.*$",
    r"^\s*copyright\s+\S*\s*\d{4}.*$",
    r"^\s*all rights reserved\.?\s*$",
]
_AD_LINE_RE = [re.compile(p, re.IGNORECASE) for p in _AD_LINE_PATTERNS]


def strip_html(text: str | None) -> str:
    """Remove HTML tags and decode HTML entities into plain readable text.

    Uses BeautifulSoup (already a project dependency) so malformed markup
    -- common in scraped `<content:encoded>` blocks -- doesn't blow up a
    regex-based stripper. Falls back to a plain entity-unescape if the
    input has no tags at all (cheap no-op path for RSS teaser text).
    """
    if not text:
        return ""

    if "<" in text and ">" in text:
        soup = BeautifulSoup(text, "lxml")
        # Drop script/style/iframe blocks outright -- never article content.
        for tag in soup(["script", "style", "iframe", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n")

    # Decode any leftover named/numeric entities (&amp;, &#39;, ...).
    text = html.unescape(text)
    return text


def remove_ads(text: str | None) -> str:
    """Drop lines that are clearly ad/promo boilerplate, keep everything else.

    Operates line-by-line so a single ad line embedded between two real
    paragraphs doesn't take the surrounding content with it.
    """
    if not text:
        return ""

    kept_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if any(pattern.match(stripped) for pattern in _AD_LINE_RE):
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines)
