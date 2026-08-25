"""
utils/text_utils.py
====================
Small, dependency-light helpers shared by every scraper:

- clean_text        -> normalise whitespace/newlines in scraped text
- detect_language    -> quick Urdu vs English detection (script-based,
                        with an optional langdetect fallback)
- compute_content_hash -> stable hash used for de-duplication in the DB
- safe_iso_now       -> current UTC timestamp in ISO-8601
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Optional

_WHITESPACE_RE = re.compile(r"[ \t\u00A0]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_URDU_RANGE_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F]")


def clean_text(text: Optional[str]) -> str:
    """Collapse repeated whitespace/newlines and strip stray characters."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE_RE.sub(" ", text)
    text = re.sub(r" *\n *", "\n", text)   # drop spaces hugging a newline
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def detect_language(text: Optional[str]) -> str:
    """Return 'ur' (Urdu), 'en' (English) or 'unknown'.

    Uses a fast Unicode-range heuristic first (cheap, no dependency, and
    reliable for Urdu since its script range is unambiguous). Falls back
    to `langdetect` for other/ambiguous cases if it's installed.
    """
    if not text or not text.strip():
        return "unknown"

    urdu_chars = len(_URDU_RANGE_RE.findall(text))
    letters = len(re.findall(r"[^\W\d_]", text, re.UNICODE))
    if letters == 0:
        return "unknown"

    if urdu_chars / max(letters, 1) > 0.3:
        return "ur"

    try:
        from langdetect import detect, LangDetectException  # type: ignore

        try:
            code = detect(text)
            return "en" if code == "en" else code
        except LangDetectException:
            pass
    except ImportError:
        pass

    # Fallback heuristic: mostly ASCII letters => assume English.
    ascii_letters = len(re.findall(r"[A-Za-z]", text))
    return "en" if ascii_letters / max(letters, 1) > 0.5 else "unknown"


def compute_content_hash(*parts: str) -> str:
    """Deterministic SHA-256 hash used as the DB de-duplication key.

    Prefer passing the article URL (most unique). If no URL is available,
    combine title + first N chars of content so near-identical scrapes of
    the same story still collapse to one row.
    """
    joined = "||".join(p.strip().lower() for p in parts if p)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def safe_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def truncate(text: str, max_len: int = 200) -> str:
    text = text or ""
    return text if len(text) <= max_len else text[: max_len - 1].rstrip() + "…"
