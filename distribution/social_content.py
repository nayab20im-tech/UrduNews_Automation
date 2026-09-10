"""
distribution/social_content.py
=================================
Stage 4.2 (content) — generate platform-specific social-media post text
from article metadata (headline, summary, category, YouTube URL).

All functions are pure string builders with no external API calls.
"""

from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TWITTER_MAX_CHARS = 280
URL_PLACEHOLDER_CHARS = 23  # t.co short URL length

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    """Collapse whitespace."""
    return re.sub(r"\s+", " ", text).strip()


def _truncate(text: str, max_chars: int, suffix: str = "…") -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - len(suffix)].rsplit(" ", 1)[0] + suffix


# ---------------------------------------------------------------------------
# Facebook / Meta
# ---------------------------------------------------------------------------

def generate_facebook_post(
    headline: str,
    summary: str = "",
    category: str = "",
    youtube_url: str = "",
) -> str:
    """Generate a Facebook caption for a news video."""
    parts = [f"📰 {headline}"]
    if summary:
        parts.append("")
        parts.append(_truncate(summary, 500))
    if category:
        parts.append(f"\n📂 {category}")
    if youtube_url:
        parts.append(f"\n🎥 Watch: {youtube_url}")
    parts.append("\n#UrduNews #اردو_نیوز")
    if category:
        tag = re.sub(r"[^\w]", "", category)
        if tag:
            parts.append(f"#{tag}")
    return _clean("\n".join(parts))


# ---------------------------------------------------------------------------
# X / Twitter
# ---------------------------------------------------------------------------

def generate_twitter_post(
    headline: str,
    summary: str = "",
    category: str = "",
    youtube_url: str = "",
) -> str:
    """Generate a tweet (max 280 chars) for a news video."""
    hashtag_block = " #UrduNews"
    url_part = f" {youtube_url}" if youtube_url else ""
    available = TWITTER_MAX_CHARS - len(hashtag_block) - len(url_part) - 2  # spaces

    # Prefer headline; add summary if room
    text = headline
    if summary and len(headline) + len(summary) + 3 <= available:
        text = f"{headline}\n{summary}"

    text = _truncate(text, available)
    parts = [text]
    if youtube_url:
        parts.append(youtube_url)
    parts.append(hashtag_block.strip())
    return _clean(" ".join(parts))


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

def generate_telegram_caption(
    headline: str,
    summary: str = "",
    category: str = "",
    youtube_url: str = "",
) -> str:
    """Generate a Telegram message caption with inline link."""
    parts = [f"📰 *{headline}*"]
    if summary:
        parts.append("")
        parts.append(_truncate(summary, 400))
    if category:
        parts.append(f"\n📂 _{category}_")
    if youtube_url:
        parts.append(f"\n🎥 [Watch Full Video]({youtube_url})")
    parts.append("\n#UrduNews #اردو_نیوز")
    return _clean("\n".join(parts))


# ---------------------------------------------------------------------------
# WhatsApp Business
# ---------------------------------------------------------------------------

def generate_whatsapp_message(
    headline: str,
    summary: str = "",
    category: str = "",
    youtube_url: str = "",
) -> str:
    """Generate a WhatsApp-compatible message (plain text, no Markdown)."""
    parts = [f"📰 {headline}"]
    if summary:
        parts.append("")
        parts.append(_truncate(summary, 400))
    if category:
        parts.append(f"\n📂 {category}")
    if youtube_url:
        parts.append(f"\n🎥 {youtube_url}")
    parts.append("\n#UrduNews")
    return _clean("\n".join(parts))


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------

def generate_all(
    headline: str,
    summary: str = "",
    category: str = "",
    youtube_url: str = "",
) -> dict:
    """Return a dict with post text for every supported platform."""
    return {
        "facebook": generate_facebook_post(headline, summary, category, youtube_url),
        "twitter": generate_twitter_post(headline, summary, category, youtube_url),
        "telegram": generate_telegram_caption(headline, summary, category, youtube_url),
        "whatsapp": generate_whatsapp_message(headline, summary, category, youtube_url),
    }
