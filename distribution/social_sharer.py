"""
distribution/social_sharer.py
================================
Stage 4.2 (sharing) — post article links and captions to Facebook,
X/Twitter, Telegram, and WhatsApp via their official APIs.

Each platform method returns ``{post_id, url, status, error}``.  When
API credentials are missing the platform is marked ``not_configured``
without raising an error.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

import requests

from data_acquisition.utils.logger import get_logger

from . import config as dc
from . import social_content

logger = get_logger("distribution.social_sharer")

_TIMEOUT = 30  # HTTP request timeout


# ---------------------------------------------------------------------------
# Facebook / Meta Graph API
# ---------------------------------------------------------------------------

def _post_facebook(
    text: str,
    youtube_url: str = "",
) -> Dict[str, Any]:
    """Post to Facebook Page via Graph API."""
    if not dc.is_facebook_configured():
        return {"post_id": "", "url": "", "status": "not_configured",
                "error": "Facebook credentials not set"}

    if dc.DISTRIBUTION_MOCK_MODE:
        mock_id = f"FB_MOCK_{uuid.uuid4().hex[:8]}"
        logger.info("[MOCK] Facebook post: %s", mock_id)
        return {"post_id": mock_id, "url": f"https://facebook.com/{mock_id}",
                "status": "mock_posted", "error": ""}

    api_url = (
        f"https://graph.facebook.com/{dc.FACEBOOK_API_VERSION}"
        f"/{dc.FACEBOOK_PAGE_ID}/feed"
    )
    payload = {"message": text, "access_token": dc.FACEBOOK_ACCESS_TOKEN}
    if youtube_url:
        payload["link"] = youtube_url

    try:
        resp = requests.post(api_url, data=payload, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        post_id = data.get("id", "")
        logger.info("Facebook post succeeded: %s", post_id)
        return {"post_id": post_id,
                "url": f"https://facebook.com/{post_id}",
                "status": "posted", "error": ""}
    except Exception as exc:
        logger.error("Facebook post failed: %s", exc)
        return {"post_id": "", "url": "", "status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# X / Twitter API v2
# ---------------------------------------------------------------------------

def _post_twitter(text: str) -> Dict[str, Any]:
    """Post a tweet via X/Twitter API v2."""
    if not dc.is_twitter_configured():
        return {"post_id": "", "url": "", "status": "not_configured",
                "error": "Twitter credentials not set"}

    if dc.DISTRIBUTION_MOCK_MODE:
        mock_id = f"TW_MOCK_{uuid.uuid4().hex[:8]}"
        logger.info("[MOCK] Twitter post: %s", mock_id)
        return {"post_id": mock_id, "url": f"https://x.com/status/{mock_id}",
                "status": "mock_posted", "error": ""}

    try:
        from requests_oauthlib import OAuth1  # type: ignore
        auth = OAuth1(
            dc.TWITTER_API_KEY, dc.TWITTER_API_SECRET,
            dc.TWITTER_ACCESS_TOKEN, dc.TWITTER_ACCESS_TOKEN_SECRET,
        )
        resp = requests.post(
            "https://api.twitter.com/2/tweets",
            json={"text": text},
            auth=auth,
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json().get("data", {})
        tweet_id = data.get("id", "")
        logger.info("Twitter post succeeded: %s", tweet_id)
        return {"post_id": tweet_id,
                "url": f"https://x.com/status/{tweet_id}",
                "status": "posted", "error": ""}
    except ImportError:
        # Fallback: use requests with Bearer Token if available
        logger.warning("requests_oauthlib not installed; trying Bearer token fallback")
        return {"post_id": "", "url": "", "status": "error",
                "error": "requests_oauthlib not installed"}
    except Exception as exc:
        logger.error("Twitter post failed: %s", exc)
        return {"post_id": "", "url": "", "status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# Telegram Bot API
# ---------------------------------------------------------------------------

def _post_telegram(text: str) -> Dict[str, Any]:
    """Send a message via Telegram Bot API."""
    if not dc.is_telegram_configured():
        return {"post_id": "", "url": "", "status": "not_configured",
                "error": "Telegram credentials not set"}

    if dc.DISTRIBUTION_MOCK_MODE:
        mock_id = f"TG_MOCK_{uuid.uuid4().hex[:8]}"
        logger.info("[MOCK] Telegram post: %s", mock_id)
        return {"post_id": mock_id, "url": f"https://t.me/{mock_id}",
                "status": "mock_posted", "error": ""}

    api_url = f"https://api.telegram.org/bot{dc.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": dc.TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": False,
    }

    try:
        resp = requests.post(api_url, json=payload, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json().get("result", {})
        msg_id = str(data.get("message_id", ""))
        chat_id = data.get("chat", {}).get("id", dc.TELEGRAM_CHAT_ID)
        logger.info("Telegram post succeeded: %s", msg_id)
        return {"post_id": msg_id,
                "url": f"https://t.me/c/{chat_id}/{msg_id}",
                "status": "posted", "error": ""}
    except Exception as exc:
        logger.error("Telegram post failed: %s", exc)
        return {"post_id": "", "url": "", "status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# WhatsApp Business API
# ---------------------------------------------------------------------------

def _post_whatsapp(text: str) -> Dict[str, Any]:
    """Send a message template via WhatsApp Business API."""
    if not dc.is_whatsapp_configured():
        return {"post_id": "", "url": "", "status": "not_configured",
                "error": "WhatsApp credentials not set"}

    if dc.DISTRIBUTION_MOCK_MODE:
        mock_id = f"WA_MOCK_{uuid.uuid4().hex[:8]}"
        logger.info("[MOCK] WhatsApp post: %s", mock_id)
        return {"post_id": mock_id, "url": "",
                "status": "mock_posted", "error": ""}

    api_url = (
        f"{dc.WHATSAPP_API_URL}/{dc.WHATSAPP_PHONE_NUMBER_ID}/messages"
    )
    headers = {
        "Authorization": f"Bearer {dc.WHATSAPP_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": dc.TELEGRAM_CHAT_ID,  # placeholder — real impl needs recipient config
        "type": "text",
        "text": {"body": text},
    }

    try:
        resp = requests.post(api_url, json=payload, headers=headers, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        msg_id = data.get("messages", [{}])[0].get("id", "")
        logger.info("WhatsApp post succeeded: %s", msg_id)
        return {"post_id": msg_id, "url": "",
                "status": "posted", "error": ""}
    except Exception as exc:
        logger.error("WhatsApp post failed: %s", exc)
        return {"post_id": "", "url": "", "status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# Public batch API
# ---------------------------------------------------------------------------

def share_article(
    headline: str,
    summary: str = "",
    category: str = "",
    youtube_url: str = "",
    platforms: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Generate social content and post to all configured platforms.

    Returns a dict keyed by platform name with per-platform results.
    """
    if platforms is None:
        platforms = ["facebook", "twitter", "telegram", "whatsapp"]

    content = social_content.generate_all(headline, summary, category, youtube_url)
    results: Dict[str, Dict[str, Any]] = {}

    dispatch = {
        "facebook": lambda: _post_facebook(content["facebook"], youtube_url),
        "twitter": lambda: _post_twitter(content["twitter"]),
        "telegram": lambda: _post_telegram(content["telegram"]),
        "whatsapp": lambda: _post_whatsapp(content["whatsapp"]),
    }

    for platform in platforms:
        handler = dispatch.get(platform)
        if handler:
            results[platform] = handler()
        else:
            results[platform] = {"post_id": "", "url": "", "status": "unknown_platform",
                                 "error": f"Unknown platform: {platform}"}

    return results


def run(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Batch share articles to social media.

    Each article dict should contain: ``article_id``, ``headline``,
    ``youtube_url`` (from Stage 4.1), ``full_script``, ``category``.
    Returns per-article result dicts.
    """
    results: List[Dict[str, Any]] = []
    for art in articles:
        youtube_url = art.get("youtube_url", "")
        summary = art.get("summary", "")
        if not summary and art.get("full_script"):
            summary = art["full_script"][:300]

        platforms = share_article(
            headline=art.get("headline", ""),
            summary=summary,
            category=art.get("category", ""),
            youtube_url=youtube_url,
        )

        result = {
            "article_id": art.get("article_id"),
            "platforms": platforms,
        }
        # Flatten per-platform status for DB storage
        for plat, res in platforms.items():
            result[f"{plat}_post_id"] = res.get("post_id", "")
            result[f"{plat}_status"] = res.get("status", "")

        results.append(result)

    posted = sum(
        1 for r in results
        if any(
            r.get(f"{p}_status", "") in ("posted", "mock_posted")
            for p in ("facebook", "twitter", "telegram", "whatsapp")
        )
    )
    logger.info("Social sharing batch: %d/%d articles had at least one platform post", posted, len(results))
    return results
