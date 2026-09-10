"""
distribution/youtube_uploader.py
===================================
Stage 4.1 — YouTube video upload, thumbnail setting, and metadata
generation via the YouTube Data API v3.

When OAuth credentials are missing the module returns
``status="not_configured"`` and logs clear setup instructions — it
never fakes success.  In mock mode (``DISTRIBUTION_MOCK_MODE=true``)
all API calls are simulated.
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from data_acquisition.utils.logger import get_logger

from . import config as dc

logger = get_logger("distribution.youtube")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# YouTube category ID 25 = "News & Politics"
_DEFAULT_CATEGORY = "25"


def _build_title(headline: str, channel_name: str = "") -> str:
    """Build the YouTube title from the configured template."""
    template = dc.YOUTUBE_TITLE_TEMPLATE or "{headline} | {channel_name}"
    title = template.replace("{headline}", headline).replace(
        "{channel_name}", channel_name or dc.CHANNEL_NAME
    )
    # YouTube title limit is 100 chars
    return title[:100]


def _build_description(
    headline: str,
    full_script: str = "",
    category: str = "",
    youtube_url: str = "",
) -> str:
    """Build a YouTube description from article content."""
    parts: List[str] = []
    if full_script:
        # Trim script to ~800 chars to leave room for metadata
        snippet = full_script[:800].rsplit(" ", 1)[0]
        if len(full_script) > 800:
            snippet += " …"
        parts.append(snippet)
    parts.append("")
    if category:
        parts.append(f"📂 Category: {category}")
    if youtube_url:
        parts.append(f"🔗 {youtube_url}")
    parts.append("")
    parts.append(f"📺 {dc.CHANNEL_NAME}")
    parts.append("#UrduNews #اردو_نیوز")
    if category:
        tag = re.sub(r"[^\w]", "", category)
        if tag:
            parts.append(f"#{tag}")
    text = "\n".join(parts)
    # YouTube description limit is 5000 chars
    return text[:5000]


def _build_tags(headline: str, category: str = "") -> List[str]:
    """Auto-generate YouTube tags from headline words + category."""
    words = [w for w in re.split(r"[^\w\u0600-\u06FF]+", headline) if len(w) > 2]
    tags = list(dict.fromkeys(words))  # deduplicate, keep order
    tags.extend(["Urdu News", "اردو نیوز", dc.CHANNEL_NAME])
    if category:
        tags.append(category)
    # YouTube allows max 500 chars total for tags
    result: List[str] = []
    total = 0
    for t in tags:
        if total + len(t) + 1 > 490:
            break
        result.append(t)
        total += len(t) + 1
    return result


# ---------------------------------------------------------------------------
# OAuth helpers
# ---------------------------------------------------------------------------

def _get_credentials():
    """Return valid Google OAuth2 credentials, or None."""
    try:
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
    except ImportError:
        logger.warning(
            "google-api-python-client / google-auth-oauthlib not installed. "
            "Run: pip install google-api-python-client google-auth-oauthlib"
        )
        return None

    token_path = Path(dc.YOUTUBE_TOKEN_FILE)
    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), dc.YOUTUBE_SCOPES)

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            token_path.parent.mkdir(parents=True, exist_ok=True)
            token_path.write_text(creds.to_json())
            return creds
        except Exception as exc:
            logger.warning("Token refresh failed: %s", exc)

    # Need interactive login
    secrets_file = dc.YOUTUBE_CLIENT_SECRETS_FILE
    if not secrets_file or not Path(secrets_file).exists():
        logger.info(
            "YouTube OAuth client_secrets.json not found at '%s'. "
            "Set YOUTUBE_CLIENT_SECRETS_FILE to enable uploads.",
            secrets_file,
        )
        return None

    try:
        flow = InstalledAppFlow.from_client_secrets_file(secrets_file, dc.YOUTUBE_SCOPES)
        creds = flow.run_local_server(port=0)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json())
        logger.info("YouTube OAuth token saved to %s", token_path)
        return creds
    except Exception as exc:
        logger.error("YouTube OAuth flow failed: %s", exc)
        return None


def _build_youtube_service(credentials):
    """Build the YouTube Data API v3 service object."""
    try:
        from googleapiclient.discovery import build
        return build("youtube", "v3", credentials=credentials)
    except Exception as exc:
        logger.error("Failed to build YouTube service: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def upload_video(
    article_id: int,
    video_path: str | Path,
    headline: str,
    full_script: str = "",
    category: str = "",
    privacy: str = "",
    schedule_time: str = "",
) -> Dict[str, Any]:
    """Upload a video to YouTube and return metadata.

    Returns a dict with keys: ``video_id``, ``url``, ``status``, ``error``.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        return {"video_id": "", "url": "", "status": "error",
                "error": f"Video file not found: {video_path}"}

    privacy = privacy or dc.YOUTUBE_PRIVACY_STATUS
    title = _build_title(headline)
    description = _build_description(headline, full_script, category)
    tags = _build_tags(headline, category)
    category_id = dc.YOUTUBE_CATEGORY_ID or _DEFAULT_CATEGORY

    # --- Mock mode ---
    if dc.DISTRIBUTION_MOCK_MODE:
        mock_id = f"MOCK_{article_id}_{uuid.uuid4().hex[:8]}"
        logger.info("[MOCK] YouTube upload: %s -> %s", video_path.name, mock_id)
        return {
            "video_id": mock_id,
            "url": f"https://www.youtube.com/watch?v={mock_id}",
            "status": "mock_published",
            "error": "",
        }

    # --- Real mode ---
    creds = _get_credentials()
    if creds is None:
        return {"video_id": "", "url": "", "status": "not_configured",
                "error": "YouTube OAuth credentials not available"}

    service = _build_youtube_service(creds)
    if service is None:
        return {"video_id": "", "url": "", "status": "error",
                "error": "Failed to build YouTube API service"}

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": category_id,
            "defaultLanguage": "ur",
            "defaultAudioLanguage": "ur",
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": False,
        },
    }

    if schedule_time and privacy == "private":
        body["status"]["publishAt"] = schedule_time

    try:
        from googleapiclient.http import MediaFileUpload

        media = MediaFileUpload(str(video_path), mimetype="video/mp4", resumable=True)
        request = service.videos().insert(part="snippet,status", body=body, media_body=media)
        response = None
        while response is None:
            _, response = request.next_chunk()

        video_id = response["id"]
        url = f"https://www.youtube.com/watch?v={video_id}"
        logger.info("YouTube upload succeeded: article=%d, video_id=%s", article_id, video_id)
        return {"video_id": video_id, "url": url, "status": "published", "error": ""}
    except Exception as exc:
        logger.error("YouTube upload failed for article %d: %s", article_id, exc)
        return {"video_id": "", "url": "", "status": "error", "error": str(exc)}


def set_thumbnail(video_id: str, thumbnail_path: str | Path) -> Dict[str, Any]:
    """Set a custom thumbnail for an uploaded video."""
    thumbnail_path = Path(thumbnail_path)
    if not thumbnail_path.exists():
        return {"status": "error", "error": f"Thumbnail not found: {thumbnail_path}"}

    if dc.DISTRIBUTION_MOCK_MODE:
        logger.info("[MOCK] YouTube thumbnail set: %s -> %s", video_id, thumbnail_path.name)
        return {"status": "mock_ok", "error": ""}

    creds = _get_credentials()
    if creds is None:
        return {"status": "not_configured", "error": "YouTube credentials not available"}

    service = _build_youtube_service(creds)
    if service is None:
        return {"status": "error", "error": "Failed to build YouTube API service"}

    try:
        from googleapiclient.http import MediaFileUpload

        media = MediaFileUpload(str(thumbnail_path), mimetype="image/jpeg")
        service.thumbnails().set(videoId=video_id, media_body=media).execute()
        logger.info("YouTube thumbnail set: video_id=%s", video_id)
        return {"status": "ok", "error": ""}
    except Exception as exc:
        logger.warning("YouTube thumbnail set failed for %s: %s", video_id, exc)
        return {"status": "error", "error": str(exc)}


def run(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Batch upload videos for a list of articles.

    Each article dict should contain: ``article_id``, ``video_path``
    (or ``subtitled_video``), ``headline``, ``full_script``, ``category``.
    Returns a list of result dicts.
    """
    results: List[Dict[str, Any]] = []
    for art in articles:
        video_path = art.get("subtitled_video") or art.get("video_path", "")
        if not video_path or not Path(video_path).exists():
            results.append({
                "article_id": art.get("article_id"),
                "video_id": "", "url": "", "status": "skipped",
                "error": "No valid video path",
            })
            continue

        result = upload_video(
            article_id=art.get("article_id", 0),
            video_path=video_path,
            headline=art.get("headline", ""),
            full_script=art.get("full_script", ""),
            category=art.get("category", ""),
            schedule_time=dc.YOUTUBE_SCHEDULE_TIME,
        )

        # Set thumbnail if upload succeeded
        thumb = art.get("thumbnail_path", "")
        if result["status"] == "published" and thumb:
            thumb_result = set_thumbnail(result["video_id"], thumb)
            result["thumbnail_status"] = thumb_result["status"]

        result["article_id"] = art.get("article_id")
        results.append(result)

    ok = sum(1 for r in results if r["status"] in ("published", "mock_published"))
    logger.info("YouTube upload batch: %d/%d succeeded", ok, len(results))
    return results
