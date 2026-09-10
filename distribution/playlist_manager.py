"""
distribution/playlist_manager.py
===================================
Stage 4.3 — YouTube playlist management: find or create per-category
playlists and assign uploaded videos to them.

Playlist ID mappings are cached in the ``playlist_map`` DB table so
subsequent runs skip the API search.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from data_acquisition.utils.logger import get_logger
from data_acquisition.utils.text_utils import safe_iso_now

from . import config as dc
from .distribution_db import DistributionDatabase

logger = get_logger("distribution.playlist")


# ---------------------------------------------------------------------------
# YouTube API helpers
# ---------------------------------------------------------------------------

def _get_youtube_service():
    """Build a YouTube service using the same credentials as the uploader."""
    from .youtube_uploader import _get_credentials, _build_youtube_service

    creds = _get_credentials()
    if creds is None:
        return None
    return _build_youtube_service(creds)


def _search_playlist(service, title: str) -> Optional[str]:
    """Search for an existing playlist by title on the authenticated channel."""
    try:
        resp = service.search().list(
            part="snippet",
            q=title,
            type="playlist",
            forMine=True,
            maxResults=5,
        ).execute()
        for item in resp.get("items", []):
            snippet = item.get("snippet", {})
            if snippet.get("title", "").lower() == title.lower():
                return item["id"]["playlistId"]
    except Exception as exc:
        logger.warning("Playlist search failed: %s", exc)
    return None


def _create_playlist(service, title: str, privacy: str = "") -> Optional[str]:
    """Create a new YouTube playlist and return its ID."""
    privacy = privacy or dc.PLAYLIST_PRIVACY
    try:
        resp = service.playlists().insert(
            part="snippet,status",
            body={
                "snippet": {
                    "title": title,
                    "description": f"Auto-generated playlist for {title}",
                    "defaultLanguage": "ur",
                },
                "status": {"privacyStatus": privacy},
            },
        ).execute()
        playlist_id = resp["id"]
        logger.info("Created playlist '%s' -> %s", title, playlist_id)
        return playlist_id
    except Exception as exc:
        logger.error("Playlist creation failed for '%s': %s", title, exc)
        return None


def _add_to_playlist(service, playlist_id: str, video_id: str) -> bool:
    """Add a video to a playlist."""
    try:
        service.playlistItems().insert(
            part="snippet",
            body={
                "snippet": {
                    "playlistId": playlist_id,
                    "resourceId": {
                        "kind": "youtube#video",
                        "videoId": video_id,
                    },
                },
            },
        ).execute()
        logger.info("Added video %s to playlist %s", video_id, playlist_id)
        return True
    except Exception as exc:
        logger.warning("Failed to add %s to playlist %s: %s", video_id, playlist_id, exc)
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_or_create_playlist(
    category: str,
    db: Optional[DistributionDatabase] = None,
) -> Dict[str, Any]:
    """Return the playlist ID for a category, creating it if necessary.

    Uses the DB cache first, then YouTube API search, then creates.
    """
    playlist_name = _build_playlist_name(category)

    # 1. Check DB cache
    if db is not None:
        cached = db.get_playlist(category)
        if cached and cached.get("playlist_id"):
            return {"playlist_id": cached["playlist_id"],
                    "playlist_name": cached["playlist_name"],
                    "status": "cached"}

    # 2. Mock mode
    if dc.DISTRIBUTION_MOCK_MODE:
        mock_id = f"PL_MOCK_{uuid.uuid4().hex[:8]}"
        logger.info("[MOCK] Playlist get_or_create: %s -> %s", category, mock_id)
        if db is not None:
            db.save_playlist(category, mock_id, playlist_name, safe_iso_now())
        return {"playlist_id": mock_id, "playlist_name": playlist_name,
                "status": "mock_created"}

    # 3. YouTube API
    service = _get_youtube_service()
    if service is None:
        return {"playlist_id": "", "playlist_name": playlist_name,
                "status": "not_configured"}

    # Search first
    playlist_id = _search_playlist(service, playlist_name)
    status = "found"

    # Create if not found
    if not playlist_id:
        playlist_id = _create_playlist(service, playlist_name)
        status = "created"

    if playlist_id and db is not None:
        db.save_playlist(category, playlist_id, playlist_name, safe_iso_now())

    return {"playlist_id": playlist_id or "",
            "playlist_name": playlist_name,
            "status": status if playlist_id else "error"}


def add_video_to_playlist(
    video_id: str,
    playlist_id: str,
) -> Dict[str, Any]:
    """Add a video to a specific playlist."""
    if not video_id or not playlist_id:
        return {"status": "skipped", "error": "Missing video_id or playlist_id"}

    if dc.DISTRIBUTION_MOCK_MODE:
        logger.info("[MOCK] Add %s to playlist %s", video_id, playlist_id)
        return {"status": "mock_ok", "error": ""}

    service = _get_youtube_service()
    if service is None:
        return {"status": "not_configured", "error": "YouTube credentials not available"}

    ok = _add_to_playlist(service, playlist_id, video_id)
    return {"status": "ok" if ok else "error",
            "error": "" if ok else "API call failed"}


def _build_playlist_name(category: str) -> str:
    """Build the full playlist name from prefix + category."""
    prefix = dc.PLAYLIST_NAME_PREFIX or dc.CHANNEL_NAME
    if category:
        return f"{prefix} - {category}"
    return prefix


def run(
    articles: List[Dict[str, Any]],
    db: Optional[DistributionDatabase] = None,
) -> List[Dict[str, Any]]:
    """Assign uploaded videos to per-category playlists.

    Each article should have ``article_id``, ``category``, ``youtube_video_id``.
    Returns per-article result dicts.
    """
    results: List[Dict[str, Any]] = []
    _db = db or DistributionDatabase()
    _owned = db is None  # track if we created the db

    try:
        for art in articles:
            video_id = art.get("youtube_video_id", "")
            category = art.get("category", "General")

            if not video_id:
                results.append({
                    "article_id": art.get("article_id"),
                    "playlist_id": "", "status": "skipped",
                    "error": "No youtube_video_id",
                })
                continue

            pl = get_or_create_playlist(category, db=_db)
            playlist_id = pl.get("playlist_id", "")

            if playlist_id:
                add_result = add_video_to_playlist(video_id, playlist_id)
                results.append({
                    "article_id": art.get("article_id"),
                    "playlist_id": playlist_id,
                    "playlist_name": pl.get("playlist_name", ""),
                    "status": add_result["status"],
                    "error": add_result.get("error", ""),
                })
            else:
                results.append({
                    "article_id": art.get("article_id"),
                    "playlist_id": "",
                    "playlist_name": pl.get("playlist_name", ""),
                    "status": pl["status"],
                    "error": "No playlist ID available",
                })

        ok = sum(1 for r in results if r["status"] in ("ok", "mock_ok"))
        logger.info("Playlist batch: %d/%d videos assigned", ok, len(results))
    finally:
        if _owned:
            _db.close()

    return results
