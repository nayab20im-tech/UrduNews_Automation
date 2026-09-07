"""
publishing/youtube.py
=====================
Publishes content to YouTube. 
Wraps the existing robust Google API OAuth implementation.
"""

from typing import Dict, Any, Optional
from pathlib import Path
from distribution import youtube_uploader

class YouTubePublisher:
    def __init__(self):
        pass
        
    def post(
        self, 
        video_path: str, 
        headline: str, 
        full_script: str = "", 
        category: str = "", 
        thumbnail_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Uploads a video to YouTube and sets its thumbnail."""
        
        # Use article_id = 9999 for bulletin or standalone uploads
        result = youtube_uploader.upload_video(
            article_id=9999,
            video_path=video_path,
            headline=headline,
            full_script=full_script,
            category=category
        )
        
        if result.get("status") in ("published", "mock_published") and thumbnail_path:
            youtube_uploader.set_thumbnail(result["video_id"], thumbnail_path)
            
        return result
