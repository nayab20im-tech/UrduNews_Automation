"""
pipeline/media_matcher.py
=========================
Matches and scores scraped media (images/videos) against the article content.
Filters out irrelevant stock images and promotes highly relevant ones.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import re

from data_acquisition.utils.logger import get_logger

logger = get_logger("pipeline.media_matcher")

@dataclass
class MediaRelevanceResult:
    article_id: int
    image_url: str = ""
    video_url: str = ""
    relevance_score: float = 0.0
    status: str = "ok"


class MediaMatcher:
    def __init__(self, relevance_threshold: float = 0.3):
        self.relevance_threshold = relevance_threshold
        # Common generic words that indicate a stock image or non-specific logo
        self.stock_keywords = {"logo", "placeholder", "stock", "avatar", "icon", "blank", "default"}

    def run(self, raw_articles: List[Dict[str, Any]]) -> List[MediaRelevanceResult]:
        results = []
        for raw in raw_articles:
            try:
                res = self._process_article(raw)
                results.append(res)
            except Exception as e:
                logger.error("Media matching failed for article %s: %s", raw.get("id"), e)
                results.append(MediaRelevanceResult(article_id=raw["id"], status="error"))
        return results

    def _process_article(self, raw: Dict[str, Any]) -> MediaRelevanceResult:
        article_id = raw["id"]
        title = str(raw.get("title", "")).lower()
        content = str(raw.get("content", "")).lower()
        
        extra_json = raw.get("extra_json")
        if not extra_json:
            return MediaRelevanceResult(article_id=article_id, status="no_media")
            
        try:
            extra = json.loads(extra_json)
        except json.JSONDecodeError:
            return MediaRelevanceResult(article_id=article_id, status="error")

        image_url = extra.get("image_url", "")
        video_url = extra.get("video_url", "")
        
        if not image_url and not video_url:
            return MediaRelevanceResult(article_id=article_id, status="no_media")

        score = self._score_media(title, content, image_url, video_url)
        
        if score < self.relevance_threshold:
            return MediaRelevanceResult(article_id=article_id, relevance_score=score, status="low_relevance")
            
        return MediaRelevanceResult(
            article_id=article_id, 
            image_url=image_url, 
            video_url=video_url, 
            relevance_score=score, 
            status="ok"
        )
        
    def _score_media(self, title: str, content: str, img_url: str, vid_url: str) -> float:
        """
        Heuristic scoring since we are avoiding an extra LLM call here just for URL analysis.
        If the url contains keywords from the title, score goes up.
        If it contains stock keywords, score goes down.
        """
        score = 0.5 # Base score for just having a media URL attached to the feed item
        
        url_text = (img_url + " " + vid_url).lower()
        
        # Penalize generic/stock image keywords in URL
        if any(sk in url_text for sk in self.stock_keywords):
            score -= 0.4
            
        # Extract long words from title to check if URL slug matches them
        title_words = [w for w in re.split(r'\W+', title) if len(w) > 4]
        matches = sum(1 for w in title_words if w in url_text)
        
        if title_words and matches > 0:
            score += 0.2 * (matches / len(title_words))
            
        return min(max(score, 0.0), 1.0)
