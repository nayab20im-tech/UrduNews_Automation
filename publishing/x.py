"""
publishing/x.py
================
Publishes content to X (Twitter) using the tweepy package.
"""

import os
import tweepy
from typing import Dict, Any
from data_acquisition.utils.logger import get_logger

logger = get_logger("publishing.x")

class TwitterPublisher:
    def __init__(self):
        self.api_key = os.getenv("TWITTER_API_KEY", "")
        self.api_secret = os.getenv("TWITTER_API_SECRET", "")
        self.access_token = os.getenv("TWITTER_ACCESS_TOKEN", "")
        self.access_token_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET", "")
        self.bearer_token = os.getenv("TWITTER_BEARER_TOKEN", "")
        
        if self.api_key and self.api_secret and self.access_token and self.access_token_secret:
            self.client = tweepy.Client(
                bearer_token=self.bearer_token,
                consumer_key=self.api_key,
                consumer_secret=self.api_secret,
                access_token=self.access_token,
                access_token_secret=self.access_token_secret
            )
        else:
            self.client = None
            
    def is_configured(self) -> bool:
        return bool(self.client)
        
    def post(self, text: str) -> Dict[str, Any]:
        if not self.is_configured():
            return {"post_id": "", "url": "", "status": "not_configured", "error": "Missing credentials"}
            
        try:
            response = self.client.create_tweet(text=text)
            if response.data:
                tweet_id = response.data.get("id", "")
                return {
                    "post_id": tweet_id,
                    "url": f"https://x.com/user/status/{tweet_id}",
                    "status": "posted",
                    "error": ""
                }
            else:
                return {"post_id": "", "url": "", "status": "error", "error": "Unknown API error"}
        except tweepy.TweepyException as e:
            logger.error(f"Tweepy Error: {e}")
            return {"post_id": "", "url": "", "status": "error", "error": str(e)}
        except Exception as e:
            logger.error(f"Failed to post to X: {e}")
            return {"post_id": "", "url": "", "status": "error", "error": str(e)}
