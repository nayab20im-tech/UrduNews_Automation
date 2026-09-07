"""
publishing/facebook.py
======================
Publishes content to Facebook using the official facebook-sdk package.
"""

import os
import facebook
from typing import Dict, Any
from data_acquisition.utils.logger import get_logger

logger = get_logger("publishing.facebook")

class FacebookPublisher:
    def __init__(self):
        self.access_token = os.getenv("FACEBOOK_ACCESS_TOKEN", "")
        self.page_id = os.getenv("FACEBOOK_PAGE_ID", "me")
        if self.access_token:
            self.graph = facebook.GraphAPI(access_token=self.access_token, version="3.1")
        else:
            self.graph = None
            
    def is_configured(self) -> bool:
        return bool(self.graph)
        
    def post(self, message: str, link: str = "") -> Dict[str, Any]:
        if not self.is_configured():
            return {"post_id": "", "url": "", "status": "not_configured", "error": "Missing credentials"}
            
        try:
            kwargs = {"message": message}
            if link:
                kwargs["link"] = link
                
            response = self.graph.put_object(parent_object=self.page_id, connection_name='feed', **kwargs)
            post_id = response.get("id", "")
            return {
                "post_id": post_id,
                "url": f"https://facebook.com/{post_id}",
                "status": "posted",
                "error": ""
            }
        except facebook.GraphAPIError as e:
            logger.error(f"Facebook Graph API Error: {e}")
            return {"post_id": "", "url": "", "status": "error", "error": str(e)}
        except Exception as e:
            logger.error(f"Failed to post to Facebook: {e}")
            return {"post_id": "", "url": "", "status": "error", "error": str(e)}
