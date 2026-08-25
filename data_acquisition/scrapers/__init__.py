from .base_scraper import BaseScraper
from .news_website_scraper import NewsWebsiteScraper
from .rss_scraper import RSSScraper
from .news_api_client import NewsAPIClient
from .trends_scraper import GoogleTrendsScraper
from .social_media_scraper import SocialMediaScraper
from .manual_input_loader import ManualInputLoader

__all__ = [
    "BaseScraper",
    "NewsWebsiteScraper",
    "RSSScraper",
    "NewsAPIClient",
    "GoogleTrendsScraper",
    "SocialMediaScraper",
    "ManualInputLoader",
]
