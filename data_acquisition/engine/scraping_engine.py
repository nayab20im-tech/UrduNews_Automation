"""
engine/scraping_engine.py
==========================
Implements Section 1's "Web Scraping Engine" box:
    Python + BeautifulSoup (default) with a Selenium fallback for
    JavaScript-heavy pages.

This is the single place that actually talks to news websites. Every
scraper in `scrapers/` goes through this engine instead of calling
`requests` directly, so retry policy, rate limiting, User-Agent rotation,
and robots.txt compliance are enforced consistently everywhere.
"""

from __future__ import annotations

import random
import time
import urllib.robotparser as robotparser
from typing import Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .. import config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ScrapingEngine:
    """Reusable HTTP + parsing engine shared by all scrapers.

    Parameters
    ----------
    respect_robots_txt: honour robots.txt disallow rules (default: True,
        overridable via config.RESPECT_ROBOTS_TXT).
    min_delay: minimum seconds between two requests to the *same* host,
        to stay polite and avoid tripping rate limits.
    """

    def __init__(
        self,
        timeout: int = config.REQUEST_TIMEOUT,
        retries: int = config.REQUEST_RETRIES,
        backoff_factor: float = config.REQUEST_BACKOFF_FACTOR,
        min_delay: float = config.MIN_DELAY_BETWEEN_REQUESTS,
        respect_robots_txt: bool = config.RESPECT_ROBOTS_TXT,
    ):
        self.timeout = timeout
        self.min_delay = min_delay
        self.respect_robots_txt = respect_robots_txt

        self._session = requests.Session()
        retry_strategy = Retry(
            total=retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET", "HEAD"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

        self._last_request_time: dict[str, float] = {}
        self._robots_cache: dict[str, robotparser.RobotFileParser] = {}

    # -- politeness helpers ------------------------------------------------
    def _random_headers(self) -> dict:
        return {
            "User-Agent": random.choice(config.USER_AGENTS),
            "Accept-Language": "en-US,en;q=0.9,ur;q=0.8",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

    def _throttle(self, host: str) -> None:
        last = self._last_request_time.get(host)
        if last is not None:
            elapsed = time.time() - last
            wait = self.min_delay - elapsed
            if wait > 0:
                time.sleep(wait)
        self._last_request_time[host] = time.time()

    def _is_allowed_by_robots(self, url: str) -> bool:
        if not self.respect_robots_txt:
            return True
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        rp = self._robots_cache.get(base)
        if rp is None:
            rp = robotparser.RobotFileParser()
            rp.set_url(base + "/robots.txt")
            try:
                rp.read()
            except Exception:
                # If robots.txt is unreachable, fail open (allow) rather
                # than blocking collection entirely.
                logger.debug("Could not read robots.txt for %s; allowing by default", base)
                self._robots_cache[base] = rp
                return True
            self._robots_cache[base] = rp
        try:
            return rp.can_fetch(config.USER_AGENTS[0], url)
        except Exception:
            return True

    # -- public fetch API ----------------------------------------------------
    def get(self, url: str) -> Optional[requests.Response]:
        """GET a URL politely (robots.txt + rate limit + retries)."""
        if not self._is_allowed_by_robots(url):
            logger.warning("Blocked by robots.txt, skipping: %s", url)
            return None

        host = urlparse(url).netloc
        self._throttle(host)

        try:
            response = self._session.get(
                url, headers=self._random_headers(), timeout=self.timeout
            )
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            logger.error("Request failed for %s: %s", url, exc)
            return None

    def get_soup(self, url: str, parser: str = "lxml") -> Optional[BeautifulSoup]:
        """GET a URL and return a parsed BeautifulSoup tree, or None."""
        response = self.get(url)
        if response is None:
            return None
        response.encoding = response.encoding or "utf-8"
        try:
            return BeautifulSoup(response.text, parser)
        except Exception as exc:
            logger.error("Failed to parse HTML for %s: %s", url, exc)
            return None

    def get_dynamic_html(self, url: str, wait_seconds: float = 3.0) -> Optional[str]:
        """Render a JavaScript-heavy page with headless Selenium/Chrome.

        Only used when a site's content doesn't appear in the plain HTML
        (BeautifulSoup path). Selenium is an optional dependency: if it
        (or a browser driver) isn't installed, this logs a clear error and
        returns None instead of crashing the whole pipeline.
        """
        if not self._is_allowed_by_robots(url):
            logger.warning("Blocked by robots.txt, skipping (dynamic): %s", url)
            return None

        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
        except ImportError:
            logger.error(
                "Selenium is not installed. Run `pip install selenium` and "
                "ensure a matching browser driver is available to use "
                "get_dynamic_html()."
            )
            return None

        host = urlparse(url).netloc
        self._throttle(host)

        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument(f"user-agent={random.choice(config.USER_AGENTS)}")

        driver = None
        try:
            driver = webdriver.Chrome(options=options)
            driver.set_page_load_timeout(self.timeout)
            driver.get(url)
            time.sleep(wait_seconds)
            return driver.page_source
        except Exception as exc:
            logger.error("Selenium fetch failed for %s: %s", url, exc)
            return None
        finally:
            if driver is not None:
                driver.quit()

    def close(self) -> None:
        self._session.close()
