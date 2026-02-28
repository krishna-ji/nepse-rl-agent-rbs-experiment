"""
Base scraper for nepsetrading.com API.

All NepseTrading endpoint-specific scrapers derive from this class,
which adds NepseTrading-specific headers and base URL to :class:`BaseScraper`.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests


class BaseScraper:
    """
    Base class for all API scrapers.
    
    Provides common HTTP functionality with session management,
    timeout handling, and response processing.
    """
    
    BASE_URL: str = ""
    DEFAULT_HEADERS: Dict[str, str] = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:148.0) Gecko/20100101 Firefox/148.0",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    def __init__(
        self,
        timeout: int = 30,
        out_dir: Optional[Path] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> None:
        self.timeout = timeout
        self.out_dir = out_dir or Path("data")  
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Create HTTP session with headers
        self.session = requests.Session()
        self.session.headers.update(self.DEFAULT_HEADERS)
        
        # Ensure output directory exists
        self.out_dir.mkdir(parents=True, exist_ok=True)
    
    def get(
        self, 
        path: str, 
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        url = f"{self.BASE_URL.rstrip('/')}/{path.lstrip('/')}"
        
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.timeout,
                    **kwargs
                )
                response.raise_for_status()
                
                if not response.text.strip():
                    raise requests.RequestException(f"Empty response from {url}")
                
                return response.json()
                
            except (requests.RequestException, ValueError) as e:
                if attempt == self.max_retries:
                    raise requests.RequestException(f"Failed to fetch {url} after {self.max_retries + 1} attempts: {e}")
                
                if self.retry_delay > 0:
                    time.sleep(self.retry_delay * (2 ** attempt))


class NepseTradingScraper(BaseScraper):
    """
    Abstract base for all nepsetrading.com API scrapers.

    The NepseTrading API doesn't require authentication but has rate limiting.
    """

    BASE_URL = "https://api.nepsetrading.com"
    DEFAULT_HEADERS: dict[str, str] = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:148.0) Gecko/20100101 Firefox/148.0"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "Referer": "https://nepsetrading.com/",
        "Origin": "https://nepsetrading.com",
    }
