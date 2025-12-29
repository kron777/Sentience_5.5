#!/usr/bin/env python3
"""
Web_IO_Node — Sentience 5.5

Role:
- Controlled read-only internet access
- Fetch real webpages on demand
- Extract factual text only
- No hallucination, no guessing
- Return source + confidence
- Designed to be governed by Orchestrator / Self_Model

This node DOES NOT decide on its own.
It responds only when asked.
"""

import requests
import time
import re
from typing import Dict, Any, Optional
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from datetime import datetime, timezone


# ---------------------------------------------------------------------
# Logging (quiet by default)
# ---------------------------------------------------------------------
def log(level: str, msg: str):
    ts = datetime.now(timezone.utc).isoformat()
    print(f"[{ts}] [WEB_IO] [{level}] {msg}")


# ---------------------------------------------------------------------
# Web IO Node
# ---------------------------------------------------------------------
class WebIONode:
    def __init__(self, timeout: int = 10):
        self.node_name = "Web_IO_Node"
        self.timeout = timeout

        # Very conservative allowed domains (expand later)
        self.allowed_domains = {
            "wikipedia.org",
            "github.com",
            "raw.githubusercontent.com",
            "docs.python.org",
            "stackoverflow.com"
        }

        log("INFO", "Web_IO_Node initialized")

    # -----------------------------------------------------------------
    # Domain control
    # -----------------------------------------------------------------
    def _domain_allowed(self, url: str) -> bool:
        try:
            domain = urlparse(url).netloc.lower()
            return any(domain.endswith(d) for d in self.allowed_domains)
        except Exception:
            return False

    # -----------------------------------------------------------------
    # Fetch page
    # -----------------------------------------------------------------
    def fetch(self, url: str) -> Dict[str, Any]:
        """
        Fetch and extract text from a webpage.
        Returns structured, grounded data.
        """
        if not self._domain_allowed(url):
            return {
                "success": False,
                "error": "Domain not permitted",
                "url": url
            }

        try:
            headers = {
                "User-Agent": "Sentience/5.5 (grounded research node)"
            }

            resp = requests.get(url, headers=headers, timeout=self.timeout)
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")

            # Remove scripts, styles, junk
            for tag in soup(["script", "style", "noscript", "header", "footer"]):
                tag.decompose()

            text = soup.get_text(separator=" ", strip=True)
            text = self._clean_text(text)

            return {
                "success": True,
                "url": url,
                "domain": urlparse(url).netloc,
                "retrieved_at": time.time(),
                "content": text[:5000],  # hard cap to prevent memory bloat
                "confidence": 0.95,
                "source_type": "web"
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "url": url
            }

    # -----------------------------------------------------------------
    # Text cleanup
    # -----------------------------------------------------------------
    def _clean_text(self, text: str) -> str:
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # -----------------------------------------------------------------
    # High-level query helper
    # -----------------------------------------------------------------
    def research(self, topic: str) -> Dict[str, Any]:
        """
        Very conservative research helper.
        Currently Wikipedia-first.
        """
        wiki_url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
        result = self.fetch(wiki_url)

        if not result.get("success"):
            return {
                "success": False,
                "message": "Unable to verify information online",
                "topic": topic
            }

        return {
            "success": True,
            "topic": topic,
            "summary": result["content"][:1500],
            "source": result["url"],
            "confidence": result["confidence"]
        }
