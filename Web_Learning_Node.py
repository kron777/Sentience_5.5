#!/usr/bin/env python3
"""
Web_Learning_Node
Sentience 5.5

Role:
- Controlled internet reading
- Source-attributed learning
- Zero hallucination
"""

import requests
import time
from datetime import datetime, timezone
from typing import Dict
from bs4 import BeautifulSoup


def log(level: str, msg: str):
    ts = datetime.now(timezone.utc).isoformat()
    print(f"[{ts}] [WEB_IO] [{level}] {msg}")


class WebLearningNode:
    def __init__(self, memory_node=None):
        self.node_name = "Web_Learning_Node"
        self.memory = memory_node
        self.timeout = 10
        self.max_chars = 8000

        log("INFO", "Web learning node initialized (read-only, controlled)")

    # -------------------------------------------------
    # Public interface
    # -------------------------------------------------
    def fetch_and_learn(self, url: str) -> Dict:
        """
        Fetch a webpage and extract grounded knowledge.
        """
        if not url.startswith(("http://", "https://")):
            return self._fail("Invalid URL")

        try:
            response = requests.get(
                url,
                timeout=self.timeout,
                headers={"User-Agent": "Sentience/5.5"}
            )
        except Exception as e:
            return self._fail(str(e))

        if response.status_code != 200:
            return self._fail(f"HTTP {response.status_code}")

        text = self._extract_text(response.text)

        if not text:
            return self._fail("No readable content found")

        summary = self._summarize(text)

        record = {
            "source": url,
            "timestamp": time.time(),
            "summary": summary
        }

        if self.memory:
            self.memory.store_external_knowledge(record)

        return {
            "status": "learned",
            "source": url,
            "summary": summary
        }

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------
    def _extract_text(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")

        # Remove scripts and junk
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        return text[:self.max_chars]

    def _summarize(self, text: str) -> str:
        """
        Deterministic summary (NO LLM hallucination)
        """
        sentences = text.split(". ")
        if len(sentences) <= 3:
            return text

        return ". ".join(sentences[:3]) + "."

    def _fail(self, reason: str) -> Dict:
        return {
            "status": "failed",
            "reason": reason
        }
