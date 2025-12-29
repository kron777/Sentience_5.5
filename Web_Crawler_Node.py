#!/usr/bin/env python3
"""
Web_Crawler_Node.py
Sentience 5.5 – Controlled web crawling and data assimilation

Purpose:
- Fetch web pages when explicitly triggered
- Extract clean text content
- Log all activity
- Store raw data for evolution/learning
- Never run autonomously — event-driven only
- Doctrine Section 6 compliant
"""

import requests
from bs4 import BeautifulSoup
import time
import os
from typing import Optional, Dict, List
from urllib.parse import urljoin, urlparse

class WebCrawlerNode:
    def __init__(self, memory, log_path: str = "crawl_log.txt"):
        self.memory = memory
        self.log_path = log_path
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Sentience/5.5 (local research agent; +local)"
        })
        self.visited_urls = set()
        self.max_depth = 2
        self.max_pages_per_crawl = 10

        # Create log if not exists
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                f.write(f"Web Crawl Log - Started {time.ctime()}\n\n")

    def log(self, message: str):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] {message}\n"
        print(entry.strip())
        with open(self.log_path, "a") as f:
            f.write(entry)

    def is_valid_url(self, url: str) -> bool:
        parsed = urlparse(url)
        return bool(parsed.scheme and parsed.netloc)

    def fetch_page(self, url: str) -> Optional[str]:
        if not self.is_valid_url(url):
            return None

        if url in self.visited_urls:
            return None

        try:
            self.log(f"Fetching: {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            self.visited_urls.add(url)
            return response.text

        except Exception as e:
            self.log(f"Error fetching {url}: {str(e)}")
            return None

    def extract_text(self, html: str) -> str:
        soup = BeautifulSoup(html, 'html.parser')

        # Remove script and style
        for script in soup(["script", "style", "nav", "footer", "aside"]):
            script.decompose()

        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)

        return text[:10000]  # Truncate to prevent bloat

    def crawl(self, start_url: str, query: str = "") -> List[Dict]:
        """
        Perform a focused crawl starting from start_url
        Returns list of assimilated page data
        """
        if not self.is_valid_url(start_url):
            self.log(f"Invalid start URL: {start_url}")
            return []

        results = []
        queue = [(start_url, 0)]  # (url, depth)
        pages_crawled = 0

        self.log(f"Starting focused crawl on: {start_url} (query context: {query})")

        while queue and pages_crawled < self.max_pages_per_crawl:
            url, depth = queue.pop(0)

            html = self.fetch_page(url)
            if not html:
                continue

            text = self.extract_text(html)

            page_data = {
                "url": url,
                "title": BeautifulSoup(html, 'html.parser').title.string if BeautifulSoup(html, 'html.parser').title else "No title",
                "content": text,
                "timestamp": time.time(),
                "depth": depth,
                "context_query": query
            }

            results.append(page_data)
            pages_crawled += 1

            # Store in memory for later evolution
            self.memory.store(
                text=f"CRAWL_RESULT: {url}",
                classification="web_data",
                response=str(page_data)
            )

            self.log(f"Assimilated {len(text)} chars from {url}")

            # Follow links only if depth allows and query is relevant
            if depth < self.max_depth:
                soup = BeautifulSoup(html, 'html.parser')
                for link in soup.find_all('a', href=True)[:8]:  # limit branching
                    next_url = urljoin(url, link['href'])
                    if self.is_valid_url(next_url) and next_url not in self.visited_urls:
                        if query.lower() in link.get_text().lower() or query.lower() in next_url.lower():
                            queue.append((next_url, depth + 1))

        self.log(f"Crawl complete. Assimilated {len(results)} pages.")
        return results

    def search_and_crawl(self, query: str, start_urls: List[str] = None) -> List[Dict]:
        """
        High-level interface: search-like behavior via known start points
        """
        default_starts = [
            "https://en.wikipedia.org/wiki/Main_Page",
            "https://arxiv.org",
            "https://news.ycombinator.com",
        ]

        urls = start_urls or default_starts

        all_results = []
        for url in urls:
            if query.lower() in url or "wiki" in url:
                results = self.crawl(url, query=query)
                all_results.extend(results)

        return all_results
