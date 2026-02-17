#!/usr/bin/env python3
"""
Robust Web Search Engine for GPTSwarm.
Uses ddgs (DuckDuckGo Search) with 'lite' backend + Wikipedia API.
"""

import hashlib
import re
import time
import random
import requests
from urllib.parse import quote_plus
from threading import Lock

from swarm.utils.log import logger

try:
    from ddgs import DDGS
    HAS_DDGS = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        HAS_DDGS = True
    except ImportError:
        HAS_DDGS = False


class MCPWebSearchEngine:
    """Robust search engine using ddgs lite backend + Wikipedia."""

    _global_lock = Lock()
    _global_last_request = 0
    _global_cache = {}
    _global_cache_lock = Lock()

    def __init__(self):
        self._min_interval = 3.0  # seconds between requests

    def _global_rate_limit(self):
        """Enforce global rate limiting."""
        with MCPWebSearchEngine._global_lock:
            now = time.time()
            elapsed = now - MCPWebSearchEngine._global_last_request
            if elapsed < self._min_interval:
                wait = self._min_interval - elapsed + random.uniform(0.2, 0.8)
                time.sleep(wait)
            MCPWebSearchEngine._global_last_request = time.time()

    def _cache_key(self, query: str) -> str:
        return hashlib.md5(query.lower().strip().encode()).hexdigest()

    def _clean_query(self, query: str) -> str:
        """Clean up LLM-generated search queries."""
        # Remove markdown formatting
        query = re.sub(r'\*\*([^*]+)\*\*', r'\1', query)
        query = re.sub(r'\*([^*]+)\*', r'\1', query)
        # Remove quotes that might wrap the entire query
        query = query.strip('"').strip("'").strip()
        # Remove numbered list prefixes
        query = re.sub(r'^\d+[\.\)]\s*', '', query)
        # Remove "Search Queries" or similar headers
        query = re.sub(r'^(Search\s+Quer(y|ies)|Queries?)\s*:?\s*', '', query, flags=re.IGNORECASE)
        # Remove newlines - take only first line
        query = query.split('\n')[0].strip()
        # Limit length
        if len(query) > 150:
            query = query[:150]
        return query

    def search(self, query: str, num: int = 3) -> str:
        """Search using ddgs lite + Wikipedia with caching."""
        query = self._clean_query(query)
        if not query:
            return "No valid search query provided."

        key = self._cache_key(query)
        with MCPWebSearchEngine._global_cache_lock:
            if key in MCPWebSearchEngine._global_cache:
                return MCPWebSearchEngine._global_cache[key]

        backends = [
            ("DDGS-lite", self._ddgs_search),
            ("Wikipedia", self._wikipedia_search),
        ]

        for name, backend_fn in backends:
            try:
                self._global_rate_limit()
                result = backend_fn(query, num)
                if result and result.strip() and not result.startswith("No search"):
                    with MCPWebSearchEngine._global_cache_lock:
                        MCPWebSearchEngine._global_cache[key] = result
                    return result
            except Exception as e:
                logger.debug(f"{name} search failed: {e}")
                continue

        return f"No search results found for: {query}"

    def _ddgs_search(self, query: str, num: int = 3) -> str:
        """Search via ddgs using 'lite' backend (avoids Brave rate limiting)."""
        if not HAS_DDGS:
            return ""
        
        for backend in ['lite', 'html']:
            try:
                ddgs = DDGS()
                results = list(ddgs.text(query, max_results=num, backend=backend))
                if results:
                    snippets = []
                    for r in results[:num]:
                        body = r.get("body", "")
                        title = r.get("title", "")
                        if body:
                            snippets.append(f"{title}: {body}" if title else body)
                    if snippets:
                        return '\n'.join(snippets)
            except Exception as e:
                logger.debug(f"DDGS {backend} failed: {e}")
                continue
        return ""

    def _wikipedia_search(self, query: str, num: int = 3) -> str:
        """Search via Wikipedia API."""
        wiki_url = (
            f"https://en.wikipedia.org/w/api.php?"
            f"action=query&list=search&srsearch={quote_plus(query)}"
            f"&format=json&srlimit={num}"
        )
        resp = requests.get(wiki_url, headers={
            "User-Agent": "GPTSwarm/1.0 (research project)"
        }, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("query", {}).get("search", [])
        if results:
            snippets = []
            for r in results[:num]:
                snippet = re.sub(r'<[^>]+>', '', r.get("snippet", ""))
                snippets.append(f"{r['title']}: {snippet}")
            return '\n'.join(snippets)
        return ""


if __name__ == "__main__":
    engine = MCPWebSearchEngine()
    queries = [
        "What is the capital of Australia?",
        "Eliud Kipchoge marathon world record pace",
        "Mercedes Sosa studio albums 2000 2009",
    ]
    for q in queries:
        result = engine.search(q, num=2)
        print(f"Q: {q}")
        print(f"R: {result[:200]}")
        print()
