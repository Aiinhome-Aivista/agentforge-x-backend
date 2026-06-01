"""Web search service.

Provides a single ``search(query, max_results)`` function. The provider is
selected at runtime in this priority order:

  1. Brave Search API   — if BRAVE_SEARCH_API_KEY is set
  2. SerpAPI            — if SERPAPI_KEY is set
  3. DuckDuckGo HTML    — fallback, no API key required (best-effort
                          HTML scraping using only the standard library
                          + the already-installed ``requests`` package)

Each provider returns a normalised list of dicts:

    [
      { "title": "...", "url": "https://...", "snippet": "..." },
      ...
    ]

The fallback scraper deliberately uses only stdlib ``html.parser`` so no
new pip dependencies are introduced — DuckDuckGo's HTML markup occasionally
changes, but this parser is intentionally lenient: when an unrecognised
shape is returned it surfaces a clear "no results" rather than crashing
the request.
"""

from __future__ import annotations

import os
import re
import logging
import urllib.parse
from html.parser import HTMLParser
from typing import List, Dict, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = float(os.getenv("WEB_SEARCH_TIMEOUT", "8.0"))
USER_AGENT = (
    os.getenv("WEB_SEARCH_USER_AGENT")
    or "Mozilla/5.0 (Linux x86_64) AgentForgeX/1.0 Search"
)


# ── Public entry point ─────────────────────────────────────────────────────
def search(query: str, max_results: int = 8) -> Dict:
    """Search the web for ``query``. Returns:

        { "provider": "brave"|"serpapi"|"duckduckgo",
          "results":  [ {title, url, snippet}, ... ],
          "query":    "..." }
    """
    q = (query or "").strip()
    if not q:
        return {"provider": None, "results": [], "query": q}

    n = max(1, min(int(max_results or 8), 20))

    if os.getenv("BRAVE_SEARCH_API_KEY"):
        try:
            return {
                "provider": "brave",
                "query": q,
                "results": _brave(q, n),
            }
        except Exception as e:
            logger.warning("Brave search failed, falling back: %s", e)

    if os.getenv("SERPAPI_KEY"):
        try:
            return {
                "provider": "serpapi",
                "query": q,
                "results": _serpapi(q, n),
            }
        except Exception as e:
            logger.warning("SerpAPI failed, falling back: %s", e)

    return {
        "provider": "duckduckgo",
        "query": q,
        "results": _duckduckgo(q, n),
    }


def provider_name() -> str:
    """Which provider would `search` use right now? (for /config endpoints)"""
    if os.getenv("BRAVE_SEARCH_API_KEY"):
        return "brave"
    if os.getenv("SERPAPI_KEY"):
        return "serpapi"
    return "duckduckgo"


# ── Provider: Brave Search API ─────────────────────────────────────────────
def _brave(q: str, n: int) -> List[Dict]:
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": os.environ["BRAVE_SEARCH_API_KEY"],
    }
    params = {"q": q, "count": n, "safesearch": "moderate"}
    r = requests.get(
        "https://api.search.brave.com/res/v1/web/search",
        headers=headers, params=params, timeout=DEFAULT_TIMEOUT,
    )
    r.raise_for_status()
    body = r.json() or {}
    web = ((body.get("web") or {}).get("results") or [])
    out = []
    for item in web[:n]:
        out.append({
            "title":   _strip_html(item.get("title") or "")[:200],
            "url":     item.get("url") or "",
            "snippet": _strip_html(item.get("description") or "")[:500],
        })
    return out


# ── Provider: SerpAPI ──────────────────────────────────────────────────────
def _serpapi(q: str, n: int) -> List[Dict]:
    params = {
        "engine":  "google",
        "q":       q,
        "num":     n,
        "api_key": os.environ["SERPAPI_KEY"],
    }
    r = requests.get("https://serpapi.com/search.json",
                     params=params, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    body = r.json() or {}
    out = []
    for item in (body.get("organic_results") or [])[:n]:
        out.append({
            "title":   _strip_html(item.get("title") or "")[:200],
            "url":     item.get("link") or item.get("url") or "",
            "snippet": _strip_html(item.get("snippet") or "")[:500],
        })
    return out


# ── Provider: DuckDuckGo HTML (no API key) ─────────────────────────────────
def _duckduckgo(q: str, n: int) -> List[Dict]:
    """Scrape DuckDuckGo's lite HTML SERP. No key, no client cookies."""
    try:
        params = {"q": q}
        r = requests.post(
            "https://html.duckduckgo.com/html/",
            data=params,
            headers={
                "User-Agent": USER_AGENT,
                "Accept-Language": "en-US,en;q=0.9",
            },
            timeout=DEFAULT_TIMEOUT,
        )
        r.raise_for_status()
        return _parse_ddg_html(r.text, n)
    except Exception as e:
        logger.warning("DuckDuckGo HTML fetch failed: %s", e)
        return []


# ── DuckDuckGo HTML parser (stdlib only) ───────────────────────────────────
class _DDGParser(HTMLParser):
    """Pulls (title, url, snippet) tuples out of the DuckDuckGo HTML SERP.

    DuckDuckGo's lite layout uses `<a class="result__a" href="...">title</a>`
    for each title and `<a class="result__snippet">snippet</a>` for the
    description. URLs are wrapped in DDG's own redirector
    (//duckduckgo.com/l/?uddg=...) which we unwrap below.
    """

    def __init__(self) -> None:
        super().__init__()
        self.results: List[Dict[str, str]] = []
        # Per-record state.
        self._capture: Optional[str] = None         # 'title' | 'snippet'
        self._buf: List[str] = []
        self._cur: Dict[str, str] = {}

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        a = dict(attrs)
        cls = (a.get("class") or "")
        if "result__a" in cls:
            href = a.get("href") or ""
            self._cur = {"url": _clean_ddg_url(href), "title": "", "snippet": ""}
            self._capture = "title"
            self._buf = []
        elif "result__snippet" in cls and self._cur:
            self._capture = "snippet"
            self._buf = []

    def handle_endtag(self, tag):
        if tag != "a" or self._capture is None:
            return
        text = _collapse_ws("".join(self._buf))
        if self._capture == "title":
            self._cur["title"] = text[:200]
        elif self._capture == "snippet":
            self._cur["snippet"] = text[:500]
            if self._cur.get("url") and self._cur.get("title"):
                self.results.append(self._cur)
            self._cur = {}
        self._capture = None
        self._buf = []

    def handle_data(self, data):
        if self._capture is not None:
            self._buf.append(data)


def _parse_ddg_html(html_text: str, n: int) -> List[Dict[str, str]]:
    p = _DDGParser()
    try:
        p.feed(html_text)
    except Exception as e:
        logger.debug("DDG parser bailed: %s", e)
    return p.results[:n]


def _clean_ddg_url(href: str) -> str:
    """Unwrap DuckDuckGo's redirect URL (//duckduckgo.com/l/?uddg=...)."""
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    try:
        parsed = urllib.parse.urlparse(href)
        if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
            qs = urllib.parse.parse_qs(parsed.query)
            wrapped = (qs.get("uddg") or [""])[0]
            if wrapped:
                return urllib.parse.unquote(wrapped)
    except Exception:
        pass
    return href


# ── Tiny utilities ─────────────────────────────────────────────────────────
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _strip_html(s: str) -> str:
    if not s:
        return ""
    return _collapse_ws(_TAG_RE.sub(" ", s))


def _collapse_ws(s: str) -> str:
    return _WS_RE.sub(" ", s or "").strip()
