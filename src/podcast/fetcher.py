import logging

import feedparser
import trafilatura
from ddgs import DDGS

logger = logging.getLogger(__name__)


RSS_FETCH_TIMEOUT = 30


def fetch_rss_articles(feed_urls: list[str], max_per_feed: int = 10) -> list[dict]:
    """Fetch recent articles from RSS feeds."""
    import urllib.request
    articles = []
    seen_urls = set()

    for url in feed_urls:
        try:
            resp = urllib.request.urlopen(url, timeout=RSS_FETCH_TIMEOUT)
            feed = feedparser.parse(resp.read())
            for entry in feed.entries[:max_per_feed]:
                link = entry.get("link", "")
                if not link or link in seen_urls:
                    continue
                seen_urls.add(link)
                articles.append({
                    "title": entry.get("title", "Untitled"),
                    "url": link,
                    "summary": entry.get("summary", ""),
                    "source": "rss",
                })
        except Exception as e:
            logger.debug("Failed to fetch RSS feed %s: %s", url, e)
            continue

    return articles


def search_web_articles(text: str, max_results: int = 10) -> list[dict]:
    """Search DuckDuckGo for recent articles matching the given text."""
    lines = [l.strip("- ").strip() for l in text.splitlines() if l.strip().startswith("-")]
    if lines:
        query = "latest news " + ", ".join(lines[:3])
    else:
        # Use the first ~30 words as a search query
        words = text.split()[:30]
        query = " ".join(words) if words else "latest tech news"

    articles = []
    try:
        with DDGS() as ddgs:
            results = ddgs.news(query, max_results=max_results)
            for r in results:
                articles.append({
                    "title": r.get("title", "Untitled"),
                    "url": r.get("url", ""),
                    "summary": r.get("body", ""),
                    "source": "web_search",
                })
    except Exception as e:
        logger.debug("Web search failed: %s", e)

    return articles


def extract_article_text(url: str) -> str | None:
    """Extract clean article text from a URL."""
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            return trafilatura.extract(downloaded)
    except Exception as e:
        logger.debug("Failed to extract text from %s: %s", url, e)
    return None
