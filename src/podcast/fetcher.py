import feedparser
import trafilatura
from ddgs import DDGS


def fetch_rss_articles(feed_urls: list[str], max_per_feed: int = 10) -> list[dict]:
    """Fetch recent articles from RSS feeds."""
    articles = []
    seen_urls = set()

    for url in feed_urls:
        try:
            feed = feedparser.parse(url)
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
        except Exception:
            continue

    return articles


def search_web_articles(interests: str, max_results: int = 10) -> list[dict]:
    """Search DuckDuckGo for recent articles matching interests."""
    # Build a concise query from interests
    lines = [l.strip("- ").strip() for l in interests.splitlines() if l.strip().startswith("-")]
    query = "latest news " + ", ".join(lines[:3]) if lines else "latest tech news"

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
    except Exception:
        pass

    return articles


def extract_article_text(url: str) -> str | None:
    """Extract clean article text from a URL."""
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            return trafilatura.extract(downloaded)
    except Exception:
        pass
    return None
