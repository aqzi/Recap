import os

from core.llm import generate_podcast_script, rank_articles
from podcast.fetcher import extract_article_text, fetch_rss_articles, search_web_articles


def generate_podcast(
    interests: str,
    config: dict,
    llm_model: str,
    progress,
) -> tuple[str, str]:
    """Full podcast generation pipeline.

    Returns (script_text, sources_markdown).
    """
    podcast_cfg = config.get("podcast", {})
    sources_cfg = config.get("sources", {})

    max_articles = podcast_cfg.get("max_articles", 5)
    style = podcast_cfg.get("style", "solo")
    target_length = podcast_cfg.get("target_length", "medium")

    feed_urls = sources_cfg.get("feeds", [
        "https://hnrss.org/newest?points=100",
        "https://feeds.arstechnica.com/arstechnica/technology-lab",
    ])
    web_search_enabled = sources_cfg.get("web_search", True)

    # Stage 1: Fetch articles
    task_fetch = progress.add_task("Fetching articles...", total=None)
    all_articles = fetch_rss_articles(feed_urls)

    if web_search_enabled:
        web_articles = search_web_articles(interests)
        # Dedupe by URL
        seen = {a["url"] for a in all_articles}
        for a in web_articles:
            if a["url"] not in seen:
                all_articles.append(a)
                seen.add(a["url"])

    progress.update(task_fetch, total=1, completed=1)

    if not all_articles:
        raise RuntimeError("No articles found from any source.")

    # Stage 2: Rank articles by relevance
    task_rank = progress.add_task("Ranking articles by relevance...", total=1)
    top_indices = rank_articles(all_articles, interests, max_articles, llm_model)
    selected = [all_articles[i] for i in top_indices]
    progress.update(task_rank, completed=1)

    # Stage 3: Extract full text for selected articles
    task_extract = progress.add_task("Extracting article content...", total=len(selected))
    for article in selected:
        text = extract_article_text(article["url"])
        article["content"] = text or article.get("summary", "")
        progress.advance(task_extract)

    # Stage 4: Generate script
    task_script = progress.add_task("Writing podcast script...", total=1)
    script = generate_podcast_script(selected, interests, style, target_length, llm_model)
    progress.update(task_script, completed=1)

    # Build sources markdown
    sources_md = "# Podcast Sources\n\n"
    for i, a in enumerate(selected, 1):
        sources_md += f"{i}. [{a['title']}]({a['url']})\n"

    return script, sources_md


def write_podcast_output(
    script: str,
    sources_md: str,
    output_dir: str,
) -> tuple[str, str]:
    """Write script.md and sources.md to output directory.

    Returns (script_path, sources_path).
    """
    os.makedirs(output_dir, exist_ok=True)

    script_path = os.path.join(output_dir, "script.md")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write("# Podcast Script\n\n")
        f.write(script)

    sources_path = os.path.join(output_dir, "sources.md")
    with open(sources_path, "w", encoding="utf-8") as f:
        f.write(sources_md)

    return script_path, sources_path
