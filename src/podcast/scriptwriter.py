import os

from core.llm import generate_podcast_script, rank_articles
from podcast.fetcher import extract_article_text, fetch_rss_articles, search_web_articles


def generate_podcast(
    input_text: str,
    config: dict,
    llm_model: str,
    progress,
    source_files: list[str] | None = None,
    kb=None,
    output_language: str = "en",
) -> tuple[str, str]:
    """Full podcast generation pipeline.

    Args:
        input_text: Primary source material for the podcast.
        source_files: List of input file paths for the sources markdown.
        kb: Optional KnowledgeBase instance for RAG context.

    Returns (script_text, sources_markdown).
    """
    podcast_cfg = config.get("podcast", {})
    enrichment_cfg = config.get("enrichment", {})

    max_articles = podcast_cfg.get("max_articles", 5)
    style = podcast_cfg.get("style", "solo")
    target_length = podcast_cfg.get("target_length", "medium")

    feed_urls = enrichment_cfg.get("feeds", [])
    web_search_enabled = enrichment_cfg.get("web_search", False)
    enrichment_enabled = bool(feed_urls) or web_search_enabled

    selected = []

    if enrichment_enabled:
        # Stage 1: Fetch articles
        task_fetch = progress.add_task("Fetching articles...", total=None)
        all_articles = fetch_rss_articles(feed_urls) if feed_urls else []

        if web_search_enabled:
            web_articles = search_web_articles(input_text)
            seen = {a["url"] for a in all_articles}
            for a in web_articles:
                if a["url"] not in seen:
                    all_articles.append(a)
                    seen.add(a["url"])

        progress.update(task_fetch, total=1, completed=1)

        if all_articles:
            # Stage 2: Rank articles by relevance to input text
            task_rank = progress.add_task("Ranking articles by relevance...", total=1)
            top_indices = rank_articles(all_articles, input_text, max_articles, llm_model)
            selected = [all_articles[i] for i in top_indices]
            progress.update(task_rank, completed=1)

            # Stage 3: Extract full text for selected articles
            if selected:
                task_extract = progress.add_task("Extracting article content...", total=len(selected))
                for article in selected:
                    text = extract_article_text(article["url"])
                    article["content"] = text or article.get("summary", "")
                    progress.advance(task_extract)

    # Stage 4: Generate script
    task_script = progress.add_task("Writing podcast script...", total=1)
    script_kb = None
    if kb:
        script_kb = kb.retrieve(input_text[:1000], top_k=5, max_chars=3000)
        script_kb = script_kb or None
    script = generate_podcast_script(
        input_text, style, target_length, llm_model,
        articles=selected or None,
        kb_context=script_kb,
        output_language=output_language,
    )
    progress.update(task_script, completed=1)

    # Build sources markdown
    sources_md = "# Podcast Sources\n\n"
    sources_md += "## Primary Input\n"
    if source_files:
        for f in source_files:
            sources_md += f"- {f}\n"
    else:
        sources_md += "- (provided text)\n"
    if selected:
        sources_md += "\n## Supplementary Articles\n"
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
        f.write(script)

    sources_path = os.path.join(output_dir, "sources.md")
    with open(sources_path, "w", encoding="utf-8") as f:
        f.write(sources_md)

    return script_path, sources_path
