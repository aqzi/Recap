import json
import logging
import time

import ollama

logger = logging.getLogger(__name__)

from core.prompts import (
    ARTICLE_RANKING_SYSTEM,
    KB_ENHANCE_PODCAST_SYSTEM,
    SOLO_SCRIPT_SYSTEM,
    TWO_HOST_SCRIPT_SYSTEM,
    YOUTUBE_SCORE_SYSTEM,
    article_ranking_prompt,
    chunk_summary_prompt,
    chunk_summary_system,
    consolidation_prompt,
    consolidation_system,
    kb_enhance_podcast_prompt,
    kb_enhance_prompt,
    kb_enhance_system,
    remarks_prompt,
    remarks_system,
    solo_script_prompt,
    two_host_script_prompt,
    youtube_score_prompt,
)

DEFAULT_NUM_CTX = 8192


def call_llm(
    prompt: str,
    system_prompt: str,
    llm_model: str,
    num_ctx: int = DEFAULT_NUM_CTX,
    retries: int = 1,
) -> str:
    for attempt in range(retries + 1):
        try:
            response = ollama.chat(
                model=llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                options={"num_ctx": num_ctx, "temperature": 0.3},
            )
            return response.message.content
        except Exception:
            if attempt < retries:
                time.sleep(2)
                continue
            raise


def summarize_chunk(
    chunk: dict, chunk_index: int, total_chunks: int, llm_model: str,
    content_type: str = "meeting",
) -> str:
    prompt = chunk_summary_prompt(chunk, chunk_index, total_chunks, content_type)
    return call_llm(prompt, chunk_summary_system(content_type), llm_model)


def consolidate_summaries(
    chunk_summaries: list[str], llm_model: str,
    content_type: str = "meeting",
) -> str:
    prompt = consolidation_prompt(chunk_summaries, content_type)
    return call_llm(prompt, consolidation_system(content_type), llm_model, num_ctx=16384)


def enhance_with_kb(
    summary: str, kb_context: str, llm_model: str,
    content_type: str = "meeting",
) -> str:
    """Enhance a summary with knowledge base context (second pass)."""
    prompt = kb_enhance_prompt(summary, kb_context, content_type)
    return call_llm(prompt, kb_enhance_system(content_type), llm_model, num_ctx=16384)


def generate_remarks(
    consolidated_summary: str, llm_model: str,
    content_type: str = "meeting",
) -> str:
    prompt = remarks_prompt(consolidated_summary, content_type)
    return call_llm(prompt, remarks_system(content_type), llm_model, num_ctx=16384)


def generate_watch_score(consolidated_summary: str, llm_model: str, user_interests: str | None = None) -> str:
    prompt = youtube_score_prompt(consolidated_summary, user_interests)
    return call_llm(prompt, YOUTUBE_SCORE_SYSTEM, llm_model)


# --- Podcast functions ---

def rank_articles(
    articles: list[dict], interests: str, max_articles: int, llm_model: str,
) -> list[int]:
    """Return indices of the most relevant articles, ordered by relevance."""
    prompt = article_ranking_prompt(articles, interests, max_articles)
    response = call_llm(prompt, ARTICLE_RANKING_SYSTEM, llm_model)

    # Parse JSON array from response
    try:
        # Find the JSON array in the response
        text = response.strip()
        start = text.index("[")
        end = text.index("]") + 1
        indices = json.loads(text[start:end])
        # Filter to valid indices
        return [i for i in indices if isinstance(i, int) and 0 <= i < len(articles)][:max_articles]
    except (ValueError, json.JSONDecodeError):
        logger.warning("Article ranking failed to parse LLM response, using default order")
        return list(range(min(max_articles, len(articles))))


def generate_podcast_script(
    articles: list[dict], interests: str, style: str, target_length: str, llm_model: str,
) -> str:
    """Generate a podcast script in solo or two_host style."""
    if style == "two_host":
        prompt = two_host_script_prompt(articles, interests, target_length)
        return call_llm(prompt, TWO_HOST_SCRIPT_SYSTEM, llm_model, num_ctx=16384)
    else:
        prompt = solo_script_prompt(articles, interests, target_length)
        return call_llm(prompt, SOLO_SCRIPT_SYSTEM, llm_model, num_ctx=16384)


def enhance_podcast_with_kb(
    script: str, kb_context: str, style: str, llm_model: str,
) -> str:
    """Enhance a podcast script with knowledge base context (second pass)."""
    prompt = kb_enhance_podcast_prompt(script, kb_context, style)
    return call_llm(prompt, KB_ENHANCE_PODCAST_SYSTEM, llm_model, num_ctx=16384)
