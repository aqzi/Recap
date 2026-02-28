import json
import logging
import os
import time

import litellm

logger = logging.getLogger(__name__)

from core.prompts import (
    ARTICLE_RANKING_SYSTEM,
    SOLO_SCRIPT_SYSTEM,
    TWO_HOST_SCRIPT_SYSTEM,
    article_ranking_prompt,
    chunk_summary_prompt,
    chunk_summary_system,
    consolidation_prompt,
    consolidation_system,
    solo_script_prompt,
    two_host_script_prompt,
)

# Suppress litellm's verbose logging
litellm.suppress_debug_info = True

DEFAULT_NUM_CTX = 8192


def detect_provider(model_name: str) -> str:
    """Detect the LLM provider from the model name."""
    lower = model_name.lower()
    openai_prefixes = ("gpt-", "o1-", "o3-", "o4-", "chatgpt-")
    if any(lower.startswith(p) for p in openai_prefixes):
        return "openai"
    if lower.startswith("claude-"):
        return "anthropic"
    return "ollama"


def _set_api_keys_from_config(llm_config: dict | None) -> None:
    """Set provider API keys in the environment from llm_config if not already set."""
    if not llm_config:
        return
    llm = llm_config.get("llm", {})
    key_map = {
        "openai_api_key": "OPENAI_API_KEY",
        "anthropic_api_key": "ANTHROPIC_API_KEY",
    }
    for config_key, env_key in key_map.items():
        value = llm.get(config_key)
        if value and not os.environ.get(env_key):
            os.environ[env_key] = value


def call_llm(
    prompt: str,
    system_prompt: str,
    llm_model: str,
    num_ctx: int = DEFAULT_NUM_CTX,
    retries: int = 1,
    timeout: int = 120,
) -> str:
    provider = detect_provider(llm_model)

    # Build the model string for litellm
    if provider == "ollama":
        model_str = f"ollama/{llm_model}"
    else:
        model_str = llm_model

    for attempt in range(retries + 1):
        try:
            kwargs = {
                "model": model_str,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
                "timeout": timeout,
            }
            if provider == "ollama":
                kwargs["num_ctx"] = num_ctx

            response = litellm.completion(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            if attempt < retries:
                logger.warning("LLM call attempt %d/%d failed: %s", attempt + 1, retries + 1, e)
                time.sleep(2)
                continue
            raise


def summarize_chunk(
    chunk: dict, chunk_index: int, total_chunks: int, llm_model: str,
    hint: str | None = None, kb_context: str | None = None,
) -> str:
    prompt = chunk_summary_prompt(chunk, chunk_index, total_chunks)
    return call_llm(prompt, chunk_summary_system(hint, kb_context=kb_context), llm_model)


def consolidate_summaries(
    chunk_summaries: list[str], llm_model: str,
    hint: str | None = None, kb_context: str | None = None,
) -> str:
    prompt = consolidation_prompt(chunk_summaries)
    return call_llm(prompt, consolidation_system(hint, kb_context=kb_context), llm_model, num_ctx=16384, timeout=300)


# --- Podcast functions ---

def rank_articles(
    articles: list[dict], reference_text: str, max_articles: int, llm_model: str,
) -> list[int]:
    """Return indices of the most relevant articles, ordered by relevance."""
    prompt = article_ranking_prompt(articles, reference_text, max_articles)
    response = call_llm(prompt, ARTICLE_RANKING_SYSTEM, llm_model)

    # Parse JSON array from response
    try:
        # Find the JSON array in the response
        text = response.strip()
        start = text.index("[")
        end = text.rindex("]") + 1
        indices = json.loads(text[start:end])
        # Filter to valid indices
        return [i for i in indices if isinstance(i, int) and 0 <= i < len(articles)][:max_articles]
    except (ValueError, json.JSONDecodeError):
        logger.warning("Article ranking failed to parse LLM response, using default order")
        return list(range(min(max_articles, len(articles))))


def generate_podcast_script(
    input_text: str, style: str, target_length: str, llm_model: str,
    articles: list[dict] | None = None,
    interests: str | None = None,
    kb_context: str | None = None,
) -> str:
    """Generate a podcast script in solo or two_host style."""
    if style == "two_host":
        prompt = two_host_script_prompt(
            input_text, target_length,
            articles=articles, interests=interests, kb_context=kb_context,
        )
        return call_llm(prompt, TWO_HOST_SCRIPT_SYSTEM, llm_model, num_ctx=16384, timeout=300)
    else:
        prompt = solo_script_prompt(
            input_text, target_length,
            articles=articles, interests=interests, kb_context=kb_context,
        )
        return call_llm(prompt, SOLO_SCRIPT_SYSTEM, llm_model, num_ctx=16384, timeout=300)
