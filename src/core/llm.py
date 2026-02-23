import time

import ollama

from core.prompts import (
    YOUTUBE_SCORE_SYSTEM,
    chunk_summary_prompt,
    chunk_summary_system,
    consolidation_prompt,
    consolidation_system,
    remarks_prompt,
    remarks_system,
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


def generate_remarks(
    consolidated_summary: str, llm_model: str,
    content_type: str = "meeting",
) -> str:
    prompt = remarks_prompt(consolidated_summary, content_type)
    return call_llm(prompt, remarks_system(content_type), llm_model, num_ctx=16384)


def generate_watch_score(consolidated_summary: str, llm_model: str, user_interests: str | None = None) -> str:
    prompt = youtube_score_prompt(consolidated_summary, user_interests)
    return call_llm(prompt, YOUTUBE_SCORE_SYSTEM, llm_model)
