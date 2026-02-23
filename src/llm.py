import time

import ollama

from prompts import (
    CHUNK_SUMMARY_SYSTEM,
    CONSOLIDATION_SYSTEM,
    REMARKS_SYSTEM,
    chunk_summary_prompt,
    consolidation_prompt,
    remarks_prompt,
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
    chunk: dict, chunk_index: int, total_chunks: int, llm_model: str
) -> str:
    prompt = chunk_summary_prompt(chunk, chunk_index, total_chunks)
    return call_llm(prompt, CHUNK_SUMMARY_SYSTEM, llm_model)


def consolidate_summaries(chunk_summaries: list[str], llm_model: str) -> str:
    prompt = consolidation_prompt(chunk_summaries)
    return call_llm(prompt, CONSOLIDATION_SYSTEM, llm_model, num_ctx=16384)


def generate_remarks(consolidated_summary: str, llm_model: str) -> str:
    prompt = remarks_prompt(consolidated_summary)
    return call_llm(prompt, REMARKS_SYSTEM, llm_model, num_ctx=16384)
