from utils.formatting import format_timestamp


def _hint_line(hint: str | None) -> str:
    """Return a hint instruction line for system prompts, or empty string."""
    if hint:
        return f"\nAdditional context about this audio: {hint}"
    return ""


def _kb_context_block(kb_context: str | None) -> str:
    """Return a KB reference block for system prompts, or empty string."""
    if kb_context:
        return (
            "\n\nBACKGROUND REFERENCE (from knowledge base):\n"
            "Use this ONLY to understand domain-specific terms, acronyms, and context "
            "that appear in the input. Do not add topics, claims, or information from "
            "this reference that are not discussed in the input. If none of it relates "
            "to the input, ignore it entirely.\n"
            f"{kb_context}"
        )
    return ""


def chunk_summary_system(hint: str | None = None, kb_context: str | None = None) -> str:
    return f"""\
You are an audio analyst. You produce concise, accurate summaries of audio transcript segments.
Focus on key topics discussed, decisions made, action items mentioned, and important statements.
Do not invent information. If the transcript is unclear, say so.
IMPORTANT: Always write your response in English, even if the transcript is in another language.{_hint_line(hint)}{_kb_context_block(kb_context)}"""


def chunk_summary_prompt(
    chunk: dict, chunk_index: int, total_chunks: int,
) -> str:
    start = format_timestamp(chunk["start_time"])
    end = format_timestamp(chunk["end_time"])

    return f"""\
Summarize this audio transcript segment (Part {chunk_index + 1} of {total_chunks}, \
covering {start} to {end}).

TRANSCRIPT:
{chunk["text"]}

Provide a structured summary with:
- **Topics Discussed**: Key subjects covered in this segment
- **Key Points**: Important statements, data, or arguments
- **Decisions**: Any decisions reached
- **Action Items**: Tasks or follow-ups mentioned (include who is responsible if mentioned)"""


def consolidation_system(hint: str | None = None, kb_context: str | None = None) -> str:
    return f"""\
You are an audio analyst. You merge multiple segment summaries into a single coherent \
summary. Produce a concise, well-structured document — keep all sections short and to the point.
IMPORTANT: Always write your response in English, even if the original audio was in another language.{_hint_line(hint)}{_kb_context_block(kb_context)}"""


def consolidation_prompt(chunk_summaries: list[str]) -> str:
    combined = "\n\n---\n\n".join(
        f"### Segment {i + 1}\n{s}" for i, s in enumerate(chunk_summaries)
    )

    return f"""\
Below are summaries of consecutive segments from a single audio recording. \
Merge them into one cohesive summary.

SEGMENT SUMMARIES:
{combined}

Produce a final summary in this format:

# Summary

## TL;DR
(3-5 bullet points capturing the most important takeaways — be very brief and direct)

## Key Topics
(For each major topic, a brief subsection with key points — a few sentences each, not exhaustive)

## Pros
(Brief bullet points — positive aspects, good decisions, strengths identified)

## Cons
(Brief bullet points — risks, weaknesses, problems, or concerns raised or implied)

## Remarks
(Brief additional observations, nuances, or noteworthy points)

## Action Items
(Bulleted list: task, responsible person if known, deadline if mentioned — omit this section entirely if there are no action items)"""


# --- Podcast prompts ---

ARTICLE_RANKING_SYSTEM = """\
You are a content curator. You rank articles by relevance to a piece of reference text.
IMPORTANT: Always respond in English. Return ONLY a JSON array of indices, nothing else."""


def article_ranking_prompt(articles: list[dict], reference_text: str, max_articles: int) -> str:
    article_list = "\n".join(
        f"{i}. [{a['title']}] — {a.get('summary', '')[:200]}"
        for i, a in enumerate(articles)
    )
    return f"""\
Given these articles and the reference text, return the indices of the top {max_articles} \
most relevant articles as a JSON array (e.g. [2, 0, 5, 1, 3]). Most relevant first.

REFERENCE TEXT:
{reference_text[:2000]}

ARTICLES:
{article_list}

Return ONLY a JSON array of integer indices, nothing else."""


LENGTH_WORD_TARGETS = {
    "short": 400,
    "medium": 900,
    "long": 1800,
}

SOLO_SCRIPT_SYSTEM = """\
You are a professional podcast host. You create engaging, informative podcast scripts \
for a solo narrator. Your primary focus is the source material provided. Write in a natural \
spoken style — conversational but informative. Use transitions between topics. Start with a \
brief intro and end with a short outro.
IMPORTANT: Always write in English. Output ONLY the script text, no stage directions or metadata.
CRITICAL: This script will be read aloud by a text-to-speech engine. Do NOT use any markdown \
formatting (no #, *, **, -, ```, links, etc.). Do NOT use bullet points, numbered lists, or \
any special symbols. Write everything as natural spoken paragraphs. Spell out abbreviations \
and symbols where needed (e.g. write "number one" not "1.", write "about 50 percent" not "~50%")."""


def solo_script_prompt(
    input_text: str, target_length: str,
    articles: list[dict] | None = None,
    interests: str | None = None,
    kb_context: str | None = None,
) -> str:
    word_target = LENGTH_WORD_TARGETS.get(target_length, 900)

    sections = [
        f"Write a podcast script (~{word_target} words). The primary source material below "
        f"is the main focus of the podcast. Cover it thoroughly.",
        f"\nPRIMARY SOURCE MATERIAL:\n{input_text[:6000]}",
    ]

    if articles:
        article_blocks = "\n\n---\n\n".join(
            f"### {a['title']}\nSource: {a['url']}\n\n{a.get('content', a.get('summary', ''))[:2000]}"
            for a in articles
        )
        sections.append(
            f"\nSUPPLEMENTARY ARTICLES (use to add depth, but keep focus on the primary material):\n{article_blocks}"
        )

    if interests:
        sections.append(f"\nLISTENER INTERESTS (guide tone and emphasis):\n{interests}")

    if kb_context:
        sections.append(_kb_context_block(kb_context))

    sections.append(
        "\nWrite a natural, engaging script for a single host. The primary source material "
        "should be the core of the podcast. Supplementary articles, if any, should only add "
        "context or depth.\nRemember: plain spoken text only, no markdown, no special symbols, no lists."
    )

    return "\n".join(sections)


TWO_HOST_SCRIPT_SYSTEM = """\
You are a podcast scriptwriter. You write engaging two-host discussion scripts. \
Host 1 (Alex) leads the conversation and introduces topics. Host 2 (Sam) adds analysis, \
asks questions, and offers counterpoints. The primary focus is always the source material provided. \
Write natural dialogue — not formal, not scripted-sounding.
IMPORTANT: Always write in English. Prefix every line with either "ALEX:" or "SAM:" with no exceptions.
CRITICAL: This script will be read aloud by a text-to-speech engine. Do NOT use any markdown \
formatting (no #, *, **, -, ```, links, etc.). Do NOT use bullet points, numbered lists, or \
any special symbols. Write everything as natural spoken dialogue. Spell out abbreviations \
and symbols where needed (e.g. write "number one" not "1.", write "about 50 percent" not "~50%")."""


def two_host_script_prompt(
    input_text: str, target_length: str,
    articles: list[dict] | None = None,
    interests: str | None = None,
    kb_context: str | None = None,
) -> str:
    word_target = LENGTH_WORD_TARGETS.get(target_length, 900)

    sections = [
        f"Write a two-host podcast script (~{word_target} words). The primary source material "
        f"below is the main focus of the podcast. Cover it thoroughly.",
        f"\nPRIMARY SOURCE MATERIAL:\n{input_text[:6000]}",
    ]

    if articles:
        article_blocks = "\n\n---\n\n".join(
            f"### {a['title']}\nSource: {a['url']}\n\n{a.get('content', a.get('summary', ''))[:2000]}"
            for a in articles
        )
        sections.append(
            f"\nSUPPLEMENTARY ARTICLES (use to add depth, but keep focus on the primary material):\n{article_blocks}"
        )

    if interests:
        sections.append(f"\nLISTENER INTERESTS (guide tone and emphasis):\n{interests}")

    if kb_context:
        sections.append(_kb_context_block(kb_context))

    sections.append(
        "\nWrite a natural conversation between ALEX and SAM. The primary source material "
        "should be the core of the discussion. Supplementary articles, if any, should only "
        "add context or depth. Every line MUST start with \"ALEX:\" or \"SAM:\".\n"
        "Remember: plain spoken text only, no markdown, no special symbols, no lists."
    )

    return "\n".join(sections)


