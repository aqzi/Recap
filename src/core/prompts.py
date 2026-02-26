from utils.formatting import format_timestamp


def _context_line(context: str | None) -> str:
    """Return a context instruction line for system prompts, or empty string."""
    if context:
        return f"\nAdditional context about this audio: {context}"
    return ""


def chunk_summary_system(context: str | None = None) -> str:
    return f"""\
You are an audio analyst. You produce concise, accurate summaries of audio transcript segments.
Focus on key topics discussed, decisions made, action items mentioned, and important statements.
Do not invent information. If the transcript is unclear, say so.
IMPORTANT: Always write your response in English, even if the transcript is in another language.{_context_line(context)}"""


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


def consolidation_system(context: str | None = None) -> str:
    return f"""\
You are an audio analyst. You merge multiple segment summaries into a single coherent \
summary. Produce a concise, well-structured document — keep all sections short and to the point.
IMPORTANT: Always write your response in English, even if the original audio was in another language.{_context_line(context)}"""


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


def kb_enhance_system() -> str:
    return """\
You are an audio analyst. You enhance existing summaries by adding domain-specific \
nuance and connections from a private knowledge base.
CRITICAL RULES:
- The original summary content is the ground truth. Do NOT remove, replace, or contradict anything from it.
- Only ADD nuance, clarify terminology, or note connections where the knowledge base is clearly relevant.
- If the knowledge base context has NO meaningful connection to the summary content, return the summary UNCHANGED.
- Keep the exact same structure and format as the original summary.
- Keep it concise — do not make the summary significantly longer.
IMPORTANT: Always write your response in English."""


def kb_enhance_prompt(summary: str, kb_context: str) -> str:
    return f"""\
Below is a summary followed by excerpts from a private knowledge base.

Your task: enhance the summary by weaving in relevant context from the knowledge base — \
but ONLY where there is a genuine connection. Add brief clarifications, terminology links, \
or contextual notes within the existing sections. Do not add new sections or topics that \
were not discussed in the original audio.

If the knowledge base content is unrelated to the summary, return the original summary exactly as-is.

SUMMARY:
{summary}

KNOWLEDGE BASE CONTEXT:
{kb_context}

Return the enhanced summary, keeping the same markdown structure and format."""


# --- Podcast prompts ---

ARTICLE_RANKING_SYSTEM = """\
You are a content curator. You rank articles by relevance to a reader's interests.
IMPORTANT: Always respond in English. Return ONLY a JSON array of indices, nothing else."""


def article_ranking_prompt(articles: list[dict], interests: str, max_articles: int) -> str:
    article_list = "\n".join(
        f"{i}. [{a['title']}] — {a.get('summary', '')[:200]}"
        for i, a in enumerate(articles)
    )
    return f"""\
Given these articles and the reader's interests, return the indices of the top {max_articles} \
most relevant articles as a JSON array (e.g. [2, 0, 5, 1, 3]). Most relevant first.

READER INTERESTS:
{interests}

ARTICLES:
{article_list}

Return ONLY a JSON array of integer indices, nothing else."""


LENGTH_WORD_TARGETS = {
    "short": 400,
    "medium": 900,
    "long": 1800,
}

SOLO_SCRIPT_SYSTEM = """\
You are a professional podcast host. You create engaging, informative tech podcast scripts \
for a solo narrator. Write in a natural spoken style — conversational but informative. \
Use transitions between topics. Start with a brief intro and end with a short outro.
IMPORTANT: Always write in English. Output ONLY the script text, no stage directions or metadata.
CRITICAL: This script will be read aloud by a text-to-speech engine. Do NOT use any markdown \
formatting (no #, *, **, -, ```, links, etc.). Do NOT use bullet points, numbered lists, or \
any special symbols. Write everything as natural spoken paragraphs. Spell out abbreviations \
and symbols where needed (e.g. write "number one" not "1.", write "about 50 percent" not "~50%")."""


def solo_script_prompt(
    articles: list[dict], interests: str, target_length: str,
) -> str:
    word_target = LENGTH_WORD_TARGETS.get(target_length, 900)
    article_blocks = "\n\n---\n\n".join(
        f"### {a['title']}\nSource: {a['url']}\n\n{a.get('content', a.get('summary', ''))[:2000]}"
        for a in articles
    )

    return f"""\
Write a podcast script (~{word_target} words) covering the following articles. \
Focus on what matters most to the listener based on their interests.

LISTENER INTERESTS:
{interests}

ARTICLES:
{article_blocks}

Write a natural, engaging script for a single host. Cover the most interesting stories, \
explain why they matter, and connect them to the listener's interests where relevant.
Remember: plain spoken text only, no markdown, no special symbols, no lists."""


TWO_HOST_SCRIPT_SYSTEM = """\
You are a podcast scriptwriter. You write engaging two-host discussion scripts. \
Host 1 (Alex) leads the conversation and introduces topics. Host 2 (Sam) adds analysis, \
asks questions, and offers counterpoints. Write natural dialogue — not formal, not scripted-sounding.
IMPORTANT: Always write in English. Prefix every line with either "ALEX:" or "SAM:" with no exceptions.
CRITICAL: This script will be read aloud by a text-to-speech engine. Do NOT use any markdown \
formatting (no #, *, **, -, ```, links, etc.). Do NOT use bullet points, numbered lists, or \
any special symbols. Write everything as natural spoken dialogue. Spell out abbreviations \
and symbols where needed (e.g. write "number one" not "1.", write "about 50 percent" not "~50%")."""


def two_host_script_prompt(
    articles: list[dict], interests: str, target_length: str,
) -> str:
    word_target = LENGTH_WORD_TARGETS.get(target_length, 900)
    article_blocks = "\n\n---\n\n".join(
        f"### {a['title']}\nSource: {a['url']}\n\n{a.get('content', a.get('summary', ''))[:2000]}"
        for a in articles
    )

    return f"""\
Write a two-host podcast script (~{word_target} words) covering the following articles. \
Focus on what matters most to the listeners based on their interests.

LISTENER INTERESTS:
{interests}

ARTICLES:
{article_blocks}

Write a natural conversation between ALEX and SAM. Alex introduces topics, Sam adds depth. \
They discuss the most interesting stories, debate implications, and connect them to the \
listener's interests. Every line MUST start with "ALEX:" or "SAM:".
Remember: plain spoken text only, no markdown, no special symbols, no lists."""


KB_ENHANCE_PODCAST_SYSTEM = """\
You are a podcast script editor. You enhance existing podcast scripts by weaving in \
relevant background context from the listener's private knowledge base.
CRITICAL RULES:
- The original script is the primary content. Do NOT remove or replace any of it.
- Only ADD brief remarks, connections, or context where the knowledge base is clearly relevant \
to topics already discussed in the script.
- If the knowledge base has NO meaningful connection to the script, return the script UNCHANGED.
- Preserve the exact format: plain spoken text, no markdown, no special symbols.
IMPORTANT: Always write in English."""


def kb_enhance_podcast_prompt(script: str, kb_context: str, style: str) -> str:
    if style == "two_host":
        format_note = ("Keep every line prefixed with \"ALEX:\" or \"SAM:\". "
                       "Add KB-informed remarks as natural dialogue exchanges.")
    else:
        format_note = "Keep the single-host narrative style."

    return f"""\
Below is a podcast script followed by excerpts from the listener's private knowledge base.

Your task: enhance the script by weaving in relevant knowledge base context — but ONLY where \
there is a genuine connection to topics already discussed. Add brief contextual remarks, \
connections to the listener's work, or deeper background where it fits naturally.

If the knowledge base content is unrelated to the script topics, return the script exactly as-is.

PODCAST SCRIPT:
{script}

KNOWLEDGE BASE CONTEXT:
{kb_context}

{format_note}
Remember: plain spoken text only, no markdown formatting, no special symbols, no lists.
Return the enhanced script."""
