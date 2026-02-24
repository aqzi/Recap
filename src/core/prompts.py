def chunk_summary_system(content_type: str) -> str:
    return f"""\
You are a {content_type} analyst. You produce concise, accurate summaries of {content_type} transcript segments.
Focus on key topics discussed, decisions made, action items mentioned, and important statements.
Do not invent information. If the transcript is unclear, say so.
IMPORTANT: Always write your response in English, even if the transcript is in Dutch or another language."""


def chunk_summary_prompt(
    chunk: dict, chunk_index: int, total_chunks: int,
    content_type: str,
) -> str:
    start = format_timestamp(chunk["start_time"])
    end = format_timestamp(chunk["end_time"])

    return f"""\
Summarize this {content_type} transcript segment (Part {chunk_index + 1} of {total_chunks}, \
covering {start} to {end}).

TRANSCRIPT:
{chunk["text"]}

Provide a structured summary with:
- **Topics Discussed**: Key subjects covered in this segment
- **Key Points**: Important statements, data, or arguments
- **Decisions**: Any decisions reached
- **Action Items**: Tasks or follow-ups mentioned (include who is responsible if mentioned)"""


def consolidation_system(content_type: str) -> str:
    return f"""\
You are a {content_type} analyst. You merge multiple segment summaries into a single coherent \
{content_type} summary. Eliminate redundancy, maintain chronological flow, and produce a well-structured document.
IMPORTANT: Always write your response in English, even if the original {content_type} was in Dutch or another language."""


def consolidation_prompt(
    chunk_summaries: list[str], content_type: str,
) -> str:
    label = content_type.capitalize()
    combined = "\n\n---\n\n".join(
        f"### Segment {i + 1}\n{s}" for i, s in enumerate(chunk_summaries)
    )

    return f"""\
Below are summaries of consecutive segments from a single {content_type}. \
Merge them into one cohesive {content_type} summary.

SEGMENT SUMMARIES:
{combined}

Produce a final summary in this format:

# {label} Summary

## Overview
(2-3 sentence high-level overview of the {content_type} purpose and outcome)

## Key Topics
(For each major topic discussed, provide a subsection with key points)

## Decisions Made
(Bulleted list of all decisions reached during the {content_type})

## Action Items
(Bulleted list: each item should include the task, responsible person if known, and deadline if mentioned)

## Open Questions
(Any unresolved questions or topics that need follow-up)"""


def kb_enhance_system(content_type: str) -> str:
    return f"""\
You are a {content_type} analyst. You enhance existing summaries by adding domain-specific \
nuance and connections from a private knowledge base.
CRITICAL RULES:
- The original summary content is the ground truth. Do NOT remove, replace, or contradict anything from it.
- Only ADD nuance, clarify terminology, or note connections where the knowledge base is clearly relevant.
- If the knowledge base context has NO meaningful connection to the summary content, return the summary UNCHANGED.
- Keep the exact same structure and format as the original summary.
IMPORTANT: Always write your response in English."""


def kb_enhance_prompt(summary: str, kb_context: str, content_type: str) -> str:
    label = content_type.capitalize()
    return f"""\
Below is a {content_type} summary followed by excerpts from a private knowledge base.

Your task: enhance the summary by weaving in relevant context from the knowledge base — \
but ONLY where there is a genuine connection. Add brief clarifications, terminology links, \
or contextual notes within the existing sections. Do not add new sections or topics that \
were not discussed in the original {content_type}.

If the knowledge base content is unrelated to the summary, return the original summary exactly as-is.

{label.upper()} SUMMARY:
{summary}

KNOWLEDGE BASE CONTEXT:
{kb_context}

Return the enhanced summary, keeping the same markdown structure and format."""


def remarks_system(content_type: str) -> str:
    return f"""\
You are a senior {content_type} facilitator and organizational consultant. \
You analyze {content_type} summaries and provide candid, actionable feedback about \
{content_type} effectiveness, team dynamics, and follow-up priorities.
IMPORTANT: Always write your response in English, even if the original {content_type} was in Dutch or another language."""


def remarks_prompt(consolidated_summary: str, content_type: str) -> str:
    label = content_type.capitalize()
    return f"""\
Based on this {content_type} summary, provide remarks and actionable suggestions.

{label.upper()} SUMMARY:
{consolidated_summary}

Provide your analysis in this format:

# {label} Remarks & Suggestions

## {label} Effectiveness
(Was this {content_type} productive? Were goals achieved? What could improve the {content_type} format?)

## Key Risks & Concerns
(Any risks, blockers, or concerns implied by the discussion that the team should be aware of)

## Priority Actions
(Rank the action items by urgency/impact. Highlight any that seem at risk of being dropped)

## Suggestions
(Concrete suggestions for follow-up, process improvements, or next steps that weren't \
explicitly discussed but would be valuable)

## Follow-up Recommendations
(Should there be a follow-up? What should the agenda include? Who should attend?)"""


YOUTUBE_SCORE_SYSTEM = """\
You are a content analyst. You evaluate video summaries and determine whether the viewer \
should watch the original video themselves or if the summary is sufficient.
IMPORTANT: Always write your response in English."""


def youtube_score_prompt(consolidated_summary: str, user_interests: str | None = None) -> str:
    interests_block = ""
    if user_interests:
        interests_block = f"""

The viewer has the following interests and goals:
{user_interests}

Factor these interests heavily into your score. A high score means the video is highly relevant \
to these interests and worth watching in full. A low score means the summary is sufficient \
given what the viewer cares about."""

    return f"""\
Based on this video summary, score how important it is for someone to watch the original video \
rather than just reading the summary.

VIDEO SUMMARY:
{consolidated_summary}{interests_block}

Score from 1 to 10:
- 1-3: The summary captures everything. No need to watch.
- 4-6: The summary covers the main points, but some nuance or context is lost.
- 7-10: Strongly recommended to watch. The video likely contains visual demos, \
complex explanations, emotional nuance, or interactive elements that text cannot capture.

Respond in EXACTLY this format (no other text):

SCORE: <number>/10
VERDICT: <one sentence explaining why>"""


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


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"
