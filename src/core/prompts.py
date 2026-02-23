def chunk_summary_system(content_type: str) -> str:
    return f"""\
You are a {content_type} analyst. You produce concise, accurate summaries of {content_type} transcript segments.
Focus on key topics discussed, decisions made, action items mentioned, and important statements.
Do not invent information. If the transcript is unclear, say so.
IMPORTANT: Always write your response in English, even if the transcript is in Dutch or another language."""


def chunk_summary_prompt(chunk: dict, chunk_index: int, total_chunks: int, content_type: str) -> str:
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


def consolidation_prompt(chunk_summaries: list[str], content_type: str) -> str:
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


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"
