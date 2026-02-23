CHUNK_SUMMARY_SYSTEM = """\
You are a meeting analyst. You produce concise, accurate summaries of meeting transcript segments.
Focus on key topics discussed, decisions made, action items mentioned, and important statements.
Do not invent information. If the transcript is unclear, say so.
IMPORTANT: Always write your response in English, even if the transcript is in Dutch or another language."""


def chunk_summary_prompt(chunk: dict, chunk_index: int, total_chunks: int) -> str:
    start = format_timestamp(chunk["start_time"])
    end = format_timestamp(chunk["end_time"])
    return f"""\
Summarize this meeting transcript segment (Part {chunk_index + 1} of {total_chunks}, \
covering {start} to {end}).

TRANSCRIPT:
{chunk["text"]}

Provide a structured summary with:
- **Topics Discussed**: Key subjects covered in this segment
- **Key Points**: Important statements, data, or arguments
- **Decisions**: Any decisions reached
- **Action Items**: Tasks or follow-ups mentioned (include who is responsible if mentioned)"""


CONSOLIDATION_SYSTEM = """\
You are a meeting analyst. You merge multiple segment summaries into a single coherent \
meeting summary. Eliminate redundancy, maintain chronological flow, and produce a well-structured document.
IMPORTANT: Always write your response in English, even if the original meeting was in Dutch or another language."""


def consolidation_prompt(chunk_summaries: list[str]) -> str:
    combined = "\n\n---\n\n".join(
        f"### Segment {i + 1}\n{s}" for i, s in enumerate(chunk_summaries)
    )
    return f"""\
Below are summaries of consecutive segments from a single meeting. \
Merge them into one cohesive meeting summary.

SEGMENT SUMMARIES:
{combined}

Produce a final summary in this format:

# Meeting Summary

## Overview
(2-3 sentence high-level overview of the meeting purpose and outcome)

## Key Topics
(For each major topic discussed, provide a subsection with key points)

## Decisions Made
(Bulleted list of all decisions reached during the meeting)

## Action Items
(Bulleted list: each item should include the task, responsible person if known, and deadline if mentioned)

## Open Questions
(Any unresolved questions or topics that need follow-up)"""


REMARKS_SYSTEM = """\
You are a senior meeting facilitator and organizational consultant. \
You analyze meeting summaries and provide candid, actionable feedback about \
meeting effectiveness, team dynamics, and follow-up priorities.
IMPORTANT: Always write your response in English, even if the original meeting was in Dutch or another language."""


def remarks_prompt(consolidated_summary: str) -> str:
    return f"""\
Based on this meeting summary, provide remarks and actionable suggestions.

MEETING SUMMARY:
{consolidated_summary}

Provide your analysis in this format:

# Meeting Remarks & Suggestions

## Meeting Effectiveness
(Was this meeting productive? Were goals achieved? What could improve the meeting format?)

## Key Risks & Concerns
(Any risks, blockers, or concerns implied by the discussion that the team should be aware of)

## Priority Actions
(Rank the action items by urgency/impact. Highlight any that seem at risk of being dropped)

## Suggestions
(Concrete suggestions for follow-up, process improvements, or next steps that weren't \
explicitly discussed but would be valuable)

## Follow-up Meeting Recommendations
(Should there be a follow-up? What should the agenda include? Who should attend?)"""


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"
