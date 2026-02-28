from utils.formatting import format_timestamp

_LANGUAGE_NAMES = {
    "en": "English",
    "nl": "Dutch",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "pl": "Polish",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "fi": "Finnish",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "cs": "Czech",
    "ro": "Romanian",
    "hu": "Hungarian",
    "el": "Greek",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
}


def _language_name(code: str) -> str:
    """Map a language code to its full name, falling back to the code itself."""
    return _LANGUAGE_NAMES.get(code.lower(), code)


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


def chunk_summary_system(
    hint: str | None = None,
    kb_context: str | None = None,
    output_language: str = "en",
    is_audio: bool = True,
) -> str:
    lang = _language_name(output_language)
    role = "an audio analyst" if is_audio else "a text analyst"
    source = "audio transcript segments" if is_audio else "text document segments"
    return f"""\
You are {role}. You produce concise, accurate summaries of {source}.
Focus on key topics discussed, decisions made, action items mentioned, and important statements.
Do not invent information. If the content is unclear, say so.
IMPORTANT: Always write your response in {lang}, even if the input is in another language.{_hint_line(hint)}{_kb_context_block(kb_context)}"""


def chunk_summary_prompt(
    chunk: dict, chunk_index: int, total_chunks: int,
    is_audio: bool = True,
) -> str:
    if is_audio:
        start = format_timestamp(chunk["start_time"])
        end = format_timestamp(chunk["end_time"])
        header = (
            f"Summarize this audio transcript segment (Part {chunk_index + 1} of "
            f"{total_chunks}, covering {start} to {end})."
        )
    else:
        header = f"Summarize this text segment (Part {chunk_index + 1} of {total_chunks})."

    return f"""\
{header}

CONTENT:
{chunk["text"]}

Provide a structured summary with:
- **Topics Discussed**: Key subjects covered in this segment
- **Key Points**: Important statements, data, or arguments
- **Decisions**: Any decisions reached
- **Action Items**: Tasks or follow-ups mentioned (include who is responsible if mentioned)"""


def consolidation_system(
    hint: str | None = None,
    kb_context: str | None = None,
    output_language: str = "en",
    is_audio: bool = True,
) -> str:
    lang = _language_name(output_language)
    role = "an audio analyst" if is_audio else "a text analyst"
    source = "segment summaries" if is_audio else "section summaries"
    return f"""\
You are {role}. You merge multiple {source} into a single coherent \
summary. Produce a concise, well-structured document — keep all sections short and to the point.
IMPORTANT: Always write your response in {lang}, even if the original content was in another language.{_hint_line(hint)}{_kb_context_block(kb_context)}"""


def consolidation_prompt(chunk_summaries: list[str], is_audio: bool = True) -> str:
    combined = "\n\n---\n\n".join(
        f"### Segment {i + 1}\n{s}" for i, s in enumerate(chunk_summaries)
    )
    source_type = "a single audio recording" if is_audio else "a single document"

    return f"""\
Below are summaries of consecutive segments from {source_type}. \
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
You are a content curator. You select articles that are genuinely relevant to a piece of reference text.
IMPORTANT: Always respond in English. Return ONLY a JSON array of indices, nothing else.
If no articles are relevant, return an empty array: []"""


def article_ranking_prompt(articles: list[dict], reference_text: str, max_articles: int) -> str:
    article_list = "\n".join(
        f"{i}. [{a['title']}] — {a.get('summary', '')[:200]}"
        for i, a in enumerate(articles)
    )
    return f"""\
Given these articles and the reference text, select ONLY articles that are genuinely relevant \
to the reference text. Return their indices as a JSON array, most relevant first.

Rules:
- Only include articles that directly relate to topics in the reference text.
- Return at most {max_articles} articles.
- If no articles are relevant, return an empty array: []
- It is better to return fewer highly relevant articles than many loosely related ones.

REFERENCE TEXT:
{reference_text[:2000]}

ARTICLES:
{article_list}

Return ONLY a JSON array of integer indices (e.g. [2, 0, 5]) or [] if none are relevant."""


LENGTH_WORD_TARGETS = {
    "short": 400,
    "medium": 900,
    "long": 1800,
}


def solo_script_system(output_language: str = "en") -> str:
    lang = _language_name(output_language)
    return f"""\
You are a professional podcast host. You create engaging, informative podcast scripts \
for a solo narrator. Write in a natural spoken style — conversational but informative.
Your output is the FINAL spoken script. It will be fed directly into a text-to-speech engine. \
Output ONLY the words to be spoken — no titles, headers, labels, stage directions, or metadata.
IMPORTANT: Always write in {lang}.
CRITICAL: Do NOT use any markdown formatting (no #, *, **, -, ```, links, etc.). \
Do NOT use bullet points, numbered lists, or any special symbols. Write everything as \
natural spoken paragraphs. Spell out abbreviations and symbols where needed \
(e.g. write "number one" not "1.", write "about 50 percent" not "~50%")."""


def solo_script_prompt(
    input_text: str, target_length: str,
    articles: list[dict] | None = None,
    kb_context: str | None = None,
) -> str:
    word_target = LENGTH_WORD_TARGETS.get(target_length, 900)

    # Merge all material into one block so the LLM sees a single pool of content
    material_parts = [input_text[:6000]]

    if articles:
        for a in articles:
            content = a.get("content", a.get("summary", ""))[:2000]
            material_parts.append(f"[Related: {a['title']}]\n{content}")

    combined_material = "\n\n---\n\n".join(material_parts)

    kb_section = _kb_context_block(kb_context) if kb_context else ""

    return f"""\
Write a podcast script (~{word_target} words).

IMPORTANT STRUCTURAL RULES:
- The script must be ONE cohesive narrative from start to finish.
- NEVER split the script into separate sections or segments per source.
- NEVER use transition phrases like "now let's look at", "moving on to", \
"in related news", or "let's switch gears" to introduce external material.
- When you reference information from a related article, blend it in naturally as \
supporting evidence, a real-world example, added context, or a deeper explanation \
within the point you are already making.

SOURCE MATERIAL:
{combined_material}
{kb_section}

Output ONLY the spoken script. No title, no headers, no labels. Start directly with \
the host speaking. Plain spoken paragraphs only, no markdown, no special symbols, no lists."""


def two_host_script_system(output_language: str = "en") -> str:
    lang = _language_name(output_language)
    return f"""\
You are a podcast scriptwriter. You write engaging two-host discussion scripts. \
Host 1 (Alex) leads the conversation and introduces topics. Host 2 (Sam) adds analysis, \
asks questions, and offers counterpoints. Write natural dialogue — not formal, not scripted-sounding.
Your output is the FINAL spoken script. It will be fed directly into a text-to-speech engine. \
Output ONLY the dialogue — no titles, headers, labels, stage directions, or metadata. \
Prefix every line with either "ALEX:" or "SAM:" with no exceptions.
IMPORTANT: Always write in {lang}.
CRITICAL: Do NOT use any markdown formatting (no #, *, **, -, ```, links, etc.). \
Do NOT use bullet points, numbered lists, or any special symbols. Write everything as \
natural spoken dialogue. Spell out abbreviations and symbols where needed \
(e.g. write "number one" not "1.", write "about 50 percent" not "~50%")."""


def two_host_script_prompt(
    input_text: str, target_length: str,
    articles: list[dict] | None = None,
    kb_context: str | None = None,
) -> str:
    word_target = LENGTH_WORD_TARGETS.get(target_length, 900)

    # Merge all material into one block so the LLM sees a single pool of content
    material_parts = [input_text[:6000]]

    if articles:
        for a in articles:
            content = a.get("content", a.get("summary", ""))[:2000]
            material_parts.append(f"[Related: {a['title']}]\n{content}")

    combined_material = "\n\n---\n\n".join(material_parts)

    kb_section = _kb_context_block(kb_context) if kb_context else ""

    return f"""\
Write a two-host podcast script (~{word_target} words).

IMPORTANT STRUCTURAL RULES:
- The script must be ONE cohesive conversation from start to finish.
- NEVER split the conversation into separate sections or segments per source.
- NEVER use transition phrases like "now let's look at", "moving on to", \
"in related news", or "let's switch gears" to introduce external material.
- When you reference information from a related article, blend it in naturally as \
supporting evidence, a real-world example, added context, or a deeper explanation \
within the point being discussed.

SOURCE MATERIAL:
{combined_material}
{kb_section}

Output ONLY the spoken dialogue. No title, no headers, no labels. Start directly with \
ALEX speaking. Every line MUST start with "ALEX:" or "SAM:". \
Plain spoken dialogue only, no markdown, no special symbols, no lists."""
