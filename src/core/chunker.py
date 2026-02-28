CHUNK_TARGET_WORDS = 500


def chunk_text(text: str, chunk_words: int = CHUNK_TARGET_WORDS) -> list[dict]:
    """Split plain text into word-count-based chunks on paragraph boundaries.

    Returns a list of dicts with 'text' and 'chunk_index' keys.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if not paragraphs:
        return []

    chunks = []
    current_parts: list[str] = []
    current_words = 0
    chunk_idx = 0

    for para in paragraphs:
        para_words = len(para.split())
        if current_words + para_words > chunk_words and current_parts:
            chunks.append({
                "text": "\n\n".join(current_parts),
                "chunk_index": chunk_idx,
            })
            chunk_idx += 1
            current_parts = []
            current_words = 0

        current_parts.append(para)
        current_words += para_words

    if current_parts:
        chunks.append({
            "text": "\n\n".join(current_parts),
            "chunk_index": chunk_idx,
        })

    return chunks


def chunk_transcript(segments: list[dict], chunk_minutes: int = 10) -> list[dict]:
    """Split transcript segments into time-based chunks.

    Each chunk is a dict with start_time, end_time, text, and segments.
    """
    chunk_seconds = chunk_minutes * 60
    chunks = []
    current_chunk_segments = []
    chunk_start = 0.0

    for seg in segments:
        current_chunk_segments.append(seg)

        if seg["end"] - chunk_start >= chunk_seconds:
            chunk_text = " ".join(s["text"] for s in current_chunk_segments)
            chunks.append({
                "start_time": chunk_start,
                "end_time": seg["end"],
                "text": chunk_text,
                "segments": current_chunk_segments,
            })
            chunk_start = seg["end"]
            current_chunk_segments = []

    # Last partial chunk
    if current_chunk_segments:
        chunk_text = " ".join(s["text"] for s in current_chunk_segments)
        chunks.append({
            "start_time": chunk_start,
            "end_time": current_chunk_segments[-1]["end"],
            "text": chunk_text,
            "segments": current_chunk_segments,
        })

    return chunks
