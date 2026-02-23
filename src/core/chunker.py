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
