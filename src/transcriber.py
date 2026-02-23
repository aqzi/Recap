from faster_whisper import WhisperModel

COMPUTE_TYPE = "int8"
DEVICE = "cpu"


def transcribe(audio_file: str, model_size: str, language: str | None, progress, task_id):
    """Transcribe an audio file using faster-whisper.

    Returns a tuple of (segments_list, duration_seconds).
    """
    model = WhisperModel(model_size, device=DEVICE, compute_type=COMPUTE_TYPE)

    transcribe_kwargs = {
        "beam_size": 5,
        "vad_filter": True,
        "vad_parameters": {"min_silence_duration_ms": 500},
    }
    if language:
        transcribe_kwargs["language"] = language

    segments_gen, info = model.transcribe(audio_file, **transcribe_kwargs)

    duration = info.duration
    progress.update(task_id, total=int(duration))

    segments = []
    for seg in segments_gen:
        segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
        })
        progress.update(task_id, completed=int(seg.end))

    progress.update(task_id, completed=int(duration))
    return segments, duration
