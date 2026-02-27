import os
import re
import sys

# --------------- backend detection ---------------

_FORCE_BACKEND = os.environ.get("WHISPER_BACKEND", "").lower().strip()

_USE_MLX = False
if _FORCE_BACKEND == "mlx":
    _USE_MLX = True
elif _FORCE_BACKEND in ("faster-whisper", "faster_whisper", "cpu"):
    _USE_MLX = False
elif sys.platform == "darwin":
    try:
        import mlx_whisper as _  # noqa: F401
        _USE_MLX = True
    except ImportError:
        _USE_MLX = False

# --------------- model-size mapping ---------------

_MLX_MODEL_MAP = {
    "tiny": "mlx-community/whisper-tiny",
    "base": "mlx-community/whisper-base",
    "small": "mlx-community/whisper-small",
    "medium": "mlx-community/whisper-medium",
    "large-v2": "mlx-community/whisper-large-v2",
    "large-v3": "mlx-community/whisper-large-v3-turbo",
}

# --------------- faster-whisper backend ---------------


def _transcribe_faster_whisper(audio_file, model_size, language, progress, task_id):
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    kwargs = {
        "beam_size": 5,
        "vad_filter": True,
        "vad_parameters": {"min_silence_duration_ms": 500},
    }
    if language:
        kwargs["language"] = language

    segments_gen, info = model.transcribe(audio_file, **kwargs)

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

# --------------- mlx-whisper backend ---------------


def _transcribe_mlx(audio_file, model_size, language, progress, task_id):
    import mlx_whisper

    repo = _MLX_MODEL_MAP.get(model_size)
    if repo is None:
        raise ValueError(
            f"Unknown model size '{model_size}' for mlx-whisper. "
            f"Supported: {', '.join(_MLX_MODEL_MAP)}"
        )

    # mlx-whisper returns all at once — no streaming progress
    progress.update(task_id, total=None)

    kwargs = {"path_or_hf_repo": repo}
    if language:
        kwargs["language"] = language

    result = mlx_whisper.transcribe(audio_file, **kwargs)

    raw_segments = result.get("segments", [])
    segments = [
        {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
        for s in raw_segments
    ]

    duration = segments[-1]["end"] if segments else 0.0
    progress.update(task_id, total=int(duration), completed=int(duration))
    return segments, duration

# --------------- public API ---------------


def parse_transcript(path: str) -> list[dict]:
    """Parse a transcript file and return segments in the same format as transcribe().

    Supports the markdown format written by write_transcript():
        **[H:MM:SS]** text   or   **[M:SS]** text

    Plain text files (no timestamps) are treated as a single segment starting at 0.
    """
    with open(path, encoding="utf-8") as f:
        content = f.read()

    pattern = re.compile(r"\*\*\[(\d+(?::\d{2}){1,2})\]\*\*\s+(.+)")
    matches = pattern.findall(content)

    if not matches:
        # Plain text fallback — single segment
        text = content.strip()
        if text:
            return [{"start": 0.0, "end": 1.0, "text": text}]
        return []

    segments = []
    for ts_str, text in matches:
        parts = ts_str.split(":")
        if len(parts) == 3:
            seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        else:
            seconds = int(parts[0]) * 60 + int(parts[1])
        segments.append({"start": float(seconds), "end": 0.0, "text": text.strip()})

    # Fill in end times: each segment ends when the next one starts
    for i in range(len(segments) - 1):
        segments[i]["end"] = segments[i + 1]["start"]
    if segments:
        segments[-1]["end"] = segments[-1]["start"] + 1.0

    return segments


def transcribe(audio_file: str, model_size: str, language: str | None, progress, task_id):
    """Transcribe an audio file.

    Uses mlx-whisper (GPU) on macOS Apple Silicon if available, else faster-whisper (CPU).
    Override with env var WHISPER_BACKEND=faster-whisper or WHISPER_BACKEND=mlx.

    Returns a tuple of (segments_list, duration_seconds).
    """
    if _USE_MLX:
        return _transcribe_mlx(audio_file, model_size, language, progress, task_id)
    return _transcribe_faster_whisper(audio_file, model_size, language, progress, task_id)
