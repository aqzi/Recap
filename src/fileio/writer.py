import os

from utils.formatting import format_timestamp


def write_transcript(segments: list[dict], output_dir: str) -> str:
    path = os.path.join(output_dir, "transcript.md")
    lines = ["# Transcript\n"]
    for seg in segments:
        ts = format_timestamp(seg["start"])
        lines.append(f"**[{ts}]** {seg['text']}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def write_summary(summary_text: str, output_dir: str) -> str:
    path = os.path.join(output_dir, "summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    return path


def write_summary_named(summary_text: str, output_dir: str, name: str) -> str:
    """Write a summary file named summary_<name>.md for per-file mode."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"summary_{name}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    return path
