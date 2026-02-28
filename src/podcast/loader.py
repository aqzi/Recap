"""Load input text from a file or directory for podcast generation."""

import os

from core.knowledge_base import SUPPORTED_EXTENSIONS, extract_text, find_supported_files


def load_input_text(path: str) -> tuple[str, list[str]]:
    """Load text from a file or directory.

    Args:
        path: Path to a single file or a directory of supported files.

    Returns:
        (combined_text, list_of_source_file_paths)

    Raises:
        ValueError: If no text could be extracted.
    """
    if os.path.isfile(path):
        text = extract_text(path)
        if not text or not text.strip():
            raise ValueError(f"Could not extract text from {path}")
        return text.strip(), [path]

    if os.path.isdir(path):
        files = find_supported_files(path)
        if not files:
            raise ValueError(
                f"No supported files found in {path} "
                f"(supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))})"
            )
        parts = []
        used_files = []
        for filepath in files:
            text = extract_text(filepath)
            if text and text.strip():
                filename = os.path.basename(filepath)
                parts.append(f"--- {filename} ---\n{text.strip()}")
                used_files.append(filepath)
        if not parts:
            raise ValueError(f"No extractable text found in files under {path}")
        return "\n\n".join(parts), used_files

    raise ValueError(f"Path does not exist: {path}")
