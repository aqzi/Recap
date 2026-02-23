import os
import sys

import click

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".wma", ".aac", ".webm"}


def check_audio_file(audio_file: str) -> None:
    ext = os.path.splitext(audio_file)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        click.echo(f"Error: Unsupported audio format '{ext}'.")
        click.echo(f"Supported formats: {supported}")
        sys.exit(1)


def check_ollama(llm_model: str) -> None:
    try:
        import ollama
        models_response = ollama.list()
    except Exception:
        click.echo("Error: Ollama server is not running.")
        click.echo("Start it with: ollama serve")
        sys.exit(1)

    available = [m.model for m in models_response.models]
    model_base = llm_model.split(":")[0]
    if not any(llm_model in name or name.startswith(model_base) for name in available):
        click.echo(f"Error: Model '{llm_model}' not found locally.")
        click.echo(f"Pull it with: ollama pull {llm_model}")
        sys.exit(1)
