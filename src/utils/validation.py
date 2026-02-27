import os
import sys

from fileio.progress import console

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".wma", ".aac", ".webm"}


def check_audio_file(audio_file: str) -> None:
    ext = os.path.splitext(audio_file)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        console.print(f"[bold red]Error:[/bold red] Unsupported audio format '{ext}'.")
        console.print(f"Supported formats: {supported}")
        sys.exit(1)


def check_ollama(llm_model: str) -> None:
    try:
        import ollama
        models_response = ollama.list()
    except Exception:
        console.print("[bold red]Error:[/bold red] Ollama server is not running.")
        console.print("Start it with: ollama serve")
        sys.exit(1)

    available = [m.model for m in models_response.models]
    has_tag = ":" in llm_model
    if has_tag:
        # User specified a tag (e.g. "llama3.2:3b") — require exact match
        found = any(llm_model == name for name in available)
    else:
        # No tag (e.g. "llama3.2") — match any variant with that base name
        available_bases = [name.split(":")[0] for name in available]
        found = any(llm_model == abase for abase in available_bases)
    if not found:
        console.print(f"[bold red]Error:[/bold red] Model '{llm_model}' not found locally.")
        console.print(f"Pull it with: ollama pull {llm_model}")
        sys.exit(1)


def check_llm_model(llm_model: str) -> None:
    """Validate the LLM model based on its provider."""
    from core.llm import detect_provider

    provider = detect_provider(llm_model)

    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            console.print("[bold red]Error:[/bold red] OPENAI_API_KEY not set.")
            console.print("Set it via environment variable or config.yaml.")
            sys.exit(1)
    elif provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            console.print("[bold red]Error:[/bold red] ANTHROPIC_API_KEY not set.")
            console.print("Set it via environment variable or config.yaml.")
            sys.exit(1)
    else:
        check_ollama(llm_model)
