import os

from fileio.progress import console
from rich.prompt import Confirm, Prompt

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
LANGUAGES = ["auto", "nl", "en"]


def interactive_mode():
    """Guide the user through mode selection and options step by step."""
    console.print()
    console.print("[bold]Recap[/bold]")
    console.print()
    console.print("  [bold cyan][1][/bold cyan] Summarize a local audio file")
    console.print("  [bold cyan][2][/bold cyan] Summarize a YouTube video")
    console.print("  [bold cyan][3][/bold cyan] Generate a podcast")
    console.print("  [bold cyan][4][/bold cyan] Record audio")
    console.print()

    mode = Prompt.ask("Select mode", choices=["1", "2", "3", "4"])

    # Record mode — handled separately (no shared options)
    if mode == "4":
        from cli.recorder import run_recorder
        rec_name = Prompt.ask("Recording name (empty for timestamp)", default="")
        out_input = Prompt.ask("Output directory (empty for default)", default="")
        console.print()
        run_recorder(out_input or None, rec_name or None)
        return

    # Mode-specific input
    audio_file = None
    youtube = None

    if mode == "1":
        while True:
            audio_file = Prompt.ask("Path to audio file")
            if os.path.isfile(audio_file):
                break
            console.print(f"  [red]File not found:[/red] {audio_file}")

    elif mode == "2":
        youtube = Prompt.ask("YouTube URL")

    # Defaults
    whisper_model = "medium"
    llm_model = "llama3.1:8b"
    language = "auto"
    output_dir = None
    kb_dir = None
    kb_rebuild = False
    embedding_model = None
    chunk_minutes = 10

    # Advanced options
    console.print()
    customize = Confirm.ask("Customize settings?", default=False)

    if customize:
        llm_model = Prompt.ask("Ollama model", default="llama3.1:8b")

        if mode in ("1", "2"):
            whisper_model = Prompt.ask(
                f"Whisper model ({', '.join(WHISPER_MODELS)})",
                choices=WHISPER_MODELS, default="medium",
            )

        if mode == "1":
            language = Prompt.ask(
                f"Language ({', '.join(LANGUAGES)})",
                choices=LANGUAGES, default="auto",
            )

        kb_input = Prompt.ask("Knowledge base directory (empty to skip)", default="")
        if kb_input:
            if os.path.isdir(kb_input):
                kb_dir = kb_input
                kb_rebuild = Confirm.ask("Rebuild KB index?", default=False)
            else:
                console.print(f"  [yellow]Warning:[/yellow] Directory not found: {kb_input}, skipping KB.")

        out_input = Prompt.ask("Output directory (empty for auto)", default="")
        if out_input:
            output_dir = out_input

    console.print()

    # Dispatch
    if mode == "3":
        from cli.podcast import run_podcast
        run_podcast(output_dir, llm_model, kb_dir=kb_dir, kb_rebuild=kb_rebuild,
                    embedding_model=embedding_model)
    else:
        from cli.summarizer import run_summarizer
        run_summarizer(audio_file, youtube, whisper_model, output_dir, llm_model,
                       language, chunk_minutes, kb_dir=kb_dir, kb_rebuild=kb_rebuild,
                       embedding_model=embedding_model)
