import os

from fileio.progress import console
from rich.prompt import Confirm, Prompt

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
LANGUAGES = ["auto", "nl", "en"]


def _choose_llm_model(llm_config: dict) -> tuple[str, dict]:
    """Interactive LLM model selection with provider choice and hardware recommendation.

    Returns (chosen_model, updated_llm_config).
    """
    saved_model = llm_config.get("llm", {}).get("model", "llama3.1:8b") or "llama3.1:8b"

    console.print()
    console.print("  LLM provider:")
    console.print("    [bold cyan][1][/bold cyan] Local (Ollama)")
    console.print("    [bold cyan][2][/bold cyan] Cloud (OpenAI, Anthropic, Gemini, ...)")
    console.print()
    provider_choice = Prompt.ask("Select provider", choices=["1", "2"], default="1")

    if provider_choice == "1":
        # Local (Ollama) path
        recommend = Confirm.ask("Get a model recommendation based on your hardware?", default=True)

        if recommend:
            console.print()
            console.print("  Model speed:")
            console.print("    [bold cyan][1][/bold cyan] Fast (smaller model, quicker responses)")
            console.print("    [bold cyan][2][/bold cyan] Medium (balanced)")
            console.print("    [bold cyan][3][/bold cyan] Slow (largest model, best quality)")
            console.print()
            speed_choice = Prompt.ask("Select speed", choices=["1", "2", "3"], default="2")
            speed_map = {"1": "fast", "2": "medium", "3": "slow"}
            speed = speed_map[speed_choice]

            from core.hardware import recommend_model
            model_name, explanation = recommend_model(speed)

            if model_name:
                console.print(f"  [bold green]Recommendation: {model_name}[/bold green]")
                console.print(f"  [dim]{explanation}[/dim]")
                default_model = model_name
            else:
                console.print(f"  [yellow]{explanation}[/yellow]")
                default_model = saved_model
        else:
            default_model = saved_model

        llm_model = Prompt.ask("LLM model", default=default_model)

    else:
        # Cloud provider path
        from core.llm import detect_provider

        default_cloud = saved_model if detect_provider(saved_model) != "ollama" else "gpt-4o"
        llm_model = Prompt.ask("Model name (e.g. gpt-4o, claude-sonnet-4-6)", default=default_cloud)

        provider = detect_provider(llm_model)
        if provider == "openai":
            existing_key = llm_config.get("llm", {}).get("openai_api_key", "")
            if not existing_key and not os.environ.get("OPENAI_API_KEY"):
                key = Prompt.ask("OpenAI API key")
                llm_config.setdefault("llm", {})["openai_api_key"] = key
        elif provider == "anthropic":
            existing_key = llm_config.get("llm", {}).get("anthropic_api_key", "")
            if not existing_key and not os.environ.get("ANTHROPIC_API_KEY"):
                key = Prompt.ask("Anthropic API key")
                llm_config.setdefault("llm", {})["anthropic_api_key"] = key

    # Persist the chosen model
    llm_config.setdefault("llm", {})["model"] = llm_model
    return llm_model, llm_config


def interactive_mode():
    """Guide the user through mode selection and options step by step."""
    from main import load_llm_config, save_llm_config, _apply_llm_config_to_env

    llm_config = load_llm_config()

    console.print()
    console.print("[bold]Recap[/bold]")
    console.print()
    console.print("  [bold cyan][1][/bold cyan] Summarize a local audio file")
    console.print("  [bold cyan][2][/bold cyan] Generate a podcast")
    console.print("  [bold cyan][3][/bold cyan] Record audio")
    console.print()

    mode = Prompt.ask("Select mode", choices=["1", "2", "3"])

    # Record mode — handled separately (no shared options)
    if mode == "3":
        from cli.recorder import run_recorder
        rec_name = Prompt.ask("Recording name (empty for timestamp)", default="")
        out_input = Prompt.ask("Output directory (empty for default)", default="")
        console.print()
        run_recorder(out_input or None, rec_name or None)
        return

    # Audio/transcript input
    audio_file = None
    transcript = None
    if mode == "1":
        console.print()
        console.print("  Input type:")
        console.print("    [bold cyan][1][/bold cyan] Audio file")
        console.print("    [bold cyan][2][/bold cyan] Existing transcript")
        console.print()
        input_type = Prompt.ask("Select input type", choices=["1", "2"], default="1")

        if input_type == "2":
            while True:
                transcript = Prompt.ask("Path to transcript file (.md)")
                if os.path.isfile(transcript):
                    break
                console.print(f"  [red]File not found:[/red] {transcript}")
        else:
            while True:
                audio_file = Prompt.ask("Path to audio file")
                if os.path.isfile(audio_file):
                    break
                console.print(f"  [red]File not found:[/red] {audio_file}")

    # Context (for summarizer mode)
    context = None
    if mode == "1":
        ctx_input = Prompt.ask("Context (e.g. 'team meeting', 'lecture', empty to skip)", default="")
        if ctx_input:
            context = ctx_input

    # Defaults
    whisper_model = "medium"
    language = "auto"
    output_dir = None
    kb_dir = None
    kb_rebuild = False
    embedding_model = None
    chunk_minutes = 10

    # Always ask about model selection
    llm_model, llm_config = _choose_llm_model(llm_config)

    # Additional settings
    console.print()
    customize = Confirm.ask("Customize other settings?", default=False)

    if customize:
        if mode == "1" and not transcript:
            whisper_model = Prompt.ask(
                f"Whisper model ({', '.join(WHISPER_MODELS)})",
                choices=WHISPER_MODELS, default="medium",
            )
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

    # Save config and apply API keys
    save_llm_config(llm_config)
    _apply_llm_config_to_env(llm_config)

    console.print()

    # Dispatch
    if mode == "2":
        from cli.podcast import run_podcast
        run_podcast(output_dir, llm_model, kb_dir=kb_dir, kb_rebuild=kb_rebuild,
                    embedding_model=embedding_model)
    else:
        from cli.summarizer import run_summarizer
        run_summarizer(audio_file, whisper_model, output_dir, llm_model,
                       language, chunk_minutes, kb_dir=kb_dir, kb_rebuild=kb_rebuild,
                       embedding_model=embedding_model, context=context,
                       transcript=transcript)
