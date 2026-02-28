import os
import sys

from core.knowledge_base import init_kb
from fileio.progress import console, create_progress
from utils.validation import check_llm_model


def run_podcast(input_path, output_dir, llm_model, kb_dir=None, kb_rebuild=False, embedding_model=None):
    """Run the podcast generation pipeline from input text."""
    from podcast.loader import load_input_text
    from podcast.scriptwriter import generate_podcast, write_podcast_output
    from podcast.tts import get_tts_engine

    from main import load_interests, load_podcast_config

    # Load primary input text
    try:
        input_text, source_files = load_input_text(input_path)
    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

    # Interests are optional — used to guide tone if available
    interests = load_interests()

    config = load_podcast_config()
    style = config.get("podcast", {}).get("style", "solo")

    enrichment_cfg = config.get("enrichment", {})
    enrichment_feeds = enrichment_cfg.get("feeds", [])
    enrichment_web = enrichment_cfg.get("web_search", False)
    enrichment_active = bool(enrichment_feeds) or enrichment_web

    if output_dir is None:
        from main import DEFAULT_OUTPUT_DIR
        # Derive output dir from input name
        input_name = os.path.splitext(os.path.basename(input_path))[0]
        if os.path.isdir(input_path):
            input_name = os.path.basename(os.path.normpath(input_path))
        output_dir = os.path.join(DEFAULT_OUTPUT_DIR, f"podcast_{input_name}")

    console.print("[bold]Podcast Generator[/bold]")
    console.print(f"  Input:    {input_path} ({len(source_files)} file(s))")
    console.print(f"  Style:    {style}")
    from core.llm import detect_provider
    provider = detect_provider(llm_model)
    console.print(f"  LLM:      {llm_model} ({provider})")
    console.print(f"  TTS:      {config.get('tts', {}).get('engine', 'piper')}")
    if enrichment_active:
        parts = []
        if enrichment_feeds:
            parts.append(f"{len(enrichment_feeds)} feed(s)")
        if enrichment_web:
            parts.append("web search")
        console.print(f"  Enrich:   {', '.join(parts)}")
    else:
        console.print("  Enrich:   none")
    if interests:
        console.print("  Interests: loaded")
    if kb_dir:
        console.print(f"  KB:       {kb_dir}")
    console.print(f"  Output:   {output_dir}")
    console.print()

    check_llm_model(llm_model)
    os.makedirs(output_dir, exist_ok=True)

    kb = None

    try:
        with create_progress() as progress:
            if kb_dir:
                kb = init_kb(kb_dir, kb_rebuild, embedding_model, progress, console)

            try:
                script, sources_md = generate_podcast(
                    input_text, config, llm_model, progress,
                    interests=interests,
                    source_files=source_files,
                    kb=kb,
                )
            except Exception as e:
                console.print(f"\n[bold red]Error:[/bold red] Podcast generation failed: {e}")
                sys.exit(1)

            task_tts = progress.add_task("Generating audio...", total=1)
            audio_path = os.path.join(output_dir, "podcast.wav")
            try:
                tts = get_tts_engine(config)
                if style == "two_host":
                    voice2 = config.get("tts", {}).get("voice_host2")
                    tts.synthesize_two_host(script, audio_path, voice2)
                else:
                    tts.synthesize(script, audio_path)
            except Exception as e:
                console.print(f"\n[bold red]Error:[/bold red] TTS failed: {e}")
                sys.exit(1)
            progress.update(task_tts, completed=1)

            task_write = progress.add_task("Writing output files...", total=2)
            script_path, sources_path = write_podcast_output(script, sources_md, output_dir)
            progress.update(task_write, completed=2)
    finally:
        if kb:
            kb.close()

    console.print()
    console.print("[bold green]Done![/bold green]")
    console.print(f"  Audio:   {audio_path}")
    console.print(f"  Script:  {script_path}")
    console.print(f"  Sources: {sources_path}")
