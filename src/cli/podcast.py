import os
import sys
from datetime import date

from core.knowledge_base import init_kb
from fileio.progress import console, create_progress
from utils.validation import check_ollama


def run_podcast(output_dir, llm_model, kb_dir=None, kb_rebuild=False, embedding_model=None):
    """Run the podcast generation pipeline."""
    from podcast.scriptwriter import generate_podcast, write_podcast_output
    from podcast.tts import get_tts_engine

    from main import load_interests, load_podcast_config

    interests = load_interests()
    if not interests:
        console.print("[bold red]Error:[/bold red] interest.md is required for podcast mode.")
        console.print("Create an interest.md file describing your interests.")
        sys.exit(1)

    config = load_podcast_config()
    style = config.get("podcast", {}).get("style", "solo")

    if output_dir is None:
        output_dir = os.path.join("output", f"podcast_{date.today().isoformat()}")

    console.print("[bold]Podcast Generator[/bold]")
    console.print(f"  Style:    {style}")
    console.print(f"  LLM:      {llm_model}")
    console.print(f"  TTS:      {config.get('tts', {}).get('engine', 'piper')}")
    if kb_dir:
        console.print(f"  KB:       {kb_dir}")
    console.print(f"  Output:   {output_dir}")
    console.print()

    check_ollama(llm_model)
    os.makedirs(output_dir, exist_ok=True)

    kb = None

    with create_progress() as progress:
        if kb_dir:
            kb = init_kb(kb_dir, kb_rebuild, embedding_model, progress, console)

        try:
            script, sources_md = generate_podcast(interests, config, llm_model, progress, kb=kb)
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] Podcast generation failed: {e}")
            if kb:
                kb.close()
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

    if kb:
        kb.close()

    console.print()
    console.print("[bold green]Done![/bold green]")
    console.print(f"  Audio:   {audio_path}")
    console.print(f"  Script:  {script_path}")
    console.print(f"  Sources: {sources_path}")
