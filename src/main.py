import os
import sys
import warnings
from datetime import date

import click
import yaml

# Suppress unclosed socket warnings from the ollama library's HTTP client
warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed.*socket")

from core.chunker import chunk_transcript
from core.llm import (
    consolidate_summaries,
    enhance_with_kb,
    generate_remarks,
    generate_watch_score,
    summarize_chunk,
)
from core.transcriber import transcribe, _USE_MLX
from fileio.downloader import download_youtube_audio
from fileio.progress import console, create_progress
from fileio.writer import write_remarks, write_summary, write_transcript
from rich.prompt import Confirm, Prompt
from utils.validation import check_audio_file, check_ollama


def derive_output_dir(audio_file: str | None, video_title: str | None) -> str:
    """Derive the default output directory from the input source."""
    if video_title:
        from fileio.downloader import sanitize_filename
        name = sanitize_filename(video_title)
    else:
        name = os.path.splitext(os.path.basename(audio_file))[0]
    return os.path.join("output", name)


def load_interests() -> str | None:
    """Load interest.md if it exists."""
    if os.path.isfile("interest.md"):
        with open("interest.md", encoding="utf-8") as f:
            text = f.read().strip()
            return text or None
    return None


def load_podcast_config() -> dict:
    """Load podcast_config.yaml or return defaults."""
    if os.path.isfile("podcast_config.yaml"):
        with open("podcast_config.yaml", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


# ---------- Podcast pipeline ----------

def run_podcast(output_dir: str | None, llm_model: str, kb_dir: str | None = None,
                kb_rebuild: bool = False, embedding_model: str | None = None):
    """Run the podcast generation pipeline."""
    from podcast.scriptwriter import generate_podcast, write_podcast_output
    from podcast.tts import get_tts_engine

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

    # Load knowledge base (if provided)
    kb = None
    if kb_dir:
        from core.knowledge_base import KnowledgeBase

        kb_kwargs = {}
        if embedding_model:
            kb_kwargs["embedding_model"] = embedding_model
        kb = KnowledgeBase(**kb_kwargs)
        console.print(f"  Embed:    {kb.embedding_model}")

        # Check for model mismatch
        mismatched_model = kb.check_model_mismatch()
        if mismatched_model:
            console.print(f"[bold red]Error:[/bold red] KB was indexed with model '{mismatched_model}' "
                          f"but you requested '{kb.embedding_model}'. "
                          f"Re-run with --kb-rebuild to re-index.")
            kb.close()
            sys.exit(1)

        if kb_rebuild:
            console.print("  Rebuilding knowledge base index...")
            kb.delete_collection()

    with create_progress() as progress:
        # Index knowledge base (if needed)
        if kb and kb_dir:
            if kb.is_indexed() and not kb_rebuild:
                console.print(f"  KB loaded from cache ({kb.chunk_count} chunks)")
            else:
                task_kb = progress.add_task("Indexing knowledge base...", total=None)
                try:
                    files_loaded = kb.index_directory(kb_dir, progress, task_kb)
                except Exception as e:
                    console.print(f"\n[bold red]Error:[/bold red] KB indexing failed: {e}")
                    kb.close()
                    sys.exit(1)
                if files_loaded == 0:
                    console.print("  [yellow]Warning:[/yellow] No supported files found in KB directory.")
                    kb.close()
                    kb = None
                else:
                    console.print(f"  Indexed {files_loaded} file(s), {kb.chunk_count} chunks")

        # Stages 1-4: Fetch, rank, extract, generate script (with KB context if available)
        try:
            script, sources_md = generate_podcast(interests, config, llm_model, progress, kb=kb)
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] Podcast generation failed: {e}")
            if kb:
                kb.close()
            sys.exit(1)

        # Stage 5: Text-to-speech
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

        # Stage 6: Write output files
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


# ---------- Summarizer pipeline ----------

def run_summarizer(audio_file, youtube, model, output_dir, llm_model, language, chunk_minutes,
                   kb_dir=None, kb_rebuild=False, embedding_model=None):
    """Run the audio/video summarization pipeline."""
    # Validate: exactly one input source
    if audio_file and youtube:
        console.print("[bold red]Error:[/bold red] Provide either an audio file or --youtube, not both.")
        sys.exit(1)
    if not audio_file and not youtube:
        console.print("[bold red]Error:[/bold red] Provide an audio file, --youtube URL, or --podcast.")
        sys.exit(1)

    is_youtube = youtube is not None
    video_title = None
    content_type = "video" if is_youtube else "meeting"

    user_interests = None
    if is_youtube:
        user_interests = load_interests()

    console.print("[bold]Audio Summarizer[/bold]")

    # Step 1: Resolve audio source
    backend = "mlx-whisper (GPU)" if _USE_MLX else "faster-whisper (CPU)"

    if is_youtube:
        console.print(f"  Source:   YouTube — {youtube}")
        console.print(f"  Whisper:  {model} — {backend}")
        console.print(f"  LLM:      {llm_model}")
        if kb_dir:
            console.print(f"  KB:       {kb_dir}")
        console.print()
        console.print("Downloading YouTube audio...")
        try:
            audio_file, video_title = download_youtube_audio(youtube)
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] YouTube download failed: {e}")
            sys.exit(1)
        console.print(f"  Downloaded: {audio_file}")
    else:
        console.print(f"  Audio:    {audio_file}")
        console.print(f"  Whisper:  {model} — {backend}")
        console.print(f"  LLM:      {llm_model}")
        console.print(f"  Language: {language}")
        if kb_dir:
            console.print(f"  KB:       {kb_dir}")

    # Step 2: Resolve output directory
    if output_dir is None:
        output_dir = derive_output_dir(audio_file, video_title)
    console.print(f"  Output:   {output_dir}")
    console.print()

    check_audio_file(audio_file)
    check_ollama(llm_model)
    os.makedirs(output_dir, exist_ok=True)

    # Step 3: Load knowledge base (if provided)
    kb = None
    if kb_dir:
        from core.knowledge_base import KnowledgeBase

        kb_kwargs = {}
        if embedding_model:
            kb_kwargs["embedding_model"] = embedding_model
        kb = KnowledgeBase(**kb_kwargs)
        console.print(f"  Embed:    {kb.embedding_model}")

        # Check for model mismatch
        mismatched_model = kb.check_model_mismatch()
        if mismatched_model:
            console.print(f"[bold red]Error:[/bold red] KB was indexed with model '{mismatched_model}' "
                          f"but you requested '{kb.embedding_model}'. "
                          f"Re-run with --kb-rebuild to re-index.")
            kb.close()
            sys.exit(1)

        if kb_rebuild:
            console.print("  Rebuilding knowledge base index...")
            kb.delete_collection()

    lang = None if language == "auto" else language

    with create_progress() as progress:
        # Index knowledge base (if needed)
        if kb and kb_dir:
            if kb.is_indexed() and not kb_rebuild:
                console.print(f"  KB loaded from cache ({kb.chunk_count} chunks)")
            else:
                task_kb = progress.add_task("Indexing knowledge base...", total=None)
                try:
                    files_loaded = kb.index_directory(kb_dir, progress, task_kb)
                except Exception as e:
                    console.print(f"\n[bold red]Error:[/bold red] KB indexing failed: {e}")
                    kb.close()
                    sys.exit(1)
                if files_loaded == 0:
                    console.print("  [yellow]Warning:[/yellow] No supported files found in KB directory.")
                    kb.close()
                    kb = None
                else:
                    console.print(f"  Indexed {files_loaded} file(s), {kb.chunk_count} chunks")

        task_transcribe = progress.add_task("Transcribing audio...", total=None)
        try:
            segments, duration = transcribe(audio_file, model, lang, progress, task_transcribe)
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] Transcription failed: {e}")
            if kb:
                kb.close()
            sys.exit(1)

        if not segments:
            console.print("\n[bold red]Error:[/bold red] No speech detected in the audio file.")
            if kb:
                kb.close()
            sys.exit(1)

        chunks = chunk_transcript(segments, chunk_minutes)
        console.print(f"  Split transcript into {len(chunks)} chunk(s) for summarization")

        task_summarize = progress.add_task("Summarizing chunks...", total=len(chunks))
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            try:
                summary = summarize_chunk(
                    chunk, i, len(chunks), llm_model, content_type,
                )
                chunk_summaries.append(summary)
            except Exception as e:
                console.print(f"\n[bold red]Error:[/bold red] Summarization failed on chunk {i + 1}: {e}")
                if kb:
                    kb.close()
                sys.exit(1)
            progress.update(task_summarize, advance=1)

        task_consolidate = progress.add_task("Consolidating summary...", total=1)
        try:
            final_summary = consolidate_summaries(
                chunk_summaries, llm_model, content_type,
            )
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] Consolidation failed: {e}")
            if kb:
                kb.close()
            sys.exit(1)
        progress.update(task_consolidate, advance=1)

        # Enhancement pass: enrich summary with KB context (if available)
        if kb:
            task_enhance = progress.add_task("Enhancing with KB context...", total=1)
            kb_context = kb.retrieve(final_summary, top_k=5, max_chars=6000)
            if kb_context:
                try:
                    final_summary = enhance_with_kb(
                        final_summary, kb_context, llm_model, content_type,
                    )
                except Exception as e:
                    console.print(f"\n[bold red]Error:[/bold red] KB enhancement failed: {e}")
                    if kb:
                        kb.close()
                    sys.exit(1)
            progress.update(task_enhance, advance=1)

        score_block = None
        if is_youtube:
            task_score = progress.add_task("Scoring watch recommendation...", total=1)
            try:
                score_block = generate_watch_score(final_summary, llm_model, user_interests)
            except Exception as e:
                console.print(f"\n[bold red]Error:[/bold red] Score generation failed: {e}")
                if kb:
                    kb.close()
                sys.exit(1)
            progress.update(task_score, advance=1)

        task_remarks = progress.add_task("Generating remarks...", total=1)
        try:
            remarks = generate_remarks(final_summary, llm_model, content_type)
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] Remarks generation failed: {e}")
            if kb:
                kb.close()
            sys.exit(1)
        progress.update(task_remarks, advance=1)

        task_write = progress.add_task("Writing output files...", total=3)
        t_path = write_transcript(segments, output_dir, content_type)
        progress.update(task_write, advance=1)
        s_path = write_summary(final_summary, output_dir)
        progress.update(task_write, advance=1)
        r_path = write_remarks(remarks, output_dir, score_block=score_block)
        progress.update(task_write, advance=1)

    if kb:
        kb.close()

    console.print()
    console.print("[bold green]Done![/bold green]")
    console.print(f"  Transcript: {t_path}")
    console.print(f"  Summary:    {s_path}")
    console.print(f"  Remarks:    {r_path}")


# ---------- Interactive wizard ----------

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
LANGUAGES = ["auto", "nl", "en"]


def interactive_mode():
    """Guide the user through mode selection and options step by step."""
    console.print()
    console.print("[bold]Audio Summarizer & Podcast Generator[/bold]")
    console.print()
    console.print("  [bold cyan][1][/bold cyan] Summarize a local audio file")
    console.print("  [bold cyan][2][/bold cyan] Summarize a YouTube video")
    console.print("  [bold cyan][3][/bold cyan] Generate a podcast")
    console.print()

    mode = Prompt.ask("Select mode", choices=["1", "2", "3"])

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
        # LLM model (all modes)
        llm_model = Prompt.ask("Ollama model", default="llama3.1:8b")

        # Whisper model (audio/youtube only)
        if mode in ("1", "2"):
            whisper_model = Prompt.ask(
                f"Whisper model ({', '.join(WHISPER_MODELS)})",
                choices=WHISPER_MODELS, default="medium",
            )

        # Language (audio file only)
        if mode == "1":
            language = Prompt.ask(
                f"Language ({', '.join(LANGUAGES)})",
                choices=LANGUAGES, default="auto",
            )

        # Knowledge base (all modes)
        kb_input = Prompt.ask("Knowledge base directory (empty to skip)", default="")
        if kb_input:
            if os.path.isdir(kb_input):
                kb_dir = kb_input
                kb_rebuild = Confirm.ask("Rebuild KB index?", default=False)
            else:
                console.print(f"  [yellow]Warning:[/yellow] Directory not found: {kb_input}, skipping KB.")

        # Output directory (all modes)
        out_input = Prompt.ask("Output directory (empty for auto)", default="")
        if out_input:
            output_dir = out_input

    console.print()

    # Dispatch
    if mode == "3":
        run_podcast(output_dir, llm_model, kb_dir=kb_dir, kb_rebuild=kb_rebuild,
                    embedding_model=embedding_model)
    else:
        run_summarizer(audio_file, youtube, whisper_model, output_dir, llm_model,
                       language, chunk_minutes, kb_dir=kb_dir, kb_rebuild=kb_rebuild,
                       embedding_model=embedding_model)


# ---------- CLI ----------

@click.command()
@click.argument("audio_file", required=False, type=click.Path(exists=True))
@click.option("--youtube", "-yt", default=None, help="YouTube URL to download and process.")
@click.option("--podcast", is_flag=True, default=False, help="Generate a podcast from your interests.")
@click.option(
    "--model", "-m",
    type=click.Choice(["tiny", "base", "small", "medium", "large-v2", "large-v3"]),
    default="medium", help="Whisper model size. Default: medium.",
)
@click.option("--output-dir", "-o", type=click.Path(), default=None, help="Directory for output files.")
@click.option("--llm-model", default="llama3.1:8b", help="Ollama model. Default: llama3.1:8b.")
@click.option(
    "--language", "-l",
    type=click.Choice(["auto", "nl", "en"]),
    default="auto", help="Audio language. Default: auto.",
)
@click.option("--chunk-minutes", type=int, default=10, help="Chunk size in minutes. Default: 10.")
@click.option("--kb", type=click.Path(exists=True, file_okay=False),
              default=None, help="Knowledge base directory for context-aware summaries.")
@click.option("--kb-rebuild", is_flag=True, default=False,
              help="Force re-index the knowledge base (use when KB files changed).")
@click.option("--embedding-model", default=None,
              help="Fastembed model for KB embeddings. Default: BAAI/bge-small-en-v1.5.")
def main(audio_file, youtube, podcast, model, output_dir, llm_model, language, chunk_minutes, kb, kb_rebuild,
         embedding_model):
    """Transcribe and summarize audio, process YouTube videos, or generate podcasts.

    \b
    Modes:
      AUDIO_FILE                  Summarize a local audio file
      --youtube URL               Summarize a YouTube video
      --podcast                   Generate a podcast from your interests
    """
    # No mode specified — launch interactive wizard
    if not audio_file and not youtube and not podcast:
        interactive_mode()
        return

    if podcast:
        if audio_file or youtube:
            console.print("[bold red]Error:[/bold red] --podcast cannot be combined with an audio file or --youtube.")
            sys.exit(1)
        run_podcast(output_dir, llm_model, kb_dir=kb, kb_rebuild=kb_rebuild,
                    embedding_model=embedding_model)
    else:
        run_summarizer(audio_file, youtube, model, output_dir, llm_model, language, chunk_minutes,
                       kb_dir=kb, kb_rebuild=kb_rebuild, embedding_model=embedding_model)


if __name__ == "__main__":
    main()
