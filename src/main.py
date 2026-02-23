import os
import sys

import click

from core.chunker import chunk_transcript
from core.llm import (
    consolidate_summaries,
    generate_remarks,
    generate_watch_score,
    summarize_chunk,
)
from core.transcriber import transcribe
from fileio.downloader import download_youtube_audio
from fileio.progress import console, create_progress
from fileio.writer import write_remarks, write_summary, write_transcript
from utils.validation import check_audio_file, check_ollama


def derive_output_dir(audio_file: str | None, video_title: str | None) -> str:
    """Derive the default output directory from the input source."""
    if video_title:
        # YouTube: use sanitized video title
        from fileio.downloader import sanitize_filename
        name = sanitize_filename(video_title)
    else:
        # Local file: use filename stem
        name = os.path.splitext(os.path.basename(audio_file))[0]
    return os.path.join("output", name)


@click.command()
@click.argument("audio_file", required=False, type=click.Path(exists=True))
@click.option(
    "--youtube",
    "-yt",
    default=None,
    help="YouTube URL to download and process.",
)
@click.option(
    "--model",
    "-m",
    type=click.Choice(["tiny", "base", "small", "medium", "large-v2", "large-v3"]),
    default="medium",
    help="Whisper model size. Default: medium.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default=None,
    help="Directory for output files. Default: output/<name>/.",
)
@click.option(
    "--llm-model",
    default="llama3.1:8b",
    help="Ollama model for summarization. Default: llama3.1:8b.",
)
@click.option(
    "--language",
    "-l",
    type=click.Choice(["auto", "nl", "en"]),
    default="auto",
    help="Audio language: auto (detect), nl (Dutch), en (English). Default: auto.",
)
@click.option(
    "--chunk-minutes",
    type=int,
    default=10,
    help="Transcript chunk size in minutes for LLM processing. Default: 10.",
)
def main(audio_file, youtube, model, output_dir, llm_model, language, chunk_minutes):
    """Transcribe and summarize a meeting audio file or YouTube video.

    Provide either an AUDIO_FILE path or --youtube URL (not both).
    Produces three output files: transcript.md, summary.md, and remarks.md.
    """
    # Validate: exactly one input source
    if audio_file and youtube:
        console.print("[bold red]Error:[/bold red] Provide either an audio file or --youtube, not both.")
        sys.exit(1)
    if not audio_file and not youtube:
        console.print("[bold red]Error:[/bold red] Provide an audio file or --youtube URL.")
        sys.exit(1)

    is_youtube = youtube is not None
    video_title = None
    content_type = "video" if is_youtube else "meeting"

    # Load user interests if available
    user_interests = None
    if is_youtube and os.path.isfile("interest.md"):
        with open("interest.md", encoding="utf-8") as f:
            user_interests = f.read().strip() or None

    console.print("[bold]Audio Summarizer[/bold]")

    # Step 1: Resolve audio source
    if is_youtube:
        console.print(f"  Source:   YouTube — {youtube}")
        console.print(f"  Whisper:  {model}")
        console.print(f"  LLM:      {llm_model}")
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
        console.print(f"  Whisper:  {model}")
        console.print(f"  LLM:      {llm_model}")
        console.print(f"  Language: {language}")

    # Step 2: Resolve output directory
    if output_dir is None:
        output_dir = derive_output_dir(audio_file, video_title)
    console.print(f"  Output:   {output_dir}")
    console.print()

    # Pre-flight checks
    check_audio_file(audio_file)
    check_ollama(llm_model)
    os.makedirs(output_dir, exist_ok=True)

    # Resolve language parameter
    lang = None if language == "auto" else language

    with create_progress() as progress:
        # Stage 1: Transcription
        task_transcribe = progress.add_task("Transcribing audio...", total=None)
        try:
            segments, duration = transcribe(audio_file, model, lang, progress, task_transcribe)
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] Transcription failed: {e}")
            sys.exit(1)

        if not segments:
            console.print("\n[bold red]Error:[/bold red] No speech detected in the audio file.")
            sys.exit(1)

        # Stage 2: Chunking
        chunks = chunk_transcript(segments, chunk_minutes)
        console.print(f"  Split transcript into {len(chunks)} chunk(s) for summarization")

        # Stage 3: Chunk summarization
        task_summarize = progress.add_task("Summarizing chunks...", total=len(chunks))
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            try:
                summary = summarize_chunk(chunk, i, len(chunks), llm_model, content_type)
                chunk_summaries.append(summary)
            except Exception as e:
                console.print(
                    f"\n[bold red]Error:[/bold red] Summarization failed on chunk {i + 1}: {e}"
                )
                sys.exit(1)
            progress.update(task_summarize, advance=1)

        # Stage 4: Consolidation
        task_consolidate = progress.add_task("Consolidating summary...", total=1)
        try:
            final_summary = consolidate_summaries(chunk_summaries, llm_model, content_type)
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] Consolidation failed: {e}")
            sys.exit(1)
        progress.update(task_consolidate, advance=1)

        # Stage 5: Watch score (YouTube only)
        score_block = None
        if is_youtube:
            task_score = progress.add_task("Scoring watch recommendation...", total=1)
            try:
                score_block = generate_watch_score(final_summary, llm_model, user_interests)
            except Exception as e:
                console.print(f"\n[bold red]Error:[/bold red] Score generation failed: {e}")
                sys.exit(1)
            progress.update(task_score, advance=1)

        # Stage 6: Remarks
        task_remarks = progress.add_task("Generating remarks...", total=1)
        try:
            remarks = generate_remarks(final_summary, llm_model, content_type)
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] Remarks generation failed: {e}")
            sys.exit(1)
        progress.update(task_remarks, advance=1)

        # Stage 7: Write files
        task_write = progress.add_task("Writing output files...", total=3)
        t_path = write_transcript(segments, output_dir, content_type)
        progress.update(task_write, advance=1)
        s_path = write_summary(final_summary, output_dir)
        progress.update(task_write, advance=1)
        r_path = write_remarks(remarks, output_dir, score_block=score_block)
        progress.update(task_write, advance=1)

    console.print()
    console.print("[bold green]Done![/bold green]")
    console.print(f"  Transcript: {t_path}")
    console.print(f"  Summary:    {s_path}")
    console.print(f"  Remarks:    {r_path}")


if __name__ == "__main__":
    main()
