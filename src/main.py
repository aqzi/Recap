import os
import sys

import click

from chunker import chunk_transcript
from llm import consolidate_summaries, generate_remarks, summarize_chunk
from progress import console, create_progress
from transcriber import transcribe
from utils import check_audio_file, check_ollama
from writer import write_remarks, write_summary, write_transcript


@click.command()
@click.argument("audio_file", type=click.Path(exists=True))
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
    default=".",
    help="Directory for output files. Default: current directory.",
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
def main(audio_file, model, output_dir, llm_model, language, chunk_minutes):
    """Transcribe and summarize a meeting audio file.

    Produces three output files: transcript.md, summary.md, and remarks.md.
    """
    console.print("[bold]Meeting Summarizer[/bold]")
    console.print(f"  Audio:    {audio_file}")
    console.print(f"  Whisper:  {model}")
    console.print(f"  LLM:      {llm_model}")
    console.print(f"  Language: {language}")
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
                summary = summarize_chunk(chunk, i, len(chunks), llm_model)
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
            final_summary = consolidate_summaries(chunk_summaries, llm_model)
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] Consolidation failed: {e}")
            sys.exit(1)
        progress.update(task_consolidate, advance=1)

        # Stage 5: Remarks
        task_remarks = progress.add_task("Generating remarks...", total=1)
        try:
            remarks = generate_remarks(final_summary, llm_model)
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] Remarks generation failed: {e}")
            sys.exit(1)
        progress.update(task_remarks, advance=1)

        # Stage 6: Write files
        task_write = progress.add_task("Writing output files...", total=3)
        t_path = write_transcript(segments, output_dir)
        progress.update(task_write, advance=1)
        s_path = write_summary(final_summary, output_dir)
        progress.update(task_write, advance=1)
        r_path = write_remarks(remarks, output_dir)
        progress.update(task_write, advance=1)

    console.print()
    console.print("[bold green]Done![/bold green]")
    console.print(f"  Transcript: {t_path}")
    console.print(f"  Summary:    {s_path}")
    console.print(f"  Remarks:    {r_path}")


if __name__ == "__main__":
    main()