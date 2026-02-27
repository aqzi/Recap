import os
import sys

from core.chunker import chunk_transcript
from core.knowledge_base import init_kb
from core.llm import (
    consolidate_summaries,
    summarize_chunk,
)
from core.transcriber import transcribe, parse_transcript, _USE_MLX
from fileio.progress import console, create_progress
from fileio.writer import write_summary, write_transcript
from utils.validation import check_audio_file, check_llm_model


def run_summarizer(audio_file, model, output_dir, llm_model, language, chunk_minutes,
                   kb_dir=None, kb_rebuild=False, embedding_model=None, hint=None,
                   transcript=None):
    """Run the audio summarization pipeline."""
    if not audio_file and not transcript:
        console.print("[bold red]Error:[/bold red] Provide an audio file, --transcript, or use --podcast.")
        sys.exit(1)

    if transcript:
        console.print("[bold]Transcript Summarizer[/bold]")
        console.print(f"  Transcript: {transcript}")
    else:
        console.print("[bold]Audio Summarizer[/bold]")
        backend = "mlx-whisper (GPU)" if _USE_MLX else "faster-whisper (CPU)"
        console.print(f"  Audio:    {audio_file}")
        console.print(f"  Whisper:  {model} — {backend}")
        console.print(f"  Language: {language}")

    from core.llm import detect_provider
    provider = detect_provider(llm_model)
    console.print(f"  LLM:      {llm_model} ({provider})")
    if hint:
        console.print(f"  Hint:     {hint}")
    if kb_dir:
        console.print(f"  KB:       {kb_dir}")

    if output_dir is None:
        from main import derive_output_dir
        source = audio_file if audio_file else transcript
        output_dir = derive_output_dir(source)
    console.print(f"  Output:   {output_dir}")
    console.print()

    if not transcript:
        check_audio_file(audio_file)
    check_llm_model(llm_model)
    os.makedirs(output_dir, exist_ok=True)

    kb = None
    lang = None if language == "auto" else language

    try:
        with create_progress() as progress:
            if kb_dir:
                kb = init_kb(kb_dir, kb_rebuild, embedding_model, progress, console)

            if transcript:
                task_parse = progress.add_task("Parsing transcript...", total=1)
                segments = parse_transcript(transcript)
                progress.update(task_parse, advance=1)
            else:
                task_transcribe = progress.add_task("Transcribing audio...", total=None)
                try:
                    segments, duration = transcribe(audio_file, model, lang, progress, task_transcribe)
                except Exception as e:
                    console.print(f"\n[bold red]Error:[/bold red] Transcription failed: {e}")
                    sys.exit(1)

            if not segments:
                console.print("\n[bold red]Error:[/bold red] No speech detected in the audio file.")
                sys.exit(1)

            chunks = chunk_transcript(segments, chunk_minutes)
            console.print(f"  Split transcript into {len(chunks)} chunk(s) for summarization")

            task_summarize = progress.add_task("Summarizing chunks...", total=len(chunks))
            chunk_summaries = []
            for i, chunk in enumerate(chunks):
                chunk_kb = kb.retrieve(chunk["text"], top_k=3, max_chars=1500) if kb else None
                chunk_kb = chunk_kb or None
                try:
                    summary = summarize_chunk(
                        chunk, i, len(chunks), llm_model,
                        hint=hint, kb_context=chunk_kb,
                    )
                    chunk_summaries.append(summary)
                except Exception as e:
                    console.print(f"\n[bold red]Error:[/bold red] Summarization failed on chunk {i + 1}: {e}")
                    sys.exit(1)
                progress.update(task_summarize, advance=1)

            task_consolidate = progress.add_task("Consolidating summary...", total=1)
            consolidation_kb = (
                kb.retrieve_multi(chunk_summaries, top_k_per_query=3, max_chars=3000)
                if kb else None
            )
            consolidation_kb = consolidation_kb or None
            try:
                final_summary = consolidate_summaries(
                    chunk_summaries, llm_model,
                    hint=hint, kb_context=consolidation_kb,
                )
            except Exception as e:
                console.print(f"\n[bold red]Error:[/bold red] Consolidation failed: {e}")
                sys.exit(1)
            progress.update(task_consolidate, advance=1)

            if transcript:
                task_write = progress.add_task("Writing output files...", total=1)
                s_path = write_summary(final_summary, output_dir)
                progress.update(task_write, advance=1)
            else:
                task_write = progress.add_task("Writing output files...", total=2)
                t_path = write_transcript(segments, output_dir)
                progress.update(task_write, advance=1)
                s_path = write_summary(final_summary, output_dir)
                progress.update(task_write, advance=1)
    finally:
        if kb:
            kb.close()

    console.print()
    console.print("[bold green]Done![/bold green]")
    if not transcript:
        console.print(f"  Transcript: {t_path}")
    console.print(f"  Summary:    {s_path}")
