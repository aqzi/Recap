import os
import sys

from core.chunker import chunk_transcript
from core.knowledge_base import init_kb
from core.llm import (
    consolidate_summaries,
    enhance_with_kb,
    summarize_chunk,
)
from core.transcriber import transcribe, _USE_MLX
from fileio.progress import console, create_progress
from fileio.writer import write_summary, write_transcript
from utils.validation import check_audio_file, check_ollama


def run_summarizer(audio_file, model, output_dir, llm_model, language, chunk_minutes,
                   kb_dir=None, kb_rebuild=False, embedding_model=None, context=None):
    """Run the audio summarization pipeline."""
    if not audio_file:
        console.print("[bold red]Error:[/bold red] Provide an audio file or use --podcast.")
        sys.exit(1)

    console.print("[bold]Audio Summarizer[/bold]")

    backend = "mlx-whisper (GPU)" if _USE_MLX else "faster-whisper (CPU)"

    console.print(f"  Audio:    {audio_file}")
    console.print(f"  Whisper:  {model} — {backend}")
    console.print(f"  LLM:      {llm_model}")
    console.print(f"  Language: {language}")
    if context:
        console.print(f"  Context:  {context}")
    if kb_dir:
        console.print(f"  KB:       {kb_dir}")

    if output_dir is None:
        from main import derive_output_dir
        output_dir = derive_output_dir(audio_file)
    console.print(f"  Output:   {output_dir}")
    console.print()

    check_audio_file(audio_file)
    check_ollama(llm_model)
    os.makedirs(output_dir, exist_ok=True)

    kb = None
    lang = None if language == "auto" else language

    with create_progress() as progress:
        if kb_dir:
            kb = init_kb(kb_dir, kb_rebuild, embedding_model, progress, console)

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
                summary = summarize_chunk(chunk, i, len(chunks), llm_model, context=context)
                chunk_summaries.append(summary)
            except Exception as e:
                console.print(f"\n[bold red]Error:[/bold red] Summarization failed on chunk {i + 1}: {e}")
                if kb:
                    kb.close()
                sys.exit(1)
            progress.update(task_summarize, advance=1)

        task_consolidate = progress.add_task("Consolidating summary...", total=1)
        try:
            final_summary = consolidate_summaries(chunk_summaries, llm_model, context=context)
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] Consolidation failed: {e}")
            if kb:
                kb.close()
            sys.exit(1)
        progress.update(task_consolidate, advance=1)

        if kb:
            task_enhance = progress.add_task("Enhancing with KB context...", total=1)
            kb_context = kb.retrieve(final_summary, top_k=5, max_chars=6000)
            if kb_context:
                try:
                    final_summary = enhance_with_kb(final_summary, kb_context, llm_model)
                except Exception as e:
                    console.print(f"\n[bold red]Error:[/bold red] KB enhancement failed: {e}")
                    kb.close()
                    sys.exit(1)
            progress.update(task_enhance, advance=1)

        task_write = progress.add_task("Writing output files...", total=2)
        t_path = write_transcript(segments, output_dir)
        progress.update(task_write, advance=1)
        s_path = write_summary(final_summary, output_dir)
        progress.update(task_write, advance=1)

    if kb:
        kb.close()

    console.print()
    console.print("[bold green]Done![/bold green]")
    console.print(f"  Transcript: {t_path}")
    console.print(f"  Summary:    {s_path}")
