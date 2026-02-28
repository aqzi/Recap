import os
import sys

from core.chunker import chunk_text, chunk_transcript
from core.knowledge_base import init_kb, extract_text, find_supported_files, SUPPORTED_EXTENSIONS
from core.llm import (
    consolidate_summaries,
    summarize_chunk,
)
from fileio.progress import console, create_progress
from fileio.writer import write_summary, write_summary_named, write_transcript
from utils.validation import check_audio_file, check_llm_model


def _summarize_chunks(chunks, llm_model, hint, kb, output_language, is_audio, progress):
    """Summarize a list of chunks and consolidate into a final summary."""
    task_summarize = progress.add_task("Summarizing chunks...", total=len(chunks))
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        chunk_kb = kb.retrieve(chunk["text"], top_k=3, max_chars=1500) if kb else None
        chunk_kb = chunk_kb or None
        try:
            summary = summarize_chunk(
                chunk, i, len(chunks), llm_model,
                hint=hint, kb_context=chunk_kb,
                output_language=output_language, is_audio=is_audio,
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
            output_language=output_language, is_audio=is_audio,
        )
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] Consolidation failed: {e}")
        sys.exit(1)
    progress.update(task_consolidate, advance=1)
    return final_summary


def run_summarizer(audio_file, model, output_dir, llm_model, language, chunk_minutes,
                   kb_dir=None, kb_rebuild=False, embedding_model=None, hint=None,
                   input_path=None, per_file=False, output_language="en"):
    """Run the summarization pipeline.

    Input modes:
      - audio_file set: transcribe audio -> time-chunk -> summarize
      - input_path is a file: load text -> chunk_text -> summarize
      - input_path is a directory: load all files -> summarize combined or per-file
    """
    if not audio_file and not input_path:
        console.print("[bold red]Error:[/bold red] Provide an audio file, --summarize PATH, or use --podcast.")
        sys.exit(1)

    is_audio = bool(audio_file)

    # Determine what we're summarizing and print header
    if input_path and os.path.isdir(input_path):
        console.print("[bold]Text Summarizer[/bold]")
        console.print(f"  Directory: {input_path}")
        console.print(f"  Per-file:  {per_file}")
    elif input_path:
        console.print("[bold]Text Summarizer[/bold]")
        console.print(f"  File: {input_path}")
    else:
        console.print("[bold]Audio Summarizer[/bold]")
        from core.transcriber import _USE_MLX
        backend = "mlx-whisper (GPU)" if _USE_MLX else "faster-whisper (CPU)"
        console.print(f"  Audio:    {audio_file}")
        console.print(f"  Whisper:  {model} — {backend}")
        console.print(f"  Language: {language}")

    from core.llm import detect_provider
    provider = detect_provider(llm_model)
    console.print(f"  LLM:      {llm_model} ({provider})")
    if output_language != "en":
        from core.prompts import _language_name
        console.print(f"  Output:   {_language_name(output_language)}")
    if hint:
        console.print(f"  Hint:     {hint}")
    if kb_dir:
        console.print(f"  KB:       {kb_dir}")

    if output_dir is None:
        from main import derive_output_dir
        source = audio_file or input_path
        output_dir = derive_output_dir(source)
    console.print(f"  Output:   {output_dir}")
    console.print()

    if audio_file:
        check_audio_file(audio_file)
    check_llm_model(llm_model)
    os.makedirs(output_dir, exist_ok=True)

    kb = None

    try:
        with create_progress() as progress:
            if kb_dir:
                kb = init_kb(kb_dir, kb_rebuild, embedding_model, progress, console)

            # --- Directory per-file mode ---
            if input_path and os.path.isdir(input_path) and per_file:
                files = find_supported_files(input_path)
                if not files:
                    console.print(
                        f"\n[bold red]Error:[/bold red] No supported files in {input_path} "
                        f"(supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))})"
                    )
                    sys.exit(1)

                summary_paths = []
                for filepath in files:
                    fname = os.path.basename(filepath)
                    name_stem = os.path.splitext(fname)[0]
                    console.print(f"  Processing: {fname}")

                    text = extract_text(filepath)
                    if not text or not text.strip():
                        console.print(f"  [yellow]Skipping (no text):[/yellow] {fname}")
                        continue

                    chunks = chunk_text(text)
                    if not chunks:
                        console.print(f"  [yellow]Skipping (empty):[/yellow] {fname}")
                        continue

                    final_summary = _summarize_chunks(
                        chunks, llm_model, hint, kb, output_language, False, progress,
                    )
                    s_path = write_summary_named(final_summary, output_dir, name_stem)
                    summary_paths.append(s_path)

                console.print()
                console.print("[bold green]Done![/bold green]")
                for p in summary_paths:
                    console.print(f"  Summary: {p}")
                return

            # --- Directory combined or single text file ---
            if input_path:
                if os.path.isdir(input_path):
                    from podcast.loader import load_input_text
                    try:
                        text, source_files = load_input_text(input_path)
                    except ValueError as e:
                        console.print(f"\n[bold red]Error:[/bold red] {e}")
                        sys.exit(1)
                    console.print(f"  Loaded {len(source_files)} file(s)")
                else:
                    text = extract_text(input_path)
                    if not text or not text.strip():
                        console.print(f"\n[bold red]Error:[/bold red] Could not extract text from {input_path}")
                        sys.exit(1)

                chunks = chunk_text(text)
                if not chunks:
                    console.print("\n[bold red]Error:[/bold red] No content to summarize.")
                    sys.exit(1)

                console.print(f"  Split text into {len(chunks)} chunk(s) for summarization")

                final_summary = _summarize_chunks(
                    chunks, llm_model, hint, kb, output_language, False, progress,
                )

                task_write = progress.add_task("Writing output files...", total=1)
                s_path = write_summary(final_summary, output_dir)
                progress.update(task_write, advance=1)

                console.print()
                console.print("[bold green]Done![/bold green]")
                console.print(f"  Summary: {s_path}")
                return

            # --- Audio mode ---
            from core.transcriber import transcribe, parse_transcript

            task_transcribe = progress.add_task("Transcribing audio...", total=None)
            try:
                segments, duration = transcribe(audio_file, model,
                                                None if language == "auto" else language,
                                                progress, task_transcribe)
            except Exception as e:
                console.print(f"\n[bold red]Error:[/bold red] Transcription failed: {e}")
                sys.exit(1)

            if not segments:
                console.print("\n[bold red]Error:[/bold red] No speech detected in the audio file.")
                sys.exit(1)

            chunks = chunk_transcript(segments, chunk_minutes)
            console.print(f"  Split transcript into {len(chunks)} chunk(s) for summarization")

            final_summary = _summarize_chunks(
                chunks, llm_model, hint, kb, output_language, True, progress,
            )

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
    console.print(f"  Transcript: {t_path}")
    console.print(f"  Summary:    {s_path}")
