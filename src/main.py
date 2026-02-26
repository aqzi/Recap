import os
import warnings

import click
import yaml

warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed.*socket")

from fileio.progress import console


def derive_output_dir(audio_file):
    """Derive the default output directory from the audio filename."""
    name = os.path.splitext(os.path.basename(audio_file))[0]
    return os.path.join("output", name)


def load_interests():
    """Load interest.md if it exists."""
    if os.path.isfile("interest.md"):
        with open("interest.md", encoding="utf-8") as f:
            text = f.read().strip()
            return text or None
    return None


def load_podcast_config():
    """Load podcast_config.yaml or return defaults."""
    if os.path.isfile("podcast_config.yaml"):
        with open("podcast_config.yaml", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


# ---------- CLI ----------

@click.command()
@click.argument("audio_file", required=False, type=click.Path(exists=True))
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
@click.option("--context", "-c", default=None,
              help="Additional context about the audio (e.g., 'team meeting', 'university lecture').")
@click.option("--record", "record_flag", is_flag=True, default=False, help="Record audio from an input device.")
@click.option("--record-name", default=None, help="Optional name for the recording file.")
def main(audio_file, podcast, model, output_dir, llm_model, language, chunk_minutes, kb, kb_rebuild,
         embedding_model, context, record_flag, record_name):
    """Transcribe and summarize audio, generate podcasts, or record audio.

    \b
    Modes:
      AUDIO_FILE                  Summarize a local audio file
      --podcast                   Generate a podcast from your interests
      --record                    Record audio from an input device
    """
    if record_flag:
        from cli.recorder import run_recorder
        run_recorder(output_dir, record_name)
        return

    if not audio_file and not podcast:
        from cli.interactive import interactive_mode
        interactive_mode()
        return

    if podcast:
        if audio_file:
            console.print("[bold red]Error:[/bold red] --podcast cannot be combined with an audio file.")
            import sys
            sys.exit(1)
        from cli.podcast import run_podcast
        run_podcast(output_dir, llm_model, kb_dir=kb, kb_rebuild=kb_rebuild,
                    embedding_model=embedding_model)
    else:
        from cli.summarizer import run_summarizer
        run_summarizer(audio_file, model, output_dir, llm_model, language, chunk_minutes,
                       kb_dir=kb, kb_rebuild=kb_rebuild, embedding_model=embedding_model, context=context)


if __name__ == "__main__":
    main()
