import os
import warnings
from pathlib import Path

import click
import yaml

warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed.*socket")

from fileio.progress import console

# Resolve paths relative to the project root (parent of src/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

LLM_CONFIG_PATH = str(_PROJECT_ROOT / "config.yaml")
PODCAST_CONFIG_PATH = str(_PROJECT_ROOT / "podcast_config.yaml")
DEFAULT_OUTPUT_DIR = str(_PROJECT_ROOT / "output")
DEFAULT_LLM_MODEL = "llama3.1:8b"


def derive_output_dir(source_path):
    """Derive the default output directory from the source filename or directory."""
    if os.path.isdir(source_path):
        name = os.path.basename(os.path.normpath(source_path))
    else:
        name = os.path.splitext(os.path.basename(source_path))[0]
    return os.path.join(DEFAULT_OUTPUT_DIR, name)


def load_podcast_config():
    """Load podcast_config.yaml or return defaults."""
    if os.path.isfile(PODCAST_CONFIG_PATH):
        with open(PODCAST_CONFIG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def load_llm_config() -> dict:
    """Load config.yaml or create it with defaults if missing."""
    if os.path.isfile(LLM_CONFIG_PATH):
        with open(LLM_CONFIG_PATH, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
            # Ensure the llm key exists
            config.setdefault("llm", {})
            config["llm"].setdefault("model", DEFAULT_LLM_MODEL)
            config.setdefault("output_language", "en")
            return config

    # Create default config
    config = {
        "llm": {
            "model": DEFAULT_LLM_MODEL,
            "openai_api_key": "",
            "anthropic_api_key": "",
        },
        "output_language": "en",
    }
    save_llm_config(config)
    return config


def save_llm_config(config: dict) -> None:
    """Write config back to config.yaml."""
    with open(LLM_CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_last_input_path(config: dict) -> str:
    """Return the last input path from config, or empty string."""
    return config.get("last_input_path", "") or ""


def set_last_input_path(config: dict, path: str) -> None:
    """Save the last input path to config and persist it."""
    config["last_input_path"] = path
    save_llm_config(config)


def get_output_language(config: dict) -> str:
    """Return the saved output language from config, or 'en'."""
    return config.get("output_language", "en") or "en"


def set_output_language(config: dict, lang: str) -> None:
    """Save the output language to config and persist it."""
    config["output_language"] = lang
    save_llm_config(config)


def _apply_llm_config_to_env(llm_config: dict) -> None:
    """Set API keys from config into environment variables."""
    from core.llm import _set_api_keys_from_config
    _set_api_keys_from_config(llm_config)


# ---------- CLI ----------


@click.command()
@click.argument("audio_file", required=False, type=click.Path(exists=True))
@click.option("--podcast", type=click.Path(exists=True), default=None,
              help="Generate a podcast from a file or directory of text.")
@click.option(
    "--model",
    type=click.Choice(["tiny", "base", "small", "medium", "large-v2", "large-v3"]),
    default="medium", help="Whisper model size. Default: medium.",
)
@click.option("--output-dir", type=click.Path(), default=None, help="Directory for output files.")
@click.option("--llm-model", default=None,
              help="LLM model. Supports Ollama, OpenAI (gpt-*), Anthropic (claude-*). Default from config.yaml.")
@click.option("--input-language", default="auto",
              help="Audio language for Whisper transcription (e.g. auto, en, nl, de, fr, es, ja, zh, ...). Default: auto.")
@click.option("--output-language", default=None,
              help="Output language for summaries and podcasts (e.g. en, nl, de, fr). Default from config or en.")
@click.option("--chunk-minutes", type=click.IntRange(min=1), default=10, help="Chunk size in minutes. Default: 10.")
@click.option("--kb", type=click.Path(exists=True, file_okay=False),
              default=None, help="Directory of reference docs for domain-aware summaries (RAG).")
@click.option("--kb-rebuild", is_flag=True, default=False,
              help="Force re-index the knowledge base (use when KB files changed).")
@click.option("--embedding-model", default=None,
              help="Fastembed model for KB embeddings. Default: BAAI/bge-small-en-v1.5.")
@click.option("--hint", default=None,
              help="Short label for the audio type (e.g., 'team meeting', 'lecture'). Guides tone/structure.")
@click.option("--record", "record_flag", is_flag=True, default=False, help="Record audio from an input device.")
@click.option("--record-name", default=None, help="Optional name for the recording file.")
@click.option("--summarize", type=click.Path(exists=True),
              default=None, help="Summarize a text file or directory (md, pdf, docx, txt, ...).")
@click.option("--per-file", is_flag=True, default=False,
              help="When summarizing a directory, produce one summary per file instead of one combined summary.")
def main(audio_file, podcast, model, output_dir, llm_model, input_language, output_language,
         chunk_minutes, kb, kb_rebuild, embedding_model, hint, record_flag, record_name,
         summarize, per_file):
    """Transcribe and summarize audio, summarize text, generate podcasts, or record audio.

    \b
    Modes:
      AUDIO_FILE                  Summarize a local audio file
      --summarize PATH            Summarize a text file or directory
      --podcast PATH              Generate a podcast from a file or directory
      --record                    Record audio from an input device
    """
    if record_flag:
        from cli.recorder import run_recorder
        run_recorder(output_dir, record_name)
        return

    if audio_file and summarize:
        console.print("[bold red]Error:[/bold red] Cannot use both an audio file and --summarize at the same time.")
        import sys
        sys.exit(1)

    if not audio_file and podcast is None and not summarize:
        from cli.interactive import interactive_mode
        interactive_mode()
        return

    # Load LLM config and apply API keys to environment
    llm_config = load_llm_config()
    _apply_llm_config_to_env(llm_config)

    # Resolve output language: CLI flag > config > default
    if output_language is None:
        output_language = get_output_language(llm_config)
    else:
        set_output_language(llm_config, output_language)

    # Prompt for model selection if --llm-model not explicitly provided
    if llm_model is None:
        from cli.interactive import _choose_llm_model
        llm_model, llm_config = _choose_llm_model(llm_config)
        save_llm_config(llm_config)
        _apply_llm_config_to_env(llm_config)
        console.print()

    if podcast is not None:
        from cli.podcast import run_podcast
        run_podcast(podcast, output_dir, llm_model, kb_dir=kb, kb_rebuild=kb_rebuild,
                    embedding_model=embedding_model, output_language=output_language)
    else:
        from cli.summarizer import run_summarizer
        run_summarizer(audio_file, model, output_dir, llm_model, input_language, chunk_minutes,
                       kb_dir=kb, kb_rebuild=kb_rebuild, embedding_model=embedding_model, hint=hint,
                       input_path=summarize, per_file=per_file, output_language=output_language)


if __name__ == "__main__":
    main()
