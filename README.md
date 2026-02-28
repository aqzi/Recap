# Recap

A CLI tool that summarizes audio, text files, or entire directories, records from input devices, or generates personalized tech podcasts. Supports multi-language output. Runs locally by default with Ollama, or connect to cloud LLMs (OpenAI, Anthropic, etc.) via LiteLLM.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) for local models, or an API key for a cloud provider (OpenAI, Anthropic, etc.)
- ffmpeg
- TTS engine for podcast audio: macOS `say` (built-in, zero setup) or [Piper TTS](https://github.com/rhasspy/piper)

## Setup

```bash
pip install -r requirements.txt

# For local models (default):
ollama pull llama3.1:8b

# Or use a cloud model — set your API key:
export OPENAI_API_KEY=sk-...
# or
export ANTHROPIC_API_KEY=sk-ant-...
```

Config files (`config.yaml`, `podcast_config.yaml`) and output directories (`output/`, `data/`) are resolved relative to the project root, so the tool works correctly regardless of which directory you run it from.

## Modes

### 1. Summarize

#### Audio file

```bash
python src/main.py meeting.mp3
python src/main.py meeting.mp3 --hint "team standup meeting"
```

Output: `output/meeting/transcript.md`, `summary.md`

Use `--hint` to tell the summarizer what kind of audio it is (e.g., "team meeting", "university lecture", "interview"). This is a short label that guides tone and structure — it doesn't add domain knowledge (use `--kb` for that).

#### Text file or directory

Summarize any supported text file (`.md`, `.pdf`, `.docx`, `.txt`, `.html`, `.csv`):

```bash
python src/main.py --summarize notes.pdf
python src/main.py --summarize report.md --output-language nl
```

Summarize an entire directory of files into one combined summary:

```bash
python src/main.py --summarize ./meeting_notes/
```

Or produce one summary per file with `--per-file`:

```bash
python src/main.py --summarize ./meeting_notes/ --per-file
```

Output: `output/<name>/summary.md` (combined) or `output/<name>/summary_<filename>.md` (per-file)

### 2. Generate a podcast

```bash
python src/main.py --podcast input.md
python src/main.py --podcast ./notes/ --output-language fr
```

Generates a podcast script from input text and converts it to audio. Can optionally enrich with articles from RSS feeds and web search.

Output: `output/podcast_<name>/podcast.wav`, `script.md`, `sources.md`

### 3. Record audio

```bash
python src/main.py --record
python src/main.py --record --record-name "standup" --output-dir ./recordings
```

Records audio from a selected input device and saves it as a WAV file. The recording is standalone — it does not auto-start transcription or summarization. To summarize a recording afterwards, pass the file to the summarizer:

```bash
python src/main.py output/recordings/standup.wav
```

#### System audio capture setup

To record audio from meeting apps (Teams, Zoom, Google Meet), you need to capture system audio. This requires platform-specific setup:

**macOS** — Install [BlackHole](https://github.com/ExistentialAudio/BlackHole) virtual audio driver:
```bash
brew install blackhole-2ch
```
Then open **Audio MIDI Setup**, click `+` to create a **Multi-Output Device** that includes both your speakers and BlackHole. Set this as your system output. When recording, select the BlackHole device. For more info, take a look here: https://github.com/ExistentialAudio/BlackHole/wiki/Multi-Output-Device

**Windows** — Enable **Stereo Mix** in Sound settings (disabled by default on most systems), or install [VB-Cable](https://vb-audio.com/Cable/) virtual audio device. Select it as the input device when recording.

**Linux** — PulseAudio monitor sources are usually available by default. Select the monitor source for your output device (e.g. `Monitor of Built-in Audio`).

### Options

```
AUDIO_FILE             Path to audio file (optional)
--summarize PATH       Summarize a text file or directory (md, pdf, docx, txt, ...)
--per-file             When summarizing a directory, produce one summary per file
--podcast PATH         Generate a podcast from a file or directory of text
--record               Record audio from an input device
--record-name          Optional name for the recording file
--hint                 Short label for audio type (e.g., 'team meeting', 'lecture') — guides tone
--kb                   Directory of reference docs for domain-aware summaries (RAG)
--kb-rebuild           Force re-index the knowledge base
--embedding-model      Fastembed model for KB embeddings (default: BAAI/bge-small-en-v1.5)
--model                Whisper model size (default: medium)
--output-dir           Output directory (default: output/<name>/)
--llm-model            LLM model — Ollama, OpenAI (gpt-*), Anthropic (claude-*). Default from config.yaml
--input-language       Audio language for Whisper transcription (e.g. auto, en, nl, de, fr, ja, zh). Default: auto
--output-language      Output language for summaries and podcasts (e.g. en, nl, de, fr). Default from config or en
--chunk-minutes        Chunk size in minutes (default: 10)
```

### GPU acceleration on macOS (Apple Silicon)

On Apple Silicon Macs, the tool automatically uses `mlx-whisper` for GPU-accelerated transcription via Apple's MLX framework. This is significantly faster than the CPU-based `faster-whisper` backend.

- **Automatic**: If `mlx-whisper` is installed and you're on macOS, it's used by default
- **Override**: Set `WHISPER_BACKEND=faster-whisper` to force CPU, or `WHISPER_BACKEND=mlx` to force MLX
- Models are downloaded automatically from HuggingFace on first use

| CLI Model | MLX HuggingFace Repo |
|---|---|
| `tiny` | `mlx-community/whisper-tiny` |
| `base` | `mlx-community/whisper-base` |
| `small` | `mlx-community/whisper-small` |
| `medium` | `mlx-community/whisper-medium` |
| `large-v2` | `mlx-community/whisper-large-v2` |
| `large-v3` | `mlx-community/whisper-large-v3-turbo` |

### Knowledge base (RAG)

Add a `--kb` flag pointing to a directory of reference documents to make summaries and podcasts more domain-aware:

```bash
python src/main.py meeting.mp3 --kb ./my_docs/
python src/main.py --podcast --kb ./my_docs/
```

KB content is used as background reference during summarization — it helps the LLM understand domain-specific terms, acronyms, and context, but doesn't steer or add to the output. Only chunks that score above a relevance threshold are included.

Supported formats: `.txt`, `.md`, `.pdf`, `.docx`, `.html`, `.csv`

On first run, documents are chunked, embedded, and stored in a local Qdrant vector store (`data/kb_store/`). Subsequent runs reuse the cached index. Use `--kb-rebuild` to re-index when files change:

```bash
python src/main.py meeting.mp3 --kb ./my_docs/ --kb-rebuild
```

#### Embedding model

By default the KB uses `BAAI/bge-small-en-v1.5` (~130 MB, 384 dimensions). For better retrieval quality, use a larger model:

```bash
python src/main.py meeting.mp3 --kb ./my_docs/ --embedding-model BAAI/bge-base-en-v1.5
python src/main.py meeting.mp3 --kb ./my_docs/ --embedding-model BAAI/bge-large-en-v1.5
```

Popular fastembed models (downloaded automatically on first use):

| Model | Size | Dimensions |
|---|---|---|
| `BAAI/bge-small-en-v1.5` (default) | ~130 MB | 384 |
| `BAAI/bge-base-en-v1.5` | ~440 MB | 768 |
| `BAAI/bge-large-en-v1.5` | ~1.2 GB | 1024 |
| `sentence-transformers/all-MiniLM-L6-v2` | ~90 MB | 384 |
| `nomic-ai/nomic-embed-text-v1.5` | ~560 MB | 768 |

Changing the embedding model requires re-indexing. The tool will detect the mismatch and ask you to add `--kb-rebuild`.

## Configuration

### `config.yaml` — LLM settings

Every time you run the tool, it asks which AI model you want to use. Your choice is saved to `config.yaml` so it becomes the default next time.

**First run:** The tool will walk you through picking a model:
1. Choose between **local** (runs on your machine via Ollama) or **cloud** (OpenAI, Anthropic, etc.)
2. For local models, it checks your hardware (RAM, GPU) and suggests the best model for your machine
3. For cloud models, it asks for your API key once — then remembers it

**Subsequent runs:** The tool uses your last chosen model as the default. Just hit Enter to keep it, or pick a different one.

Your output language preference is also saved here (as `output_language`). Set it once with `--output-language nl` and it becomes the default for future runs.

**Skip the prompt entirely** by passing a model on the command line:

```bash
python src/main.py meeting.mp3 --llm-model gpt-4o
python src/main.py meeting.mp3 --llm-model claude-sonnet-4-6
python src/main.py meeting.mp3 --llm-model mistral:7b
```

### `podcast_config.yaml`

Controls podcast generation. Edit to change voice, style, or sources:

```yaml
tts:
  engine: piper                     # piper | macos_say
  voice: en_US-lessac-medium        # Piper model name (or macOS voice name)
  voice_host2: en_US-ryan-medium    # Second voice for two_host mode
  speed: 1.0

podcast:
  style: solo                       # solo | two_host
  max_articles: 5
  target_length: medium             # short (~3min) | medium (~7min) | long (~15min)

sources:
  feeds:
    - https://hnrss.org/newest?points=100
    - https://feeds.arstechnica.com/arstechnica/technology-lab
    - https://arxiv.org/rss/cs.AI
  web_search: true
```

### Piper TTS voice setup

When using Piper, you need to download voice model files (`.onnx` + `.onnx.json`) and place them in the `voices/` directory. Browse available voices at https://github.com/rhasspy/piper/blob/master/VOICES.md.

```bash
mkdir -p voices && cd voices
python3 -m piper.download_voices en_US-ryan-medium
python3 -m piper.download_voices en_US-lessac-medium
```

The `voice` value in `podcast_config.yaml` must match the filename without `.onnx` (e.g. `en_US-lessac-medium`).

For macOS without extra setup, use `engine: macos_say` with a system voice name like `Daniel`.

## Running tests

```bash
pip install pytest
python -m pytest tests/ -v
```