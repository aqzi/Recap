# Recap

A fully local CLI tool that transcribes meetings, summarizes YouTube videos, or generates personalized tech podcasts. Runs entirely on your machine — no cloud APIs, no data leaves your device.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) running locally with a model pulled (default: `llama3.1:8b`)
- ffmpeg
- TTS engine for podcast audio: macOS `say` (built-in, zero setup) or [Piper TTS](https://github.com/rhasspy/piper)

## Setup

```bash
pip install -r requirements.txt
ollama pull llama3.1:8b
```

## Modes

### 1. Summarize a local audio file

```bash
python src/main.py meeting.mp3
```

Output: `output/meeting/transcript.md`, `summary.md`, `remarks.md`

### 2. Summarize a YouTube video

```bash
python src/main.py --youtube "https://www.youtube.com/watch?v=abc123"
```

Audio is downloaded to `data/`, output goes to `output/<video_title>/`. Includes a **Watch Recommendation Score** (1-10) in `remarks.md`.

### 3. Generate a podcast

```bash
python src/main.py --podcast
```

Fetches recent articles from RSS feeds and web search, filters by your interests, generates a podcast script, and converts it to audio.

Output: `output/podcast_<date>/podcast.wav`, `script.md`, `sources.md`

Requires `interest.md` — see below.

### 4. Record audio

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
--youtube, -yt         YouTube URL to download and process
--podcast              Generate a podcast from your interests
--record               Record audio from an input device
--record-name          Optional name for the recording file
--kb                   Knowledge base directory for context-aware summaries
--kb-rebuild           Force re-index the knowledge base
--embedding-model      Fastembed model for KB embeddings (default: BAAI/bge-small-en-v1.5)
--model, -m            Whisper model size (default: medium)
--output-dir, -o       Output directory (default: output/<name>/)
--llm-model            Ollama model (default: llama3.1:8b)
--language, -l         Audio language: auto, nl, en (default: auto)
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

### 4. Knowledge base (RAG)

Add a `--kb` flag pointing to a directory of reference documents to make summaries and podcasts more domain-aware:

```bash
python src/main.py meeting.mp3 --kb ./my_docs/
python src/main.py --youtube "URL" --kb ./my_docs/
python src/main.py --podcast --kb ./my_docs/
```

For meetings and YouTube videos, relevant KB content is injected into the summarization prompts. For podcasts, fetched articles are discussed in the context of your knowledge base.

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

### `interest.md`

Personalizes YouTube watch scores and podcast content. Create in the project root:

```markdown
I'm interested in AI/ML engineering, startup strategy, and Python tooling.
I don't care about marketing or social media growth.
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

Supports Dutch and English audio. Output is always in English.
