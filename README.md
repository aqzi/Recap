# Audio Summarizer & Podcast Generator

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

### Options

```
AUDIO_FILE             Path to audio file (optional)
--youtube, -yt         YouTube URL to download and process
--podcast              Generate a podcast from your interests
--model, -m            Whisper model size (default: medium)
--output-dir, -o       Output directory (default: output/<name>/)
--llm-model            Ollama model (default: llama3.1:8b)
--language, -l         Audio language: auto, nl, en (default: auto)
--chunk-minutes        Chunk size in minutes (default: 10)
```

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

When using Piper, you need to download voice model files (`.onnx` + `.onnx.json`) and place them in the project root. Browse available voices at https://github.com/rhasspy/piper/blob/master/VOICES.md.

```bash
# Example: download the default voice
python3 -m piper.download_voices en_US-ryan-medium
python3 -m piper.download_voices en_US-lessac-medium
```

The `voice` value in `podcast_config.yaml` must match the filename without `.onnx` (e.g. `en_US-lessac-medium`).

For macOS without extra setup, use `engine: macos_say` with a system voice name like `Samantha` or `Daniel`.

Supports Dutch and English audio. Output is always in English.
