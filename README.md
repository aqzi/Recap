# Meeting Summarizer

A fully local CLI tool that transcribes meeting audio (or YouTube videos) and generates a structured summary with actionable remarks. Runs entirely on your machine — no cloud APIs, no data leaves your device.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) running locally
- An Ollama model pulled (default: `llama3.1:8b`)
- ffmpeg (for YouTube audio extraction)

## Setup

```bash
pip install -r requirements.txt
ollama pull llama3.1:8b
```

## Usage

### From a local audio file

```bash
python src/main.py meeting.mp3
```

Output goes to `output/meeting/` by default:

```
output/meeting/
├── transcript.md    # Full transcript with timestamps
├── summary.md       # Structured meeting summary (English)
└── remarks.md       # Remarks and actionable suggestions (English)
```

### From a YouTube URL

```bash
python src/main.py --youtube "https://www.youtube.com/watch?v=abc123"
```

Audio is downloaded to `data/`, output goes to `output/<video_title>/`. The `remarks.md` file includes a **Watch Recommendation Score** (1-10) at the top indicating whether you should watch the video yourself.

### Personalized scoring with `interest.md`

Create an `interest.md` file in the project root to personalize YouTube watch scores based on your interests:

```markdown
I'm interested in AI/ML engineering, startup strategy, and Python tooling.
I don't care about marketing or social media growth.
```

The score will factor in how relevant the video is to your interests. If no `interest.md` file exists, scoring works normally without personalization.

### Options

```
AUDIO_FILE             Path to audio file (optional if --youtube is used)
--youtube, -yt         YouTube URL to download and process
--model, -m            Whisper model size: tiny, base, small, medium (default), large-v2, large-v3
--output-dir, -o       Output directory (default: output/<name>/)
--llm-model            Ollama model name (default: llama3.1:8b)
--language, -l         Audio language: auto (default), nl (Dutch), en (English)
--chunk-minutes        Transcript chunk size in minutes for LLM processing (default: 10)
```

### Examples

```bash
# Dutch meeting
python src/main.py vergadering.mp3 --language nl

# Custom output directory
python src/main.py meeting.wav --output-dir ./my_notes

# YouTube video with large Whisper model
python src/main.py --youtube "https://youtu.be/abc123" --model large-v2
```

## How it works

1. **Download** (YouTube only) — yt-dlp extracts audio to `data/`
2. **Transcribe** — faster-whisper converts audio to text locally
3. **Chunk** — transcript is split into ~10-minute segments
4. **Summarize** — each chunk is summarized via Ollama, then consolidated
5. **Score** (YouTube only) — LLM rates if the video is worth watching yourself
6. **Remarks** — summary is analyzed for risks, priorities, and suggestions
7. **Write** — markdown files are written to the output directory

Supports Dutch and English audio. Output is always in English.
