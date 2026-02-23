# Meeting Summarizer

A fully local CLI tool that transcribes meeting audio and generates a structured summary with actionable remarks. Runs entirely on your machine — no cloud APIs, no data leaves your device.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) running locally
- An Ollama model pulled (default: `llama3.1:8b`)

## Setup

```bash
pip install -r requirements.txt
ollama pull llama3.1:8b
```

## Usage

```bash
python src/main.py meeting.mp3
```

This produces three files in the current directory:

| File | Contents |
|---|---|
| `transcript.md` | Full transcript with timestamps (original language) |
| `summary.md` | Structured meeting summary (English) |
| `remarks.md` | Remarks and actionable suggestions (English) |

### Options

```
--model, -m        Whisper model size: tiny, base, small, medium (default), large-v2, large-v3
--output-dir, -o   Output directory (default: current directory)
--llm-model        Ollama model name (default: llama3.1:8b)
--language, -l     Audio language: auto (default), nl (Dutch), en (English)
--chunk-minutes    Transcript chunk size in minutes for LLM processing (default: 10)
```

### Examples

```bash
# Dutch meeting, output to a folder
python src/main.py vergadering.mp3 --language nl --output-dir ./output

# Use a larger Whisper model for better accuracy
python src/main.py meeting.wav --model large-v2

# Use a smaller Whisper model for faster processing
python src/main.py long_meeting.mp3 --model small
```

## How it works

1. **Transcribe** — faster-whisper converts audio to text locally (with timestamps)
2. **Chunk** — the transcript is split into ~10-minute segments
3. **Summarize** — each chunk is summarized independently via Ollama, then consolidated into one summary
4. **Remarks** — the summary is analyzed for risks, priorities, and suggestions
5. **Write** — three markdown files are written to disk

Supports Dutch and English audio. Output is always in English.
