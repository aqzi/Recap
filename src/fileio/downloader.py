import os
import re

import yt_dlp


def sanitize_filename(name: str) -> str:
    """Remove characters that are unsafe for filenames."""
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = name.strip().replace(" ", "_")
    return name[:100]


def download_youtube_audio(url: str, data_dir: str = "data") -> tuple[str, str]:
    """Download audio from a YouTube URL as mp3.

    Returns (audio_file_path, video_title).
    """
    os.makedirs(data_dir, exist_ok=True)

    # Use a temp template, then rename after we know the title
    temp_template = os.path.join(data_dir, "%(title)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "outtmpl": temp_template,
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "video")

    safe_title = sanitize_filename(title)
    audio_file = os.path.join(data_dir, safe_title + ".mp3")

    # yt-dlp uses the raw title for the filename; rename to sanitized version
    raw_file = os.path.join(data_dir, title + ".mp3")
    if not os.path.isfile(audio_file) and os.path.isfile(raw_file):
        os.rename(raw_file, audio_file)

    return audio_file, title
