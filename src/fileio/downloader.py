import os
import re

import yt_dlp


def sanitize_filename(name: str) -> str:
    """Remove characters that are unsafe for filenames."""
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = name.strip().replace(" ", "_")
    return name[:100]  # limit length


def download_youtube_audio(url: str, data_dir: str = "data") -> tuple[str, str]:
    """Download audio from a YouTube URL as mp3.

    Returns (audio_file_path, video_title).
    """
    os.makedirs(data_dir, exist_ok=True)

    # First pass: extract info to get the title
    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
        info = ydl.extract_info(url, download=False)
        title = info.get("title", "video")

    safe_title = sanitize_filename(title)
    output_path = os.path.join(data_dir, safe_title)

    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "outtmpl": output_path,
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    audio_file = output_path + ".mp3"
    return audio_file, title
