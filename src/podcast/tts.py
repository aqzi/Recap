import os
import re
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

# Project root: src/podcast/tts.py -> parents[2] = project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_VOICES_DIR = _PROJECT_ROOT / "voices"

SUBPROCESS_TIMEOUT = 300  # 5 minutes


class TTSEngine(ABC):
    """Base interface for TTS engines."""

    @abstractmethod
    def synthesize(self, text: str, output_path: str) -> None:
        """Convert text to audio and save to output_path."""

    def synthesize_two_host(self, script: str, output_path: str, voice2: str | None = None) -> None:
        """Synthesize a two-host script with different voices.

        Default implementation: just synthesize as single voice.
        Subclasses can override for real multi-voice support.
        """
        clean = re.sub(r"^(ALEX|SAM):\s*", "", script, flags=re.MULTILINE)
        self.synthesize(clean, output_path)


def _resolve_piper_model(voice: str) -> str:
    """Resolve a Piper voice name to a full .onnx model path.

    Looks for <voice>.onnx in the voices/ directory.
    If voice is already an absolute path or ends with .onnx, use as-is.
    """
    if os.path.isabs(voice) or voice.endswith(".onnx"):
        return voice

    candidate = _VOICES_DIR / f"{voice}.onnx"
    if candidate.is_file():
        return str(candidate)

    # Fallback: return as-is and let piper handle the error
    return voice


class PiperTTS(TTSEngine):
    """Piper TTS — local, fast, good quality."""

    def __init__(self, voice: str = "en_US-lessac-medium", speed: float = 1.0):
        self.voice = _resolve_piper_model(voice)
        self.speed = speed

    def synthesize(self, text: str, output_path: str) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(text)
            text_path = f.name

        try:
            cmd = [
                "piper",
                "--model", self.voice,
                "--output_file", output_path,
                "--length-scale", str(1.0 / self.speed),
            ]
            with open(text_path) as stdin:
                subprocess.run(cmd, stdin=stdin, check=True, capture_output=True,
                               timeout=SUBPROCESS_TIMEOUT)
        finally:
            os.unlink(text_path)

    def synthesize_two_host(self, script: str, output_path: str, voice2: str | None = None) -> None:
        if not voice2:
            super().synthesize_two_host(script, output_path)
            return

        voice2 = _resolve_piper_model(voice2)

        segments = _parse_two_host_script(script)
        if not segments:
            self.synthesize(script, output_path)
            return

        from pydub import AudioSegment
        combined = AudioSegment.empty()
        temp_files = []

        try:
            for i, (speaker, text) in enumerate(segments):
                temp_path = os.path.join(tempfile.gettempdir(), f"segment_{i}.wav")
                temp_files.append(temp_path)
                voice = self.voice if speaker == "ALEX" else voice2

                with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                    f.write(text)
                    text_path = f.name

                try:
                    cmd = [
                        "piper",
                        "--model", voice,
                        "--output_file", temp_path,
                        "--length-scale", str(1.0 / self.speed),
                    ]
                    with open(text_path) as stdin:
                        subprocess.run(cmd, stdin=stdin, check=True, capture_output=True,
                                       timeout=SUBPROCESS_TIMEOUT)
                finally:
                    os.unlink(text_path)

                segment_audio = AudioSegment.from_wav(temp_path)
                combined += segment_audio
                combined += AudioSegment.silent(duration=300)

            combined.export(output_path, format="wav")
        finally:
            for f in temp_files:
                if os.path.exists(f):
                    os.unlink(f)


class MacOSSay(TTSEngine):
    """macOS built-in say command — zero setup fallback."""

    def __init__(self, voice: str = "Daniel", speed: float = 1.0):
        self.voice = voice
        # macOS say uses words per minute, default ~175
        self.rate = int(175 * speed)

    def synthesize(self, text: str, output_path: str) -> None:
        aiff_path = output_path.replace(".wav", ".aiff")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(text)
            text_path = f.name

        try:
            subprocess.run(
                ["say", "-v", self.voice, "-r", str(self.rate), "-o", aiff_path, "-f", text_path],
                check=True, capture_output=True, timeout=SUBPROCESS_TIMEOUT,
            )
        finally:
            os.unlink(text_path)

        subprocess.run(
            ["ffmpeg", "-y", "-i", aiff_path, output_path],
            check=True, capture_output=True, timeout=SUBPROCESS_TIMEOUT,
        )
        if os.path.exists(aiff_path):
            os.unlink(aiff_path)

    def synthesize_two_host(self, script: str, output_path: str, voice2: str | None = None) -> None:
        if not voice2:
            voice2 = "Daniel"

        segments = _parse_two_host_script(script)
        if not segments:
            self.synthesize(script, output_path)
            return

        from pydub import AudioSegment
        combined = AudioSegment.empty()
        temp_files = []

        try:
            for i, (speaker, text) in enumerate(segments):
                temp_path = os.path.join(tempfile.gettempdir(), f"segment_{i}.wav")
                temp_files.append(temp_path)
                voice = self.voice if speaker == "ALEX" else voice2

                aiff_path = temp_path.replace(".wav", ".aiff")
                temp_files.append(aiff_path)

                with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tf:
                    tf.write(text)
                    seg_text_path = tf.name

                try:
                    subprocess.run(
                        ["say", "-v", voice, "-r", str(self.rate), "-o", aiff_path, "-f", seg_text_path],
                        check=True, capture_output=True, timeout=SUBPROCESS_TIMEOUT,
                    )
                finally:
                    os.unlink(seg_text_path)
                subprocess.run(
                    ["ffmpeg", "-y", "-i", aiff_path, temp_path],
                    check=True, capture_output=True, timeout=SUBPROCESS_TIMEOUT,
                )

                segment_audio = AudioSegment.from_wav(temp_path)
                combined += segment_audio
                combined += AudioSegment.silent(duration=300)

            combined.export(output_path, format="wav")
        finally:
            for f in temp_files:
                if os.path.exists(f):
                    os.unlink(f)


def _parse_two_host_script(script: str) -> list[tuple[str, str]]:
    """Parse a two-host script into (speaker, text) segments."""
    segments = []
    current_speaker = None
    current_lines = []

    for line in script.splitlines():
        line = line.strip()
        if not line:
            continue

        match = re.match(r"^(ALEX|SAM):\s*(.*)", line)
        if match:
            if current_speaker and current_lines:
                segments.append((current_speaker, " ".join(current_lines)))
            current_speaker = match.group(1)
            current_lines = [match.group(2)] if match.group(2) else []
        elif current_speaker:
            current_lines.append(line)

    if current_speaker and current_lines:
        segments.append((current_speaker, " ".join(current_lines)))

    return segments


def get_tts_engine(config: dict) -> TTSEngine:
    """Factory: returns the right TTS engine based on config."""
    tts_cfg = config.get("tts", {})
    engine_name = tts_cfg.get("engine", "piper")
    voice = tts_cfg.get("voice", "en_US-lessac-medium")
    speed = tts_cfg.get("speed", 1.0)

    if engine_name == "macos_say":
        if sys.platform != "darwin":
            from fileio.progress import console
            console.print("[bold red]Error:[/bold red] macos_say TTS engine is only available on macOS.")
            console.print("Change tts.engine to 'piper' in podcast_config.yaml.")
            sys.exit(1)
        return MacOSSay(voice=voice, speed=speed)
    else:
        return PiperTTS(voice=voice, speed=speed)
