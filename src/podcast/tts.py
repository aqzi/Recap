import os
import re
import subprocess
import tempfile
from abc import ABC, abstractmethod


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
        # Strip speaker labels and synthesize as one voice
        clean = re.sub(r"^(ALEX|SAM):\s*", "", script, flags=re.MULTILINE)
        self.synthesize(clean, output_path)


def _resolve_piper_model(voice: str) -> str:
    """Resolve a Piper voice name to a full .onnx model path.

    Looks for <voice>.onnx in the project root directory.
    If voice is already an absolute path or ends with .onnx, use as-is.
    """
    if os.path.isabs(voice) or voice.endswith(".onnx"):
        return voice

    # Look in project root (cwd)
    candidate = os.path.join(os.getcwd(), f"{voice}.onnx")
    if os.path.isfile(candidate):
        return candidate

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
                subprocess.run(cmd, stdin=stdin, check=True, capture_output=True)
        finally:
            os.unlink(text_path)

    def synthesize_two_host(self, script: str, output_path: str, voice2: str | None = None) -> None:
        if not voice2:
            super().synthesize_two_host(script, output_path)
            return

        voice2 = _resolve_piper_model(voice2)

        # Parse script into segments
        segments = _parse_two_host_script(script)
        if not segments:
            self.synthesize(script, output_path)
            return

        # Synthesize each segment with the appropriate voice
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
                        subprocess.run(cmd, stdin=stdin, check=True, capture_output=True)
                finally:
                    os.unlink(text_path)

                segment_audio = AudioSegment.from_wav(temp_path)
                combined += segment_audio
                # Small pause between speakers
                combined += AudioSegment.silent(duration=300)

            combined.export(output_path, format="wav")
        finally:
            for f in temp_files:
                if os.path.exists(f):
                    os.unlink(f)


class MacOSSay(TTSEngine):
    """macOS built-in say command — zero setup fallback."""

    def __init__(self, voice: str = "Samantha", speed: float = 1.0):
        self.voice = voice
        # macOS say uses words per minute, default ~175
        self.rate = int(175 * speed)

    def synthesize(self, text: str, output_path: str) -> None:
        # macOS say outputs AIFF, we'll convert to WAV
        aiff_path = output_path.replace(".wav", ".aiff")

        # Write text to temp file to avoid shell argument length limits
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(text)
            text_path = f.name

        try:
            subprocess.run(
                ["say", "-v", self.voice, "-r", str(self.rate), "-o", aiff_path, "-f", text_path],
                check=True,
                capture_output=True,
            )
        finally:
            os.unlink(text_path)

        # Convert to WAV using ffmpeg
        subprocess.run(
            ["ffmpeg", "-y", "-i", aiff_path, output_path],
            check=True,
            capture_output=True,
        )
        if os.path.exists(aiff_path):
            os.unlink(aiff_path)

    def synthesize_two_host(self, script: str, output_path: str, voice2: str | None = None) -> None:
        if not voice2:
            voice2 = "Daniel"  # Default second macOS voice

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
                        check=True,
                        capture_output=True,
                    )
                finally:
                    os.unlink(seg_text_path)
                subprocess.run(
                    ["ffmpeg", "-y", "-i", aiff_path, temp_path],
                    check=True,
                    capture_output=True,
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
        return MacOSSay(voice=voice, speed=speed)
    else:
        return PiperTTS(voice=voice, speed=speed)
