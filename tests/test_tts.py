"""Tests for podcast.tts module — path handling and script parsing."""
from pathlib import Path

from podcast.tts import _parse_two_host_script


class TestAiffPathHandling:
    """Verify Path.with_suffix produces correct aiff paths."""

    def test_simple_wav(self):
        result = str(Path("/tmp/output.wav").with_suffix(".aiff"))
        assert result == "/tmp/output.aiff"

    def test_wav_in_directory_name(self):
        # If directory contains ".wav", with_suffix only changes the file suffix
        result = str(Path("/tmp/my.wav.files/output.wav").with_suffix(".aiff"))
        assert result == "/tmp/my.wav.files/output.aiff"

    def test_no_extension(self):
        result = str(Path("/tmp/output").with_suffix(".aiff"))
        assert result == "/tmp/output.aiff"


class TestParseTwoHostScript:
    def test_basic_parsing(self):
        script = "ALEX: Hello everyone.\nSAM: Welcome to the show."
        segments = _parse_two_host_script(script)
        assert len(segments) == 2
        assert segments[0] == ("ALEX", "Hello everyone.")
        assert segments[1] == ("SAM", "Welcome to the show.")

    def test_multiline_speaker(self):
        script = "ALEX: First line.\nSecond line.\nSAM: Response."
        segments = _parse_two_host_script(script)
        assert len(segments) == 2
        assert segments[0] == ("ALEX", "First line. Second line.")
        assert segments[1] == ("SAM", "Response.")

    def test_empty_script(self):
        assert _parse_two_host_script("") == []

    def test_no_speakers(self):
        assert _parse_two_host_script("Just some text without speakers.") == []
