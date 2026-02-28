"""Tests for podcast.loader module."""
import os

import pytest

from podcast.loader import load_input_text


class TestLoadInputTextFile:
    def test_single_text_file(self, tmp_path):
        f = tmp_path / "notes.txt"
        f.write_text("Hello, this is a test document.")
        text, files = load_input_text(str(f))
        assert "Hello, this is a test document." in text
        assert files == [str(f)]

    def test_single_md_file(self, tmp_path):
        f = tmp_path / "notes.md"
        f.write_text("# Title\n\nSome content here.")
        text, files = load_input_text(str(f))
        assert "Some content here." in text
        assert len(files) == 1

    def test_empty_file_raises(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        with pytest.raises(ValueError, match="Could not extract text"):
            load_input_text(str(f))

    def test_whitespace_only_file_raises(self, tmp_path):
        f = tmp_path / "blank.txt"
        f.write_text("   \n\n  ")
        with pytest.raises(ValueError, match="Could not extract text"):
            load_input_text(str(f))


class TestLoadInputTextDirectory:
    def test_directory_with_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("First document.")
        (tmp_path / "b.md").write_text("Second document.")
        text, files = load_input_text(str(tmp_path))
        assert "First document." in text
        assert "Second document." in text
        assert len(files) == 2

    def test_directory_with_separator(self, tmp_path):
        (tmp_path / "a.txt").write_text("Content A")
        (tmp_path / "b.txt").write_text("Content B")
        text, _ = load_input_text(str(tmp_path))
        assert "--- a.txt ---" in text
        assert "--- b.txt ---" in text

    def test_empty_directory_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No supported files found"):
            load_input_text(str(tmp_path))

    def test_directory_with_unsupported_files_only(self, tmp_path):
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        (tmp_path / "data.bin").write_bytes(b"\x00\x01")
        with pytest.raises(ValueError, match="No supported files found"):
            load_input_text(str(tmp_path))

    def test_skips_empty_files(self, tmp_path):
        (tmp_path / "good.txt").write_text("Real content.")
        (tmp_path / "empty.txt").write_text("")
        text, files = load_input_text(str(tmp_path))
        assert "Real content." in text
        assert len(files) == 1


class TestLoadInputTextInvalidPath:
    def test_nonexistent_path_raises(self):
        with pytest.raises(ValueError, match="Path does not exist"):
            load_input_text("/nonexistent/path/to/file.txt")
