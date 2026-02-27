"""Tests for core.knowledge_base module — text chunking and extraction helpers."""

from core.knowledge_base import _chunk_text


class TestChunkText:
    def test_short_text_single_chunk(self):
        text = "Short paragraph."
        chunks = _chunk_text(text, "test.txt")
        assert len(chunks) == 1
        assert chunks[0]["text"] == "Short paragraph."
        assert chunks[0]["source"] == "test.txt"
        assert chunks[0]["chunk_index"] == 0

    def test_empty_text(self):
        assert _chunk_text("", "test.txt") == []

    def test_multiple_chunks(self):
        # Create text that exceeds CHUNK_TARGET_WORDS (500)
        para1 = " ".join(["word"] * 300)
        para2 = " ".join(["word"] * 300)
        text = f"{para1}\n\n{para2}"
        chunks = _chunk_text(text, "test.txt")
        assert len(chunks) == 2
        assert chunks[0]["chunk_index"] == 0
        assert chunks[1]["chunk_index"] == 1

    def test_chunk_source_preserved(self):
        chunks = _chunk_text("Hello world", "readme.md")
        assert all(c["source"] == "readme.md" for c in chunks)
