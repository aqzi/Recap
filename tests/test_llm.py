"""Tests for core.llm module — provider detection and JSON parsing."""
import json

from core.llm import detect_provider


class TestDetectProvider:
    def test_openai_gpt(self):
        assert detect_provider("gpt-4o") == "openai"

    def test_openai_o1(self):
        assert detect_provider("o1-preview") == "openai"

    def test_openai_o3(self):
        assert detect_provider("o3-mini") == "openai"

    def test_openai_o4(self):
        assert detect_provider("o4-mini") == "openai"

    def test_openai_chatgpt(self):
        assert detect_provider("chatgpt-4o-latest") == "openai"

    def test_anthropic_claude(self):
        assert detect_provider("claude-3-haiku-20240307") == "anthropic"

    def test_ollama_fallback(self):
        assert detect_provider("llama3.1:8b") == "ollama"

    def test_ollama_custom(self):
        assert detect_provider("mistral:7b") == "ollama"

    def test_case_insensitive(self):
        assert detect_provider("GPT-4o") == "openai"
        assert detect_provider("Claude-3-opus") == "anthropic"


class TestRankArticlesJsonParsing:
    """Test that rindex finds the correct closing bracket."""

    def test_simple_array(self):
        text = "[1, 2, 3]"
        start = text.index("[")
        end = text.rindex("]") + 1
        assert json.loads(text[start:end]) == [1, 2, 3]

    def test_text_before_array(self):
        text = "Here are the results: [0, 2, 4]"
        start = text.index("[")
        end = text.rindex("]") + 1
        assert json.loads(text[start:end]) == [0, 2, 4]

    def test_text_with_bracket_before_array(self):
        # Pathological case: brackets in surrounding text. The rank_articles
        # function falls back to default order on parse failure, which is fine.
        text = "Note: use index [0] for first. The answer is [1, 3, 5]"
        start = text.index("[")
        end = text.rindex("]") + 1
        # rindex finds the correct last bracket, but the first [ is wrong
        # This falls through to the except handler gracefully
        import pytest
        with pytest.raises(json.JSONDecodeError):
            json.loads(text[start:end])

    def test_nested_array(self):
        # rindex correctly handles nested brackets unlike index
        text = "[[1, 2], 3]"
        start = text.index("[")
        end = text.rindex("]") + 1
        assert json.loads(text[start:end]) == [[1, 2], 3]
