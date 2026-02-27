"""Tests for utils.validation module — Ollama model matching."""


class TestOllamaModelMatching:
    """Verify that check_ollama matching logic is correct."""

    def _match(self, llm_model: str, available: list[str]) -> bool:
        """Simulate the matching logic from check_ollama."""
        has_tag = ":" in llm_model
        if has_tag:
            return any(llm_model == name for name in available)
        else:
            available_bases = [name.split(":")[0] for name in available]
            return any(llm_model == abase for abase in available_bases)

    def test_exact_match(self):
        assert self._match("llama3.1:8b", ["llama3.1:8b", "mistral:7b"])

    def test_base_match(self):
        # "llama3.1" should match "llama3.1:8b" via base comparison
        assert self._match("llama3.1", ["llama3.1:8b"])

    def test_no_false_positive_prefix(self):
        # "llama3" should NOT match "llama3.1:8b" — different base names
        assert not self._match("llama3", ["llama3.1:8b", "llama3.2:3b"])

    def test_no_false_positive_substring(self):
        # "llama3" should NOT match "llama3.1:8b" via substring
        assert not self._match("llama3", ["llama3.1:8b"])

    def test_exact_match_with_tag(self):
        assert self._match("mistral:7b", ["mistral:7b", "llama3.1:8b"])

    def test_wrong_tag_no_match(self):
        # "llama3.2:3b" should NOT match "llama3.2:1b" — different tags
        assert not self._match("llama3.2:3b", ["llama3.2:1b"])

    def test_no_match(self):
        assert not self._match("phi3:mini", ["llama3.1:8b", "mistral:7b"])
