import csv
import json
import os

from qdrant_client import QdrantClient

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".html", ".htm", ".csv"}
CHUNK_TARGET_WORDS = 500
COLLECTION_NAME = "knowledge_base"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
MODEL_META_FILE = "embedding_model.json"


class KnowledgeBase:
    """RAG knowledge base backed by a persistent local Qdrant vector store."""

    def __init__(self, store_path: str = "data/kb_store",
                 embedding_model: str = DEFAULT_EMBEDDING_MODEL):
        os.makedirs(store_path, exist_ok=True)
        self.client = QdrantClient(path=store_path)
        self.client.set_model(embedding_model)
        self.collection = COLLECTION_NAME
        self.embedding_model = embedding_model
        self.store_path = store_path
        self._chunk_count = 0

    def is_indexed(self) -> bool:
        """Check if KB collection already exists from a previous run."""
        return self.client.collection_exists(self.collection)

    def get_indexed_model(self) -> str | None:
        """Return the embedding model used for the existing index, or None."""
        meta_path = os.path.join(self.store_path, MODEL_META_FILE)
        if os.path.isfile(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                data = json.load(f)
            return data.get("embedding_model")
        return None

    def check_model_mismatch(self) -> str | None:
        """If the index was built with a different model, return that model name."""
        if not self.is_indexed():
            return None
        indexed_model = self.get_indexed_model()
        if indexed_model and indexed_model != self.embedding_model:
            return indexed_model
        return None

    def _save_model_meta(self) -> None:
        """Save the embedding model name alongside the store."""
        meta_path = os.path.join(self.store_path, MODEL_META_FILE)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"embedding_model": self.embedding_model}, f)

    def delete_collection(self) -> None:
        """Delete existing collection (for --kb-rebuild)."""
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)

    def index_directory(self, directory: str, progress=None, task_id=None) -> int:
        """Scan directory, extract text, chunk, embed and store in Qdrant.

        Returns the number of files processed.
        """
        files = _find_supported_files(directory)
        if progress and task_id is not None:
            progress.update(task_id, total=len(files))

        all_documents = []
        all_metadata = []
        files_processed = 0

        for filepath in files:
            text = _extract_text(filepath)
            if text and text.strip():
                filename = os.path.basename(filepath)
                chunks = _chunk_text(text, filename)
                for chunk in chunks:
                    all_documents.append(chunk["text"])
                    all_metadata.append({
                        "source": chunk["source"],
                        "chunk_index": chunk["chunk_index"],
                    })
                files_processed += 1
            if progress and task_id is not None:
                progress.advance(task_id)

        if all_documents:
            self.client.add(
                collection_name=self.collection,
                documents=all_documents,
                metadata=all_metadata,
            )
            self._save_model_meta()

        self._chunk_count = len(all_documents)
        return files_processed

    @property
    def chunk_count(self) -> int:
        """Number of chunks in the KB."""
        if self._chunk_count > 0:
            return self._chunk_count
        if self.is_indexed():
            info = self.client.get_collection(self.collection)
            return info.points_count or 0
        return 0

    def retrieve(self, query: str, top_k: int = 5, max_chars: int = 4500) -> str:
        """Query Qdrant for the most relevant KB chunks.

        Returns a formatted context string, truncated to max_chars.
        Returns empty string if KB is empty or no results found.
        """
        if not self.is_indexed():
            return ""

        results = self.client.query(
            collection_name=self.collection,
            query_text=query,
            limit=top_k,
        )

        if not results:
            return ""

        parts = []
        total_chars = 0
        for point in results:
            source = point.metadata.get("source", "unknown")
            text = point.document
            part = f"[From: {source}]\n{text}"

            if total_chars + len(part) > max_chars:
                remaining = max_chars - total_chars
                if remaining > 100:
                    parts.append(part[:remaining] + "...")
                break

            parts.append(part)
            total_chars += len(part)

        return "\n\n".join(parts)

    def close(self) -> None:
        """Close the Qdrant client."""
        self.client.close()


# --- File discovery ---


def _find_supported_files(directory: str) -> list[str]:
    """Recursively find all supported files in directory."""
    found = []
    for root, _dirs, files in os.walk(directory):
        for fname in sorted(files):
            if fname.startswith("."):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                found.append(os.path.join(root, fname))
    return found


# --- Text extraction (lazy imports for optional deps) ---


def _extract_text(filepath: str) -> str | None:
    """Extract plain text from a file based on its extension."""
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext in (".txt", ".md"):
            return _extract_plaintext(filepath)
        elif ext == ".pdf":
            return _extract_pdf(filepath)
        elif ext == ".docx":
            return _extract_docx(filepath)
        elif ext in (".html", ".htm"):
            return _extract_html(filepath)
        elif ext == ".csv":
            return _extract_csv(filepath)
    except Exception:
        return None
    return None


def _extract_plaintext(filepath: str) -> str:
    with open(filepath, encoding="utf-8", errors="replace") as f:
        return f.read()


def _extract_pdf(filepath: str) -> str:
    import pymupdf

    text_parts = []
    with pymupdf.open(filepath) as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts)


def _extract_docx(filepath: str) -> str:
    from docx import Document

    doc = Document(filepath)
    return "\n\n".join(para.text for para in doc.paragraphs if para.text.strip())


def _extract_html(filepath: str) -> str:
    import trafilatura

    with open(filepath, encoding="utf-8", errors="replace") as f:
        html_content = f.read()
    result = trafilatura.extract(html_content)
    return result or ""


def _extract_csv(filepath: str) -> str:
    with open(filepath, encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return ""
    header = rows[0]
    lines = []
    for row in rows[1:]:
        pairs = [f"{h}: {v}" for h, v in zip(header, row) if v.strip()]
        lines.append("; ".join(pairs))
    return "\n".join(lines)


# --- Text chunking ---


def _chunk_text(text: str, source: str) -> list[dict]:
    """Split text into ~500-word chunks on paragraph boundaries."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if not paragraphs:
        return []

    chunks = []
    current_parts = []
    current_words = 0
    chunk_idx = 0

    for para in paragraphs:
        para_words = len(para.split())
        if current_words + para_words > CHUNK_TARGET_WORDS and current_parts:
            chunks.append({
                "text": "\n\n".join(current_parts),
                "source": source,
                "chunk_index": chunk_idx,
            })
            chunk_idx += 1
            current_parts = []
            current_words = 0

        current_parts.append(para)
        current_words += para_words

    if current_parts:
        chunks.append({
            "text": "\n\n".join(current_parts),
            "source": source,
            "chunk_index": chunk_idx,
        })

    return chunks
