import re
import uuid
from pathlib import Path
from typing import List, Dict, Any


class DocumentProcessor:
    """
    Handles loading and chunking of financial documents.
    Supports PDF and plain text. Chunking uses a token-aware
    sliding window with configurable overlap.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process(self, file_path: str, source_name: str = None) -> List[Dict[str, Any]]:
        path = Path(file_path)
        source = source_name or path.name

        if path.suffix.lower() == ".pdf":
            text = self._load_pdf(file_path)
        elif path.suffix.lower() == ".txt":
            text = self._load_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        text = self._clean_text(text)
        chunks = self._chunk_text(text)

        return [
            {
                "id": str(uuid.uuid4()),
                "content": chunk,
                "source": source,
                "chunk_id": i,
                "total_chunks": len(chunks),
            }
            for i, chunk in enumerate(chunks)
        ]

    def _load_pdf(self, path: str) -> str:
        try:
            import pypdf
            reader = pypdf.PdfReader(path)
            pages = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text.strip())
            return "\n\n".join(pages)
        except ImportError:
            raise ImportError("Install pypdf: pip install pypdf")

    def _load_txt(self, path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    def _clean_text(self, text: str) -> str:
        # Normalize whitespace and remove common PDF artifacts
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)  # Fix hyphenation
        text = re.sub(r"\.{3,}", "...", text)
        return text.strip()

    def _chunk_text(self, text: str) -> List[str]:
        """
        Approximate token-based chunking using word count as proxy.
        ~1.3 words per token on average for financial English.
        """
        words = text.split()
        word_chunk = int(self.chunk_size * 1.3)
        word_overlap = int(self.chunk_overlap * 1.3)

        chunks = []
        start = 0

        while start < len(words):
            end = min(start + word_chunk, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            if end == len(words):
                break
            start += word_chunk - word_overlap

        return [c for c in chunks if len(c.strip()) > 50]
