"""Load and chunk PDF/DOCX documents."""

import glob
import logging
import os
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.core.config import Config
from app.core.interfaces import IDocumentLoader
from pypdf import PdfReader


def _load_pdf(path: str) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(f"[Page {i + 1}]\n{text}")
    return "\n\n".join(pages)


def _load_docx(path: str) -> str:
    """Extract text from a DOCX file."""
    from docx import Document as DocxDocument

    doc = DocxDocument(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def load_document(path: str) -> str:
    """Load a single document by extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _load_pdf(path)
    elif ext in (".docx", ".doc"):
        return _load_docx(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def load_all_documents(directory: str) -> str:
    """Load all supported documents from a directory."""
    patterns = ["*.pdf", "*.docx", "*.doc"]
    texts = []
    for pattern in patterns:
        for path in sorted(glob.glob(os.path.join(directory, pattern))):
            print(f"  Loading: {os.path.basename(path)}")
            texts.append(load_document(path))
    if not texts:
        raise FileNotFoundError(
            f"No PDF or DOCX files found in '{directory}'. "
            "Please place your document there and try again."
        )
    return "\n\n".join(texts)


def chunk_text(text: str) -> List[Document]:
    """Split text into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.create_documents([text])
    logging.getLogger(__name__).info(
        "Created %d chunks (size=%d, overlap=%d)",
        len(chunks), Config.CHUNK_SIZE, Config.CHUNK_OVERLAP,
    )
    return chunks


class DocumentLoader(IDocumentLoader):
    """IDocumentLoader implementation backed by pypdf / python-docx.

    Accepts an injected Config so chunk sizes are configurable per-instance.
    """

    def __init__(self, config: Config) -> None:
        self._config = config

    def load(self, path: str) -> str:
        """Load a single document."""
        return load_document(path)

    def load_all(self, directory: str) -> str:
        """Load all supported documents from a directory."""
        return load_all_documents(directory)

    def chunk(self, text: str) -> List[Document]:
        """Chunk text using config values injected at construction time."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._config.CHUNK_SIZE,
            chunk_overlap=self._config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.create_documents([text])
        print(f"  Created {len(chunks)} chunks (size={self._config.CHUNK_SIZE}, overlap={self._config.CHUNK_OVERLAP})")
        return chunks
