"""Embedding generation and FAISS index management."""

import json
import logging
import os
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

from app.core.config import Config
from app.core.interfaces import IEmbeddingManager

logger = logging.getLogger(__name__)


class EmbeddingManager(IEmbeddingManager):
    """Manages embedding model, FAISS index, and chunk storage."""

    def __init__(self, config: Config) -> None:
        self._config = config
        logger.info("Loading embedding model: %s", config.EMBEDDING_MODEL)
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: List[Document] = []

    def build_index(self, chunks: List[Document]) -> None:
        """Create a FAISS index from document chunks."""
        self.chunks = chunks
        texts = [chunk.page_content for chunk in chunks]

        logger.info("Generating embeddings for %d chunks", len(texts))
        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings, dtype="float32")

        # Normalize for cosine similarity (using inner product on normalized vectors)
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        logger.info("FAISS index built: %d vectors, dim=%d", self.index.ntotal, dim)

    def save_index(self, path: str = None) -> None:
        """Persist FAISS index, chunks, and a metadata stamp to disk.

        ``index_meta.json`` records the embedding model name, vector
        dimension, and number of vectors so a saved index loaded under a
        different embedding model can be detected before it silently
        corrupts retrieval results.
        """
        path = path or self._config.INDEX_PATH
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))

        chunk_data = [c.page_content for c in self.chunks]
        with open(os.path.join(path, "chunks.json"), "w") as f:
            json.dump(chunk_data, f)

        meta = {
            "embedding_model": self._config.EMBEDDING_MODEL,
            "dim": int(self.index.d) if self.index is not None else None,
            "n_vectors": int(self.index.ntotal) if self.index is not None else 0,
            "chunk_size": self._config.CHUNK_SIZE,
            "chunk_overlap": self._config.CHUNK_OVERLAP,
        }
        with open(os.path.join(path, "index_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("Index saved to %s/ (meta=%s)", path, meta)

    def load_index(self, path: str = None) -> bool:
        """Load a previously saved index. Returns True if successful.

        If ``index_meta.json`` is present and records a different
        embedding model than the one currently configured, refuse to load
        — silently mixing embedding spaces is the most insidious failure
        mode for a research RAG pipeline.
        """
        path = path or self._config.INDEX_PATH
        index_file = os.path.join(path, "index.faiss")
        chunks_file = os.path.join(path, "chunks.json")
        meta_file = os.path.join(path, "index_meta.json")

        if not (os.path.exists(index_file) and os.path.exists(chunks_file)):
            return False

        if os.path.exists(meta_file):
            with open(meta_file, "r") as f:
                meta = json.load(f)
            saved_model = meta.get("embedding_model")
            if saved_model and saved_model != self._config.EMBEDDING_MODEL:
                logger.error(
                    "Refusing to load index: saved with embedding_model=%r but "
                    "config.EMBEDDING_MODEL=%r. Delete %s/ or switch models.",
                    saved_model, self._config.EMBEDDING_MODEL, path,
                )
                return False

        self.index = faiss.read_index(index_file)
        with open(chunks_file, "r") as f:
            chunk_data = json.load(f)
        self.chunks = [Document(page_content=text) for text in chunk_data]
        logger.info("Loaded existing index: %d vectors", self.index.ntotal)
        return True

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        vec = self.model.encode([query])
        vec = np.array(vec, dtype="float32")
        faiss.normalize_L2(vec)
        return vec
