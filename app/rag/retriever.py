"""Retrieve relevant chunks from the FAISS index."""

from typing import List, Optional

from app.core.config import Config
from app.core.interfaces import IEmbeddingManager


class RetrievalResult:
    """A single retrieved chunk with its similarity score."""

    def __init__(self, content: str, score: float, chunk_index: int) -> None:
        self.content = content
        self.score = score
        self.chunk_index = chunk_index


def retrieve(
    query: str,
    embedding_manager: IEmbeddingManager,
    top_k: Optional[int] = None,
    config: Optional[Config] = None,
) -> List[RetrievalResult]:
    """Search the index for the most relevant chunks.

    ``top_k`` is resolved in this priority order so DI overrides actually
    take effect: explicit argument → injected ``config`` → process Config.
    """
    if top_k is None:
        top_k = (config or Config).TOP_K

    if embedding_manager.index is None or not embedding_manager.chunks:
        return []

    query_vec = embedding_manager.embed_query(query)

    scores, indices = embedding_manager.index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append(
            RetrievalResult(
                content=embedding_manager.chunks[idx].page_content,
                score=float(score),
                chunk_index=int(idx),
            )
        )
    return results
