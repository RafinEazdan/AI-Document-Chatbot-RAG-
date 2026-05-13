"""FastAPI dependency providers — wire app.state and config into route handlers."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Request

from app.core.config import Config
from app.core.interfaces import IEmbeddingManager, ILLMProvider, IGuard, IDocumentLoader
from app.memory.memory import ConversationMemory


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Return a process-wide cached Config instance."""
    return Config()


def get_embedding_manager(request: Request) -> IEmbeddingManager:
    """Pull the EmbeddingManager from app.state (populated during lifespan startup)."""
    return request.app.state.embedding_manager


def get_memory(request: Request) -> ConversationMemory:
    """Pull the single ConversationMemory from app.state."""
    return request.app.state.memory


def get_llm_provider(
    config: Annotated[Config, Depends(get_config)],
) -> ILLMProvider:
    from app.rag.llm import GeminiProvider
    return GeminiProvider(config)


def get_guard(
    config: Annotated[Config, Depends(get_config)],
) -> IGuard:
    from pathlib import Path
    from app.rag.guard import TwoTierGuard
    return TwoTierGuard(
        config,
        patterns_path=Path(config.GUARD_PATTERNS_PATH),
        judge_prompt_path=Path(config.GUARD_JUDGE_PROMPT_PATH),
        trace_path=Path(config.GUARD_TRACE_PATH),
        enable_trace=config.GUARD_TRACE_ENABLED,
    )


def get_document_loader(
    config: Annotated[Config, Depends(get_config)],
) -> IDocumentLoader:
    from app.document_loader import DocumentLoader
    return DocumentLoader(config)
