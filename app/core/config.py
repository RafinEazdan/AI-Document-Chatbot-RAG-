"""Centralized configuration loaded from environment variables."""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Gemini (used by the main answering LLM only)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

    # Tier-2 guard runs locally on Ollama to stay off the Gemini quota.
    LLM_GUARD_MODEL: str = os.getenv("LLM_GUARD_MODEL", "llama3.2")
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_TIMEOUT: float = float(os.getenv("OLLAMA_TIMEOUT", "30.0"))

    # Reproducibility — pinned to 0.0 by default so eval runs are deterministic.
    # Override per-experiment via env when intentionally sweeping temperature.
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    LLM_GUARD_TEMPERATURE: float = float(os.getenv("LLM_GUARD_TEMPERATURE", "0.0"))

    # RAG settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    TOP_K: int = int(os.getenv("TOP_K", "4"))
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Paths
    DOCUMENT_PATH: str = os.getenv("DOCUMENT_PATH", "documents/")
    INDEX_PATH: str = os.getenv("INDEX_PATH", "vector_store/")

    # Guard configuration paths (versioned, externally editable artifacts).
    GUARD_PATTERNS_PATH: str = os.getenv("GUARD_PATTERNS_PATH", "configs/guard/patterns.yaml")
    GUARD_JUDGE_PROMPT_PATH: str = os.getenv("GUARD_JUDGE_PROMPT_PATH", "configs/guard/judge_prompt.txt")
    GUARD_TRACE_PATH: str = os.getenv("GUARD_TRACE_PATH", "logs/guard_trace.jsonl")
    GUARD_TRACE_ENABLED: bool = os.getenv("GUARD_TRACE_ENABLED", "true").lower() == "true"

    # Hallucination control
    NOT_FOUND_RESPONSE: str = (
        "This information is not present in the provided document."
    )
