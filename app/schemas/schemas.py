"""Pydantic schemas for API request/response models."""

from pydantic import BaseModel


# ── Document Schemas ──

class UploadResponse(BaseModel):
    message: str
    filename: str
    total_chunks: int


class ReindexResponse(BaseModel):
    message: str
    total_chunks: int


class IndexStatusResponse(BaseModel):
    indexed: bool
    total_vectors: int
    document_path: str


# ── Chat Schemas ──

class ChatRequest(BaseModel):
    question: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"question": "What is the company's leave policy?"}
            ]
        }
    }


class SourceChunk(BaseModel):
    chunk_index: int
    score: float
    preview: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    blocked: bool = False


class ClearMemoryResponse(BaseModel):
    message: str


# ── Health Schema ──

class HealthResponse(BaseModel):
    service: str
    status: str
    docs_url: str
    endpoints: dict[str, str]
