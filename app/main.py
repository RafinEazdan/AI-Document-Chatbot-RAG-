"""RAG Document Chatbot — FastAPI Server."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routers import document_router, chat_router
from app.core.config import Config
from app.core.dependencies import get_config
from app.memory.memory import ConversationMemory
from app.rag.embeddings import EmbeddingManager
from app.schemas.schemas import HealthResponse


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared state on startup; clean up on shutdown."""
    config: Config = get_config()

    # Single-user conversation memory
    app.state.memory = ConversationMemory(max_turns=10)

    # Attempt to load a previously saved FAISS index
    em = EmbeddingManager(config)
    if em.load_index():
        logger.info("Loaded existing vector index on startup.")
    else:
        logger.info("No existing index found. Upload a document via POST /documents/upload")

    app.state.embedding_manager = em

    yield


app = FastAPI(lifespan=lifespan)

app.include_router(document_router)
app.include_router(chat_router)


@app.get("/", tags=["Health"], response_model=HealthResponse)
async def root():
    """Health check and API info."""
    return HealthResponse(
        service="RAG Document Chatbot API",
        status="running",
        docs_url="/docs",
        endpoints={
            "upload_document": "POST /documents/upload",
            "index_status": "GET /documents/status",
            "ask_question": "POST /chat/ask",
            "clear_memory": "POST /chat/clear",
        },
    )
