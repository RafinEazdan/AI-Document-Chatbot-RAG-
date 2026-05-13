"""Chat service — handles question answering and conversation memory."""

from app.memory.chain import ask
from app.core.interfaces import IEmbeddingManager, ILLMProvider, IGuard
from app.memory.memory import ConversationMemory


def ask_question(
    question: str,
    embedding_manager: IEmbeddingManager,
    memory: ConversationMemory,
    llm_provider: ILLMProvider,
    guard: IGuard,
) -> dict:
    """Process a question through the RAG pipeline.

    The chain returns a single retrieval result set; we render sources
    from those exact results so the API response and the prompt context
    can never disagree (single-source-of-truth retrieval).
    """
    out = ask(
        question,
        embedding_manager,
        memory,
        llm_provider=llm_provider,
        guard=guard,
    )

    sources = [
        {
            "chunk_index": r.chunk_index,
            "score": round(r.score, 4),
            "preview": r.content[:120].replace("\n", " "),
        }
        for r in out["results"]
    ]

    return {
        "answer": out["answer"],
        "sources": sources,
        "blocked": out.get("blocked", False),
    }


def clear_memory(memory: ConversationMemory) -> str:
    """Clear conversation memory."""
    memory.clear()
    return "Conversation memory cleared."
