# 📄 RAG Document Chatbot

A production-grade, hallucination-free chatbot that answers questions **strictly** from an uploaded PDF or DOCX document using a Retrieval-Augmented Generation (RAG) pipeline, powered by Google Gemini and served through a FastAPI REST API.

---

## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
  - [Project Structure](#project-structure)
  - [RAG Pipeline](#rag-pipeline)
  - [Two-Tier Prompt Injection Guard](#two-tier-prompt-injection-guard)
- [Technical Explanation](#technical-explanation)
  - [Document Ingestion](#1-document-ingestion)
  - [Embedding & Indexing](#2-embedding--indexing)
  - [Retrieval](#3-retrieval)
  - [Prompt Construction & LLM Generation](#4-prompt-construction--llm-generation)
  - [Conversation Memory](#5-conversation-memory)
  - [Hallucination Control](#6-hallucination-control)
- [Libraries & Tools Used](#libraries--tools-used)
- [Design Decisions & Justifications](#design-decisions--justifications)
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [Configuration](#configuration)
- [Docker Deployment](#docker-deployment)
- [Research Use](#research-use)
- [Reproducibility Statement](#reproducibility-statement)

---

## Features

| Feature | Description |
|---------|-------------|
| 📄 **Document grounding** | Answers only from the uploaded document — never guesses |
| 📎 **Source citations** | Every answer cites the chunk(s) it was derived from |
| 📊 **Similarity scores** | Each retrieved chunk includes a cosine-similarity score |
| 💬 **Conversation memory** | Sliding-window history for multi-turn follow-up questions |
| 🛡️ **Two-tier injection guard** | Regex heuristics + LLM confirmation to block prompt injection |
| 🔌 **Dependency injection** | Interface-driven design for testability and swappability |
| 🐳 **Docker support** | One-command containerized deployment |
| ⚡ **Atomic document upload** | Uploading a new document resets the document directory and rebuilds the index for simplicity and atomicity |

---

## Architecture Overview

### Project Structure

```
csn-demo/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── documents/                ← uploaded PDF/DOCX files (mounted volume)
├── vector_store/             ← persisted FAISS index + chunks (mounted volume)
└── app/
    ├── main.py               ← FastAPI application entry point & lifespan
    ├── document_loader.py    ← PDF/DOCX parsing + recursive text chunking
    ├── api/
    │   └── routers.py        ← REST endpoints (documents, chat)
    ├── core/
    │   ├── config.py         ← Centralized env-based configuration
    │   ├── interfaces.py     ← Abstract base classes (DI contracts)
    │   └── dependencies.py   ← FastAPI Depends() provider functions
    ├── rag/
    │   ├── embeddings.py     ← SentenceTransformer + FAISS index management
    │   ├── retriever.py      ← Similarity search (Top-K nearest neighbors)
    │   ├── llm.py            ← Google Gemini LLM provider
    │   └── guard.py          ← Two-tier prompt injection guard
    ├── memory/
    │   ├── memory.py         ← Sliding-window conversation memory
    │   └── chain.py          ← RAG chain: retrieve → prompt → LLM → citations
    ├── schemas/
    │   └── schemas.py        ← Pydantic request/response models
    └── services/
        ├── chat_service.py   ← Chat business logic
        └── document_service.py ← Upload, indexing & status logic
```

### RAG Pipeline

```
                          User Question
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Two-Tier Guard     │
                    │  ┌───────────────┐  │
                    │  │ Tier 1: Regex │──┼──▶ Pass → continue
                    │  └───────┬───────┘  │
                    │     Suspicious?      │
                    │          ▼           │
                    │  ┌───────────────┐  │
                    │  │ Tier 2: LLM   │──┼──▶ Confirm → Block
                    │  │  (Gemini)     │  │    Deny   → Pass
                    │  └───────────────┘  │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │  Embed Query        │  sentence-transformers
                    │  (all-MiniLM-L6-v2) │  → 384-dim vector
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │  FAISS Search       │  Cosine similarity (Inner Product
                    │  (IndexFlatIP)      │  on L2-normalized vectors)
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │  Build Prompt       │  System prompt + conversation
                    │                     │  history + retrieved chunks
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │  Gemini LLM Call    │  gemini-2.5-flash-lite
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │  Answer + Citations │  [Chunk N] references
                    │  + Similarity Scores│  with cosine scores
                    └─────────────────────┘
```

### Two-Tier Prompt Injection Guard

The system's headline contribution is a **layered, measurable defense**
against prompt-injection attacks. The full design document — including
attack-family taxonomy, falsifiable claims, evaluation protocol, and
ablations — lives at [docs/two_tier_guard.md](docs/two_tier_guard.md).

| Tier | Method | Cost | When it runs |
|------|--------|------|-------------|
| **Tier 1** | Regex pattern bank ([configs/guard/patterns.yaml](configs/guard/patterns.yaml)) | Microseconds, free | Every request |
| **Tier 2** | LLM judge (Gemini, temperature=0, single-token output) | ~200ms, 1 API call | Only when Tier 1 fires |

**How it works:**

1. **Tier 1 (Regex Guard)** — 44 IGNORECASE patterns grouped by attack
   family (`instruction_override`, `role_hijack`, `prompt_extraction`,
   `policy_bypass`, `privilege_claim`, `encoding_evasion`). Externalized
   to a versioned YAML so researchers can edit and ablate the pattern
   bank without touching Python; the version string is stamped into
   every decision record.

2. **Tier 2 (LLM Guard)** — A focused security-classifier prompt
   ([configs/guard/judge_prompt.txt](configs/guard/judge_prompt.txt))
   that emits a single `yes`/`no` token. Catches Tier 1 false positives
   (legitimate questions containing flagged surface forms like *"act as
   a guide and walk me through Article 7"*).

**Why two tiers?** A regex-only guard has high recall but low precision
on adversarial benigns. An LLM-only guard is accurate but pays the cost
on every request. Two-tier gives the precision of an LLM-only filter at
the cost of regex-only on the dominant clean-input path. The
[evaluation harness](benchmarks/run_guard_eval.py) measures and reports
this trade-off as `llm_invocation_rate`.

**Observability.** Every guard decision is appended as a JSON line to
`logs/guard_trace.jsonl` with per-tier latency, matched pattern id, LLM
verdict, and pattern-bank version — so any production decision can be
replayed offline for error analysis without re-running the pipeline.

**Reproduce the numbers:**

```bash
python -m benchmarks.run_guard_eval              # full eval (needs GEMINI_API_KEY)
python -m benchmarks.run_guard_eval --skip-llm   # offline, regex-only
```

---

## Technical Explanation

### 1. Document Ingestion

- **Supported formats:** PDF (via `pypdf`) and DOCX (via `python-docx`).
- **Upload flow:** When a new document is uploaded via `POST /documents/upload`, the system **clears all existing files from the `documents/` directory** before saving the new file. This atomic-reset design ensures the chatbot always operates on exactly one document, maintaining simplicity and consistency.
- **Text extraction:** PDFs are parsed page-by-page with page markers (`[Page N]`). DOCX files are extracted paragraph-by-paragraph.

### 2. Embedding & Indexing

- **Chunking:** Extracted text is split into overlapping chunks using LangChain's `RecursiveCharacterTextSplitter` with configurable `CHUNK_SIZE` (default: 500 chars) and `CHUNK_OVERLAP` (default: 100 chars). The recursive strategy splits on `\n\n` → `\n` → `. ` → ` ` → `""`, preserving semantic boundaries.
- **Embedding model:** `all-MiniLM-L6-v2` from Sentence Transformers generates 384-dimensional dense vectors. This model offers an excellent balance of speed, size (~80 MB), and quality for semantic similarity tasks.
- **Vector index:** FAISS `IndexFlatIP` (inner product) is used on L2-normalized vectors, which is mathematically equivalent to cosine similarity but leverages FAISS's optimized inner-product kernels.
- **Persistence:** The FAISS index (`index.faiss`) and chunk texts (`chunks.json`) are saved to the `vector_store/` directory and automatically reloaded on server startup.

### 3. Retrieval

- Top-K (default: 4) chunks are retrieved by cosine similarity.
- Each result includes the chunk content, its index, and the similarity score.
- Results with `idx == -1` (FAISS sentinel for no match) are filtered out.

### 4. Prompt Construction & LLM Generation

- A **strict system prompt** instructs the LLM to answer only from the provided context, cite chunks using `[Chunk N]` notation, and refuse off-document questions with a specific fallback message.
- The prompt assembles: `system prompt` → `conversation history` → `context chunks` → `user question`.
- The Google Gemini API is called with multi-turn chat support (prior history is passed as alternating user/model turns).

### 5. Conversation Memory

- A `ConversationMemory` class maintains a **sliding window** of the last 10 turns (20 messages: 10 user + 10 assistant).
- History is injected into the prompt so the LLM can handle follow-up questions that reference earlier answers.
- Memory can be cleared via `POST /chat/clear`.

### 6. Hallucination Control

The system uses **four layers** of hallucination prevention:

| Layer | Mechanism |
|-------|-----------|
| **Strict system prompt** | Explicitly tells the LLM to only use provided context |
| **Context-only prompting** | The LLM sees only retrieved chunks, not the full document |
| **Chunk citations** | Forces the model to ground answers in specific chunks |
| **Explicit fallback** | If the answer isn't in the context: *"This information is not present in the provided document."* |

---

## Libraries & Tools Used

| Library | Version | Purpose |
|---------|---------|---------|
| **FastAPI** | 0.135.2 | Async REST API framework with auto-generated OpenAPI docs |
| **Uvicorn** | 0.42.0 | ASGI server to run the FastAPI application |
| **Pydantic** | 2.12.5 | Request/response validation and serialization |
| **google-generativeai** | 0.8.6 | Google Gemini API client for LLM calls (chat + guard) |
| **sentence-transformers** | 5.3.0 | Pre-trained embedding models (`all-MiniLM-L6-v2`) |
| **faiss-cpu** | 1.13.2 | Facebook AI Similarity Search for fast vector retrieval |
| **langchain-text-splitters** | 1.1.1 | Recursive character text splitting with semantic boundaries |
| **langchain-core** | 1.2.23 | `Document` data model for chunk representation |
| **pypdf** | 6.9.2 | PDF text extraction |
| **python-docx** | 1.2.0 | DOCX text extraction |
| **python-dotenv** | 1.2.2 | Load configuration from `.env` files |
| **NumPy** | 2.4.4 | Vector operations and array handling |
| **Docker** | — | Containerized deployment with volume mounts |

---

## Design Decisions & Justifications

### Why single-document atomicity?

When a new document is uploaded, the system **deletes all existing files** from `documents/` before saving the new one. This is a deliberate simplification:

- **Atomicity:** The index always corresponds to exactly one document. There is no risk of stale chunks from a previously uploaded file bleeding into answers.
- **Simplicity:** Users don't need to manage document inventories or worry about conflicts.
- **Consistency:** Every question is answered from a single, well-defined source of truth.

### Why FAISS `IndexFlatIP` over approximate methods?

- For document-scale datasets (hundreds to low thousands of chunks), exact search is fast enough and eliminates the complexity of tuning approximate nearest-neighbor parameters (e.g., nprobe, nlist).
- L2-normalization + inner product is mathematically equivalent to cosine similarity but avoids the overhead of FAISS's cosine-specific index types.

### Why `all-MiniLM-L6-v2`?

- ~80 MB model size — loads quickly even in containerized environments.
- 384-dimensional output — compact vectors reduce memory and search time.
- Consistently ranks among the top lightweight models on the [MTEB benchmark](https://huggingface.co/spaces/mteb/leaderboard) for semantic similarity tasks.

### Why a two-tier guard instead of regex-only or LLM-only?

- **Regex-only** is fast but brittle — legitimate questions containing trigger words (e.g., *"override"* in a safety manual) get blocked.
- **LLM-only** is accurate but expensive — every request incurs an API call and ~200ms latency even for clearly benign inputs.
- **Two-tier** combines both: regex pre-screens cheaply, LLM confirms only when needed. In practice, >95% of legitimate requests pass Tier 1 instantly, and the LLM guard only fires for genuinely ambiguous inputs.

### Why dependency injection with abstract base classes?

The `core/interfaces.py` module defines `IEmbeddingManager`, `ILLMProvider`, `IGuard`, and `IDocumentLoader` as abstract base classes. Route handlers receive implementations through FastAPI's `Depends()` system:

- **Testability:** Unit tests can inject mock implementations without touching real APIs or file systems.
- **Swappability:** Switching from Gemini to OpenAI (or adding a new LLM provider) requires only a new class and a one-line change in `dependencies.py`.
- **Separation of concerns:** Route handlers don't know or care which LLM, embedding model, or storage backend is in use.

### Why sliding-window memory instead of full history?

- LLM context windows have token limits. Sending the entire conversation history would eventually exceed them.
- A 10-turn window (configurable) keeps recent context available for follow-up questions while bounding prompt size.
- Older turns naturally become less relevant as the conversation topic shifts.

---

## Quick Start

### 1. Setup

```bash
# Clone and enter project
cd csn-demo

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp app/.env.example app/.env
# Edit app/.env with your Gemini API key
```

### 2. Configure

Edit `app/.env`:

```env
GEMINI_API_KEY=your-gemini-api-key-here
GEMINI_MODEL=gemini-2.5-flash-lite
LLM_GUARD_MODEL=gemini-2.5-flash-lite
```

### 3. Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The server starts at `http://localhost:8000`. Interactive docs are available at `/docs`.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check and API info |
| `POST` | `/documents/upload` | Upload a PDF/DOCX and build the vector index |
| `GET` | `/documents/status` | Check if an index is loaded and its vector count |
| `POST` | `/chat/ask` | Ask a question about the uploaded document |
| `POST` | `/chat/clear` | Clear conversation memory |

### Example: Ask a question

```bash
curl -X POST http://localhost:8000/chat/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the company leave policy?"}'
```

**Response:**

```json
{
  "answer": "According to the document, employees are entitled to ... [Chunk 3]",
  "sources": [
    {
      "chunk_index": 3,
      "score": 0.8721,
      "preview": "All full-time employees are entitled to 20 days of annual leave..."
    }
  ]
}
```

---

## Configuration

All settings are loaded from environment variables (via `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | — | Your Google Gemini API key |
| `GEMINI_MODEL` | `gemini-2.5-flash-lite` | Model for answer generation |
| `LLM_GUARD_MODEL` | `gemini-2.5-flash-lite` | Model for Tier 2 injection classification |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between consecutive chunks |
| `TOP_K` | `4` | Number of chunks to retrieve per query |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model name |
| `DOCUMENT_PATH` | `documents/` | Directory for uploaded documents |
| `INDEX_PATH` | `vector_store/` | Directory for persisted FAISS index |

---

# 🐳 Docker Deployment

You can run this project either by building the image locally (recommended for development) or by using the prebuilt image from Docker Hub (recommended for quick testing).

## 🔹 Option 1: Build Locally (Development)

```bash
# Build and run using Docker Compose
docker-compose up --build
```

Or manually:

```bash
docker build -t eazdan-rafin-rag-chatbot .
```

```bash
docker run -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/documents:/app/documents \
  -v $(pwd)/vector_store:/app/vector_store \
  eazdan-rafin-rag-chatbot
```

The Docker image uses `python:3.11-slim` as the base, exposes port 8000, and mounts two volumes:
- `documents/` — stores input documents for indexing
- `vector_store/` — persists FAISS index and embeddings


# Render Deployment

During deployment on Render, the application exceeded the 512MB memory limit due to the embedding model and in-memory FAISS index.

Since the primary goal of this project was to demonstrate the RAG pipeline and conversational behavior, I did not optimize for low-memory deployment environments.

In a production setting, this could be addressed by using a lighter embedding model, external vector databases (such as. Pinecone etc), or a higher-memory hosting plan.

---

## Research Use

This codebase is designed to double as a research platform for studies on
prompt-injection defenses, retrieval, and grounded generation. Most of the
research surface lives **alongside** the production code rather than inside it,
so the API stays clean.

### Headline contribution — two-tier prompt-injection guard

A measurable, layered defense (Tier 1 regex bank → Tier 2 LLM judge) with
an externalized, versioned pattern bank, a labeled benchmark dataset,
JSONL decision tracing, and a one-command evaluation harness that
compares three configurations (`regex_only`, `llm_only`, `two_tier`) on
the same data and writes a CSV/manifest pair you can drop straight into
a paper.

Full design + claims + protocol: [docs/two_tier_guard.md](docs/two_tier_guard.md).

### Research artifacts in this repo

| Path | Purpose |
|---|---|
| [docs/two_tier_guard.md](docs/two_tier_guard.md) | Standalone design + evaluation document for the two-tier guard. |
| [configs/guard/patterns.yaml](configs/guard/patterns.yaml) | Versioned regex pattern bank, grouped by attack family. |
| [configs/guard/judge_prompt.txt](configs/guard/judge_prompt.txt) | LLM judge system prompt (editable, no code change). |
| [benchmarks/injections.jsonl](benchmarks/injections.jsonl) | 80-row labeled dataset (50 injections × 6 families + 30 benigns inc. adversarial). |
| [benchmarks/run_guard_eval.py](benchmarks/run_guard_eval.py) | Evaluation harness — precision/recall/F1/latency/cost per config. |
| [benchmarks/results/](benchmarks/results/) | Versioned per-run outputs (`summary.csv`, `per_example.jsonl`, `manifest.json`). |
| [tests/test_guard.py](tests/test_guard.py) | 18 deterministic, offline unit tests for the guard. |
| `logs/guard_trace.jsonl` | Append-only JSONL trace of every guard decision in production. |

### Running the evaluation

```bash
# Offline (regex-only baseline, no API key)
python -m benchmarks.run_guard_eval --skip-llm

# Full eval (regex_only + llm_only + two_tier)
python -m benchmarks.run_guard_eval

# Ablations
python -m benchmarks.run_guard_eval --patterns my_patterns.yaml
python -m benchmarks.run_guard_eval --judge-prompt my_prompt.txt
python -m benchmarks.run_guard_eval --dataset multilingual.jsonl
```

Each run writes a timestamped folder under `benchmarks/results/` containing:
* `summary.csv` — one row per configuration.
* `per_example.jsonl` — every decision (id, tier reached, latencies, verdict).
* `manifest.json` — dataset SHA1, pattern-bank version, judge model + temperature.

### Tests

```bash
pip install pytest
pytest tests/                     # 18 tests, fully offline
```

---

## Reproducibility Statement

The intent of this section is that a second researcher with this repo
and a Gemini API key can re-derive the reported numbers byte-for-byte.

* **Pinned dependencies.** `requirements.txt` uses exact `==` versions
  for every package the eval depends on. No `>=` ranges remain.
* **Determinism.**
  * Tier-1 regex is deterministic by construction.
  * Both the answering LLM and the Tier-2 judge are configured with
    `temperature=0.0` (`LLM_TEMPERATURE`, `LLM_GUARD_TEMPERATURE`).
  * Embedding model: `all-MiniLM-L6-v2` (deterministic for a given
    `sentence-transformers` version).
  * FAISS uses `IndexFlatIP` over L2-normalised vectors → exact cosine
    search, no approximate-search noise.
* **Index integrity.** Every saved index writes
  `vector_store/index_meta.json` with the embedding model name and
  vector dimension. `EmbeddingManager.load_index` refuses to load an
  index that was saved under a different embedding model — silently
  mixing embedding spaces is impossible.
* **Run provenance.** Every eval run writes `manifest.json` recording:
  dataset path + SHA1, pattern-bank `version:` field, judge model name,
  judge temperature, UTC timestamp, run id.
* **Trace provenance.** Every production decision in
  `logs/guard_trace.jsonl` records `pattern_version`, so a trace can be
  matched against the exact pattern bank that produced it.
* **Hardware.** All numbers in `benchmarks/results/` should be tagged
  with the machine that produced them when reported externally
  (CPU / RAM / OS). Latency is wall-clock and will vary across machines;
  precision / recall / F1 are deterministic and will not.
* **What is NOT yet pinned.** The Gemini judge model is a hosted API and
  can change on Google's side without a version bump. We pin the model
  *name* (`gemini-2.5-flash-lite` by default) and the request-time
  parameters; the model weights themselves are not under our control.
  Re-runs against the same model name on a different date may shift
  Tier-2 numbers slightly. Mitigation: include the manifest UTC
  timestamp when citing results.