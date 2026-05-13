# Research Readiness Report — RAG Document Chatbot

This report reviews the entire codebase and gives a plain-language answer to one question: *Is this project ready to be used for research, and if not, what should be added or changed to make it research-ready?*

I have **not modified any code**. Everything below is observation and recommendation.

---

## 1. What This Project Is Today

It is a clean, **production-style demo** of a Retrieval-Augmented Generation (RAG) chatbot:

- Upload one PDF/DOCX → text is chunked → embeddings stored in FAISS.
- A user asks a question → top-K chunks are retrieved → Gemini generates an answer with citations.
- A two-tier (regex + LLM) guard blocks prompt-injection attempts.
- A sliding-window conversation memory supports follow-up questions.
- FastAPI exposes everything over a small REST API; Docker support is included.

The architecture is well-organized: clear layers (`api`, `services`, `rag`, `memory`, `core`), interface-driven dependency injection, and a sensible README. **As a working demo, it is solid.**

But "solid demo" and "research-ready" are two different things. A research codebase needs **reproducibility, evaluation, comparability, observability, and configurable experimentation** — and that's where this project is currently thin.

---

## 2. Short Verdict

**Not yet research-ready, but the foundation is good.** With moderate, well-scoped additions (mostly *around* the existing code rather than inside it) it can become a strong research platform. Most of what's missing is **measurement infrastructure**, not core engineering.

I rate the readiness areas as follows:

| Area | Status | Notes |
|---|---|---|
| Code structure & architecture | Good | Clean DI, clear modules. |
| Documentation (README) | Good | Explains pipeline well. |
| Reproducibility | Weak | No pinned deps, no seeds, no lock file. |
| Evaluation framework | Missing | No metrics, no test set, no golden Q&A pairs. |
| Experiment tracking | Missing | No MLflow / W&B / CSV results logging. |
| Automated testing | Missing | No `tests/` folder. |
| Logging / observability | Weak | Only `print()` statements. |
| Configurability for ablations | Partial | Some configs available, but key choices hardcoded. |
| Multi-document / multi-user research | Limited | Single-doc-by-design; single global memory. |
| Robustness / error handling | Partial | Several silent failure paths. |
| Comparability (baselines) | Missing | No way to swap retriever, reranker, embedding model side-by-side. |

---

## 3. Strengths to Build On

1. **Interface-driven design** ([app/core/interfaces.py](app/core/interfaces.py)): `IEmbeddingManager`, `ILLMProvider`, `IGuard`, `IDocumentLoader` make it easy to add alternative implementations for ablation studies.
2. **Centralized config** ([app/core/config.py](app/core/config.py)): Most knobs (chunk size, top-k, embedding model) come from environment variables — handy for sweep experiments.
3. **Deterministic-ish retrieval**: FAISS `IndexFlatIP` with L2-normalized vectors gives **exact** cosine search. No approximate-search noise to worry about.
4. **Persistence**: FAISS index and chunks are saved to disk, so once indexed you can re-run experiments without re-encoding.
5. **Two-tier guard**: A nice piece of applied research itself — regex + LLM classifier with cost-vs-accuracy reasoning is already a publishable design pattern.
6. **Citations + similarity scores**: The system already returns retrieval scores in API responses ([app/services/chat_service.py:28-35](app/services/chat_service.py#L28-L35)). This is exactly the kind of signal evaluation pipelines need.

---

## 4. Gaps That Block Research Use

### 4.1 No evaluation framework (the biggest gap)

There is no way to **measure** answer quality, retrieval quality, or hallucination rate. Research requires numbers. You currently have none of:

- **Golden Q&A set** — a JSON/CSV file of `(question, expected_answer, expected_chunks)` triples.
- **Retrieval metrics** — Recall@K, MRR, nDCG, Hit@K.
- **Generation metrics** — exact match, F1, ROUGE, BLEU, BERTScore, or LLM-as-judge.
- **Hallucination metrics** — RAGAS faithfulness/answer-relevance, Trulens groundedness, or simple "is answer entailed by context?" check.
- **Guard metrics** — false-positive / false-negative rate on a labeled injection dataset.
- **Latency & cost metrics** — per-stage timings, token counts, $/query.

Without these, you cannot defend any claim, tune any parameter, or compare any change.

### 4.2 No automated tests

There is no `tests/` directory. A research codebase needs at minimum:

- Unit tests for chunking, embedding shapes, retrieval indices, memory rollover, guard regex behavior.
- Integration tests for the upload → index → query path with a tiny fixture document.
- Regression tests so an embedding-model swap doesn't silently break retrieval.

### 4.3 Reproducibility is fragile

- `requirements.txt` uses `>=` ranges, not pins. Two researchers running `pip install` weeks apart can get different versions of `sentence-transformers`, `faiss`, `google-generativeai`, etc.
- No lock file (`requirements.lock`, `poetry.lock`, `uv.lock`).
- No random seed control — `SentenceTransformer` and Gemini calls are not seeded; Gemini is non-deterministic by default (no `temperature=0` is set in [app/rag/llm.py](app/rag/llm.py)).
- Index version not stamped — the saved FAISS file does not record which embedding model produced it. Loading it under a different model would silently corrupt results.
- No data versioning (DVC, Git LFS) for documents or indices.

### 4.4 Logging and telemetry are weak

The codebase uses `print()` throughout (see [app/document_loader.py:69](app/document_loader.py#L69), [app/rag/embeddings.py:30](app/rag/embeddings.py#L30), [app/main.py:26](app/main.py#L26)). For research you want:

- A real `logging` module setup (structured, leveled).
- Per-request trace IDs.
- Persistent JSONL logs of `{question, retrieved_chunks, scores, latency, answer}` for offline analysis.
- Optional integration with OpenTelemetry / Langfuse / Phoenix Arize / W&B.

### 4.5 Some hardcoded behavior limits ablation

- `Config.TOP_K` is read **statically** in [app/rag/retriever.py:24](app/rag/retriever.py#L24) (`top_k or Config.TOP_K`) instead of going through the injected config. This means changing TOP_K at runtime via the dependency-injected config object will not always take effect.
- The chunking separators (`["\n\n", "\n", ". ", " ", ""]`) are hardcoded inside both `chunk_text` and `DocumentLoader.chunk` ([app/document_loader.py:66](app/document_loader.py#L66), [app/document_loader.py:95](app/document_loader.py#L95)).
- The system prompt is a constant in [app/memory/chain.py:11](app/memory/chain.py#L11) — research often needs prompt sweeps.
- The 40+ injection regex patterns are hardcoded in [app/rag/guard.py:10-55](app/rag/guard.py#L10-L55) — should be loadable from a YAML/JSON file so researchers can edit and version them without touching Python.
- The LLM guard system instruction is hardcoded ([app/rag/guard.py:86-90](app/rag/guard.py#L86-L90)).
- LLM `temperature`, `top_p`, `max_output_tokens` are never set — Gemini defaults are used silently.

### 4.6 Single global conversation memory

[app/main.py:21](app/main.py#L21) creates **one** `ConversationMemory` shared across all incoming requests. For research this means:

- You cannot run a multi-user evaluation in parallel without state leaking between sessions.
- You cannot benchmark conversational metrics across many threads.
- A `session_id` request field with per-session memory is needed before any multi-turn evaluation makes sense.

### 4.7 Single-document atomicity is by design — but limits research

Uploading a new file deletes the old one ([app/services/document_service.py:39-42](app/services/document_service.py#L39-L42)). Fine for a demo, but for research you want:

- Multiple corpora indexed side-by-side.
- The ability to switch corpora at query time (e.g. for cross-domain tests).
- Index naming / versioning (`index_v1_minilm/`, `index_v2_bge/`).

### 4.8 Robustness gaps

- `response.text` is read directly in [app/rag/llm.py:46](app/rag/llm.py#L46) and [app/rag/guard.py:97](app/rag/guard.py#L97) — Gemini can block on safety filters and return no `.text`, causing an exception. No retry or fallback.
- The retriever calls `embedding_manager.index.search` without checking that the index is loaded ([app/rag/retriever.py:27](app/rag/retriever.py#L27)). If a user queries before uploading, this will throw an unfriendly error.
- `chat_service.ask_question` calls `retrieve()` **twice** — once inside `ask()` ([app/memory/chain.py:64](app/memory/chain.py#L64)) and once again in [app/services/chat_service.py:27](app/services/chat_service.py#L27) to extract sources. This wastes compute and, more importantly, **the two retrievals could disagree** if any randomness were introduced. A research pipeline needs single-source-of-truth retrieval results.
- Guard regex is English-only — cannot test non-English injections (e.g. Bangladesh Constitution PDF is in English, but multilingual research is blocked).

### 4.9 No baselines / no comparators

For a RAG paper or report you typically need to compare:

- Different embedding models (MiniLM vs BGE vs E5 vs OpenAI-3-small).
- Different chunk sizes / strategies (fixed vs recursive vs semantic).
- Dense-only vs hybrid (BM25 + dense) vs hybrid + reranker (e.g. `bge-reranker`).
- Different LLMs (Gemini, GPT-4o, Claude, Llama 3).
- With vs without query rewriting (HyDE, multi-query).

The interfaces support adding these, but **no second implementation of any interface exists** today.

### 4.10 Minor cleanup / hygiene

- `ReindexResponse` schema is defined but never used ([app/schemas/schemas.py:14](app/schemas/schemas.py#L14)).
- `app/.env` is committed to the working tree (only `.env` at root is in `.gitignore`). Verify your real API key isn't already in git history.
- No `.dockerignore` — the Docker build copies `venv/`, `__pycache__/`, etc. into the image, bloating it.
- `documents/` is in `.gitignore` but `documents/Bangladesh Constitution.pdf` is present locally — fine, but make corpus origin / version explicit somewhere.
- No CI (GitHub Actions) running tests / linting.
- No type checker config (`mypy`, `pyright`) or formatter (`black`, `ruff`) configured.

---

## 5. What "Research-Ready" Should Look Like

A research-ready version of this project would, at minimum, have:

1. **A `benchmarks/` (or `eval/`) directory** containing:
   - `golden_qa.jsonl` — at least 50–100 hand-labeled `{question, ideal_answer, supporting_chunk_ids}` rows for the Bangladesh Constitution PDF (and any other corpora).
   - `injections.jsonl` — labeled prompt-injection vs benign inputs for guard evaluation.
   - `run_eval.py` — a script that loads the index, runs every question through the pipeline, and computes Recall@K, MRR, faithfulness, answer-relevance, latency, and cost.
   - `results/` — versioned per-run CSV/JSON outputs.

2. **Pinned, locked dependencies** + a `Makefile` or `pyproject.toml` with `make eval`, `make test`, `make sweep`.

3. **Experiment configuration via YAML** (e.g. `configs/exp_minilm_top4.yaml`) with overrides for embedding model, chunk size, top-K, prompt, LLM, temperature. The script reads the YAML, runs the eval, and writes results stamped with the config hash.

4. **Structured logging + per-query JSONL traces** so any run can be diffed and post-analyzed.

5. **At least two implementations of each interface** — e.g. a second `IEmbeddingManager` using BGE or E5; a second `ILLMProvider` using OpenAI; a second `IGuard` (LLM-only baseline) — so ablations are real.

6. **Optional but high-leverage components**: a reranker stage, a hybrid (BM25 + dense) retriever, a query rewriter (HyDE / multi-query). These are standard in modern RAG research.

7. **Per-session memory**: `session_id` in `ChatRequest`, dictionary of `ConversationMemory` keyed by session.

8. **CI**: GitHub Actions runs unit tests + a tiny smoke eval on every push.

9. **Reproducibility statement** in the README: model versions, seeds, hardware, dataset sources, exact commands.

---

## 6. Suggested Roadmap (Priority Order)

I'd add the items below roughly in this order. None require touching the existing core code in destructive ways — most live in **new files** alongside the current structure.

### Phase 1 — Make experiments reproducible (1–2 days)
- [ ] Pin `requirements.txt` to exact versions (`==`) and add `pip-compile` lock.
- [ ] Set Gemini `temperature=0` (or a fixed value) and seed any randomness.
- [ ] Stamp the embedding model name and dimension into `chunks.json` / a sibling `index_meta.json`.
- [ ] Replace `print()` with the `logging` module; add a JSONL trace logger.
- [ ] Add a `.dockerignore`.

### Phase 2 — Build the evaluation harness (3–5 days)
- [ ] Create `benchmarks/golden_qa.jsonl` for the existing Bangladesh Constitution corpus.
- [ ] Create `benchmarks/injections.jsonl` for the guard.
- [ ] Write `benchmarks/run_eval.py` that computes Recall@K, MRR, faithfulness (LLM-judge), latency, and writes `results/<timestamp>.csv`.
- [ ] Add a tiny `tests/` folder (chunking, embedding shape, memory rollover, regex guard).
- [ ] Add GitHub Actions for tests + lint.

### Phase 3 — Make the system configurable for ablations (2–4 days)
- [ ] Move the system prompt and guard patterns into editable YAML/text files.
- [ ] Add a `configs/` directory with per-experiment YAMLs.
- [ ] Fix the `Config.TOP_K` static reference in `retriever.py` so DI overrides actually win.
- [ ] Add `session_id` to `ChatRequest` + a `MemoryStore` keyed by session.
- [ ] Allow multiple named indices on disk; let upload accept an optional `corpus_name`.

### Phase 4 — Add baselines & advanced components (1–2 weeks)
- [ ] Second embedding backend (e.g. BGE or E5).
- [ ] Second LLM provider (OpenAI or local Llama via vLLM/Ollama).
- [ ] BM25 + dense hybrid retriever and a `bge-reranker` stage.
- [ ] Query rewriting (HyDE or multi-query) as an optional step in the chain.
- [ ] LLM-only and rules-only guard baselines for the two-tier comparison.

### Phase 5 — Reporting (ongoing)
- [ ] Notebook (`reports/analysis.ipynb`) loading `results/*.csv` and producing the comparison plots/tables that go into a paper or report.
- [ ] Reproducibility README section: exact commands, model versions, seeds, hardware.

---

## 7. TL;DR

- **As an engineering demo**, this codebase is well-built and clearly documented.
- **As a research codebase**, it is missing the four pillars of empirical work: an **evaluation harness**, **labeled datasets**, **experiment tracking**, and **comparable baselines**.
- The **interfaces are already in place** to add those things without rewriting the core.
- The fastest path to "research-ready" is: (1) pin everything for reproducibility, (2) build a small golden Q&A set + eval script, (3) add per-session memory and configurable prompts, (4) drop in a second embedding model and a reranker so you have something to compare against.

Once those four steps are done, the project becomes a credible platform for studies on chunking strategies, embedding-model trade-offs, prompt-injection defenses, hallucination control, or multi-turn grounding behavior.
