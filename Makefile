# Research workflows for the RAG Document Chatbot.
#
# Usage:  make <target>
# All targets assume the project venv is on PATH (e.g. `source venv/bin/activate`).

PYTHON ?= python

.PHONY: help install test eval eval-offline serve clean-results clean-logs

help:
	@echo "Targets:"
	@echo "  install        Install pinned dependencies."
	@echo "  test           Run unit tests (offline, no API calls)."
	@echo "  eval           Run the full guard evaluation (requires GEMINI_API_KEY)."
	@echo "  eval-offline   Run the regex-only baseline (no API key needed)."
	@echo "  serve          Run the FastAPI server on :8000."
	@echo "  clean-results  Delete benchmarks/results/ (asks first)."
	@echo "  clean-logs     Truncate logs/guard_trace.jsonl."

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest tests/ -v

eval:
	$(PYTHON) -m benchmarks.run_guard_eval

eval-offline:
	$(PYTHON) -m benchmarks.run_guard_eval --skip-llm

serve:
	$(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

clean-results:
	@echo "About to delete benchmarks/results/* — Ctrl-C to abort."
	@read _
	rm -rf benchmarks/results/*
	touch benchmarks/results/.gitkeep

clean-logs:
	: > logs/guard_trace.jsonl
