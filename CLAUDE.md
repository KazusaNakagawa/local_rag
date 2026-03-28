# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`local-rag` is a fully local RAG system for querying Obsidian vaults. All processing is offline using Ollama for embedding and LLM inference. It combines BM25 keyword search + FAISS vector search (MMR) with a Streamlit chat UI and SQLite-backed chat history.

## Prerequisites

```bash
brew install ollama
ollama serve           # must be running before ingest or query
ollama pull nomic-embed-text
ollama pull qwen2.5:7b
```

## Commands

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Ingest Obsidian vault (run from src/)
cd src && python ingest.py

# Query via CLI (run from src/)
cd src && python query.py "your question"

# Start Streamlit UI (run from src/, opens http://localhost:8501)
cd src && streamlit run app.py

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_session_repo.py -v

# Run a single test
pytest tests/test_session_repo.py::test_create_session -v
```

## Architecture

### Data Flow

```
Obsidian Vault → ObsidianLoader → RecursiveCharacterTextSplitter (500 chars, 100 overlap)
    ├── FAISS vectorstore  (nomic-embed-text embeddings, saved to data/vectorstore/)
    └── BM25 cache         (pickled docs, saved to data/docs_cache.pkl)

User Query → Hybrid retrieval (BM25 + FAISS/MMR) → deduplicate → TOP_K=6 chunks
    → Prompt (src/prompts.py) + qwen2.5:7b → Streamed answer
    → SQLite (data/chat.db) for session persistence
```

### Key Modules

| File | Role |
|------|------|
| `src/config.py` | Central config — all paths, model names, tuning params (edit here first) |
| `src/ingest.py` | One-time vault ingestion; must re-run after vault changes |
| `src/app.py` | Streamlit UI — thin layer delegating to rag.py, db/, log_config.py |
| `src/rag.py` | RAG pipeline — load_resources, make_llm, hybrid retrieve, invoke_answer, stream_answer |
| `src/query.py` | CLI interface — BM25 + FAISS hybrid (same pipeline as app.py) |
| `src/log_config.py` | Logging setup — daily rotating file handler (logs/YYYYMMDD-app.log) |
| `src/db/` | SQLite layer: `connection.py`, `models.py`, `session_repo.py`, `message_repo.py`, `export.py` |

### Hybrid Search (src/app.py)

BM25 uses a custom Japanese tokenizer (`japanese_tokenizer()`) that produces character unigrams and bigrams to support Japanese text. Results from BM25 and FAISS are merged and deduplicated before passing to the LLM.

### Configuration (src/config.py)

Key tuning parameters:
- `OBSIDIAN_VAULT_PATH` — set this to your vault location
- `CHUNK_SIZE` / `CHUNK_OVERLAP` — affects recall vs. context noise
- `MAX_CHUNK_CHARS` = 1500 — truncates chunks to avoid LLM context overflow
- `TOP_K` = 6, `FETCH_K` = 20 — MMR diversity controls
- `LLM_MODEL` — switch to `qwen2.5:14b` for better quality on M-series with Metal GPU

### Database Schema

```sql
sessions (id TEXT PRIMARY KEY, title TEXT, created_at DATETIME)
messages (id, session_id → sessions(id) ON DELETE CASCADE, role CHECK('user'|'assistant'), content, created_at)
```

Tests use an in-memory SQLite fixture (`conftest.py::tmp_db`) — no real DB is touched during test runs.

## Coding Conventions

### Modularization

**Each file must have a single, clearly defined responsibility. Do not place multiple unrelated concerns in the same file.**

| Responsibility | Where it belongs |
|----------------|-----------------|
| RAG pipeline logic (retrieval, prompts, LLM calls) | `src/rag.py` |
| Streamlit UI and user interaction | `src/app.py` |
| Logging configuration | `src/log_config.py` |
| Central config and constants | `src/config.py` |
| DB connection | `src/db/connection.py` |
| DB schema / migrations | `src/db/models.py` |
| Session CRUD | `src/db/session_repo.py` |
| Message CRUD | `src/db/message_repo.py` |
| Export helpers | `src/db/export.py` |

**Rules:**
- When adding a new cross-cutting concern (logging, metrics, auth, etc.), create a dedicated module rather than embedding the logic in an existing file.
- `app.py` must remain a thin UI layer — it delegates to `rag.py`, `log_config.py`, and `db/` rather than implementing logic itself.
- If a function or class does not belong to the current file's stated responsibility, extract it to the appropriate module before adding it.
