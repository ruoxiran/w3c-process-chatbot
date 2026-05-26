# W3C Process Chatbot

Internal W3C Process assistant built with Harness Engineering, deterministic workflow gates, and RAG over authoritative W3C Process and Guidebook sources.

## What This Includes

- `apps/api`: FastAPI service with scope classification, source allowlist checks, RAG retrieval hooks, citation checks, audit-ready responses, and eval endpoints.
- `apps/web`: Next.js UI styled after the W3C Design System and Manual of Style.
- `deploy/docker-compose.yml`: local development stack for API, web, PostgreSQL, Redis, Qdrant, and a vLLM-compatible service slot.
- `packages/ui`: shared W3C design tokens.

The first implementation is intentionally safe by default: if no local model or indexed corpus is available, the API still starts and returns grounded fallback answers or scope refusals. Connect vLLM, Qdrant, and the ingestion worker to enable full RAG-backed generation.

## Quick Start

```bash
cp .env.example .env
python3 -m venv .venv
source .venv/bin/activate
pip install -r apps/api/requirements.txt
uvicorn app.main:app --reload --app-dir apps/api
```

Install the optional production RAG adapters when you are ready to connect LangGraph, LlamaIndex, and Qdrant:

```bash
pip install -r apps/api/requirements-rag.txt
```

In another terminal:

```bash
cd apps/web
npm install
npm run dev
```

Open `http://localhost:3000`.

## Starting A New Agent Session

For a fresh coding session, load the memory files first:

```text
Please read SESSION_BOOTSTRAP.md, PROJECT_MEMORY.md, and PLAN.md before making changes.
```

- `SESSION_BOOTSTRAP.md`: shortest operational context for a new session.
- `PROJECT_MEMORY.md`: project decisions, completed work, TODOs, important files, and architecture notes.
- `PLAN.md`: detailed evolving implementation plan.
- `SYSTEM_ARCHITECTURE.md`: current end-to-end system design, authority model, data flow, and runtime workflow.

## Docker Compose

```bash
cp .env.example .env
docker compose -f deploy/docker-compose.yml up --build
```

The compose file includes a placeholder vLLM service profile. Start it only on a GPU host and update `LLM_MODEL` as needed.

## Environment

See `.env.example` for all configuration. Important settings:

- `LLM_BASE_URL`: OpenAI-compatible vLLM endpoint.
- `LLM_PROVIDER`: `ollama`, `openai-compatible`, `openai`, `openrouter`, or `template`.
- `LLM_MODEL`: local instruct model name for Ollama.
- `OPENAI_COMPATIBLE_BASE_URL`: online or internal OpenAI-compatible `/v1` endpoint.
- `OPENAI_COMPATIBLE_API_KEY`: API key for the online/internal provider.
- `OPENAI_COMPATIBLE_MODEL`: default model when using the OpenAI-compatible provider.
- `QDRANT_URL`: vector database endpoint.
- `SOURCE_ALLOWLIST`: authoritative domains and repositories.
- `ENABLE_MEMBER_ONLY_SOURCES`: keep `false` for v1 unless per-user retrieval filtering is implemented.

## Safety Model

The workflow treats user input as untrusted. W3C Process and Guidebook context is retrieved only from allowlisted sources. Normative claims require citations; Guidebook-only claims are labelled as practice guidance. Out-of-scope questions are refused with a short explanation.

## Useful Commands

```bash
pytest apps/api
npm --prefix apps/web run lint
```

Run the backend regression harness:

```bash
curl -s -X POST http://127.0.0.1:8000/eval/run
```

The eval harness checks more than scope: expected workflow intent, source families, required citation URLs, entity grounding, compiled-context use, answer/next-step terms, and forbidden misgrounding terms.

The web UI also exposes this in the right-side `Quality` tab. The default eval run is deterministic and offline-friendly, so it is suitable as a quick regression signal before retrieval or model changes.

## Dense Retrieval Cache

The local retriever always supports BM25-style lexical scoring and sparse TF-IDF similarity. To add dense retrieval with Ollama embeddings:

```bash
ollama pull qwen3-embedding:4b
./.venv/bin/python scripts/build_embedding_cache.py --model qwen3-embedding:4b --resume
```

Then set:

```env
RETRIEVAL_DENSE_ENABLED=true
OLLAMA_EMBEDDING_MODEL=qwen3-embedding:4b
```

If the cache or embedding model is unavailable, retrieval falls back to the existing lexical/sparse path.

## Online Model Provider

The assistant can use any OpenAI-compatible chat-completions provider while keeping the same W3C safety harness:

```env
LLM_PROVIDER=openai-compatible
OPENAI_COMPATIBLE_BASE_URL=https://api.openai.com/v1
OPENAI_COMPATIBLE_API_KEY=...
OPENAI_COMPATIBLE_MODEL=gpt-4.1
```

For OpenRouter or an internal model gateway, point `OPENAI_COMPATIBLE_BASE_URL` at that provider's `/v1` endpoint and set `OPENAI_COMPATIBLE_MODEL` to the provider model id. The model improves synthesis quality, but Process / Guidebook citations and evidence checks still control the answer.
