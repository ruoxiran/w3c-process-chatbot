# W3C Process Chatbot

A scope-gated chat assistant for the W3C Process Document and Guidebook. It
only answers W3C standards-workflow questions, grounds every claim in
allowlisted sources, and exposes the full retrieval/workflow trace next to
each answer.

It is not an open chatbot. It is a deterministic workflow with the model in
a constrained synthesis role.

## Layout

```text
apps/api      FastAPI backend, workflow, retrieval, eval harness
apps/web      Next.js chat UI with a workflow / sources / entities inspector
packages/ui   Shared W3C design tokens
data/         Corpus, compiled spec pages, runtime caches (gitignored)
deploy/       Docker Compose for local + RAG infra stack
scripts/      Ingestion, embeddings cache, evaluation utilities
```

## Quick start

```bash
cp .env.example .env
python3 -m venv .venv && source .venv/bin/activate
pip install -r apps/api/requirements.txt
uvicorn app.main:app --reload --app-dir apps/api
```

In another terminal:

```bash
cd apps/web && npm install && npm run dev
```

Open <http://localhost:3000>.

The API and web both start without a model or vector index; the workflow
returns grounded fallback answers and scope refusals. Wire up a provider via
`.env` to get full LLM generation.

## Model providers

`LLM_PROVIDER` picks the backend:

| Provider           | When to use                                              |
|--------------------|----------------------------------------------------------|
| `ollama`           | Local model via Ollama (`LLM_MODEL`, e.g. `qwen3:8b`)    |
| `openai-compatible`| Any OpenAI-shaped `/v1/chat/completions` endpoint        |
| `openai`           | OpenAI                                                   |
| `openrouter`       | OpenRouter                                               |
| `template`         | No LLM, deterministic template only                      |

For an OpenAI-compatible provider:

```env
LLM_PROVIDER=openai-compatible
OPENAI_COMPATIBLE_BASE_URL=https://api.openai.com/v1
OPENAI_COMPATIBLE_API_KEY=...
OPENAI_COMPATIBLE_MODEL=gpt-4.1
```

The model only synthesizes language. Process and Guidebook citations
and the evidence checks still gate what the answer can claim.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md). Short version:

1. **Scope classifier** rejects anything that isn't a W3C Process workflow
   question.
2. **Task planner + entity resolver** figure out the intent and resolve any
   specifications or groups against the public W3C API.
3. **Retrieval** runs against a local corpus of Process, Guidebook, and
   related policy text, optionally augmented by compiled per-spec context
   and read-only snippets from official W3C GitHub repos.
4. **Evidence check** verifies the right sources are present before the
   model is allowed to synthesize.
5. **Answer generator** is the model — bounded by the prompt and the
   retrieved chunks.
6. **Citation + injection checks** validate the result before it leaves the
   workflow.

Every step is logged to a workflow trace that the UI surfaces alongside the
answer.

## Safety

- User input is untrusted; conversation history is used only to resolve
  references in follow-ups, never as authority.
- Sources are allowlisted by domain (`SOURCE_ALLOWLIST`); nothing else can
  be cited.
- W3C Process citations are normative; Guidebook citations are guidance;
  W3C API + GitHub + compiled context are grounding hints, not authority.
- Out-of-scope questions return a short refusal.

## Docker Compose

```bash
cp .env.example .env
docker compose -f deploy/docker-compose.yml up --build
```

The compose stack includes Postgres, Redis, Qdrant, and a placeholder vLLM
service profile (start it only on a GPU host).

## Evaluation

Structured offline harness (deterministic, no LLM required):

```bash
curl -s -X POST http://127.0.0.1:8000/eval/run
```

It checks intent classification, source families, citation URLs, entity
grounding, compiled-context use, required answer terms, and forbidden
misgrounding terms. The web UI exposes the same run under the `Quality`
tab.

LLM-as-judge run (requires a provider configured):

```bash
curl -s -X POST http://127.0.0.1:8000/eval/llm-judge
```

## Dense retrieval (optional)

The retriever defaults to BM25 + TF-IDF on the local corpus. To enable
dense retrieval with Ollama embeddings:

```bash
ollama pull qwen3-embedding:4b
./.venv/bin/python scripts/build_embedding_cache.py \
  --model qwen3-embedding:4b --resume
```

```env
RETRIEVAL_DENSE_ENABLED=true
OLLAMA_EMBEDDING_MODEL=qwen3-embedding:4b
```

If the cache or embedding model is missing, retrieval falls back to lexical.

## Tests

```bash
cd apps/api && python -m pytest -q
cd apps/web && npm run lint
```
