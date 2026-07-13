# W3C Process Chatbot

A chat assistant for the W3C Process Document and Guidebook. It only answers
questions about the W3C standards workflow, and every answer is built from
allowlisted sources with the retrieval and workflow trace shown next to it.

It is not an open chatbot. The model does not answer on its own. It sits
inside a fixed workflow and only writes the final text from the sources the
workflow retrieved.

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
# Python 3.11 to 3.13 (3.14 not yet supported by pinned numpy/torch)
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r apps/api/requirements.txt
uvicorn app.main:app --reload --app-dir apps/api
```

In another terminal:

```bash
cd apps/web && npm install && npm run dev
```

Open <http://localhost:3000>.

The API and web both start without a model or vector index. In that state the
workflow returns grounded template answers and scope refusals. Set a provider
in `.env` to get full LLM answers.

## Model providers

`LLM_PROVIDER` picks the backend:

| Provider           | When to use                                              |
|--------------------|----------------------------------------------------------|
| `ollama`           | Local model via Ollama (`LLM_MODEL`, e.g. `qwen3:8b`)    |
| `openai-compatible`| Any OpenAI-shaped `/v1/chat/completions` endpoint        |
| `openai`           | OpenAI                                                   |
| `openrouter`       | OpenRouter                                               |
| `bedrock`          | AWS Bedrock via the boto3 Converse API (`LLM_MODEL`)     |
| `template`         | No LLM, deterministic template only                      |

OpenAI-compatible provider:

```env
LLM_PROVIDER=openai-compatible
OPENAI_COMPATIBLE_BASE_URL=https://api.openai.com/v1
OPENAI_COMPATIBLE_API_KEY=...
OPENAI_COMPATIBLE_MODEL=gpt-4.1
```

AWS Bedrock (Claude, Nova, Llama, Titan, and others through the same Converse
API):

```env
LLM_PROVIDER=bedrock
LLM_MODEL=us.anthropic.claude-sonnet-5
BEDROCK_REGION=us-east-1
BEDROCK_API_KEY=...        # generated in the Bedrock console under "API keys"
```

Auth uses a Bedrock API key (a bearer token), passed to boto3 through
`AWS_BEARER_TOKEN_BEDROCK`. The ambient AWS credential chain is not used. Two
things to watch:

- Claude models on Bedrock need a cross-region inference profile, so use the
  `us.` prefix (`us.anthropic.claude-sonnet-5`), not the bare model id. Run
  `aws bedrock list-inference-profiles` to see what your account allows.
- The key's IAM identity needs `bedrock:InvokeModel` and
  `bedrock:InvokeModelWithResponseStream`. An explicit deny in an AWS
  Organizations SCP overrides any grant, so an org admin may need to lift it.

The model only writes language. The citations and evidence checks still decide
what the answer is allowed to claim.

### Bedrock Knowledge Base retrieval (optional)

You can also pull passages from an AWS Bedrock Knowledge Base into retrieval.
KB passages join the local corpus in the candidate pool and are reranked,
grounded, and cited like any other chunk. Use it for content that only lives
in a managed KB.

```env
BEDROCK_KB_ENABLED=true
BEDROCK_KB_ID=XXXXXXXXXX
BEDROCK_KB_REGION=us-east-1        # optional; defaults to BEDROCK_REGION
BEDROCK_KB_MAX_RESULTS=8
# reuses BEDROCK_API_KEY
```

KB retrieval is independent of the generation provider and uses a different
IAM action (`bedrock:Retrieve`). It can work even where `bedrock:InvokeModel`
is blocked. If it fails, the local-corpus retrieval still runs.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md). In short:

1. The scope classifier rejects anything that is not a W3C Process workflow
   question.
2. The task planner and entity resolver work out the intent and resolve any
   specs or groups against the public W3C API.
3. Retrieval runs against a local corpus of Process, Guidebook, and related
   policy text, plus optional per-spec context and read-only snippets from
   official W3C GitHub repos.
4. The evidence check confirms the right sources are present before the model
   is allowed to write.
5. The model writes the answer, bounded by the prompt and the retrieved
   chunks.
6. Citation and injection checks validate the result before it is returned.

Every step is logged to a workflow trace that the UI shows next to the answer.

## Safety

- User input is untrusted. Conversation history is used only to resolve
  references in follow-ups, never as a source.
- Sources are allowlisted by domain (`SOURCE_ALLOWLIST`). Nothing else can be
  cited.
- Process citations are normative. Guidebook citations are guidance. W3C API,
  GitHub, and compiled context are grounding hints, not authority.
- Out-of-scope questions get a short refusal.

## Docker Compose

```bash
cp .env.example .env
docker compose -f deploy/docker-compose.yml up --build
```

The stack includes Postgres, Redis, Qdrant, and a placeholder vLLM service
profile (start it only on a GPU host).

## Evaluation

Offline harness (deterministic, no LLM needed):

```bash
curl -s -X POST http://127.0.0.1:8000/eval/run
```

It checks intent classification, source families, citation URLs, entity
grounding, compiled-context use, required answer terms, and forbidden
misgrounding terms. The web UI runs the same thing under the `Quality` tab.

LLM-as-judge run (needs a provider configured):

```bash
curl -s -X POST http://127.0.0.1:8000/eval/llm-judge
```

## Dense retrieval (optional)

The retriever defaults to BM25 + TF-IDF on the local corpus. To turn on dense
retrieval with Ollama embeddings:

```bash
ollama pull qwen3-embedding:4b
./.venv/bin/python scripts/build_embedding_cache.py \
  --model qwen3-embedding:4b --resume
```

```env
RETRIEVAL_DENSE_ENABLED=true
OLLAMA_EMBEDDING_MODEL=qwen3-embedding:4b
```

If the cache or the embedding model is missing, retrieval falls back to
lexical.

## Tests

```bash
cd apps/api && python -m pytest -q
cd apps/web && npm run lint
```
