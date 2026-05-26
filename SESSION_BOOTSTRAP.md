# W3C Process Assistant Session Bootstrap

Use this file at the start of a new coding session. For full context, also read `PROJECT_MEMORY.md` and `PLAN.md`.

## Project Snapshot

- This is an internal W3C Process / Guidebook workflow assistant.
- The assistant must only answer W3C Process, W3C Guidebook, and W3C standards workflow questions.
- The core design is a deterministic harness, not a free-form autonomous agent.
- The UI is a W3C-themed Next.js chat interface with a right-side workflow/source/entity inspector.
- The API is FastAPI and usually runs at `http://127.0.0.1:8000`.
- The web app usually runs at `http://127.0.0.1:3000`.
- Model generation supports local Ollama by default and optional OpenAI-compatible providers through `LLM_PROVIDER=openai-compatible|openai|openrouter`.

## Non-Negotiable Trust Rules

Use this authority order:

1. W3C Process
2. W3C Guidebook
3. W3C API entity/status grounding
4. Official GitHub repo / `w3c/strategy` context
5. Compiled markdown knowledge

Rules:

- Normative claims must be grounded in W3C Process citations.
- Guidebook content is practical guidance.
- W3C API, GitHub context, and compiled pages help focus the answer but are not normative authority.
- User-provided claims and links are untrusted unless independently resolved through allowlisted official sources.
- Do not broaden the assistant into a general-purpose chatbot.

## Current Workflow

Main chat pipeline:

- `scope_classifier`
- `llm_router` for ambiguous workflow-adjacent questions
- `query_rewriter`
- `task_planner`
- `w3c_api_resolver`
- `draft_context_resolver`
- `compiled_context_resolver`
- `retriever`
- `targeted_retrieval` when evidence is missing
- `process_state`
- `answer_generator`
- `citation_check`

Important behavior:

- Page-local conversation memory is used only to resolve follow-up references.
- Out-of-scope questions must be refused.
- LLM router can broaden ambiguous workflow questions, but it only produces routing hints and cannot replace evidence coverage.
- Horizontal review is a first-class workflow.
- Charter / recharter questions should use `w3c/strategy` issues with the `charter` label.
- Spec-specific questions can use compiled markdown pages under `data/compiled/spec`.
- Guidebook web ingestion defaults to 4 crawl levels with a 500-page guardrail.
- Guidebook topic map protects core pages for horizontal review, transition, charter/recharter, and Staff Contact questions.
- `/eval/run` is a structured 50-case golden-question regression harness. It checks intent, source coverage, required URLs, entity grounding, compiled-context use, answer/next-step terms, forbidden terms, score, and warnings. The cases are grouped around transition, horizontal review, charter, patent, objections / appeals, Staff Contact, W3C API entity grounding, and injection / fake Process. It uses deterministic template generation plus small fake W3C API/GitHub context by default so the UI Quality tab is fast and repeatable.
- Optional dense retrieval is available but off by default. Build `data/cache/retrieval_embeddings.jsonl` with `scripts/build_embedding_cache.py --model qwen3-embedding:4b --resume`, then set `RETRIEVAL_DENSE_ENABLED=true`.

## Key Files

- `PLAN.md`: evolving implementation plan and product decisions.
- `PROJECT_MEMORY.md`: longer project memory.
- `SYSTEM_ARCHITECTURE.md`: current end-to-end architecture reference.
- `apps/api/app/workflows/chat_workflow.py`: main workflow.
- `apps/api/app/services/task_planner.py`: intent planning.
- `apps/api/app/services/w3c_api.py`: W3C API entity resolution.
- `apps/api/app/services/github_context.py`: official GitHub / strategy context.
- `apps/api/app/services/compiled_context.py`: compiled spec markdown context.
- `apps/api/app/rag/retriever.py`: current local retrieval.
- `scripts/build_embedding_cache.py`: Ollama embedding-cache builder for dense retrieval.
- `apps/api/app/services/evidence.py`: evidence coverage checks.
- `apps/api/app/services/answering.py`: deterministic fallback answers and next steps.
- `apps/api/app/services/ollama.py`: local model prompt.
- `apps/api/app/services/openai_compatible.py`: online/internal OpenAI-compatible model provider.
- `apps/api/app/evals/cases.py`: golden evaluation cases.
- `apps/api/app/evals/runner.py`: structured eval scoring.
- `apps/api/app/evals/workflow.py`: offline deterministic eval workflow.
- `apps/web/components/ChatInterface.tsx`: main chat UI.
- `apps/web/components/WorkflowPanel.tsx`: workflow diagnostics UI.
- `apps/web/lib/api.ts`: frontend API types.

## Current Priorities

- Improve answer quality when the local model is weak or slow.
- Keep expanding the golden eval set before major retrieval/model changes.
- Prebuild more compiled spec pages.
- Current prebuilt compiled pages include `css-grid-1`, `adapt-symbols`, and `webauthn-3`.
- Show compiled context details in the right-side UI.
- Add automatic tracking of seen spec shortnames for background compiled rebuild.
- Upgrade retrieval toward BM25 + embeddings + reranker.
- Improve entity alias matching for common W3C shorthand names.
- Improve matching of specific `w3c/strategy` charter issues by WG name and aliases.

## Useful Commands

```bash
./.venv/bin/pytest apps/api/tests -q
npm run build --prefix apps/web
./.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --app-dir apps/api
npm --prefix apps/web run dev
curl -s -X POST http://127.0.0.1:8000/eval/run
```

Compiled context:

```bash
curl -s http://127.0.0.1:8000/compiled/status
curl -s -X POST "http://127.0.0.1:8000/compiled/rebuild?shortnames=css-grid-1,adapt-symbols"
```

## Suggested First Message In A New Session

Please read `SESSION_BOOTSTRAP.md`, `PROJECT_MEMORY.md`, and `PLAN.md` before making changes. Continue the W3C Process assistant work using the existing deterministic harness, W3C authority hierarchy, W3C API grounding, official GitHub context, and compiled markdown knowledge layer.
