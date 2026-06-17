# Architecture

The system is a deterministic workflow with the model in a constrained
synthesis role. Everything outside answer generation is rule-based and
inspectable. The UI shows the full trace next to every answer.

## Authority

When sources disagree, the higher row wins.

| Source | Treated as |
|---|---|
| W3C Process Document | Normative |
| W3C Guidebook | Practical guidance |
| W3C API entity / status data | Grounding only |
| Official GitHub (`w3c`, `w3ctag`, `w3cping`) | Draft / operational context |
| Compiled markdown spec dossiers | Derivative orchestration context |
| Live web page fetch | Supporting reference, may be more current |
| Conversation history | Used only to resolve "this" / "that" in follow-ups |
| User claims | Never authoritative |

The LLM prompt repeats this order so the model can't get clever.

## Request pipeline

Entry point: `POST /chat`, implemented in
[`apps/api/app/workflows/chat_workflow.py`](apps/api/app/workflows/chat_workflow.py).
Each node emits a `WorkflowStep` that the UI surfaces.

```text
question
   │
   ▼  resolve "this/that" against last 8 turns
contextual query rewriter
   │
   ▼  L1 keyword match → L2 contextual re-classify → L3 LLM router
scope classifier  ──► refusal (if out of scope)
   │
   ▼  intent, stage, spec/group, needed sources
task planner
   │
   ▼  parallel: W3C API entity resolve
            ║   GitHub draft context (only for resolved official repos)
            ║   compiled markdown dossier (per high-confidence entity)
   ▼
hybrid retriever  (BM25 + TF-IDF + optional dense, with topic / source boosts)
   │
   ▼  if needs_more_evidence → second targeted retrieval
evidence coverage check
   │
   ▼  current_stage / target_stage / risk_flags
process state extractor
   │
   ▼  optional, off by default; only when evidence is insufficient
live page fetch (allowlisted URLs)
   │
   ▼  ollama | openai-compatible | template fallback
answer generator
   │
   ▼  citation requirement + injection guard
ChatResponse  (answer + citations + workflow_trace + audit)
```

The router LLM call runs in parallel with W3C API entity resolution to keep
end-to-end latency down. GitHub draft and compiled context resolve in
parallel too.

## Scope classifier

[`services/scope.py`](apps/api/app/services/scope.py). Three layers, each
with explicit confidence:

| Layer | What it does |
|---|---|
| L1 keyword | Match against ~70 W3C terms; 0.9 for strong (`fpwd`, `cr`, `formal objection`, `charter`...), 0.5 for weak (`w3c`, `process`...) |
| L2 contextual | Re-run on the context-resolved query if L1 missed, gated to avoid scope-leak from history |
| L3 LLM router | JSON-only LLM call; can rescue misses or override weak L1 hits |

Injection detection is independent and runs on every message regardless of
scope outcome — patterns in `INJECTION_PATTERNS` (English and Chinese)
trigger a `safety_note` in the audit blob.

## Task planner

[`services/task_planner.py`](apps/api/app/services/task_planner.py).
Produces a `TaskPlan` with `intent_type` (~28 categories including
`advance_specification`, `horizontal_review`, `handle_objection_or_appeal`,
`charter_or_recharter`), stages, needed sources, search queries, risk flags.
The plan steers retrieval and shapes the answer.

## Entity and context resolvers

- **W3C API** [`services/w3c_api.py`](apps/api/app/services/w3c_api.py) —
  resolves specifications and groups against `api.w3.org`. Disk-cached
  (TTL 6h). Grounding only, not authority.
- **GitHub draft context**
  [`services/github_context.py`](apps/api/app/services/github_context.py) —
  reads README, recent commits, open issue counts, and small file snippets
  from allowed orgs (`GITHUB_CONTEXT_ALLOWED_ORGS`). Optional
  `GITHUB_TOKEN` for higher rate limits.
- **Compiled context**
  [`services/compiled_context.py`](apps/api/app/services/compiled_context.py)
  — hand-curated per-spec markdown dossiers in
  [`data/compiled/spec/`](data/compiled/spec/). Loaded only when an entity
  with confidence ≥ `COMPILED_CONTEXT_MIN_ENTITY_CONFIDENCE` is resolved.

## Retriever

[`rag/retriever.py`](apps/api/app/rag/retriever.py). Hybrid scorer over
[`data/corpus/chunks.jsonl`](data/corpus/chunks.jsonl):

```text
score = bm25
      + 18 * tfidf_cosine
      + dense_weight * dense_cosine     (optional, Ollama embeddings)
      + topic_bonus                     (horizontal review, charter, ...)
      + source_priority                 (process=4, guide=2)
      + quality_bonus
      + min(heading_overlap, 10)
      + relevance_adjustment
```

BM25 tuning: `k1=1.45, b=0.72`. Early-exit when BM25 and dense are both ≤ 0.

After scoring, the retriever guarantees up to 4 Process + 2 Guidebook hits
and injects Guidebook entry-point pages based on the question topic.

Paths resolve through
[`core/paths.py`](apps/api/app/core/paths.py) so the retriever works whether
uvicorn is launched from the repo root or `apps/api/`.

## Evidence and process state

[`services/evidence.py`](apps/api/app/services/evidence.py) classifies
coverage as `sufficient` / `needs_more_evidence` / `insufficient`. If
`needs_more_evidence`, the workflow runs a second targeted retrieval pass
with the queries the checker proposed.

[`services/process_state.py`](apps/api/app/services/process_state.py)
extracts a `ProcessState` (current/target stage, group type, deliverable
type, likely workflow, missing information, risk flags). Used to keep the
answer focused on the user's actual situation rather than the full Process
chapter.

## Answer generation

`LLM_PROVIDER` picks the backend. The prompt
([`services/ollama.py::_build_prompt`](apps/api/app/services/ollama.py))
enforces:

- per-claim citation labels (`[S1]`, `[S2]`, ...)
- separation between Process (normative) and Guidebook (guidance)
- no exposure of the system prompt
- no acceptance of user-supplied Process claims
- output shape: short conclusion → 3–5 bullets → optional clarifying note
- `<think>...</think>` is allowed for internal reasoning and stripped
  before return

Empty LLM output falls back to the deterministic template answer composed
from citations.

## Citation and injection guard

The final node verifies an in-scope answer has at least one citation
(`REQUIRE_CITATIONS=true` by default). The injection guard flags any
response where the original message matched an injection pattern and adds
a `safety_note` to the audit blob.

## API surface

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness |
| `POST` | `/chat` | Main chat entry |
| `POST` | `/classify` | Scope-only (no retrieval) |
| `GET` | `/models` | Provider's model list |
| `GET` | `/sources/status` | Allowlist + source versions |
| `GET` | `/compiled/status` | Loaded compiled dossiers |
| `POST` | `/compiled/rebuild` | Rebuild compiled dossiers |
| `POST` | `/refresh-index` | Rebuild preview index from corpus |
| `POST` | `/eval/run` | Structural eval (50 + 30 adversarial) |
| `POST` | `/eval/llm-judge` | LLM-as-judge over live workflow |
| `POST` | `/feedback` | Thumbs up/down + optional comment |
| `GET` | `/feedback/stats` | Aggregate feedback counters |

OpenAPI at `/openapi.json`, `/docs`, `/redoc`.

## Data layout

```text
data/
├── corpus/
│   ├── chunks.jsonl              indexed Process + Guidebook chunks
│   └── manifest.json             ingestion provenance
├── compiled/spec/
│   ├── adapt-symbols.md          hand-curated spec dossier
│   ├── css-grid-1.md
│   └── webauthn-3.md
├── cache/
│   ├── w3c_api_cache.json        6h TTL
│   └── retrieval_embeddings.jsonl  optional dense cache
└── feedback/
    └── feedback.jsonl            append-only thumbs up/down records
```

Each chunk carries `source_url`, `source_type`
(`process`/`guide`/`related_policy`/`repo`), `title`, `heading_path`,
`section_id`, `text`, `commit_sha`, `published_version_date`,
`quality_score`. Built by
[`scripts/import_w3c_sources.py`](scripts/import_w3c_sources.py) from
`www.w3.org/policies/process/`, `www.w3.org/guide/`, and the corresponding
GitHub repos.

## Evaluation

Three independent layers; each measures something different.

**Structural eval** —
[`evals/runner.py`](apps/api/app/evals/runner.py). 80 cases. Checks scope
verdict, intent classification, source families, citation URLs, entity
shortnames, compiled-context use, required/forbidden answer terms,
confidence floor. Deterministic, offline, no LLM needed. Adversarial cases
live in
[`evals/adversarial_cases.py`](apps/api/app/evals/adversarial_cases.py) and
cover prompt-injection, fabricated sections, role reset, prompt-leak
attempts, multi-section reasoning, detail-correctness, and scope-boundary
edges.

**LLM-as-judge** —
[`evals/llm_judge.py`](apps/api/app/evals/llm_judge.py). Runs the live
LLM-backed workflow on each case, then asks a judge LLM to score the
answer 0–5 on accuracy, groundedness, relevance, harm avoidance. Pass
threshold: average ≥ 3.5. Slow (1–3 min per case); use before releases or
on a cron.

**User feedback** — every assistant message has 👍/👎 in the UI. 👎
expands a comment box. Each submission lands in
[`data/feedback/feedback.jsonl`](data/feedback/feedback.jsonl) with the
full audit blob (workflow_trace, source_version, evidence_coverage,
process_state) so a single record is enough to diagnose a regression.
Stats at `/feedback/stats`.

**pytest** — 63 tests in [`apps/api/tests/`](apps/api/tests/). Covers
scope, retriever, github_context, w3c_api, workflow, evidence,
process_state, task_planner, path resolution.

## Configuration

All `.env`-driven. See [`.env.example`](.env.example) for the full list.

| Key | Default | Controls |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` / `openai-compatible` / `template` |
| `LLM_MODEL` | `qwen3:8b` | Default chat + judge model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama endpoint |
| `LLM_ROUTER_ENABLED` | `true` | Layer-3 scope router |
| `LLM_ROUTER_MIN_CONFIDENCE` | `0.55` | Confidence floor for router verdict |
| `LIVE_FETCH_ENABLED` | `false` | Fetch URL when evidence is insufficient |
| `COMPILED_CONTEXT_ENABLED` | `true` | Use pre-built spec dossiers |
| `COMPILED_CONTEXT_MIN_ENTITY_CONFIDENCE` | `0.7` | Entity confidence floor for compiled context |
| `RETRIEVAL_DENSE_ENABLED` | `false` | Use the Ollama embeddings cache |
| `REQUIRE_CITATIONS` | `true` | Refuse in-scope answer without citations |
| `SOURCE_ALLOWLIST` | `w3.org,api.w3.org,github.com/w3c,...` | Retrieval URL trust boundary |
| `CORS_ALLOW_ORIGINS` | `http://localhost:3000,...` | Browser origin allowlist |

## Production gaps

Open before any non-local deployment:

- Authentication on `/chat`, `/feedback`, `/eval/*`
- Rate limiting (one `/chat` may run a router LLM + main generation;
  abusers can saturate quickly)
- TLS / security headers behind a reverse proxy
- Secrets sourced from a real secret store, not `.env`
- Structured logs with request IDs, metrics, error reporting
- Clear UI disclaimer that the Process Document is authoritative
- Narrower `CORS_ALLOW_ORIGINS` for production

## Known limitations

- Corpus is a snapshot; refresh via `/refresh-index` or rerun
  `scripts/import_w3c_sources.py`.
- Dense retrieval requires a pre-built embeddings cache via
  `scripts/build_embedding_cache.py`. Without it, retrieval is BM25 +
  TF-IDF only.
- Compiled spec dossiers are hand-curated; expanding the set is manual.
- Web UI conversation is page-local; refreshing clears history.

## Where to change things

| To change... | Look at... |
|---|---|
| What "in scope" means | [`services/scope.py`](apps/api/app/services/scope.py) |
| Intent categories | [`services/task_planner.py`](apps/api/app/services/task_planner.py) |
| Retrieval scoring | [`rag/retriever.py`](apps/api/app/rag/retriever.py) (module constants) |
| LLM prompt | [`services/ollama.py::_build_prompt`](apps/api/app/services/ollama.py) |
| Workflow ordering | [`workflows/chat_workflow.py`](apps/api/app/workflows/chat_workflow.py) |
| Add an eval case | [`evals/cases.py`](apps/api/app/evals/cases.py) or [`evals/adversarial_cases.py`](apps/api/app/evals/adversarial_cases.py) |
| Judge scoring | [`evals/llm_judge.py`](apps/api/app/evals/llm_judge.py) |
| API endpoints | [`apps/api/app/main.py`](apps/api/app/main.py) |
| UI feedback widget | [`apps/web/components/ChatInterface.tsx`](apps/web/components/ChatInterface.tsx) |
| Frontend API client | [`apps/web/lib/api.ts`](apps/web/lib/api.ts) |
| Default paths | [`apps/api/app/core/config.py`](apps/api/app/core/config.py) + [`apps/api/app/core/paths.py`](apps/api/app/core/paths.py) |
