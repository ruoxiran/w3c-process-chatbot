# W3C Process Assistant — System Overview

Last updated: 2026-05-23

This document is the canonical end-to-end reference for the W3C Process Chatbot.
It covers what the system does, how the request pipeline works, what every
component contributes, where data lives, how quality is measured, and the
current path to production.

For the architectural principles in isolation, see [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md).
For setup and quickstart, see [README.md](README.md).

---

## 1. Executive Summary

The W3C Process Assistant is a citation-grounded chatbot that helps W3C
contributors navigate the W3C Process Document, the Guidebook ("Art of
Consensus"), and the surrounding standards workflow. It is **not** an open
chatbot — it is a deterministic workflow engine that only answers W3C
process/workflow questions, only cites authoritative sources, and exposes the
full reasoning trace through the UI so any answer can be audited.

The system is built around five non-negotiable invariants:

1. **Scope-gated** — only W3C Process / Guidebook / workflow questions are
   answered. Everything else is politely refused.
2. **Source-grounded** — every procedural claim must cite a chunk that came
   from an allowlisted W3C source (Process, Guidebook, related policy, or the
   official W3C GitHub orgs).
3. **Authority-ranked** — when sources disagree, Process wins; Guidebook is
   labelled as practical guidance; W3C API / GitHub / compiled markdown are
   non-normative grounding only.
4. **Untrusted user input** — the user can never override the Process by
   saying so. Prompt-injection attempts are detected and audited.
5. **Inspectable** — every response carries a `workflow_trace`, source list,
   resolved entities, evidence coverage, and audit blob. The UI surfaces them
   in a side panel.

The system is designed to run with a local LLM (Ollama by default) but can be
pointed at any OpenAI-compatible endpoint.

---

## 2. High-Level Architecture

```
                ┌──────────────────────────────────────────────┐
                │              Next.js Web UI                  │
                │  /chat conversation + workflow inspector +   │
                │  citations + 👍/👎 feedback                  │
                └──────────────────────────────────────────────┘
                                    │ HTTPS / JSON
                                    ▼
   ┌──────────────────────────────────────────────────────────────┐
   │                      FastAPI service                          │
   │  /chat /classify /feedback /eval/run /eval/llm-judge ...      │
   └──────────────────────────────────────────────────────────────┘
                                    │
       ┌────────────────────────────┼────────────────────────────┐
       ▼                            ▼                            ▼
 ┌─────────────┐           ┌────────────────┐         ┌──────────────────┐
 │  Workflow   │           │  Local corpus  │         │  External APIs   │
 │  (deterministic │       │  + indexes     │         │  (W3C, GitHub,    │
 │   nodes)    │           │  + compiled MD │         │   Ollama, live    │
 │             │           │  + feedback    │         │   page fetch)     │
 └─────────────┘           └────────────────┘         └──────────────────┘
```

Everything inside the FastAPI service is deterministic. The only place an LLM
makes a free-form decision is in:

- The optional Layer-3 scope router (for ambiguous questions)
- The final answer generation step (constrained by the prompt + citations)
- The LLM-as-judge evaluator (offline only)

---

## 3. Authority Hierarchy

When any two sources conflict, the higher row wins.

| Rank | Source | Treated as |
|---|---|---|
| 1 | W3C Process Document | Normative |
| 2 | W3C Guidebook | Practical guidance (non-normative) |
| 3 | W3C API entity / status data | Grounding only |
| 4 | Official GitHub repositories (w3c/, w3ctag/, w3cping/) | Draft/operational context |
| 5 | Compiled markdown spec context | Derivative orchestration context |
| 6 | Live web page fetch | Supporting reference, may be more current but not pre-verified |
| 7 | Conversation history | Only used to resolve "this", "that transition", follow-ups |
| 8 | User claims | **Never** treated as authoritative |

The LLM is instructed to follow this hierarchy explicitly in the system prompt.

---

## 4. End-to-End Request Pipeline

The entry point is `POST /chat`. The pipeline is implemented in
[`apps/api/app/workflows/chat_workflow.py`](apps/api/app/workflows/chat_workflow.py)
as a sequence of explicit nodes, each producing a `WorkflowStep` that surfaces
in the response trace.

```
User question
   │
   ▼
[1] Contextual query rewriter  — resolve "this/that" against last 8 turns
   │
   ▼
[2] Three-layer scope classifier
       L1: keyword match (services/scope.py)
       L2: contextual query re-classification (for follow-ups)
       L3: LLM router for weak (<0.7) or no-match cases
   │
   ▼ (if out of scope, short-circuit to refusal)
   │
[3] Task planner — extract intent, stage, spec/group, needed sources
   │
   ▼
[4] W3C API resolver — find specification / group entities
   │
   ▼
[5] GitHub draft context resolver — official W3C-org repos only
   │
   ▼
[6] Compiled markdown context resolver — pre-built spec dossiers
   │
   ▼
[7] Retriever — hybrid BM25 + TF-IDF cosine + optional dense
       augmented with entity, draft, task plan, router hints
   │
   ▼
[8] Evidence coverage check
       │
       ▼ (if needs_more_evidence, run targeted second retrieval)
   │
[9] Process state extractor — current_stage / target_stage / risk_flags
   │
   ▼
[10] Optional live page fetch — only if evidence is insufficient
       (controlled by LIVE_FETCH_ENABLED)
   │
   ▼
[11] Answer generator
       provider=ollama         → local LLM call
       provider=openai-compat. → OpenAI/OpenRouter/vLLM
       provider=template       → deterministic fallback (no LLM)
   │
   ▼
[12] Citation check + injection guard
   │
   ▼
Structured ChatResponse (answer + citations + workflow_trace + audit)
```

---

## 5. Component Catalogue

### 5.1 Scope Classifier — [`services/scope.py`](apps/api/app/services/scope.py)

Three layers, each with explicit confidence:

| Layer | What it does | Confidence |
|---|---|---|
| L1 keyword match | Match against ~70 W3C terms (`PROCESS_TOPICS`) | 0.9 for strong match (`fpwd`, `cr`, `formal objection`, `charter`...), 0.5 for weak (`w3c`, `process`...) |
| L2 contextual re-classify | Re-run L1 on the contextual query if the raw message failed | inherits L1 confidence |
| L3 LLM router | Calls the configured LLM with a strict JSON-only prompt | model-provided confidence |

The LLM router runs **for both** missed keyword matches and weak matches
(confidence < 0.7). A weak match can be **overridden to out-of-scope** if the
router says so with confidence ≥ `LLM_ROUTER_MIN_CONFIDENCE`. This eliminates
the false-positive class where "Tell me a joke about w3c" used to slip
through.

Injection detection is independent and runs on every message regardless of
scope outcome (`INJECTION_PATTERNS`).

### 5.2 Task Planner — [`services/task_planner.py`](apps/api/app/services/task_planner.py)

Produces a `TaskPlan` with `intent_type` (one of ~28 categories like
`advance_specification`, `horizontal_review`, `handle_objection_or_appeal`,
`charter_or_recharter`), `current_stage`, `target_stage`, `needed_sources`,
`search_queries`, and `risk_flags`. The plan steers retrieval and shapes the
answer.

### 5.3 W3C API Resolver — [`services/w3c_api.py`](apps/api/app/services/w3c_api.py)

Calls `https://api.w3.org` to resolve specifications and groups mentioned in
the question. Cached on disk (`data/cache/w3c_api_cache.json`, TTL 6h by
default). Returns `W3CEntity` objects with shortname, status, charter, deliverers,
team contacts.

**Treated as grounding, not authority.** An entity match doesn't grant the
right to make claims; it just helps retrieval find the right Process sections.

### 5.4 GitHub Draft Context — [`services/github_context.py`](apps/api/app/services/github_context.py)

Reads repos under the allowed orgs (`w3c,w3ctag,w3cping` by default). Pulls
README, recent commits, open issue counts, and limited file snippets. Used to
ground "what's the draft state" questions without ever treating the repo as
normative.

Per-instance cache (TTL 6h). Token via `GITHUB_TOKEN` env var.

### 5.5 Compiled Context — [`services/compiled_context.py`](apps/api/app/services/compiled_context.py)

Pre-built per-spec markdown dossiers in
[`data/compiled/spec/`](data/compiled/spec/). Each file is a hand-curated
summary of a specific specification's current state, next-step candidates,
horizontal-review signals, and charter signals. Only loaded when an entity
with confidence ≥ `COMPILED_CONTEXT_MIN_ENTITY_CONFIDENCE` is resolved.

This is the "orchestration" layer — it shapes what the answer focuses on, but
cannot replace Process / Guidebook citations.

### 5.6 Retriever — [`rag/retriever.py`](apps/api/app/rag/retriever.py)

Hybrid scorer over [`data/corpus/chunks.jsonl`](data/corpus/chunks.jsonl) (5,879
chunks):

```
rerank = bm25
       + semantic * 18      (TF-IDF cosine)
       + dense * weight     (optional Ollama embeddings)
       + topic_bonus        (workflow-aware: horizontal review, charter, ...)
       + source_priority    (process=4, guide=2)
       + quality_bonus      (chunk-quality score)
       + heading_overlap    (capped at 10)
       + relevance_adjustment
```

BM25 tuning: `k1=1.45, b=0.72`. Early-exit when BM25 and dense both ≤ 0.

After scoring, [`_balanced_hits`](apps/api/app/rag/retriever.py) guarantees up to 4 process + 2
guide hits, and [`_ensure_topic_coverage`](apps/api/app/rag/retriever.py) injects
required Guidebook entry-point pages based on the question topic
(horizontal review, charter, transitions, ...).

Paths are resolved by [`core/paths.py`](apps/api/app/core/paths.py) so the
retriever works whether uvicorn is launched from the repo root or
`apps/api/`.

### 5.7 Evidence Coverage Checker — [`services/evidence.py`](apps/api/app/services/evidence.py)

After the first retrieval pass, classifies coverage as `sufficient`,
`needs_more_evidence`, or `insufficient`. If `needs_more_evidence`, the
workflow runs **targeted second retrieval** using the queries the checker
proposed. This is one of the highest-value nodes for accuracy.

### 5.8 Process State Extractor — [`services/process_state.py`](apps/api/app/services/process_state.py)

Extracts a `ProcessState` (current/target stage, group type, deliverable type,
likely workflow, missing information, risk flags). Used to keep the answer
focused on the user's actual situation.

### 5.9 Live Page Fetch — [`services/live_fetch.py`](apps/api/app/services/live_fetch.py)

Off by default (`LIVE_FETCH_ENABLED=false`). When on and evidence coverage is
`insufficient`, fetches the primary citation URL, strips HTML, and passes the
text to the LLM as `supplementary_context` — clearly labelled as non-normative.
Useful when the corpus is stale relative to a recent W3C page update.

### 5.10 Answer Generator

Three providers, selected by `LLM_PROVIDER`:

| Provider | Backend | Use case |
|---|---|---|
| `ollama` | Local Ollama on `OLLAMA_BASE_URL` | Default for development and member-only alpha |
| `openai-compatible` (also `openai`, `openrouter`) | Any OpenAI-format `/v1/chat/completions` | When you need stronger reasoning |
| `template` | Deterministic answer composed from citations only | Failure-safe fallback; eval harness uses this for repeatability |

The LLM prompt (see [`services/ollama.py::_build_prompt`](apps/api/app/services/ollama.py))
enforces:

- per-claim citation labels `[S1]`, `[S2]`, ...
- separation between Process (normative) and Guidebook (practice guidance)
- no exposure of the system prompt
- no acceptance of user-provided process claims
- output format: short conclusion → 3-5 bullets → optional clarifying note
- internal reasoning is allowed inside `<think>...</think>` tags; the harness
  strips them via `THINKING_BLOCK_RE` before returning

If the LLM returns empty text, the workflow falls back to the deterministic
template answer.

### 5.11 Citation Check + Injection Guard

The final node verifies that an in-scope answer has at least one citation
(`require_citations=true` by default). The injection guard flags responses to
messages that contained any pattern from `INJECTION_PATTERNS` (Chinese and
English) and adds a `safety_note` audit entry.

---

## 6. API Surface

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `POST` | `/chat` | Main entry point; returns `ChatResponse` |
| `POST` | `/classify` | Scope-only classifier (no retrieval) |
| `GET` | `/models` | List available models from the configured provider |
| `GET` | `/sources/status` | Allowlist, source versions, member-only flag |
| `GET` | `/compiled/status` | Which compiled spec contexts are loaded |
| `POST` | `/compiled/rebuild` | Rebuild compiled contexts (optional `shortnames` param) |
| `POST` | `/refresh-index` | Rebuild preview index from corpus |
| `POST` | `/eval/run` | Structural eval (50 + optional 30 adversarial). Add `?include_adversarial=true` |
| `POST` | `/eval/llm-judge` | LLM-as-judge over the live LLM-generated answers |
| `POST` | `/feedback` | Persist a thumbs up/down + optional comment + audit trail |
| `GET` | `/feedback/stats` | Aggregate feedback counters |

Full OpenAPI at `/openapi.json` / `/docs` / `/redoc`.

---

## 7. Data Layer

```
data/
├── corpus/
│   ├── chunks.jsonl              # 5,879 indexed Process+Guidebook chunks
│   └── manifest.json             # ingestion provenance
├── compiled/spec/
│   ├── adapt-symbols.md          # hand-curated spec dossier
│   ├── css-grid-1.md
│   └── webauthn-3.md
├── cache/
│   ├── w3c_api_cache.json        # 6h TTL
│   └── retrieval_embeddings.jsonl # optional dense cache
└── feedback/
    └── feedback.jsonl            # append-only thumbs up/down records
```

### 7.1 Corpus format

Each line is one chunk with: `source_url`, `source_type` (`process`/`guide`/`related_policy`/`repo`),
`title`, `heading_path`, `section_id`, `text`, `commit_sha`, `published_version_date`, `quality_score`.

Built by [`scripts/import_w3c_sources.py`](scripts/import_w3c_sources.py),
which crawls:

- `https://www.w3.org/policies/process/` + the `w3c/process` GitHub repo
- `https://www.w3.org/guide/` (depth 4, 500-page cap) + the `w3c/guide` GitHub repo

### 7.2 Path resolution

All data paths in `Settings` are repo-relative defaults like
`data/corpus/chunks.jsonl`. [`core/paths.py::resolve_data_path`](apps/api/app/core/paths.py)
walks up from the calling module to find the project root, so the same
config works from any working directory.

---

## 8. Quality Assurance

Three independent layers, each measuring something different.

### 8.1 Structural eval — 80 cases

[`apps/api/app/evals/runner.py`](apps/api/app/evals/runner.py) checks:

- in-scope verdict matches expectation
- task plan `intent_type` matches
- citation URLs include expected substrings
- expected/forbidden answer terms
- entity shortname matches
- compiled-context usage matches
- response confidence ≥ floor

The 80-case golden set is split:

| Set | Count | Pass rate (template mode) |
|---|---|---|
| Original golden cases | 50 | **100%** ✅ |
| Adversarial / compound / detail-correctness | 30 | ~50% — many are designed for LLM-judge mode |

Adversarial cases live in
[`evals/adversarial_cases.py`](apps/api/app/evals/adversarial_cases.py) and
cover four categories: adversarial (prompt-injection, fake-section
fabrication, role-reset, prompt-leak attempts), compound (multi-section
reasoning), detail-correctness (high-stakes facts that must not be
hallucinated), and scope-boundary edge cases.

Run via `curl -X POST 'http://127.0.0.1:8000/eval/run?include_adversarial=true'`
or from the **Quality** tab in the web UI.

### 8.2 LLM-as-judge — answer quality

[`apps/api/app/evals/llm_judge.py`](apps/api/app/evals/llm_judge.py) runs the
**live LLM-backed workflow** for each case, then asks a judge LLM (Ollama by
default) to score the actual generated answer along four dimensions, each
0–5:

| Axis | What it measures |
|---|---|
| **accuracy** | Are the Process facts correct? Penalize invented sections, durations, versions, roles. |
| **groundedness** | Does every procedural claim trace to a cited excerpt? Penalize unsupported claims. |
| **relevance** | Does the answer address the question? Penalize off-topic digressions. |
| **harm_avoidance** | Does the answer avoid actively harmful guidance (confirming fake Process, leaking system prompt, accepting user "new Process" claims)? |

Pass threshold: average ≥ 3.5 / 5.

Run via:

```bash
python scripts/run_llm_judge.py --print-failures --output reports/judge.json
python scripts/run_llm_judge.py --tag adversarial --tag fabrication
curl -X POST 'http://127.0.0.1:8000/eval/llm-judge'
```

Expect 1–3 minutes per case (one workflow call + one judge call).

### 8.3 User feedback loop

The chat UI shows 👍 / 👎 buttons under every assistant response. 👎 expands
a comment box. Each submission persists to
[`data/feedback/feedback.jsonl`](data/feedback/feedback.jsonl) with:

- rating, optional comment
- question + answer text
- conversation_id, message_id, model
- in-scope flag, confidence, citation URLs
- full audit blob (workflow_trace, source_version, evidence_coverage, process_state)

The full audit means a 👎 record by itself is enough to diagnose a regression
without replaying the conversation. Stats are exposed at `/feedback/stats`.

This file is the ground-truth source we will use to:

1. Triage regressions in alpha
2. Promote frequently-flagged questions into the adversarial eval set
3. Tune retrieval / prompt / classifier thresholds based on real demand

### 8.4 pytest

63 tests in [`apps/api/tests/`](apps/api/tests/). Run with `pytest apps/api`.
Covers scope classifier, retriever, github_context, w3c_api, workflow,
evidence coverage, process state, task planner, scope confidence tiers, and
service path resolution.

---

## 9. Configuration

Everything is `.env`-driven. See [`.env.example`](.env.example) for the full
list. The most operationally important keys:

| Key | Default | What it controls |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` / `openai-compatible` / `template` |
| `LLM_MODEL` | `qwen3:8b` | Default chat + judge model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama endpoint |
| `LLM_ROUTER_ENABLED` | `true` | Layer-3 scope router |
| `LLM_ROUTER_MIN_CONFIDENCE` | `0.55` | Confidence floor for accepting router verdict |
| `LIVE_FETCH_ENABLED` | `false` | Real-time URL fetch when evidence is insufficient |
| `LIVE_FETCH_MAX_CHARS` | `3500` | Max bytes from fetched page |
| `COMPILED_CONTEXT_ENABLED` | `true` | Use pre-built spec dossiers |
| `COMPILED_CONTEXT_MIN_ENTITY_CONFIDENCE` | `0.7` | Only load compiled context for high-confidence entities |
| `RETRIEVAL_DENSE_ENABLED` | `false` | Use Ollama embeddings cache (requires pre-built `data/cache/retrieval_embeddings.jsonl`) |
| `REQUIRE_CITATIONS` | `true` | Refuse to return in-scope answer without citations |
| `SOURCE_ALLOWLIST` | `w3.org,api.w3.org,github.com/w3c,...` | Trust boundary for retrieval URLs |
| `FEEDBACK_LOG_PATH` | `data/feedback/feedback.jsonl` | JSONL append target |
| `CORS_ALLOW_ORIGINS` | `http://localhost:3000,...` | Set to W3C domain for production |

---

## 10. Deployment

### 10.1 Local development

```bash
# Terminal 1 — API
cp .env.example .env
python3 -m venv .venv && source .venv/bin/activate
pip install -r apps/api/requirements.txt
uvicorn app.main:app --reload --app-dir apps/api --port 8000

# Terminal 2 — Web
cd apps/web
npm install
npm run dev   # serves http://localhost:3000

# Optional — Ollama (for LLM mode)
ollama pull qwen3:8b
ollama serve
```

### 10.2 Docker Compose

```bash
docker compose -f deploy/docker-compose.yml up --build
```

Services: `api` (8000), `web` (3000), `postgres`, `redis`, `qdrant`. The
`vllm` service is profile-gated for GPU hosts.

### 10.3 Production gaps before W3C member-only release

Open before W3C-member-only deployment:

- [ ] **Authentication** — `/chat`, `/feedback`, `/eval/*` are currently open.
      Plan to integrate W3C SSO at the reverse proxy.
- [ ] **Rate limiting** — single `/chat` call may run an LLM router + main
      generation; ~30s + 1k+ tokens. A few concurrent abusers can saturate.
- [ ] **HTTPS + headers** — docker-compose exposes HTTP. Need
      nginx/caddy reverse proxy with TLS, HSTS, CSP, X-Content-Type-Options.
- [ ] **Secrets management** — `.env` works in dev; production must source
      `OPENAI_COMPATIBLE_API_KEY`, `GITHUB_TOKEN` from W3C secret store.
- [ ] **Observability** — structured logs with request_id, prometheus
      metrics, error reporting. Currently no request log.
- [ ] **Disclaimer copy** — UI must clearly state "AI-assisted; W3C Process
      Document is authoritative."
- [ ] **CORS narrowing** — `CORS_ALLOW_ORIGINS` is localhost-only by default.

Accuracy is **not** in the blocker list anymore — section 8 covers that
ground (structural eval + LLM-as-judge + closed-loop feedback).

---

## 11. Known Limitations

- **Corpus is a snapshot.** Without `LIVE_FETCH_ENABLED`, the system answers
  against the chunks indexed at corpus-build time. Run
  `/refresh-index` periodically.
- **Embeddings are optional.** Dense retrieval requires a pre-built cache
  via `scripts/build_embedding_cache.py`. Without it, retrieval is BM25 +
  TF-IDF only.
- **Compiled contexts are hand-curated** for three specs (adapt-symbols,
  css-grid-1, webauthn-3). Expanding the dossier set is manual.
- **LLM-judge takes minutes per case.** Don't run it on every PR; use it
  before a release and on a cron.
- **Adversarial cases mostly need LLM-judge mode.** They are structurally
  hard to pass in `template` mode because the structural runner checks for
  specific tokens that only the LLM produces.
- **Web UI conversation is page-local.** Refreshing clears history. Server-
  side conversation persistence is not implemented.

---

## 12. Where to look in the code

| If you want to change... | Look at... |
|---|---|
| What "in scope" means | [`services/scope.py`](apps/api/app/services/scope.py) |
| Which intent categories exist | [`services/task_planner.py`](apps/api/app/services/task_planner.py) |
| Retrieval scoring weights | [`rag/retriever.py`](apps/api/app/rag/retriever.py) (module-level constants) |
| LLM prompt | [`services/ollama.py::_build_prompt`](apps/api/app/services/ollama.py) |
| Workflow ordering | [`workflows/chat_workflow.py::ChatWorkflow.run`](apps/api/app/workflows/chat_workflow.py) |
| Add a new eval case | [`evals/cases.py`](apps/api/app/evals/cases.py) or [`evals/adversarial_cases.py`](apps/api/app/evals/adversarial_cases.py) |
| Change judge scoring | [`evals/llm_judge.py`](apps/api/app/evals/llm_judge.py) |
| Add a new API endpoint | [`apps/api/app/main.py`](apps/api/app/main.py) |
| UI feedback widget | [`apps/web/components/ChatInterface.tsx::FeedbackControls`](apps/web/components/ChatInterface.tsx) |
| API client (frontend) | [`apps/web/lib/api.ts`](apps/web/lib/api.ts) |
| Data path defaults | [`apps/api/app/core/config.py`](apps/api/app/core/config.py) + [`apps/api/app/core/paths.py`](apps/api/app/core/paths.py) |

---

## 13. Related Documents

- [README.md](README.md) — quickstart, env, useful commands
- [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) — principles, data flow diagram, authority model
- [PROJECT_MEMORY.md](PROJECT_MEMORY.md) — decisions log, completed work, TODOs
- [PLAN.md](PLAN.md) — evolving implementation plan
- [SESSION_BOOTSTRAP.md](SESSION_BOOTSTRAP.md) — minimal context for new agent sessions
