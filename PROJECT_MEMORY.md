# W3C Process Assistant Project Memory

Last updated: 2026-05-01  
Purpose: a compact project memory file for bootstrapping a new chat session quickly.

## 1. Quick Load Summary

- Project: internal W3C Process / Guidebook workflow assistant.
- Goal: answer only W3C Process-related workflow questions with grounded citations, guide users toward the next concrete standards-process step, and surface relevant W3C pages / repos / operational context.
- Current stack:
  - API: FastAPI
  - Web: Next.js + TypeScript
- Model provider: local Ollama by default
- Online model provider: optional OpenAI-compatible chat-completions endpoint for OpenAI, OpenRouter, vLLM, or internal gateways
- Retrieval: local corpus file + deterministic lexical/TF-IDF retriever
- Entity grounding: W3C API
- Operational context: official GitHub repos on demand
- Compiled knowledge: spec-first markdown pages under `data/compiled/spec`
- Evaluation: structured golden-question harness exposed at `/eval/run`
- Important design rule: compiled knowledge, W3C API data, and GitHub repo context are all non-normative helpers. Normative procedural claims must still be grounded in W3C Process citations. Guidebook is practical guidance, not authority over Process.

## 2. Product Boundaries

### In scope

- W3C Process
- W3C Guidebook
- W3C standards workflow
- Recommendation-track transitions
- Horizontal review
- Charter / recharter workflow
- Staff Contact / Team Contact role questions
- Draft repo context when tied to an official W3C specification

### Out of scope

- General web questions
- Non-W3C policy/process questions
- Arbitrary GitHub or third-party repos
- Fully autonomous external action execution

### Required refusal behavior

- If the user asks something outside W3C Process / Guidebook / W3C workflow scope, the assistant must refuse with a short explanation.

## 3. Core Architectural Decisions

### 3.1 Deterministic harness over free-form agents

- The system intentionally uses a deterministic workflow rather than a broad autonomous multi-agent setup.
- Primary workflow shape:
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

Why:

- Stronger safety and scope control
- Easier provenance and debugging
- Better protection against prompt injection and authority confusion

### 3.2 Authority hierarchy

Use this precedence order consistently:

1. W3C Process
2. W3C Guidebook
3. W3C API entity/status data
4. Official GitHub repo / strategy issue context
5. Compiled markdown knowledge

Interpretation:

- Process decides normative facts.
- Guidebook explains practice and workflow.
- API/GitHub/compiled context help focus retrieval and answer shape.

### 3.3 Spec-first compiled knowledge layer

- Inspired by the Karpathy-style compiled wiki pattern, but adapted as a hybrid layer instead of replacing the current workflow.
- Current choice:
  - storage format: markdown
  - coverage: spec-first only in v1
  - update strategy: background rebuild / manual rebuild endpoint
- Compiled pages live under:
  - `data/compiled/spec/<shortname>.md`

Each compiled page is intended to summarize:

- spec identity
- current public state
- likely next Process step
- Guidebook workflow hints
- horizontal review signals
- charter signals when relevant
- draft repo context summary
- provenance and freshness metadata

### 3.4 GitHub context strategy

- Do not index the whole `github.com/w3c/` universe.
- Only resolve official repo context on demand when a concrete entity is known.
- Allowed orgs:
  - `w3c`
  - `w3ctag`
  - `w3cping`

Special case:

- Charter / recharter questions use `w3c/strategy` issues with the `charter` label as tracking context.

### 3.5 Horizontal review as first-class workflow

- Horizontal review is not treated as generic review.
- It is explicitly modeled with:
  - GitHub request repos
  - `*-tracker`
  - `*-needs-resolution`
  - tracker-board checks before transitions

## 4. Current Runtime Behavior

### 4.1 Chat behavior

- Page-local memory exists inside the web UI conversation while the page stays open.
- Refreshing or closing the page clears the conversation.
- Follow-up questions are resolved using recent chat turns only as reference-resolution context, never as authoritative source material.
- Scope is now rule-first and LLM-assisted for ambiguous questions. The router only suggests intent and retrieval hints; evidence coverage still decides whether the answer is grounded.

### 4.2 Retrieval behavior

- Current local retriever uses lexical BM25-style scoring plus TF-IDF cosine similarity over the corpus file.
- Optional dense retrieval is now available through an Ollama embedding cache. Build it with `scripts/build_embedding_cache.py --model qwen3-embedding:4b --resume`, then set `RETRIEVAL_DENSE_ENABLED=true`.
- Dense retrieval fuses cached embedding cosine with BM25, sparse TF-IDF similarity, topic boosts, source priority, heading overlap, and quality adjustments. Missing cache/model automatically falls back to lexical/sparse retrieval.
- Retrieval is source-priority aware.
- Guidebook deep crawl exists and is no longer limited to the homepage. The default crawl depth is now 4, with a 500-page guardrail.
- Horizontal review retrieval has explicit boosts for key Guidebook pages.
- Guidebook workflow topic map protects core pages for horizontal review, transition, charter/recharter, and Staff Contact questions.
- For spec-specific questions, compiled spec context should now load before raw retrieval if a matching high-confidence spec entity and local compiled page exist.

### 4.3 Local model behavior

- Default model path currently points to a local Ollama model.
- The API can run without a model and still return deterministic fallback answers.
- `LLM_PROVIDER=openai-compatible|openai|openrouter` can route answer generation through `OPENAI_COMPATIBLE_BASE_URL` and `OPENAI_COMPATIBLE_MODEL`; the same source/evidence/citation harness still applies.
- Ollama generation is constrained by a prompt that distinguishes:
  - trusted Process excerpts
  - trusted Guidebook excerpts
  - W3C API grounding
  - GitHub draft context
  - compiled spec context
  - untrusted conversation history

### 4.4 Evaluation behavior

- `/eval/run` now runs structured golden cases rather than a simple scope smoke test.
- Each eval case can check expected workflow intent, required source types, URL substrings, answer terms, next-step terms, forbidden terms, entity shortname, compiled-context use, and minimum confidence.
- Current coverage includes 50 cases grouped around transition, horizontal review, charter, Patent Policy, Formal Objection / Appeal, Staff Contact, W3C API entity grounding, and injection / fake Process.
- The response includes score, passed/total counts, tags, actual intent/source/entity/URL diagnostics, confidence, and warnings.
- Current deterministic eval baseline after this update: `50/50` eval cases pass with score `1.0`.
- The web inspector has a `Quality` tab that runs `/eval/run` and displays score, failures, warnings, and all golden cases.

## 5. Important Completed Work

### Retrieval / grounding

- Added exact/direct W3C API shortname resolution before fuzzy catalog search.
- Fixed bad entity matches such as `wai-adapt symbol` resolving to unrelated accessibility specs.
- Added stopword filtering and stronger overlap rules in W3C API entity resolution.
- Added guidebook crawl depth and deeper ingestion of Guidebook content.
- Added topic-aware retrieval boosts for:
  - transitions
  - Staff Contact
  - charter
  - horizontal review

### Workflow

- Added structured `task_plan`.
- Added structured `process_state`.
- Added structured `evidence_coverage`.
- Added targeted second-pass retrieval when important evidence is missing.
- Added workflow visualization on the right-side panel in the UI.

### GitHub / live context

- Added on-demand official draft repo context resolution.
- Added charter/recharter tracking via `w3c/strategy` issues.
- Added tracking of open/closed strategy issues, timestamps, labels, and TiLT-readiness hints.
- Added handling for horizontal review labels such as:
  - `Horizontal review requested`
  - completed review labels
  - `*-needs-resolution`

### UI

- Switched the UI to English.
- Moved to a ChatGPT-like layout with conversation memory in-page and workflow inspector on the right.
- Kept W3C theme while refining toward a more polished Apple-like density and layout feel.
- Added Enter-to-submit behavior.
- Replaced logo usage with the W3C Member sub-brand asset.

### Compiled knowledge layer

- Added `CompiledContext` schema.
- Added `compiled_context_resolver` to the workflow.
- Added compiled status and rebuild endpoints.
- Added first compiled spec page support under `data/compiled/spec`.
- Added compiled-context-aware evidence coverage and prompt shaping.
- Prebuilt compiled pages currently include `css-grid-1`, `adapt-symbols`, and `webauthn-3`.

### Evaluation harness

- Added richer `EvalCase` expectations in [apps/api/app/evals/cases.py](/Users/roy/Developer/W3C-process/apps/api/app/evals/cases.py).
- Added structured scoring in [apps/api/app/evals/runner.py](/Users/roy/Developer/W3C-process/apps/api/app/evals/runner.py).
- `/eval/run` now returns score, counts, tags, actual vs expected diagnostics, confidence, and warnings.

## 6. Current To-Do / Next Priorities

### High priority

- Improve end-to-end answer quality when the local model is weak or slow.
- Expand the golden eval set before large retrieval/model changes so quality gains are measurable.
- Prebuild more compiled spec pages instead of relying on just one or a few.
- Surface compiled context details in the right-side UI, not only as an internal flag.
- Add automatic tracking of “seen spec shortnames” so chat traffic can feed background compiled rebuild targets.
- Strengthen deep linking and multi-level retrieval for Guidebook sections.

### Retrieval quality

- Move from the current local lexical/TF-IDF approach toward a stronger hybrid retrieval setup:
  - BM25
  - embeddings
  - reranker
- Improve entity alias mapping for common W3C shorthand names.
- Improve precision for spec/group references that are colloquial rather than exact shortnames.

### Live operational context

- Match specific `w3c/strategy` issues more accurately by WG name and aliases.
- When a question mentions a specific WG or shortname, combine:
  - current W3C API state
  - strategy issue state
  - relevant repo context
  - Process + Guidebook guidance

### Product / UX

- Show compiled context freshness, provenance, workflow signals, and next-step candidates in the inspector.
- Add admin visibility for compiled page status and refresh health.
- Consider a lightweight operator memory / runbook layer, but do not replace the deterministic harness.

## 7. Important Files and What They Matter For

### Root docs

- [README.md](/Users/roy/Developer/W3C-process/README.md)
  - quick-start and high-level repo overview
- [PLAN.md](/Users/roy/Developer/W3C-process/PLAN.md)
  - authoritative implementation plan and evolving design decisions
- [PROJECT_MEMORY.md](/Users/roy/Developer/W3C-process/PROJECT_MEMORY.md)
  - this session bootstrap memory file
- [SYSTEM_ARCHITECTURE.md](/Users/roy/Developer/W3C-process/SYSTEM_ARCHITECTURE.md)
  - current end-to-end architecture, data flow, authority hierarchy, and runtime workflow

### Backend workflow and reasoning

- [apps/api/app/workflows/chat_workflow.py](/Users/roy/Developer/W3C-process/apps/api/app/workflows/chat_workflow.py)
  - main deterministic chat pipeline
- [apps/api/app/services/task_planner.py](/Users/roy/Developer/W3C-process/apps/api/app/services/task_planner.py)
  - intent detection, source needs, focused retrieval queries
- [apps/api/app/services/process_state.py](/Users/roy/Developer/W3C-process/apps/api/app/services/process_state.py)
  - structured workflow/state extraction
- [apps/api/app/services/evidence.py](/Users/roy/Developer/W3C-process/apps/api/app/services/evidence.py)
  - evidence sufficiency checks and targeted retrieval triggers
- [apps/api/app/services/answering.py](/Users/roy/Developer/W3C-process/apps/api/app/services/answering.py)
  - deterministic fallback answer shaping and next-step selection
- [apps/api/app/services/ollama.py](/Users/roy/Developer/W3C-process/apps/api/app/services/ollama.py)
  - local-model prompt construction and cleanup
- [apps/api/app/services/openai_compatible.py](/Users/roy/Developer/W3C-process/apps/api/app/services/openai_compatible.py)
  - OpenAI-compatible online/internal model provider for answer generation and JSON routing

### Grounding and live context

- [apps/api/app/services/w3c_api.py](/Users/roy/Developer/W3C-process/apps/api/app/services/w3c_api.py)
  - W3C API resolution, shortname handling, caching
- [apps/api/app/services/github_context.py](/Users/roy/Developer/W3C-process/apps/api/app/services/github_context.py)
  - official GitHub draft context and `w3c/strategy` charter issue context
- [apps/api/app/services/context.py](/Users/roy/Developer/W3C-process/apps/api/app/services/context.py)
  - follow-up question resolution and entity-aware query augmentation

### Retrieval and corpus

- [apps/api/app/rag/retriever.py](/Users/roy/Developer/W3C-process/apps/api/app/rag/retriever.py)
  - current local retriever implementation, including optional dense embedding cache fusion
- [scripts/build_embedding_cache.py](/Users/roy/Developer/W3C-process/scripts/build_embedding_cache.py)
  - builds `data/cache/retrieval_embeddings.jsonl` from the corpus using an Ollama embedding model
- [scripts/import_w3c_sources.py](/Users/roy/Developer/W3C-process/scripts/import_w3c_sources.py)
  - source import / crawl pipeline
- [data/corpus/chunks.jsonl](/Users/roy/Developer/W3C-process/data/corpus/chunks.jsonl)
  - local retrieval corpus
- [data/cache/w3c_api_cache.json](/Users/roy/Developer/W3C-process/data/cache/w3c_api_cache.json)
  - persisted W3C API cache

### Compiled knowledge

- [apps/api/app/services/compiled_context.py](/Users/roy/Developer/W3C-process/apps/api/app/services/compiled_context.py)
  - compiled spec context loader / compiler / rebuild logic
- [data/compiled/spec](/Users/roy/Developer/W3C-process/data/compiled/spec)
  - compiled spec pages, currently including `css-grid-1`, `adapt-symbols`, and `webauthn-3`

### API surface

- [apps/api/app/main.py](/Users/roy/Developer/W3C-process/apps/api/app/main.py)
  - `/chat`, `/models`, `/sources/status`, `/refresh-index`, `/eval/run`, `/compiled/rebuild`, `/compiled/status`
- [apps/api/app/models/schemas.py](/Users/roy/Developer/W3C-process/apps/api/app/models/schemas.py)
  - response/request schemas including `TaskPlan`, `EvidenceCoverage`, `W3CEntity`, `DraftContext`, `CompiledContext`
- [apps/api/app/evals/cases.py](/Users/roy/Developer/W3C-process/apps/api/app/evals/cases.py)
  - golden question expectations for answer-quality regression
- [apps/api/app/evals/runner.py](/Users/roy/Developer/W3C-process/apps/api/app/evals/runner.py)
  - eval scorer for intent/source/URL/entity/compiled-context/forbidden-term checks

### Frontend

- [apps/web/components/ChatInterface.tsx](/Users/roy/Developer/W3C-process/apps/web/components/ChatInterface.tsx)
  - main chat layout and in-page conversation memory
- [apps/web/components/WorkflowPanel.tsx](/Users/roy/Developer/W3C-process/apps/web/components/WorkflowPanel.tsx)
  - right-side workflow diagnostics panel
- [apps/web/lib/api.ts](/Users/roy/Developer/W3C-process/apps/web/lib/api.ts)
  - frontend API types and fetch layer
- [apps/web/app/styles.css](/Users/roy/Developer/W3C-process/apps/web/app/styles.css)
  - main W3C-themed visual styling and layout density

## 8. Recent Important Modification History

This is not a git changelog; it is a human summary of high-impact changes.

### Phase 1: workflow and scope safety

- Built the initial deterministic W3C Process harness.
- Added refusal path for out-of-scope questions.
- Added citation-aware fallback answering.

### Phase 2: UI and session UX

- Switched the UI into an in-page chat app with memory during the page session.
- Added workflow/source/entity/version inspector tabs.
- Refined layout and styling toward denser, more polished interaction.

### Phase 3: entity and live context grounding

- Added W3C API entity resolution.
- Added GitHub draft context resolution.
- Added `w3c/strategy` charter issue tracking context.
- Improved handling for horizontal review and strategy labels.

### Phase 4: retrieval quality and workflow specialization

- Added targeted retrieval.
- Added horizontal review as a first-class planner intent.
- Improved next-step generation to use more Guidebook-driven operational guidance.

### Phase 5: compiled knowledge layer

- Added spec-first compiled markdown pages.
- Added compiled context loading into the chat workflow.
- Added compiled rebuild and status endpoints.
- Added compiled-aware evidence coverage and prompt context.

### Phase 6: architecture and inspector visibility

- Added Guidebook workflow topic map for horizontal review, transition, charter/recharter, and Staff Contact.
- Added compiled context visibility in the right-side Entities inspector.
- Added `SYSTEM_ARCHITECTURE.md` as the current end-to-end architecture reference.

### Phase 7: measurable answer-quality regression

- Upgraded `/eval/run` from an `in_scope` smoke test to a structured golden-question harness.
- Expanded cases to 50 golden questions covering transition, horizontal review, charter, Patent Policy, Formal Objection / Appeal, Staff Contact, W3C API entity grounding, and injection / fake Process, including CSS Grid / WebAuthn / WAI-Adapt entity grounding and fake Process authority checks.
- Added tests for eval scoring and misgrounded entity detection.
- Added a deterministic offline eval workflow using template generation plus fake W3C API/GitHub context for entity and charter cases.
- Added the frontend `Quality` inspector tab for running and viewing eval results.

## 9. Operational Notes

- Web typically runs on:
  - `http://127.0.0.1:3000`
- API typically runs on:
  - `http://127.0.0.1:8000`
- Current `.env.example` includes compiled context settings:
  - `COMPILED_CONTEXT_DIR`
  - `COMPILED_CONTEXT_ENABLED`
  - `COMPILED_CONTEXT_MIN_ENTITY_CONFIDENCE`

## 10. Suggested Bootstrap Prompt For Next Session

Use something like this in a new session:

> Please load [PROJECT_MEMORY.md](/Users/roy/Developer/W3C-process/PROJECT_MEMORY.md) and [PLAN.md](/Users/roy/Developer/W3C-process/PLAN.md) as the current project memory before making changes. This project is a W3C Process assistant with deterministic workflow, W3C API grounding, GitHub strategy/draft context, and a spec-first compiled markdown knowledge layer.

## 11. Known Constraints and Cautions

- Do not treat W3C API or GitHub repo data as normative process authority.
- Do not broaden the assistant into general-purpose chat.
- Do not replace the deterministic harness with a free-form autonomous agent architecture unless there is a very strong reason.
- Be careful with answer quality judgments: some weak answers may be caused by the local model, but many failures historically came from retrieval/entity/context issues rather than “the model is dumb.”
- Use `/eval/run` before and after retrieval/model/workflow changes; avoid relying only on subjective chat testing.
- The full `github.com/w3c/` universe is too large to index blindly; keep using on-demand official-context resolution.
