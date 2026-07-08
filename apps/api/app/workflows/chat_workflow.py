import logging
import re
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TypedDict

from app.core.config import Settings
from app.core.logging_setup import get_request_id, log_event, set_request_id
from app.models.schemas import ChatRequest, ChatResponse, ClassifyResponse, Citation, CompiledContext, DraftContext, LLMRouterDecision, NextStep, ProcessState, EvidenceCoverage, SourceVersion, TaskPlan, W3CEntity, WorkflowStep
from app.rag.retriever import Retriever
from app.services.answering import build_grounded_answer, build_next_step_details, build_refusal
from app.services.compiled_context import CompiledContextStore
from app.services.context import FOLLOW_UP_MARKERS, build_contextual_query, build_entity_augmented_query
from app.services.evidence import check_evidence_coverage
from app.services.github_context import GitHubDraftContextClient, build_draft_context_augmented_query
from app.services.llm_router import LLMRouter
from app.services.bedrock import BedrockClient
from app.services.bedrock_kb import BedrockKnowledgeBaseClient
from app.services.ollama import OllamaClient
from app.services.openai_compatible import OpenAICompatibleClient
from app.services.process_state import extract_process_state
from app.services.live_fetch import fetch_page_excerpt
from app.services.scope import classify_scope
from app.services.action_surfaces import (
    format_surfaces_for_prompt,
    linkify_bare_action_urls,
    surfaces_for_intent,
)
from app.services.citation_verifier import verify_citations
from app.services.claim_verifier import verify_claims
from app.services.cross_encoder_reranker import (
    CrossEncoderRerankResult,
    MissingDependencyError as _CrossEncoderMissing,
    rerank_with_cross_encoder,
)
from app.services.provider_override import ProviderOverrideError, build_override_client
from app.services.hyde import generate_hypothetical_passage
from app.services.query_rewriter import rewrite_query
from app.services.reranker import rerank_citations
from app.services.task_planner import build_planned_retrieval_query, plan_task
from app.services.w3c_terminology import expand_acronyms_for_retrieval
from app.services.w3c_api import W3CAPIClient


logger = logging.getLogger(__name__)


# Set of llm_provider values that route through the OpenAI-compatible
# client (everything else goes through the Ollama client). Centralised
# so adding a new provider — say "anthropic" via an OpenAI-compatible
# proxy — is a one-line change. The duplicate ad-hoc set literals that
# used to live at each call site were a known source of router bugs.
_OPENAI_COMPATIBLE_PROVIDERS: frozenset[str] = frozenset(
    {"openai", "openai-compatible", "openrouter"}
)


def is_openai_compatible_provider(provider: str) -> bool:
    """Public predicate exported for /v1/models in main.py."""
    return provider in _OPENAI_COMPATIBLE_PROVIDERS


def _uses_lighter_prompt(provider: str) -> bool:
    """Providers that get the softer formatting prompt instead of the strict one.

    External token APIs (OpenAI-compatible cluster) and Bedrock models follow
    the light-touch formatting guidance well; local Ollama needs the strict
    block spelled out.
    """
    return is_openai_compatible_provider(provider) or provider == "bedrock"


class ChatState(TypedDict, total=False):
    request: ChatRequest
    classification: ClassifyResponse
    response: ChatResponse


@dataclass
class _RunContext:
    """Mutable per-request state shared across workflow stages.

    The run() method used to be a single 770-line block carrying its
    state in closure variables (``trace``, ``audit``, ``degraded``,
    ``_stage_clock``) and closure helpers (``_record``, ``_emit_stage``,
    ``_degrade``). Lifting that state onto an object lets the stages
    sit as methods on ``ChatWorkflow`` instead of nested closures, so
    each one can be read, tested, and reasoned about in isolation.
    """

    request: ChatRequest
    source_version: SourceVersion
    trace: list[WorkflowStep] = field(default_factory=list)
    audit: dict = field(default_factory=dict)
    degraded: list[str] = field(default_factory=list)
    stream_sink: object | None = None
    stage_sink: object | None = None
    _request_start: float = field(default_factory=time.perf_counter)
    _last_stage_at: float = field(default_factory=time.perf_counter)

    def record(self, step: "WorkflowStep") -> None:
        """Append ``step`` to the trace AND emit a structured stage event.

        SSE consumers see each completed workflow node as it lands,
        not all at once at the end of the request. Per-stage timing
        is the wall-clock delta since the previous record() call.
        """
        self.trace.append(step)
        now = time.perf_counter()
        previous = self._last_stage_at
        self._last_stage_at = now
        try:
            log_event(
                logger,
                step.id,
                status=step.status,
                duration_ms=(now - previous) * 1000.0,
            )
        except Exception:  # pragma: no cover - logging must never break the request
            pass
        if self.stage_sink is None:
            return
        try:
            self.stage_sink(step)  # type: ignore[misc]
        except Exception:  # pragma: no cover - sink misbehaviour
            logger.warning("stage_sink raised; continuing without progress events")

    def degrade(self, tag: str) -> None:
        """Record that the workflow fell back to a worse path on this run."""
        if tag not in self.degraded:
            self.degraded.append(tag)


@dataclass
class _RetrievalResult:
    """Output of the retrieval phase.

    Bundles every piece of evidence the generator and verifier need
    plus the deterministic template answer (used as a fallback when
    the LLM is unavailable or returns empty text).
    """

    task_plan: TaskPlan
    routed_query: str
    retrieval_query: str
    resolved_entities: list[W3CEntity]
    draft_contexts: list[DraftContext]
    compiled_context: CompiledContext | None
    citations: list[Citation]
    process_state: ProcessState
    evidence_coverage: EvidenceCoverage
    next_steps: list[str]
    next_step_details: list[NextStep]
    template_answer: str
    supplementary_context: str | None
    action_surfaces_text: str


@dataclass
class _GenerationResult:
    """Output of the generation phase.

    ``answer`` is the final text (LLM output or template fallback);
    ``model_generation`` is the audit tag identifying which path
    produced it ("ollama" / "openai_compatible_stream" /
    "template_fallback" / "override_rejected" / ...).
    """

    answer: str
    model_generation: str
    selected_model: str
    router_model: str


@dataclass
class _ScopeResult:
    """Output of the scope phase.

    When ``refusal`` is non-None the workflow short-circuits — that's
    the response to send back as-is. Otherwise the question is in
    scope and the remaining stages should run with the rest of the
    fields as their starting state.
    """

    refusal: ChatResponse | None = None
    classification: ClassifyResponse | None = None
    router_decision: LLMRouterDecision | None = None
    contextual_query: str = ""
    used_contextual_query: bool = False
    prefetched_entities: list[W3CEntity] | None = None
    prefetch_error: str | None = None
    router_model: str = ""


class ChatWorkflow:
    """Deterministic harness workflow.

    LangGraph is the intended production graph runtime. This class keeps the
    node contract explicit and testable while the project is still a scaffold.
    """

    def __init__(
        self,
        settings: Settings,
        retriever: Retriever | None = None,
        ollama_client: OllamaClient | None = None,
        openai_compatible_client: OpenAICompatibleClient | None = None,
        bedrock_client: BedrockClient | None = None,
        bedrock_kb_client: BedrockKnowledgeBaseClient | None = None,
        w3c_api_client: W3CAPIClient | None = None,
        github_context_client: GitHubDraftContextClient | None = None,
        compiled_context_store: CompiledContextStore | None = None,
        llm_router: LLMRouter | None = None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever or Retriever()
        self.ollama_client = ollama_client or OllamaClient(
            settings.ollama_base_url,
            settings.ollama_timeout_seconds,
        )
        self.openai_compatible_client = openai_compatible_client or OpenAICompatibleClient(
            settings.openai_compatible_base_url,
            settings.openai_compatible_api_key,
            settings.openai_compatible_timeout_seconds,
        )
        # The boto3 client inside BedrockClient is built lazily on first use, so
        # constructing this eagerly costs nothing and needs no AWS round-trip.
        self.bedrock_client = bedrock_client or BedrockClient(
            settings.bedrock_region,
            settings.bedrock_access_key_id,
            settings.bedrock_secret_access_key,
            settings.bedrock_session_token,
            settings.bedrock_timeout_seconds,
        )
        # Optional Bedrock Knowledge Base retriever. Only built when enabled and
        # configured; None means the KB augmentation pass is skipped entirely.
        self.bedrock_kb_client = bedrock_kb_client
        if self.bedrock_kb_client is None and settings.bedrock_kb_enabled and settings.bedrock_kb_id:
            self.bedrock_kb_client = BedrockKnowledgeBaseClient(
                settings.bedrock_kb_id,
                settings.bedrock_kb_region or settings.bedrock_region,
                settings.bedrock_access_key_id,
                settings.bedrock_secret_access_key,
                settings.bedrock_session_token,
                settings.bedrock_kb_max_results,
                settings.bedrock_timeout_seconds,
            )
        self.w3c_api_client = w3c_api_client or W3CAPIClient(settings)
        self.github_context_client = github_context_client or GitHubDraftContextClient(settings)
        self.llm_router = llm_router or LLMRouter(settings, self._resolve_llm_client())
        self.compiled_context_store = compiled_context_store or CompiledContextStore(
            settings,
            retriever=self.retriever,
            w3c_api_client=self.w3c_api_client,
            github_context_client=self.github_context_client,
        )

    def _resolve_llm_client(self) -> OllamaClient | OpenAICompatibleClient | BedrockClient:
        """Pick the client based on the configured ``llm_provider``.

        The router, reranker, citation verifier, and generator paths
        used to each carry their own copy of this branch. Centralising
        it here means adding a new provider — say an Anthropic gateway
        served via OpenAI-compatible shape — is one edit, not five.
        """
        if self.settings.llm_provider == "bedrock":
            return self.bedrock_client
        if is_openai_compatible_provider(self.settings.llm_provider):
            return self.openai_compatible_client
        return self.ollama_client

    def _default_model_id(self) -> str:
        """Default model id matching the resolved client.

        Ollama and Bedrock share ``llm_model``; only the OpenAI-compatible
        cluster carries its own model id.
        """
        if is_openai_compatible_provider(self.settings.llm_provider):
            return self.settings.openai_compatible_model
        return self.settings.llm_model

    def classify(self, request: ChatRequest) -> ClassifyResponse:
        history_text = "\n".join(turn.content for turn in request.history)
        decision = classify_scope(request.message, history_text=history_text)
        return ClassifyResponse(
            in_scope=decision.in_scope,
            reason=decision.reason,
            matched_topics=decision.matched_topics,
            injection_risk=decision.injection_risk,
            confidence=decision.confidence,
        )

    def _run_scope_stage(self, ctx: _RunContext) -> _ScopeResult:
        """Scope phase: classify, route, optionally short-circuit with a refusal.

        Three layers run, each able to override the previous one:

          1. Keyword classifier on the raw message.
          2. Keyword classifier on the conversation-context-enriched query
             (for follow-up questions like "and how about X?").
          3. LLM-assisted router for questions keyword matching missed,
             and to validate weak keyword matches as false positives.

        While the router runs we speculatively prefetch W3C API entities
        in parallel so the retrieval phase can skip an extra network
        round-trip when scope holds up.

        Side-effects on ``ctx``:
            * adds the ``scope_classifier`` (always) and ``llm_router``
              (when attempted) trace steps
            * initialises ``audit`` with workflow / provider / model /
              router fields
            * appends ``router_failed`` to ``degraded`` if the router
              call errored
            * when scope rejects the question, populates the refusal
              trace tail (``retriever`` skipped + ``final_response``)
              and returns it as ``_ScopeResult.refusal``

        Returns ``_ScopeResult.refusal != None`` iff the workflow
        should short-circuit. Otherwise the result carries the final
        classification + router decision plus any speculative
        prefetches for the retrieval stage.
        """
        request = ctx.request
        contextual_query = build_contextual_query(request.message, request.history)
        used_contextual_query = contextual_query != request.message
        classification = self.classify(request)
        router_decision: LLMRouterDecision | None = None

        # Layer 2: re-classify using the context-enriched query for
        # follow-up questions. IMPORTANT: a follow-up marker alone (e.g.
        # "how about X") does not justify inheriting scope from history —
        # that lets unrelated topics ride along because PROCESS_TOPICS
        # keywords in prior turns leak into contextual_query. Only flip
        # to in_scope when the contextual query produces a STRONG match
        # AND the original message is either purely referential (no new
        # noun) or itself contains at least one weak topic word.
        if not classification.in_scope and used_contextual_query:
            contextual_decision = classify_scope(contextual_query)
            message_decision = classify_scope(request.message)
            is_pure_reference = _is_pure_reference(request.message)
            if (
                contextual_decision.in_scope
                and contextual_decision.confidence >= 0.9
                and (is_pure_reference or message_decision.in_scope)
            ):
                classification = ClassifyResponse(
                    in_scope=True,
                    reason="Follow-up question resolved against recent W3C Process conversation context.",
                    matched_topics=contextual_decision.matched_topics,
                    injection_risk=classification.injection_risk or contextual_decision.injection_risk,
                    confidence=contextual_decision.confidence,
                )

        router_model = (
            request.model
            or self.settings.llm_router_model
            or self._default_model_id()
        )

        # Layer 3: LLM router rescues misses and validates weak matches.
        # Optimization: when the router needs to run AND classification
        # is already weakly in-scope, speculatively prefetch W3C API
        # entities in parallel — router hints only annotate the query
        # for retrieval, they don't affect entity matching.
        _needs_router = not classification.in_scope or classification.confidence < 0.7
        prefetched_entities: list[W3CEntity] | None = None
        prefetch_error: str | None = None
        if _needs_router:
            should_prefetch = classification.in_scope
            with ThreadPoolExecutor(max_workers=2) as executor:
                router_future = executor.submit(
                    self.llm_router.route,
                    contextual_query,
                    request.history,
                    router_model,
                )
                entities_future = (
                    executor.submit(self.w3c_api_client.resolve_entities, contextual_query)
                    if should_prefetch
                    else None
                )
                router_decision = router_future.result()
                if entities_future is not None:
                    try:
                        prefetched_entities = entities_future.result()
                    except Exception as exc:  # pragma: no cover - external service fallback
                        prefetch_error = type(exc).__name__
                        logger.warning("W3C API entity prefetch failed", exc_info=exc)
            if router_decision.attempted:
                if router_decision.likely_in_scope and router_decision.confidence >= self.settings.llm_router_min_confidence:
                    classification = ClassifyResponse(
                        in_scope=True,
                        reason="LLM-assisted router classified this as likely W3C workflow scope; final answer remains evidence-gated.",
                        matched_topics=[f"llm_router:{router_decision.intent_type}"],
                        injection_risk=classification.injection_risk,
                        confidence=router_decision.confidence,
                    )
                elif classification.in_scope and not router_decision.likely_in_scope and router_decision.confidence >= self.settings.llm_router_min_confidence:
                    # Weak keyword match overridden: LLM says out of scope
                    classification = ClassifyResponse(
                        in_scope=False,
                        reason=f"Weak keyword match overridden by LLM router: {router_decision.reason}",
                        matched_topics=classification.matched_topics,
                        injection_risk=classification.injection_risk,
                        confidence=router_decision.confidence,
                    )

        # Scope classifier step always lands first — even on refusals, so
        # operators can see why the workflow short-circuited.
        ctx.trace.append(
            WorkflowStep(
                id="scope_classifier",
                label="Scope classifier",
                status="completed",
                detail=(
                    "Question accepted as W3C Process scope."
                    if classification.in_scope
                    else "Question rejected because it is outside W3C Process scope."
                ),
            )
        )

        override = request.provider_override
        ctx.audit.update({
            "workflow": "harness_v1",
            "matched_topics": classification.matched_topics,
            "injection_risk": classification.injection_risk,
            # Audit reflects what the workflow WILL run with. For user
            # overrides we record kind+model only — never base_url and
            # never api_key, so the audit blob is safe to surface or
            # persist.
            "llm_provider": f"override:{override.kind}" if override else self.settings.llm_provider,
            "llm_model": override.model if override else _selected_model(request, self.settings),
            "used_contextual_query": used_contextual_query,
            "degraded": ctx.degraded,
        })

        if router_decision:
            ctx.audit["llm_router"] = router_decision.model_dump(mode="json")
            if router_decision.attempted and router_decision.error:
                ctx.degrade("router_failed")
            ctx.record(
                WorkflowStep(
                    id="llm_router",
                    label="LLM-assisted router",
                    status=(
                        "completed"
                        if router_decision.attempted and router_decision.likely_in_scope
                        else "skipped"
                    ),
                    detail=_router_detail(router_decision),
                )
            )

        if not classification.in_scope:
            answer = build_refusal(request.locale, request.message)
            ctx.trace.extend([
                WorkflowStep(
                    id="retriever",
                    label="Authoritative source retrieval",
                    status="skipped",
                    detail="Retrieval was skipped because the question is outside scope.",
                ),
                WorkflowStep(
                    id="final_response",
                    label="Final conclusion",
                    status="completed",
                    detail=answer,
                ),
            ])
            return _ScopeResult(
                refusal=ChatResponse(
                    answer=answer,
                    in_scope=False,
                    confidence=0.98,
                    source_version=ctx.source_version,
                    refusal_reason=classification.reason,
                    workflow_trace=ctx.trace,
                    audit=ctx.audit,
                ),
            )

        if used_contextual_query:
            ctx.record(
                WorkflowStep(
                    id="query_rewriter",
                    label="Context resolver",
                    status="completed",
                    detail=(
                        "Resolved the current follow-up against recent page-session turns before retrieval. "
                        "Conversation text was used only to understand references, not as authoritative source material."
                    ),
                )
            )

        return _ScopeResult(
            classification=classification,
            router_decision=router_decision,
            contextual_query=contextual_query,
            used_contextual_query=used_contextual_query,
            prefetched_entities=prefetched_entities,
            prefetch_error=prefetch_error,
            router_model=router_model,
        )

    def run(
        self,
        request: ChatRequest,
        *,
        stream_sink: object | None = None,
        stage_sink: object | None = None,
    ) -> ChatResponse:
        """Execute the full workflow and return the assembled ChatResponse.

        When ``stream_sink`` is provided AND the resolved LLM client supports
        token streaming, the workflow uses the streaming generation path and
        invokes ``stream_sink(delta)`` for each text chunk as it arrives.

        When ``stage_sink`` is provided, the workflow invokes
        ``stage_sink(WorkflowStep)`` AS each pre-LLM stage completes — scope
        classifier, task planner, entity resolver, retriever, reranker,
        evidence coverage, process state. This lets the SSE consumer paint
        the workflow inspector progressively rather than waiting for the
        whole response. Both sinks are independent; either, both, or
        neither can be set.
        """

        source_version = SourceVersion(indexed_at=datetime.now(timezone.utc).isoformat())
        ctx = _RunContext(
            request=request,
            source_version=source_version,
            stream_sink=stream_sink,
            stage_sink=stage_sink,
        )
        scope = self._run_scope_stage(ctx)
        if scope.refusal is not None:
            return scope.refusal

        retrieval = self._run_retrieval_stage(ctx, scope)
        generation = self._run_generation_stage(ctx, scope, retrieval)

        return self._run_verification_stage(
            ctx,
            answer=generation.answer,
            citations=retrieval.citations,
            model_generation=generation.model_generation,
            router_model=generation.router_model,
            classification=scope.classification,
            next_steps=retrieval.next_steps,
            next_step_details=retrieval.next_step_details,
            task_plan=retrieval.task_plan,
            evidence_coverage=retrieval.evidence_coverage,
            process_state=retrieval.process_state,
            compiled_context=retrieval.compiled_context,
            resolved_entities=retrieval.resolved_entities,
            draft_contexts=retrieval.draft_contexts,
        )

    def _run_retrieval_stage(
        self, ctx: _RunContext, scope: _ScopeResult
    ) -> _RetrievalResult:
        """Retrieval phase: plan, resolve, retrieve, rerank, evidence-check.

        Steps in order:
            1. Task planner — turns the routed query into a structured
               intent (advance_specification / file_review / ...).
            2. W3C API entity resolver — uses the prefetch from the
               scope phase when available, otherwise issues a new call.
            3. GitHub draft + compiled context resolvers in parallel
               (network + filesystem; independent).
            4. Query augmentation with entity + draft hints.
            5. Optional LLM query rewriter (multi-query expansion;
               skipped in template mode for deterministic evals).
            6. Hybrid retrieval over the corpus.
            7. Reranker — cross-encoder when available, falls back to
               LLM-as-reranker. Skipped silently for <4 candidates.
            8. Process state extraction + evidence coverage check.
            9. Optional targeted second-pass retrieval when coverage
               flags missing evidence.
           10. Optional live page fetch when coverage is still
               insufficient and live_fetch is enabled.
           11. Build the deterministic template answer (used as the
               fallback when the LLM is unavailable).

        All trace events and audit fields land on ``ctx`` as side
        effects. Degradations are recorded via ``ctx.degrade`` so the
        ``audit["degraded"]`` channel stays accurate.
        """
        request = ctx.request
        classification = scope.classification
        router_decision = scope.router_decision
        contextual_query = scope.contextual_query
        prefetched_entities = scope.prefetched_entities
        prefetch_error = scope.prefetch_error
        router_model = scope.router_model
        audit = ctx.audit

        # Shim for the still-inline trace-emission style copied from the
        # legacy method body. Lets the (large) block below stay
        # textually identical to the pre-refactor version — easier to
        # diff-review than a wholesale rewrite. Future cleanup can
        # convert these to direct ``ctx.record`` calls.
        def _record(_unused_trace_list, step: "WorkflowStep") -> None:
            ctx.record(step)

        def _degrade(tag: str) -> None:
            ctx.degrade(tag)

        trace = ctx.trace  # alias used by legacy block below

        routed_query = _router_augmented_query(contextual_query, router_decision)
        task_plan = plan_task(routed_query, request.history)
        audit["task_plan"] = task_plan.model_dump(mode="json")
        _record(trace, 
            WorkflowStep(
                id="task_planner",
                label="Task planner",
                status="completed",
                detail=_task_plan_detail(task_plan),
            )
        )

        resolved_entities: list[W3CEntity] = []
        if prefetched_entities is not None:
            # Reuse parallel prefetch result (router_decision lane); avoid duplicate network call.
            resolved_entities = prefetched_entities
            audit["resolved_entities"] = [entity.model_dump(mode="json") for entity in resolved_entities]
            audit["w3c_api_prefetched"] = True
            _record(trace, 
                WorkflowStep(
                    id="w3c_api_resolver",
                    label="W3C API entity resolver",
                    status="completed",
                    detail=_entity_resolution_detail(resolved_entities),
                )
            )
        elif prefetch_error is not None:
            audit["w3c_api_error"] = prefetch_error
            _degrade("w3c_api_unavailable")
            _record(trace, 
                WorkflowStep(
                    id="w3c_api_resolver",
                    label="W3C API entity resolver",
                    status="failed",
                    detail="W3C API entity lookup failed; continuing with Process and Guidebook corpus retrieval.",
                )
            )
        else:
            try:
                resolved_entities = self.w3c_api_client.resolve_entities(routed_query)
                audit["resolved_entities"] = [entity.model_dump(mode="json") for entity in resolved_entities]
                _record(trace, 
                    WorkflowStep(
                        id="w3c_api_resolver",
                        label="W3C API entity resolver",
                        status="completed",
                        detail=_entity_resolution_detail(resolved_entities),
                    )
                )
            except Exception as exc:  # pragma: no cover - external service fallback
                audit["w3c_api_error"] = type(exc).__name__
                _degrade("w3c_api_unavailable")
                logger.warning("W3C API entity resolution failed", exc_info=exc)
                _record(trace, 
                    WorkflowStep(
                        id="w3c_api_resolver",
                        label="W3C API entity resolver",
                        status="failed",
                        detail="W3C API entity lookup failed; continuing with Process and Guidebook corpus retrieval.",
                    )
                )

        draft_contexts: list[DraftContext] = []
        compiled_context: CompiledContext | None = None
        # Parallelize: GitHub draft (network) + compiled context (filesystem) are independent.
        with ThreadPoolExecutor(max_workers=2) as executor:
            draft_future = executor.submit(
                self.github_context_client.resolve_contexts,
                routed_query,
                resolved_entities,
                task_plan,
            )
            compiled_future = executor.submit(
                self.compiled_context_store.resolve,
                resolved_entities,
            )
            try:
                draft_contexts = draft_future.result()
                audit["draft_contexts"] = [context.model_dump(mode="json") for context in draft_contexts]
                _record(trace, 
                    WorkflowStep(
                        id="draft_context_resolver",
                        label="GitHub draft context resolver",
                        status="completed" if draft_contexts else "skipped",
                        detail=_draft_context_detail(draft_contexts),
                    )
                )
            except Exception as exc:  # pragma: no cover - external service fallback
                audit["draft_context_error"] = type(exc).__name__
                _degrade("draft_contexts_unavailable")
                logger.warning("GitHub draft context resolution failed", exc_info=exc)
                _record(trace, 
                    WorkflowStep(
                        id="draft_context_resolver",
                        label="GitHub draft context resolver",
                        status="failed",
                        detail=(
                            "GitHub draft context lookup failed; continuing with Process and Guidebook corpus retrieval. "
                            "Draft repository data is never treated as normative Process authority."
                        ),
                    )
                )
            try:
                compiled_context = compiled_future.result()
                if compiled_context:
                    audit["compiled_context"] = compiled_context.model_dump(mode="json")
                _record(trace, 
                    WorkflowStep(
                        id="compiled_context_resolver",
                        label="Compiled knowledge resolver",
                        status="completed" if compiled_context else "skipped",
                        detail=_compiled_context_detail(compiled_context),
                    )
                )
            except Exception as exc:  # pragma: no cover - local filesystem fallback
                audit["compiled_context_error"] = type(exc).__name__
                _degrade("compiled_context_unavailable")
                logger.warning("Compiled context resolution failed", exc_info=exc)
                _record(trace, 
                    WorkflowStep(
                        id="compiled_context_resolver",
                        label="Compiled knowledge resolver",
                        status="failed",
                        detail="Compiled spec context lookup failed; continuing with raw Process and Guidebook retrieval.",
                    )
                )

        planned_query = build_planned_retrieval_query(routed_query, task_plan)
        retrieval_query = build_entity_augmented_query(planned_query, resolved_entities)
        retrieval_query = build_draft_context_augmented_query(retrieval_query, draft_contexts)
        # Acronym widening: users overwhelmingly write "from CR to REC",
        # the corpus has both the acronym and the long form, but BM25
        # scores them as different terms. Silently append the long-form
        # expansion for any bare acronym so lexical retrieval hits
        # either form. The original query text is preserved verbatim
        # so dense + topic-relevance ranking still uses what the user
        # actually typed.
        retrieval_query = expand_acronyms_for_retrieval(retrieval_query)
        used_entity_augmented_query = retrieval_query != planned_query
        audit["used_entity_augmented_query"] = used_entity_augmented_query
        if used_entity_augmented_query:
            _record(trace, 
                WorkflowStep(
                    id="entity_query_enricher",
                    label="Entity-aware retrieval query",
                    status="completed",
                    detail=(
                        "Expanded the retrieval query with resolved W3C API specification/group status hints. "
                        "These hints steer Process and Guidebook retrieval but are not treated as normative rules."
                    ),
                )
            )

        # Multi-query retrieval AND HyDE in parallel. Both are LLM
        # calls that produce alternative retrieval queries:
        #   * query_rewriter → 1-3 W3C-canonical short rewrites
        #     (helps lexical recall — BM25 widens by vocabulary)
        #   * hyde → one 2-4 sentence hypothetical answer passage
        #     (helps dense recall — semantic shape of an answer is
        #     closer to actual answer chunks than the question is)
        # They're complementary, so we run both. Running in parallel
        # means wall-clock cost is bounded by the slower call, not
        # the sum. Skipped entirely in template mode to keep evals
        # deterministic and offline.
        rewrite_variants: list[str] = []
        hypothetical_passage: str = ""
        if self.settings.llm_provider != "template":
            client = self._resolve_llm_client()
            with ThreadPoolExecutor(max_workers=2) as executor:
                rewrite_future = executor.submit(
                    rewrite_query,
                    request.message,
                    settings=self.settings,
                    client=client,
                    model=router_model,
                )
                hyde_future = (
                    executor.submit(
                        generate_hypothetical_passage,
                        request.message,
                        settings=self.settings,
                        client=client,
                        model=router_model,
                    )
                    if self.settings.hyde_enabled
                    else None
                )
                try:
                    rewrite_result = rewrite_future.result()
                    rewrite_variants = rewrite_result.variants
                    audit["query_rewrites"] = rewrite_variants
                    if rewrite_result.error:
                        audit["query_rewrite_error"] = rewrite_result.error
                        _degrade("query_rewriter_failed")
                except Exception as exc:  # pragma: no cover - non-fatal
                    logger.warning("Query rewriter failed; continuing with single-query retrieval", exc_info=exc)
                    audit["query_rewrite_error"] = type(exc).__name__
                    _degrade("query_rewriter_failed")
                if hyde_future is not None:
                    try:
                        hyde_result = hyde_future.result()
                        hypothetical_passage = hyde_result.passage
                        if hyde_result.error:
                            audit["hyde_error"] = hyde_result.error
                            _degrade("hyde_failed")
                    except Exception as exc:  # pragma: no cover - non-fatal
                        logger.warning("HyDE failed; continuing without hypothetical passage", exc_info=exc)
                        audit["hyde_error"] = type(exc).__name__
                        _degrade("hyde_failed")

        # Pass the raw user message separately so the retriever can rank by
        # what the user actually asked instead of being biased by the 10+
        # lines of task-planner / entity / router metadata in retrieval_query.
        citations = self.retriever.retrieve(retrieval_query, user_message=request.message)

        # Bedrock Knowledge Base augmentation — one query per request (on the
        # raw user message, best for the KB's semantic search). KB passages are
        # merged ahead of the corpus hits so they're favoured, then the whole
        # pool is reranked and grounded exactly like corpus chunks. Failures are
        # non-fatal: the corpus retrieval still stands.
        if self.bedrock_kb_client is not None:
            try:
                kb_hits = self.bedrock_kb_client.retrieve(request.message)
            except Exception as exc:  # pragma: no cover - external service fallback
                kb_hits = []
                logger.warning("Bedrock KB retrieval failed; continuing without it", exc_info=exc)
                audit["bedrock_kb_error"] = type(exc).__name__
                _degrade("bedrock_kb_failed")
            if kb_hits:
                audit["bedrock_kb_hits"] = len(kb_hits)
                citations = _merge_citations(kb_hits, citations, limit=12)
                _record(trace,
                    WorkflowStep(
                        id="bedrock_kb",
                        label="Bedrock Knowledge Base retrieval",
                        status="completed",
                        detail=(
                            f"Retrieved {len(kb_hits)} passage(s) from the configured Bedrock "
                            "Knowledge Base and merged them with the local corpus for reranking."
                        ),
                        references=kb_hits,
                    )
                )

        if rewrite_variants:
            extra_hits: list[Citation] = []
            for variant in rewrite_variants:
                try:
                    extra_hits.extend(
                        self.retriever.retrieve(
                            expand_acronyms_for_retrieval(variant),
                            user_message=variant,
                        )
                    )
                except Exception as exc:  # pragma: no cover - non-fatal
                    logger.warning("Variant retrieval failed for %r", variant, exc_info=exc)
            citations = _merge_citations(citations, extra_hits, limit=12)
            _record(trace,
                WorkflowStep(
                    id="query_rewriter",
                    label="LLM query rewriter",
                    status="completed",
                    detail=(
                        f"Generated {len(rewrite_variants)} W3C-canonical rewrites and merged their "
                        "citations with the primary retrieval. Original message remains the authoritative query."
                    ),
                )
            )

        # HyDE second pass: embed the hypothetical answer instead of
        # the bare question. The retriever splits BM25 input (``query``)
        # from ranking input (``user_message``); for HyDE we pass the
        # passage as ``query`` so both lexical and dense paths benefit
        # — and keep the original user message as the ranking signal so
        # topic-relevance scoring doesn't drift from what the user asked.
        if hypothetical_passage:
            try:
                hyde_hits = self.retriever.retrieve(
                    hypothetical_passage, user_message=request.message
                )
                citations = _merge_citations(citations, hyde_hits, limit=12)
                audit["hyde_passage_chars"] = len(hypothetical_passage)
                _record(trace,
                    WorkflowStep(
                        id="hyde_retrieval",
                        label="HyDE hypothetical-passage retrieval",
                        status="completed",
                        detail=(
                            "Generated a hypothetical answer passage and ran an extra retrieval "
                            "pass against it — broadens dense-retrieval recall over chunks whose "
                            "answer-shaped prose matches the passage better than the bare question. "
                            "The hypothesis is informational only; the final answer remains grounded "
                            "in the citations the user sees."
                        ),
                    )
                )
            except Exception as exc:  # pragma: no cover - non-fatal
                logger.warning("HyDE retrieval pass failed; continuing without it", exc_info=exc)
                audit["hyde_error"] = type(exc).__name__
                _degrade("hyde_failed")

        # Reranker. Try the local cross-encoder first (fast, no LLM quota
        # cost, model-trained for relevance); fall back to the LLM-as-
        # reranker if sentence-transformers isn't installed. Skipped
        # silently for template provider (eval mode) or <4 candidates.
        used_cross_encoder = False
        if len(citations) >= 4:
            if self.settings.reranker_cross_encoder_enabled:
                ce_result: CrossEncoderRerankResult = rerank_with_cross_encoder(
                    request.message,
                    citations,
                    model_name=self.settings.reranker_model,
                )
                if ce_result.reordered:
                    citations = ce_result.citations
                    audit["reranker_kind"] = "cross_encoder"
                    audit["reranker_model"] = self.settings.reranker_model
                    used_cross_encoder = True
                    _record(trace, 
                        WorkflowStep(
                            id="reranker",
                            label="Cross-encoder reranker",
                            status="completed",
                            detail=(
                                f"Reordered the hybrid retrieval candidates with the local "
                                f"{self.settings.reranker_model} cross-encoder. Hybrid order is "
                                "the tie-breaker; the reranker only reorders, never drops citations."
                            ),
                            references=citations,
                        )
                    )
                elif ce_result.skipped_reason and "unavailable" not in (ce_result.skipped_reason or ""):
                    audit["reranker_skipped"] = ce_result.skipped_reason
                elif ce_result.skipped_reason and "unavailable" in ce_result.skipped_reason:
                    # CE itself couldn't load (e.g. sentence-transformers missing
                    # or model download failed). Record the degradation so
                    # operators see we're running on the LLM reranker fallback.
                    _degrade("cross_encoder_unavailable")
            if not used_cross_encoder and self.settings.llm_provider != "template":
                rerank_result = rerank_citations(
                    request.message,
                    citations,
                    settings=self.settings,
                    client=self._resolve_llm_client(),
                    model=router_model,
                )
                if rerank_result.reordered:
                    citations = rerank_result.citations
                    audit["reranker_kind"] = "llm"
                    audit["reranker_model"] = rerank_result.model
                    _record(trace, 
                        WorkflowStep(
                            id="reranker",
                            label="LLM relevance reranker",
                            status="completed",
                            detail=(
                                "Reordered the hybrid retrieval candidates by LLM-judged relevance to "
                                "the user's question. Hybrid scores remain the tie-breaker; the model "
                                "cannot drop a citation from the result list."
                            ),
                            references=citations,
                        )
                    )
                elif rerank_result.skipped_reason:
                    audit["reranker_skipped"] = rerank_result.skipped_reason

        process_state = extract_process_state(retrieval_query, citations, resolved_entities)
        evidence_coverage = check_evidence_coverage(
            plan=task_plan,
            citations=citations,
            entities=resolved_entities,
            process_state=process_state,
            compiled_context=compiled_context,
            query=routed_query,
        )
        _record(trace, 
            WorkflowStep(
                id="retriever",
                label="Authoritative source retrieval",
                status="completed",
                detail=(
                    "Retrieved trusted W3C Process, Guidebook, or repository sections from the local corpus. "
                    "Process excerpts are treated as normative; Guidebook excerpts are treated as practice guidance."
                ),
                references=citations,
            )
        )

        if evidence_coverage.status == "needs_more_evidence" and evidence_coverage.targeted_queries:
            attempted_targeted_queries = list(evidence_coverage.targeted_queries)
            targeted_hits: list[Citation] = []
            for targeted_query in attempted_targeted_queries:
                targeted_hits.extend(
                    self.retriever.retrieve(targeted_query, user_message=request.message)
                )
            citations = _merge_citations(citations, targeted_hits)
            process_state = extract_process_state(retrieval_query, citations, resolved_entities)
            evidence_coverage = check_evidence_coverage(
                plan=task_plan,
                citations=citations,
                entities=resolved_entities,
                process_state=process_state,
                compiled_context=compiled_context,
            )
            _record(trace, 
                WorkflowStep(
                    id="targeted_retrieval",
                    label="Targeted second retrieval",
                    status="completed",
                    detail=(
                        "Ran focused follow-up retrieval for missing evidence: "
                        f"{'; '.join(attempted_targeted_queries)}."
                    ),
                    references=citations,
                )
            )

        audit["evidence_coverage"] = evidence_coverage.model_dump(mode="json")
        _record(trace, 
            WorkflowStep(
                id="evidence_coverage",
                label="Evidence coverage check",
                status="completed" if evidence_coverage.status == "sufficient" else "failed",
                detail=evidence_coverage.summary,
                references=citations,
            )
        )
        _record(trace, 
            WorkflowStep(
                id="process_state",
                label="Process state extraction",
                status="completed",
                detail=(
                    f"Identified likely workflow: {process_state.likely_workflow}; "
                    f"intent: {process_state.intent}; "
                    f"missing information: {', '.join(process_state.missing_information) or 'none detected'}."
                ),
                references=citations,
            )
        )

        supplementary_context: str | None = None
        if self.settings.live_fetch_enabled and evidence_coverage.status == "insufficient":
            primary_url = next(
                (str(c.url) for c in citations if c.url),
                None,
            )
            if primary_url:
                supplementary_context = fetch_page_excerpt(
                    primary_url,
                    max_chars=self.settings.live_fetch_max_chars,
                    timeout=self.settings.live_fetch_timeout_seconds,
                    allowlist=self.settings.allowlist_entries,
                )
            _record(trace, 
                WorkflowStep(
                    id="live_fetch",
                    label="Live page fetch",
                    status="completed" if supplementary_context else "skipped",
                    detail=(
                        f"Fetched live page content from {primary_url} to supplement insufficient corpus evidence."
                        if supplementary_context
                        else "Live fetch skipped: no primary URL available or fetch failed."
                    ),
                )
            )
            if supplementary_context:
                audit["live_fetch_url"] = primary_url

        answer, next_steps = build_grounded_answer(
            routed_query,
            citations,
            request.locale,
            draft_contexts,
            compiled_context=compiled_context,
        )
        next_step_details = build_next_step_details(routed_query, citations, next_steps, compiled_context=compiled_context)
        # Resolve concrete W3C action surfaces (issue trackers, mailing lists,
        # forms) for this intent so the model can end each step with a
        # "do X at Y" instruction instead of vague guidance.
        action_surfaces_text = format_surfaces_for_prompt(
            surfaces_for_intent(task_plan.intent_type if task_plan else None)
        )
        if action_surfaces_text:
            audit["action_surfaces_intent"] = task_plan.intent_type if task_plan else None

        return _RetrievalResult(
            task_plan=task_plan,
            routed_query=routed_query,
            retrieval_query=retrieval_query,
            resolved_entities=resolved_entities,
            draft_contexts=draft_contexts,
            compiled_context=compiled_context,
            citations=citations,
            process_state=process_state,
            evidence_coverage=evidence_coverage,
            next_steps=next_steps,
            next_step_details=next_step_details,
            template_answer=answer,
            supplementary_context=supplementary_context,
            action_surfaces_text=action_surfaces_text,
        )

    def _run_generation_stage(
        self,
        ctx: _RunContext,
        scope: _ScopeResult,
        retrieval: _RetrievalResult,
    ) -> _GenerationResult:
        """Generation phase: prompt assembly + LLM call (sync or stream).

        Three paths are possible depending on configuration and what
        the caller supplied:

          * **override**: the request carries a per-request
            ProviderOverride (user's own OpenAI-compatible / Ollama
            endpoint). A one-shot client is built; the api_key is
            consumed and never written to ``audit``.
          * **stream**: ``ctx.stream_sink`` is set AND the resolved
            client exposes ``stream_answer``. Each token delta is
            forwarded to the sink as it arrives; the final answer is
            the cleaned concatenation.
          * **sync**: a single ``generate_answer`` blocking call.

        On any LLM exception (timeout, 500, unsupported model, ...)
        we fall back to ``retrieval.template_answer`` — the
        deterministic, citation-grounded answer the retrieval phase
        already built. ``audit["degraded"]`` gets the
        ``llm_generation_failed`` tag so operators see the path that
        served the user.

        Returns the final answer text + the ``model_generation`` audit
        tag identifying which path produced it. Also appends the
        ``answer_generator`` trace step as a side effect.
        """
        request = ctx.request
        classification = scope.classification
        router_model = scope.router_model
        override = request.provider_override

        audit = ctx.audit
        answer = retrieval.template_answer
        citations = retrieval.citations

        # When the user supplies a per-request provider override we use
        # THEIR endpoint and model id; otherwise the server defaults.
        # The api_key never leaves this function — it is consumed by
        # the one-shot client and intentionally never written to audit.
        if override is not None:
            selected_model = override.model
            generation_provider = override.kind
            audit["model_provider_source"] = "override"
        else:
            selected_model = _selected_model(request, self.settings)
            generation_provider = self.settings.llm_provider
            audit["model_provider_source"] = "default"
        model_generation = "template"

        # When injection language is detected we throw away the
        # conversation history before handing it to the model. The
        # audit field already records that injection was suspected;
        # this is the enforcement that makes the safety_note safe.
        safe_history = [] if classification.injection_risk else request.history

        # Resolve which client handles this request. Override clients
        # are built per-request and never cached.
        generation_client: object | None = None
        if override is not None:
            try:
                generation_client = build_override_client(override, self.settings)
            except ProviderOverrideError as exc:
                model_generation = "override_rejected"
                audit["model_generation"] = model_generation
                audit["model_error"] = "ProviderOverrideError"
                audit["model_error_detail"] = str(exc)
                ctx.degrade("provider_override_rejected")
                logger.info("Provider override rejected: %s", exc)
        elif generation_provider == "ollama":
            generation_client = self.ollama_client
        elif generation_provider == "bedrock":
            generation_client = self.bedrock_client
        elif is_openai_compatible_provider(generation_provider):
            generation_client = self.openai_compatible_client

        # Audit/log names use underscore form for backwards compatibility
        # with existing tests, logs, and eval cases.
        provider_label = generation_provider.replace("-", "_")

        stream_sink = ctx.stream_sink

        if generation_client is not None:
            # Trust mode: external token-API providers (OpenAI, Kimi/
            # moonshot, OpenRouter, ...) get the lighter prompt with
            # softer formatting prescriptions. The safety + grounding
            # rules are unchanged; only the format-policing block
            # (numbered-list incrementing, no bold-on-labels, the
            # 5-sub-bullet template) is swapped for a single "use
            # your judgement" line. Local Ollama keeps the strict
            # prompt because it needs the structure spelled out. Bedrock
            # models (Claude et al.) follow the softer prompt well too.
            lighter_mode = _uses_lighter_prompt(generation_provider)
            if lighter_mode:
                audit["prompt_mode"] = "lighter"
            generation_kwargs = dict(
                model=selected_model,
                question=request.message,
                locale=request.locale,
                citations=citations,
                fallback_answer=answer,
                fallback_next_steps=retrieval.next_steps,
                history=safe_history,
                entities=retrieval.resolved_entities,
                task_plan=retrieval.task_plan,
                process_state=retrieval.process_state,
                evidence_coverage=retrieval.evidence_coverage,
                draft_contexts=retrieval.draft_contexts,
                compiled_context=retrieval.compiled_context,
                supplementary_context=retrieval.supplementary_context,
                action_surfaces_text=retrieval.action_surfaces_text,
                lighter_mode=lighter_mode,
            )
            stream_supported = stream_sink is not None and hasattr(generation_client, "stream_answer")
            try:
                if stream_supported:
                    # Real-token streaming path. Each delta is forwarded
                    # to the sink as it arrives; the assembled answer
                    # is the concatenation cleaned with the same
                    # harness post-processing as the sync path.
                    from app.services.ollama import _clean_model_text  # local to avoid moving import to top

                    chunks: list[str] = []
                    for delta in generation_client.stream_answer(**generation_kwargs):  # type: ignore[union-attr]
                        if delta:
                            chunks.append(delta)
                            try:
                                stream_sink(delta)  # type: ignore[misc]
                            except Exception:  # pragma: no cover - sink misbehaviour
                                logger.warning("stream_sink raised; continuing without forwarding deltas")
                                stream_sink = None
                    streamed_text = _clean_model_text("".join(chunks))
                    if streamed_text:
                        answer = streamed_text
                        model_generation = f"{provider_label}_stream"
                        audit["model_generation"] = model_generation
                    else:
                        model_generation = f"{provider_label}_stream_empty_fallback"
                        audit["model_generation"] = model_generation
                else:
                    generation = generation_client.generate_answer(**generation_kwargs)  # type: ignore[union-attr]
                    if generation.text:
                        answer = generation.text
                        model_generation = provider_label
                        audit["model_generation"] = model_generation
                    else:
                        model_generation = f"{provider_label}_empty_fallback"
                        audit["model_generation"] = model_generation
            except Exception as exc:
                model_generation = "template_fallback"
                audit["model_generation"] = model_generation
                model_error = type(exc).__name__
                # Surface the upstream status code (429 rate-limit vs 5xx
                # outage vs 401 auth) in the audit blob — a bare exception
                # class name is not enough to diagnose provider failures.
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                if status_code is not None:
                    model_error = f"{model_error}:{status_code}"
                audit["model_error"] = model_error
                ctx.degrade("llm_generation_failed")
                logger.warning(
                    "Answer generation via %s failed; using template fallback",
                    provider_label,
                    exc_info=exc,
                )
        else:
            audit["model_generation"] = model_generation
        # Defensive: explicitly drop the secret before this scope ends so any
        # future debugger / serializer can't reach it from the local frame.
        generation_client = None

        # Belt-and-suspenders: if the model emitted a bare action URL
        # instead of ``[label](url)`` markdown — common with lighter-
        # prompt models that sometimes underweight the link rule — wrap
        # those URLs so the frontend renders them as clickable links
        # instead of inert plain text. Only the curated surface URLs
        # for this intent are touched.
        if retrieval.task_plan is not None:
            try:
                wrapped = linkify_bare_action_urls(
                    answer,
                    surfaces_for_intent(retrieval.task_plan.intent_type),
                )
                if wrapped != answer:
                    audit["linkified_bare_action_urls"] = True
                    answer = wrapped
            except Exception:  # pragma: no cover - linkifier must never break the request
                pass

        ctx.record(
            WorkflowStep(
                id="answer_generator",
                label="Answer generation",
                status="completed",
                detail=_model_generation_detail(model_generation, selected_model),
                references=citations,
            )
        )

        return _GenerationResult(
            answer=answer,
            model_generation=model_generation,
            selected_model=selected_model,
            router_model=router_model,
        )

    def _run_verification_stage(
        self,
        ctx: _RunContext,
        *,
        answer: str,
        citations: list[Citation],
        model_generation: str,
        router_model: str,
        classification: ClassifyResponse,
        next_steps: list[str],
        next_step_details: list[NextStep],
        task_plan: TaskPlan | None,
        evidence_coverage: EvidenceCoverage | None,
        process_state: ProcessState | None,
        compiled_context: CompiledContext | None,
        resolved_entities: list[W3CEntity],
        draft_contexts: list[DraftContext],
    ) -> ChatResponse:
        """Citation-verifier + injection-guard + final assembly.

        Runs the LLM-backed citation verifier (post-generation) to
        strip ``[Sn]`` tags whose excerpts don't actually support the
        surrounding claim. The claim text is preserved; only the
        misleading attribution is removed. Skipped in template mode,
        when the answer came from a template fallback, or when there
        are no citations to verify.

        If the scope phase flagged the question as injection-risk we
        also append a visible ``injection_guard`` trace step so users
        can see why their custom claims weren't treated as Process
        authority.

        Returns the assembled ``ChatResponse`` ready to send.
        """
        if (
            self.settings.llm_provider != "template"
            and model_generation not in {"template", "template_fallback", "override_rejected"}
            and citations
        ):
            verification = verify_citations(
                answer,
                citations,
                settings=self.settings,
                client=self._resolve_llm_client(),
                model=router_model,
            )
            if verification.stripped_pairs:
                answer = verification.answer
                ctx.audit["citation_verifier"] = {
                    "model": verification.model,
                    "flagged_count": len(verification.stripped_pairs),
                }
                ctx.record(
                    WorkflowStep(
                        id="citation_verifier",
                        label="Citation verifier",
                        status="completed",
                        detail=(
                            f"Flagged {len(verification.stripped_pairs)} citation tag(s) the verifier judged unsupported "
                            "by their excerpts. Flagged tags keep their source link but carry an inline [unverified] "
                            "marker; tags pointing at nonexistent sources are removed."
                        ),
                    )
                )
            elif verification.skipped_reason:
                ctx.audit["citation_verifier_skipped"] = verification.skipped_reason

            # Chain-of-verification — second pass, opposite direction.
            # citation_verifier above strips ``[Sn]`` tags whose
            # claims don't survive scrutiny. claim_verifier finds
            # CLAIM-SENTENCES that have no [Sn] tag at all and
            # appends an inline ``[unverified]`` marker so users
            # see which assertions lack source grounding. Off by
            # default; flip ``claim_verification_enabled`` to turn on.
            if self.settings.claim_verification_enabled:
                claim_result = verify_claims(
                    answer,
                    citations,
                    settings=self.settings,
                    client=self._resolve_llm_client(),
                    model=router_model,
                )
                if claim_result.unsupported_claims:
                    answer = claim_result.annotated_answer
                    ctx.audit["claim_verifier"] = {
                        "model": claim_result.model,
                        "annotated_count": len(claim_result.unsupported_claims),
                    }
                    ctx.record(
                        WorkflowStep(
                            id="claim_verifier",
                            label="Unsourced-claim auditor",
                            status="completed",
                            detail=(
                                f"Found {len(claim_result.unsupported_claims)} factual claim(s) "
                                "without a citation tag that the cited excerpts don't support. "
                                "Each is marked inline as ``[unverified]`` so the user can see "
                                "which assertions aren't grounded."
                            ),
                        )
                    )
                elif claim_result.skipped_reason:
                    ctx.audit["claim_verifier_skipped"] = claim_result.skipped_reason

        if classification.injection_risk:
            ctx.audit["safety_note"] = "Potential prompt injection detected; answer constrained to trusted sources."
            ctx.record(
                WorkflowStep(
                    id="injection_guard",
                    label="Prompt-injection guard",
                    status="completed",
                    detail="Potential injection language was detected; user-provided process claims were not treated as authoritative.",
                )
            )

        return self._finalize_in_scope_response(
            answer=answer,
            citations=citations,
            next_steps=next_steps,
            next_step_details=next_step_details,
            task_plan=task_plan,
            evidence_coverage=evidence_coverage,
            process_state=process_state,
            compiled_context=compiled_context,
            resolved_entities=resolved_entities,
            draft_contexts=draft_contexts,
            source_version=ctx.source_version,
            trace=ctx.trace,
            audit=ctx.audit,
        )

    def _finalize_in_scope_response(
        self,
        *,
        answer: str,
        citations: list[Citation],
        next_steps: list[str],
        next_step_details: list[NextStep],
        task_plan: TaskPlan | None,
        evidence_coverage: EvidenceCoverage | None,
        process_state: ProcessState | None,
        compiled_context: CompiledContext | None,
        resolved_entities: list[W3CEntity],
        draft_contexts: list[DraftContext],
        source_version: SourceVersion,
        trace: list[WorkflowStep],
        audit: dict,
    ) -> ChatResponse:
        """Append the citation_check + final_response trace steps and return
        the assembled ChatResponse. Used by both sync ``run`` and the
        upcoming ``run_stream`` paths so the final shape is identical.
        """
        trace.extend(
            [
                WorkflowStep(
                    id="citation_check",
                    label="Citation and source check",
                    status="completed",
                    detail=_citation_detail(citations),
                    references=citations,
                ),
                WorkflowStep(
                    id="final_response",
                    label="Final conclusion",
                    status="completed",
                    detail=answer,
                    references=citations,
                ),
            ]
        )
        return ChatResponse(
            answer=answer,
            in_scope=True,
            citations=citations,
            next_steps=next_steps,
            next_step_details=next_step_details,
            task_plan=task_plan,
            evidence_coverage=evidence_coverage,
            process_state=process_state,
            compiled_context=compiled_context,
            compiled_context_used=bool(compiled_context),
            resolved_entities=resolved_entities,
            draft_contexts=draft_contexts,
            confidence=0.72 if citations else 0.45,
            source_version=source_version,
            notice=_degradation_notice(audit),
            workflow_trace=trace,
            audit=audit,
        )

    def run_stream(self, request: ChatRequest) -> Iterator[dict]:
        """Generator that yields workflow events for SSE streaming.

        Event shapes:
          - ``{"type": "stage", "step": <WorkflowStep>}`` — a pre-LLM
            workflow stage just completed (scope classifier, task plan,
            retriever, reranker, ...). Lets the UI paint the inspector
            progressively, before the LLM has even started.
          - ``{"type": "delta", "text": <str>}`` — one LLM token chunk
          - ``{"type": "response", "response": ChatResponse}`` — final answer
          - ``{"type": "error", "message": <str>}`` — runner crashed

        The actual workflow runs in a worker thread, with ``run``
        configured to push stage updates AND LLM token deltas into a
        shared ``queue.Queue``. This generator drains the queue,
        forwarding each item as the appropriate event, and finally
        yields ``response`` once the worker thread completes.

        End-to-end time isn't reduced — the pre-LLM phase still blocks
        on retrieval / W3C API / GitHub calls — but the user sees the
        workflow inspector populate stage-by-stage as those steps
        complete, and the answer text streams in the moment the LLM
        starts producing tokens.
        """
        import queue
        import threading

        sentinel = object()
        event_queue: "queue.Queue[object]" = queue.Queue()
        result_holder: dict[str, ChatResponse | BaseException] = {}

        def push_delta(delta: str) -> None:
            event_queue.put(("delta", delta))

        def push_stage(step: "WorkflowStep") -> None:
            event_queue.put(("stage", step))

        # Capture the request id BEFORE entering the worker thread.
        # ``threading.Thread`` does not inherit contextvars (unlike
        # ``concurrent.futures.Executor.submit``), so without this the
        # worker's structured log lines would all carry request_id="-"
        # and lose correlation with the originating /chat/stream call.
        parent_request_id = get_request_id()

        def worker() -> None:
            set_request_id(parent_request_id)
            try:
                response = self.run(
                    request, stream_sink=push_delta, stage_sink=push_stage
                )
                result_holder["response"] = response
            except BaseException as exc:
                # Propagated to the SSE caller via the ``error`` event in
                # the generator's drain loop. We use BaseException so
                # KeyboardInterrupt / SystemExit are also surfaced
                # rather than silently swallowed inside the worker.
                result_holder["error"] = exc
            finally:
                event_queue.put(sentinel)

        thread = threading.Thread(target=worker, name="chat-workflow-stream", daemon=True)
        thread.start()

        while True:
            item = event_queue.get()
            if item is sentinel:
                break
            kind, payload = item  # type: ignore[misc]
            if kind == "delta":
                yield {"type": "delta", "text": str(payload)}
            elif kind == "stage":
                yield {"type": "stage", "step": payload}
        thread.join()

        if "error" in result_holder:
            yield {"type": "error", "message": str(result_holder["error"])}
            return

        response = result_holder.get("response")
        if response is not None:
            yield {"type": "response", "response": response}


_ENGLISH_STOPWORDS = frozenset({
    "the", "a", "an", "of", "for", "to", "in", "on", "is", "are", "was", "were",
    "do", "does", "did", "can", "could", "will", "would", "should", "may", "might",
    "and", "or", "but", "not", "tell", "me", "us", "you", "i", "we", "they",
    "what", "why", "how", "when", "where", "who", "now", "again", "more", "also",
    "any", "all", "some", "please", "ok", "okay",
})
_CJK_FUNCTION_CHARS = frozenset("的是吗呢了过被把就也都还又咋么么了着")


_REF_PUNCT_RE = re.compile(r"[^a-z0-9一-鿿\s]+")
_REF_TOKEN_RE = re.compile(r"[a-z0-9]+|[一-鿿]+")
_REF_ENGLISH_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _is_pure_reference(message: str) -> bool:
    """True when message has no new content beyond referential/function words.

    Used to allow follow-ups like "then?", "继续", "那下一步呢？" to inherit scope,
    while blocking "how about Beijing" (which carries the new noun "Beijing").
    """
    text = message.strip().lower()
    if not text:
        return True
    # Strip known follow-up markers (these signal continuation, not new content).
    cleaned = text
    for marker in sorted(FOLLOW_UP_MARKERS, key=len, reverse=True):
        cleaned = cleaned.replace(marker.lower(), " ")
    cleaned = _REF_PUNCT_RE.sub(" ", cleaned).strip()
    if not cleaned:
        return True
    for token in _REF_TOKEN_RE.findall(cleaned):
        if _REF_ENGLISH_TOKEN_RE.fullmatch(token):
            if token not in _ENGLISH_STOPWORDS:
                return False
        else:
            if not all(ch in _CJK_FUNCTION_CHARS for ch in token):
                return False
    return True


def _degradation_notice(audit: dict) -> str | None:
    """User-facing notice when the answer is a limited fallback, not a real
    model generation. Returns None for healthy generations and for the
    deliberately-offline ``template`` provider.
    """
    generation = audit.get("model_generation")
    if generation == "template_fallback":
        error = audit.get("model_error")
        detail = f" ({error})" if error else ""
        return (
            f"The language model could not be reached{detail}. This is a limited "
            "answer built only from the retrieved sources — check the model "
            "provider configuration."
        )
    if generation == "override_rejected":
        detail = audit.get("model_error_detail")
        suffix = f": {detail}" if detail else ""
        return (
            f"The requested model provider was rejected{suffix}. This is a "
            "limited answer built only from the retrieved sources."
        )
    if isinstance(generation, str) and generation.endswith("empty_fallback"):
        return (
            "The language model returned an empty response. This is a limited "
            "answer built only from the retrieved sources."
        )
    return None


def _model_generation_detail(model_generation: str, model: str) -> str:
    if model_generation == "ollama":
        return f"Used local Ollama model {model}; output was accepted after harness cleanup."
    if model_generation == "openai_compatible":
        return f"Used OpenAI-compatible model {model}; output was accepted after harness cleanup."
    if model_generation == "openai_compatible_empty_fallback":
        return (
            f"Called OpenAI-compatible model {model}, but its usable answer was empty, "
            "so the workflow used the validated conservative answer."
        )
    if model_generation == "bedrock":
        return f"Used AWS Bedrock model {model}; output was accepted after harness cleanup."
    if model_generation == "bedrock_empty_fallback":
        return (
            f"Called AWS Bedrock model {model}, but its usable answer was empty, "
            "so the workflow used the validated conservative answer."
        )
    if model_generation == "ollama_empty_fallback":
        return (
            f"Called local Ollama model {model}, but its usable answer was empty after removing "
            "thinking content, so the workflow used the validated conservative answer."
        )
    if model_generation == "template_fallback":
        return f"Attempted local Ollama model {model}, then fell back to the validated conservative answer."
    return "Used the deterministic validated answer template."


def _selected_model(request: ChatRequest, settings: Settings) -> str:
    if request.model:
        return request.model
    if is_openai_compatible_provider(settings.llm_provider):
        return settings.openai_compatible_model
    # Ollama and Bedrock both read llm_model.
    return settings.llm_model


def _router_detail(decision: LLMRouterDecision) -> str:
    if not decision.attempted:
        return decision.reason or "Router was not attempted for this question."
    verdict = "likely in scope" if decision.likely_in_scope else "not likely in scope"
    pieces = [
        f"Router judged the question as {verdict}",
        f"intent={decision.intent_type}",
        f"confidence={round(decision.confidence * 100)}%",
    ]
    if decision.needed_sources:
        pieces.append(f"sources={', '.join(source.value for source in decision.needed_sources)}")
    if decision.error:
        pieces.append(f"error={decision.error}")
    return "; ".join(pieces) + "."


def _router_augmented_query(query: str, decision: LLMRouterDecision | None) -> str:
    if not decision or not decision.attempted or not decision.likely_in_scope:
        return query
    lines = [
        query,
        "",
        "LLM router hints for retrieval only; not authoritative source material:",
        f"intent_type={decision.intent_type}",
        f"confidence={decision.confidence}",
    ]
    if decision.needed_sources:
        lines.append(f"needed_sources={', '.join(source.value for source in decision.needed_sources)}")
    if decision.entities_to_resolve:
        lines.append(f"entities_to_resolve={', '.join(decision.entities_to_resolve)}")
    if decision.search_hints:
        lines.append("search_hints:")
        lines.extend(f"- {hint}" for hint in decision.search_hints)
    # ``risk_flags`` is intentionally NOT appended to the retrieval query.
    # The router emits them as categorical labels (e.g. "Horizontal Review",
    # "Transition"), and the retriever's downstream substring matchers
    # (``_is_horizontal_review_query``, topic-coverage injection) would then
    # treat the flag value as if the user had asked about that topic. The
    # flags are still captured in the audit trace for inspection.
    return "\n".join(lines)


def _draft_context_detail(contexts: list[DraftContext]) -> str:
    if not contexts:
        return (
            "No official GitHub draft repository was resolved from the W3C API entities, or the question did not "
            "need draft-level context. The workflow did not search the full github.com/w3c organization."
        )
    repos = ", ".join(context.repo_full_name for context in contexts[:3])
    return (
        f"Resolved limited read-only draft context from official GitHub repositories ({repos}). "
        "This context is used to understand the draft/repo, not as normative W3C Process authority."
    )


def _compiled_context_detail(context: CompiledContext | None) -> str:
    if not context:
        return (
            "No compiled spec page was available for a high-confidence specification entity. "
            "The workflow continued with raw Process and Guidebook retrieval."
        )
    return (
        f"Loaded compiled {context.kind} context for {context.key} to shape answer focus and next-step candidates. "
        "Raw Process and Guidebook citations remain required for normative claims."
    )


def _task_plan_detail(task_plan: TaskPlan) -> str:
    pieces = [
        f"Intent: {task_plan.intent_type}",
        f"goal: {task_plan.user_goal}",
    ]
    if task_plan.current_stage or task_plan.target_stage:
        pieces.append(f"transition: {task_plan.current_stage or 'unknown'} -> {task_plan.target_stage or 'unknown'}")
    if task_plan.spec_or_group:
        pieces.append(f"subject: {task_plan.spec_or_group}")
    if task_plan.needed_sources:
        pieces.append(f"needed sources: {', '.join(source.value for source in task_plan.needed_sources)}")
    return "; ".join(pieces) + "."


def _citation_detail(citations: list[Citation]) -> str:
    if not citations:
        return "No authoritative citations were available, so the answer remained conservative."
    return (
        "Checked that the answer references trusted W3C sources. Normative claims must be grounded "
        "in Process citations; Guidebook-only material is treated as practice guidance."
    )


def _merge_citations(primary: list[Citation], extra: list[Citation], limit: int = 10) -> list[Citation]:
    merged: list[Citation] = []
    seen: set[str] = set()
    for citation in [*primary, *extra]:
        key = f"{citation.url}#{citation.section_id or ''}#{citation.heading_path or ''}"
        if key in seen:
            continue
        seen.add(key)
        merged.append(citation)
        if len(merged) >= limit:
            break
    return merged


def _entity_resolution_detail(entities: list[W3CEntity]) -> str:
    if not entities:
        return (
            "Checked the public W3C API for matching specifications and groups, but no strong entity "
            "match was found. API data was not used as normative Process authority."
        )
    labels = ", ".join(f"{entity.entity_type}: {entity.title}" for entity in entities[:3])
    return (
        f"Resolved public W3C API entities ({labels}). API data is used for entity/status grounding only, "
        "not as normative Process authority."
    )
