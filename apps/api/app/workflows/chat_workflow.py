import logging
import re
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TypedDict

from app.core.config import Settings
from app.models.schemas import ChatRequest, ChatResponse, ClassifyResponse, Citation, CompiledContext, DraftContext, LLMRouterDecision, NextStep, ProcessState, EvidenceCoverage, SourceVersion, TaskPlan, W3CEntity, WorkflowStep
from app.rag.retriever import Retriever
from app.services.answering import build_grounded_answer, build_next_step_details, build_refusal
from app.services.compiled_context import CompiledContextStore
from app.services.context import FOLLOW_UP_MARKERS, build_contextual_query, build_entity_augmented_query
from app.services.evidence import check_evidence_coverage
from app.services.github_context import GitHubDraftContextClient, build_draft_context_augmented_query
from app.services.llm_router import LLMRouter
from app.services.ollama import OllamaClient
from app.services.openai_compatible import OpenAICompatibleClient
from app.services.process_state import extract_process_state
from app.services.live_fetch import fetch_page_excerpt
from app.services.scope import classify_scope
from app.services.action_surfaces import format_surfaces_for_prompt, surfaces_for_intent
from app.services.citation_verifier import verify_citations
from app.services.cross_encoder_reranker import (
    CrossEncoderRerankResult,
    MissingDependencyError as _CrossEncoderMissing,
    rerank_with_cross_encoder,
)
from app.services.provider_override import ProviderOverrideError, build_override_client
from app.services.query_rewriter import rewrite_query
from app.services.reranker import rerank_citations
from app.services.task_planner import build_planned_retrieval_query, plan_task
from app.services.w3c_api import W3CAPIClient


logger = logging.getLogger(__name__)


class ChatState(TypedDict, total=False):
    request: ChatRequest
    classification: ClassifyResponse
    response: ChatResponse


@dataclass
class _PreparedAnswer:
    """All workflow state produced before the final LLM generation call.

    The pre-LLM phase of the workflow is identical for sync (``run``) and
    streaming (``run_stream``) paths — both consume one of these. The two
    methods only diverge in HOW they invoke the LLM (a single
    ``generate_answer`` blocking call vs an iterator of token deltas).

    Keeping this state in one bag means the assembly phase (workflow trace
    tail + ``ChatResponse`` construction) is also single-source.
    """

    request: ChatRequest
    classification: ClassifyResponse
    citations: list[Citation]
    next_steps: list[str]
    next_step_details: list[NextStep]
    trace: list[WorkflowStep]
    audit: dict
    source_version: SourceVersion
    task_plan: TaskPlan | None
    evidence_coverage: EvidenceCoverage | None
    process_state: ProcessState | None
    compiled_context: CompiledContext | None
    resolved_entities: list[W3CEntity]
    draft_contexts: list[DraftContext]
    supplementary_context: str | None
    template_answer: str
    selected_model: str
    generation_provider: str
    provider_label: str
    generation_client: object | None
    safe_history: list
    action_surfaces_text: str
    initial_model_generation: str
    router_model: str


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
        self.w3c_api_client = w3c_api_client or W3CAPIClient(settings)
        self.github_context_client = github_context_client or GitHubDraftContextClient(settings)
        router_client = (
            self.openai_compatible_client
            if settings.llm_provider in {"openai", "openai-compatible", "openrouter"}
            else self.ollama_client
        )
        self.llm_router = llm_router or LLMRouter(settings, router_client)
        self.compiled_context_store = compiled_context_store or CompiledContextStore(
            settings,
            retriever=self.retriever,
            w3c_api_client=self.w3c_api_client,
            github_context_client=self.github_context_client,
        )

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

        def _emit_stage(step: "WorkflowStep") -> None:
            if stage_sink is None:
                return
            try:
                stage_sink(step)
            except Exception:  # pragma: no cover - sink misbehaviour
                logger.warning("stage_sink raised; continuing without progress events")

        def _record(trace_list: list, step: "WorkflowStep") -> None:
            """Append ``step`` to the trace and emit a stage event.

            Use instead of ``_record(trace, step)`` so SSE consumers see
            each completed workflow node as it lands, not all at once
            at the end of the request.
            """
            trace_list.append(step)
            _emit_stage(step)

        contextual_query = build_contextual_query(request.message, request.history)
        used_contextual_query = contextual_query != request.message
        classification = self.classify(request)
        router_decision: LLMRouterDecision | None = None

        # Layer 2: re-classify using the context-enriched query for follow-up questions.
        # IMPORTANT: a follow-up marker alone (e.g. "how about X") does not justify
        # inheriting scope from history — that lets unrelated topics ride along
        # because PROCESS_TOPICS keywords in prior turns leak into contextual_query.
        # Only flip to in_scope when the contextual query produces a STRONG match
        # AND the original message is either purely referential (no new noun) or
        # itself contains at least one weak topic word.
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
            or (
                self.settings.openai_compatible_model
                if self.settings.llm_provider in {"openai", "openai-compatible", "openrouter"}
                else self.settings.llm_model
            )
        )

        # Layer 3a: LLM router rescues questions that keyword matching missed
        # Layer 3b: LLM router also validates weak keyword matches (confidence < 0.7) to filter false positives
        # Optimization: when router needs to run AND classification is already weakly in-scope,
        # speculatively prefetch W3C API entities in parallel — router hints only annotate
        # the query for retrieval, they don't affect entity matching.
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
        source_version = SourceVersion(indexed_at=datetime.now(timezone.utc).isoformat())
        trace: list[WorkflowStep] = [
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
        ]
        override = request.provider_override
        audit = {
            "workflow": "harness_v1",
            "matched_topics": classification.matched_topics,
            "injection_risk": classification.injection_risk,
            # Audit reflects what the workflow WILL run with. For user overrides
            # we record kind+model only — never base_url and never api_key,
            # so the audit blob is safe to surface or persist.
            "llm_provider": f"override:{override.kind}" if override else self.settings.llm_provider,
            "llm_model": override.model if override else _selected_model(request, self.settings),
            "used_contextual_query": used_contextual_query,
        }
        if router_decision:
            audit["llm_router"] = router_decision.model_dump(mode="json")
            _record(trace, 
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
            answer = build_refusal(request.locale)
            trace.extend(
                [
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
                ]
            )
            return ChatResponse(
                answer=answer,
                in_scope=False,
                confidence=0.98,
                source_version=source_version,
                refusal_reason=classification.reason,
                workflow_trace=trace,
                audit=audit,
            )

        if used_contextual_query:
            _record(trace, 
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

        # Multi-query retrieval: ask the LLM for 1-3 W3C-canonical rewrites of
        # the user's question and merge their citations with the primary pass.
        # Skipped when the model provider is "template" (e.g. eval workflow)
        # to keep evals deterministic and offline.
        rewrite_variants: list[str] = []
        if self.settings.llm_provider != "template":
            router_client = (
                self.openai_compatible_client
                if self.settings.llm_provider in {"openai", "openai-compatible", "openrouter"}
                else self.ollama_client
            )
            try:
                rewrite_result = rewrite_query(
                    request.message,
                    settings=self.settings,
                    client=router_client,
                    model=router_model,
                )
                rewrite_variants = rewrite_result.variants
                audit["query_rewrites"] = rewrite_variants
                if rewrite_result.error:
                    audit["query_rewrite_error"] = rewrite_result.error
            except Exception as exc:  # pragma: no cover - non-fatal
                logger.warning("Query rewriter failed; continuing with single-query retrieval", exc_info=exc)
                audit["query_rewrite_error"] = type(exc).__name__

        # Pass the raw user message separately so the retriever can rank by
        # what the user actually asked instead of being biased by the 10+
        # lines of task-planner / entity / router metadata in retrieval_query.
        citations = self.retriever.retrieve(retrieval_query, user_message=request.message)
        if rewrite_variants:
            extra_hits: list[Citation] = []
            for variant in rewrite_variants:
                try:
                    extra_hits.extend(
                        self.retriever.retrieve(variant, user_message=variant)
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
            if not used_cross_encoder and self.settings.llm_provider != "template":
                router_client = (
                    self.openai_compatible_client
                    if self.settings.llm_provider in {"openai", "openai-compatible", "openrouter"}
                    else self.ollama_client
                )
                rerank_result = rerank_citations(
                    request.message,
                    citations,
                    settings=self.settings,
                    client=router_client,
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
        # When the user supplies a per-request provider override we use THEIR
        # endpoint and model id; otherwise we fall back to the server defaults.
        # The api_key never leaves this function — it is consumed by the
        # one-shot client and intentionally never written to ``audit``.
        if override is not None:
            selected_model = override.model
            generation_provider = override.kind
            audit["model_provider_source"] = "override"
        else:
            selected_model = _selected_model(request, self.settings)
            generation_provider = self.settings.llm_provider
            audit["model_provider_source"] = "default"
        model_generation = "template"

        # When injection language is detected we throw away the conversation
        # history before handing it to the model. The audit field already
        # records that injection was suspected; this is the enforcement that
        # makes the safety_note actually safe.
        safe_history = [] if classification.injection_risk else request.history

        # Resolve which client handles this request. Override clients are
        # built per-request and never cached.
        generation_client = None
        if override is not None:
            try:
                generation_client = build_override_client(override, self.settings)
            except ProviderOverrideError as exc:
                model_generation = "override_rejected"
                audit["model_generation"] = model_generation
                audit["model_error"] = "ProviderOverrideError"
                audit["model_error_detail"] = str(exc)
                logger.info("Provider override rejected: %s", exc)
        elif generation_provider == "ollama":
            generation_client = self.ollama_client
        elif generation_provider in {"openai", "openai-compatible", "openrouter"}:
            generation_client = self.openai_compatible_client

        # Audit/log names use underscore form for backwards compatibility
        # with existing tests, logs, and eval cases.
        provider_label = generation_provider.replace("-", "_")
        # Resolve concrete W3C action surfaces (issue trackers, mailing lists,
        # forms) for this intent so the model can end each step with a
        # "do X at Y" instruction instead of vague guidance.
        action_surfaces_text = format_surfaces_for_prompt(
            surfaces_for_intent(task_plan.intent_type if task_plan else None)
        )
        if action_surfaces_text:
            audit["action_surfaces_intent"] = task_plan.intent_type if task_plan else None
        if generation_client is not None:
            generation_kwargs = dict(
                model=selected_model,
                question=request.message,
                locale=request.locale,
                citations=citations,
                fallback_answer=answer,
                fallback_next_steps=next_steps,
                history=safe_history,
                entities=resolved_entities,
                task_plan=task_plan,
                process_state=process_state,
                evidence_coverage=evidence_coverage,
                draft_contexts=draft_contexts,
                compiled_context=compiled_context,
                supplementary_context=supplementary_context,
                action_surfaces_text=action_surfaces_text,
            )
            stream_supported = stream_sink is not None and hasattr(generation_client, "stream_answer")
            try:
                if stream_supported:
                    # Real-token streaming path. Each delta is forwarded to
                    # the sink as it arrives; the assembled answer is the
                    # concatenation cleaned with the same harness post-
                    # processing as the sync path.
                    from app.services.ollama import _clean_model_text  # local to avoid moving import to top

                    chunks: list[str] = []
                    for delta in generation_client.stream_answer(**generation_kwargs):
                        if delta:
                            chunks.append(delta)
                            try:
                                stream_sink(delta)
                            except Exception:  # pragma: no cover - sink misbehaviour
                                logger.warning("stream_sink raised; continuing without forwarding deltas")
                                stream_sink = None  # type: ignore[assignment]
                    streamed_text = _clean_model_text("".join(chunks))
                    if streamed_text:
                        answer = streamed_text
                        model_generation = f"{provider_label}_stream"
                        audit["model_generation"] = model_generation
                    else:
                        model_generation = f"{provider_label}_stream_empty_fallback"
                        audit["model_generation"] = model_generation
                else:
                    generation = generation_client.generate_answer(**generation_kwargs)
                    if generation.text:
                        answer = generation.text
                        model_generation = provider_label
                        audit["model_generation"] = model_generation
                    else:
                        model_generation = f"{provider_label}_empty_fallback"
                        audit["model_generation"] = model_generation
            except Exception as exc:  # pragma: no cover - external service fallback
                model_generation = "template_fallback"
                audit["model_generation"] = model_generation
                audit["model_error"] = type(exc).__name__
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

        _record(trace, 
            WorkflowStep(
                id="answer_generator",
                label="Answer generation",
                status="completed",
                detail=_model_generation_detail(model_generation, selected_model),
                references=citations,
            )
        )

        # Post-generation citation verification: for each ``[Sn]`` tag in the
        # answer, check that the cited excerpt actually supports the
        # surrounding claim. Strip tags that fail verification — the claim
        # stays, but it's no longer falsely attributed. Skipped in template
        # mode (eval), when the answer came from template fallback, or when
        # no LLM client is available.
        if (
            self.settings.llm_provider != "template"
            and model_generation not in {"template", "template_fallback", "override_rejected"}
            and citations
        ):
            verifier_client = (
                self.openai_compatible_client
                if self.settings.llm_provider in {"openai", "openai-compatible", "openrouter"}
                else self.ollama_client
            )
            verification = verify_citations(
                answer,
                citations,
                settings=self.settings,
                client=verifier_client,
                model=router_model,
            )
            if verification.stripped_pairs:
                answer = verification.answer
                audit["citation_verifier"] = {
                    "model": verification.model,
                    "stripped_count": len(verification.stripped_pairs),
                }
                _record(trace, 
                    WorkflowStep(
                        id="citation_verifier",
                        label="Citation verifier",
                        status="completed",
                        detail=(
                            f"Stripped {len(verification.stripped_pairs)} citation tag(s) the verifier judged unsupported "
                            "by their excerpts. The claim text is preserved; only the misleading attribution is removed."
                        ),
                    )
                )
            elif verification.skipped_reason:
                audit["citation_verifier_skipped"] = verification.skipped_reason

        if classification.injection_risk:
            audit["safety_note"] = "Potential prompt injection detected; answer constrained to trusted sources."
            _record(trace, 
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
            source_version=source_version,
            trace=trace,
            audit=audit,
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

        def worker() -> None:
            try:
                response = self.run(
                    request, stream_sink=push_delta, stage_sink=push_stage
                )
                result_holder["response"] = response
            except BaseException as exc:  # pragma: no cover - propagated to caller
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
    if settings.llm_provider in {"openai", "openai-compatible", "openrouter"}:
        return settings.openai_compatible_model
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
