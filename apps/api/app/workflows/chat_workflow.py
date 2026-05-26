from datetime import datetime, timezone
from typing import TypedDict

from app.core.config import Settings
from app.models.schemas import ChatRequest, ChatResponse, ClassifyResponse, Citation, CompiledContext, DraftContext, LLMRouterDecision, SourceVersion, TaskPlan, W3CEntity, WorkflowStep
from app.rag.retriever import Retriever
from app.services.answering import build_grounded_answer, build_next_step_details, build_refusal
from app.services.compiled_context import CompiledContextStore
from app.services.context import build_contextual_query, build_entity_augmented_query
from app.services.evidence import check_evidence_coverage
from app.services.github_context import GitHubDraftContextClient, build_draft_context_augmented_query
from app.services.llm_router import LLMRouter
from app.services.ollama import OllamaClient
from app.services.openai_compatible import OpenAICompatibleClient
from app.services.process_state import extract_process_state
from app.services.live_fetch import fetch_page_excerpt
from app.services.scope import classify_scope
from app.services.task_planner import build_planned_retrieval_query, plan_task
from app.services.w3c_api import W3CAPIClient


class ChatState(TypedDict, total=False):
    request: ChatRequest
    classification: ClassifyResponse
    response: ChatResponse


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
        decision = classify_scope(request.message)
        return ClassifyResponse(
            in_scope=decision.in_scope,
            reason=decision.reason,
            matched_topics=decision.matched_topics,
            injection_risk=decision.injection_risk,
            confidence=decision.confidence,
        )

    def run(self, request: ChatRequest) -> ChatResponse:
        contextual_query = build_contextual_query(request.message, request.history)
        used_contextual_query = contextual_query != request.message
        classification = self.classify(request)
        router_decision: LLMRouterDecision | None = None

        # Layer 2: re-classify using the context-enriched query for follow-up questions
        if not classification.in_scope and used_contextual_query:
            contextual_decision = classify_scope(contextual_query)
            if contextual_decision.in_scope:
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
        _needs_router = not classification.in_scope or classification.confidence < 0.7
        if _needs_router:
            router_decision = self.llm_router.route(
                contextual_query,
                request.history,
                model=router_model,
            )
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
        audit = {
            "workflow": "harness_v1",
            "matched_topics": classification.matched_topics,
            "injection_risk": classification.injection_risk,
            "llm_provider": self.settings.llm_provider,
            "llm_model": _selected_model(request, self.settings),
            "used_contextual_query": used_contextual_query,
        }
        if router_decision:
            audit["llm_router"] = router_decision.model_dump(mode="json")
            trace.append(
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
            trace.append(
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
        trace.append(
            WorkflowStep(
                id="task_planner",
                label="Task planner",
                status="completed",
                detail=_task_plan_detail(task_plan),
            )
        )

        resolved_entities: list[W3CEntity] = []
        try:
            resolved_entities = self.w3c_api_client.resolve_entities(routed_query)
            audit["resolved_entities"] = [entity.model_dump(mode="json") for entity in resolved_entities]
            trace.append(
                WorkflowStep(
                    id="w3c_api_resolver",
                    label="W3C API entity resolver",
                    status="completed",
                    detail=_entity_resolution_detail(resolved_entities),
                )
            )
        except Exception as exc:  # pragma: no cover - external service fallback
            audit["w3c_api_error"] = str(exc)
            trace.append(
                WorkflowStep(
                    id="w3c_api_resolver",
                    label="W3C API entity resolver",
                    status="failed",
                    detail="W3C API entity lookup failed; continuing with Process and Guidebook corpus retrieval.",
                )
            )

        draft_contexts: list[DraftContext] = []
        compiled_context: CompiledContext | None = None
        try:
            draft_contexts = self.github_context_client.resolve_contexts(routed_query, resolved_entities, task_plan)
            audit["draft_contexts"] = [context.model_dump(mode="json") for context in draft_contexts]
            trace.append(
                WorkflowStep(
                    id="draft_context_resolver",
                    label="GitHub draft context resolver",
                    status="completed" if draft_contexts else "skipped",
                    detail=_draft_context_detail(draft_contexts),
                )
            )
        except Exception as exc:  # pragma: no cover - external service fallback
            audit["draft_context_error"] = str(exc)
            trace.append(
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
            compiled_context = self.compiled_context_store.resolve(resolved_entities)
            if compiled_context:
                audit["compiled_context"] = compiled_context.model_dump(mode="json")
            trace.append(
                WorkflowStep(
                    id="compiled_context_resolver",
                    label="Compiled knowledge resolver",
                    status="completed" if compiled_context else "skipped",
                    detail=_compiled_context_detail(compiled_context),
                )
            )
        except Exception as exc:  # pragma: no cover - local filesystem fallback
            audit["compiled_context_error"] = str(exc)
            trace.append(
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
            trace.append(
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

        citations = self.retriever.retrieve(retrieval_query)
        process_state = extract_process_state(retrieval_query, citations, resolved_entities)
        evidence_coverage = check_evidence_coverage(
            plan=task_plan,
            citations=citations,
            entities=resolved_entities,
            process_state=process_state,
            compiled_context=compiled_context,
        )
        trace.append(
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
                targeted_hits.extend(self.retriever.retrieve(targeted_query))
            citations = _merge_citations(citations, targeted_hits)
            process_state = extract_process_state(retrieval_query, citations, resolved_entities)
            evidence_coverage = check_evidence_coverage(
                plan=task_plan,
                citations=citations,
                entities=resolved_entities,
                process_state=process_state,
                compiled_context=compiled_context,
            )
            trace.append(
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
        trace.append(
            WorkflowStep(
                id="evidence_coverage",
                label="Evidence coverage check",
                status="completed" if evidence_coverage.status == "sufficient" else "failed",
                detail=evidence_coverage.summary,
                references=citations,
            )
        )
        trace.append(
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
                )
            trace.append(
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
        selected_model = _selected_model(request, self.settings)
        model_generation = "template"

        if self.settings.llm_provider == "ollama":
            try:
                generation = self.ollama_client.generate_answer(
                    model=selected_model,
                    question=request.message,
                    locale=request.locale,
                    citations=citations,
                    fallback_answer=answer,
                    fallback_next_steps=next_steps,
                    history=request.history,
                    entities=resolved_entities,
                    task_plan=task_plan,
                    process_state=process_state,
                    evidence_coverage=evidence_coverage,
                    draft_contexts=draft_contexts,
                    compiled_context=compiled_context,
                    supplementary_context=supplementary_context,
                )
                if generation.text:
                    answer = generation.text
                    model_generation = "ollama"
                    audit["model_generation"] = model_generation
                else:
                    model_generation = "ollama_empty_fallback"
                    audit["model_generation"] = model_generation
            except Exception as exc:  # pragma: no cover - external service fallback
                model_generation = "template_fallback"
                audit["model_generation"] = model_generation
                audit["model_error"] = str(exc)
        elif self.settings.llm_provider in {"openai", "openai-compatible", "openrouter"}:
            try:
                generation = self.openai_compatible_client.generate_answer(
                    model=selected_model,
                    question=request.message,
                    locale=request.locale,
                    citations=citations,
                    fallback_answer=answer,
                    fallback_next_steps=next_steps,
                    history=request.history,
                    entities=resolved_entities,
                    task_plan=task_plan,
                    process_state=process_state,
                    evidence_coverage=evidence_coverage,
                    draft_contexts=draft_contexts,
                    compiled_context=compiled_context,
                    supplementary_context=supplementary_context,
                )
                if generation.text:
                    answer = generation.text
                    model_generation = "openai_compatible"
                    audit["model_generation"] = model_generation
                else:
                    model_generation = "openai_compatible_empty_fallback"
                    audit["model_generation"] = model_generation
            except Exception as exc:  # pragma: no cover - external service fallback
                model_generation = "template_fallback"
                audit["model_generation"] = model_generation
                audit["model_error"] = str(exc)
        else:
            audit["model_generation"] = model_generation

        trace.append(
            WorkflowStep(
                id="answer_generator",
                label="Answer generation",
                status="completed",
                detail=_model_generation_detail(model_generation, selected_model),
                references=citations,
            )
        )

        if classification.injection_risk:
            audit["safety_note"] = "Potential prompt injection detected; answer constrained to trusted sources."
            trace.append(
                WorkflowStep(
                    id="injection_guard",
                    label="Prompt-injection guard",
                    status="completed",
                    detail="Potential injection language was detected; user-provided process claims were not treated as authoritative.",
                )
            )

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
    if decision.risk_flags:
        lines.append(f"risk_flags={', '.join(decision.risk_flags)}")
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
