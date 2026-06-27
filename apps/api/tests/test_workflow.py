from app.core.config import Settings
from app.models.schemas import ChatRequest, ChatTurn, Citation, CompiledContext, CompiledFreshness, CompiledProvenance, LLMRouterDecision, SourceType, W3CEntity
from app.rag.retriever import DEFAULT_PROCESS_CITATION
from app.workflows.chat_workflow import ChatWorkflow


def _test_settings() -> Settings:
    return Settings(llm_provider="template", w3c_api_enabled=False)


def test_workflow_refuses_out_of_scope_question() -> None:
    response = ChatWorkflow(_test_settings()).run(ChatRequest(message="Tell me a joke"))
    assert not response.in_scope
    assert response.refusal_reason
    assert response.workflow_trace[-1].id == "final_response"


def test_workflow_returns_citations_for_process_question() -> None:
    response = ChatWorkflow(_test_settings()).run(ChatRequest(message="W3C Process 中 CR 到 REC 怎么走？"))
    assert response.in_scope
    assert response.citations
    assert response.next_steps
    # Allow the reranker step to be present or absent: the cross-encoder
    # reranker only fires when sentence-transformers is installed AND the
    # query produces at least 4 candidates. We assert the load-bearing
    # invariants instead of the exact list, so the test passes in both the
    # vanilla-deps environment and the cross-encoder-enabled one.
    step_ids = [step.id for step in response.workflow_trace]
    must_have = [
        "scope_classifier",
        "task_planner",
        "w3c_api_resolver",
        "draft_context_resolver",
        "compiled_context_resolver",
        "retriever",
        "evidence_coverage",
        "process_state",
        "answer_generator",
        "citation_check",
        "final_response",
    ]
    for required in must_have:
        assert required in step_ids, f"missing workflow step: {required}"
    # Final conclusion stays last regardless of optional reranker.
    assert step_ids[-1] == "final_response"
    assert response.process_state
    assert response.task_plan
    assert response.evidence_coverage


def test_workflow_audit_includes_degraded_field_empty_on_healthy_run() -> None:
    """``audit["degraded"]`` is the canonical "what fell back" channel.
    On a healthy template-mode run nothing should be in it; the field
    exists so operators can distinguish "no degradation" (empty list)
    from "field missing" (old build, can't tell)."""
    response = ChatWorkflow(_test_settings()).run(
        ChatRequest(message="What is a Working Draft?")
    )
    assert "degraded" in response.audit
    assert isinstance(response.audit["degraded"], list)
    # Template mode never calls external services, so degradation tags
    # like w3c_api_unavailable / llm_generation_failed should not appear.
    assert response.audit["degraded"] == []


def test_workflow_returns_topic_specific_answers() -> None:
    workflow = ChatWorkflow(_test_settings())

    transition = workflow.run(
        ChatRequest(
            message="What should a CSS specification do next to move from CR to REC?",
            locale="en",
        )
    )
    objection = workflow.run(
        ChatRequest(
            message="How does the W3C Process handle a Formal Objection?",
            locale="en",
        )
    )

    assert transition.answer != objection.answer
    assert "transition" in transition.answer.lower()
    assert "formal objection" in objection.answer.lower()
    assert transition.process_state
    assert transition.process_state.current_stage == "CR"
    assert transition.process_state.target_stage == "REC"


def test_workflow_returns_contextual_next_steps() -> None:
    workflow = ChatWorkflow(_test_settings())

    transition = workflow.run(ChatRequest(message="怎么准备 transition request 和 milestones？"))
    staff_contact = workflow.run(ChatRequest(message="Staff Contact 的职责是什么？"))
    chair_meeting = workflow.run(ChatRequest(message="Chair 怎么准备 W3C group meeting？"))

    assert transition.next_steps != staff_contact.next_steps
    assert staff_contact.next_steps != chair_meeting.next_steps
    assert transition.next_step_details
    assert any(step.source_type == "guide" for step in transition.next_step_details)
    assert any(step.source_url for step in staff_contact.next_step_details)
    assert any("Staff Contact" in step for step in staff_contact.next_steps)
    assert any("meeting" in step.lower() or "会议" in step for step in chair_meeting.next_steps)


def test_workflow_resolves_follow_up_questions_with_history() -> None:
    workflow = ChatWorkflow(_test_settings())

    response = workflow.run(
        ChatRequest(
            message="那下一步呢？",
            history=[
                ChatTurn(role="user", content="Staff Contact 的职责是什么？"),
                ChatTurn(
                    role="assistant",
                    content="Staff Contact helps coordinate Working Group process work using Process and Guidebook sources.",
                ),
            ],
        )
    )

    assert response.in_scope
    assert response.audit["used_contextual_query"]
    assert "query_rewriter" in [step.id for step in response.workflow_trace]
    assert "task_planner" in [step.id for step in response.workflow_trace]
    assert response.process_state
    assert response.process_state.intent in {"coordinate_with_staff_contact", "advance_specification"}
    assert any("staff contact" in (citation.heading_path or "").lower() for citation in response.citations)


class FakeW3CAPIClient:
    def resolve_entities(self, query: str) -> list[W3CEntity]:
        if "css grid" not in query.lower():
            return []
        return [
            W3CEntity(
                entity_type="specification",
                title="CSS Grid Layout Module Level 1",
                shortname="css-grid-1",
                api_url="https://api.w3.org/specifications/css-grid-1",
                public_url="https://www.w3.org/TR/css-grid-1/",
                editor_draft_url="https://w3c.github.io/csswg-drafts/css-grid-1/",
                status="Candidate Recommendation Draft",
                latest_version_url="https://api.w3.org/specifications/css-grid-1/versions/20250326",
                latest_version_date="2025-03-26",
                process_rules_url="https://www.w3.org/policies/process/20231103/",
                deliverers=["Cascading Style Sheets (CSS) Working Group"],
                confidence=0.9,
            )
        ]


class RecordingRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []
        self.user_messages: list[str] = []

    def retrieve(self, query: str, *, user_message: str | None = None) -> list[Citation]:
        self.queries.append(query)
        self.user_messages.append(user_message or query)
        return [DEFAULT_PROCESS_CITATION]


class FakeGitHubDraftContextClient:
    def resolve_contexts(self, query: str, entities: list[W3CEntity], task_plan):  # type: ignore[no-untyped-def]
        from app.models.schemas import DraftContext, DraftSnippet

        if task_plan.intent_type == "charter_or_recharter":
            return [
                DraftContext(
                    repo_full_name="w3c/strategy",
                    repo_url="https://github.com/w3c/strategy",
                    resolved_from="https://github.com/w3c/strategy/issues?q=label%3Acharter",
                    snippets=[
                        DraftSnippet(
                            path="issues/123",
                            title="Review Foo Working Group charter",
                            text="w3c/strategy issue #123; state=open; labels=charter.",
                            url="https://github.com/w3c/strategy/issues/123",
                        )
                    ],
                    retrieval_hints=["w3c/strategy", "charter label", "charter review issue tracker"],
                    confidence=0.88,
                )
            ]
        if "draft" not in query.lower() and "github" not in query.lower():
            return []
        return [
            DraftContext(
                repo_full_name="w3c/csswg-drafts",
                repo_url="https://github.com/w3c/csswg-drafts",
                resolved_from="https://w3c.github.io/csswg-drafts/css-grid-1/",
                default_branch="main",
                latest_commit_sha="abc123def456",
                snippets=[
                    DraftSnippet(
                        path="css-grid-1/Overview.bs",
                        title="CSS Grid Layout Module Level 1",
                        text="Specification source for CSS Grid Layout Module Level 1.",
                        url="https://github.com/w3c/csswg-drafts/blob/main/css-grid-1/Overview.bs",
                    )
                ],
                retrieval_hints=["w3c/csswg-drafts", "CSS Grid Layout Module Level 1"],
                confidence=0.9,
            )
        ]


class FakeCompiledContextStore:
    def resolve(self, entities):  # type: ignore[no-untyped-def]
        shortname = next((entity.shortname for entity in entities if entity.shortname), None)
        if shortname != "css-grid-1":
            return None
        return CompiledContext(
            kind="spec",
            key="css-grid-1",
            title="CSS Grid Layout Module Level 1",
            summary="Compiled CSS Grid summary.",
            current_state="Candidate Recommendation Draft | 2025-03-26",
            next_step_candidates=[
                "Confirm the next Recommendation-track transition and gather the required Process evidence."
            ],
            guide_signals=["Transitions: https://www.w3.org/guide/transitions/"],
            horizontal_review_signals=["Check horizontal review request state before the next transition."],
            charter_signals=[],
            freshness=CompiledFreshness(compiled_at="2026-04-26T00:00:00Z"),
            provenance=CompiledProvenance(
                normative_urls=["https://www.w3.org/policies/process/"],
                guide_urls=["https://www.w3.org/guide/transitions/"],
                operational_urls=["https://api.w3.org/specifications/css-grid-1"],
            ),
            confidence=0.9,
        )


class FakeLLMRouter:
    def route(self, question, history=None, model=None):  # type: ignore[no-untyped-def]
        if "published now" not in question.lower():
            return LLMRouterDecision(reason="Not router-worthy.")
        return LLMRouterDecision(
            attempted=True,
            likely_in_scope=True,
            intent_type="advance_specification",
            needed_sources=[SourceType.process, SourceType.guide],
            entities_to_resolve=["the document"],
            search_hints=["publication transition request current status"],
            risk_flags=["Transition"],
            confidence=0.74,
            reason="The question asks whether a standards document can be published.",
            model=model,
        )


class FakeOpenAICompatibleClient:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.last_kwargs: dict = {}

    def generate_answer(self, **kwargs):  # type: ignore[no-untyped-def]
        from app.services.openai_compatible import OpenAICompatibleGeneration

        self.calls.append(kwargs["model"])
        self.last_kwargs = kwargs
        return OpenAICompatibleGeneration(
            text="Online model grounded answer with transition guidance [S1].",
            model=kwargs["model"],
        )


def test_workflow_resolves_w3c_api_entities() -> None:
    workflow = ChatWorkflow(_test_settings(), w3c_api_client=FakeW3CAPIClient())  # type: ignore[arg-type]

    response = workflow.run(
        ChatRequest(
            message="What should CSS Grid do next from CR to REC?",
            locale="en",
        )
    )

    assert response.resolved_entities
    assert response.resolved_entities[0].shortname == "css-grid-1"
    assert response.resolved_entities[0].deliverers == ["Cascading Style Sheets (CSS) Working Group"]
    assert response.process_state
    assert response.process_state.current_stage == "CR"
    assert response.process_state.group_type == "Working Group"
    assert "w3c_api_resolver" in [step.id for step in response.workflow_trace]


class FakeWAIAdaptAPIClient:
    def resolve_entities(self, query: str) -> list[W3CEntity]:
        return [
            W3CEntity(
                entity_type="specification",
                title="WAI-Adapt: Symbols Module",
                shortname="adapt-symbols",
                api_url="https://api.w3.org/specifications/adapt-symbols",
                public_url="https://www.w3.org/TR/adapt-symbols/",
                editor_draft_url="https://w3c.github.io/personalization-semantics/content/",
                status="Candidate Recommendation Snapshot",
                latest_version_url="https://api.w3.org/specifications/adapt-symbols/versions/20230105",
                latest_version_date="2023-01-05",
                deliverers=["Accessible Platform Architectures Working Group"],
                confidence=0.95,
            )
        ]


def test_workflow_keeps_wai_adapt_cr_to_rec_as_transition_not_charter() -> None:
    workflow = ChatWorkflow(
        _test_settings(),
        w3c_api_client=FakeWAIAdaptAPIClient(),  # type: ignore[arg-type]
    )

    response = workflow.run(
        ChatRequest(
            message="now wai-adapt symbol in CR, how to publish it in rec",
            locale="en",
        )
    )

    assert response.resolved_entities
    assert response.resolved_entities[0].shortname == "adapt-symbols"
    assert response.task_plan
    assert response.task_plan.intent_type == "advance_specification"
    assert response.process_state
    assert response.process_state.current_stage == "CR"
    assert response.process_state.target_stage == "REC"
    assert "transition" in response.answer.lower()
    assert "charter work" not in response.answer.lower()


def test_workflow_uses_w3c_api_entities_to_enhance_retrieval_query() -> None:
    retriever = RecordingRetriever()
    workflow = ChatWorkflow(
        _test_settings(),
        retriever=retriever,  # type: ignore[arg-type]
        w3c_api_client=FakeW3CAPIClient(),  # type: ignore[arg-type]
    )

    response = workflow.run(
        ChatRequest(
            message="What should the CSS Grid specification do next?",
            locale="en",
        )
    )

    assert response.audit["used_entity_augmented_query"]
    assert "entity_query_enricher" in [step.id for step in response.workflow_trace]
    assert retriever.queries
    query = retriever.queries[0]
    assert "Task plan retrieval requirements" in query
    assert "css-grid-1" in query
    assert "Candidate Recommendation Draft" in query
    assert "Cascading Style Sheets (CSS) Working Group" in query
    assert "transitioning to Recommendation" in query


def test_workflow_resolves_official_github_draft_context_when_needed() -> None:
    retriever = RecordingRetriever()
    workflow = ChatWorkflow(
        _test_settings(),
        retriever=retriever,  # type: ignore[arg-type]
        w3c_api_client=FakeW3CAPIClient(),  # type: ignore[arg-type]
        github_context_client=FakeGitHubDraftContextClient(),  # type: ignore[arg-type]
    )

    response = workflow.run(
        ChatRequest(
            message="Use the CSS Grid editor draft GitHub repo context to tell me what process step is next.",
            locale="en",
        )
    )

    assert response.draft_contexts
    assert response.draft_contexts[0].repo_full_name == "w3c/csswg-drafts"
    assert "draft_context_resolver" in [step.id for step in response.workflow_trace]
    assert retriever.queries
    assert "Resolved official GitHub draft context" in retriever.queries[0]
    assert "w3c/csswg-drafts" in retriever.queries[0]


def test_workflow_adds_w3c_strategy_context_for_charter_workflow() -> None:
    retriever = RecordingRetriever()
    workflow = ChatWorkflow(
        _test_settings(),
        retriever=retriever,  # type: ignore[arg-type]
        github_context_client=FakeGitHubDraftContextClient(),  # type: ignore[arg-type]
    )

    response = workflow.run(
        ChatRequest(
            message="How should we track a recharter review?",
            locale="en",
        )
    )

    assert response.task_plan
    assert response.task_plan.intent_type == "charter_or_recharter"
    assert response.draft_contexts
    assert response.draft_contexts[0].repo_full_name == "w3c/strategy"
    assert "w3c/strategy" in retriever.queries[0]
    assert "Current `w3c/strategy` charter issue signals" in response.answer
    assert "https://github.com/w3c/strategy/issues?q=label%3Acharter" in " ".join(response.next_steps)
    assert "closed Strategy issues" in " ".join(response.next_steps)
    assert "TiLT" in " ".join(response.next_steps)


def test_workflow_uses_compiled_context_when_available() -> None:
    workflow = ChatWorkflow(
        _test_settings(),
        w3c_api_client=FakeW3CAPIClient(),  # type: ignore[arg-type]
        compiled_context_store=FakeCompiledContextStore(),  # type: ignore[arg-type]
    )

    response = workflow.run(
        ChatRequest(
            message="What should CSS Grid do next from CR to REC?",
            locale="en",
        )
    )

    assert response.compiled_context
    assert response.compiled_context_used
    assert response.compiled_context.key == "css-grid-1"
    assert response.evidence_coverage
    assert response.evidence_coverage.has_compiled_context
    assert "compiled_context_resolver" in [step.id for step in response.workflow_trace]


def test_workflow_uses_llm_router_for_ambiguous_process_question() -> None:
    retriever = RecordingRetriever()
    workflow = ChatWorkflow(
        _test_settings(),
        retriever=retriever,  # type: ignore[arg-type]
        llm_router=FakeLLMRouter(),  # type: ignore[arg-type]
    )

    response = workflow.run(ChatRequest(message="Can this document be published now?", locale="en"))

    assert response.in_scope
    assert response.task_plan
    assert response.task_plan.intent_type == "advance_specification"
    assert response.audit["llm_router"]["likely_in_scope"]
    assert "llm_router" in [step.id for step in response.workflow_trace]
    assert retriever.queries
    assert "publication transition request current status" in retriever.queries[0]


def test_workflow_uses_openai_compatible_provider_when_configured() -> None:
    client = FakeOpenAICompatibleClient()
    workflow = ChatWorkflow(
        Settings(
            llm_provider="openai-compatible",
            openai_compatible_model="gpt-test",
            w3c_api_enabled=False,
        ),
        openai_compatible_client=client,  # type: ignore[arg-type]
    )

    response = workflow.run(
        ChatRequest(
            message="What should a CSS specification do next to move from CR to REC?",
            locale="en",
        )
    )

    assert response.answer == "Online model grounded answer with transition guidance [S1]."
    assert response.audit["model_generation"] == "openai_compatible"
    assert client.calls == ["gpt-test"]
    answer_step = next(step for step in response.workflow_trace if step.id == "answer_generator")
    assert "OpenAI-compatible model gpt-test" in answer_step.detail


def test_workflow_sends_lighter_prompt_to_external_api_providers() -> None:
    """External token-API providers (OpenAI / Kimi / OpenRouter) get the
    lighter prompt — strict formatting rules are swapped for "use your
    judgement". Local Ollama still gets the strict prompt because it
    needs the structure spelled out."""
    client = FakeOpenAICompatibleClient()
    workflow = ChatWorkflow(
        Settings(
            llm_provider="openai-compatible",
            openai_compatible_model="gpt-test",
            w3c_api_enabled=False,
        ),
        openai_compatible_client=client,  # type: ignore[arg-type]
    )
    response = workflow.run(
        ChatRequest(message="What should a CSS spec do next from CR to REC?", locale="en")
    )
    assert client.last_kwargs.get("lighter_mode") is True
    assert response.audit.get("prompt_mode") == "lighter"


def test_workflow_linkifies_bare_action_urls_in_lighter_mode_output() -> None:
    """Belt-and-suspenders: when the model emits a bare ``https://…``
    URL instead of ``[label](url)`` markdown, the workflow post-pass
    must wrap it so the frontend renders a clickable link. Only the
    curated surface URLs for the intent are touched (never random
    URLs from elsewhere in the answer)."""

    class _RawUrlClient:
        """Emits a bare GitHub URL — the failure mode we're guarding."""

        def generate_answer(self, **kwargs):  # type: ignore[no-untyped-def]
            from app.services.openai_compatible import OpenAICompatibleGeneration
            return OpenAICompatibleGeneration(
                text=(
                    "File the request at https://github.com/w3c/i18n-request/issues/new/choose "
                    "before requesting transition [S1]."
                ),
                model=kwargs["model"],
            )

    workflow = ChatWorkflow(
        Settings(
            llm_provider="openai-compatible",
            openai_compatible_model="gpt-test",
            w3c_api_enabled=False,
        ),
        openai_compatible_client=_RawUrlClient(),  # type: ignore[arg-type]
    )
    response = workflow.run(
        ChatRequest(message="How do I file an i18n review request?", locale="en")
    )
    # The bare URL must be wrapped in markdown link syntax.
    assert "[https://github.com/w3c/i18n-request/issues/new/choose](https://github.com/w3c/i18n-request/issues/new/choose)" in response.answer
    # Audit records that the post-pass fired.
    assert response.audit.get("linkified_bare_action_urls") is True


def test_workflow_keeps_strict_prompt_for_ollama() -> None:
    """The strict (default) mode stays for local Ollama — template mode
    also defaults to lighter_mode=False since it never hits build_prompt
    via these code paths."""
    response = ChatWorkflow(_test_settings()).run(
        ChatRequest(message="What is a Working Draft?")
    )
    # Template provider doesn't even go through build_prompt, so
    # ``prompt_mode`` is not set on the audit — distinct from the
    # explicit "lighter" tag above.
    assert response.audit.get("prompt_mode") is None


def test_workflow_runs_targeted_retrieval_when_guide_evidence_is_missing() -> None:
    retriever = RecordingRetriever()
    workflow = ChatWorkflow(
        _test_settings(),
        retriever=retriever,  # type: ignore[arg-type]
        w3c_api_client=FakeW3CAPIClient(),  # type: ignore[arg-type]
    )

    response = workflow.run(
        ChatRequest(
            message="What should the CSS Grid specification do next?",
            locale="en",
        )
    )

    assert response.evidence_coverage
    assert response.evidence_coverage.status in {"needs_more_evidence", "insufficient"}
    assert "targeted_retrieval" in [step.id for step in response.workflow_trace]
    assert len(retriever.queries) > 1


def test_workflow_answers_horizontal_review_with_github_operational_steps() -> None:
    workflow = ChatWorkflow(_test_settings())

    response = workflow.run(
        ChatRequest(
            message="How should a WG request horizontal review and handle *-needs-resolution labels before CR?",
            locale="en",
        )
    )

    assert response.task_plan
    assert response.task_plan.intent_type == "horizontal_review"
    assert response.process_state
    assert response.process_state.likely_workflow == "horizontal_review"
    assert any("documentreview" in str(citation.url) for citation in response.citations)
    assert any("github" in step.lower() for step in response.next_steps)
    assert any("needs-resolution" in step.lower() for step in response.next_steps)
    assert any("tracker" in step.lower() for step in response.next_steps)
    assert any("a11y-request" in step.lower() for step in response.next_steps)
    assert any("i18n-request" in step.lower() for step in response.next_steps)
    assert any("privacy-request" in step.lower() for step in response.next_steps)
    assert any("security-request" in step.lower() for step in response.next_steps)


def test_workflow_keeps_charter_horizontal_review_focused_on_horizontal_review() -> None:
    workflow = ChatWorkflow(_test_settings())

    response = workflow.run(
        ChatRequest(
            message="How does a proposed charter request horizontal review?",
            locale="en",
        )
    )

    assert response.task_plan
    assert response.task_plan.intent_type == "horizontal_review"
    assert response.process_state
    assert response.process_state.likely_workflow == "horizontal_review"
    assert "horizontal review" in response.answer.lower()
    assert "github" in " ".join(response.next_steps).lower()
