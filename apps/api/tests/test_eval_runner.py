from app.evals.cases import EVAL_CASES, EvalCase
from app.evals.runner import run_eval_cases
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    CompiledContext,
    SourceType,
    SourceVersion,
    TaskPlan,
    W3CEntity,
)


def test_eval_cases_cover_core_accuracy_risks() -> None:
    names = {case.name for case in EVAL_CASES}

    assert {
        "cr-to-rec",
        "horizontal-review-github",
        "charter-strategy-tracking",
        "wai-adapt-cr-to-rec",
        "non-process",
        "injection",
    }.issubset(names)
    assert any(case.expected_url_substrings for case in EVAL_CASES)
    assert any(case.forbidden_terms for case in EVAL_CASES)


def test_eval_runner_scores_intent_sources_urls_entities_and_compiled_context() -> None:
    case = EvalCase(
        name="wai-adapt",
        message="now wai-adapt symbol in CR, how to publish it in rec",
        expected_in_scope=True,
        expected_intent="advance_specification",
        expected_source_types=("process", "guide"),
        expected_url_substrings=("w3.org/policies/process", "w3.org/guide/transitions"),
        expected_answer_terms=("Recommendation",),
        expected_next_step_terms=("transition",),
        forbidden_terms=("mobile-accessibility-mapping",),
        expected_entity_shortname="adapt-symbols",
        expected_compiled_context=True,
        min_confidence=0.5,
        tags=("entity-grounding",),
    )

    def runner(_: ChatRequest) -> ChatResponse:
        return ChatResponse(
            answer="Use the Recommendation-track transition path toward Recommendation.",
            in_scope=True,
            citations=[
                Citation(
                    title="W3C Process",
                    url="https://www.w3.org/policies/process/",
                    source_type=SourceType.process,
                ),
                Citation(
                    title="Transitions",
                    url="https://www.w3.org/guide/transitions/",
                    source_type=SourceType.guide,
                ),
            ],
            next_steps=["Prepare the transition request."],
            task_plan=TaskPlan(intent_type="advance_specification", user_goal="Advance the spec"),
            compiled_context=CompiledContext(
                kind="spec",
                key="adapt-symbols",
                title="WAI-Adapt: Symbols Module",
                summary="Compiled context",
            ),
            compiled_context_used=True,
            resolved_entities=[
                W3CEntity(
                    entity_type="specification",
                    title="WAI-Adapt: Symbols Module",
                    shortname="adapt-symbols",
                    api_url="https://api.w3.org/specifications/adapt-symbols",
                )
            ],
            confidence=0.7,
            source_version=SourceVersion(),
        )

    response = run_eval_cases([case], runner)

    assert response.passed
    assert response.score == 1.0
    assert response.passed_count == 1
    assert response.total_count == 1
    assert response.results[0].actual_intent == "advance_specification"
    assert response.results[0].actual_entity_shortnames == ["adapt-symbols"]
    assert response.results[0].actual_compiled_context is True


def test_eval_runner_fails_on_misgrounded_entity() -> None:
    case = EvalCase(
        name="wai-adapt",
        message="now wai-adapt symbol in CR, how to publish it in rec",
        expected_in_scope=True,
        expected_entity_shortname="adapt-symbols",
        forbidden_terms=("mobile-accessibility-mapping",),
    )

    def runner(_: ChatRequest) -> ChatResponse:
        return ChatResponse(
            answer="Use mobile-accessibility-mapping as the entity.",
            in_scope=True,
            citations=[],
            resolved_entities=[
                W3CEntity(
                    entity_type="specification",
                    title="Mobile Accessibility Mapping",
                    shortname="mobile-accessibility-mapping",
                    api_url="https://api.w3.org/specifications/mobile-accessibility-mapping",
                )
            ],
            confidence=0.7,
            source_version=SourceVersion(),
        )

    response = run_eval_cases([case], runner)

    assert not response.passed
    assert response.score == 0.0
    assert "expected entity shortname adapt-symbols" in response.results[0].details
    assert "forbidden term appeared" in response.results[0].details
