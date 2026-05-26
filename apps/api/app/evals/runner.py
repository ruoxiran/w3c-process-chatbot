from __future__ import annotations

from collections.abc import Callable

from app.evals.cases import EvalCase
from app.models.schemas import ChatRequest, ChatResponse, EvalCaseResult, EvalRunResponse


WorkflowRunner = Callable[[ChatRequest], ChatResponse]


def run_eval_cases(cases: list[EvalCase], runner: WorkflowRunner) -> EvalRunResponse:
    results = [_evaluate_case(case, runner(ChatRequest(message=case.message, locale="en"))) for case in cases]
    passed_count = sum(1 for result in results if result.passed)
    score = passed_count / len(results) if results else 0.0
    return EvalRunResponse(
        passed=all(result.passed for result in results),
        score=round(score, 4),
        passed_count=passed_count,
        total_count=len(results),
        results=results,
    )


def _evaluate_case(case: EvalCase, response: ChatResponse) -> EvalCaseResult:
    failures: list[str] = []
    warnings: list[str] = []

    if response.in_scope != case.expected_in_scope:
        failures.append(f"expected in_scope={case.expected_in_scope}, got {response.in_scope}")

    if case.expected_intent:
        actual = response.task_plan.intent_type if response.task_plan else None
        if actual != case.expected_intent:
            failures.append(f"expected intent={case.expected_intent}, got {actual}")

    if case.expected_entity_shortname:
        shortnames = {entity.shortname for entity in response.resolved_entities if entity.shortname}
        if case.expected_entity_shortname not in shortnames:
            failures.append(
                f"expected entity shortname {case.expected_entity_shortname}, got {sorted(shortnames) or 'none'}"
            )

    if case.expected_compiled_context is not None and response.compiled_context_used != case.expected_compiled_context:
        failures.append(
            f"expected compiled_context_used={case.expected_compiled_context}, got {response.compiled_context_used}"
        )

    actual_source_types = {citation.source_type.value for citation in response.citations}
    if response.draft_contexts:
        actual_source_types.add("repo")
    for source_type in case.expected_source_types:
        if source_type not in actual_source_types:
            failures.append(f"missing source type {source_type}")

    citation_urls = [str(citation.url) for citation in response.citations]
    draft_urls = [str(context.repo_url) for context in response.draft_contexts]
    combined_urls = [*citation_urls, *draft_urls]
    for expected_url in case.expected_url_substrings:
        if not _contains_any(combined_urls, expected_url):
            failures.append(f"missing URL containing {expected_url}")

    answer_text = response.answer
    next_steps_text = "\n".join([*response.next_steps, *(step.text for step in response.next_step_details)])
    all_text = f"{answer_text}\n{next_steps_text}"
    for term in case.expected_answer_terms:
        if term.lower() not in answer_text.lower():
            failures.append(f"answer missing term {term!r}")
    for term in case.expected_next_step_terms:
        if term.lower() not in next_steps_text.lower():
            failures.append(f"next steps missing term {term!r}")
    for term in case.forbidden_terms:
        if term.lower() in all_text.lower():
            failures.append(f"forbidden term appeared: {term!r}")

    if case.min_confidence is not None and response.confidence < case.min_confidence:
        failures.append(f"confidence {response.confidence:.2f} below minimum {case.min_confidence:.2f}")

    if response.in_scope and not response.citations:
        failures.append("in-scope answer has no citations")
    if response.evidence_coverage and response.evidence_coverage.status != "sufficient":
        warnings.append(f"evidence coverage is {response.evidence_coverage.status}")
    if response.in_scope and response.task_plan and not response.task_plan.search_queries:
        warnings.append("task plan has no focused search queries")

    passed = not failures
    details = "passed" if passed else "; ".join(failures)
    if warnings:
        details = f"{details}; warnings: {'; '.join(warnings)}"

    return EvalCaseResult(
        name=case.name,
        passed=passed,
        details=details,
        tags=list(case.tags),
        expected_in_scope=case.expected_in_scope,
        actual_in_scope=response.in_scope,
        expected_intent=case.expected_intent,
        actual_intent=response.task_plan.intent_type if response.task_plan else None,
        expected_source_types=list(case.expected_source_types),
        actual_source_types=sorted(actual_source_types),
        expected_url_substrings=list(case.expected_url_substrings),
        actual_urls=combined_urls[:12],
        expected_entity_shortname=case.expected_entity_shortname,
        actual_entity_shortnames=[
            entity.shortname for entity in response.resolved_entities if entity.shortname
        ],
        expected_compiled_context=case.expected_compiled_context,
        actual_compiled_context=response.compiled_context_used,
        confidence=response.confidence,
        warnings=warnings,
    )


def _contains_any(values: list[str], needle: str) -> bool:
    lowered = needle.lower()
    return any(lowered in value.lower() for value in values)
