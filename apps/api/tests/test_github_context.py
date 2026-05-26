from app.core.config import Settings
from app.models.schemas import TaskPlan, W3CEntity
from app.services.github_context import (
    GitHubDraftContextClient,
    _should_resolve_strategy_charter_context,
    _repo_candidate_from_url,
    build_draft_context_augmented_query,
)


def test_repo_candidate_from_w3c_github_io_url() -> None:
    candidate = _repo_candidate_from_url("https://w3c.github.io/csswg-drafts/css-grid-1/", {"w3c"})

    assert candidate
    assert candidate.full_name == "w3c/csswg-drafts"


def test_repo_candidate_rejects_untrusted_github_org() -> None:
    candidate = _repo_candidate_from_url("https://github.com/example/fake-process", {"w3c"})

    assert candidate is None


def test_github_context_is_gated_by_question_and_w3c_api_entity() -> None:
    client = GitHubDraftContextClient(Settings(github_context_enabled=True))
    plan = TaskPlan(user_goal="Answer a process question")
    entity = W3CEntity(
        entity_type="specification",
        title="CSS Grid Layout Module Level 1",
        shortname="css-grid-1",
        api_url="https://api.w3.org/specifications/css-grid-1",
        editor_draft_url="https://w3c.github.io/csswg-drafts/css-grid-1/",
    )

    assert client.resolve_contexts("What is the next Process step?", [entity], plan) == []


def test_draft_context_augmented_query_marks_context_as_non_normative() -> None:
    from app.models.schemas import DraftContext, DraftSnippet

    query = build_draft_context_augmented_query(
        "What should CSS Grid do next?",
        [
            DraftContext(
                repo_full_name="w3c/csswg-drafts",
                repo_url="https://github.com/w3c/csswg-drafts",
                snippets=[
                    DraftSnippet(
                        path="css-grid-1/Overview.bs",
                        title="CSS Grid Layout Module Level 1",
                        text="Spec source",
                    )
                ],
            )
        ],
    )

    assert "Resolved official GitHub draft context" in query
    assert "not normative Process rules" in query
    assert "w3c/csswg-drafts" in query


class FakeStrategyGitHubClient(GitHubDraftContextClient):
    def _get_json(self, path: str) -> object:
        if path == "/repos/w3c/strategy":
            return {
                "html_url": "https://github.com/w3c/strategy",
                "default_branch": "main",
                "description": "W3C strategy funnel",
                "open_issues_count": 42,
            }
        if path.startswith("/repos/w3c/strategy/issues"):
            return [
                {
                    "number": 123,
                    "title": "Review Foo Working Group charter",
                    "state": "open",
                    "html_url": "https://github.com/w3c/strategy/issues/123",
                    "updated_at": "2026-04-20T00:00:00Z",
                    "labels": [{"name": "charter"}],
                }
            ]
        raise RuntimeError(path)


def test_strategy_charter_context_is_triggered_by_charter_workflow() -> None:
    plan = TaskPlan(
        intent_type="charter_or_recharter",
        user_goal="Determine the charter workflow.",
    )

    assert _should_resolve_strategy_charter_context("How do we recharter this WG?", plan)


def test_strategy_charter_context_fetches_charter_label_issues() -> None:
    client = FakeStrategyGitHubClient(Settings(github_context_enabled=True))
    plan = TaskPlan(
        intent_type="charter_or_recharter",
        user_goal="Determine the charter workflow.",
    )

    contexts = client.resolve_contexts("How do we recharter this WG?", [], plan)

    assert contexts
    assert contexts[0].repo_full_name == "w3c/strategy"
    assert "charter label" in contexts[0].retrieval_hints
    assert contexts[0].snippets[0].path == "issues/123"


class FakeStrategyGitHubStatusClient(GitHubDraftContextClient):
    def _get_json(self, path: str) -> object:
        if path == "/repos/w3c/strategy":
            return {
                "html_url": "https://github.com/w3c/strategy",
                "default_branch": "main",
                "description": "W3C strategy funnel",
                "open_issues_count": 42,
            }
        if path.startswith("/repos/w3c/strategy/issues"):
            return [
                {
                    "number": 508,
                    "title": "[wg/ag] Accessibility Guidelines Group Charter",
                    "state": "open",
                    "html_url": "https://github.com/w3c/strategy/issues/508",
                    "created_at": "2025-05-23T10:49:06Z",
                    "updated_at": "2026-04-21T21:12:23Z",
                    "labels": [
                        {"name": "Horizontal review requested"},
                        {"name": "Accessibility review completed"},
                        {"name": "Internationalization review completed"},
                        {"name": "In Charter Refinement"},
                        {"name": "charter"},
                    ],
                },
                {
                    "number": 542,
                    "title": "[wg/math] Math Working Group Charter",
                    "state": "closed",
                    "html_url": "https://github.com/w3c/strategy/issues/542",
                    "created_at": "2026-03-17T12:45:07Z",
                    "updated_at": "2026-04-02T12:41:41Z",
                    "closed_at": "2026-04-02T12:41:41Z",
                    "labels": [{"name": "charter"}],
                },
            ]
        raise RuntimeError(path)


def test_strategy_charter_context_includes_review_status_timing_and_closed_issues() -> None:
    client = FakeStrategyGitHubStatusClient(Settings(github_context_enabled=True))
    plan = TaskPlan(
        intent_type="charter_or_recharter",
        user_goal="Determine the charter workflow.",
    )

    contexts = client.resolve_contexts("How is the AG charter horizontal review going?", [], plan)
    text = " ".join(snippet.text for snippet in contexts[0].snippets)

    assert "horizontal_review_requested=True" in text
    assert "Accessibility review completed" in text
    assert "Internationalization review completed" in text
    assert "tilt_readiness_signal=possible_staff_contact_tilt_check" in text
    assert "state=closed" in text
    assert "closed_at=2026-04-02T12:41:41Z" in text
