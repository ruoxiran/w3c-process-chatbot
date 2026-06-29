from __future__ import annotations

from app.core.config import Settings
from app.models.schemas import DraftContext, DraftSnippet, W3CEntity
from app.services.compiled_context import CompiledContextStore
from app.workflows.chat_workflow import ChatWorkflow


def build_eval_workflow(settings: Settings) -> ChatWorkflow:
    """Build a deterministic, offline-friendly workflow for regression evals.

    The product chat path can use live W3C API and GitHub context. The eval path
    should be fast and repeatable, so it supplies small fake live-context clients
    for the high-risk cases that require entity or charter grounding.
    """
    # llm_router_enabled also forced off — otherwise the router
    # fires whenever local Ollama happens to be running, making
    # eval results depend on the developer's machine state.
    # Deterministic eval = no LLM in the loop, anywhere.
    eval_settings = settings.model_copy(update={
        "llm_provider": "template",
        "llm_router_enabled": False,
    })
    w3c_api = EvalW3CAPIClient()
    github = EvalGitHubContextClient()
    compiled = CompiledContextStore(
        eval_settings,
        w3c_api_client=w3c_api,  # type: ignore[arg-type]
        github_context_client=github,  # type: ignore[arg-type]
    )
    return ChatWorkflow(
        eval_settings,
        w3c_api_client=w3c_api,  # type: ignore[arg-type]
        github_context_client=github,  # type: ignore[arg-type]
        compiled_context_store=compiled,
    )


class EvalW3CAPIClient:
    def resolve_entities(self, query: str) -> list[W3CEntity]:
        text = query.lower()
        if "wai-adapt" in text or "adapt-symbol" in text or "adapt symbol" in text:
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
                    retrieval_hints=[
                        "adapt-symbols",
                        "Candidate Recommendation Snapshot",
                        "Accessible Platform Architectures Working Group",
                    ],
                    confidence=0.95,
                )
            ]
        if "css grid" in text:
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
                    deliverers=["Cascading Style Sheets (CSS) Working Group"],
                    retrieval_hints=[
                        "css-grid-1",
                        "Candidate Recommendation Draft",
                        "Cascading Style Sheets (CSS) Working Group",
                    ],
                    confidence=0.9,
                )
            ]
        if "webauthn" in text or "web authentication" in text:
            return [
                W3CEntity(
                    entity_type="specification",
                    title="Web Authentication: An API for accessing Public Key Credentials Level 3",
                    shortname="webauthn-3",
                    api_url="https://api.w3.org/specifications/webauthn-3",
                    public_url="https://www.w3.org/TR/webauthn-3/",
                    editor_draft_url="https://w3c.github.io/webauthn/",
                    status="Candidate Recommendation Draft",
                    latest_version_url="https://api.w3.org/specifications/webauthn-3/versions/20250501",
                    latest_version_date="2025-05-01",
                    deliverers=["Web Authentication Working Group"],
                    retrieval_hints=[
                        "webauthn-3",
                        "Candidate Recommendation Draft",
                        "Web Authentication Working Group",
                    ],
                    confidence=0.92,
                )
            ]
        return []


class EvalGitHubContextClient:
    def resolve_contexts(self, query, entities, task_plan, limit=2):  # type: ignore[no-untyped-def]
        if task_plan.intent_type == "charter_or_recharter":
            return [
                DraftContext(
                    repo_full_name="w3c/strategy",
                    repo_url="https://github.com/w3c/strategy",
                    resolved_from="https://github.com/w3c/strategy/issues?q=label%3Acharter",
                    description="W3C strategy issue tracker for charter and recharter reviews.",
                    snippets=[
                        DraftSnippet(
                            path="issues/123",
                            title="Review Example Working Group charter",
                            text=(
                                "state=open; labels=charter, Horizontal review requested, "
                                "i18n-review-completed, privacy-review-completed; no *-needs-resolution blockers."
                            ),
                            url="https://github.com/w3c/strategy/issues/123",
                        )
                    ],
                    retrieval_hints=[
                        "w3c/strategy",
                        "charter label",
                        "charter review issue tracker",
                        "closed charter issues",
                        "Horizontal review requested",
                        "horizontal review completed labels",
                        "TiLT review readiness",
                        "https://github.com/w3c/strategy/issues?q=label%3Acharter",
                    ],
                    confidence=0.88,
                )
            ][:limit]
        return []
