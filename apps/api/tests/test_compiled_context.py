from pathlib import Path

from app.core.config import Settings
from app.models.schemas import (
    Citation,
    CompiledContext,
    CompiledFreshness,
    CompiledProvenance,
    DraftContext,
    DraftSnippet,
    SourceType,
    W3CEntity,
)
from app.services.compiled_context import CompiledContextStore


class FakeRetriever:
    def retrieve(self, query: str) -> list[Citation]:
        return [
            Citation(
                title="W3C Process Document",
                url="https://www.w3.org/policies/process/#transition",
                source_type=SourceType.process,
                heading_path="Transitioning to Recommendation",
                quote="Transition requirements for Recommendation track publications.",
            ),
            Citation(
                title="The Art of Consensus: W3C Guidebook",
                url="https://www.w3.org/guide/transitions/",
                source_type=SourceType.guide,
                heading_path="Transitions",
                quote="Guidebook transition planning for specifications.",
            ),
        ]


class FakeGitHubContextClient:
    def resolve_contexts(self, query, entities, task_plan, limit=2):  # type: ignore[no-untyped-def]
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
                        text="Specification source for CSS Grid.",
                        url="https://github.com/w3c/csswg-drafts/blob/main/css-grid-1/Overview.bs",
                    )
                ],
                confidence=0.9,
            )
        ]


def test_compiled_context_store_reads_existing_markdown(tmp_path: Path) -> None:
    compiled_dir = tmp_path / "compiled"
    compiled_dir.mkdir(parents=True)
    payload = CompiledContext(
        kind="spec",
        key="adapt-symbols",
        title="WAI-Adapt: Symbols Module",
        summary="Compiled summary",
        freshness=CompiledFreshness(compiled_at="2026-04-26T00:00:00Z"),
        provenance=CompiledProvenance(
            normative_urls=["https://www.w3.org/policies/process/"],
            guide_urls=["https://www.w3.org/guide/"],
            operational_urls=["https://api.w3.org/specifications/adapt-symbols"],
        ),
    )
    (compiled_dir / "adapt-symbols.md").write_text(
        "---\n"
        + payload.model_dump_json(indent=2)
        + "\n---\n# WAI-Adapt: Symbols Module\n",
        encoding="utf-8",
    )
    store = CompiledContextStore(Settings(compiled_context_dir=str(compiled_dir), llm_provider="template", w3c_api_enabled=False))

    resolved = store.resolve(
        [
            W3CEntity(
                entity_type="specification",
                title="WAI-Adapt: Symbols Module",
                shortname="adapt-symbols",
                api_url="https://api.w3.org/specifications/adapt-symbols",
                confidence=0.95,
            )
        ]
    )

    assert resolved
    assert resolved.key == "adapt-symbols"
    assert resolved.summary == "Compiled summary"


def test_compiled_context_store_compiles_entity_to_markdown(tmp_path: Path) -> None:
    store = CompiledContextStore(
        Settings(compiled_context_dir=str(tmp_path / "compiled"), llm_provider="template", w3c_api_enabled=False),
        retriever=FakeRetriever(),  # type: ignore[arg-type]
        github_context_client=FakeGitHubContextClient(),  # type: ignore[arg-type]
    )
    entity = W3CEntity(
        entity_type="specification",
        title="CSS Grid Layout Module Level 1",
        shortname="css-grid-1",
        api_url="https://api.w3.org/specifications/css-grid-1",
        public_url="https://www.w3.org/TR/css-grid-1/",
        editor_draft_url="https://w3c.github.io/csswg-drafts/css-grid-1/",
        status="Candidate Recommendation Draft",
        latest_version_date="2025-03-26",
        deliverers=["Cascading Style Sheets (CSS) Working Group"],
        confidence=0.94,
    )

    result = store.compile_entity(entity)

    assert result.context
    assert result.context.key == "css-grid-1"
    assert result.context.next_step_candidates
    assert "https://www.w3.org/policies/process/#transition" in result.context.provenance.normative_urls
    assert (tmp_path / "compiled" / "css-grid-1.md").exists()
