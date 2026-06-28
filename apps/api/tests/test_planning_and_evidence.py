from app.models.schemas import Citation, CompiledContext, CompiledFreshness, CompiledProvenance, SourceType, W3CEntity
from app.services.evidence import check_evidence_coverage
from app.services.task_planner import plan_task


def test_task_planner_focuses_recommendation_transition() -> None:
    plan = plan_task("What should CSS Grid do next from CR to REC?")

    assert plan.intent_type == "advance_specification"
    assert plan.current_stage == "CR"
    assert plan.target_stage == "REC"
    assert SourceType.process in plan.needed_sources
    assert SourceType.guide in plan.needed_sources
    assert any("transition" in query.lower() for query in plan.search_queries)


def test_task_planner_promotes_horizontal_review_to_first_class_workflow() -> None:
    plan = plan_task("Before CR, how should we handle horizontal review and *-needs-resolution labels?")

    assert plan.intent_type == "horizontal_review"
    assert plan.answer_shape == "horizontal_review_checklist_with_github_request_links_and_tracker_checks"
    assert SourceType.process in plan.needed_sources
    assert SourceType.guide in plan.needed_sources
    assert "Horizontal Review" in plan.risk_flags
    assert any("github" in query.lower() for query in plan.search_queries)
    assert any("needs-resolution" in query.lower() for query in plan.search_queries)


def test_task_planner_routes_spec_authoring_tool_questions_to_author_spec() -> None:
    """Questions about ReSpec / Bikeshed / Pubrules / Echidna / HTMLdiff
    are spec-AUTHORING / publication tooling — they used to fall
    through every keyword rule and end up labelled
    ``advance_specification``, which steered retrieval toward REC-
    track transition content instead of editor / repo-management /
    publication tooling. Pin the fix so a future refactor doesn't
    drop authoring-tool routing on the floor."""
    for query in [
        "ReSpec or Bikeshed for my new spec?",
        "how do I run pubrules before publishing",
        "configure Echidna for auto-publication",
        "use HTMLdiff to produce the diff for CR",
        "how do I author a W3C spec",
        "用 respec 还是 bikeshed",
    ]:
        plan = plan_task(query)
        assert plan.intent_type == "author_spec", f"misclassified: {query!r} → {plan.intent_type}"
    # Retrieval seeds must point at the editor / publication chapters,
    # not at REC-track transition pages.
    plan = plan_task("how do I author a W3C spec")
    joined = " ".join(plan.search_queries).lower()
    assert "editor" in joined
    assert "publication" in joined or "pubrules" in joined or "echidna" in joined


def test_task_planner_routes_scribing_questions_to_run_group_process() -> None:
    """"how to scribe?" used to fall through to ``advance_specification``
    because ``scribe`` wasn't in the run-group-process keyword list,
    so retrieval went hunting for REC-transition chunks and missed
    the dedicated zakim.html / rrsagent.html / scribe.html guide
    pages entirely. Pin the fix so a future keyword-list refactor
    doesn't drop scribing on the floor again."""
    for query in [
        "how to scribe?",
        "how do I use Zakim?",
        "what does RRSAgent do during a meeting?",
        "scribe.perl conventions for IRC minutes",
        "how to invite Zakim and RRSAgent to my meeting",
    ]:
        plan = plan_task(query)
        assert plan.intent_type == "run_group_process", f"misclassified: {query!r}"
    # The retrieval seeds for run_group_process must include tool-
    # specific phrases so dense + lexical retrieval lands on the
    # actual guide chapters, not generic meeting/minutes pages.
    plan = plan_task("how to scribe?")
    joined = " ".join(plan.search_queries).lower()
    assert "zakim" in joined
    assert "rrsagent" in joined
    assert "scribe" in joined


def test_task_planner_marks_draft_context_when_question_mentions_github_repo() -> None:
    plan = plan_task("Use the CSS Grid editor draft GitHub repo context to decide the next Process step.")

    assert SourceType.repo in plan.needed_sources
    assert "Draft Context" in plan.risk_flags
    assert any("github repository context" in query.lower() for query in plan.search_queries)


def test_task_planner_includes_strategy_repo_for_charter_workflow() -> None:
    plan = plan_task("How should we track a recharter review?")

    assert plan.intent_type == "charter_or_recharter"
    assert plan.answer_shape == "charter_recharter_steps_with_w3c_strategy_issue_tracking"
    assert SourceType.repo in plan.needed_sources
    assert any("w3c strategy" in query.lower() for query in plan.search_queries)


def test_evidence_checker_requests_guide_when_missing() -> None:
    plan = plan_task("What should CSS Grid do next from CR to REC?")
    plan.spec_or_group = "CSS Grid"
    entity = W3CEntity(
        entity_type="specification",
        title="CSS Grid Layout Module Level 1",
        shortname="css-grid-1",
        api_url="https://api.w3.org/specifications/css-grid-1",
        status="Candidate Recommendation Draft",
    )
    coverage = check_evidence_coverage(
        plan=plan,
        entities=[entity],
        citations=[
            Citation(
                title="W3C Process Document",
                url="https://www.w3.org/policies/process/",
                source_type=SourceType.process,
                heading_path="Transitioning to Recommendation",
            )
        ],
    )

    assert coverage.status == "needs_more_evidence"
    assert not coverage.has_compiled_context
    assert "Guidebook practice guidance" in coverage.missing_evidence
    assert coverage.targeted_queries


def test_evidence_checker_accepts_compiled_context_for_spec_questions() -> None:
    plan = plan_task("What should CSS Grid do next from CR to REC?")
    plan.spec_or_group = "CSS Grid"
    entity = W3CEntity(
        entity_type="specification",
        title="CSS Grid Layout Module Level 1",
        shortname="css-grid-1",
        api_url="https://api.w3.org/specifications/css-grid-1",
        status="Candidate Recommendation Draft",
        confidence=0.92,
    )
    compiled = CompiledContext(
        kind="spec",
        key="css-grid-1",
        title="CSS Grid Layout Module Level 1",
        summary="Compiled summary",
        freshness=CompiledFreshness(compiled_at="2026-04-26T00:00:00Z"),
        provenance=CompiledProvenance(),
    )
    coverage = check_evidence_coverage(
        plan=plan,
        entities=[entity],
        compiled_context=compiled,
        citations=[
            Citation(
                title="W3C Process Document",
                url="https://www.w3.org/policies/process/",
                source_type=SourceType.process,
                heading_path="Transitioning to Recommendation",
            ),
            Citation(
                title="The Art of Consensus: W3C Guidebook",
                url="https://www.w3.org/guide/transitions/",
                source_type=SourceType.guide,
                heading_path="Transitions",
            ),
        ],
    )

    assert coverage.has_compiled_context
    assert "compiled spec context" not in coverage.missing_evidence
