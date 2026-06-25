from __future__ import annotations

from app.models.schemas import Citation, CompiledContext, EvidenceCoverage, ProcessState, SourceType, TaskPlan, W3CEntity


_COMPOUND_LANDMARKS = (
    "fpwd", "first public", "working draft", "candidate recommendation", "cr",
    "snapshot", "draft note", "proposed recommendation", "rec", "recommendation",
    "charter", "recharter", "horizontal review", "wide review", "ac review",
    "formal objection", "appeal", "patent", "exclusion", "transition", "transitioning",
)


def _is_compound_question(query: str | None) -> bool:
    """Heuristic: the question touches three or more distinct Process landmarks.

    Two landmarks (e.g. "CR to REC") describe a single transition and a single
    Process citation can still cover it. The bar is three so we only fire on
    genuinely multi-topic questions (e.g. "transition + horizontal review + patent").
    """
    if not query:
        return False
    text = query.lower()
    hits = {landmark for landmark in _COMPOUND_LANDMARKS if landmark in text}
    return len(hits) >= 3


def check_evidence_coverage(
    *,
    plan: TaskPlan,
    citations: list[Citation],
    entities: list[W3CEntity],
    process_state: ProcessState | None = None,
    compiled_context: CompiledContext | None = None,
    query: str | None = None,
) -> EvidenceCoverage:
    has_compiled_context = compiled_context is not None
    process_citations = sum(1 for c in citations if c.source_type == SourceType.process)
    has_process = process_citations > 0
    has_guide = any(citation.source_type == SourceType.guide for citation in citations)
    has_entity_status = bool(entities)
    missing: list[str] = []

    if SourceType.process in plan.needed_sources and not has_process:
        missing.append("normative W3C Process evidence")
    if SourceType.guide in plan.needed_sources and not has_guide:
        missing.append("Guidebook practice guidance")
    if _needs_entity(plan) and not has_entity_status:
        missing.append("matching W3C specification or group status from the public W3C API")
    if _needs_compiled_context(plan, entities) and not has_compiled_context:
        missing.append("compiled spec context")
    if plan.intent_type == "advance_specification":
        if not (plan.current_stage or (process_state and process_state.current_stage)):
            missing.append("current maturity stage")
        if not (plan.target_stage or (process_state and process_state.target_stage)):
            missing.append("target transition or maturity stage")
    if plan.intent_type == "charter_or_recharter" and not _citation_mentions(citations, ["charter"]):
        missing.append("charter review or approval evidence")
    if plan.intent_type == "plan_or_complete_review" and not _citation_mentions(
        citations, ["wide review", "horizontal review", "review"]
    ):
        missing.append("review-specific evidence")
    if plan.intent_type == "coordinate_with_staff_contact" and not _citation_mentions(
        citations, ["staff contact", "team contact", "teamcontact"]
    ):
        missing.append("Staff Contact or Team Contact evidence")
    if _is_compound_question(query) and process_citations < 2:
        missing.append("a second Process citation covering the other stage or rule the question touches")

    targeted_queries = _targeted_queries(plan, missing)
    if not citations:
        status = "insufficient"
    elif missing and targeted_queries:
        status = "needs_more_evidence"
    elif missing:
        status = "insufficient"
    else:
        status = "sufficient"

    confidence = 0.42
    confidence += 0.08 if has_compiled_context else 0
    confidence += 0.16 if has_process else 0
    confidence += 0.13 if has_guide else 0
    confidence += 0.08 if has_entity_status else 0
    confidence += 0.08 if process_state and process_state.confidence >= 0.65 else 0
    confidence -= min(len(missing) * 0.06, 0.24)

    return EvidenceCoverage(
        status=status,
        has_compiled_context=has_compiled_context,
        has_process=has_process,
        has_guide=has_guide,
        has_entity_status=has_entity_status,
        missing_evidence=_dedupe(missing)[:6],
        targeted_queries=targeted_queries,
        summary=_summary(status, missing),
        confidence=max(0.2, min(confidence, 0.92)),
    )


def _needs_entity(plan: TaskPlan) -> bool:
    return bool(plan.spec_or_group)


def _needs_compiled_context(plan: TaskPlan, entities: list[W3CEntity]) -> bool:
    return bool(
        plan.intent_type == "advance_specification"
        and any(entity.entity_type == "specification" and (entity.confidence or 0) >= 0.7 for entity in entities)
    )


def _targeted_queries(plan: TaskPlan, missing: list[str]) -> list[str]:
    if not missing:
        return []
    subject = plan.spec_or_group or "the W3C deliverable"
    queries: list[str] = []
    for item in missing:
        lower = item.lower()
        if "process" in lower or "stage" in lower or "transition" in lower:
            queries.append(f"{subject} W3C Process {plan.intent_type} transition requirements")
        if "compiled spec context" in lower:
            queries.append(f"{subject} compiled spec context refresh")
        if "guidebook" in lower or "practice" in lower:
            queries.append(f"{subject} W3C Guidebook {plan.intent_type} practical next steps")
        if "status" in lower:
            queries.append(f"{subject} W3C specification group current status")
        if "charter" in lower:
            queries.append(f"{subject} charter review approval W3C Process Guidebook")
        if "review" in lower:
            queries.append(f"{subject} wide review horizontal review W3C Process Guidebook")
        if "staff contact" in lower or "team contact" in lower:
            queries.append("Staff Contact Team Contact responsibilities W3C Process Guidebook")
        if "second process citation" in lower:
            queries.append(f"{subject} W3C Process detailed transition requirements full criteria")
    if not queries:
        queries.extend(plan.search_queries[1:3])
    return _dedupe(queries)[:4]


def _citation_mentions(citations: list[Citation], needles: list[str]) -> bool:
    for citation in citations:
        text = " ".join(
            [
                citation.title,
                str(citation.url),
                citation.heading_path or "",
                citation.section_id or "",
                citation.quote or "",
            ]
        ).lower()
        if any(needle in text for needle in needles):
            return True
    return False


def _summary(status: str, missing: list[str]) -> str:
    if status == "sufficient":
        return "Retrieved evidence covers the planned answer requirements."
    if status == "needs_more_evidence":
        return f"Retrieved evidence is useful but missing: {', '.join(_dedupe(missing)[:4])}."
    return f"Retrieved evidence is not enough for a precise answer; missing: {', '.join(_dedupe(missing)[:4])}."


def _dedupe(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = " ".join(value.split())
        key = normalized.lower()
        if normalized and key not in seen:
            output.append(normalized)
            seen.add(key)
    return output
