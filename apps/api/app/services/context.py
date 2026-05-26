import re

from app.models.schemas import ChatTurn, W3CEntity


FOLLOW_UP_MARKERS = {
    "it",
    "its",
    "that",
    "this",
    "those",
    "they",
    "them",
    "there",
    "then",
    "next",
    "above",
    "previous",
    "same",
    "what about",
    "how about",
    "这个",
    "那个",
    "这",
    "那",
    "它",
    "他们",
    "上述",
    "上面",
    "刚才",
    "继续",
    "下一步",
    "然后",
    "还要",
    "谁",
    "多久",
    "需要吗",
}


def build_contextual_query(message: str, history: list[ChatTurn]) -> str:
    """Build a retrieval query that resolves short follow-up references.

    Conversation history is not a trusted source. This function only adds recent
    turn text so scope classification and retrieval can understand references
    like "that transition" or "下一步".
    """
    if not history or not _looks_like_follow_up(message):
        return message

    lines = []
    for turn in history[-6:]:
        content = _compact(turn.content, 360)
        if content:
            lines.append(f"{turn.role}: {content}")

    if not lines:
        return message

    context = "\n".join(lines)
    return (
        f"Current follow-up question: {message}\n"
        "Recent conversation for reference resolution only; not authoritative source text:\n"
        f"{context}"
    )


def build_entity_augmented_query(query: str, entities: list[W3CEntity]) -> str:
    """Add resolved W3C API entity/status hints to a retrieval query.

    The appended API facts are used only to steer Process/Guidebook retrieval
    toward the right spec, group, maturity stage, and workflow terms.
    """
    if not entities:
        return query

    lines = [query, "", "Resolved W3C API entity/status hints for retrieval only; not normative Process rules:"]
    workflow_terms: set[str] = set()

    for entity in entities[:5]:
        parts = [
            entity.entity_type,
            entity.title,
            entity.shortname or "",
            entity.status or "",
            entity.latest_version_date or "",
            ", ".join(entity.deliverers),
            entity.group_type or "",
            entity.charter_end or "",
        ]
        if entity.public_url:
            parts.append(str(entity.public_url))
        if entity.editor_draft_url:
            parts.append(str(entity.editor_draft_url))
        if entity.process_rules_url:
            parts.append(str(entity.process_rules_url))
        if entity.retrieval_hints:
            parts.append(f"retrieval_hints={', '.join(entity.retrieval_hints)}")
        lines.append("; ".join(part for part in parts if part))
        workflow_terms.update(_workflow_terms(entity))

    if workflow_terms:
        lines.append(f"Derived workflow retrieval terms: {', '.join(sorted(workflow_terms))}")

    return "\n".join(lines)


def _workflow_terms(entity: W3CEntity) -> set[str]:
    terms: set[str] = set()
    status = (entity.status or "").lower()
    if "candidate recommendation" in status:
        terms.update(
            {
                "Candidate Recommendation",
                "CR",
                "Recommendation track",
                "transition requirements",
                "transitioning to Recommendation",
                "AC Review",
                "implementation experience",
                "Exclusion Opportunity",
            }
        )
    if "working draft" in status:
        terms.update({"Working Draft", "FPWD", "wide review", "horizontal review"})
    if "proposed recommendation" in status:
        terms.update({"Proposed Recommendation", "Advisory Committee Review"})
    if entity.deliverers:
        terms.update({"Working Group", "charter", "Staff Contact", "Team Contact"})
    if entity.charter_url or entity.charter_end:
        terms.update({"active charter", "charter review", "charter end"})
    if entity.patent_policy_url:
        terms.add("Patent Policy")
    return terms


def _looks_like_follow_up(message: str) -> bool:
    text = message.strip().lower()
    if not text:
        return False
    if any(marker in text for marker in FOLLOW_UP_MARKERS):
        return True
    words = re.findall(r"[a-z0-9]+", text)
    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", text))
    return len(words) <= 6 or (has_cjk and len(text) <= 18)


def _compact(text: str, limit: int) -> str:
    compacted = " ".join(text.split())
    if len(compacted) <= limit:
        return compacted
    return f"{compacted[:limit].rsplit(' ', 1)[0]}..."
