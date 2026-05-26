import re

from app.models.schemas import Citation, ProcessState, SourceType, W3CEntity


STAGE_KEYWORDS = [
    ("FPWD", ["fpwd", "first public working draft", "first public", "首次公开"]),
    ("WD", ["working draft", " wd", "工作草案"]),
    ("CR", ["candidate recommendation", " cr", "候选推荐"]),
    ("CRD", ["candidate recommendation draft", "crd"]),
    ("CRS", ["candidate recommendation snapshot", "crs"]),
    ("PR", ["proposed recommendation", " pr", "提案推荐"]),
    ("REC", ["recommendation", " rec", "推荐标准"]),
]


def extract_process_state(
    message: str,
    citations: list[Citation],
    entities: list[W3CEntity] | None = None,
) -> ProcessState:
    entities = entities or []
    user_text = message.lower()
    text = _context(message, citations)
    intent = _intent(user_text, text)
    current_stage = _stage(user_text, current=True) or _stage_from_entities(entities)
    target_stage = _stage(user_text, current=False)
    group_type = _group_type(user_text) or _group_type_from_entities(entities)
    deliverable_type = _deliverable_type(user_text)
    likely_workflow = _workflow(intent, text, current_stage, target_stage)
    risk_flags = _risk_flags(text, citations)
    missing_information = _missing_information(
        intent=intent,
        text=text,
        current_stage=current_stage,
        target_stage=target_stage,
        group_type=group_type,
    )
    confidence = 0.45
    confidence += 0.12 if current_stage or target_stage else 0
    confidence += 0.12 if group_type else 0
    confidence += 0.12 if citations else 0
    confidence += 0.08 if entities else 0
    confidence += 0.1 if any(c.source_type == SourceType.process for c in citations) else 0
    confidence += 0.08 if any(c.source_type == SourceType.guide for c in citations) else 0

    return ProcessState(
        intent=intent,
        current_stage=current_stage,
        target_stage=target_stage,
        group_type=group_type,
        deliverable_type=deliverable_type,
        likely_workflow=likely_workflow,
        missing_information=missing_information,
        risk_flags=risk_flags,
        confidence=min(confidence, 0.92),
    )


def _context(message: str, citations: list[Citation]) -> str:
    parts = [message]
    for citation in citations:
        parts.extend([citation.title, citation.heading_path or "", str(citation.url), citation.quote or ""])
    return " ".join(parts).lower()


def _intent(user_text: str, combined_text: str) -> str:
    if _has(user_text, ["formal objection", "appeal", "异议", "申诉"]):
        return "handle_objection_or_appeal"
    if _has(user_text, _horizontal_review_needles()):
        return "horizontal_review"
    if _has(user_text, ["staff contact", "team contact", "liaison", "职责"]):
        return "coordinate_with_staff_contact"
    if _has(user_text, ["transition", "advance", "next step", "milestone", "转换", "推进", "下一步"]):
        return "advance_specification"
    if _has(user_text, ["meeting", "chair", "minutes", "agenda", "会议", "主席"]):
        return "run_group_process"
    if _has(user_text, ["charter", "recharter", "章程"]):
        return "charter_or_recharter"
    if _has(user_text, ["review", "horizontal review", "wide review", "审查"]):
        return "plan_or_complete_review"
    if _has(user_text, ["patent", "ipr", "专利"]):
        return "check_patent_policy"

    if _has(combined_text, ["staff contact", "team contact", "liaison", "职责"]):
        return "coordinate_with_staff_contact"
    if _has(combined_text, _horizontal_review_needles()):
        return "horizontal_review"
    if _has(combined_text, ["transition", "advance", "next step", "milestone", "转换", "推进", "下一步"]):
        return "advance_specification"
    if _has(combined_text, ["charter", "recharter", "章程"]):
        return "charter_or_recharter"
    return "explain_process"


def _stage(text: str, *, current: bool) -> str | None:
    directional = text
    transition = re.search(r"\bfrom\s+([a-z0-9 -]+?)\s+to\s+([a-z0-9 -]+?)(?:[?.!,;:]|$)", text)
    if transition:
        directional = transition.group(1 if current else 2)
    elif current and " from " in text:
        directional = text.rsplit(" from ", 1)[1].split(" to ", 1)[0]
    elif not current and " to " in text:
        directional = text.rsplit(" to ", 1)[1]

    for stage, needles in STAGE_KEYWORDS:
        if any(_stage_match(directional, needle) for needle in needles):
            return stage
    return None


def _stage_from_entities(entities: list[W3CEntity]) -> str | None:
    for entity in entities:
        if entity.entity_type != "specification" or not entity.status:
            continue
        status = entity.status.lower()
        if "candidate recommendation" in status:
            return "CR"
        if "proposed recommendation" in status:
            return "PR"
        if status == "recommendation" or " recommendation" in status:
            return "REC"
        if "working draft" in status:
            return "WD"
        if "first public" in status:
            return "FPWD"
    return None


def _group_type(text: str) -> str | None:
    if _has(text, ["community group", " cg", "社区组"]):
        return "Community Group"
    if _has(text, ["interest group", " ig"]):
        return "Interest Group"
    if _has(text, ["working group", " wg", "工作组"]):
        return "Working Group"
    return None


def _group_type_from_entities(entities: list[W3CEntity]) -> str | None:
    for entity in entities:
        if entity.entity_type == "group" and entity.group_type:
            return entity.group_type.title()
        if entity.deliverers:
            for deliverer in entity.deliverers:
                lower = deliverer.lower()
                if "working group" in lower:
                    return "Working Group"
                if "interest group" in lower:
                    return "Interest Group"
                if "community group" in lower:
                    return "Community Group"
    return None


def _deliverable_type(text: str) -> str | None:
    if _has(text, ["registry"]):
        return "Registry"
    if _has(text, ["note", "group note"]):
        return "Group Note"
    if _has(text, ["spec", "specification", "technical report", "standard", "标准", "规范"]):
        return "Technical Report"
    return None


def _workflow(intent: str, text: str, current_stage: str | None, target_stage: str | None) -> str:
    if intent == "advance_specification" and (current_stage or target_stage):
        return "recommendation_track_transition"
    if intent == "charter_or_recharter":
        return "charter_review_and_approval"
    if intent == "plan_or_complete_review":
        return "wide_or_horizontal_review"
    if intent == "horizontal_review":
        return "horizontal_review"
    if intent == "handle_objection_or_appeal":
        return "formal_objection_or_appeal"
    if _has(text, ["community group", "incubation"]):
        return "cg_to_wg_incubation_transfer"
    return intent


def _missing_information(
    *,
    intent: str,
    text: str,
    current_stage: str | None,
    target_stage: str | None,
    group_type: str | None,
) -> list[str]:
    missing: list[str] = []
    if intent in {"advance_specification", "plan_or_complete_review", "horizontal_review"}:
        if not current_stage:
            missing.append("current maturity stage")
        if not target_stage and intent == "advance_specification":
            missing.append("target maturity stage or transition")
        if not group_type:
            missing.append("responsible W3C group type")
        if "deliverable" not in text and "spec" not in text and "standard" not in text and "标准" not in text:
            missing.append("specific deliverable or specification")
    if intent == "charter_or_recharter":
        if not group_type:
            missing.append("whether this is a Working Group or Interest Group charter")
        if not _has(text, ["new", "renew", "recharter", "extension", "新", "延期"]):
            missing.append("whether this is a new charter, recharter, renewal, or extension")
    return missing[:5]


def _risk_flags(text: str, citations: list[Citation]) -> list[str]:
    flags: list[str] = []
    checks = [
        ("Patent Policy", ["patent", "ipr", "专利"]),
        ("Formal Objection", ["formal objection", "异议"]),
        ("Appeal", ["appeal", "申诉"]),
        ("AC Review", ["ac review", "advisory committee review"]),
        ("Charter", ["charter", "recharter", "章程"]),
        ("Transition", ["transition", "cr", "rec", "recommendation", "转换"]),
        ("Horizontal Review", _horizontal_review_needles()),
        ("Wide Review", ["wide review", "horizontal review", "横向审查"]),
    ]
    citation_text = " ".join([citation.heading_path or "" for citation in citations]).lower()
    combined = f"{text} {citation_text}"
    for label, needles in checks:
        if _has(combined, needles):
            flags.append(label)
    return flags


def _has(text: str, needles: list[str]) -> bool:
    return any(needle in text for needle in needles)


def _horizontal_review_needles() -> list[str]:
    return [
        "horizontal review",
        "horizontal group",
        "horizontal groups",
        "横向审查",
        "a11y review",
        "accessibility review",
        "i18n review",
        "internationalization review",
        "privacy review",
        "security review",
        "tag review",
        "*-tracker",
        "*-needs-resolution",
        "needs-resolution",
        "horizontal issue tracker",
    ]


def _stage_match(text: str, needle: str) -> bool:
    stripped = needle.strip()
    if stripped.lower() in {"cr", "wd", "pr", "rec"}:
        return bool(re.search(rf"\b{re.escape(stripped.lower())}\b", text))
    return needle in text
