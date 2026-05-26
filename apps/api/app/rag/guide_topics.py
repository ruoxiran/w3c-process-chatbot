from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GuideTopic:
    id: str
    query_needles: tuple[str, ...]
    required_url_needles: tuple[str, ...]
    optional_text_needles: tuple[str, ...] = ()


GUIDE_TOPICS: tuple[GuideTopic, ...] = (
    GuideTopic(
        id="horizontal_review",
        query_needles=(
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
        ),
        required_url_needles=(
            "/policies/process/#doc-reviews",
            "/guide/documentreview",
            "/guide/process/horizontal-groups",
            "/guide/github/issue-metadata",
        ),
        optional_text_needles=(
            "i18n-request",
            "privacy-request",
            "security-request",
            "a11y-request",
            "w3ctag/design-reviews",
        ),
    ),
    GuideTopic(
        id="transition",
        query_needles=(
            "transition",
            "transition request",
            "advance",
            "advancing",
            "milestone",
            "milestones",
            "cr",
            "candidate recommendation",
            "rec",
            "recommendation",
            "推进",
            "转换",
            "候选推荐",
            "推荐标准",
        ),
        required_url_needles=(
            "/guide/transitions",
            "/guide/transitions/milestones",
            "/guide/#rec-track",
        ),
    ),
    GuideTopic(
        id="charter",
        query_needles=(
            "charter",
            "recharter",
            "charter review",
            "charter extension",
            "章程",
        ),
        required_url_needles=(
            "/guide/process/charter",
            "/guide/process/charter-extensions",
            "/guide/tools/new-group",
        ),
    ),
    GuideTopic(
        id="staff_contact",
        query_needles=(
            "staff contact",
            "team contact",
            "teamcontact",
            "staff-contact",
            "职责",
        ),
        required_url_needles=(
            "/guide/teamcontact",
            "/guide/teamcontact/role",
            "/guide/teamcontact/liaison-role",
        ),
    ),
)


def matching_guide_topics(query: str) -> list[GuideTopic]:
    text = query.lower()
    return [
        topic
        for topic in GUIDE_TOPICS
        if any(needle in text for needle in topic.query_needles)
    ]
