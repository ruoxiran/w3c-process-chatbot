from __future__ import annotations

import re

from app.models.schemas import ChatTurn, SourceType, TaskPlan
from app.services.process_state import STAGE_KEYWORDS


def plan_task(message: str, history: list[ChatTurn] | None = None) -> TaskPlan:
    """Create a small, deterministic plan before retrieval.

    The plan is deliberately conservative: it identifies the user's procedural
    goal, the source families needed to answer it, and the focused retrieval
    queries that should be attempted before answer generation.
    """
    history = history or []
    text = _compact_context(message, history)
    lower = text.lower()
    intent_type = _intent_type(lower)
    current_stage = _stage(lower, current=True)
    target_stage = _stage(lower, current=False)
    spec_or_group = _spec_or_group(text)
    needed_sources = _needed_sources(intent_type, lower)
    risk_flags = _risk_flags(lower)
    search_queries = _search_queries(
        message=message,
        intent_type=intent_type,
        current_stage=current_stage,
        target_stage=target_stage,
        spec_or_group=spec_or_group,
        needed_sources=needed_sources,
    )

    confidence = 0.48
    confidence += 0.12 if intent_type != "explain_process" else 0
    confidence += 0.1 if spec_or_group else 0
    confidence += 0.08 if current_stage or target_stage else 0
    confidence += 0.07 if history else 0

    return TaskPlan(
        intent_type=intent_type,
        user_goal=_user_goal(intent_type, message),
        current_stage=current_stage,
        target_stage=target_stage,
        spec_or_group=spec_or_group,
        needed_sources=needed_sources,
        answer_shape=_answer_shape(intent_type),
        search_queries=search_queries,
        risk_flags=risk_flags,
        confidence=min(confidence, 0.9),
    )


def build_planned_retrieval_query(base_query: str, plan: TaskPlan) -> str:
    lines = [
        base_query,
        "",
        "Task plan retrieval requirements:",
        f"intent_type={plan.intent_type}",
        f"user_goal={plan.user_goal}",
        f"answer_shape={plan.answer_shape}",
    ]
    if plan.current_stage:
        lines.append(f"current_stage={plan.current_stage}")
    if plan.target_stage:
        lines.append(f"target_stage={plan.target_stage}")
    if plan.spec_or_group:
        lines.append(f"spec_or_group={plan.spec_or_group}")
    if plan.needed_sources:
        lines.append(f"needed_sources={', '.join(source.value for source in plan.needed_sources)}")
    if plan.risk_flags:
        lines.append(f"risk_flags={', '.join(plan.risk_flags)}")
    if plan.search_queries:
        lines.append("focused_queries:")
        lines.extend(f"- {query}" for query in plan.search_queries)
    return "\n".join(lines)


def _compact_context(message: str, history: list[ChatTurn]) -> str:
    recent = " ".join(turn.content for turn in history[-4:])
    return f"{recent} {message}" if recent else message


_VALID_INTENT_TYPES = frozenset({
    "advance_specification",
    "horizontal_review",
    "charter_or_recharter",
    "coordinate_with_staff_contact",
    "run_group_process",
    "transfer_incubation_to_wg",
    "check_patent_policy",
    "handle_objection_or_appeal",
    # Spec-authoring tools (ReSpec / Bikeshed / Pubrules / Echidna /
    # HTMLdiff). The Process and Guidebook describe WHAT must happen;
    # this intent surfaces the actual tools the editor uses to produce
    # and publish the document.
    "author_spec",
    "explain_process",
})


def _intent_type(text: str) -> str:
    # Keyword rules run BEFORE deferring to a router-supplied intent_type. A
    # specific keyword match ("horizontal review", "patent exclusion",
    # "formal objection") is a strong signal that the question is really
    # about that topic; the LLM router's broad ``intent_type=explain_process``
    # label, which it emits frequently for any ambiguous question, would
    # otherwise override these correct keyword classifications.
    if _has(text, ["formal objection", "appeal", "异议", "申诉"]):
        return "handle_objection_or_appeal"
    if _has(text, ["patent", "ipr", "exclusion opportunity", "专利"]):
        return "check_patent_policy"
    if _has(text, ["tilt"]):
        return "charter_or_recharter"
    if _has(text, ["charter issue", "charter review", "charter label", "recharter"]):
        return "charter_or_recharter"
    if _has(text, ["random blog", "unofficial", "non-official"]) and _has(text, ["transition", "authority"]):
        return "advance_specification"
    if _has(text, ["ac review", "advisory committee review"]) and _has(text, ["transition", "recommendation"]):
        return "advance_specification"
    if _has(text, _horizontal_review_needles()):
        return "horizontal_review"
    if _has(text, ["charter", "recharter", "active charter", "章程"]):
        return "charter_or_recharter"
    # Stage-transition signals (CR/PR/REC + "transition") should be
    # classified as ``advance_specification`` BEFORE the broader "review"
    # keyword catches "wide review needed?" as a plain review-planning
    # question. The user's real intent in "我们的 spec 在 CR，下一步
    # transition to PR 需要 wide review 吗?" is advancing the spec, with
    # review as a sub-question.
    #
    # Stage terms must be SPECIFIC enough to not collide with neutral words.
    # E.g. " pr" alone would match "Working Group p[rocess]"; "to pr" or
    # "in cr" requires the user to be talking about the maturity stage
    # explicitly.
    has_transition_verb = _has(text, ["transition", "transitioning", "推进", "转换"])
    has_stage_term = _has(
        text,
        [
            "in cr", "at cr", "to cr", "from cr", "cr to ", "cr snapshot", "cr draft",
            "in pr", "at pr", "to pr", "from pr", "pr to ",
            "to rec", "rec snapshot", "rec track", "recommendation track",
            "fpwd", "first public working draft",
            "candidate recommendation", "proposed recommendation",
            "候选推荐", "推荐标准",
        ],
    )
    if has_transition_verb and has_stage_term:
        return "advance_specification"
    if _has(text, ["wide review", "review", "审查"]):
        return "plan_or_complete_review"
    if _has(text, ["staff contact", "team contact", "liaison", "职责"]):
        return "coordinate_with_staff_contact"
    if _has(text, [
        "chair", "meeting", "agenda", "minutes", "会议", "主席",
        # Meeting tooling — IRC bots and the scribing toolchain are
        # core to W3C meetings but the original keyword list missed
        # them, so "how to scribe?" was falling through to
        # advance_specification and retrieving REC-transition chunks
        # instead of the dedicated zakim.html / rrsagent.html /
        # scribe.html guide pages.
        "scribe", "scribing", "zakim", "rrsagent", "irc", "irc bot",
        "scribe.perl", "记录员", "会议记录",
    ]):
        return "run_group_process"
    if _has(text, ["community group", "incubation", "cg ", "转入 working group"]):
        return "transfer_incubation_to_wg"
    # Spec-authoring tools — ReSpec / Bikeshed / Pubrules / Echidna /
    # HTMLdiff / spec editing in general. Goes BEFORE the advance_
    # specification fallback below because "publication" and "publish"
    # are in that list, and the tool-question rephrasings ("how do I
    # auto-publish via echidna" / "how do I validate my draft with
    # pubrules") otherwise fall through and get the wrong intent.
    if _has(text, [
        "respec", "re-spec", "bikeshed", "pubrules", "echidna", "htmldiff",
        "spec editor", "spec authoring", "spec template", "edit a spec",
        "write a spec", "writing a spec", "author a spec", "author a w3c spec",
        "authoring a spec", "create a spec", "creating a spec",
        "spec source", "spec markup",
        "speced.github.io", "respec.org",
        # Repo / publication tooling
        "repo manager", "repo-manager",
        "auto-publication", "auto-publish", "automated publication",
        # WBS / Call for Review tooling
        "wbs", "call for review", "cfr ",
        "编辑器", "撰写规范", "编写规范",
    ]):
        return "author_spec"
    if _has(
        text,
        [
            "w3c process step",
            "recommendation track",
            "transition",
            "advance",
            "next step",
            "milestone",
            "fpwd",
            "first public working draft",
            "working draft",
            "publication",
            "publish",
            "implementation experience",
            "cr",
            "rec",
            "推进",
            "下一步",
            "转换",
        ],
    ):
        return "advance_specification"
    # No keyword rule fired. Defer to the router's ``intent_type=...`` label
    # if it produced something other than the catch-all ``explain_process``
    # — the router has more context than the keyword table for unusual
    # phrasings. Otherwise return the catch-all.
    explicit = re.search(r"\bintent_type=([a-z_]+)\b", text)
    if explicit:
        value = explicit.group(1)
        if value in _VALID_INTENT_TYPES and value != "explain_process":
            return value
    return "explain_process"


def _needed_sources(intent_type: str, text: str) -> list[SourceType]:
    sources = [SourceType.process]
    if intent_type in {
        "advance_specification",
        "coordinate_with_staff_contact",
        "run_group_process",
        "transfer_incubation_to_wg",
        "plan_or_complete_review",
        "horizontal_review",
    } or _has(text, ["how", "next", "怎么", "下一步", "guide", "practice"]):
        sources.append(SourceType.guide)
    if intent_type in {"check_patent_policy", "handle_objection_or_appeal"}:
        sources.append(SourceType.related_policy)
    if intent_type == "charter_or_recharter" or _has(text, _draft_context_needles()):
        sources.append(SourceType.repo)
    return sources


def _search_queries(
    *,
    message: str,
    intent_type: str,
    current_stage: str | None,
    target_stage: str | None,
    spec_or_group: str | None,
    needed_sources: list[SourceType],
) -> list[str]:
    seed = [message]
    subject = spec_or_group or "the W3C deliverable"
    if intent_type == "advance_specification":
        transition = " ".join(part for part in [current_stage, "to", target_stage] if part)
        seed.extend(
            [
                f"{subject} recommendation track transition requirements {transition}".strip(),
                f"{subject} Guidebook transition request milestones next step",
                "advancing on the Recommendation track transition requirements implementation experience AC Review",
            ]
        )
    elif intent_type == "charter_or_recharter":
        seed.extend(
            [
                f"{subject} charter review approval",
                "Guidebook charter recharter practical steps",
                "w3c strategy charter label issue tracker charter recharter review",
            ]
        )
    elif intent_type == "plan_or_complete_review":
        seed.extend(["wide review horizontal review process requirements", "Guidebook wide review horizontal review planning"])
    elif intent_type == "horizontal_review":
        seed.extend(
            [
                f"{subject} horizontal review wide review Guidebook document review",
                f"{subject} horizontal review GitHub request issue tracker labels",
                "Guidebook How to get horizontal review accessibility i18n privacy security TAG GitHub request",
                "horizontal review labels *-tracker *-needs-resolution issue metadata tracker boards",
                "transition request horizontal *-needs-resolution tracker boards",
            ]
        )
    elif intent_type == "coordinate_with_staff_contact":
        seed.extend(["Staff Contact Team Contact responsibilities", "Guidebook Staff Contact role"])
    elif intent_type == "run_group_process":
        seed.extend([
            "W3C group meeting process chair minutes decision",
            "Guidebook chair meetings agenda minutes",
            # Tool-specific seeds so retrieval lands on the dedicated
            # zakim.html / rrsagent.html / scribe.html guide pages
            # when the user asks "how to scribe?", "what is Zakim?",
            # "how do I queue with rrsagent?". Without these the
            # generic meeting/minutes seeds drift to chapters that
            # only mention scribing tangentially.
            "Zakim IRC bot agenda queue start meeting take up",
            "RRSAgent IRC bot record minutes log scribe.perl",
            "scribe IRC bot scribe.perl meeting minutes record",
        ])
    elif intent_type == "transfer_incubation_to_wg":
        seed.extend(["Community Group specification transfer Working Group incubation", "Guidebook CG transition Working Group"])
    elif intent_type == "author_spec":
        # Spec-authoring tools — most of the canonical docs live
        # OUTSIDE the corpus (respec.org / speced.github.io /
        # services.w3.org/htmldiff are reference action surfaces,
        # not indexed). What IS in the corpus: the Guidebook editor
        # role chapter, repo-management chapter, pubrules-conformance
        # content, github/w3c.json chapter. Steer retrieval to those
        # so the model has SOMETHING to ground its "use ReSpec or
        # Bikeshed" recommendation against.
        seed.extend([
            "Guidebook editor role spec author responsibilities",
            "Guidebook github repo management w3c.json automated publication",
            "pubrules publication rules SOTD boilerplate conformance",
            "echidna automated publication working draft snapshot",
            f"{subject} editor draft publication tooling",
        ])
    elif intent_type == "handle_objection_or_appeal":
        seed.extend(["Formal Objection appeal process", "Guidebook formal objection escalation"])
    elif intent_type == "check_patent_policy":
        seed.extend(["Patent Policy exclusion opportunity Recommendation track", "W3C Process Patent Policy"])

    if SourceType.guide in needed_sources:
        seed.append(f"{subject} Guidebook practical guidance next steps")
    if SourceType.repo in needed_sources:
        if intent_type == "charter_or_recharter":
            seed.append(f"{subject} w3c strategy GitHub charter issue tracker label")
        else:
            seed.append(f"{subject} official editor draft GitHub repository context issues source files")
    return _dedupe(seed)[:6]


def _stage(text: str, *, current: bool) -> str | None:
    transition = re.search(r"\bfrom\s+([a-z0-9 -]+?)\s+to\s+([a-z0-9 -]+?)(?:[?.!,;:]|$)", text)
    directional = text
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


def _spec_or_group(text: str) -> str | None:
    patterns = [
        r"\b([A-Z][A-Za-z0-9 ]{1,60}?(?:Working Group|Interest Group|Community Group))\b",
        r"\b([A-Z][A-Za-z0-9 ]{1,60}?(?:specification|spec|standard|module|API))\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return " ".join(match.group(1).split())
    quoted = re.search(r"['\"]([^'\"]{2,80})['\"]", text)
    if quoted:
        return quoted.group(1).strip()
    return None


def _user_goal(intent_type: str, message: str) -> str:
    goals = {
        "advance_specification": "Determine the current Recommendation-track position and the concrete next transition work.",
        "charter_or_recharter": "Determine the charter/recharter workflow and the approvals or review steps needed.",
        "plan_or_complete_review": "Determine what review work is required and how to plan or complete it.",
        "horizontal_review": "Determine horizontal review obligations, GitHub request paths, labels, trackers, and transition-readiness checks.",
        "coordinate_with_staff_contact": "Clarify Staff Contact or Team Contact responsibilities for the workflow.",
        "run_group_process": "Clarify how to run the relevant W3C group process step.",
        "transfer_incubation_to_wg": "Determine how incubated Community Group work can move toward Working Group standardization.",
        "handle_objection_or_appeal": "Determine how a Formal Objection or appeal should be handled.",
        "check_patent_policy": "Determine Patent Policy implications and required checks.",
    }
    return goals.get(intent_type, f"Answer the W3C Process question: {message[:180]}")


def _answer_shape(intent_type: str) -> str:
    if intent_type == "charter_or_recharter":
        return "charter_recharter_steps_with_w3c_strategy_issue_tracking"
    if intent_type in {"advance_specification", "transfer_incubation_to_wg"}:
        return "conclusion_current_state_next_steps_missing_information"
    if intent_type == "horizontal_review":
        return "horizontal_review_checklist_with_github_request_links_and_tracker_checks"
    if intent_type in {"check_patent_policy", "handle_objection_or_appeal"}:
        return "risk_first_conclusion_required_escalation_steps"
    return "short_conclusion_with_actionable_steps"


def _risk_flags(text: str) -> list[str]:
    checks = [
        ("Patent Policy", ["patent", "ipr", "exclusion opportunity", "专利"]),
        ("Formal Objection", ["formal objection", "异议"]),
        ("Appeal", ["appeal", "申诉"]),
        ("AC Review", ["ac review", "advisory committee review"]),
        ("Charter", ["charter", "recharter", "章程"]),
        ("Transition", ["transition", "cr", "rec", "recommendation", "转换"]),
        ("Horizontal Review", _horizontal_review_needles()),
        ("Draft Context", _draft_context_needles()),
        ("Wide Review", ["wide review", "横向审查"]),
    ]
    return [label for label, needles in checks if _has(text, needles)]


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
        "tag design review",
        "design review",
        "*-tracker",
        "*-needs-resolution",
        "needs-resolution",
        "horizontal issue tracker",
    ]


def _draft_context_needles() -> list[str]:
    return [
        "editor draft",
        "editors draft",
        "draft repo",
        "github repo",
        "github issue",
        "github issues",
        "pull request",
        "source file",
        "repo context",
        "草案",
        "仓库",
        "github",
        "上下文",
    ]


def _stage_match(text: str, needle: str) -> bool:
    stripped = needle.strip()
    if stripped.lower() in {"cr", "wd", "pr", "rec"}:
        return bool(re.search(rf"\b{re.escape(stripped.lower())}\b", text))
    return needle in text


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
