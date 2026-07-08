from app.models.schemas import Citation, CompiledContext, DraftContext, NextStep, SourceType


TopicAnswer = tuple[str, list[str]]


def _wants_english(locale: str, message: str) -> bool:
    """Resolve the answer language. Explicit locales win; ``auto`` (the API
    default) detects from the message — a message with no CJK characters gets
    the English templates, not the Chinese ones."""
    if locale.startswith("en"):
        return True
    if locale.startswith("zh"):
        return False
    return not any("一" <= ch <= "鿿" for ch in message)


def build_refusal(locale: str = "auto", message: str = "") -> str:
    if _wants_english(locale, message):
        return (
            "This assistant only answers questions about the W3C Process, the W3C Guidebook, "
            "and W3C standards workflow. Please rephrase your question around those topics."
        )
    return "本系统只回答 W3C Process、W3C Guidebook 和 W3C 标准推进流程相关问题。请把问题改写为相关流程问题。"


def build_grounded_answer(
    message: str,
    citations: list[Citation],
    locale: str = "auto",
    draft_contexts: list[DraftContext] | None = None,
    compiled_context: CompiledContext | None = None,
) -> TopicAnswer:
    topic = _detect_answer_topic(message)
    process_link = _preferred_process_link(topic, citations)
    guide_link = _preferred_guide_link(topic, citations)
    draft_contexts = draft_contexts or []

    if _wants_english(locale, message):
        answer, default_steps = _english_topic_answer(topic, process_link, guide_link)
        answer, default_steps = _apply_compiled_context(answer, default_steps, compiled_context, english=True)
        if topic == "charter":
            answer = _append_strategy_status_summary(answer, draft_contexts, english=True)
        return answer, _contextual_next_steps(message, citations, locale, default_steps)

    answer, default_steps = _chinese_topic_answer(topic, process_link, guide_link)
    answer, default_steps = _apply_compiled_context(answer, default_steps, compiled_context, english=False)
    if topic == "charter":
        answer = _append_strategy_status_summary(answer, draft_contexts, english=False)
    return answer, _contextual_next_steps(message, citations, locale, default_steps)


def build_next_step_details(
    message: str,
    citations: list[Citation],
    steps: list[str],
    compiled_context: CompiledContext | None = None,
) -> list[NextStep]:
    source_seed = message
    if compiled_context:
        source_seed = f"{message}\n{compiled_context.summary}\n" + "\n".join(compiled_context.guide_signals)
    return [
        _next_step_with_source(step, _source_for_step(source_seed, step, citations))
        for step in steps
    ]


def _detect_answer_topic(message: str) -> str:
    text = message.lower()
    if "formal objection" in text or "appeal" in text or "异议" in text or "申诉" in text:
        return "objection"
    if "patent" in text or "ipr" in text or "专利" in text:
        return "patent"
    if _has_any(text, _horizontal_review_needles()) and "tilt" not in text:
        if _has_any(
            text,
            [
                "charter issue",
                "charter label",
                "charter review issue",
                "recharter",
                "w3c/strategy",
                "strategy issue",
                "horizontal review requested",
            ],
        ):
            return "charter"
        return "horizontal_review"
    if "charter" in text or "recharter" in text or "章程" in text:
        return "charter"
    if "staff contact" in text or "team contact" in text:
        return "staff_contact"
    if "community group" in text or "incubation" in text:
        return "community_group"
    if "member submission" in text:
        return "member_submission"
    if "registry" in text:
        return "registry"
    if (
        "cr" in text
        or "candidate recommendation" in text
        or "rec" in text
        or "recommendation" in text
        or "implementation experience" in text
        or "ac review" in text
        or "候选推荐" in text
        or "推荐标准" in text
    ):
        return "recommendation_transition"
    if "wide review" in text or "horizontal review" in text or "review" in text or "审查" in text:
        return "review"
    if "fpwd" in text or "working draft" in text or "wd" in text:
        return "working_draft"
    return "general"


def _contextual_next_steps(
    message: str,
    citations: list[Citation],
    locale: str,
    default_steps: list[str],
) -> list[str]:
    context = _context_text(message, citations)
    query = message.lower()
    english = _wants_english(locale, message)
    query_mentions_charter = _has_any(query, ["charter", "recharter", "章程"])
    query_mentions_transition = _has_any(
        query,
        [
            "milestones",
            "nextstep",
            "transition request",
            "transition",
            "转换",
            "cr",
            "candidate recommendation",
            "rec",
            "recommendation",
            "候选推荐",
            "推荐标准",
        ],
    )
    query_mentions_horizontal_review = _has_any(query, _horizontal_review_needles())

    if query_mentions_charter or _has_any(
        context,
        [
            "w3c/strategy",
            "charter label",
            "charter review issue tracker",
            "recharter issue tracker",
        ],
    ):
        return _steps(
            english,
            [
                "Confirm whether this is a new charter, recharter, or charter extension and identify the owning group.",
                "Check Process requirements for group scope, deliverables, dependencies, review, and approval.",
                "Use the Guidebook charter guidance to prepare the draft and practical review checklist.",
                "Create or locate the tracking issue in the W3C Strategy repository with the `charter` label: https://github.com/w3c/strategy/issues?q=label%3Acharter.",
                "Check open and closed Strategy issues, issue dates, and the charter end date; after `Horizontal review requested` and required completion labels are present with no `*-needs-resolution` blockers, ask the Staff Contact to start the TiLT review path.",
            ],
            [
                "确认这是新建 charter、recharter 还是 charter extension，并确认所属 group。",
                "核对 Process 中关于 group scope、deliverables、依赖、review 和 approval 的要求。",
                "使用 Guidebook 的 charter 指南准备草案和实践审查清单。",
                "在 W3C Strategy 仓库中创建或找到带 `charter` label 的 tracking issue：https://github.com/w3c/strategy/issues?q=label%3Acharter。",
                "同时检查 open/closed Strategy issues、issue 时间线和 charter end date；当已有 `Horizontal review requested`、所需 review completed labels 到位且没有 `*-needs-resolution` 阻塞时，请 Staff Contact 启动 TiLT review 路径。",
            ],
        )

    if _has_any(query, ["patent", "ipr", "exclusion opportunity"]):
        return _steps(
            english,
            [
                "Identify the group, deliverable, maturity stage, and relevant Patent Policy trigger.",
                "Check the current W3C Process and referenced Patent Policy material before drawing conclusions.",
                "Do not treat Guidebook practice notes or user claims as legal or policy overrides.",
                "Escalate to the Staff Contact or W3C Team for confirmation before saying the patent question is resolved.",
            ],
            [
                "确认 group、deliverable、成熟度阶段和相关 Patent Policy trigger。",
                "先核对当前 W3C Process 和引用的 Patent Policy 材料，再下结论。",
                "不要让 Guidebook 实践建议或用户声明覆盖正式政策。",
                "在认定 patent 问题已解决前，交给 Staff Contact 或 W3C Team 确认。",
            ],
        )

    if _has_any(query, ["formal objection", "appeal", "异议", "申诉"]):
        return _steps(
            english,
            [
                "Confirm the decision, review, or transition where the Formal Objection or appeal issue arose.",
                "Locate the current Process rules for Formal Objections or appeals.",
                "Keep the objection rationale, attempted resolution, and unresolved points visible in the record.",
                "Escalate unresolved or high-risk cases to the Staff Contact or W3C Team before advising on outcomes.",
            ],
            [
                "确认 Formal Objection 或 appeal 对应的决定、审查或 transition。",
                "查找当前 Process 中 Formal Objection 或 appeal 的规则。",
                "保留异议理由、解决尝试和未解决点的记录。",
                "高风险或未解决情况先交给 Staff Contact 或 W3C Team，再判断后续路径。",
            ],
        )

    if _has_any(query, ["staff contact", "team contact"]):
        return _steps(
            english,
            [
                "Identify the specific group, deliverable, and decision where Staff Contact support is needed.",
                "Check the Process requirement for Staff Contact or Team Contact responsibilities in the cited group section.",
                "Use the Guidebook Staff Contact material to map practical duties, liaison coordination, tracker checks, and escalation points.",
                "Record which items need Staff Contact confirmation before the group treats the workflow step as complete.",
            ],
            [
                "先确认具体 group、deliverable，以及需要 Staff Contact 支持的是哪个决定或流程节点。",
                "核对引用到的 Process 章节中关于 Staff Contact / Team Contact 职责的要求。",
                "结合 Guidebook 的 Staff Contact 材料，梳理实践职责、liaison 协调、tracker 检查和需要升级的问题。",
                "记录哪些事项必须由 Staff Contact 确认后，工作组才能视为流程完成。",
            ],
        )

    if _has_any(query, ["community group", "cg", "转到 working group", "转入 working group"]):
        return _steps(
            english,
            [
                "Identify the Community Group output, its maturity, and whether the target path is a new or existing Working Group.",
                "Use the Guidebook CG transition material to prepare scope, contributors, issue history, and transfer considerations.",
                "Check the Process requirements for the Working Group charter, participation, and Patent Policy implications.",
                "Coordinate with the W3C Team before treating the Community Group work as ready for Recommendation-track standardization.",
            ],
            [
                "确认 Community Group 产出、成熟度，以及目标路径是新建 Working Group 还是转入已有 Working Group。",
                "使用 Guidebook 的 CG transition 材料整理 scope、贡献者、issue 历史和转移注意事项。",
                "核对 Process 中关于 Working Group charter、参与要求和 Patent Policy 影响的要求。",
                "在把 Community Group 工作视为可进入 Recommendation-track 标准化前，先与 W3C Team 协调确认。",
            ],
        )

    if (query_mentions_transition and not query_mentions_horizontal_review) or (
        not query_mentions_horizontal_review
        and _has_any(
            context,
            ["milestones", "nextstep", "transition request", "transition requirements"],
        )
    ):
        return _steps(
            english,
            [
                "Identify the current maturity stage and the exact target transition.",
                "Use the Guidebook transition tools or milestones guidance to plan dates and expected review windows.",
                "Check the Process transition requirements for the target stage and gather the required evidence.",
                "Confirm unresolved issues, horizontal review status, objections, dependencies, and Staff Contact or Team verification before submitting the request.",
            ],
            [
                "确认当前 maturity stage 和准确的目标 transition。",
                "使用 Guidebook 的 transition tools 或 milestones 指导规划日期和 review window。",
                "核对 Process 中目标阶段的 transition requirements，并整理所需证据。",
                "提交请求前确认未解决 issue、horizontal review 状态、异议、依赖和 Team verification。",
            ],
        )

    if query_mentions_horizontal_review or (
        not query_mentions_charter
        and not query_mentions_transition
        and _has_any(
            context,
            [
                "how to get horizontal review",
                "working with horizontal review labels",
                "horizontal groups",
                "horizontal issue tracker",
                "needs-resolution",
                "a11y-request",
                "i18n-request",
                "privacy-request",
                "security-request",
                "design-reviews",
            ],
        )
    ):
        return _steps(
            english,
            [
                "Use the Guidebook Wide Review page to plan horizontal review early, before the target transition.",
                "Request the relevant horizontal reviews through GitHub as applicable: APA/a11y (https://github.com/w3c/a11y-request/issues/new/choose), TAG (https://github.com/w3ctag/design-reviews/issues/new/choose), I18N (https://github.com/w3c/i18n-request/issues/new/choose), Privacy (https://github.com/w3cping/privacy-request/issues/new/choose), and Security (https://github.com/w3c/security-request/issues/new/choose).",
                "Create or maintain a GitHub meta-issue in the specification repository to track review requests, responses, and resolutions.",
                "Use `*-tracker` to draw horizontal group attention, but do not add or remove `*-needs-resolution`; that label belongs to horizontal groups.",
                "Before requesting transition, check the horizontal tracker boards and coordinate with the horizontal group and Staff Contact on lingering `*-needs-resolution` issues.",
                "Ask the Staff Contact to verify whether the tracked review evidence is complete enough for the intended transition.",
            ],
            [
                "先用 Guidebook 的 Wide Review 页面规划 horizontal review，并尽量早于目标 transition 发起。",
                "按需要通过 GitHub 请求 APA/a11y（https://github.com/w3c/a11y-request/issues/new/choose）、TAG（https://github.com/w3ctag/design-reviews/issues/new/choose）、I18N（https://github.com/w3c/i18n-request/issues/new/choose）、Privacy（https://github.com/w3cping/privacy-request/issues/new/choose）、Security（https://github.com/w3c/security-request/issues/new/choose）等 horizontal review。",
                "在 specification repo 中创建或维护一个 GitHub meta issue，跟踪 review 请求、反馈和解决状态。",
                "可以用 `*-tracker` 提醒 horizontal group，但不要自行添加或移除 `*-needs-resolution`，这个标签由 horizontal group 使用。",
                "提交 transition request 前检查 horizontal tracker boards，并与 horizontal group 和 Staff Contact 确认未关闭的 `*-needs-resolution` issue。",
            ],
        )

    if _has_any(query, ["chair", "meeting", "meetings", "主席", "会议"]) or (
        _has_any(context, ["chair/meetings", "chairs meetings", "official meetings"])
        and not _has_any(query, ["staff contact", "team contact"])
    ):
        return _steps(
            english,
            [
                "Confirm whether the meeting is an official W3C group meeting covered by the Process meeting rules.",
                "Check the cited Process meeting requirements for participation, notice, record keeping, and decision handling.",
                "Use the Guidebook Chair meeting material to prepare agenda, facilitation, minutes, and action tracking.",
                "After the meeting, publish or link the record and make unresolved decisions or objections explicit.",
            ],
            [
                "确认该会议是否属于 Process 规则覆盖的正式 W3C group meeting。",
                "核对引用到的 Process meeting 要求，包括参与、通知、记录和决策处理。",
                "使用 Guidebook 中 Chair meeting 材料准备议程、主持方式、会议记录和 action tracking。",
                "会后发布或链接记录，并明确未解决的决定、异议或后续 action。",
            ],
        )

    if _has_any(context, ["editor", "editors-draft", "editor's role"]):
        return _steps(
            english,
            [
                "Identify the editor responsibility being asked about: draft maintenance, issue handling, publication preparation, or tooling.",
                "Use the Guidebook editor material to map the practical work expected from editors.",
                "Check whether the Process or publication rules impose requirements beyond Guidebook practice.",
                "Confirm the division of responsibility with Chairs and the Staff Contact.",
            ],
            [
                "先确认问题涉及 editor 的哪类职责：草案维护、issue 处理、发布准备还是工具链。",
                "使用 Guidebook 的 editor 材料梳理 editor 的实际工作内容。",
                "核对是否存在 Process 或 publication rules 中高于 Guidebook 实践建议的正式要求。",
                "与 Chairs 和 Staff Contact 确认职责分工。",
            ],
        )

    return _dedupe_steps(default_steps)


def _apply_compiled_context(
    answer: str,
    default_steps: list[str],
    compiled_context: CompiledContext | None,
    *,
    english: bool,
) -> tuple[str, list[str]]:
    if not compiled_context:
        return answer, default_steps
    prefix = compiled_context.summary.strip()
    if prefix:
        answer = f"{prefix}\n\n{answer}"
    if compiled_context.next_step_candidates:
        lead = (
            "Compiled spec context suggests these concrete next checks:"
            if english
            else "结合 compiled spec context，当前更适合优先检查这些具体步骤："
        )
        answer = f"{answer}\n\n{lead}"
        return answer, _dedupe_steps([*compiled_context.next_step_candidates, *default_steps])
    return answer, default_steps


def _context_text(message: str, citations: list[Citation]) -> str:
    parts = [message]
    for citation in citations:
        parts.extend(
            [
                citation.title,
                str(citation.url),
                citation.heading_path or "",
                citation.section_id or "",
                citation.quote or "",
            ]
        )
    return " ".join(parts).lower()


def _append_strategy_status_summary(answer: str, draft_contexts: list[DraftContext], *, english: bool) -> str:
    strategy = next((context for context in draft_contexts if context.repo_full_name == "w3c/strategy"), None)
    if not strategy or not strategy.snippets:
        return answer

    items = []
    for snippet in strategy.snippets[:3]:
        text = snippet.text
        labels = _field(text, "labels")
        state = _field(text, "state")
        updated = _field(text, "updated_at")
        closed = _field(text, "closed_at")
        horizontal = _field(text, "horizontal_review_requested")
        completed = _field(text, "completed_horizontal_reviews")
        blockers = _field(text, "needs_resolution_labels")
        tilt = _field(text, "tilt_readiness_signal")
        if english:
            items.append(
                f"- {snippet.title or snippet.path}: state={state or 'unknown'}, updated={updated or 'unknown'}, "
                f"closed={closed or 'open'}, horizontal_requested={horizontal or 'unknown'}, "
                f"completed_reviews={completed or '(none)'}, blockers={blockers or '(none)'}, "
                f"TiLT signal={tilt or 'not_yet_clear'}."
            )
        else:
            items.append(
                f"- {snippet.title or snippet.path}：state={state or 'unknown'}，updated={updated or 'unknown'}，"
                f"closed={closed or 'open'}，horizontal_requested={horizontal or 'unknown'}，"
                f"completed_reviews={completed or '(none)'}，blockers={blockers or '(none)'}，"
                f"TiLT signal={tilt or 'not_yet_clear'}。"
            )

    if english:
        summary = (
            "\n\nCurrent `w3c/strategy` charter issue signals from the GitHub tracker:\n"
            + "\n".join(items)
        )
    else:
        summary = (
            "\n\n当前 `w3c/strategy` charter issue tracker 中的状态信号：\n"
            + "\n".join(items)
        )
    return f"{answer}{summary}"


def _field(text: str, name: str) -> str | None:
    marker = f"{name}="
    if marker not in text:
        return None
    value = text.split(marker, 1)[1].split(";", 1)[0].strip()
    return value or None


def _preferred_process_link(topic: str, citations: list[Citation]) -> str:
    if topic == "horizontal_review":
        matched = _find_citation(citations, SourceType.process, ["#doc-reviews", "review responsibilities", "wide review"])
        if matched:
            return str(matched.url)
    return next((str(c.url) for c in citations if c.source_type == SourceType.process), "https://www.w3.org/policies/process/")


def _preferred_guide_link(topic: str, citations: list[Citation]) -> str:
    if topic == "horizontal_review":
        matched = _find_citation(citations, SourceType.guide, ["documentreview", "horizontal review", "wide review"])
        if matched:
            return str(matched.url)
    return next((str(c.url) for c in citations if c.source_type == SourceType.guide), "https://www.w3.org/guide/")


def _has_any(text: str, needles: list[str]) -> bool:
    return any(needle.lower() in text for needle in needles)


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


def _steps(english: bool, en_steps: list[str], zh_steps: list[str]) -> list[str]:
    return _dedupe_steps(en_steps if english else zh_steps)


def _dedupe_steps(steps: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for step in steps:
        normalized = step.strip()
        key = normalized.lower()
        if normalized and key not in seen:
            deduped.append(normalized)
            seen.add(key)
    return deduped[:5]


def _next_step_with_source(step: str, citation: Citation | None) -> NextStep:
    if citation is None:
        return NextStep(text=step)
    return NextStep(
        text=step,
        source_title=citation.title,
        source_url=citation.url,
        source_type=citation.source_type,
        source_heading=citation.heading_path,
    )


def _source_for_step(message: str, step: str, citations: list[Citation]) -> Citation | None:
    step_text = f"{message} {step}".lower()
    step_only = step.lower()

    if _has_any(step_only, ["process", "正式要求", "规范", "requirement", "requirements", "team verification"]):
        matched = _find_citation(citations, SourceType.process, [])
        if matched:
            return matched

    source_preferences: list[tuple[list[str], SourceType, list[str]]] = [
        (
            [
                "horizontal review",
                "wide review",
                "a11y",
                "accessibility",
                "i18n",
                "privacy",
                "security",
                "tag",
                "tracker",
                "needs-resolution",
                "横向审查",
            ],
            SourceType.guide,
            ["documentreview", "horizontal-groups", "issue-metadata", "transitions"],
        ),
        (["chair", "meeting", "meetings", "会议", "agenda", "minutes"], SourceType.guide, ["chair/meetings", "#run"]),
        (["staff contact", "team contact", "liaison", "职责"], SourceType.guide, ["teamcontact"]),
        (["community group", "cg ", "cg-", "转到 working group", "转入 working group"], SourceType.guide, ["cg-transition", "incubation"]),
        (["milestones", "nextstep", "transition tools", "dates", "review window"], SourceType.guide, ["transitions", "#rec-track"]),
        (["editor", "editors"], SourceType.guide, ["editor"]),
        (["guidebook", "guidebook", "实践", "practice"], SourceType.guide, []),
        (["process", "requirement", "requirements", "规范", "要求", "patent policy"], SourceType.process, []),
    ]

    for needles, source_type, source_needles in source_preferences:
        if not _has_any(step_text, needles):
            continue
        matched = _find_citation(citations, source_type, source_needles)
        if matched:
            return matched

    return _find_citation(citations, SourceType.guide, []) or _find_citation(citations, SourceType.process, [])


def _find_citation(citations: list[Citation], source_type: SourceType, needles: list[str]) -> Citation | None:
    typed = [citation for citation in citations if citation.source_type == source_type]
    if not needles:
        return typed[0] if typed else None

    for citation in typed:
        haystack = " ".join(
            [
                citation.title,
                str(citation.url),
                citation.heading_path or "",
                citation.section_id or "",
                citation.quote or "",
            ]
        ).lower()
        if any(needle.lower() in haystack for needle in needles):
            return citation

    return typed[0] if typed else None


def _english_topic_answer(topic: str, process_link: str, guide_link: str) -> TopicAnswer:
    answers: dict[str, TopicAnswer] = {
        "recommendation_transition": (
            "For a Recommendation-track transition such as Candidate Recommendation (CR) to Recommendation (REC), treat the W3C Process Document "
            f"as the controlling source ({process_link}). The safe next move is to verify the spec's "
            "current maturity, collect implementation experience and other transition evidence required by the Process, confirm reviews including AC Review when applicable, "
            f"and unresolved issues, then coordinate the transition path with the W3C Team. Use the Guidebook "
            f"only for operational practice ({guide_link}).",
            [
                "Identify the current maturity stage and exact target transition.",
                "Check the latest Process transition requirements for that target stage.",
                "Review implementation evidence, horizontal review status, AC Review relevance, dependencies, and open objections before preparing a request.",
                "Coordinate with the Staff Contact or W3C Team before treating the transition as ready.",
            ],
        ),
        "objection": (
            "For a Formal Objection or appeal-related question, stay close to the W3C Process Document "
            f"({process_link}) and avoid informal shortcuts. The safe answer is to identify where the objection "
            "was raised, preserve the objection record, check the Process section governing objections or appeals, "
            "and involve the relevant Staff Contact or W3C Team before advising on outcomes.",
            [
                "Confirm the decision, review, or transition where the objection was raised.",
                "Locate the current Process rules for Formal Objections or appeals.",
                "Keep the objection rationale and any attempted resolution visible in the record.",
                "Escalate high-risk or unresolved cases to the Staff Contact or W3C Team.",
            ],
        ),
        "charter": (
            "For Working Group charter work, use the W3C Process Document as the normative source "
            f"({process_link}) and the Guidebook for practical drafting guidance ({guide_link}). The safe "
            "next step is to verify the group's scope, deliverables, dependencies, review expectations, and "
            "approval path, and track the charter/recharter review in the W3C Strategy repository issue tracker "
            "using the `charter` label before presenting the charter as ready.",
            [
                "Confirm whether this is a new charter, recharter, or charter extension.",
                "Check Process requirements for group creation, scope, deliverables, and approval.",
                "Review dependencies, liaison needs, horizontal review expectations, and patent implications.",
                "Create or locate the W3C Strategy tracking issue with the `charter` label: https://github.com/w3c/strategy/issues?q=label%3Acharter.",
                "Check open and closed Strategy issues, issue dates, and the charter end date; when horizontal review has been requested and required completion labels are present without `*-needs-resolution`, ask the Staff Contact to start the TiLT review path.",
            ],
        ),
        "review": (
            "For review questions, use the Process Document to identify required review points and the Guidebook "
            f"for practical coordination ({guide_link}). The safe next step is to map which reviews apply, "
            "record review requests and responses, and resolve or explicitly track issues before relying on the review as complete.",
            [
                "Identify the maturity stage and which reviews are expected.",
                "Check whether horizontal, wide, AC, or other reviews apply.",
                "Record review requests, responses, unresolved issues, and rationale.",
                "Confirm with the Staff Contact before using review completion as transition evidence.",
            ],
        ),
        "horizontal_review": (
            "For horizontal review, use the Guidebook Wide Review guidance as the operational playbook and the "
            f"W3C Process as the normative source for wide-review requirements ({process_link}). The practical path is "
            "to request the relevant horizontal reviews through their GitHub request repositories, track progress in "
            "the specification repository, respect horizontal review labels, and check tracker boards before transition.",
            [
                "Start from the Guidebook Wide Review / Horizontal Review guidance.",
                "Request APA/a11y, TAG, I18N, Privacy, and Security reviews through the appropriate GitHub request repositories: https://github.com/w3c/a11y-request/issues/new/choose, https://github.com/w3ctag/design-reviews/issues/new/choose, https://github.com/w3c/i18n-request/issues/new/choose, https://github.com/w3cping/privacy-request/issues/new/choose, and https://github.com/w3c/security-request/issues/new/choose.",
                "Track the overall wide-review status in a GitHub meta-issue in the specification repository.",
                "Use `*-tracker` for horizontal attention, but leave `*-needs-resolution` to horizontal groups.",
                "Before transition, check horizontal tracker boards and resolve or coordinate any lingering `*-needs-resolution` issues.",
            ],
        ),
        "patent": (
            "For Patent Policy or IPR questions, do not rely on a chatbot as the final authority. Use the W3C "
            f"Process entry point ({process_link}) to find the applicable policy path and involve the Staff Contact "
            "or W3C Team. The assistant can help organize the checklist, but it should not give legal advice.",
            [
                "Identify the group, deliverable, maturity stage, and relevant patent-policy trigger.",
                "Check the current W3C Process and referenced Patent Policy material.",
                "Avoid treating Guidebook practice notes as legal or policy overrides.",
                "Escalate to the Staff Contact or W3C Team for confirmation.",
            ],
        ),
        "working_draft": (
            "For Working Draft or FPWD questions, use the W3C Process Document as the normative source "
            f"({process_link}). The safe next step is to confirm the group has authority to publish, verify the "
            "document status and publication expectations, and plan the Recommendation-track path forward with the Guidebook only for practical preparation.",
            [
                "Confirm the owning group and whether the publication is FPWD or another Working Draft.",
                "Check the Process requirements for that publication type.",
                "Review status text, issue tracking, dependencies, and publication readiness.",
                "Coordinate with the Staff Contact before publication.",
            ],
        ),
        "staff_contact": (
            "For Staff Contact or Team Contact questions, use the W3C Process for the formal group "
            f"role and the Guidebook for practical coordination ({guide_link}). The Staff Contact should "
            "help the group connect Process requirements, publication or transition preparation, review "
            "coordination, and escalation to the W3C Team when needed.",
            [
                "Identify the group, deliverable, decision, and Process step where Staff Contact support is needed.",
                "Check the cited Process rule for the relevant group or transition responsibility.",
                "Use the Guidebook Staff Contact material to map practical coordination and escalation points.",
                "Record which items need Staff Contact confirmation before treating the workflow step as complete.",
            ],
        ),
        "community_group": (
            "For Community Group incubation moving toward Working Group standardization, treat the Process "
            f"as controlling for Working Group authority and chartering ({process_link}), and use the Guidebook "
            f"for practical transfer preparation ({guide_link}). The key is to confirm scope, contributors, "
            "issue history, Patent Policy implications, and whether the target is a new or existing Working Group.",
            [
                "Identify the Community Group output, maturity, contributors, and issue history.",
                "Decide whether the target path is a new Working Group or an existing Working Group.",
                "Check Working Group charter and Patent Policy implications under the Process.",
                "Coordinate with the W3C Team before treating the incubated work as Recommendation-track ready.",
            ],
        ),
        "member_submission": (
            "For a W3C Member Submission, use the W3C Process as the normative source "
            f"({process_link}). The answer should identify who is submitting, what is being submitted, "
            "whether the submission route is appropriate, and which W3C Team checks or publication steps apply.",
            [
                "Confirm the submitting Member organization and submitted material.",
                "Check the current Process rules for Member Submissions.",
                "Separate Member Submission handling from Recommendation-track advancement.",
                "Coordinate with the W3C Team before publication or public positioning.",
            ],
        ),
        "registry": (
            "For Registry-track questions, use the W3C Process to identify the applicable Registry rules "
            f"({process_link}). The safe next step is to confirm whether the document is actually on the "
            "Registry track, who maintains it, and which publication or update rules apply.",
            [
                "Confirm whether the document is a Registry-track document.",
                "Check the Process rules for Registry publication or updates.",
                "Identify the maintaining group and update procedure.",
                "Coordinate with the Staff Contact for unusual registry or transition questions.",
            ],
        ),
    }
    return answers.get(topic, _english_general_answer(process_link, guide_link))


def _english_general_answer(process_link: str, guide_link: str) -> TopicAnswer:
    return (
        "This appears to be a W3C Process workflow question. Use the latest W3C Process Document "
        f"as the normative source ({process_link}) and the W3C Guidebook as practice guidance "
        f"({guide_link}). The assistant has not yet connected section-level vector retrieval, so it is "
        "returning a conservative checklist instead of a definitive process determination.",
        [
            "Confirm the specification's current maturity stage and owning group.",
            "Check the latest W3C Process section for the target workflow or transition.",
            "Use the Guidebook for operational guidance, but do not treat it as overriding the Process.",
            "Ask the Staff Contact or W3C Team for high-risk items such as Patent Policy, Formal Objection, Appeal, or AC Review.",
        ],
    )


def _chinese_topic_answer(topic: str, process_link: str, guide_link: str) -> TopicAnswer:
    answers: dict[str, TopicAnswer] = {
        "recommendation_transition": (
            f"对于 CR 到 REC 这类 Recommendation-track 转换，应以 W3C Process Document 为规范依据（{process_link}）。安全的下一步是确认当前成熟度阶段、核对目标转换要求、整理实现与审查证据、处理未解决问题，并与 Staff Contact 或 W3C Team 确认转换路径。Guidebook 只能作为实践参考（{guide_link}）。",
            ["确认当前成熟度阶段和目标转换。", "核对最新版 Process 中该转换的要求。", "整理实现证据、审查状态、依赖和未解决异议。", "与 Staff Contact 或 W3C Team 确认是否可以准备转换请求。"],
        ),
        "objection": (
            f"对于 Formal Objection 或 appeal 相关问题，应紧贴 W3C Process Document（{process_link}），不要用非正式经验替代流程。安全的下一步是确认异议发生在哪个决定或转换点，保留异议记录，核对 Process 中关于异议或申诉的规则，并让 Staff Contact 或 W3C Team 参与确认。",
            ["确认异议对应的决定、审查或转换。", "查找最新版 Process 中 Formal Objection 或 appeal 的规则。", "保留异议理由、处理尝试和未解决点的记录。", "高风险或未解决情况交给 Staff Contact 或 W3C Team 确认。"],
        ),
        "charter": (
            f"对于 Working Group charter，应以 W3C Process Document 为规范依据（{process_link}），Guidebook 作为起草实践参考（{guide_link}）。安全的下一步是确认这是新 charter、recharter 还是延期，核对 scope、deliverables、依赖、审查期望和批准路径，并在 W3C Strategy 仓库中用带 `charter` label 的 issue 跟踪整个审阅流程。",
            [
                "确认是新建、recharter 还是 charter extension。",
                "核对 Process 中关于 group、scope、deliverables 和批准的要求。",
                "检查依赖、liaison、horizontal review 和 patent 影响。",
                "在 W3C Strategy 仓库中创建或找到带 `charter` label 的 tracking issue：https://github.com/w3c/strategy/issues?q=label%3Acharter。",
                "检查 open/closed Strategy issues、issue 时间线和 charter end date；当 horizontal review 已请求、所需 review completed labels 已到位且没有 `*-needs-resolution` 阻塞时，请 Staff Contact 启动 TiLT review 路径。",
            ],
        ),
        "review": (
            f"对于 review 问题，应用 Process Document 判断哪些 review 适用，并用 Guidebook 协调实际操作（{guide_link}）。安全的下一步是映射需要的 review，记录请求与反馈，处理或明确跟踪未解决问题，再把 review completion 作为转换证据。",
            ["确认当前成熟度阶段和适用 review。", "判断是否需要 horizontal、wide、AC 或其他 review。", "记录 review 请求、反馈、未解决 issue 和理由。", "使用 review 作为转换依据前先与 Staff Contact 确认。"],
        ),
        "horizontal_review": (
            f"对于 horizontal review，应把 Guidebook 的 Wide Review / Horizontal Review 页面作为操作手册，同时以 W3C Process 中 wide review 要求作为规范依据（{process_link}）。实践路径是通过各 horizontal group 的 GitHub request repo 发起 review，在 specification repo 中用 meta issue 跟踪进展，正确使用 `*-tracker` / `*-needs-resolution` 标签，并在 transition 前检查 horizontal tracker boards。",
            [
                "从 Guidebook 的 Wide Review / Horizontal Review 指南开始规划。",
                "按需要通过 GitHub 请求 APA/a11y、TAG、I18N、Privacy、Security 等 review：https://github.com/w3c/a11y-request/issues/new/choose、https://github.com/w3ctag/design-reviews/issues/new/choose、https://github.com/w3c/i18n-request/issues/new/choose、https://github.com/w3cping/privacy-request/issues/new/choose、https://github.com/w3c/security-request/issues/new/choose。",
                "在 specification repo 里建立 GitHub meta issue，集中跟踪 review 请求、反馈和解决状态。",
                "可以使用 `*-tracker` 提醒 horizontal group，但不要自行添加或移除 `*-needs-resolution`。",
                "transition 前检查 horizontal tracker boards，并与 horizontal group / Staff Contact 处理遗留 `*-needs-resolution` issue。",
            ],
        ),
        "patent": (
            f"对于 Patent Policy 或 IPR 问题，不应把 chatbot 当作最终权威。请从 W3C Process 入口（{process_link}）找到相关政策路径，并联系 Staff Contact 或 W3C Team。系统可以帮你整理清单，但不会提供法律意见。",
            ["确认 group、deliverable、成熟度阶段和相关 patent trigger。", "核对当前 Process 和引用的 Patent Policy 材料。", "不要让 Guidebook 实践建议覆盖正式政策。", "交给 Staff Contact 或 W3C Team 做最终确认。"],
        ),
        "working_draft": (
            f"对于 Working Draft 或 FPWD，应以 W3C Process Document 为规范依据（{process_link}）。安全的下一步是确认工作组有发布权限，核对文档状态和发布要求，并用 Guidebook 作为准备工作的实践参考。",
            ["确认所属 group，以及是 FPWD 还是普通 Working Draft。", "核对 Process 中对应 publication type 的要求。", "检查 status text、issue tracking、依赖和发布准备情况。", "发布前与 Staff Contact 确认。"],
        ),
    }
    return answers.get(topic, _chinese_general_answer(process_link, guide_link))


def _chinese_general_answer(process_link: str, guide_link: str) -> TopicAnswer:
    return (
        f"这看起来是一个 W3C Process 工作流问题。请以最新版 W3C Process Document 作为规范性依据（{process_link}），以 W3C Guidebook 作为实践操作参考（{guide_link}）。当前还没有接入 section-level 向量检索，因此先返回保守清单，而不是给出最终流程判断。",
        ["确认当前成熟度阶段和所属 Working Group。", "在最新版 Process 中核对目标流程或转换要求。", "使用 Guidebook 获取操作建议，但不要让它覆盖 Process 的规范要求。", "涉及 Patent Policy、Formal Objection、Appeal、AC Review 等高风险事项时，联系 Staff Contact 或 W3C Team 确认。"],
    )
