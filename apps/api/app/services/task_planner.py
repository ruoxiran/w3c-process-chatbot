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
    # Announcing publications / Call-for-Review / press releases.
    # Maps to ``w3t-comm@w3.org`` and the public-review-announce /
    # www-announce lists. Distinct from ``advance_specification`` —
    # that's the technical transition step; this is the
    # communications follow-up that goes through the W3C
    # Communications Team, NOT ``w3t-tr@w3.org``.
    "communications_announcement",
    # TPAC / workshops / AC meetings — attending or hosting a W3C
    # event. The operational answers live in the
    # /guide/meetings/ chapters (hosting, workshops, hybrid) and in
    # Process §3.1.1 about General Meetings + §3.2 AC Meetings.
    "attend_or_host_event",
    # W3C membership — how to join, what members get, member-only
    # vs public, member dues. Mostly maps to /guide/members and the
    # Process §2.1 Members chapter.
    "w3c_membership",
    # Group lifecycle endings: closing a WG, suspending a
    # participant, rescinding a published Recommendation, what
    # happens to Notes / repos after group closure. Distinct from
    # ``charter_or_recharter`` (which is about starting / extending)
    # and from ``advance_specification`` (which is about advancing
    # through stages, not ending). Lives in /guide/process/
    # closing-wg-implementation.html + suspension.html + Process
    # §6.x rescind.
    "group_lifecycle",
    # AB / TAG elections, nominations, elected-body composition.
    # Distinct from ``coordinate_with_staff_contact`` (Team Contact
    # roles) and from ``handle_objection_or_appeal`` (FO/appeal
    # mechanics). Has ~105 chunks of corpus grounding —
    # /guide/process/election.html + /guide/process/elections.html +
    # /other/elected-body-communication-guidelines.html.
    "elected_body",
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
    # Group-lifecycle endings — close a WG, suspend a participant,
    # rescind a Recommendation, what happens to a Note after closure.
    # Placed early so it catches BEFORE charter / advance / review
    # rules grab keywords like "Working Group" or "Recommendation".
    if _has(text, [
        "close a working group", "close the working group",
        "close a wg", "close the wg", "close the group",
        "closing a working group", "closing a wg",
        "group closure", "wind down",
        "terminate a group", "terminate the group",
        "rescind", "rescinding", "rescinded",
        "obsolete recommendation", "supersede a recommendation",
        "obsolete a rec",
        # Rollback / revoke a published REC — same operational
        # category as rescind.
        "rollback a recommendation", "rollback the recommendation",
        "revoke a recommendation", "revoke the recommendation",
        "suspend a participant", "suspend participant",
        "participant suspension", "suspension of a participant",
        # Looser end-of-life phrasings — "after the group closes",
        # "after the WG closes", "the group closes", etc.
        "group closes", "wg closes",
        "after the group closes", "after a group closes",
        "after the wg closes",
        "what happens to a note", "what happens to a rec",
        "what happens to the spec when",
        "关闭工作组", "终止", "暂停",
    ]):
        return "group_lifecycle"
    # AB / TAG elections + nominations + elected-body composition.
    # Placed before ``advance_specification`` / staff_contact so
    # questions like "how do I vote in an AB election?" don't get
    # caught as a transition or a Team-Contact question.
    if _has(text, [
        "ab election", "ab elections", "tag election", "tag elections",
        "advisory board election", "advisory board nominat",
        "tag nominat", "nominate to the tag", "nominate to the ab",
        # Keep loose phrasings — "nominate someone to the TAG" should
        # work even though the longer literal "nominate to the tag"
        # doesn't appear due to the intervening "someone".
        "nominate to tag", "nominate for tag", "nominate to ab",
        "nominate for ab", "nominee to the tag", "nominee to the ab",
        # "nominate someone to the TAG" / "nominate a person to AB"
        # — short keywords so any intervening pronoun works.
        "nominate", "nominee", "nominated", "nomination",
        "elected body", "elected bodies",
        "ab member", "ab members", "tag member", "tag members",
        "advisory board",
        "technical architecture group",
        "who's on the advisory board", "who is on the advisory board",
        "ab seat", "tag seat",
        "nomination period", "nomination procedure",
        "ab/tag", "ab and tag",
        "选举", "提名",
    ]):
        return "elected_body"
    if _has(text, [
        "patent", "ipr", "exclusion opportunity", "专利",
        # Employer / contribution / IPR commitment questions are
        # Patent Policy territory — they're about the legal basis
        # of contributing to a spec, not about authoring tools.
        "employer veto", "employer approval", "veto my contribution",
        "ipr commitment", "contribution license",
        "non-participant commitment",
    ]):
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
    # Stage-shorthand + "next" / "next step" / "what's next" — without
    # an explicit ``transition`` verb but clearly about advancing
    # through stages. e.g. "from CR to REC, what's next?",
    # "I'm at CR, what's the next milestone?". Distinct from the
    # earlier branch which required the transition verb.
    if has_stage_term and _has(text, [
        "next", "next step", "what's next", "what is next",
        "milestone", "下一步", "下一关",
    ]):
        return "advance_specification"
    # Communications / announcement / press release / Call-for-Review
    # / AC review mechanics / mailing-list subscription. Goes BEFORE
    # the broad "review" keyword check below because "Call for
    # Review" / "AC review" otherwise match via the bare ``review``
    # substring and route to plan_or_complete_review — which feeds
    # the model the wrong action surface (``w3t-tr@w3.org`` instead
    # of ``w3t-comm@w3.org``).
    if _has(text, [
        "announce", "announcement", "announcing",
        "press release", "press contact",
        "publicize", "publicise",
        "communications team", "comm team", "w3c communications",
        "w3t-comm", "www-announce", "public-review-announce",
        "blog post", "blog about", "social media post",
        "call for review", "cfr ",
        # AC procedural mechanics — voting, decisions, approval.
        # These are coordinated by the Communications Team via the
        # WBS survey system; same action-surface set applies.
        "ac vote", "ac votes", "advisory committee vote",
        "ac decision", "advisory committee decision",
        "ac approval", "advisory committee approval",
        "vote on a recommendation", "vote on a rec",
        # Mailing-list subscription / management. The Communications
        # Team owns list creation + subscription policy. Keep these
        # short — "subscribe to a W3C mailing list" must match even
        # with an intervening "W3C" / "public" / "member" qualifier.
        "mailing list",
        "mailing-list",
        "subscribe to a", "subscribe to the",
        "subscribe to public-", "subscribe to member-",
        "邮件列表", "订阅",
        "通讯团队", "公告", "宣布发布",
    ]):
        return "communications_announcement"
    if _has(text, ["wide review", "review", "审查"]):
        return "plan_or_complete_review"
    if _has(text, ["staff contact", "team contact", "liaison", "职责"]):
        return "coordinate_with_staff_contact"
    # TPAC / workshops / AC face-to-face. Goes BEFORE the generic
    # meeting / chair check so "how to host a workshop" doesn't get
    # caught as a plain group-meeting question and lose its event-
    # specific action surfaces (workshop chapter, TPAC homepage,
    # breakout proposal form).
    if _has(text, [
        "tpac", "annual technical plenary", "annual meeting",
        "workshop", "workshops", "research workshop",
        "host an event", "host a meeting", "host a workshop",
        "breakout", "breakouts", "breakout session",
        "f2f", "face-to-face", "face to face meeting",
        "register for tpac", "tpac registration",
        "hybrid meeting", "hybrid event",
        # AC plenary as an event (logistical). The procedural side
        # — "how does the AC vote / decide" — is intercepted EARLIER
        # by communications_announcement.
        "advisory committee meeting", "ac meeting",
        "研讨会", "线下会议", "面对面会议",
    ]):
        return "attend_or_host_event"
    # W3C membership — join, dues, benefits, member-only access.
    # Goes BEFORE meeting / chair to avoid "member representative
    # meeting" collisions catching this as run_group_process.
    if _has(text, [
        "become a member", "become a w3c member", "join w3c",
        "join the w3c", "w3c membership", "member dues",
        "member fees", "member benefits", "members benefits",
        "member-only", "member only", "membership agreement",
        "how do i join", "what does w3c membership", "w3c funding",
        "how is w3c funded", "w3c finance",
        "成为成员", "会员", "会费",
    ]):
        return "w3c_membership"
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
        # Group governance / behavior — Code of Conduct + Antitrust
        # Policy live as action surfaces under this intent (round 24).
        # Without these keywords "what does the W3C Code of Conduct
        # say" / "antitrust rules for cross-company participation"
        # fell through to advance_specification and missed the
        # policy chunks entirely.
        "code of conduct", "coc ", "positive work environment",
        "antitrust", "competition policy", "cross-company",
        "行为准则", "反垄断",
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
        # Adding / removing / registering editors on a spec.
        # Without these, "how to add a new editor?" fell through to
        # advance_specification because it has no transition keyword
        # but DOES eventually match generic "spec" terms.
        "add an editor", "add a new editor", "remove an editor",
        "new editor", "additional editor", "register as a spec editor",
        "register as an editor", "register an editor",
        # SOTD content rules are an editor concern — the editor
        # writes the SOTD section in the spec source.
        "sotd", "status of this document", "sotd section",
        "sotd boilerplate",
        # Repo / publication tooling
        "repo manager", "repo-manager",
        "auto-publication", "auto-publish", "automated publication",
        # WBS tooling — note: "call for review" is intentionally NOT
        # here. CfR is a communications/announcement step that goes
        # through w3t-comm@w3.org, not a spec-authoring tool. Moved
        # to ``communications_announcement`` below.
        "wbs",
        "编辑器", "撰写规范", "编写规范",
    ]):
        return "author_spec"
    # ``communications_announcement`` is handled earlier (before the
    # broad ``review`` keyword) so a "Call for Review" question gets
    # the comms intent, not plan_or_complete_review.
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
            # Living-CR-REC / post-publication maintenance — adding
            # features after CR, maintaining a published REC. These
            # are advance-spec questions, not group-lifecycle, even
            # though they happen post-publication.
            "candidate additions", "candidate amendments",
            "adding features after", "feature after",
            "maintenance update", "rec maintenance",
            "post-rec maintenance", "post-publication maintenance",
            "existing recommendation", "published recommendation",
            # Bare 2-letter "cr" / "rec" REMOVED in round 35 — they
            # substring-match harmless words ("di-REC-tor" /
            # "REC-ommend" / "REC-ipe", "con-CR-ete" / "se-CR-et").
            # Real stage-transition questions are still caught
            # earlier by the ``has_transition_verb`` +
            # ``has_stage_term`` check, which uses specific phrases
            # like "to rec" / "in cr" / "candidate recommendation".
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
        # Branch by sub-topic — run_group_process covers three quite
        # different sub-domains: meetings/scribing, group governance
        # (Code of Conduct), and cross-company antitrust. Meeting
        # seeds bias retrieval toward meeting pages, which is wrong
        # when the user asks about antitrust or CoC. Emit only the
        # seeds that match the question's actual sub-topic.
        lower_msg = message.lower()
        if "antitrust" in lower_msg or "competition policy" in lower_msg or "反垄断" in lower_msg:
            seed.extend([
                "W3C antitrust policy participants cross-company",
                "antitrust competition rules group participation",
            ])
        elif "code of conduct" in lower_msg or "positive work environment" in lower_msg or "行为准则" in lower_msg:
            seed.extend([
                "W3C Code of Conduct positive work environment unacceptable behavior",
                "Code of Conduct reporting harassment ethical guidelines",
            ])
        else:
            seed.extend([
                "W3C group meeting process chair minutes decision",
                "Guidebook chair meetings agenda minutes",
                # Tool-specific seeds so retrieval lands on the
                # dedicated zakim.html / rrsagent.html / scribe.html
                # guide pages when the user asks "how to scribe?",
                # "what is Zakim?", "how do I queue with rrsagent?".
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
    elif intent_type == "communications_announcement":
        # Announcement / press / Call-for-Review path. The canonical
        # contact is ``w3t-comm@w3.org`` (NOT the transitions list
        # ``w3t-tr@w3.org``). Retrieval seeds steer BM25 toward the
        # Guidebook charter chapter (which has the most Comms-Team
        # interaction text), the Process publication-and-communication
        # section, and the "Speaking about your work" landing page.
        seed.extend([
            "W3C Communications Team w3t-comm@w3.org announce publication AC",
            "Call for Review AC review announcement Comms Team",
            "Guidebook Speaking about your work press blog announcement",
            "Process publication communication dissemination press release",
            "public-review-announce mailing list publication announcement",
        ])
    elif intent_type == "attend_or_host_event":
        # TPAC / workshops / AC face-to-face. Steer BM25 at the
        # /guide/meetings/ chapters (hosting.html, workshops.html,
        # hybrid-meeting.html) and Process §3.1.1 General Meetings
        # + §3.2 AC Meetings.
        seed.extend([
            "Guidebook host workshop W3C event organize",
            "TPAC annual technical plenary breakout schedule registration",
            "Guidebook hosting face-to-face meeting venue logistics",
            "Process General Meetings AC meeting Advisory Committee",
            "hybrid meeting remote participation continuity",
        ])
    elif intent_type == "w3c_membership":
        # Membership — Process §2.1 Members chapter and the
        # /guide/ landing's membership-related sections.
        seed.extend([
            "W3C membership join member dues benefits agreement",
            "Process Members Member Agreement Patent Policy commitment",
            "Guidebook becoming a W3C member application",
            "member-only access invited expert participation",
        ])
    elif intent_type == "group_lifecycle":
        # Closing a WG / suspending a participant / rescinding a
        # Recommendation. Steer at /guide/process/
        # closing-wg-implementation.html + Process §3.x suspension
        # + Process §6.7 obsolete-rescinded-superseded.
        seed.extend([
            "Guidebook closing Working Group implementation wind down",
            "Process suspension participant suspend conformance",
            "Process rescind obsolete superseded Recommendation",
            "Guide closing-wg what happens to spec note after closure",
            f"{subject} group closure rescind",
        ])
    elif intent_type == "elected_body":
        # AB / TAG elections + nominations.
        seed.extend([
            "Guidebook AB TAG election nomination procedure timeline",
            "Process Advisory Board Technical Architecture Group composition",
            "elected body communication guidelines AB TAG",
            f"{subject} AB TAG election",
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
        # Security/Privacy Questionnaire — the concrete checklist
        # spec authors fill in for privacy/security review. Indexed
        # in round 27. Routes through horizontal_review because the
        # questionnaire IS a horizontal-review artifact.
        "security and privacy questionnaire",
        "security/privacy questionnaire",
        "privacy questionnaire",
        "self-review questionnaire",
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
