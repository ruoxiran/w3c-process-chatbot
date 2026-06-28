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


_GUIDE_TOPICS_BY_ID = {topic.id: topic for topic in GUIDE_TOPICS}


def is_topic_match(query: str, topic_id: str) -> bool:
    """True iff ``query`` matches at least one query needle of the named topic."""
    text = query.lower()
    topic = _GUIDE_TOPICS_BY_ID.get(topic_id)
    if topic is None:
        return False
    return any(needle in text for needle in topic.query_needles)


# ---- Process ↔ Guidebook pairing -----------------------------------------
#
# Process is the normative rule ("what is required"). Guidebook is the
# operational manual ("how to do it"). They are written and maintained
# as complementary documents — the Guidebook chapters explicitly link
# to the Process sections they operationalise. A good answer to "how
# do I X" cites BOTH sides: the rule for authority, the operational
# chapter for the steps.
#
# This table was extracted from the corpus by mining inline
# ``[link](https://www.w3.org/policies/process/#X)`` references from
# every Guidebook chunk back to its Process anchor. It's regenerated
# manually when the Process or Guidebook structure changes; see
# scripts/mine_process_guide_pairs.py in git history for the script.
#
# Key  = Process Document section_id (the anchor after ``#``)
# Value = guide-topic keys (path under /guide/ or under
#         /github/w3c/guide/blob/<sha>/, normalised — e.g.
#         ``documentreview/index``, ``process/charter``,
#         ``meetings/hybrid-meeting``)
PROCESS_GUIDE_PAIRS: dict[str, tuple[str, ...]] = {
    'AB': ('other/elected-body-communication-guidelines',),
    'AB-TAG-participation': ('other/elected-body-communication-guidelines',),
    'CharterReview': ('process/charter', 'process/predicting-milestones'),
    'Consensus': ('chair/role',),
    'FormalObjection': ('council/index',),
    'GAEvents': ('meetings/continuity',),
    'GAGeneral': ('teamcontact/role',),
    'GeneralMeetings': ('meetings/organize', 'process/predicting-milestones'),
    'GeneralTermination': ('process/closing-wg-implementation',),
    'GroupsIG': ('teamcontact/role',),
    'GroupsWG': ('teamcontact/role',),
    'Liaisons': ('index',),
    'MemberSubmission': ('process/member-submission',),
    'ParticipationCriteria': ('process/suspension',),
    'Reports': ('transitions/details', 'transitions/wide-review-request', 'process/tilt/normative-references'),
    'ReqsAllGroups': ('process/charter', 'chair/index', 'teamcontact/index'),
    'Submission': ('process/member-submission', 'incubation', 'standards-track/rec-tips'),
    'TAG': ('other/elected-body-communication-guidelines',),
    'TAG-appointments': ('other/elected-body-communication-guidelines',),
    'Team': ('process/tilt/index',),
    'Team-only': ('process/member-submission',),
    'WGCharter': ('process/charter',),
    'WGCharterDevelopment': ('chair/role', 'process/predicting-milestones'),
    'addressing-fo': ('process/charter',),
    'advisory-board': ('council/index',),
    'advisory-committee-appeal': ('process/ac-appeal',),
    'advisory-committee-review': ('process/charter',),
    'appeal-vote': ('process/charter',),
    'candidate-amendments': ('process/living-cr-rec',),
    'candidate-rec': ('process/predicting-milestones', 'github/issue-metadata'),
    'cfp': ('standards-track/index',),
    'change-review': ('documentreview/index',),
    'charter-development': ('process/charter', 'process/adv-notice'),
    'charter-extension': ('process/charter',),
    'charter-initiation': ('process/adv-notice', 'process/charter'),
    'contributor-license': ('process/non-participant-commitment',),
    'council': ('council/index',),
    'council-chairing': ('council/council',),
    'council-composition': ('council/council-steps',),
    'council-decision': ('council/council',),
    'council-delegation': ('council/council-steps', 'council/council'),
    'council-deliberations': ('council/council',),
    'council-participation': ('council/council',),
    'council-short-circuit': ('council/council-steps', 'council/council'),
    'decision-types': ('standards-track/index',),
    'def-w3c-decision': ('council/council', 'standards-track/index', 'process/closing-wg'),
    'distributed-meeting': ('meetings/organize',),
    'editorial-change': ('process/member-submission',),
    'errata': ('process/predicting-milestones',),
    'first-wd': ('documentreview/index', 'github/issue-metadata'),
    'formally-addressed': ('documentreview/index',),
    'general-requirements': ('chair/role',),
    'group-decision': ('council/council',),
    'group-lifecyle': ('process/charter', 'process/closing-wg'),
    'implementation-experience': ('process/living-cr-rec', 'process/tilt/normative-references'),
    'initiating-rec-review': ('process/predicting-milestones',),
    'maturity-stages': ('standards-track/index',),
    'meeting-schedules': ('tools/new-group',),
    'note-track': ('github/w3c.json',),
    'publication': ('editor/index',),
    'publishing': ('documentreview/index',),
    'rec-publication': ('github/issue-metadata',),
    'rec-rescind': ('process/obsolete-rescinded-supserseded',),
    'rec-track': ('standards-track/rec-tips', 'github/w3c.json', 'process/living-cr-rec'),
    'rec-track-regression': ('process/living-cr-rec',),
    'ref-for-team': ('chair/role',),
    'registering-objections': ('council/council',),
    'registries': ('github/w3c.json',),
    'requirements-and-definitions': ('transitions/details',),
    'submitter': ('process/member-submission',),
    'substantive-change': ('github/repo-management', 'editor/versioning'),
    'team': ('process/member-submission', 'council/index', 'chair/index', 'teamcontact/index'),
    'team-decision': ('process/charter',),
    'team-fo-mediation': ('council/council',),
    'technical-architecture-group': ('council/index',),
    'tooling': ('meetings/hybrid-meeting', 'tools/index'),
    'transition-cr': ('documentreview/index',),
    'update-reqs': ('process/living-cr-rec',),
    'update-requests': ('documentreview/index',),
    'w3c-council': ('process/charter',),
    'wide-review': ('documentreview/index', 'process/horizontal-groups', 'process/living-cr-rec'),
}


def paired_guide_topics(section_id: str | None) -> tuple[str, ...]:
    """Return the Guidebook topic keys that operationalise this Process section.

    Returns an empty tuple when the section isn't in the table — caller
    should treat that as "no specific operational chapter is paired";
    don't fabricate one.
    """
    if not section_id:
        return ()
    return PROCESS_GUIDE_PAIRS.get(section_id, ())


# ---- Scoring rules --------------------------------------------------------
#
# The retriever needs to nudge results toward authoritative sections for
# specific query patterns ("horizontal review questions → the document-
# review chapter", "CR/REC transition queries → the transition guide",
# ...). The rules used to live as ~140 lines of ``if X in text and Y in
# heading: score += N`` inside ``retriever._topic_bonus`` /
# ``retriever._relevance_adjustment``. That mixed two concerns — domain
# knowledge about W3C topic taxonomy and the scoring algorithm — into
# one file.
#
# Rules now live here as data; the retriever holds only the generic
# scorer. Adding a new boost is one row in ``SCORING_RULES``, not a
# patch to the ranker.


@dataclass(frozen=True)
class ScoringRule:
    """One declarative rule contributing a score adjustment.

    Every needle list is OR semantics — if at least one of the
    candidates appears in the relevant field the predicate passes.
    Between fields the semantics is AND — the rule fires only when
    every populated check passes. Empty tuples are no-ops.

    Negative ``score`` values are penalties (used by the relevance
    adjustment table to demote stale snapshots, off-topic namespace
    chunks, etc.).
    """

    id: str
    score: int
    query_any: tuple[str, ...] = ()
    query_topic_id: str | None = None  # alternative to query_any
    query_none: tuple[str, ...] = ()
    title_any: tuple[str, ...] = ()
    heading_any: tuple[str, ...] = ()
    heading_none: tuple[str, ...] = ()
    body_any: tuple[str, ...] = ()
    combined_any: tuple[str, ...] = ()  # matches against heading + " " + body
    url_any: tuple[str, ...] = ()
    url_none: tuple[str, ...] = ()
    source_type_any: tuple[str, ...] = ()


def _any_match(needles: tuple[str, ...], haystack: str) -> bool:
    """OR-of-substrings, with the empty needle tuple meaning "no constraint"."""
    if not needles:
        return True
    return any(needle in haystack for needle in needles)


def _none_match(needles: tuple[str, ...], haystack: str) -> bool:
    if not needles:
        return True
    return not any(needle in haystack for needle in needles)


def apply_scoring_rules(
    rules: tuple[ScoringRule, ...],
    *,
    query: str,
    title: str,
    heading: str,
    body: str,
    url: str = "",
    source_type: str = "",
) -> int:
    """Sum the scores of every rule whose every constraint passes.

    The retriever calls this twice per candidate (once for topic
    bonuses, once for relevance adjustments) — both tables share the
    same predicate shape and pay one normalisation up front.
    """
    text = query.lower()
    title_l = title.lower()
    heading_l = heading.lower()
    body_l = body.lower()
    combined = f"{heading_l} {body_l}"
    url_l = url.lower()

    total = 0
    for rule in rules:
        if rule.query_topic_id is not None and not is_topic_match(text, rule.query_topic_id):
            continue
        if not _any_match(rule.query_any, text):
            continue
        if not _none_match(rule.query_none, text):
            continue
        if not _any_match(rule.title_any, title_l):
            continue
        if not _any_match(rule.heading_any, heading_l):
            continue
        if not _none_match(rule.heading_none, heading_l):
            continue
        if not _any_match(rule.body_any, body_l):
            continue
        if not _any_match(rule.combined_any, combined):
            continue
        if not _any_match(rule.url_any, url_l):
            continue
        if not _none_match(rule.url_none, url_l):
            continue
        if rule.source_type_any and source_type not in rule.source_type_any:
            continue
        total += rule.score
    return total


# Topic-bonus rules — boost the retrieval score when the query is
# asking about a specific area AND the candidate's heading/body
# matches the relevant W3C section. Ordering does not matter (scores
# are summed). Comments tag the original ``_topic_bonus`` line for
# easier diff review during the extraction commit.
TOPIC_BONUS_RULES: tuple[ScoringRule, ...] = (
    ScoringRule(id="formal_objection_heading", score=10,
        query_any=("formal objection",), heading_any=("formal objection",)),
    # Smaller boost when the term appears in the body but NOT the heading.
    # Original code used ``elif`` so heading+body matches did NOT also
    # collect the +5 — we preserve that via heading_none.
    ScoringRule(id="formal_objection_body_only", score=5,
        query_any=("formal objection",),
        body_any=("formal objection",),
        heading_none=("formal objection",)),
    ScoringRule(id="rec_track_transition_heading", score=12,
        query_any=("cr", "rec"),
        heading_any=("transitioning to recommendation",)),
    ScoringRule(id="rec_track_advancing", score=12,
        query_any=(
            "cr", "rec", "candidate recommendation", "recommendation",
            "候选推荐", "推荐标准", "推进",
        ),
        heading_any=("advancing on the recommendation track",)),
    ScoringRule(id="charter_review_heading", score=12,
        query_any=("charter",),
        heading_any=("charter review and approval",)),
    ScoringRule(id="charter_starting_group", score=10,
        query_any=("charter", "章程"),
        heading_any=("starting a group",)),
    ScoringRule(id="working_group_heading", score=5,
        query_any=("working group", "工作组"),
        heading_any=("groups",)),
    ScoringRule(id="patent_heading", score=8,
        query_any=("patent",),
        heading_any=("patent",)),
    ScoringRule(id="wide_review_heading", score=8,
        query_any=("wide review",),
        heading_any=("wide review",)),
    # Horizontal review cluster — the highest-value query pattern.
    ScoringRule(id="hr_reviews_responsibilities", score=34,
        query_topic_id="horizontal_review",
        heading_any=("reviews and review responsibilities",)),
    ScoringRule(id="hr_doc_reviews_anchor", score=22,
        query_topic_id="horizontal_review",
        combined_any=("#doc-reviews",)),
    ScoringRule(id="hr_how_to_get", score=30,
        query_topic_id="horizontal_review",
        heading_any=("how to get horizontal review",)),
    ScoringRule(id="hr_working_with_labels", score=28,
        query_topic_id="horizontal_review",
        heading_any=("working with horizontal review labels",)),
    ScoringRule(id="hr_needs_resolution", score=22,
        query_topic_id="horizontal_review",
        combined_any=("needs-resolution",)),
    ScoringRule(id="hr_issue_trackers", score=18,
        query_topic_id="horizontal_review",
        combined_any=("issue trackers", "tracker boards")),
    ScoringRule(id="hr_horizontal_groups", score=18,
        query_topic_id="horizontal_review",
        heading_any=("horizontal groups",)),
    ScoringRule(id="hr_labels_metadata", score=18,
        query_topic_id="horizontal_review",
        title_any=("labels and other metadata",),
        heading_any=("horizontal reviews",)),
    ScoringRule(id="hr_transition_horizontal", score=14,
        query_topic_id="horizontal_review",
        title_any=("organize a technical report transition",),
        body_any=("horizontal",)),
    # Workshops — surface the dedicated chapter ahead of generic
    # meeting / hosting pages that share the parent /guide/meetings.
    ScoringRule(id="workshop_pages", score=18,
        query_any=("workshop", "workshops", "研讨会"),
        combined_any=("workshop", "workshops.html")),
    ScoringRule(id="formal_objection_zh", score=6,
        query_any=("formal objection", "异议"),
        combined_any=("formal objection",)),
    ScoringRule(id="appeal", score=8,
        query_any=("appeal", "申诉"),
        combined_any=("appeal",)),
    ScoringRule(id="recharter", score=10,
        query_any=("recharter", "rechartering"),
        combined_any=("charter", "rechartering")),
    ScoringRule(id="fpwd", score=10,
        query_any=("fpwd", "first public working draft"),
        combined_any=("first public working draft", "fpwd")),
    ScoringRule(id="ac_review", score=10,
        query_any=("ac review", "advisory committee"),
        combined_any=("advisory committee", "ac review")),
    ScoringRule(id="next_step_finder", score=8,
        query_any=("next step", "下一步"),
        combined_any=("next step finder",)),
)


# Relevance-adjustment rules — same shape, but allowed to subtract
# (negative scores) to demote stale snapshots and off-topic chunks.
RELEVANCE_RULES: tuple[ScoringRule, ...] = (
    ScoringRule(id="process_canonical_url", score=10,
        source_type_any=("process",),
        url_any=("w3.org/policies/process/",)),
    ScoringRule(id="process_github_snapshot_penalty", score=-10,
        source_type_any=("process",),
        url_any=("github.com/w3c/process", "/snapshots/")),
    ScoringRule(id="snapshot_no_intent_penalty", score=-8,
        url_any=("/snapshots/",), query_none=("snapshot",)),
    # ---- transition cluster (shared query needles) ----
    ScoringRule(id="transition_heading_boost", score=18,
        query_any=(
            "cr", "candidate recommendation", "rec", "recommendation",
            "transition", "advance", "推进", "候选推荐", "推荐标准",
        ),
        heading_any=("transitioning to recommendation", "advancing on the recommendation track")),
    ScoringRule(id="transition_process_heading_boost", score=18,
        query_any=(
            "cr", "candidate recommendation", "rec", "recommendation",
            "transition", "advance", "推进", "候选推荐", "推荐标准",
        ),
        source_type_any=("process",),
        heading_any=("transitioning to recommendation",)),
    ScoringRule(id="transition_guide_url_boost", score=14,
        query_any=(
            "cr", "candidate recommendation", "rec", "recommendation",
            "transition", "advance", "推进", "候选推荐", "推荐标准",
        ),
        title_any=("organize a technical report transition",)),
    ScoringRule(id="transition_guide_url_boost_alt", score=14,
        query_any=(
            "cr", "candidate recommendation", "rec", "recommendation",
            "transition", "advance", "推进", "候选推荐", "推荐标准",
        ),
        url_any=("/guide/transitions",)),
    ScoringRule(id="namespace_off_topic_penalty", score=-14,
        query_any=(
            "cr", "candidate recommendation", "rec", "recommendation",
            "transition", "advance", "推进", "候选推荐", "推荐标准",
        ),
        query_none=("namespace",),
        body_any=("namespace",)),
    ScoringRule(id="comment_invited_penalty", score=-12,
        query_any=(
            "cr", "candidate recommendation", "rec", "recommendation",
            "transition", "advance", "推进", "候选推荐", "推荐标准",
        ),
        body_any=("comment is invited on the draft",)),
    # ---- staff/team contact cluster ----
    ScoringRule(id="staff_contact_heading", score=24,
        query_any=("staff contact", "team contact"),
        heading_any=("staff contacts",)),
    ScoringRule(id="staff_contact_url", score=24,
        query_any=("staff contact", "team contact"),
        url_any=("teamcontact",)),
    ScoringRule(id="staff_contact_body", score=14,
        query_any=("staff contact", "team contact"),
        body_any=("staff contact", "team contact")),
    ScoringRule(id="staff_contact_chair_role_penalty", score=-8,
        query_any=("staff contact", "team contact"),
        source_type_any=("guide",),
        url_any=("chair/role",),
        # Don't penalise chair/role pages that DO discuss staff contacts.
        heading_none=("staff contact",)),
    # ---- meetings ----
    ScoringRule(id="meeting_pages", score=12,
        query_any=("meeting", "chair", "会议"),
        url_any=("chair/meetings",)),
    ScoringRule(id="meeting_pages_heading", score=12,
        query_any=("meeting", "chair", "会议"),
        heading_any=("meeting",)),
    # ---- horizontal review cluster ----
    ScoringRule(id="hr_doc_reviews_process", score=42,
        query_topic_id="horizontal_review",
        source_type_any=("process",),
        url_any=("#doc-reviews",)),
    ScoringRule(id="hr_documentreview_url", score=34,
        query_topic_id="horizontal_review",
        url_any=("/guide/documentreview",)),
    ScoringRule(id="hr_horizontal_groups_url", score=26,
        query_topic_id="horizontal_review",
        url_any=("/guide/process/horizontal-groups",)),
    ScoringRule(id="hr_issue_metadata_url", score=24,
        query_topic_id="horizontal_review",
        url_any=("/guide/github/issue-metadata",)),
    ScoringRule(id="hr_transitions_with_horizontal", score=18,
        query_topic_id="horizontal_review",
        url_any=("/guide/transitions",),
        body_any=("needs-resolution", "horizontal")),
    ScoringRule(id="hr_guide_other_penalty", score=-6,
        query_topic_id="horizontal_review",
        url_any=("github.com/w3c/guide",),
        url_none=("documentreview", "horizontal-groups")),
    # ---- meeting-tooling / scribing cluster -------------------------------
    # "how to scribe" questions need to land on the workflow entry
    # points (join IRC → invite bots → scribe.perl conventions) not
    # on niche bot features. The lexical "scribe" frequency naturally
    # pulls the ``#pickvictim`` chunk to the top because that section
    # repeats the word — these rules counterweight that.
    ScoringRule(id="scribe_irc_chapter", score=32,
        query_any=("scribe", "scribing", "zakim", "rrsagent", "irc"),
        url_any=("/guide/meetings/irc.html",)),
    ScoringRule(id="scribe_zakim_entry_sections", score=28,
        query_any=("scribe", "scribing", "zakim"),
        url_any=(
            "/guide/meetings/zakim.html#invite",
            "/guide/meetings/zakim.html#startmeeting",
            "/guide/meetings/zakim.html#general",
            "/guide/meetings/zakim.html#addressing",
        )),
    ScoringRule(id="scribe_rrsagent_chapter", score=26,
        query_any=("scribe", "scribing", "rrsagent", "minutes log"),
        url_any=("/guide/meetings/rrsagent.html",)),
    # The Zakim "pick a scribe" feature is technically a scribing
    # topic but it's an ADVANCED helper, not where a new scribe
    # should start. Demote it strongly for general "how to scribe"
    # queries so the entry-point chunks above out-rank it.
    ScoringRule(id="scribe_pickvictim_penalty", score=-30,
        query_any=("how to scribe", "how do i scribe", "how to take minutes",
                   "how do i take minutes", "how to do scribe"),
        url_any=("#pickvictim",)),
    # Prefer the published w3.org HTML chapter over its github.com
    # markdown twin — citations should send the user to the rendered
    # page with the section anchor, not the raw .md file.
    ScoringRule(id="meetings_guide_html_preferred", score=10,
        query_any=("scribe", "scribing", "zakim", "rrsagent", "irc", "minutes log"),
        url_any=("www.w3.org/guide/meetings/",)),
    ScoringRule(id="meetings_guide_md_penalty", score=-10,
        query_any=("scribe", "scribing", "zakim", "rrsagent", "irc", "minutes log"),
        url_any=("github.com/w3c/guide/blob/", "meetings/zakim.md",
                 "meetings/rrsagent.md", "meetings/irc.md")),
)
