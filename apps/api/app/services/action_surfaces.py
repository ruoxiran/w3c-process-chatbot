"""Concrete W3C "do this here" surfaces that turn answers into instructions.

The retrieval layer surfaces *what the rule is*. This module surfaces
*where to act on it*: the actual issue tracker to file in, the mailing
list to email, the form to submit, the role to contact. The workflow
injects the action surfaces matching the question's intent into the
prompt, and the rule layer tells the model to end each step with a
concrete action — a URL, a mailto, or a named tracker.

These are deliberately stable, well-known surfaces. They are not the
authority — Process and Guidebook citations remain that. They are the
operational handles a W3C contributor needs to actually move work
forward.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ActionSurface:
    label: str
    """Short human label used in the prompt (e.g. "File a transition request")."""

    url: str | None = None
    """Best concrete URL to act on — a form, tracker page, or guide section."""

    email: str | None = None
    """Mailing list or role address, when emailing is the action."""

    repo: str | None = None
    """GitHub repo to file an issue / open a PR (``owner/repo`` shorthand)."""

    notes: str | None = None
    """One short sentence — when this surface is the right one, any caveats."""


# Action surfaces grouped by ``intent_type`` (TaskPlan field). Each intent
# may carry multiple surfaces; the workflow picks them all and lets the
# model choose the one(s) actually relevant to the user's situation.
ACTION_SURFACES: dict[str, list[ActionSurface]] = {
    "advance_specification": [
        ActionSurface(
            label="Request a transition (Guidebook entry point)",
            url="https://www.w3.org/guide/transitions/",
            notes="Lists the prerequisites and links to the transition-request form.",
        ),
        ActionSurface(
            label="Submit the transition-request form (W3C Webmaster sysreq)",
            url="https://www.w3.org/Webmaster/Group/transition.html",
            notes="The actual form that asks the Director to evaluate a Recommendation-track transition.",
        ),
        ActionSurface(
            label="Email the W3C Transitions Team",
            email="w3t-tr@w3.org",
            notes=(
                "INTERNAL list for transition-meeting scheduling and "
                "clarifying questions about an in-flight transition "
                "request. NOT for AC announcements / press releases / "
                "publication notices — those go through w3t-comm@w3.org "
                "under the communications_announcement intent."
            ),
        ),
        ActionSurface(
            label="W3C Process — Recommendation Track",
            url="https://www.w3.org/policies/process/#recs-and-notes",
        ),
        ActionSurface(
            label="Implementation report template",
            url="https://www.w3.org/guide/transitions/implementation-report.html",
            notes="Use when documenting two-implementer evidence ahead of CR → PR transition.",
        ),
        # Publication tooling — what an editor actually clicks/runs
        # when the rule says "publish a new draft / snapshot".
        ActionSurface(
            label="W3C Pubrules — pre-publication validator",
            url="https://www.w3.org/pubrules/",
            notes="Run before requesting publication; surfaces SOTD / SoTD / boilerplate / Process-conformance issues.",
        ),
        ActionSurface(
            label="Echidna — automated publication",
            url="https://github.com/w3c/echidna/wiki",
            notes="Lets a WG auto-publish a WD or CR Snapshot from a repo without a manual Team handover. Requires a configured ``w3c.json`` + green pubrules.",
        ),
        ActionSurface(
            label="HTMLdiff — diff two spec versions",
            url="https://services.w3.org/htmldiff",
            notes="Use to produce the change-summary required for CR / PR transitions of an existing REC.",
        ),
    ],
    "horizontal_review": [
        ActionSurface(
            label="File an accessibility (a11y) review request",
            repo="w3c/a11y-request",
            url="https://github.com/w3c/a11y-request/issues/new/choose",
        ),
        ActionSurface(
            label="File an internationalization (i18n) review request",
            repo="w3c/i18n-request",
            url="https://github.com/w3c/i18n-request/issues/new/choose",
        ),
        ActionSurface(
            label="File a privacy review request",
            repo="w3cping/privacy-request",
            url="https://github.com/w3cping/privacy-request/issues/new",
        ),
        ActionSurface(
            label="File a security review request",
            repo="w3cping/security-request",
            url="https://github.com/w3cping/security-request/issues/new",
        ),
        ActionSurface(
            label="Request a TAG design review",
            repo="w3ctag/design-reviews",
            url="https://github.com/w3ctag/design-reviews/issues/new",
        ),
        ActionSurface(
            label="Guidebook — how to get horizontal review",
            url="https://www.w3.org/guide/documentreview/",
        ),
        ActionSurface(
            label="W3C Security and Privacy Questionnaire (TR)",
            url="https://www.w3.org/TR/security-privacy-questionnaire/",
            notes="The concrete checklist the PING and security IGs expect spec authors to work through before requesting privacy/security review. Now indexed in the corpus too.",
        ),
    ],
    "charter_or_recharter": [
        ActionSurface(
            label="Open a charter / recharter tracking issue",
            repo="w3c/strategy",
            url="https://github.com/w3c/strategy/issues",
            notes="Strategy tracks charter and recharter status; new charters land here first.",
        ),
        ActionSurface(
            label="Charter development guide",
            url="https://www.w3.org/guide/process/charter.html",
        ),
        ActionSurface(
            label="Charter extensions guide",
            url="https://www.w3.org/guide/process/charter-extensions.html",
        ),
        # Setup tooling — what the chair / Team contact uses AFTER
        # a charter ships to actually stand the group up.
        ActionSurface(
            label="New-group request form (Guidebook tools)",
            url="https://www.w3.org/guide/tools/new-group.html",
            notes="Walks through the chair-confirmation + Team-resource ticketing the W3C Operations team needs.",
        ),
        ActionSurface(
            label="W3C Repo Manager — provision GitHub repos for a group",
            url="https://labs.w3.org/repo-manager/",
            notes="Used after a charter ships to spin up the WG/IG's GitHub repos with the right w3c.json / labels / branch protections.",
        ),
    ],
    "handle_objection_or_appeal": [
        ActionSurface(
            label="Process — Formally addressing an issue / Formal Objection",
            url="https://www.w3.org/policies/process/#WGArchiveMinorityViews",
        ),
        ActionSurface(
            label="Process — Appeal by Advisory Committee",
            url="https://www.w3.org/policies/process/#ACAppeal",
        ),
        ActionSurface(
            label="Email AC chair for an appeal",
            email="ac-forum@w3.org",
            notes="Use only for AC-level escalations; informal disagreement should be raised in the working group first.",
        ),
    ],
    "check_patent_policy": [
        # Use the canonical /policies/patent-policy/ URL — same path
        # that's now in the corpus after round 24. The legacy
        # /Consortium/Patent-Policy/ URL redirects but breaks the
        # ``link === citation excerpt URL`` invariant the renderer
        # uses for the source pill.
        ActionSurface(
            label="W3C Patent Policy",
            url="https://www.w3.org/policies/patent-policy/",
        ),
        ActionSurface(
            label="Exclusion mechanism (Patent Policy §4)",
            url="https://www.w3.org/policies/patent-policy/#sec-exclude-mech",
            notes="Exclusion notices follow strict timing tied to CR Snapshot publication.",
        ),
    ],
    "coordinate_with_staff_contact": [
        ActionSurface(
            label="Find your Staff Contact via the group page",
            url="https://www.w3.org/groups/",
            notes="Each WG/IG/CG page lists its Staff (Team) Contact and chairs.",
        ),
        ActionSurface(
            label="Team Contact role responsibilities",
            url="https://www.w3.org/guide/teamcontact/role.html",
        ),
    ],
    "run_group_process": [
        ActionSurface(
            label="Group meeting hosting guide",
            url="https://www.w3.org/guide/meetings/hosting.html",
        ),
        ActionSurface(
            label="W3C Workshops",
            url="https://www.w3.org/guide/meetings/workshops.html",
        ),
        # Meeting-tooling surfaces — the W3C IRC bots that take
        # attendance, manage the queue, and produce minutes. The
        # corpus has dedicated chapters for each; these surfaces let
        # the model wire its scribing answer to the operational pages
        # ("how to invoke Zakim" / "how to start RRSAgent recording"
        # / "scribe.perl post-processing").
        ActionSurface(
            label="Zakim IRC bot (queue + agenda + attendance)",
            url="https://www.w3.org/guide/meetings/zakim.html",
        ),
        ActionSurface(
            label="RRSAgent IRC bot (logs IRC + generates minutes)",
            url="https://www.w3.org/guide/meetings/rrsagent.html",
        ),
        ActionSurface(
            label="Scribe handbook (scribe.perl conventions)",
            url="https://www.w3.org/2008/04/scribe.html",
        ),
        ActionSurface(
            label="W3C IRC conventions",
            url="https://www.w3.org/guide/meetings/irc.html",
        ),
        ActionSurface(
            label="Open an operational issue for your group",
            notes="Use your group's own GitHub repository's issue tracker for spec-level discussion; the W3C Team for process questions.",
        ),
        # Behavior + cross-company conduct: separate normative docs
        # the Process + Guidebook reference. Surface them so the
        # model can deep-link to the actual rule the user needs.
        ActionSurface(
            label="W3C Code of Conduct (positive work environment + reporting)",
            url="https://www.w3.org/policies/code-of-conduct/",
        ),
        ActionSurface(
            label="W3C Antitrust and competition policy",
            url="https://www.w3.org/policies/antitrust/",
        ),
    ],
    "transfer_incubation_to_wg": [
        ActionSurface(
            label="Incubation overview",
            url="https://www.w3.org/guide/incubation/",
        ),
        ActionSurface(
            label="W3C Strategy — charter / transition issues",
            repo="w3c/strategy",
            url="https://github.com/w3c/strategy/issues",
        ),
        ActionSurface(
            label="Open an incubation transfer issue",
            repo="w3c/strategy",
            url="https://github.com/w3c/strategy/issues/new/choose",
            notes="Use the 'WG to incubate' / 'CG to WG' templates.",
        ),
    ],
    "plan_or_complete_review": [
        ActionSurface(
            label="Wide-review guidance",
            url="https://www.w3.org/guide/documentreview/",
        ),
        ActionSurface(
            label="Open a wide-review request",
            url="https://www.w3.org/guide/transitions/wide-review-request.html",
        ),
        ActionSurface(
            label="Process — Reviews and Review Responsibilities",
            url="https://www.w3.org/policies/process/#doc-reviews",
        ),
    ],
    "communications_announcement": [
        # The CORRECT mailing list. Round 31 fix: pre-existing
        # advance_specification surface was bleeding ``w3t-tr@w3.org``
        # into announcement answers because "publish/publication"
        # words routed announcement questions to advance_specification.
        # w3t-comm is the W3C Communications Team — handles Call for
        # Review, press releases, AC announcements, public-review-
        # announce posts. w3t-tr is the transitions list, totally
        # different team.
        ActionSurface(
            label="Email the W3C Communications Team",
            email="w3t-comm@w3.org",
            notes="Canonical contact for any AC Call for Review, press release, blog post, or public publication announcement.",
        ),
        ActionSurface(
            label="public-review-announce mailing list",
            email="public-review-announce@w3.org",
            notes="The public list the W3C uses to broadcast each new WD / CR / PR / REC publication. Default notice goes here automatically when the spec is published; subscribe to follow new W3C publications.",
        ),
        ActionSurface(
            label="W3C Process — Publication and Communication (§7.1)",
            url="https://www.w3.org/policies/process/#pub-com",
            notes="Normative rule that the Team manages public communications about W3C publications.",
        ),
        ActionSurface(
            label="Guidebook — Speaking about your work",
            url="https://www.w3.org/guide/#speaking",
            notes="Practical guidance on blog posts, press interviews, and amplifying a publication.",
        ),
        ActionSurface(
            label="Guidebook — Charter Call for Review (Comms Team workflow)",
            url="https://www.w3.org/guide/process/charter.html#cfr",
            notes="Walks through the standard email-to-w3t-comm pattern for a Call for Review; the same pattern applies to other AC announcements.",
        ),
    ],
    "author_spec": [
        # Spec authoring toolchain. Process tells WHAT the document
        # needs to contain (SOTD, references, conformance, ...);
        # these tools are HOW the editor actually produces the
        # marked-up document that satisfies pubrules.
        ActionSurface(
            label="ReSpec — JS-based spec authoring framework",
            url="https://respec.org/docs/",
            notes="Editors write HTML with respec markers; the framework injects boilerplate (SOTD, references, conformance) at render time. Most W3C specs use ReSpec.",
        ),
        ActionSurface(
            label="Bikeshed — preprocessor for spec markup",
            url="https://speced.github.io/bikeshed/",
            notes="Alternative to ReSpec. Source-file → HTML preprocessor with strong cross-referencing. Used by CSS, WHATWG-style specs.",
        ),
        ActionSurface(
            label="W3C Pubrules — pre-publication validator",
            url="https://www.w3.org/pubrules/",
            notes="Validates a draft against W3C's publication conformance rules. Run before requesting publication or invoking Echidna.",
        ),
        ActionSurface(
            label="Echidna — automated publication",
            url="https://github.com/w3c/echidna/wiki",
            notes="WGs configured for Echidna can auto-publish WD / CR-Snapshot from a repo without a manual Team handover.",
        ),
        ActionSurface(
            label="HTMLdiff — generate version diffs",
            url="https://services.w3.org/htmldiff",
            notes="Required output for CR / PR / REC re-publication: a human-readable diff between the new draft and the previous published version.",
        ),
        ActionSurface(
            label="Editor's role guide (Guidebook)",
            url="https://www.w3.org/guide/editor/",
            notes="Process-side responsibilities of a spec editor — what to publish when, who reviews, exit criteria documentation.",
        ),
        ActionSurface(
            label="Repository management guide (Guidebook)",
            url="https://www.w3.org/guide/github/repo-management.html",
            notes="How to set up the spec repository: w3c.json, branch protection, automated publication triggers.",
        ),
    ],
    "attend_or_host_event": [
        # TPAC / workshops / AC face-to-face events. Logistical
        # surfaces — registration, breakout proposals, venue booking,
        # hybrid-meeting tooling.
        ActionSurface(
            label="TPAC homepage (current year)",
            url="https://www.w3.org/events/tpac/",
            notes="The current TPAC event page. Registration, schedule, breakouts, group meeting times — all live here.",
        ),
        ActionSurface(
            label="Guidebook — hosting a W3C meeting (logistics)",
            url="https://www.w3.org/guide/meetings/hosting.html",
            notes="Practical guide for venue booking, scheduling, attendance, hybrid-meeting setup.",
        ),
        ActionSurface(
            label="Guidebook — W3C Workshops",
            url="https://www.w3.org/guide/meetings/workshops.html",
            notes="How to propose, organize, and run a W3C workshop. Includes the workshop charter template.",
        ),
        ActionSurface(
            label="Guidebook — hybrid meetings",
            url="https://www.w3.org/guide/meetings/hybrid-meeting.html",
            notes="Equipment + facilitation guidance for mixed in-person + remote sessions.",
        ),
        ActionSurface(
            label="Email the W3C Events / Operations Team",
            email="w3t-events@w3.org",
            notes="Internal Team list for event logistics. Use for TPAC venue / breakout / registration questions the public pages don't cover.",
        ),
        ActionSurface(
            label="Process — General Meetings (§3.1.1)",
            url="https://www.w3.org/policies/process/#GeneralMeetings",
            notes="Normative requirements for meeting announcements, participation, and minutes.",
        ),
    ],
    "w3c_membership": [
        # Becoming / being a W3C member. Member benefits, dues,
        # invited-expert status, member-only resources.
        ActionSurface(
            label="W3C Membership homepage",
            url="https://www.w3.org/membership/",
            notes="The public membership landing page. Member categories, fee schedule, application form link.",
        ),
        ActionSurface(
            label="Members benefits + how to join",
            url="https://www.w3.org/membership/",
            notes="What W3C members get + the joining process. Note: detailed dues are member-only.",
        ),
        ActionSurface(
            label="Process — Members (§2.1)",
            url="https://www.w3.org/policies/process/#Members",
            notes="Normative definition of W3C Member status, Member Agreement, and Member responsibilities.",
        ),
        ActionSurface(
            label="Invited Expert participation",
            url="https://www.w3.org/invited-experts/",
            notes="Non-Member individuals can participate in a Working Group as Invited Experts — separate from Membership but a useful adjacent path.",
        ),
        ActionSurface(
            label="Email W3C Membership for sales / dues questions",
            email="membership@w3.org",
            notes="Canonical contact for prospective members. Fee structure depends on org size + country tier — Team will guide.",
        ),
    ],
    "group_lifecycle": [
        # Group endings — close a WG, suspend a participant,
        # rescind a Recommendation, post-closure note handling.
        ActionSurface(
            label="Guidebook — closing a Working Group",
            url="https://www.w3.org/guide/process/closing-wg-implementation.html",
            notes="Practical guide for winding down a WG: what to publish, what happens to active drafts, repository transfer.",
        ),
        ActionSurface(
            label="Guidebook — participant suspension",
            url="https://www.w3.org/guide/process/suspension.html",
            notes="The mechanics of suspending a participant from a group (rare; usually after Code-of-Conduct escalation).",
        ),
        ActionSurface(
            label="Guidebook — obsolete, rescinded, superseded",
            url="https://www.w3.org/guide/process/obsolete-rescinded-supserseded.html",
            notes="How to mark a published Recommendation as obsolete, rescinded, or superseded by a successor spec.",
        ),
        ActionSurface(
            label="Process — Rescinding a Recommendation (§6.7)",
            url="https://www.w3.org/policies/process/#rec-rescind",
            notes="Normative procedure for rescinding a published REC. Includes the AC Review requirement.",
        ),
        ActionSurface(
            label="Email the W3C Team about a group closure",
            email="w3t@w3.org",
            notes="Generic Team contact — for group-closure logistics, repo transfer, list archival, the Team will route internally.",
        ),
    ],
    "elected_body": [
        # AB / TAG elections, nominations, composition.
        ActionSurface(
            label="Guidebook — organizing an AB or TAG election",
            url="https://www.w3.org/guide/process/election.html",
            notes="Step-by-step for running an election: timeline, nomination period, voting window, results announcement.",
        ),
        ActionSurface(
            label="Guidebook — Elected Body Communication Guidelines",
            url="https://www.w3.org/guide/other/elected-body-communication-guidelines.html",
            notes="How AB / TAG members should communicate — what's confidential, what's public, what's W3C-only.",
        ),
        ActionSurface(
            label="W3C Advisory Board page",
            url="https://www.w3.org/2002/ab/",
            notes="The AB landing page — current members, charter, meeting minutes.",
        ),
        ActionSurface(
            label="W3C TAG page",
            url="https://www.w3.org/2001/tag/",
            notes="The TAG landing page — current members, design reviews, findings.",
        ),
        ActionSurface(
            label="Process — Advisory Board (§2.5)",
            url="https://www.w3.org/policies/process/#AB",
            notes="Normative definition of the Advisory Board: role, composition, term length.",
        ),
        ActionSurface(
            label="Process — Technical Architecture Group (§2.6)",
            url="https://www.w3.org/policies/process/#TAG",
            notes="Normative definition of the TAG: role, composition, term length.",
        ),
    ],
    "explain_process": [
        ActionSurface(
            label="W3C Process Document",
            url="https://www.w3.org/policies/process/",
            notes="Normative source of all Process rules.",
        ),
        ActionSurface(
            label="W3C Guidebook (Art of Consensus)",
            url="https://www.w3.org/guide/",
            notes="Practice guidance, day-to-day workflow.",
        ),
        ActionSurface(
            label="Ask in the W3C Process Community Group",
            url="https://github.com/w3c/w3process/issues",
            notes="Use for clarification on Process language or to propose Process changes.",
        ),
    ],
}


def surfaces_for_intent(intent_type: str | None) -> list[ActionSurface]:
    """Return the action surfaces registered for ``intent_type``.

    Unknown / unset intents return an empty list — the prompt then falls
    back to relying on the citation URLs alone for action targets.
    """
    if not intent_type:
        return []
    return ACTION_SURFACES.get(intent_type, [])


def format_surfaces_for_prompt(surfaces: list[ActionSurface]) -> str:
    """Render surfaces as a plain bulleted list (no ``[An]`` labels).

    Format is chosen to nudge the model toward markdown-link output —
    the rendering rule in the system prompt asks for
    ``[descriptive label](url)`` so the frontend can paint the URL
    as a clickable inline link instead of ``url=https://...``
    surfacing as literal text.

    No ``[An]`` reference tags: those would just be echoed back into
    the answer and the UI's citation renderer doesn't know how to
    dereference them.
    """
    if not surfaces:
        return ""
    lines = []
    for surface in surfaces:
        target = surface.url or (f"mailto:{surface.email}" if surface.email else None) or surface.repo
        if target:
            line = f"- {surface.label} — {target}"
        else:
            line = f"- {surface.label}"
        if surface.notes:
            line = f"{line}\n    note: {surface.notes}"
        lines.append(line)
    return "\n".join(lines)


# Pattern used by ``linkify_bare_action_urls`` to skip URLs that are
# already inside markdown link syntax. A URL is considered "already
# wrapped" when ``](`` appears immediately before it (the href slot of
# ``[label](url)``). We can't use a negative lookbehind directly inside
# the alternation pattern because the URLs differ in length, so we
# build a single combined pattern with the lookbehind upfront.
_MARKDOWN_HREF_PREFIX = re.compile(r"\]\(")


def linkify_bare_action_urls(text: str, surfaces: list[ActionSurface]) -> str:
    """Wrap bare action-surface URLs in the answer with ``[url](url)``.

    Belt-and-suspenders fix for the case where the model emits the
    raw ``https://github.com/...`` instead of ``[label](url)`` — the
    frontend renders raw URLs as inert plain text, so a bare URL
    shows up as an unclickable string. This post-pass scans the
    finished answer for URLs from THIS request's surface list and
    wraps them so the renderer turns them into clickable links.

    Constraints:
      * Only the curated surface URLs are touched — never arbitrary
        URLs from elsewhere in the answer. Keeps the XSS surface at
        zero and prevents corrupting citation URLs.
      * Skips URLs already inside ``[label](url)`` syntax so we don't
        double-wrap. Detected by ``](`` directly preceding the URL.
      * Skips URLs that ARE the label part (``[https://…]``) by
        requiring no ``[`` immediately before. Rare but possible.
    """
    if not text or not surfaces:
        return text
    urls = sorted(
        {surface.url for surface in surfaces if surface.url},
        key=len,
        reverse=True,  # match longest first so a prefix URL doesn't beat its parent
    )
    if not urls:
        return text
    for url in urls:
        if url not in text:
            continue
        # Walk the text, replacing each bare occurrence. We do this
        # ourselves (instead of ``str.replace``) so we can check the
        # immediately-preceding characters and skip already-wrapped
        # instances.
        out: list[str] = []
        i = 0
        url_len = len(url)
        while True:
            idx = text.find(url, i)
            if idx == -1:
                out.append(text[i:])
                break
            preceding = text[max(0, idx - 2): idx]
            already_wrapped = preceding.endswith("](") or text[max(0, idx - 1): idx] == "["
            out.append(text[i:idx])
            if already_wrapped:
                out.append(url)
            else:
                out.append(f"[{url}]({url})")
            i = idx + url_len
        text = "".join(out)
    return text
