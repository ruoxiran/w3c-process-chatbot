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
            label="Email the W3C Team about a transition",
            email="w3t-tr@w3.org",
            notes="Internal Team list for transition-meeting scheduling and clarifying questions.",
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
        ActionSurface(
            label="W3C Patent Policy",
            url="https://www.w3.org/Consortium/Patent-Policy/",
        ),
        ActionSurface(
            label="File an Exclusion via the Patent Policy form",
            url="https://www.w3.org/Consortium/Patent-Policy/#sec-Exclusion",
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
        ActionSurface(
            label="Open an operational issue for your group",
            notes="Use your group's own GitHub repository's issue tracker for spec-level discussion; the W3C Team for process questions.",
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

    The model should embed the concrete URL / mailto / repo directly into
    the answer ("file at https://github.com/w3c/i18n-request/issues/new/choose"),
    not invent a reference tag. Labels like ``[A1]`` would just be echoed
    back into the answer text and the UI's citation renderer doesn't know
    how to dereference them.
    """
    if not surfaces:
        return ""
    lines = []
    for surface in surfaces:
        bits: list[str] = [f"- {surface.label}"]
        if surface.url:
            bits.append(f"url={surface.url}")
        if surface.email:
            bits.append(f"mailto={surface.email}")
        if surface.repo:
            bits.append(f"repo={surface.repo}")
        line = "; ".join(bits)
        if surface.notes:
            line = f"{line}\n    note: {surface.notes}"
        lines.append(line)
    return "\n".join(lines)
