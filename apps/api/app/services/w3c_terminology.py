"""W3C-specific terminology helpers: acronym expansion + maturity-stage mapping.

Two distinct jobs, both small but high-leverage for accuracy.

1. Acronym expansion for retrieval
   Queries from real users overwhelmingly use the short form: "from CR
   to REC", "AC review", "FPWD". The corpus uses both — sometimes the
   acronym is in the heading, sometimes only the expansion is. BM25
   scores them as different terms. Expanding bare acronyms in the
   retrieval query (silently appended, not replacing the user's words)
   makes lexical retrieval match either form.

2. Maturity-stage canonicalisation
   The W3C API ``status`` field returns user-facing labels like
   "Candidate Recommendation Snapshot". The Process Document and the
   workflow inspector use the 2-3 letter code (CR). Mapping API
   strings to canonical stages lets the prompt show
   "status=Candidate Recommendation (CR); next track stage: PR" —
   actionable hint instead of just a status name.
"""

from __future__ import annotations

import re


# Canonical acronym → expansion. Lower-case keys; matched case-
# insensitively. Multi-word entries are intentional — "horizontal
# review" gets expanded with its concrete sub-tracks so retrieval
# hits the i18n/a11y/privacy/security pages too.
W3C_TERMS: dict[str, str] = {
    # Maturity stages on the Recommendation track
    "fpwd": "First Public Working Draft",
    "wd": "Working Draft",
    "cr": "Candidate Recommendation",
    "crs": "Candidate Recommendation Snapshot",
    "crd": "Candidate Recommendation Draft",
    "pr": "Proposed Recommendation",
    "rec": "Recommendation",
    # Non-REC-track deliverables
    "note": "Group Note",
    "wgn": "Working Group Note",
    "ign": "Interest Group Note",
    # Groups + roles
    "ac": "Advisory Committee",
    "ab": "Advisory Board",
    "tag": "Technical Architecture Group",
    "wg": "Working Group",
    "ig": "Interest Group",
    "cg": "Community Group",
    "bg": "Business Group",
    "tf": "Task Force",
    # Process surfaces
    "ipr": "Intellectual Property Rights",
    "sotd": "Status of This Document",
    "fo": "Formal Objection",
    "cfc": "Call for Consensus",
    "cfr": "Call for Review",
    # Horizontal review tracks
    "i18n": "internationalization",
    "a11y": "accessibility",
    "horizontal review": "horizontal review (i18n internationalization, a11y accessibility, privacy review, security review, TAG review)",
}


# Word-boundary pattern that matches each acronym from ``W3C_TERMS``.
# Built once at import time; sorted by length descending so longer
# multi-word entries match before their prefixes (so "horizontal
# review" wins over "review" — there's no bare "review" entry today,
# but the ordering future-proofs adding one).
_ACRONYM_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(W3C_TERMS.keys(), key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE,
)


def expand_acronyms_for_retrieval(query: str) -> str:
    """Return ``query`` plus appended expansions for any bare acronyms.

    The original query text is preserved verbatim (so the model still
    sees what the user actually typed); expansions are appended as a
    space-separated trailer so BM25 + TF-IDF index hits on either
    form. Expansions for terms already present in long-form are
    skipped to avoid pointless inflation.

    Examples
    --------
    >>> expand_acronyms_for_retrieval("from CR to REC")
    'from CR to REC Candidate Recommendation Recommendation'
    >>> expand_acronyms_for_retrieval("Working Draft to CR")
    'Working Draft to CR Candidate Recommendation'
    >>> expand_acronyms_for_retrieval("how to scribe?")
    'how to scribe?'
    """
    if not query:
        return query
    text_lower = query.lower()
    extras: list[str] = []
    seen: set[str] = set()
    for match in _ACRONYM_PATTERN.finditer(query):
        acronym = match.group(1).lower()
        expansion = W3C_TERMS.get(acronym)
        if expansion is None:
            continue
        if expansion.lower() in text_lower:
            # User already wrote the long form; don't duplicate.
            continue
        if expansion in seen:
            continue
        seen.add(expansion)
        extras.append(expansion)
    if not extras:
        return query
    return f"{query} {' '.join(extras)}"


# Lower-case API ``status`` substring → canonical 2-3 letter stage.
# Ordered longest-first inside the matcher so "candidate recommendation
# snapshot" wins over the bare "candidate recommendation" prefix.
_STAGE_FROM_API_STATUS: tuple[tuple[str, str], ...] = (
    ("first public working draft", "FPWD"),
    ("candidate recommendation snapshot", "CR"),
    ("candidate recommendation draft", "CR"),
    ("candidate recommendation", "CR"),
    ("proposed recommendation", "PR"),
    ("working draft", "WD"),
    ("recommendation", "REC"),
    ("group note", "NOTE"),
    ("working group note", "NOTE"),
    ("interest group note", "NOTE"),
    ("retired", "RETIRED"),
    ("superseded", "SUPERSEDED"),
)


def canonical_maturity_stage(api_status: str | None) -> str | None:
    """Map a W3C API status string to the canonical Process stage code.

    Returns ``None`` for unrecognised inputs rather than guessing; the
    caller should treat unknown stages as "no actionable hint" rather
    than fabricating one.
    """
    if not api_status:
        return None
    text = api_status.lower()
    for needle, stage in _STAGE_FROM_API_STATUS:
        if needle in text:
            return stage
    return None


# Recommendation-track ordering. The next-stage suggestion is only
# emitted for specs ON the track; NOTE / RETIRED / SUPERSEDED don't
# have a "next" stage in the same sense.
_REC_TRACK: tuple[str, ...] = ("FPWD", "WD", "CR", "PR", "REC")


def next_recommendation_track_stage(current_stage: str | None) -> str | None:
    """Return the next stage on the Recommendation track, or ``None``.

    ``None`` covers: unknown stage, off-track stage (NOTE / RETIRED /
    SUPERSEDED), or already at the terminal stage (REC).
    """
    if not current_stage:
        return None
    stage = current_stage.upper()
    try:
        idx = _REC_TRACK.index(stage)
    except ValueError:
        return None
    if idx + 1 >= len(_REC_TRACK):
        return None
    # WD → CR skips FPWD when the spec has already had a first public
    # publication — but the Process treats FPWD as a one-time milestone,
    # not a stage you go BACK to. Same for FPWD → CR. Keep the simple
    # linear progression here; the prompt can refine when needed.
    return _REC_TRACK[idx + 1]
