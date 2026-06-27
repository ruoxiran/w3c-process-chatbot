"""Unit tests for the W3C-acronym + maturity-stage helpers."""

from __future__ import annotations

import pytest

from app.services.w3c_terminology import (
    canonical_maturity_stage,
    expand_acronyms_for_retrieval,
    next_recommendation_track_stage,
)


# ---------- acronym expansion ---------------------------------------------


@pytest.mark.parametrize(
    "query, expected_appended",
    [
        # Bare acronym: long form appended.
        ("from CR to REC", ("Candidate Recommendation", "Recommendation")),
        ("AC review process", ("Advisory Committee",)),
        ("FPWD then WD", ("First Public Working Draft", "Working Draft")),
        # Case-insensitive matching.
        ("how does the tag review work", ("Technical Architecture Group",)),
        # Multi-word terms get the full sub-track hint.
        ("horizontal review on a spec",
         ("horizontal review (i18n internationalization, a11y accessibility, privacy review, security review, TAG review)",)),
    ],
)
def test_expand_acronyms_appends_long_form(query: str, expected_appended: tuple[str, ...]) -> None:
    result = expand_acronyms_for_retrieval(query)
    # Original text is always preserved verbatim.
    assert query in result
    for expansion in expected_appended:
        assert expansion in result, f"missing expansion {expansion!r} in {result!r}"


def test_expand_acronyms_does_not_double_expand_when_long_form_present() -> None:
    """If the user already wrote "Candidate Recommendation" we don't
    also append "Candidate Recommendation" again — keeps the retrieval
    query from getting absurdly long."""
    query = "Working Draft to Candidate Recommendation"
    result = expand_acronyms_for_retrieval(query)
    # The bare "WD" and "CR" expansions should NOT be appended
    # because both long forms are already present.
    assert result == query


def test_expand_acronyms_handles_empty_and_no_match() -> None:
    assert expand_acronyms_for_retrieval("") == ""
    # No acronyms → query unchanged.
    assert expand_acronyms_for_retrieval("how to scribe?") == "how to scribe?"


def test_expand_acronyms_treats_acronym_inside_word_correctly() -> None:
    """Word-boundary matching: 'rec' inside 'recipe' must NOT trigger
    a Recommendation expansion."""
    result = expand_acronyms_for_retrieval("recipe for charter approval")
    assert "Recommendation" not in result


# ---------- maturity stage canonicalisation -------------------------------


@pytest.mark.parametrize(
    "api_status, expected_stage",
    [
        ("Candidate Recommendation Snapshot", "CR"),
        ("Candidate Recommendation Draft", "CR"),
        ("Candidate Recommendation", "CR"),
        ("Proposed Recommendation", "PR"),
        ("First Public Working Draft", "FPWD"),
        ("Working Draft", "WD"),
        ("Recommendation", "REC"),
        ("Group Note", "NOTE"),
        # Case insensitive.
        ("candidate recommendation", "CR"),
        # Unknown returns None — never guess.
        ("Unicorn Stage", None),
        (None, None),
        ("", None),
    ],
)
def test_canonical_maturity_stage(api_status: str | None, expected_stage: str | None) -> None:
    assert canonical_maturity_stage(api_status) == expected_stage


# ---------- next-stage on the REC track -----------------------------------


@pytest.mark.parametrize(
    "current, expected_next",
    [
        ("FPWD", "WD"),
        ("WD", "CR"),
        ("CR", "PR"),
        ("PR", "REC"),
        # Terminal stage: no further step.
        ("REC", None),
        # Off-track deliverables: no track-style next stage.
        ("NOTE", None),
        ("RETIRED", None),
        # Unknown / missing.
        (None, None),
        ("", None),
        ("unicorn", None),
        # Case insensitive.
        ("cr", "PR"),
    ],
)
def test_next_recommendation_track_stage(current: str | None, expected_next: str | None) -> None:
    assert next_recommendation_track_stage(current) == expected_next
