"""Tests for the citation verifier post-pass.

Contract under test: out-of-range ``[Sn]`` tags are stripped (they point
at nothing renderable), while in-range tags the LLM judge marks
unsupported KEEP their source link and gain an inline ``*[unverified]*``
badge. Deleting a judged tag would remove the user's only path to the
source on the strength of a single fallible LLM verdict.
"""
from __future__ import annotations

from app.core.config import Settings
from app.models.schemas import Citation, SourceType
from app.services.citation_verifier import verify_citations


def _citations(n: int) -> list[Citation]:
    return [
        Citation(
            title=f"Source {i}",
            url=f"https://www.w3.org/policies/process/#section-{i}",
            source_type=SourceType.process,
            quote=f"Excerpt text {i}.",
        )
        for i in range(1, n + 1)
    ]


class _JudgeClient:
    """Fake JSONGenerator returning a canned verdict."""

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls = 0

    def generate_json(self, *, model: str, prompt: str, num_predict: int = 500) -> dict[str, object]:
        self.calls += 1
        return self.payload


def test_out_of_range_tag_is_stripped_without_client() -> None:
    answer = "The Process requires wide review before CR [S99]."
    result = verify_citations(answer, _citations(2), settings=Settings(), client=None)

    assert "[S99]" not in result.answer
    assert result.stripped_pairs and result.stripped_pairs[0][1] == 99


def test_judged_unsupported_tag_keeps_link_and_gains_badge() -> None:
    answer = "The Process requires horizontal review [S1]. The Guidebook explains the steps [S2]."
    client = _JudgeClient({"pairs": [{"index": 1, "supported": False}, {"index": 2, "supported": True}]})

    result = verify_citations(answer, _citations(2), settings=Settings(), client=client)

    assert "[S1] *[unverified]*" in result.answer
    assert "[S2]" in result.answer
    assert "[S2] *[unverified]*" not in result.answer
    assert len(result.stripped_pairs) == 1


def test_repeated_unsupported_tag_is_badged_everywhere_once() -> None:
    answer = "Rule one [S1]. Rule two also cites it [S1]."
    client = _JudgeClient({"pairs": [{"index": 1, "supported": False}]})

    result = verify_citations(answer, _citations(1), settings=Settings(), client=client)

    assert result.answer.count("[S1] *[unverified]*") == 2
    assert "*[unverified]* *[unverified]*" not in result.answer


def test_all_supported_leaves_answer_unchanged() -> None:
    answer = "The Process requires horizontal review [S1]."
    client = _JudgeClient({"pairs": [{"index": 1, "supported": True}]})

    result = verify_citations(answer, _citations(1), settings=Settings(), client=client)

    assert result.answer == answer
    assert result.stripped_pairs == []


def test_judge_error_keeps_answer_as_is() -> None:
    class _BrokenClient:
        def generate_json(self, *, model: str, prompt: str, num_predict: int = 500) -> dict[str, object]:
            raise RuntimeError("judge model down")

    answer = "The Process requires horizontal review [S1]."
    result = verify_citations(answer, _citations(1), settings=Settings(), client=_BrokenClient())

    assert result.answer == answer
    assert result.skipped_reason == "RuntimeError"
