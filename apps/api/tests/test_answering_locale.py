"""Tests for template-answer language selection.

The API's default locale is ``auto``. The old check (``locale.startswith("en")``)
routed every non-``en`` locale — including ``auto`` — to the Chinese
templates, so any API client that didn't explicitly send ``locale=en`` got a
Chinese fallback answer for an English question.
"""
from __future__ import annotations

from app.services.answering import build_grounded_answer, build_refusal


def test_auto_locale_english_message_gets_english_template() -> None:
    answer, _steps = build_grounded_answer("how to do a a11y review?", [], locale="auto")
    assert "horizontal review" in answer.lower() or "review" in answer.lower()
    assert not any("一" <= ch <= "鿿" for ch in answer)


def test_auto_locale_chinese_message_gets_chinese_template() -> None:
    answer, _steps = build_grounded_answer("如何进行无障碍审查？", [], locale="auto")
    assert any("一" <= ch <= "鿿" for ch in answer)


def test_explicit_en_locale_wins_over_chinese_message() -> None:
    answer, _steps = build_grounded_answer("如何进行无障碍审查？", [], locale="en")
    assert not any("一" <= ch <= "鿿" for ch in answer)


def test_explicit_zh_locale_wins_over_english_message() -> None:
    answer, _steps = build_grounded_answer("how to do a a11y review?", [], locale="zh")
    assert any("一" <= ch <= "鿿" for ch in answer)


def test_refusal_follows_message_language_on_auto() -> None:
    assert "W3C Process" in build_refusal("auto", "what's the weather today?")
    assert not any("一" <= ch <= "鿿" for ch in build_refusal("auto", "what's the weather today?"))
    assert any("一" <= ch <= "鿿" for ch in build_refusal("auto", "今天天气怎么样？"))
