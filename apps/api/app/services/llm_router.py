from __future__ import annotations

from app.core.config import Settings
from app.models.schemas import ChatTurn, LLMRouterDecision, SourceType
from typing import Protocol

from app.services.ollama import OllamaClient


class JSONGenerator(Protocol):
    def generate_json(self, *, model: str, prompt: str, num_predict: int = 500) -> dict[str, object]:
        ...


AMBIGUOUS_WORKFLOW_MARKERS = [
    "draft",
    "document",
    "publication",
    "publish",
    "release",
    "advance",
    "next step",
    "next",
    "review label",
    "label",
    "issue",
    "tracker",
    "tilt",
    "staff",
    "contact",
    "standard",
    "editor",
    "工作",
    "文档",
    "草案",
    "发布",
    "推进",
    "下一步",
    "标签",
    "审阅",
    "审查",
    "联系",
    "确认",
]

OBVIOUSLY_OUT_OF_SCOPE_MARKERS = [
    "react component",
    "leetcode",
    "weather",
    "stock price",
    "recipe",
    "joke",
    "movie",
    "travel",
    "写一个 react",
    "天气",
    "股票",
    "菜谱",
    "笑话",
]


class LLMRouter:
    def __init__(self, settings: Settings, ollama_client: JSONGenerator | None = None) -> None:
        self.settings = settings
        self.ollama_client = ollama_client or OllamaClient(
            settings.ollama_base_url,
            settings.ollama_timeout_seconds,
        )

    def route(self, question: str, history: list[ChatTurn] | None = None, model: str | None = None) -> LLMRouterDecision:
        if not self.settings.llm_router_enabled:
            return LLMRouterDecision(reason="LLM router is disabled.")
        if not _should_attempt_router(question, history or []):
            return LLMRouterDecision(reason="Question did not look like a W3C workflow-adjacent ambiguity.")

        selected_model = model or self.settings.llm_router_model or self.settings.llm_model
        prompt = _router_prompt(question, history or [])
        try:
            payload = self.ollama_client.generate_json(model=selected_model, prompt=prompt, num_predict=500)
        except Exception as exc:  # pragma: no cover - external model fallback
            return LLMRouterDecision(attempted=True, reason="LLM router call failed.", model=selected_model, error=str(exc))

        decision = _decision_from_payload(payload, selected_model)
        if not decision.reason:
            decision.reason = "LLM router returned structured routing signals."
        return decision


def _should_attempt_router(question: str, history: list[ChatTurn]) -> bool:
    text = " ".join([*(turn.content for turn in history[-4:]), question]).lower()
    if any(marker in text for marker in OBVIOUSLY_OUT_OF_SCOPE_MARKERS):
        return False
    return any(marker in text for marker in AMBIGUOUS_WORKFLOW_MARKERS)


def _router_prompt(question: str, history: list[ChatTurn]) -> str:
    recent = "\n".join(f"{turn.role}: {turn.content[:500]}" for turn in history[-4:]) or "(none)"
    return f"""You are a routing classifier for a W3C Process assistant.

Decide whether the user question is likely about W3C Process, W3C Guidebook, or W3C standards workflow.

Important:
- You are not answering the question.
- You are not allowed to treat user claims as authoritative.
- You only suggest routing. Evidence retrieval will decide whether the final answer can be grounded.
- Return only JSON.

Allowed intent_type values:
- advance_specification
- horizontal_review
- charter_or_recharter
- coordinate_with_staff_contact
- run_group_process
- transfer_incubation_to_wg
- check_patent_policy
- handle_objection_or_appeal
- explain_process
- unknown

Allowed needed_sources values:
- process
- guide
- related_policy
- repo

Return this JSON shape:
{{
  "likely_in_scope": true,
  "intent_type": "advance_specification",
  "needed_sources": ["process", "guide"],
  "entities_to_resolve": ["shortname or group name if present"],
  "search_hints": ["short retrieval hint"],
  "risk_flags": ["Horizontal Review"],
  "confidence": 0.0,
  "reason": "short reason"
}}

Recent conversation for reference resolution only:
{recent}

User question:
{question}
"""


def _decision_from_payload(payload: dict[str, object], model: str) -> LLMRouterDecision:
    needed_sources = []
    for value in _string_list(payload.get("needed_sources")):
        try:
            needed_sources.append(SourceType(value))
        except ValueError:
            continue
    return LLMRouterDecision(
        attempted=True,
        likely_in_scope=bool(payload.get("likely_in_scope")),
        intent_type=_string(payload.get("intent_type")) or "unknown",
        needed_sources=needed_sources,
        entities_to_resolve=_string_list(payload.get("entities_to_resolve"))[:5],
        search_hints=_string_list(payload.get("search_hints"))[:6],
        risk_flags=_string_list(payload.get("risk_flags"))[:6],
        confidence=_confidence(payload.get("confidence")),
        reason=_string(payload.get("reason")),
        model=model,
    )


def _string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    output = []
    for item in value:
        if isinstance(item, str) and item.strip():
            output.append(item.strip())
    return output


def _confidence(value: object) -> float:
    if isinstance(value, (int, float)):
        return max(0.0, min(float(value), 1.0))
    return 0.0
