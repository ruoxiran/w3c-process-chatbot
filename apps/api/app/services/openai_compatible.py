from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Iterator
from dataclasses import dataclass

import httpx


logger = logging.getLogger(__name__)

# Statuses we should treat as transient. 429 is the common rate-limit code;
# 5xx is upstream-transient. Everything else (4xx) is the caller's fault and
# retrying would just burn quota.
_RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
_MAX_RETRIES = 3

from app.models.schemas import ChatTurn, Citation, CompiledContext, DraftContext, EvidenceCoverage, ModelInfo, ProcessState, TaskPlan, W3CEntity
from app.services.ollama import _clean_model_text, _extract_json_object, build_prompt


@dataclass(frozen=True)
class OpenAICompatibleGeneration:
    text: str
    model: str


class OpenAICompatibleClient:
    """Client for OpenAI-compatible `/v1/chat/completions` APIs.

    This works with OpenAI, OpenRouter, self-hosted vLLM, and many internal
    model gateways as long as they implement the chat completions shape.
    """

    def __init__(self, base_url: str, api_key: str | None, timeout_seconds: float = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def list_models(self) -> list[ModelInfo]:
        response = httpx.get(
            f"{self.base_url}/models",
            headers=self._headers(),
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        models: list[ModelInfo] = []
        for item in payload.get("data", []):
            if not isinstance(item, dict):
                continue
            name = item.get("id")
            if not isinstance(name, str) or not name:
                continue
            models.append(ModelInfo(name=name, provider="openai-compatible"))
        return models

    def generate_json(self, *, model: str, prompt: str, num_predict: int = 500) -> dict[str, object]:
        text = self._chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Return only a valid JSON object. Do not include markdown or explanations.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=num_predict,
            response_format={"type": "json_object"},
        )
        return _extract_json_object(_clean_model_text(text))

    def generate_answer(
        self,
        *,
        model: str,
        question: str,
        locale: str,
        citations: list[Citation],
        fallback_answer: str,
        fallback_next_steps: list[str],
        history: list[ChatTurn] | None = None,
        entities: list[W3CEntity] | None = None,
        task_plan: TaskPlan | None = None,
        process_state: ProcessState | None = None,
        evidence_coverage: EvidenceCoverage | None = None,
        draft_contexts: list[DraftContext] | None = None,
        compiled_context: CompiledContext | None = None,
        supplementary_context: str | None = None,
        action_surfaces_text: str = "",
        lighter_mode: bool = False,
    ) -> OpenAICompatibleGeneration:
        prompt = build_prompt(
            question=question,
            locale=locale,
            citations=citations,
            fallback_answer=fallback_answer,
            fallback_next_steps=fallback_next_steps,
            history=history or [],
            entities=entities or [],
            task_plan=task_plan,
            process_state=process_state,
            evidence_coverage=evidence_coverage,
            draft_contexts=draft_contexts or [],
            compiled_context=compiled_context,
            supplementary_context=supplementary_context,
            action_surfaces_text=action_surfaces_text,
            lighter_mode=lighter_mode,
        )
        text = self._chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a W3C Process assistant constrained by source-grounded evidence. "
                        "Follow the user's prompt rules exactly and do not reveal hidden reasoning."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            # Generous ceiling: the prompt now asks for thorough, in-depth
            # answers, and thinking models (Kimi k2.5) also spend part of this
            # budget on hidden reasoning before emitting text. 8192 leaves room
            # for a long grounded answer without truncating mid-sentence.
            max_tokens=4096,
        )
        return OpenAICompatibleGeneration(text=_clean_model_text(text), model=model)

    def stream_answer(
        self,
        *,
        model: str,
        question: str,
        locale: str,
        citations: list[Citation],
        fallback_answer: str,
        fallback_next_steps: list[str],
        history: list[ChatTurn] | None = None,
        entities: list[W3CEntity] | None = None,
        task_plan: TaskPlan | None = None,
        process_state: ProcessState | None = None,
        evidence_coverage: EvidenceCoverage | None = None,
        draft_contexts: list[DraftContext] | None = None,
        compiled_context: CompiledContext | None = None,
        supplementary_context: str | None = None,
        action_surfaces_text: str = "",
        lighter_mode: bool = False,
    ) -> Iterator[str]:
        """Yield raw text deltas from the chat completions streaming API.

        Caller is responsible for assembling and post-cleaning the final text
        (call ``_clean_model_text`` on the joined result).
        """
        prompt = build_prompt(
            question=question,
            locale=locale,
            citations=citations,
            fallback_answer=fallback_answer,
            fallback_next_steps=fallback_next_steps,
            history=history or [],
            entities=entities or [],
            task_plan=task_plan,
            process_state=process_state,
            evidence_coverage=evidence_coverage,
            draft_contexts=draft_contexts or [],
            compiled_context=compiled_context,
            supplementary_context=supplementary_context,
            action_surfaces_text=action_surfaces_text,
            lighter_mode=lighter_mode,
        )
        payload: dict[str, object] = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a W3C Process assistant constrained by source-grounded evidence. "
                        "Follow the user's prompt rules exactly and do not reveal hidden reasoning."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            # Matches the sync path: a generous ceiling for thorough answers,
            # with headroom for thinking models whose reasoning shares it.
            "max_tokens": 4096,
            "stream": True,
        }
        # Mirror ``_post_with_backoff`` for the streaming connection: retry
        # transient statuses (429 rate-limit, 5xx) with exponential backoff
        # before the first token. Once deltas have been yielded downstream
        # there is no safe retry, but connection-time failures — the common
        # case with per-minute rate limits — get the same resilience as the
        # sync path.
        url = f"{self.base_url}/chat/completions"
        for attempt in range(_MAX_RETRIES + 1):
            with httpx.stream(
                "POST",
                url,
                headers=self._headers(),
                json=payload,
                timeout=self.timeout_seconds,
            ) as response:
                if _is_sampling_param_rejection(response, payload):
                    # Some models pin sampling params (e.g. Kimi k2.5 only
                    # accepts temperature=1). Drop ours and let the provider
                    # default apply — grounding comes from the prompt and the
                    # citation gates, not from a low temperature.
                    body = response.read()
                    logger.warning(
                        "Upstream rejected sampling params on %s (%s); retrying without temperature",
                        url, body[:300],
                    )
                    payload.pop("temperature", None)
                    wait = 0.0
                elif response.status_code in _RETRYABLE_STATUS_CODES and attempt < _MAX_RETRIES:
                    body = response.read()
                    retry_after = _parse_retry_after(response.headers.get("retry-after"))
                    wait = retry_after if retry_after is not None else min(16.0, (2 ** attempt) + random.random())
                    logger.warning(
                        "Upstream returned %d on streaming %s (%s); sleeping %.2fs (attempt %d/%d)",
                        response.status_code, url, body[:300], wait, attempt + 1, _MAX_RETRIES,
                    )
                else:
                    if response.status_code >= 400:
                        # Read the body before raising so the error log shows
                        # WHY upstream refused (rate limit vs overload vs auth)
                        # instead of a bare status code.
                        body = response.read()
                        logger.error(
                            "Upstream returned %d on streaming %s: %s",
                            response.status_code, url, body[:300],
                        )
                        response.raise_for_status()
                    yield from self._iter_stream_deltas(response)
                    return
            time.sleep(wait)

    @staticmethod
    def _iter_stream_deltas(response: httpx.Response) -> Iterator[str]:
        for line in response.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                if data == "[DONE]":
                    break
                continue
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = obj.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
            if not isinstance(delta, dict):
                continue
            content = delta.get("content")
            if isinstance(content, str) and content:
                yield content

    def _chat(
        self,
        *,
        model: str,
        messages: list[dict[str, object]],
        temperature: float,
        max_tokens: int,
        response_format: dict[str, str] | None = None,
    ) -> str:
        payload: dict[str, object] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format
        response = self._post_with_backoff(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout_seconds,
        )
        if _is_sampling_param_rejection(response, payload):
            # Same contract as the streaming path: models that pin sampling
            # params (Kimi k2.5: temperature must be 1) get one retry with
            # our temperature removed so the provider default applies.
            logger.warning(
                "Upstream rejected sampling params (%s); retrying without temperature",
                response.content[:300],
            )
            payload.pop("temperature", None)
            response = self._post_with_backoff(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=self.timeout_seconds,
            )
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0]
        if not isinstance(first, dict):
            return ""
        message = first.get("message")
        if not isinstance(message, dict):
            return ""
        content = message.get("content")
        return content if isinstance(content, str) else ""

    def _post_with_backoff(self, url: str, **kwargs) -> httpx.Response:
        """POST with exponential backoff on 429 / 5xx responses.

        Respects ``Retry-After`` if upstream sends one; otherwise uses
        ``1 * 2^attempt`` seconds + jitter, capped at 16 s per retry. After
        ``_MAX_RETRIES`` failed attempts the last response is returned so
        the caller can ``raise_for_status()`` and get the original error.
        """
        last_response: httpx.Response | None = None
        for attempt in range(_MAX_RETRIES + 1):
            response = httpx.post(url, **kwargs)
            if response.status_code not in _RETRYABLE_STATUS_CODES:
                return response
            last_response = response
            if attempt == _MAX_RETRIES:
                break
            retry_after = _parse_retry_after(response.headers.get("retry-after"))
            wait = retry_after if retry_after is not None else min(16.0, (2 ** attempt) + random.random())
            logger.info(
                "Upstream returned %d on %s; sleeping %.2fs (attempt %d/%d)",
                response.status_code, url, wait, attempt + 1, _MAX_RETRIES,
            )
            time.sleep(wait)
        return last_response or httpx.post(url, **kwargs)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


def _is_sampling_param_rejection(response: httpx.Response, payload: dict[str, object]) -> bool:
    """True when upstream rejected the request because of our ``temperature``.

    Some models pin sampling parameters (Kimi k2.5 returns
    ``400 invalid temperature: only 1 is allowed for this model``). Detected
    by a 400 status whose body mentions "temperature" while our payload still
    carries one — after the caller strips it, this can't match again, so the
    retry is naturally bounded to one attempt.
    """
    if response.status_code != 400 or "temperature" not in payload:
        return False
    try:
        body = response.read()
    except Exception:  # pragma: no cover - closed/consumed stream edge
        return False
    return b"temperature" in body.lower()


_RETRY_AFTER_MAX_SECONDS = 60.0


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    # Cap at 60 s so a hostile upstream can't pin a worker thread to sleep
    # for hours via a giant ``Retry-After`` header. Real provider 429s
    # almost never exceed this; if they do we'd rather fail fast and let
    # the caller decide than block the request queue.
    return min(max(0.0, parsed), _RETRY_AFTER_MAX_SECONDS)
