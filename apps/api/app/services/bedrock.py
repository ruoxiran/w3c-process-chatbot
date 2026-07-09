from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass

from app.models.schemas import (
    ChatTurn,
    Citation,
    CompiledContext,
    DraftContext,
    EvidenceCoverage,
    ModelInfo,
    ProcessState,
    TaskPlan,
    W3CEntity,
)
from app.services.ollama import _clean_model_text, _extract_json_object, build_prompt


logger = logging.getLogger(__name__)


# Selectable Bedrock model ids surfaced by GET /models for the UI dropdown.
# Curated for THIS project: general text-generation chat models suitable for
# grounded, cited RAG synthesis (strong instruction-following, multilingual —
# the bot answers in the user's locale incl. CJK). Limited to the amazon /
# anthropic / qwen / deepseek / openai providers, and deliberately excludes
# non-chat modalities (image / video / audio / embeddings / rerank) and
# task-specialised variants (coder, vision, safety-classifier) that don't fit
# a policy Q&A assistant. This is a static catalogue, not a live account query;
# override LLM_MODEL for whatever your account+region actually grants.
BEDROCK_MODEL_IDS: tuple[str, ...] = (
    # Anthropic — best instruction-following / grounded synthesis for this task.
    "anthropic.claude-sonnet-5",
    "anthropic.claude-opus-4-8",
    "anthropic.claude-haiku-4-5-20251001-v1:0",
    # Amazon Nova — text-generation tiers (premier → micro).
    "amazon.nova-premier-v1:0",
    "amazon.nova-pro-v1:0",
    "amazon.nova-lite-v1:0",
    "amazon.nova-micro-v1:0",
    # Qwen — strong multilingual (incl. Chinese, which this bot supports).
    "qwen.qwen3-235b-a22b-2507-v1:0",
    "qwen.qwen3-32b-v1:0",
    # DeepSeek — general + reasoning.
    "deepseek.v3.2",
    "deepseek.r1-v1:0",
    # OpenAI.
    "openai.gpt-5.4",
    "openai.gpt-oss-120b-1:0",
)


def bedrock_model_infos() -> list[ModelInfo]:
    """The curated selectable Bedrock chat models for GET /models."""
    return [ModelInfo(name=model_id, provider="bedrock") for model_id in BEDROCK_MODEL_IDS]


# Shared system-role framing, identical to the OpenAI-compatible client so the
# grounding + safety contract is the same regardless of provider.
_SYSTEM_PROMPT = (
    "You are a W3C Process and Patent Policy assistant constrained by source-grounded evidence. "
    "Follow the user's prompt rules exactly and do not reveal hidden reasoning."
)


def _is_sampling_param_error(exc: Exception) -> bool:
    """True when Converse rejected a sampling param (temperature / top_p).

    The newest Claude models (Sonnet 5, Opus 4.7/4.8, Fable 5) reject
    non-default sampling params and surface a ``ValidationException`` naming
    the field. We detect that so we can transparently retry without it,
    instead of falling all the way back to the template answer.
    """
    text = str(exc).lower()
    return "validationexception" in text and (
        "temperature" in text or "top_p" in text or "sampling" in text
    )


@dataclass(frozen=True)
class BedrockGeneration:
    text: str
    model: str


class BedrockClient:
    """Client for AWS Bedrock via the boto3 ``bedrock-runtime`` Converse API.

    Converse is model-agnostic — the same request shape works for Claude, Nova,
    Llama, Titan, and others — so ``model`` is any Bedrock model id the account
    has access to (e.g. ``anthropic.claude-3-5-sonnet-20241022-v2:0``).

    Authenticates with a Bedrock API key (bearer token) via the
    AWS_BEARER_TOKEN_BEDROCK mechanism; the ambient AWS credential chain is not
    used. The boto3 client is built lazily on first use so importing /
    constructing this class never requires boto3 to be installed unless Bedrock
    is the active provider.
    """

    def __init__(
        self,
        region: str,
        api_key: str | None = None,
        timeout_seconds: float = 120,
        max_answer_tokens: int = 4096,
    ) -> None:
        self.region = region
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_answer_tokens = max_answer_tokens
        self._runtime = None
        # Model ids known to reject a supplied ``temperature`` (newest Claude
        # models). Cached after the first rejection so we skip the doomed first
        # call on every subsequent request instead of always paying two.
        self._no_temperature_models: set[str] = set()

    def _client(self):
        """Build (and cache) the boto3 bedrock-runtime client on first use."""
        if self._runtime is None:
            import boto3
            from botocore.config import Config

            # boto3 reads the Bedrock API key from this env var (bearer-token
            # auth); set it from config rather than relying on the ambient env.
            if self.api_key:
                os.environ["AWS_BEARER_TOKEN_BEDROCK"] = self.api_key
            self._runtime = boto3.client(
                "bedrock-runtime",
                region_name=self.region,
                config=Config(
                    read_timeout=self.timeout_seconds,
                    connect_timeout=self.timeout_seconds,
                ),
            )
        return self._runtime

    def list_models(self) -> list:
        # Enumerating foundation models needs the ``bedrock`` control-plane
        # client, which this runtime client doesn't hold. The /models endpoint
        # reports ``settings.llm_model`` directly, so this stays empty; the
        # method exists only for interface parity with the other clients.
        return []

    def generate_json(self, *, model: str, prompt: str, num_predict: int = 500) -> dict[str, object]:
        text = self._converse(
            model=model,
            system=[{"text": "Return only a valid JSON object. Do not include markdown or explanations."}],
            prompt=prompt,
            max_tokens=num_predict,
            temperature=0,
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
    ) -> BedrockGeneration:
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
        text = self._converse(
            model=model,
            system=[{"text": _SYSTEM_PROMPT}],
            prompt=prompt,
            max_tokens=self.max_answer_tokens,
            temperature=0.1,
        )
        return BedrockGeneration(text=_clean_model_text(text), model=model)

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
        """Yield raw text deltas from ConverseStream.

        Caller assembles and post-cleans the final text (``_clean_model_text``
        on the joined result), matching the OpenAI-compatible streaming path.
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
        messages = [{"role": "user", "content": [{"text": prompt}]}]
        system = [{"text": _SYSTEM_PROMPT}]

        def _open(temperature: float | None):
            return self._client().converse_stream(
                modelId=model,
                system=system,
                messages=messages,
                inferenceConfig=self._inference_config(self.max_answer_tokens, temperature),
            )

        temperature = None if model in self._no_temperature_models else 0.1
        try:
            response = _open(temperature)
        except Exception as exc:
            if temperature is None or not _is_sampling_param_error(exc):
                raise
            logger.info("Bedrock model %s rejected temperature; retrying stream without it", model)
            self._no_temperature_models.add(model)
            response = _open(None)
        for event in response.get("stream", []):
            delta = event.get("contentBlockDelta")
            if not isinstance(delta, dict):
                continue
            text = delta.get("delta", {}).get("text")
            if isinstance(text, str) and text:
                yield text

    @staticmethod
    def _inference_config(max_tokens: int, temperature: float | None) -> dict:
        cfg: dict = {"maxTokens": max_tokens}
        if temperature is not None:
            cfg["temperature"] = temperature
        return cfg

    def _converse(
        self,
        *,
        model: str,
        system: list[dict[str, str]],
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        messages = [{"role": "user", "content": [{"text": prompt}]}]

        def _call(temp: float | None):
            return self._client().converse(
                modelId=model,
                system=system,
                messages=messages,
                inferenceConfig=self._inference_config(max_tokens, temp),
            )

        # The newest Claude models reject a supplied ``temperature``. Skip it
        # up front once we've learned this model does; otherwise retry once
        # without it (and remember) rather than dropping to the template.
        effective = None if model in self._no_temperature_models else temperature
        try:
            response = _call(effective)
        except Exception as exc:
            if effective is None or not _is_sampling_param_error(exc):
                raise
            logger.info("Bedrock model %s rejected temperature; retrying without it", model)
            self._no_temperature_models.add(model)
            response = _call(None)
        blocks = response.get("output", {}).get("message", {}).get("content", [])
        if not isinstance(blocks, list):
            return ""
        return "".join(block["text"] for block in blocks if isinstance(block, dict) and isinstance(block.get("text"), str))
