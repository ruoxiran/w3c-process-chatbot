"""Tests for the AWS Bedrock provider.

Two layers:

1. ``BedrockClient`` correctly drives the boto3 ``bedrock-runtime`` Converse /
   ConverseStream API and parses their response shapes — verified against a
   fake runtime so no AWS round-trip happens.
2. The workflow routes generation to the Bedrock client when
   ``llm_provider="bedrock"`` and uses the configured Bedrock model + the
   lighter prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from app.core.config import Settings
from app.models.schemas import ChatRequest
from app.services.bedrock import BEDROCK_MODEL_IDS, BedrockClient, BedrockGeneration, bedrock_model_infos
from app.workflows.chat_workflow import ChatWorkflow


# ---------- Fake bedrock-runtime -----------------------------------------


@dataclass
class _FakeRuntime:
    """Stands in for a boto3 ``bedrock-runtime`` client."""

    converse_text: str = '{"ok": true}'
    stream_deltas: list[str] = field(default_factory=list)
    converse_calls: list[dict] = field(default_factory=list)
    stream_calls: list[dict] = field(default_factory=list)

    def converse(self, **kwargs):
        self.converse_calls.append(kwargs)
        return {"output": {"message": {"content": [{"text": self.converse_text}]}}}

    def converse_stream(self, **kwargs):
        self.stream_calls.append(kwargs)
        events = [{"contentBlockDelta": {"delta": {"text": d}}} for d in self.stream_deltas]
        events.append({"messageStop": {"stopReason": "end_turn"}})
        return {"stream": events}


def _client_with_runtime(runtime: _FakeRuntime) -> BedrockClient:
    client = BedrockClient("us-east-1")
    # Inject the fake so ``_client()`` returns it instead of building a real one.
    client._runtime = runtime
    return client


def _answer_kwargs(model: str = "anthropic.test-model") -> dict:
    return dict(
        model=model,
        question="What does the Staff Contact do?",
        locale="en",
        citations=[],
        fallback_answer="Fallback answer.",
        fallback_next_steps=[],
    )


# ---------- Client-level parsing -----------------------------------------


def test_generate_answer_parses_converse_output() -> None:
    runtime = _FakeRuntime(converse_text="The Staff Contact coordinates with the team.")
    client = _client_with_runtime(runtime)

    result = client.generate_answer(**_answer_kwargs())

    assert isinstance(result, BedrockGeneration)
    assert result.text == "The Staff Contact coordinates with the team."
    assert result.model == "anthropic.test-model"
    # The workflow's system-grounded prompt is passed as the single user turn.
    call = runtime.converse_calls[0]
    assert call["modelId"] == "anthropic.test-model"
    assert call["messages"][0]["role"] == "user"
    assert "system" in call


def test_generate_answer_strips_thinking() -> None:
    runtime = _FakeRuntime(converse_text="<think>secret</think>Public answer.")
    client = _client_with_runtime(runtime)

    result = client.generate_answer(**_answer_kwargs())

    assert "secret" not in result.text
    assert "Public answer." in result.text


def test_generate_json_parses_object() -> None:
    runtime = _FakeRuntime(converse_text='{"likely_in_scope": true, "confidence": 0.8}')
    client = _client_with_runtime(runtime)

    payload = client.generate_json(model="anthropic.test-model", prompt="classify this")

    assert payload == {"likely_in_scope": True, "confidence": 0.8}
    assert runtime.converse_calls[0]["inferenceConfig"]["temperature"] == 0


def test_stream_answer_yields_content_deltas() -> None:
    runtime = _FakeRuntime(stream_deltas=["Hello ", "from ", "Bedrock"])
    client = _client_with_runtime(runtime)

    chunks = list(client.stream_answer(**_answer_kwargs()))

    assert chunks == ["Hello ", "from ", "Bedrock"]
    assert runtime.stream_calls[0]["modelId"] == "anthropic.test-model"


def test_list_models_is_empty() -> None:
    # The runtime client can't enumerate; /models uses bedrock_model_infos().
    assert BedrockClient("us-east-1").list_models() == []


def test_api_key_sets_bearer_token_env(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    client = BedrockClient("us-east-1", api_key="sk-bedrock-test-123")
    # Building the boto3 client applies the key via the bearer-token env var.
    client._client()
    assert os.environ["AWS_BEARER_TOKEN_BEDROCK"] == "sk-bedrock-test-123"


_ALLOWED_BEDROCK_PROVIDERS = {"amazon", "anthropic", "qwen", "deepseek", "openai"}


def test_bedrock_model_catalogue() -> None:
    infos = bedrock_model_infos()
    assert len(infos) == len(BEDROCK_MODEL_IDS)
    assert all(info.provider == "bedrock" for info in infos)
    # No embeddings/non-chat models in the curated list.
    assert not any(info.is_embedding for info in infos)
    # Limited to the five allowed model providers.
    assert {name.split(".", 1)[0] for name in BEDROCK_MODEL_IDS} == _ALLOWED_BEDROCK_PROVIDERS
    # The recommended default is present; excluded families/modalities are not.
    assert "anthropic.claude-sonnet-5" in BEDROCK_MODEL_IDS
    for excluded in (
        "amazon.nova-canvas-v1:0",       # image
        "amazon.nova-reel-v1:0",         # video
        "amazon.titan-embed-text-v1",    # embeddings
        "amazon.rerank-v1:0",            # rerank
        "qwen.qwen3-coder-next",         # coding-specialised
        "google.gemma-4-31b",            # provider not allowed
        "meta.llama4-scout-17b-instruct-v1:0",  # provider not allowed
    ):
        assert excluded not in BEDROCK_MODEL_IDS


class _TemperatureRejectingRuntime:
    """Rejects any converse call that includes ``temperature`` (like the
    newest Claude models on Bedrock), succeeds once it's dropped."""

    def __init__(self) -> None:
        self.attempts: list[dict] = []

    def converse(self, **kwargs):
        self.attempts.append(kwargs["inferenceConfig"])
        if "temperature" in kwargs["inferenceConfig"]:
            raise RuntimeError(
                "An error occurred (ValidationException): temperature is not supported"
            )
        return {"output": {"message": {"content": [{"text": "recovered answer"}]}}}


def test_generate_answer_retries_without_temperature() -> None:
    runtime = _TemperatureRejectingRuntime()
    client = BedrockClient("us-east-1")
    client._runtime = runtime

    result = client.generate_answer(**_answer_kwargs())

    assert result.text == "recovered answer"
    # First attempt carried temperature, retry dropped it.
    assert "temperature" in runtime.attempts[0]
    assert "temperature" not in runtime.attempts[1]

    # The rejection is cached: a second call skips temperature entirely (one
    # request, not two).
    runtime.attempts.clear()
    client.generate_answer(**_answer_kwargs())
    assert len(runtime.attempts) == 1
    assert "temperature" not in runtime.attempts[0]


def test_generate_answer_uses_configured_budget() -> None:
    runtime = _FakeRuntime(converse_text="ok")
    client = _client_with_runtime(runtime)
    client.max_answer_tokens = 4096

    client.generate_answer(**_answer_kwargs())

    assert runtime.converse_calls[0]["inferenceConfig"]["maxTokens"] == 4096


def test_non_sampling_error_is_not_swallowed() -> None:
    class _BoomRuntime:
        def converse(self, **kwargs):
            raise RuntimeError("An error occurred (AccessDeniedException): explicit deny")

    client = BedrockClient("us-east-1")
    client._runtime = _BoomRuntime()
    with pytest.raises(RuntimeError, match="AccessDeniedException"):
        client.generate_answer(**_answer_kwargs())


# ---------- Workflow routing ---------------------------------------------


class _SpyBedrock:
    """Fake matching the informal client interface used by the workflow."""

    def __init__(self) -> None:
        self.captured_model: str | None = None
        self.captured_lighter: bool | None = None

    def generate_answer(self, *, model, lighter_mode=False, **_kwargs):
        self.captured_model = model
        self.captured_lighter = lighter_mode
        return BedrockGeneration(text="STUB BEDROCK ANSWER", model=model)

    def generate_json(self, *, model, prompt, num_predict=500):
        return {}


def test_workflow_routes_to_bedrock() -> None:
    settings = Settings(
        app_env="development",
        llm_provider="bedrock",
        llm_model="us.anthropic.claude-workflow-test",
        w3c_api_enabled=False,
        llm_router_enabled=False,
        hyde_enabled=False,
    )
    spy = _SpyBedrock()
    workflow = ChatWorkflow(settings, bedrock_client=spy)

    response = workflow.run(ChatRequest(message="What does the Staff Contact do?"))

    assert spy.captured_model == "us.anthropic.claude-workflow-test"
    # Bedrock models get the lighter formatting prompt, like the external API.
    assert spy.captured_lighter is True
    assert response.audit["model_generation"] == "bedrock"
    assert response.audit["model_provider_source"] == "default"
    assert response.notice is None  # healthy generation → no degradation notice


class _FailingBedrock:
    """A client whose generation always fails (e.g. AccessDeniedException)."""

    def generate_answer(self, *, model, **_kwargs):
        raise RuntimeError("AccessDeniedException: explicit deny in a service control policy")

    def generate_json(self, *, model, prompt, num_predict=500):
        return {}


def test_workflow_surfaces_notice_when_generation_fails() -> None:
    settings = Settings(
        app_env="development",
        llm_provider="bedrock",
        llm_model="us.anthropic.claude-sonnet-5",
        w3c_api_enabled=False,
        llm_router_enabled=False,
        hyde_enabled=False,
    )
    workflow = ChatWorkflow(settings, bedrock_client=_FailingBedrock())

    response = workflow.run(ChatRequest(message="What does the Staff Contact do?"))

    assert response.audit["model_generation"] == "template_fallback"
    assert response.notice is not None
    # The exception type is surfaced so the UI can explain the failure.
    assert "RuntimeError" in response.notice
    assert "limited answer" in response.notice
