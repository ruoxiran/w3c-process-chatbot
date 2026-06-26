"""Tests for user-supplied per-request LLM provider overrides.

Three guarantees we lock in here:

1. The SSRF guard rejects loopback / private / link-local / cloud-metadata
   hosts for ``openai-compatible`` overrides, and rejects link-local /
   metadata addresses even for ``ollama`` overrides.
2. When the user supplies an override, the workflow uses ITS endpoint and
   model id, not the server defaults.
3. The ``api_key`` is never written to ``audit`` or to any response field
   that leaves the server.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.core.config import Settings
from app.models.schemas import ChatRequest, Citation, CompiledContext, DraftContext, EvidenceCoverage, ProcessState, ProviderOverride, TaskPlan, W3CEntity
from app.services.provider_override import ProviderOverrideError, build_override_client, validate_provider_override
from app.workflows.chat_workflow import ChatWorkflow


def _dev_settings() -> Settings:
    return Settings(app_env="development", llm_provider="template", w3c_api_enabled=False)


def _prod_settings() -> Settings:
    return Settings(app_env="production", llm_provider="template", w3c_api_enabled=False)


# ---------- SSRF guard ----------------------------------------------------


@pytest.mark.parametrize(
    "kind, url",
    [
        ("openai-compatible", "http://169.254.169.254/latest/meta-data"),
        ("openai-compatible", "http://localhost:8080/v1"),
        ("openai-compatible", "http://127.0.0.1/v1"),
        ("openai-compatible", "http://10.0.0.5/v1"),
        ("openai-compatible", "http://internal-api.corp.internal/v1"),
        ("openai-compatible", "http://something.local/v1"),
        ("ollama", "http://169.254.169.254/"),
        ("ollama", "http://something.internal/"),
    ],
)
def test_override_rejects_unsafe_hosts(kind: str, url: str) -> None:
    override = ProviderOverride(kind=kind, base_url=url, api_key="x" if kind == "openai-compatible" else None, model="m")
    with pytest.raises(ProviderOverrideError):
        validate_provider_override(override, _dev_settings())


def test_override_allows_public_openai_compatible_in_dev() -> None:
    override = ProviderOverride(
        kind="openai-compatible",
        base_url="https://api.openai.com/v1",
        api_key="sk-test",
        model="gpt-4.1",
    )
    validate_provider_override(override, _dev_settings())  # no exception


def test_override_allows_localhost_ollama_in_dev() -> None:
    override = ProviderOverride(kind="ollama", base_url="http://localhost:11434", model="qwen3:8b")
    validate_provider_override(override, _dev_settings())  # no exception


def test_override_rejects_ollama_kind_when_disabled() -> None:
    """Operators on public infra can shut Ollama overrides off entirely
    to close the residual DNS-rebinding gap."""
    settings = Settings(
        app_env="development",
        llm_provider="template",
        w3c_api_enabled=False,
        provider_override_allow_ollama=False,
    )
    override = ProviderOverride(
        kind="ollama", base_url="http://localhost:11434", model="qwen3:8b"
    )
    with pytest.raises(ProviderOverrideError, match="ollama .* disabled"):
        validate_provider_override(override, settings)


def test_override_requires_https_in_production() -> None:
    override = ProviderOverride(
        kind="openai-compatible",
        base_url="http://api.openai.com/v1",
        api_key="sk-test",
        model="gpt-4.1",
    )
    with pytest.raises(ProviderOverrideError):
        validate_provider_override(override, _prod_settings())


def test_build_client_returns_correct_type() -> None:
    from app.services.ollama import OllamaClient
    from app.services.openai_compatible import OpenAICompatibleClient

    ollama_client = build_override_client(
        ProviderOverride(kind="ollama", base_url="http://localhost:11434", model="qwen3:8b"),
        _dev_settings(),
    )
    assert isinstance(ollama_client, OllamaClient)

    openai_client = build_override_client(
        ProviderOverride(
            kind="openai-compatible",
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            model="gpt-4.1",
        ),
        _dev_settings(),
    )
    assert isinstance(openai_client, OpenAICompatibleClient)


# ---------- Workflow uses the override, not the server default -----------


@dataclass
class _GenerationCapture:
    """Records the kwargs the workflow passes to ``generate_answer``.

    A real fake client would also need ``list_models``, but the workflow
    never calls that on the per-request override path.
    """

    captured_model: str | None = None
    captured_history_len: int = -1

    def generate_answer(self, *, model, history, **_kwargs):
        self.captured_model = model
        self.captured_history_len = len(history)
        from app.services.openai_compatible import OpenAICompatibleGeneration

        return OpenAICompatibleGeneration(text="STUB ANSWER FROM OVERRIDE", model=model)


def test_workflow_uses_provider_override(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _dev_settings()
    workflow = ChatWorkflow(settings)
    capture = _GenerationCapture()

    # Replace the override-client builder so we get the spy back instead of a
    # real HTTP client. This is the seam that tests the override branch
    # without making real network calls.
    monkeypatch.setattr(
        "app.workflows.chat_workflow.build_override_client",
        lambda override, settings: capture,
    )

    response = workflow.run(
        ChatRequest(
            message="What does Staff Contact do?",
            provider_override=ProviderOverride(
                kind="openai-compatible",
                base_url="https://api.example.com/v1",
                api_key="sk-not-stored",
                model="my-custom-model",
            ),
        )
    )

    assert capture.captured_model == "my-custom-model"
    assert response.audit["model_provider_source"] == "override"
    assert response.audit["llm_provider"] == "override:openai-compatible"
    assert response.audit["llm_model"] == "my-custom-model"


def test_workflow_rejects_unsafe_override_without_crashing() -> None:
    """Bad override falls back to template, never raises out of run()."""
    workflow = ChatWorkflow(_dev_settings())
    response = workflow.run(
        ChatRequest(
            message="What does Staff Contact do?",
            provider_override=ProviderOverride(
                kind="openai-compatible",
                base_url="http://localhost:8080/v1",  # private — must be rejected
                api_key="sk-leak",
                model="evil",
            ),
        )
    )
    assert response.audit["model_generation"] == "override_rejected"
    assert response.audit["model_error"] == "ProviderOverrideError"


# ---------- API key never leaks ------------------------------------------


def test_api_key_never_appears_in_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """A response that touched a user api_key must not contain that string.

    This is the load-bearing privacy guarantee for BYO providers: the user
    accepts that the key transits our server, but they should be able to
    trust that nothing on the way back out carries it.
    """
    settings = _dev_settings()
    workflow = ChatWorkflow(settings)
    capture = _GenerationCapture()
    monkeypatch.setattr(
        "app.workflows.chat_workflow.build_override_client",
        lambda override, settings: capture,
    )

    secret = "sk-do-not-leak-this-anywhere-1234567890"
    response = workflow.run(
        ChatRequest(
            message="What does Staff Contact do?",
            provider_override=ProviderOverride(
                kind="openai-compatible",
                base_url="https://api.example.com/v1",
                api_key=secret,
                model="my-custom-model",
            ),
        )
    )

    serialized = response.model_dump_json()
    assert secret not in serialized
