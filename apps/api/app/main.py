from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.evals.adversarial_cases import ADVERSARIAL_CASES
from app.evals.cases import EVAL_CASES
from app.evals.llm_judge import run_llm_judge
from app.evals.runner import run_eval_cases
from app.evals.workflow import build_eval_workflow
from app.ingestion.indexer import build_preview_index
from app.ingestion.sources import AUTHORITATIVE_SOURCES
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    ClassifyRequest,
    ClassifyResponse,
    CompiledStatusResponse,
    EvalRunResponse,
    FeedbackRequest,
    FeedbackResponse,
    FeedbackStatsResponse,
    LLMJudgeReportResponse,
    ModelInfo,
    ModelsResponse,
    SourceStatusResponse,
    SourceVersion,
)
from app.services.compiled_context import CompiledContextStore
from app.services.feedback import FeedbackStore
from app.services.ollama import OllamaClient
from app.services.openai_compatible import OpenAICompatibleClient
from app.workflows.chat_workflow import ChatWorkflow

settings = get_settings()
app = FastAPI(title="W3C Process Chatbot API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def workflow() -> ChatWorkflow:
    return ChatWorkflow(settings)


def compiled_store() -> CompiledContextStore:
    return CompiledContextStore(settings)


def feedback_store() -> FeedbackStore:
    return FeedbackStore(settings.feedback_log_path)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return workflow().run(request)


@app.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest) -> ClassifyResponse:
    return workflow().classify(ChatRequest(message=request.message))


@app.get("/sources/status", response_model=SourceStatusResponse)
def source_status() -> SourceStatusResponse:
    return SourceStatusResponse(
        process_url=AUTHORITATIVE_SOURCES[0].url,
        guide_url=AUTHORITATIVE_SOURCES[1].url,
        process_repo=AUTHORITATIVE_SOURCES[0].repo or "",
        guide_repo=AUTHORITATIVE_SOURCES[1].repo or "",
        source_version=SourceVersion(),
        allowlist=settings.allowlist_entries,
        member_only_enabled=settings.enable_member_only_sources,
    )


@app.get("/models", response_model=ModelsResponse)
def models() -> ModelsResponse:
    if settings.llm_provider == "ollama":
        try:
            ollama_models = OllamaClient(
                settings.ollama_base_url,
                settings.ollama_timeout_seconds,
            ).list_models()
            return ModelsResponse(default_model=settings.llm_model, models=ollama_models)
        except Exception as exc:  # pragma: no cover - external service fallback
            return ModelsResponse(default_model=settings.llm_model, models=[], error=str(exc))
    if settings.llm_provider in {"openai", "openai-compatible", "openrouter"}:
        try:
            online_models = OpenAICompatibleClient(
                settings.openai_compatible_base_url,
                settings.openai_compatible_api_key,
                settings.openai_compatible_timeout_seconds,
            ).list_models()
            return ModelsResponse(default_model=settings.openai_compatible_model, models=online_models)
        except Exception as exc:  # pragma: no cover - external service fallback
            return ModelsResponse(
                default_model=settings.openai_compatible_model,
                models=[
                    ModelInfo(
                        name=settings.openai_compatible_model,
                        provider="openai-compatible",
                        is_embedding=False,
                    )
                ],
                error=str(exc),
            )
    return ModelsResponse(default_model=settings.llm_model, models=[])


@app.post("/refresh-index")
async def refresh_index() -> dict[str, object]:
    counts = await build_preview_index()
    compiled = compiled_store().rebuild_known()
    return {
        "status": "preview-index-built",
        "chunk_counts": counts,
        "compiled_contexts_rebuilt": [context.key for context in compiled],
    }


@app.post("/compiled/rebuild")
def rebuild_compiled(shortnames: str | None = None) -> dict[str, object]:
    targets = [item.strip() for item in (shortnames or "").split(",") if item.strip()] or None
    results = compiled_store().rebuild_known(targets)
    return {"status": "compiled-rebuilt", "count": len(results), "keys": [item.key for item in results]}


@app.get("/compiled/status", response_model=CompiledStatusResponse)
def compiled_status() -> CompiledStatusResponse:
    return CompiledStatusResponse(enabled=settings.compiled_context_enabled, items=compiled_store().status())


@app.post("/eval/run", response_model=EvalRunResponse)
def run_eval(include_adversarial: bool = False) -> EvalRunResponse:
    wf = build_eval_workflow(settings)
    cases = [*EVAL_CASES, *ADVERSARIAL_CASES] if include_adversarial else EVAL_CASES
    return run_eval_cases(cases, wf.run)


@app.post("/eval/llm-judge", response_model=LLMJudgeReportResponse)
def run_llm_judge_endpoint(
    include_adversarial: bool = True,
    judge_model: str | None = None,
) -> LLMJudgeReportResponse:
    """Run the live LLM-backed workflow and score each answer with a judge LLM.

    This bypasses the deterministic ``build_eval_workflow`` because we want to
    score the actual LLM output, not the template fallback.
    """
    cases = [*EVAL_CASES, *ADVERSARIAL_CASES] if include_adversarial else EVAL_CASES
    workflow_instance = ChatWorkflow(settings)
    report = run_llm_judge(
        cases=cases,
        workflow=workflow_instance,
        settings=settings,
        judge_model=judge_model,
    )
    return LLMJudgeReportResponse(**report.to_dict())


@app.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    stored = feedback_store().append(request.model_dump(mode="json"))
    return FeedbackResponse(received_at=stored["received_at"])


@app.get("/feedback/stats", response_model=FeedbackStatsResponse)
def feedback_stats() -> FeedbackStatsResponse:
    return FeedbackStatsResponse(**feedback_store().stats())
