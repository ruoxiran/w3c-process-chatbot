import json
import logging
import re
import secrets
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, StreamingResponse

from app.core.config import Settings, get_settings
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


logger = logging.getLogger(__name__)

settings = get_settings()

# Endpoints that bypass the API-key gate.
_AUTH_EXEMPT_PATHS = frozenset({"/health"})


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Constant-time API-key gate on every endpoint except ``/health``.

    Behaviour by configuration:

    - ``settings.api_key`` set → requests must send ``X-API-Key`` matching it
      or get a 401. Missing header also returns 401.
    - ``settings.api_key`` is ``None`` AND ``app_env`` is "development" → gate
      is open, lets local development work without configuration.
    - ``settings.api_key`` is ``None`` AND ``app_env`` is anything else → fail
      closed; the deployment is misconfigured.
    """

    def __init__(self, app, settings: Settings) -> None:
        super().__init__(app)
        self._settings = settings

    async def dispatch(self, request: Request, call_next):
        if request.url.path in _AUTH_EXEMPT_PATHS:
            return await call_next(request)
        expected = self._settings.api_key
        if not expected:
            if self._settings.app_env == "development":
                return await call_next(request)
            return JSONResponse({"detail": "API key not configured"}, status_code=503)
        provided = request.headers.get("X-API-Key", "")
        if not provided or not secrets.compare_digest(provided, expected):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        return await call_next(request)


# Limit by API-key when present (so different keys get independent buckets);
# otherwise fall back to remote address. This avoids collapsing all unauth dev
# traffic into one bucket while still defending against abuse.
def _rate_limit_key(request: Request) -> str:
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"key:{api_key}"
    return get_remote_address(request)


limiter = Limiter(key_func=_rate_limit_key, default_limits=[settings.rate_limit_default])


app = FastAPI(
    title="W3C Process Chatbot API",
    version="0.1.0",
    docs_url="/docs" if settings.expose_openapi_docs else None,
    redoc_url="/redoc" if settings.expose_openapi_docs else None,
    openapi_url="/openapi.json" if settings.expose_openapi_docs else None,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_handler := lambda r, e: JSONResponse(
    {"detail": "Rate limit exceeded. Please retry shortly."}, status_code=429
))

# Order matters: API-key auth runs FIRST, then rate limiting, then CORS.
# Starlette processes middleware in reverse-add order, so add CORS last.
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(APIKeyMiddleware, settings=settings)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=settings.cors_methods,
    allow_headers=settings.cors_headers,
)


@app.on_event("startup")
def _warm_retriever_caches() -> None:
    """Load the corpus index and (when enabled) the dense-embedding cache at
    startup so the first user request doesn't pay the 5-10 second cold-load.

    Each load is idempotent under the retriever's double-checked locking, so
    this is safe even if reload picks up a corpus refresh later.
    """
    try:
        retriever = _workflow_singleton().retriever
        retriever._load_index()  # noqa: SLF001 — intentional pre-warm
        if settings.retrieval_dense_enabled:
            retriever._load_dense_cache()  # noqa: SLF001 — intentional pre-warm
        logger.info("retriever caches pre-warmed at startup")
    except Exception as exc:  # pragma: no cover - startup best effort
        logger.warning("retriever pre-warm failed; first request will pay the cost", exc_info=exc)

    # Optional: pre-load the cross-encoder reranker so the first user
    # request doesn't pay the ~3 s model-load tax. Skipped silently if
    # sentence-transformers isn't installed.
    if settings.reranker_cross_encoder_enabled:
        try:
            from app.services.cross_encoder_reranker import _load_model  # noqa: SLF001 — intentional

            _load_model(settings.reranker_model)
            logger.info("cross-encoder reranker pre-warmed at startup")
        except Exception as exc:  # pragma: no cover - startup best effort
            logger.info("cross-encoder pre-warm skipped: %s", exc)


@lru_cache
def _workflow_singleton() -> ChatWorkflow:
    """Long-lived workflow instance so retriever/W3C API caches survive across requests."""
    return ChatWorkflow(settings)


def workflow() -> ChatWorkflow:
    return _workflow_singleton()


@lru_cache
def _compiled_store_singleton() -> CompiledContextStore:
    return CompiledContextStore(settings)


def compiled_store() -> CompiledContextStore:
    return _compiled_store_singleton()


@lru_cache
def _feedback_store_singleton() -> FeedbackStore:
    return FeedbackStore(settings.feedback_log_path)


def feedback_store() -> FeedbackStore:
    return _feedback_store_singleton()


def _strip_audit_if_disabled(response: ChatResponse) -> ChatResponse:
    """Hide the audit blob from clients unless ``expose_audit`` is on.

    The audit blob contains internal routing/retrieval state (provider, model,
    task plan, evidence coverage, router decision, live-fetch URL). Useful in
    dev for inspecting the workflow; in production it leaks internal architecture
    and whether prompt-injection was suspected. Strip by default.
    """
    if settings.expose_audit:
        return response
    return response.model_copy(update={"audit": {}})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(settings.rate_limit_chat)
def chat(request: Request, body: ChatRequest, wf: ChatWorkflow = Depends(workflow)) -> ChatResponse:
    return _strip_audit_if_disabled(wf.run(body))


def _sse_event(event_type: str, payload: object) -> bytes:
    """Pack an SSE event with explicit ``event:`` + JSON ``data:`` line.

    Browsers can read these via ``EventSource`` or any fetch-based reader.
    """
    body = json.dumps(payload, ensure_ascii=False, default=str)
    return f"event: {event_type}\ndata: {body}\n\n".encode("utf-8")


def _stream_chat_events(wf: ChatWorkflow, body: ChatRequest, expose_audit: bool):
    """Yield SSE events for a single chat request via real token streaming.

    The workflow runs in a worker thread; this generator drains its delta
    queue and forwards each LLM token chunk as a ``delta`` SSE event as it
    arrives. Once the worker thread finishes, the final ``ChatResponse`` is
    emitted as a ``meta`` event (everything except the answer body) followed
    by a ``done`` event with the full answer text.

    The pre-LLM phase (retrieval, entity resolution, etc.) still blocks
    before the first token arrives — but the moment the LLM starts
    producing text, the user sees it. ``meta`` lands at the end with the
    fully-populated workflow trace so the inspector renders in one shot.
    """
    final_response: ChatResponse | None = None
    error_message: str | None = None
    for event in wf.run_stream(body):
        if event["type"] == "delta":
            yield _sse_event("delta", {"text": event["text"]})
        elif event["type"] == "stage":
            # Pre-LLM workflow stages arrive as WorkflowStep instances;
            # serialise via Pydantic so the UI sees the same shape it
            # already knows from the non-streaming /chat response.
            step = event["step"]
            yield _sse_event("stage", step.model_dump(mode="json"))
        elif event["type"] == "response":
            final_response = event["response"]
        elif event["type"] == "error":
            error_message = event.get("message") or "workflow failed"
            break

    if error_message is not None or final_response is None:
        yield _sse_event(
            "error",
            {"message": error_message or "workflow ended without a response"},
        )
        return

    payload_response = (
        final_response if expose_audit else _strip_audit_if_disabled(final_response)
    )
    meta = payload_response.model_dump(mode="json")
    answer_text = meta.pop("answer", "") or ""
    yield _sse_event("meta", meta)
    yield _sse_event("done", {"answer": answer_text})


@app.post("/chat/stream")
@limiter.limit(settings.rate_limit_chat)
def chat_stream(request: Request, body: ChatRequest, wf: ChatWorkflow = Depends(workflow)) -> StreamingResponse:
    return StreamingResponse(
        _stream_chat_events(wf, body, expose_audit=settings.expose_audit),
        media_type="text/event-stream",
        headers={
            # Prevent intermediate proxies and the browser from buffering the
            # SSE response — without this Nginx / Cloudflare can hold the
            # whole stream until completion, defeating the point.
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/classify", response_model=ClassifyResponse)
def classify(request: Request, body: ClassifyRequest, wf: ChatWorkflow = Depends(workflow)) -> ClassifyResponse:
    return wf.classify(ChatRequest(message=body.message))


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
            logger.warning("ollama list_models failed", exc_info=exc)
            return ModelsResponse(default_model=settings.llm_model, models=[], error=type(exc).__name__)
    if settings.llm_provider in {"openai", "openai-compatible", "openrouter"}:
        try:
            online_models = OpenAICompatibleClient(
                settings.openai_compatible_base_url,
                settings.openai_compatible_api_key,
                settings.openai_compatible_timeout_seconds,
            ).list_models()
            return ModelsResponse(default_model=settings.openai_compatible_model, models=online_models)
        except Exception as exc:  # pragma: no cover - external service fallback
            logger.warning("openai-compatible list_models failed", exc_info=exc)
            return ModelsResponse(
                default_model=settings.openai_compatible_model,
                models=[
                    ModelInfo(
                        name=settings.openai_compatible_model,
                        provider="openai-compatible",
                        is_embedding=False,
                    )
                ],
                error=type(exc).__name__,
            )
    return ModelsResponse(default_model=settings.llm_model, models=[])


@app.post("/refresh-index")
@limiter.limit(settings.rate_limit_admin)
async def refresh_index(request: Request) -> dict[str, object]:
    counts = await build_preview_index()
    compiled = compiled_store().rebuild_known()
    return {
        "status": "preview-index-built",
        "chunk_counts": counts,
        "compiled_contexts_rebuilt": [context.key for context in compiled],
    }


_SHORTNAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,80}$")


@app.post("/compiled/rebuild")
@limiter.limit(settings.rate_limit_admin)
def rebuild_compiled(request: Request, shortnames: str | None = None) -> dict[str, object]:
    raw = [item.strip() for item in (shortnames or "").split(",") if item.strip()]
    if raw and len(raw) > 50:
        raise HTTPException(status_code=400, detail="too many shortnames; max 50")
    for token in raw:
        if not _SHORTNAME_RE.match(token):
            raise HTTPException(status_code=400, detail=f"invalid shortname: {token!r}")
    targets = raw or None
    results = compiled_store().rebuild_known(targets)
    return {"status": "compiled-rebuilt", "count": len(results), "keys": [item.key for item in results]}


@app.get("/compiled/status", response_model=CompiledStatusResponse)
@limiter.limit(settings.rate_limit_admin)
def compiled_status(request: Request) -> CompiledStatusResponse:
    return CompiledStatusResponse(enabled=settings.compiled_context_enabled, items=compiled_store().status())


@app.post("/eval/run", response_model=EvalRunResponse)
@limiter.limit(settings.rate_limit_eval)
def run_eval(request: Request, include_adversarial: bool = False) -> EvalRunResponse:
    wf = build_eval_workflow(settings)
    cases = [*EVAL_CASES, *ADVERSARIAL_CASES] if include_adversarial else EVAL_CASES
    return run_eval_cases(cases, wf.run)


@app.post("/eval/llm-judge", response_model=LLMJudgeReportResponse)
@limiter.limit(settings.rate_limit_judge)
def run_llm_judge_endpoint(
    request: Request,
    include_adversarial: bool = True,
    judge_model: str | None = Query(
        default=None,
        # Same constraint as ChatRequest.model — must be a plausible model
        # identifier, NOT free-form text. Without this, anyone who can hit
        # /eval/llm-judge can probe model names against the configured
        # backend or send arbitrary strings through the LLM gateway.
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,80}$",
        max_length=81,
    ),
) -> LLMJudgeReportResponse:
    """Run the live LLM-backed workflow and score each answer with a judge LLM.

    This bypasses the deterministic ``build_eval_workflow`` because we want to
    score the actual LLM output, not the template fallback.
    """
    cases = [*EVAL_CASES, *ADVERSARIAL_CASES] if include_adversarial else EVAL_CASES
    report = run_llm_judge(
        cases=cases,
        workflow=workflow(),
        settings=settings,
        judge_model=judge_model,
    )
    return LLMJudgeReportResponse(**report.to_dict())


@app.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    # Whitelist fields explicitly. Client-supplied ``audit`` is intentionally
    # NOT persisted: it is an unbounded dict[Any] surface that would otherwise
    # let any caller pollute the JSONL with arbitrary nested structures.
    record = {
        "rating": request.rating,
        "conversation_id": request.conversation_id,
        "message_id": request.message_id,
        "question": request.question,
        "answer": request.answer,
        "comment": request.comment,
        "model": request.model,
        "in_scope": request.in_scope,
        "confidence": request.confidence,
        "citation_urls": [str(u) for u in request.citation_urls],
    }
    stored = feedback_store().append(record)
    return FeedbackResponse(received_at=stored["received_at"])


@app.get("/feedback/stats", response_model=FeedbackStatsResponse)
@limiter.limit(settings.rate_limit_admin)
def feedback_stats(request: Request) -> FeedbackStatsResponse:
    return FeedbackStatsResponse(**feedback_store().stats())
