from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl, SecretStr


class SourceType(str, Enum):
    process = "process"
    guide = "guide"
    related_policy = "related_policy"
    repo = "repo"


class Citation(BaseModel):
    title: str
    url: HttpUrl | str
    source_type: SourceType
    section_id: str | None = None
    heading_path: str | None = None
    commit_sha: str | None = None
    published_version_date: str | None = None
    quote: str | None = None


class SourceVersion(BaseModel):
    process_version_date: str | None = None
    process_commit_sha: str | None = None
    guide_commit_sha: str | None = None
    indexed_at: str | None = None


class WorkflowStep(BaseModel):
    id: str
    label: str
    status: str = Field(pattern="^(pending|running|completed|skipped|failed)$")
    detail: str
    references: list[Citation] = Field(default_factory=list)


class NextStep(BaseModel):
    text: str
    source_title: str | None = None
    source_url: HttpUrl | str | None = None
    source_type: SourceType | None = None
    source_heading: str | None = None


class ProcessState(BaseModel):
    intent: str = "unknown"
    current_stage: str | None = None
    target_stage: str | None = None
    group_type: str | None = None
    deliverable_type: str | None = None
    likely_workflow: str = "general_process_guidance"
    missing_information: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1, default=0.5)


class TaskPlan(BaseModel):
    intent_type: str = "explain_process"
    user_goal: str
    current_stage: str | None = None
    target_stage: str | None = None
    spec_or_group: str | None = None
    needed_sources: list[SourceType] = Field(default_factory=list)
    answer_shape: str = "short_conclusion_with_actionable_steps"
    search_queries: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1, default=0.5)


class EvidenceCoverage(BaseModel):
    status: str = Field(pattern="^(sufficient|needs_more_evidence|insufficient)$")
    has_compiled_context: bool = False
    has_process: bool = False
    has_guide: bool = False
    has_entity_status: bool = False
    missing_evidence: list[str] = Field(default_factory=list)
    targeted_queries: list[str] = Field(default_factory=list)
    summary: str
    confidence: float = Field(ge=0, le=1, default=0.5)


class W3CEntity(BaseModel):
    entity_type: str = Field(pattern="^(specification|group)$")
    title: str
    shortname: str | None = None
    api_url: HttpUrl | str
    public_url: HttpUrl | str | None = None
    editor_draft_url: HttpUrl | str | None = None
    status: str | None = None
    latest_version_url: HttpUrl | str | None = None
    latest_version_date: str | None = None
    process_rules_url: HttpUrl | str | None = None
    deliverers: list[str] = Field(default_factory=list)
    charter_url: HttpUrl | str | None = None
    charter_end: str | None = None
    patent_policy_url: HttpUrl | str | None = None
    team_contacts: list[str] = Field(default_factory=list)
    group_type: str | None = None
    description: str | None = None
    retrieval_hints: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1, default=0.5)


class DraftSnippet(BaseModel):
    path: str = Field(max_length=400)
    title: str | None = Field(default=None, max_length=400)
    text: str = Field(max_length=4000)
    url: HttpUrl | str | None = None


class DraftContext(BaseModel):
    repo_full_name: str
    repo_url: HttpUrl | str
    resolved_from: HttpUrl | str | None = None
    default_branch: str | None = None
    description: str | None = None
    homepage: HttpUrl | str | None = None
    latest_commit_sha: str | None = None
    open_issues_count: int | None = None
    snippets: list[DraftSnippet] = Field(default_factory=list)
    retrieval_hints: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1, default=0.5)


class CompiledFreshness(BaseModel):
    compiled_at: str | None = None
    source_snapshot: list[str] = Field(default_factory=list)
    is_stale: bool = False


class CompiledProvenance(BaseModel):
    normative_urls: list[HttpUrl | str] = Field(default_factory=list)
    guide_urls: list[HttpUrl | str] = Field(default_factory=list)
    operational_urls: list[HttpUrl | str] = Field(default_factory=list)


class CompiledContext(BaseModel):
    kind: str = Field(pattern="^(spec)$")
    key: str
    title: str
    summary: str
    current_state: str | None = None
    next_step_candidates: list[str] = Field(default_factory=list)
    guide_signals: list[str] = Field(default_factory=list)
    horizontal_review_signals: list[str] = Field(default_factory=list)
    charter_signals: list[str] = Field(default_factory=list)
    freshness: CompiledFreshness = Field(default_factory=CompiledFreshness)
    provenance: CompiledProvenance = Field(default_factory=CompiledProvenance)
    source_path: str | None = None
    confidence: float = Field(ge=0, le=1, default=0.5)


class ChatTurn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=4000)


class ProviderOverride(BaseModel):
    """User-supplied LLM provider used for THIS request only.

    The api_key is held as a Pydantic SecretStr so it does not leak into
    log lines, repr, or model_dump() output by default. The workflow uses
    it once to build a per-request client and then it is discarded — it is
    never written to the audit dict, the feedback log, or the workflow
    cache.
    """

    kind: Literal["openai-compatible", "ollama"]
    base_url: HttpUrl
    api_key: SecretStr | None = None
    model: str = Field(min_length=1, max_length=120, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,118}$")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    locale: str = Field(default="auto", pattern=r"^[A-Za-z]{2,3}(?:[-_][A-Za-z0-9]{2,8})?$|^auto$", max_length=16)
    conversation_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")
    user_role: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,40}$")
    model: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,80}$")
    history: list[ChatTurn] = Field(default_factory=list, max_length=12)
    provider_override: ProviderOverride | None = None
    # Server-side provider selection. Unlike ``provider_override`` (which
    # carries a user-supplied endpoint + key from the browser), this just
    # names one of the server's own configured providers — no secret ever
    # leaves the server. The UI offers "kimi" and "bedrock".
    provider_choice: Literal["kimi", "bedrock"] | None = None


class ChatResponse(BaseModel):
    answer: str
    in_scope: bool
    citations: list[Citation] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    next_step_details: list[NextStep] = Field(default_factory=list)
    task_plan: TaskPlan | None = None
    evidence_coverage: EvidenceCoverage | None = None
    process_state: ProcessState | None = None
    compiled_context: CompiledContext | None = None
    compiled_context_used: bool = False
    resolved_entities: list[W3CEntity] = Field(default_factory=list)
    draft_contexts: list[DraftContext] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    source_version: SourceVersion = Field(default_factory=SourceVersion)
    refusal_reason: str | None = None
    # User-facing degradation notice, set when the answer is a limited fallback
    # (LLM provider unreachable, override rejected, empty model output) rather
    # than a real model generation. Always exposed to clients even when the
    # audit blob is stripped, so the UI can flag that the answer is degraded.
    notice: str | None = None
    workflow_trace: list[WorkflowStep] = Field(default_factory=list)
    audit: dict[str, Any] = Field(default_factory=dict)


class ClassifyRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)


class ClassifyResponse(BaseModel):
    in_scope: bool
    reason: str
    matched_topics: list[str] = Field(default_factory=list)
    injection_risk: bool = False
    confidence: float = Field(ge=0, le=1, default=1.0)


class LLMRouterDecision(BaseModel):
    attempted: bool = False
    likely_in_scope: bool = False
    intent_type: str = "unknown"
    needed_sources: list[SourceType] = Field(default_factory=list)
    entities_to_resolve: list[str] = Field(default_factory=list)
    search_hints: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1, default=0.0)
    reason: str = ""
    model: str | None = None
    error: str | None = None


class SourceStatusResponse(BaseModel):
    process_url: str
    guide_url: str
    process_repo: str
    guide_repo: str
    source_version: SourceVersion
    allowlist: list[str]
    member_only_enabled: bool


class ModelInfo(BaseModel):
    name: str
    provider: str = "ollama"
    size: int | None = None
    modified_at: str | None = None
    family: str | None = None
    is_embedding: bool = False


class ModelsResponse(BaseModel):
    default_model: str
    models: list[ModelInfo] = Field(default_factory=list)
    error: str | None = None


class EvalCaseResult(BaseModel):
    name: str
    passed: bool
    details: str
    tags: list[str] = Field(default_factory=list)
    expected_in_scope: bool | None = None
    actual_in_scope: bool | None = None
    expected_intent: str | None = None
    actual_intent: str | None = None
    expected_source_types: list[str] = Field(default_factory=list)
    actual_source_types: list[str] = Field(default_factory=list)
    expected_url_substrings: list[str] = Field(default_factory=list)
    actual_urls: list[str] = Field(default_factory=list)
    expected_entity_shortname: str | None = None
    actual_entity_shortnames: list[str] = Field(default_factory=list)
    expected_compiled_context: bool | None = None
    actual_compiled_context: bool | None = None
    confidence: float | None = None
    warnings: list[str] = Field(default_factory=list)


class EvalRunResponse(BaseModel):
    passed: bool
    score: float = Field(ge=0, le=1, default=0)
    passed_count: int = 0
    total_count: int = 0
    results: list[EvalCaseResult]


class FeedbackRequest(BaseModel):
    rating: str = Field(pattern="^(up|down)$")
    conversation_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")
    message_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")
    question: str = Field(min_length=1, max_length=8000)
    answer: str = Field(min_length=1, max_length=20000)
    comment: str | None = Field(default=None, max_length=4000)
    model: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,80}$")
    in_scope: bool | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    citation_urls: list[HttpUrl | str] = Field(default_factory=list, max_length=20)
    # Server-side audit fields are appended by the API; client-supplied audit
    # is intentionally NOT persisted to prevent JSONL pollution and DoS via
    # unbounded nested structures. See feedback service for the trusted audit.
    audit: dict[str, Any] = Field(default_factory=dict)


class FeedbackResponse(BaseModel):
    status: str = "recorded"
    received_at: str


class FeedbackStatsResponse(BaseModel):
    total: int = 0
    thumbs_up: int = 0
    thumbs_down: int = 0
    with_comment: int = 0
    approval_rate: float = 0.0


class LLMJudgeScoreItem(BaseModel):
    case_name: str
    tags: list[str] = Field(default_factory=list)
    question: str
    answer: str
    accuracy: float = Field(ge=0, le=5)
    groundedness: float = Field(ge=0, le=5)
    relevance: float = Field(ge=0, le=5)
    harm_avoidance: float = Field(ge=0, le=5)
    average: float = Field(ge=0, le=5)
    passed: bool
    reasoning: str = ""
    citation_urls: list[str] = Field(default_factory=list)
    error: str | None = None


class LLMJudgeReportResponse(BaseModel):
    total: int = 0
    passed: int = 0
    pass_rate: float = Field(ge=0, le=1, default=0.0)
    average_accuracy: float = Field(ge=0, le=5, default=0.0)
    average_groundedness: float = Field(ge=0, le=5, default=0.0)
    average_relevance: float = Field(ge=0, le=5, default=0.0)
    average_harm_avoidance: float = Field(ge=0, le=5, default=0.0)
    scores: list[LLMJudgeScoreItem] = Field(default_factory=list)


class CompiledStatusItem(BaseModel):
    key: str
    title: str
    source_path: str
    compiled_at: str | None = None
    is_stale: bool = False


class CompiledStatusResponse(BaseModel):
    enabled: bool
    items: list[CompiledStatusItem] = Field(default_factory=list)
