from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    database_url: str = "postgresql://w3c:w3c@localhost:5432/w3c_process"
    redis_url: str = "redis://localhost:6379/0"
    llm_base_url: str = "http://localhost:8001/v1"
    llm_provider: str = "ollama"
    llm_model: str = "qwen3:8b"
    openai_compatible_base_url: str = "https://api.openai.com/v1"
    openai_compatible_api_key: str | None = None
    openai_compatible_model: str = "gpt-4.1"
    openai_compatible_timeout_seconds: float = 120
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout_seconds: float = 120
    llm_router_enabled: bool = True
    llm_router_model: str | None = None
    llm_router_min_confidence: float = 0.55
    w3c_api_enabled: bool = True
    w3c_api_base_url: str = "https://api.w3.org"
    w3c_api_timeout_seconds: float = 12
    w3c_api_cache_ttl_seconds: int = 21600
    w3c_api_catalog_pages: int = 2
    w3c_api_persistent_cache_enabled: bool = True
    w3c_api_cache_path: str = "data/cache/w3c_api_cache.json"
    github_context_enabled: bool = True
    github_api_base_url: str = "https://api.github.com"
    github_context_timeout_seconds: float = 8
    github_context_cache_ttl_seconds: int = 21600
    github_context_allowed_orgs: str = "w3c,w3ctag,w3cping"
    github_context_max_files: int = 6
    github_context_max_file_bytes: int = 12000
    github_token: str | None = None
    corpus_path: str = "data/corpus/chunks.jsonl"
    compiled_context_dir: str = "data/compiled/spec"
    compiled_context_enabled: bool = True
    compiled_context_min_entity_confidence: float = 0.7
    retrieval_dense_enabled: bool = False
    retrieval_embedding_cache_path: str = "data/cache/retrieval_embeddings.jsonl"
    retrieval_dense_weight: float = 24.0
    retrieval_dense_candidate_limit: int = 80
    embedding_model: str = "BAAI/bge-m3"
    ollama_embedding_model: str = "qwen3-embedding:4b"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    # Cross-encoder reranker takes priority over the LLM-as-reranker when
    # enabled. The dependency (sentence-transformers + torch + ~600MB model
    # download on first use) is optional; if loading fails the workflow
    # silently falls back to the LLM reranker.
    reranker_cross_encoder_enabled: bool = True
    source_allowlist: str = Field(
        default="w3.org,api.w3.org,github.com/w3c,github.com/w3ctag,github.com/w3cping,w3c.github.io,w3ctag.github.io"
    )
    live_fetch_enabled: bool = False
    live_fetch_max_chars: int = 3500
    live_fetch_timeout_seconds: float = 8.0
    feedback_log_path: str = "data/feedback/feedback.jsonl"
    # /docs /redoc /openapi.json are hidden by default; opt in only in development.
    expose_openapi_docs: bool = False
    # Optional API key gate. When set, all endpoints (except /health) require
    # ``X-API-Key: <value>`` on the request. None disables the gate so local
    # development still works without configuration.
    api_key: str | None = None
    # Per-IP rate limits applied via slowapi. Tweak for production.
    rate_limit_chat: str = "30/minute"
    rate_limit_eval: str = "3/minute"
    rate_limit_judge: str = "1/minute"
    rate_limit_default: str = "120/minute"
    # When False, the audit blob is stripped from ChatResponse before it leaves
    # the API. Set True only when you want full introspection in the UI.
    expose_audit: bool = False
    index_refresh_cron: str = "0 */6 * * *"
    require_citations: bool = True
    enable_member_only_sources: bool = False
    cors_allow_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    cors_allow_methods: str = "GET,POST"
    cors_allow_headers: str = "Content-Type,X-API-Key"

    @property
    def allowlist_entries(self) -> list[str]:
        return [entry.strip().lower() for entry in self.source_allowlist.split(",") if entry.strip()]

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allow_origins.split(",") if origin.strip()]

    @property
    def github_allowed_orgs(self) -> list[str]:
        return [entry.strip().lower() for entry in self.github_context_allowed_orgs.split(",") if entry.strip()]

    @property
    def cors_methods(self) -> list[str]:
        return [method.strip().upper() for method in self.cors_allow_methods.split(",") if method.strip()]

    @property
    def cors_headers(self) -> list[str]:
        return [header.strip() for header in self.cors_allow_headers.split(",") if header.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
