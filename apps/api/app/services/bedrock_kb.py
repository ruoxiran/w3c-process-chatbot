"""Retrieval from an AWS Bedrock Knowledge Base.

Queries the managed KB via the boto3 ``bedrock-agent-runtime`` ``Retrieve``
API and maps each passage into a ``Citation`` so KB results flow through the
same reranking / grounding / display path as local-corpus chunks.

This is retrieval only — it uses the ``bedrock:Retrieve`` IAM action and is
independent of the generation provider (``bedrock:InvokeModel``). Credentials
are supplied explicitly, mirroring ``BedrockClient``; the boto3 client is
built lazily so importing this module never requires boto3.
"""

from __future__ import annotations

import logging

from app.models.schemas import Citation, SourceType


logger = logging.getLogger(__name__)

# Cap the excerpt pulled into the prompt so a large KB chunk can't blow the
# context budget; the reranker/grounding only need enough to judge relevance.
_MAX_QUOTE_CHARS = 1500


def _is_managed_kb_error(exc: Exception) -> bool:
    """True when the KB is a *managed* knowledge base that rejects the
    ``vectorSearchConfiguration`` we send by default. Managed KBs want a
    ``managedSearchConfiguration`` (not in older boto3) or simply no
    ``retrievalConfiguration`` at all — so we retry without one.
    """
    text = str(exc).lower()
    return "managed knowledge base" in text or "managedsearchconfiguration" in text


class BedrockKnowledgeBaseClient:
    def __init__(
        self,
        knowledge_base_id: str,
        region: str,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        session_token: str | None = None,
        max_results: int = 8,
        timeout_seconds: float = 30,
    ) -> None:
        self.knowledge_base_id = knowledge_base_id
        self.region = region
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.session_token = session_token
        self.max_results = max_results
        self.timeout_seconds = timeout_seconds
        self._runtime = None

    def _client(self):
        if self._runtime is None:
            import boto3
            from botocore.config import Config

            self._runtime = boto3.client(
                "bedrock-agent-runtime",
                region_name=self.region,
                aws_access_key_id=self.access_key_id,
                aws_secret_access_key=self.secret_access_key,
                aws_session_token=self.session_token,
                config=Config(
                    read_timeout=self.timeout_seconds,
                    connect_timeout=self.timeout_seconds,
                ),
            )
        return self._runtime

    def retrieve(self, query: str) -> list[Citation]:
        try:
            response = self._client().retrieve(
                knowledgeBaseId=self.knowledge_base_id,
                retrievalQuery={"text": query},
                retrievalConfiguration={
                    "vectorSearchConfiguration": {"numberOfResults": self.max_results}
                },
            )
        except Exception as exc:
            # Managed KBs reject vectorSearchConfiguration; retry with the
            # service default result count (numberOfResults isn't tunable on
            # managed KBs through older boto3).
            if not _is_managed_kb_error(exc):
                raise
            logger.info("Managed Bedrock KB detected; retrieving without vectorSearchConfiguration")
            response = self._client().retrieve(
                knowledgeBaseId=self.knowledge_base_id,
                retrievalQuery={"text": query},
            )
        citations: list[Citation] = []
        for result in response.get("retrievalResults", []):
            if not isinstance(result, dict):
                continue
            citation = self._result_to_citation(result)
            if citation is not None:
                citations.append(citation)
        return citations

    def _result_to_citation(self, result: dict) -> Citation | None:
        text = (result.get("content") or {}).get("text")
        if not isinstance(text, str) or not text.strip():
            return None
        url = self._source_url(result)
        return Citation(
            title=self._title(result, url),
            url=url,
            source_type=SourceType.related_policy,
            quote=text.strip()[:_MAX_QUOTE_CHARS],
        )

    def _source_url(self, result: dict) -> str:
        """Best-available source locator for a KB result.

        Prefers an explicit web/S3 location, then the source-uri metadata key
        Bedrock stamps on ingested documents, then a synthetic KB urn so the
        Citation always has a non-empty url.
        """
        location = result.get("location") or {}
        for key in ("webLocation",):
            block = location.get(key)
            if isinstance(block, dict) and isinstance(block.get("url"), str) and block["url"]:
                return block["url"]
        metadata = result.get("metadata") or {}
        # ``_source_uri`` is the ingestion source (often a clickable http(s)
        # link) that Bedrock stamps on managed-KB chunks; prefer it over the
        # raw S3 object uri.
        for key in ("_source_uri", "x-amz-bedrock-kb-source-uri"):
            value = metadata.get(key)
            if isinstance(value, str) and value:
                return value
        for key in ("s3Location", "confluenceLocation", "sharePointLocation", "salesforceLocation"):
            block = location.get(key)
            if isinstance(block, dict):
                value = block.get("url") or block.get("uri")
                if isinstance(value, str) and value:
                    return value
        return f"bedrock-kb://{self.knowledge_base_id}"

    @staticmethod
    def _title(result: dict, url: str) -> str:
        metadata = result.get("metadata") or {}
        for key in ("_document_title", "title", "document_title", "name", "x-amz-bedrock-kb-title"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        # Fall back to the last meaningful path segment of the source url.
        tail = url.rstrip("/").rsplit("/", 1)[-1]
        return tail or "Knowledge Base result"
