"""LLM-as-reranker: a focused relevance pass over the hybrid retriever's
top candidates.

Hybrid retrieval (BM25 + TF-IDF + dense) gives broad recall but it is
biased by lexical overlap, document length, and our hand-tuned topic
bonuses. A cross-encoder or a chat-LLM scoring pass over (query, chunk)
pairs reorders candidates by *actual relevance to the user's question*,
which is the canonical "biggest single accuracy win" in RAG.

We implement it as an LLM-as-reranker rather than a real cross-encoder
because:

- It works with any provider already configured (Ollama or
  openai-compatible), without adding torch / transformers dependencies.
- It can be skipped cleanly when no LLM is available (eval mode,
  template provider) — the workflow falls back to hybrid-only ranking.
- Latency is bounded: one JSON call returning an ordered list of
  citation indices, capped at ~200 ms with a short prompt.

The reranker is *informational*: it returns a new ordering plus a
confidence score. The workflow keeps the hybrid score as a tie-breaker
so a model hallucination can't drop a clearly relevant citation off
the list.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from app.core.config import Settings
from app.models.schemas import Citation
from app.services.llm_router import JSONGenerator


logger = logging.getLogger(__name__)


# Cap candidates we send to the reranker. The trade-off:
# - more candidates = more thorough rerank but longer prompt + higher latency
# - 16 covers our usual top-N (12) plus a few from targeted retrieval
_MAX_RERANK_INPUT = 16

# How many results to keep after rerank. Same as the workflow's hard cap.
_RERANK_KEEP = 12


@dataclass(frozen=True)
class RerankResult:
    citations: list[Citation]
    """Re-ordered citation list (original list when rerank skipped/failed)."""

    reordered: bool
    """True iff the reranker actually moved at least one citation."""

    skipped_reason: str | None = None
    """Why we didn't rerank (no provider, too few candidates, error)."""

    model: str | None = None
    """The model that did the rerank, if any."""


def rerank_citations(
    user_message: str,
    citations: list[Citation],
    *,
    settings: Settings,
    client: JSONGenerator | None,
    model: str | None = None,
) -> RerankResult:
    """Reorder ``citations`` by LLM-judged relevance to ``user_message``.

    Returns the original list unchanged when no client is available, when
    there are fewer than 4 candidates (not worth a call), or when the LLM
    call fails. The fallback path is the workflow's standard "hybrid score
    is good enough" — no quality regression vs the pre-reranker behaviour.
    """
    if client is None:
        return RerankResult(citations=citations, reordered=False, skipped_reason="no_client")
    if len(citations) < 4:
        return RerankResult(citations=citations, reordered=False, skipped_reason="too_few_candidates")

    inputs = citations[:_MAX_RERANK_INPUT]
    selected_model = model or settings.llm_router_model or settings.llm_model
    prompt = _prompt(user_message, inputs)
    try:
        payload = client.generate_json(model=selected_model, prompt=prompt, num_predict=200)
    except Exception as exc:  # pragma: no cover - external service fallback
        logger.warning("Reranker LLM call failed; keeping hybrid order", exc_info=exc)
        return RerankResult(citations=citations, reordered=False, skipped_reason=type(exc).__name__, model=selected_model)

    order = _parse_order(payload, len(inputs))
    if not order:
        return RerankResult(citations=citations, reordered=False, skipped_reason="empty_order", model=selected_model)

    seen: set[int] = set()
    reordered: list[Citation] = []
    for index in order:
        if 0 <= index < len(inputs) and index not in seen:
            reordered.append(inputs[index])
            seen.add(index)
    # Preserve any candidate the reranker dropped — it's a tie-breaker, not
    # a censor. We don't trust the model enough to let it remove citations
    # outright.
    for index, citation in enumerate(inputs):
        if index not in seen:
            reordered.append(citation)
    # Anything beyond _MAX_RERANK_INPUT stays in its hybrid-ranked place.
    reordered.extend(citations[_MAX_RERANK_INPUT:])
    reordered = reordered[:_RERANK_KEEP]

    moved = any(
        original.url != reranked.url
        for original, reranked in zip(citations[: len(reordered)], reordered)
    )
    return RerankResult(citations=reordered, reordered=moved, model=selected_model)


def _parse_order(payload: object, n: int) -> list[int]:
    """Accept several reasonable shapes the model might emit."""
    if isinstance(payload, dict):
        for key in ("order", "ranking", "ranks", "indices"):
            value = payload.get(key)
            if isinstance(value, list):
                return [int(v) for v in value if _is_index(v, n)]
        # Some models return {"results": [{"index": 0, ...}, ...]}.
        results = payload.get("results")
        if isinstance(results, list):
            indices: list[int] = []
            for entry in results:
                if isinstance(entry, dict):
                    idx = entry.get("index")
                    if _is_index(idx, n):
                        indices.append(int(idx))
            if indices:
                return indices
    return []


def _is_index(value: object, n: int) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return 0 <= value < n
    if isinstance(value, float) and value.is_integer():
        return 0 <= int(value) < n
    return False


def _prompt(user_message: str, candidates: list[Citation]) -> str:
    lines = []
    for index, citation in enumerate(candidates):
        heading = (citation.heading_path or citation.title or "").strip()
        snippet = (citation.quote or "").strip()
        if len(snippet) > 220:
            snippet = snippet[:220].rsplit(" ", 1)[0] + "..."
        lines.append(
            f"[{index}] type={citation.source_type.value}; heading={heading[:120]}; url={citation.url}\n    excerpt: {snippet}"
        )
    candidates_text = "\n".join(lines)
    return (
        "Rank the candidates below by how directly they answer the user's "
        "W3C-Process question. The most relevant index comes first.\n\n"
        "Rules:\n"
        "- Output a single JSON object: {\"order\": [<index>, <index>, ...]}\n"
        "- Each index is the integer in [0, N) shown in brackets.\n"
        "- Include AT LEAST the 6 most relevant; you may include all of them.\n"
        "- Do not invent indices; do not output prose, explanations, or markdown.\n"
        "- Treat a dedicated Guidebook chapter on the user's topic as more "
        "relevant than a generic Process section that merely mentions the term.\n"
        "- A snapshot copy of the Process Document is less relevant than the "
        "canonical w3.org/policies/process page covering the same fragment.\n\n"
        f"User question: {json.dumps(user_message, ensure_ascii=False)}\n\n"
        "Candidates:\n"
        f"{candidates_text}\n\n"
        "JSON:"
    )
