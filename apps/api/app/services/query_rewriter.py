"""LLM-driven multi-query rewriting for retrieval.

A single user message often misses high-value chunks because the question's
vocabulary doesn't overlap with the corpus. For example: "publish my draft"
vs "FPWD", or "how do I make my spec official" vs "transition to
Recommendation". Dense retrieval helps, but a query that has two or three
phrasings widens recall further at the cost of one extra LLM call.

The rewriter asks the model for up to ``N`` W3C-specific reformulations of
the user's question, with the explicit constraint that each variant uses
W3C-canonical terminology (FPWD, CR, AC review, charter, horizontal review,
etc.) instead of natural-language paraphrases. We deduplicate, validate the
shape (each must look like a question or a noun phrase), and return a
small list to the workflow, which retrieves for each and merges results.

Failure is non-fatal — if the model errors or returns nonsense, the
workflow continues with the original message and an empty rewrite list.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass

from app.core.config import Settings
from app.services.llm_router import JSONGenerator


logger = logging.getLogger(__name__)


# Process-wide LRU cache so the same user question doesn't hit the LLM twice
# within a short period. The Workflow is a singleton (lru_cache in main.py),
# so this dict lives for the life of the process. Bounded so it can't grow
# unbounded over a long-running server.
_REWRITE_CACHE_LIMIT = 256
_rewrite_cache: OrderedDict[str, "QueryRewriteResult"] = OrderedDict()
_rewrite_cache_lock = threading.Lock()


@dataclass(frozen=True)
class QueryRewriteResult:
    variants: list[str]
    model: str
    error: str | None = None


# Cap variants tightly: each variant adds a retrieval pass + a workflow node.
# Three variants is the empirical sweet spot — diminishing returns past that
# and the LLM tends to start repeating itself.
_MAX_VARIANTS = 3


def rewrite_query(
    user_message: str,
    *,
    settings: Settings,
    client: JSONGenerator,
    model: str | None = None,
) -> QueryRewriteResult:
    """Return up to ``_MAX_VARIANTS`` W3C-canonical rewrites of ``user_message``.

    Does not include the original message in the returned list — the workflow
    always runs retrieval for the original separately.

    Process-wide cached by ``(message, model)`` so repeated questions (and
    busy traffic patterns like the eval harness) don't burn rate-limit quota.
    """
    selected_model = model or settings.llm_router_model or settings.llm_model
    cache_key = f"{selected_model}::{user_message.strip()}"
    with _rewrite_cache_lock:
        cached = _rewrite_cache.get(cache_key)
        if cached is not None:
            _rewrite_cache.move_to_end(cache_key)
            return cached
    prompt = _prompt(user_message)
    try:
        payload = client.generate_json(model=selected_model, prompt=prompt, num_predict=240)
    except Exception as exc:  # pragma: no cover - external model fallback
        logger.warning("Query rewriter LLM call failed; continuing with single-query retrieval", exc_info=exc)
        # Cache the failure too — short TTL would be nicer, but for a singleton
        # workflow the simpler bounded LRU keeps quota under control.
        result = QueryRewriteResult(variants=[], model=selected_model, error=type(exc).__name__)
        with _rewrite_cache_lock:
            _rewrite_cache[cache_key] = result
            _rewrite_cache.move_to_end(cache_key)
            while len(_rewrite_cache) > _REWRITE_CACHE_LIMIT:
                _rewrite_cache.popitem(last=False)
        return result

    raw_variants: list[str] = []
    if isinstance(payload, dict):
        for key in ("variants", "rewrites", "queries"):
            value = payload.get(key)
            if isinstance(value, list):
                raw_variants = value
                break
    result = QueryRewriteResult(
        variants=_clean(raw_variants, user_message),
        model=selected_model,
    )
    with _rewrite_cache_lock:
        _rewrite_cache[cache_key] = result
        _rewrite_cache.move_to_end(cache_key)
        while len(_rewrite_cache) > _REWRITE_CACHE_LIMIT:
            _rewrite_cache.popitem(last=False)
    return result


def _clean(raw: list[object], original: str) -> list[str]:
    seen: set[str] = {original.strip().lower()}
    out: list[str] = []
    for value in raw:
        if not isinstance(value, str):
            continue
        candidate = " ".join(value.split())
        if not candidate:
            continue
        if len(candidate) < 6 or len(candidate) > 240:
            continue
        normalized = candidate.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        out.append(candidate)
        if len(out) >= _MAX_VARIANTS:
            break
    return out


def _prompt(question: str) -> str:
    return (
        "You rewrite a W3C-Process question into 1-3 alternate retrieval queries "
        "that use the canonical W3C terminology (e.g. FPWD, Candidate Recommendation, "
        "AC Review, Horizontal Review, charter, Patent Policy, transition, Staff "
        "Contact, Working Group). Each rewrite should be useful for a keyword + "
        "semantic search against the W3C Process Document and Guidebook.\n\n"
        "Rules:\n"
        "- Output a single JSON object: {\"variants\": [\"...\", \"...\", \"...\"]}\n"
        "- 1-3 variants. Skip any that would duplicate the original.\n"
        "- Each variant is a SHORT search query (10-25 words), not a long paragraph.\n"
        "- Prefer W3C-canonical terms over colloquial phrasing.\n"
        "- Do not include explanations, markdown, code fences, or any other key.\n\n"
        f"Original question: {json.dumps(question, ensure_ascii=False)}\n\n"
        "JSON:"
    )
