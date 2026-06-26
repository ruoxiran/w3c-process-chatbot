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
from dataclasses import dataclass

from app.core.config import Settings
from app.services.llm_router import JSONGenerator


logger = logging.getLogger(__name__)


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
    """
    selected_model = model or settings.llm_router_model or settings.llm_model
    prompt = _prompt(user_message)
    try:
        payload = client.generate_json(model=selected_model, prompt=prompt, num_predict=240)
    except Exception as exc:  # pragma: no cover - external model fallback
        logger.warning("Query rewriter LLM call failed; continuing with single-query retrieval", exc_info=exc)
        return QueryRewriteResult(variants=[], model=selected_model, error=type(exc).__name__)

    raw_variants: list[str] = []
    if isinstance(payload, dict):
        for key in ("variants", "rewrites", "queries"):
            value = payload.get(key)
            if isinstance(value, list):
                raw_variants = value
                break
    return QueryRewriteResult(
        variants=_clean(raw_variants, user_message),
        model=selected_model,
    )


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
