"""Cross-encoder reranker using a local sentence-transformers model.

The LLM-as-reranker in ``services/reranker.py`` works but it (a) costs an
LLM call per query and (b) the model can be wrong about relevance in
ways a dedicated reranker isn't. ``BAAI/bge-reranker-v2-m3`` is a small
(~600 MB) multilingual cross-encoder specifically trained to score
``(query, passage)`` relevance. It runs locally on CPU at ~50-150 ms
for a batch of 16 candidates, which is the same order as the LLM
reranker but without the model-call quota cost.

We treat the model as a lazy singleton: loaded on first use, kept warm
for the life of the process. If sentence-transformers / torch isn't
installed, this module's loader raises ``MissingDependencyError`` and
the workflow silently falls back to the LLM-as-reranker (the existing
hybrid path is unaffected).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from app.models.schemas import Citation


logger = logging.getLogger(__name__)


class MissingDependencyError(RuntimeError):
    """Raised when sentence-transformers / torch isn't installed."""


# Reranker models we've vetted for use. Cross-encoder loading via
# sentence-transformers will reach out to HuggingFace and download
# *whatever repo id is passed*, and PyTorch's ``.bin`` loader has a
# known unpickle-code-execution risk. So we refuse any model id not on
# this list — operators who want to add a model must opt-in by editing
# this constant rather than just flipping an env var.
_ALLOWED_RERANKER_MODELS: frozenset[str] = frozenset({
    "BAAI/bge-reranker-v2-m3",
    "BAAI/bge-reranker-base",
    "BAAI/bge-reranker-large",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
})


def _validate_model_name(model_name: str) -> None:
    if model_name not in _ALLOWED_RERANKER_MODELS:
        raise MissingDependencyError(
            f"reranker model {model_name!r} is not in the allowlist; "
            "add it to _ALLOWED_RERANKER_MODELS in cross_encoder_reranker.py "
            "after verifying it's a trusted source"
        )


@dataclass(frozen=True)
class CrossEncoderRerankResult:
    citations: list[Citation]
    """Reordered citations. Hybrid order preserved as tie-breaker."""

    reordered: bool
    skipped_reason: str | None = None


# How many top candidates to score. Same upper bound as the LLM
# reranker: rerank input cap is 16, but if more candidates are given
# the helper just scores them all (fast).
_DEFAULT_KEEP = 12


_model_lock = threading.Lock()
_model_cache: object | None = None
_model_name_loaded: str | None = None
_load_error: BaseException | None = None


def _load_model(model_name: str) -> object:
    """Lazily load the cross-encoder. Cached for the process lifetime.

    Concurrent first-call access is serialised through a lock so we don't
    load the 600 MB model twice. After the first success the unlocked
    fast-path returns the cached instance.
    """
    global _model_cache, _model_name_loaded, _load_error

    if _model_cache is not None and _model_name_loaded == model_name:
        return _model_cache
    if _load_error is not None:
        raise MissingDependencyError(str(_load_error))

    _validate_model_name(model_name)

    with _model_lock:
        if _model_cache is not None and _model_name_loaded == model_name:
            return _model_cache
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]
        except ImportError as exc:
            _load_error = exc
            raise MissingDependencyError(
                "sentence-transformers is not installed; install it or fall "
                "back to the LLM-as-reranker"
            ) from exc

        logger.info("Loading cross-encoder %s (first call; one-time download if missing)", model_name)
        _model_cache = CrossEncoder(model_name, max_length=512)
        _model_name_loaded = model_name
        return _model_cache


def rerank_with_cross_encoder(
    user_message: str,
    citations: list[Citation],
    *,
    model_name: str,
    keep: int = _DEFAULT_KEEP,
) -> CrossEncoderRerankResult:
    """Reorder ``citations`` by cross-encoder relevance to ``user_message``.

    The cross-encoder scores each ``(message, passage)`` pair where
    ``passage`` is the citation's quote (or heading_path as fallback).
    Higher score = more relevant. Hybrid order is the tie-breaker when
    scores are equal.

    Skipped silently when:
      - fewer than 4 citations (not worth the call)
      - the cross-encoder model can't be loaded (missing dependency)
      - the model raises during scoring (we log and fall back)
    """
    if len(citations) < 4:
        return CrossEncoderRerankResult(citations=citations, reordered=False, skipped_reason="too_few_candidates")

    try:
        model = _load_model(model_name)
    except MissingDependencyError as exc:
        return CrossEncoderRerankResult(
            citations=citations, reordered=False, skipped_reason=f"unavailable:{exc}"
        )

    passages = [_passage_text(c) for c in citations]
    pairs = [(user_message, passage) for passage in passages]
    try:
        scores = model.predict(pairs)  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - model error path
        logger.warning("Cross-encoder predict failed; keeping hybrid order", exc_info=exc)
        return CrossEncoderRerankResult(
            citations=citations, reordered=False, skipped_reason=type(exc).__name__
        )

    # Pair each citation with its score AND its original hybrid rank, so
    # ties (or scores within rounding distance) preserve the hybrid order
    # rather than introducing fresh non-determinism.
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda item: (-float(item[1]), item[0]))
    reordered_citations = [citations[i] for i, _ in indexed][:keep]
    moved = any(orig.url != new.url for orig, new in zip(citations[: len(reordered_citations)], reordered_citations))
    return CrossEncoderRerankResult(citations=reordered_citations, reordered=moved)


def _passage_text(citation: Citation) -> str:
    """The string the cross-encoder scores against the user query."""
    parts: list[str] = []
    heading = citation.heading_path or citation.title
    if heading:
        parts.append(heading)
    if citation.quote:
        parts.append(citation.quote)
    if not parts:
        return str(citation.url)
    text = " — ".join(parts)
    # Cross-encoder max_length=512 tokens; keep the input short so
    # truncation doesn't drop the heading. ~1800 chars is well under.
    return text[:1800]
