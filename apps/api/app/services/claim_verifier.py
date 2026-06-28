"""Chain-of-verification — find factual claims in the answer that have
no ``[Sn]`` citation backing them at all.

This is the complement to ``citation_verifier``:

  * ``citation_verifier`` looks at each ``[Sn]`` tag and asks "does the
    cited excerpt actually support the claim attached to it?". It
    strips tags that fail verification.
  * ``claim_verifier`` (this module) looks at each declarative
    sentence WITHOUT a ``[Sn]`` tag and asks "is this a substantive
    factual claim that should have been cited?". For claims that
    should have been but weren't, the workflow appends an inline
    ``[unverified]`` marker so users see which assertions lack source
    grounding.

The two together close both directions of hallucination:
  - Phantom citations (tag exists, excerpt doesn't support it)
  - Phantom claims (claim exists, no tag — pure model assertion)

One LLM call per request. Cached process-wide so the eval harness's
repeat-question pattern doesn't burn quota. Off by default — flip
``settings.claim_verification_enabled`` to turn on.
"""

from __future__ import annotations

import logging
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass

from app.core.config import Settings
from app.models.schemas import Citation
from app.services.llm_router import JSONGenerator


logger = logging.getLogger(__name__)


_CACHE_LIMIT = 256
_cache: OrderedDict[str, "ClaimVerificationResult"] = OrderedDict()
_cache_lock = threading.Lock()


_CITATION_TAG_RE = re.compile(r"\[S\d+\]")
# Sentence segmentation — same conservative regex shape as the citation
# verifier so the two modules agree on what counts as a sentence.
_SENTENCE_RE = re.compile(r"([^.!?\n]{8,}?[.!?])", re.MULTILINE)
# Words that mark a sentence as a substantive procedural claim rather
# than a connector or a meta-comment. Cheap pre-filter so we don't
# send the LLM connector sentences like "First, ..." or "In summary,..."
_CLAIM_CUE_WORDS = (
    "must", "should", "requires", "required", "may", "shall", "will",
    "needs to", "has to", "is required", "is responsible",
    "before", "after", "during", "when transitioning",
    "file ", "submit ", "request ", "open an issue", "send to",
    "process ", "policy ", "guidebook ", "wide review", "horizontal review",
    "charter", "transition", "review", "publication", "recommendation",
    "candidate", "working group", "advisory committee", "team", "chair",
    "exclusion", "patent", "antitrust", "code of conduct",
)


@dataclass(frozen=True)
class UnsupportedClaim:
    """One sentence the verifier flagged as a factual claim without ``[Sn]``."""

    sentence: str
    confidence: float
    """0–1 estimate that this is a real factual gap (not just a sentence the
    classifier mishandled)."""


@dataclass(frozen=True)
class ClaimVerificationResult:
    annotated_answer: str
    """The answer with ``[unverified]`` markers appended to each
    flagged sentence. Equal to the input answer when nothing was
    flagged or the LLM call failed."""

    unsupported_claims: tuple[UnsupportedClaim, ...]
    model: str
    error: str | None = None
    skipped_reason: str | None = None


def _candidate_unsourced_sentences(answer: str) -> list[str]:
    """First-pass filter: claim-shaped sentences WITHOUT a ``[Sn]`` tag.

    Cheap regex + cue-word check. Sentences the LLM judge then
    evaluates — if it agrees they're substantive AND unsupported,
    we mark them.
    """
    out: list[str] = []
    seen: set[str] = set()
    for match in _SENTENCE_RE.findall(answer):
        sentence = " ".join(match.split())
        if len(sentence) < 24:
            continue
        if _CITATION_TAG_RE.search(sentence):
            continue
        if not any(cue in sentence.lower() for cue in _CLAIM_CUE_WORDS):
            continue
        if sentence in seen:
            continue
        seen.add(sentence)
        out.append(sentence)
    return out


def _excerpt_for_prompt(citation: Citation, index: int) -> str:
    quote = (citation.quote or "").strip()
    if len(quote) > 280:
        quote = quote[:277] + "..."
    return f"[S{index}] {citation.heading_path or citation.title or '(no heading)'}: {quote}"


def _prompt(sentences: list[str], citations: list[Citation]) -> str:
    excerpt_lines = "\n".join(_excerpt_for_prompt(c, i) for i, c in enumerate(citations, start=1))
    sentence_lines = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, start=1))
    return f"""You are auditing a W3C Process answer for unsupported factual
claims. Below are the cited excerpts the answer was grounded in, plus
a list of sentences from the answer that DO NOT carry a ``[Sn]`` citation
tag.

For each sentence, decide:
  - ``true`` if the sentence makes a SUBSTANTIVE factual claim
    about Process / Guidebook / W3C policy AND none of the cited
    excerpts directly supports it.
  - ``false`` otherwise — either the sentence is a connector /
    summary / definition (not a procedural fact), OR one of the
    excerpts does support it (so it was just missing the tag).

Cited excerpts:
{excerpt_lines}

Sentences to evaluate (numbered):
{sentence_lines}

Output JSON only:
  {{"unsupported": [<list of sentence numbers from above that are TRUE>]}}"""


def verify_claims(
    answer: str,
    citations: list[Citation],
    *,
    settings: Settings,
    client: JSONGenerator,
    model: str | None = None,
) -> ClaimVerificationResult:
    """Find unsupported factual claims in ``answer``; annotate them.

    Returns the original answer when:
      * No citations (nothing to verify against)
      * No candidate unsourced sentences passed the pre-filter
      * LLM call failed (graceful degradation; never raises out)
    """
    selected_model = model or settings.llm_router_model or settings.llm_model
    if not answer or not citations:
        return ClaimVerificationResult(
            annotated_answer=answer,
            unsupported_claims=(),
            model=selected_model,
            skipped_reason="no_input",
        )

    candidates = _candidate_unsourced_sentences(answer)
    if not candidates:
        return ClaimVerificationResult(
            annotated_answer=answer,
            unsupported_claims=(),
            model=selected_model,
            skipped_reason="no_candidates",
        )

    # Cap how many sentences we send the LLM — long answers with
    # many uncited connector sentences shouldn't blow the prompt or
    # the LLM's context. The pre-filter is already aggressive, so 8
    # is more than enough for typical workflow answers.
    candidates = candidates[:8]

    cache_key = f"{selected_model}::{hash((answer, tuple(c.url for c in citations)))}"
    with _cache_lock:
        cached = _cache.get(cache_key)
        if cached is not None:
            _cache.move_to_end(cache_key)
            return cached

    try:
        payload = client.generate_json(
            model=selected_model,
            prompt=_prompt(candidates, citations),
            num_predict=200,
        )
    except Exception as exc:  # pragma: no cover - external model fallback
        logger.warning("Claim verifier LLM call failed; skipping annotation", exc_info=exc)
        result = ClaimVerificationResult(
            annotated_answer=answer,
            unsupported_claims=(),
            model=selected_model,
            error=type(exc).__name__,
            skipped_reason="llm_error",
        )
        _store(cache_key, result)
        return result

    flagged_indices: list[int] = []
    if isinstance(payload, dict):
        raw = payload.get("unsupported") or payload.get("indices") or []
        if isinstance(raw, list):
            for value in raw:
                try:
                    idx = int(value) - 1  # prompt uses 1-based numbering
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < len(candidates):
                    flagged_indices.append(idx)

    if not flagged_indices:
        result = ClaimVerificationResult(
            annotated_answer=answer,
            unsupported_claims=(),
            model=selected_model,
        )
        _store(cache_key, result)
        return result

    flagged_sentences = {candidates[i] for i in flagged_indices}

    # Annotate by inserting ``*[unverified]*`` just before each
    # flagged sentence's terminator (. / ! / ?). The ``_SENTENCE_RE``
    # pre-filter captures sentences WITH their terminator, so we
    # split body + terminator and reassemble with the marker in the
    # middle. Plain ``str.replace`` keeps the logic readable —
    # no regex backref subtleties.
    annotated = answer
    annotated_set: set[str] = set()
    for sentence in flagged_sentences:
        if sentence in annotated_set:
            continue
        annotated_set.add(sentence)
        if sentence and sentence[-1] in ".!?":
            body, terminator = sentence[:-1], sentence[-1]
            marked = f"{body} *[unverified]*{terminator}"
        else:
            marked = f"{sentence} *[unverified]*"
        if sentence in annotated:
            annotated = annotated.replace(sentence, marked, 1)

    unsupported = tuple(
        UnsupportedClaim(sentence=candidates[i], confidence=0.7)
        for i in flagged_indices
    )
    result = ClaimVerificationResult(
        annotated_answer=annotated,
        unsupported_claims=unsupported,
        model=selected_model,
    )
    _store(cache_key, result)
    return result


def _store(cache_key: str, result: ClaimVerificationResult) -> None:
    with _cache_lock:
        _cache[cache_key] = result
        _cache.move_to_end(cache_key)
        while len(_cache) > _CACHE_LIMIT:
            _cache.popitem(last=False)


def _clear_cache_for_tests() -> None:
    with _cache_lock:
        _cache.clear()
