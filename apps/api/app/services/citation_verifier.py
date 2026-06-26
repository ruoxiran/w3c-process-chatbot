"""Post-generation pass that verifies each ``[Sn]`` tag in the answer
actually corresponds to evidence in the cited excerpt.

The model is told (in the prompt) to attach ``[Sn]`` only to claims that
the excerpt supports. It mostly does. The cases that bite us are when
several excerpts mention the same surface term ("transition", "review")
and the model picks the wrong one — the user sees an authoritative-
looking citation that doesn't actually back the claim they care about.

We do not re-run the model on a failed citation; that would burn another
LLM call per question and the workflow already runs slow on Kimi. We
just strip the failing ``[Sn]`` tag from the answer text and record the
verification result in audit so the UI can surface a confidence hint.

A claim with no surviving citation is left in the answer but is now
visibly uncited — which is honest. The model's instructions already tell
it to mark unsupported claims; verification just enforces that
post-hoc.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from app.core.config import Settings
from app.models.schemas import Citation
from app.services.llm_router import JSONGenerator


logger = logging.getLogger(__name__)


# Cap pairs we verify. Most answers reference 4-8 sources; 12 is safe.
_MAX_PAIRS = 12

_CITATION_TAG_RE = re.compile(r"\[S(\d+)\]")
_CLAIM_SENTENCE_RE = re.compile(r"([^.!?\n]{8,}?\[S\d+\][^.!?\n]*?[.!?]?)", re.MULTILINE)


@dataclass(frozen=True)
class VerificationResult:
    answer: str
    """Possibly-rewritten answer with unsupported [Sn] tags stripped."""

    stripped_pairs: list[tuple[str, int]]
    """``(claim_snippet, citation_index_1_based)`` pairs that were removed."""

    model: str | None = None
    """Model used for verification, if any."""

    skipped_reason: str | None = None
    """Why we didn't verify (no client, no claims found, error)."""


def verify_citations(
    answer: str,
    citations: list[Citation],
    *,
    settings: Settings,
    client: JSONGenerator | None,
    model: str | None = None,
) -> VerificationResult:
    """Strip ``[Sn]`` tags that aren't supported by their cited excerpts.

    Two checks run in order:
      1. Free, no LLM call: strip ``[Sn]`` whose index is out of range
         relative to the citation list (e.g. ``[S99]`` when only 7
         citations exist). The model occasionally invents these and they
         can't be verified.
      2. LLM-based: for each in-range ``(claim, [Sn])`` pair, ask the
         verifier model whether the excerpt actually supports the claim;
         strip the tags marked unsupported.

    Step 1 always runs (it needs no client). Step 2 is skipped when no
    client is provided.
    """
    if not citations:
        return VerificationResult(answer=answer, stripped_pairs=[], skipped_reason="no_citations")

    pairs = _extract_claim_citation_pairs(answer)
    if not pairs:
        return VerificationResult(answer=answer, stripped_pairs=[], skipped_reason="no_pairs")

    # Step 1 — strip out-of-range tags first; this runs unconditionally so
    # the answer stays clean even when no verifier LLM is configured.
    n_citations = len(citations)
    out_of_range_indices = sorted({index for _, index in pairs if index < 1 or index > n_citations})
    new_answer = answer
    stripped: list[tuple[str, int]] = []
    for index in out_of_range_indices:
        tag = f"[S{index}]"
        if tag in new_answer:
            claim = next(
                (claim for claim, claim_index in pairs if claim_index == index),
                tag,
            )
            stripped.append((claim, index))
            new_answer = re.sub(r"\s*" + re.escape(tag), "", new_answer)

    in_range_pairs = [(c, i) for c, i in pairs if 1 <= i <= n_citations]

    if client is None:
        # Step 2 not available; return whatever step 1 produced.
        if stripped:
            return VerificationResult(answer=new_answer, stripped_pairs=stripped, skipped_reason="no_client")
        return VerificationResult(answer=answer, stripped_pairs=[], skipped_reason="no_client")

    if not in_range_pairs:
        if stripped:
            return VerificationResult(answer=new_answer, stripped_pairs=stripped, skipped_reason="all_out_of_range")
        return VerificationResult(answer=answer, stripped_pairs=[], skipped_reason="no_pairs_in_range")

    selected_model = model or settings.llm_router_model or settings.llm_model
    prompt = _prompt(in_range_pairs[:_MAX_PAIRS], citations)
    try:
        payload = client.generate_json(model=selected_model, prompt=prompt, num_predict=240)
    except Exception as exc:  # pragma: no cover - external service fallback
        logger.warning("Citation verifier LLM call failed; keeping answer as-is", exc_info=exc)
        # Out-of-range strips already done above; still return them.
        return VerificationResult(answer=new_answer, stripped_pairs=stripped, skipped_reason=type(exc).__name__, model=selected_model)

    unsupported = _parse_unsupported(payload)
    if not unsupported:
        if stripped:
            return VerificationResult(answer=new_answer, stripped_pairs=stripped, model=selected_model)
        return VerificationResult(answer=answer, stripped_pairs=[], model=selected_model)

    # Strip the [Sn] tags marked unsupported. We strip ALL occurrences of
    # the tag in the answer rather than just the one we evaluated — the
    # same wrong citation often appears multiple times in compound answers,
    # and the user would notice if half of them disappeared.
    for index in unsupported:
        tag = f"[S{index}]"
        if tag in new_answer:
            # Match the claim that contained the tag for the audit record.
            claim = next(
                (claim for claim, claim_index in pairs if claim_index == index),
                tag,
            )
            stripped.append((claim, index))
            # Strip the tag and the space/punct that immediately precedes it
            # (a dangling " ." after the strip looks ugly).
            new_answer = re.sub(r"\s*" + re.escape(tag), "", new_answer)
    if not stripped:
        return VerificationResult(answer=answer, stripped_pairs=[], model=selected_model)
    return VerificationResult(answer=new_answer, stripped_pairs=stripped, model=selected_model)


def _extract_claim_citation_pairs(answer: str) -> list[tuple[str, int]]:
    """Return ``(short_claim, citation_index)`` pairs found in the answer."""
    pairs: list[tuple[str, int]] = []
    for sentence in _CLAIM_SENTENCE_RE.findall(answer):
        sentence_clean = " ".join(sentence.split())
        if len(sentence_clean) < 12:
            continue
        for match in _CITATION_TAG_RE.finditer(sentence_clean):
            try:
                index = int(match.group(1))
            except ValueError:
                continue
            # Trim sentence to ~180 chars so the prompt stays small.
            claim = sentence_clean
            if len(claim) > 180:
                claim = claim[:180].rsplit(" ", 1)[0] + "..."
            pairs.append((claim, index))
            if len(pairs) >= _MAX_PAIRS:
                return pairs
    return pairs


def _parse_unsupported(payload: object) -> list[int]:
    if not isinstance(payload, dict):
        return []
    result = payload.get("pairs")
    if not isinstance(result, list):
        # Try ``unsupported_indices`` as a fallback shape.
        alt = payload.get("unsupported_indices")
        if isinstance(alt, list):
            return [int(v) for v in alt if _is_index(v)]
        return []
    indices: list[int] = []
    for entry in result:
        if not isinstance(entry, dict):
            continue
        index = entry.get("index")
        supported = entry.get("supported")
        if _is_index(index) and supported is False:
            indices.append(int(index))
    return indices


def _is_index(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int) and value >= 1:
        return True
    if isinstance(value, float) and value.is_integer() and value >= 1:
        return True
    return False


def _prompt(pairs: list[tuple[str, int]], citations: list[Citation]) -> str:
    pair_lines = []
    for claim, index in pairs:
        if index < 1 or index > len(citations):
            continue
        citation = citations[index - 1]
        excerpt = (citation.quote or "").strip()
        if len(excerpt) > 320:
            excerpt = excerpt[:320].rsplit(" ", 1)[0] + "..."
        pair_lines.append(
            "  - index: " + str(index) + "\n"
            "    claim: " + json.dumps(claim, ensure_ascii=False) + "\n"
            "    source heading: " + json.dumps((citation.heading_path or citation.title or "")[:120], ensure_ascii=False) + "\n"
            "    excerpt: " + json.dumps(excerpt, ensure_ascii=False)
        )
    return (
        "For each (claim, source excerpt) pair below, decide whether the excerpt "
        "actually supports the claim — i.e. a reader looking at only the excerpt "
        "would agree that the claim is correct.\n\n"
        "Rules:\n"
        "- Output a single JSON object: {\"pairs\": [{\"index\": <int>, \"supported\": <bool>}, ...]}\n"
        "- index is the 1-based citation index from the input.\n"
        "- ``supported`` is true ONLY if the excerpt contains language directly "
        "backing the claim. Vague topical overlap is not enough.\n"
        "- Do NOT include any other field or any prose.\n\n"
        "Pairs:\n"
        + "\n".join(pair_lines)
        + "\n\nJSON:"
    )
