from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import Settings
from app.core.paths import resolve_data_path
from app.models.schemas import (
    Citation,
    CompiledContext,
    CompiledFreshness,
    CompiledProvenance,
    CompiledStatusItem,
    DraftContext,
    TaskPlan,
    W3CEntity,
)
from app.rag.retriever import Retriever
from app.services.github_context import GitHubDraftContextClient
from app.services.task_planner import build_planned_retrieval_query
from app.services.w3c_api import W3CAPIClient


FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)


@dataclass(frozen=True)
class CompiledBuildResult:
    context: CompiledContext | None
    written: bool


class CompiledContextStore:
    def __init__(
        self,
        settings: Settings,
        *,
        retriever: Retriever | None = None,
        w3c_api_client: W3CAPIClient | None = None,
        github_context_client: GitHubDraftContextClient | None = None,
    ) -> None:
        self.settings = settings
        self.enabled = settings.compiled_context_enabled
        self.compiled_dir = resolve_data_path(settings.compiled_context_dir)
        self.retriever = retriever or Retriever(settings.corpus_path)
        self.w3c_api_client = w3c_api_client or W3CAPIClient(settings)
        self.github_context_client = github_context_client or GitHubDraftContextClient(settings)

    def resolve(self, entities: list[W3CEntity]) -> CompiledContext | None:
        if not self.enabled:
            return None
        for entity in entities:
            if entity.entity_type != "specification":
                continue
            if (entity.confidence or 0) < self.settings.compiled_context_min_entity_confidence:
                continue
            if not entity.shortname:
                continue
            context = self._load(entity.shortname)
            if context:
                return context
        return None

    def rebuild_known(self, shortnames: list[str] | None = None) -> list[CompiledContext]:
        if not self.enabled:
            return []
        targets = shortnames or self._known_shortnames()
        contexts: list[CompiledContext] = []
        for shortname in targets:
            context = self.rebuild_shortname(shortname)
            if context:
                contexts.append(context)
        return contexts

    def rebuild_shortname(self, shortname: str) -> CompiledContext | None:
        if not self.enabled:
            return None
        entities = self.w3c_api_client.resolve_entities(shortname)
        entity = next(
            (
                item
                for item in entities
                if item.entity_type == "specification" and item.shortname == shortname
            ),
            None,
        )
        if not entity:
            return None
        return self.compile_entity(entity).context

    def compile_entity(self, entity: W3CEntity) -> CompiledBuildResult:
        if not self.enabled or entity.entity_type != "specification" or not entity.shortname:
            return CompiledBuildResult(context=None, written=False)

        task_plan = TaskPlan(
            intent_type="advance_specification",
            user_goal=f"Compiled context for {entity.title}",
            current_stage=_stage_from_status(entity.status),
            spec_or_group=entity.title,
            needed_sources=[],
            answer_shape="compiled_spec_context",
            search_queries=[],
            risk_flags=[],
            confidence=0.8,
        )
        focused_query = build_planned_retrieval_query(
            f"{entity.title} {entity.shortname} W3C Process Guidebook next steps horizontal review transition",
            task_plan,
        )
        citations = self.retriever.retrieve(focused_query)
        draft_contexts = self.github_context_client.resolve_contexts(entity.title, [entity], task_plan)
        compiled = self._materialize(entity, citations, draft_contexts)
        destination = self.compiled_dir / f"{entity.shortname}.md"
        destination.parent.mkdir(parents=True, exist_ok=True)
        current = destination.read_text(encoding="utf-8") if destination.exists() else None
        rendered = _render_markdown(compiled)
        if current == rendered:
            loaded = self._load(entity.shortname)
            return CompiledBuildResult(context=loaded or compiled, written=False)
        destination.write_text(rendered, encoding="utf-8")
        loaded = self._load(entity.shortname)
        return CompiledBuildResult(context=loaded or compiled, written=True)

    def status(self) -> list[CompiledStatusItem]:
        if not self.enabled or not self.compiled_dir.exists():
            return []
        items: list[CompiledStatusItem] = []
        for path in sorted(self.compiled_dir.glob("*.md")):
            context = self._load(path.stem)
            if not context:
                continue
            items.append(
                CompiledStatusItem(
                    key=context.key,
                    title=context.title,
                    source_path=str(path),
                    compiled_at=context.freshness.compiled_at,
                    is_stale=context.freshness.is_stale,
                )
            )
        return items

    def _known_shortnames(self) -> list[str]:
        values: set[str] = set()
        cache_path = Path(self.settings.w3c_api_cache_path)
        if cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            detail = payload.get("detail") if isinstance(payload, dict) else None
            if isinstance(detail, dict):
                for key in detail:
                    match = re.search(r"/specifications/([a-z0-9-]+)$", key)
                    if match:
                        values.add(match.group(1))
        if self.compiled_dir.exists():
            for path in self.compiled_dir.glob("*.md"):
                values.add(path.stem)
        return sorted(values)

    def _load(self, shortname: str) -> CompiledContext | None:
        path = self.compiled_dir / f"{shortname}.md"
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8")
        match = FRONTMATTER_RE.match(text)
        if not match:
            return None
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
        payload["source_path"] = str(path)
        return CompiledContext.model_validate(payload)

    def _materialize(
        self,
        entity: W3CEntity,
        citations: list[Citation],
        draft_contexts: list[DraftContext],
    ) -> CompiledContext:
        process_citations = [c for c in citations if c.source_type.value == "process"]
        guide_citations = [c for c in citations if c.source_type.value == "guide"]
        normative_urls = _dedupe([str(c.url) for c in process_citations])
        guide_urls = _dedupe([str(c.url) for c in guide_citations])
        operational_urls = _dedupe(
            [str(entity.api_url), *(str(entity.public_url or ""), str(entity.editor_draft_url or ""))]
            + [str(ctx.repo_url) for ctx in draft_contexts]
        )
        current_stage = _stage_from_status(entity.status)
        current_state = " | ".join(
            part
            for part in [
                entity.status or "",
                entity.latest_version_date or "",
                ", ".join(entity.deliverers),
            ]
            if part
        ) or None
        summary = _build_summary(entity, citations, draft_contexts)
        next_steps = _build_next_steps(entity, citations)
        guide_signals = _build_guide_signals(guide_citations)
        horizontal_signals = _build_horizontal_signals(citations, draft_contexts)
        charter_signals = _build_charter_signals(entity, draft_contexts)
        freshness = CompiledFreshness(
            compiled_at=datetime.now(timezone.utc).isoformat(),
            source_snapshot=_dedupe(
                [*(normative_urls[:3]), *(guide_urls[:3]), *(operational_urls[:4])]
            ),
            is_stale=False,
        )
        return CompiledContext(
            kind="spec",
            key=entity.shortname or entity.title,
            title=entity.title,
            summary=summary,
            current_state=current_state or current_stage,
            next_step_candidates=next_steps,
            guide_signals=guide_signals,
            horizontal_review_signals=horizontal_signals,
            charter_signals=charter_signals,
            freshness=freshness,
            provenance=CompiledProvenance(
                normative_urls=normative_urls,
                guide_urls=guide_urls,
                operational_urls=[value for value in operational_urls if value],
            ),
            confidence=min(0.94, 0.58 + (0.08 if guide_citations else 0) + (0.08 if process_citations else 0)),
        )


def _render_markdown(context: CompiledContext) -> str:
    payload = context.model_dump(mode="json")
    payload.pop("source_path", None)
    lines = [
        "---",
        json.dumps(payload, indent=2, ensure_ascii=False),
        "---",
        f"# {context.title}",
        "",
        f"- Shortname: `{context.key}`",
    ]
    if context.current_state:
        lines.append(f"- Current state: {context.current_state}")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            context.summary,
            "",
            "## Next step candidates",
            "",
        ]
    )
    lines.extend([f"- {step}" for step in context.next_step_candidates] or ["- No compiled next steps yet."])
    lines.extend(["", "## Guidebook signals", ""])
    lines.extend([f"- {item}" for item in context.guide_signals] or ["- No Guidebook signals compiled."])
    lines.extend(["", "## Horizontal review signals", ""])
    lines.extend([f"- {item}" for item in context.horizontal_review_signals] or ["- No horizontal review signals compiled."])
    if context.charter_signals:
        lines.extend(["", "## Charter signals", ""])
        lines.extend([f"- {item}" for item in context.charter_signals])
    lines.extend(["", "## Provenance", ""])
    lines.append(f"- Normative: {', '.join(str(url) for url in context.provenance.normative_urls) or '(none)'}")
    lines.append(f"- Guidebook: {', '.join(str(url) for url in context.provenance.guide_urls) or '(none)'}")
    lines.append(f"- Operational: {', '.join(str(url) for url in context.provenance.operational_urls) or '(none)'}")
    return "\n".join(lines) + "\n"


def _stage_from_status(status: str | None) -> str | None:
    text = (status or "").lower()
    if "candidate recommendation" in text or text.strip() == "cr":
        return "CR"
    if "proposed recommendation" in text:
        return "PR"
    if "recommendation" in text and "candidate" not in text and "proposed" not in text:
        return "REC"
    if "working draft" in text:
        return "WD"
    return None


def _build_summary(entity: W3CEntity, citations: list[Citation], draft_contexts: list[DraftContext]) -> str:
    stage = _stage_from_status(entity.status)
    process_hit = next((citation for citation in citations if citation.source_type.value == "process"), None)
    guide_hit = next((citation for citation in citations if citation.source_type.value == "guide"), None)
    parts = [
        f"{entity.title} ({entity.shortname}) is currently tracked as {entity.status or 'an active W3C deliverable'}."
    ]
    if stage == "CR":
        parts.append("The likely near-term workflow is advancing from Candidate Recommendation toward Recommendation-track publication readiness.")
    elif stage == "WD":
        parts.append("The likely near-term workflow is still centered on Working Draft progression, review readiness, and transition planning.")
    if process_hit:
        parts.append(f"Normative grounding is available from {process_hit.title}.")
    if guide_hit:
        parts.append(f"Guidebook operational guidance is available from {guide_hit.title}.")
    if draft_contexts:
        parts.append(f"Official draft repository context is available from {draft_contexts[0].repo_full_name}.")
    return " ".join(parts)


def _build_next_steps(entity: W3CEntity, citations: list[Citation]) -> list[str]:
    stage = _stage_from_status(entity.status)
    guide_url = next((str(c.url) for c in citations if c.source_type.value == "guide"), "https://www.w3.org/guide/")
    process_url = next((str(c.url) for c in citations if c.source_type.value == "process"), "https://www.w3.org/policies/process/")
    if stage == "CR":
        return [
            f"Confirm which transition target comes next for {entity.shortname} and gather the normative transition requirements from {process_url}.",
            f"Use the Guidebook transition and review guidance at {guide_url} to plan implementation evidence, review timing, and publication preparation.",
            "Check horizontal review requests, tracker labels, unresolved issues, and Staff Contact coordination before asking for the next transition.",
        ]
    if stage == "WD":
        return [
            f"Confirm whether the next milestone is another Working Draft publication, CR preparation, or a review checkpoint using {process_url}.",
            f"Use the Guidebook workflow guidance at {guide_url} to map concrete editorial and review work.",
            "Track review dependencies and publication blockers in the spec repository before the next formal step.",
        ]
    return [
        f"Confirm the current Process position and next formal milestone in {process_url}.",
        f"Use the Guidebook workflow guidance at {guide_url} to turn that milestone into concrete operational work.",
    ]


def _build_guide_signals(citations: list[Citation]) -> list[str]:
    values: list[str] = []
    for citation in citations[:4]:
        label = citation.heading_path or citation.title
        values.append(f"{label}: {str(citation.url)}")
    return _dedupe(values)


def _build_horizontal_signals(citations: list[Citation], draft_contexts: list[DraftContext]) -> list[str]:
    values: list[str] = []
    for citation in citations:
        text = " ".join(
            [citation.title, str(citation.url), citation.heading_path or "", citation.quote or ""]
        ).lower()
        if "horizontal" in text or "needs-resolution" in text or "tracker" in text:
            values.append(f"{citation.title}: {citation.url}")
    for context in draft_contexts:
        if context.repo_full_name == "w3c/strategy":
            values.append("Strategy tracker available for related charter review and horizontal review status signals.")
    if not values:
        values.append(
            "Use the Guidebook horizontal review material and GitHub request repositories to verify review timing and unresolved issues."
        )
    return _dedupe(values)


def _build_charter_signals(entity: W3CEntity, draft_contexts: list[DraftContext]) -> list[str]:
    values: list[str] = []
    if entity.deliverers:
        values.append(f"Deliverer groups: {', '.join(entity.deliverers)}")
    strategy = next((ctx for ctx in draft_contexts if ctx.repo_full_name == "w3c/strategy"), None)
    if strategy:
        values.append("Related charter tracking may exist in w3c/strategy issues with the `charter` label.")
    if entity.charter_end:
        values.append(f"Active charter end date: {entity.charter_end}")
    return _dedupe(values)


def _dedupe(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = " ".join(value.split())
        if not normalized:
            continue
        key = normalized.lower()
        if key not in seen:
            output.append(normalized)
            seen.add(key)
    return output
