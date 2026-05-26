from __future__ import annotations

import base64
import re
import time
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from app.core.config import Settings
from app.models.schemas import DraftContext, DraftSnippet, TaskPlan, W3CEntity


CONTEXT_QUERY_MARKERS = [
    "draft",
    "editor draft",
    "editors draft",
    "repo",
    "repository",
    "github",
    "issue",
    "issues",
    "pull request",
    "pr ",
    "草案",
    "仓库",
    "上下文",
    "问题",
]

SPEC_SOURCE_FILES = [
    "README.md",
    "readme.md",
    "index.bs",
    "Overview.bs",
    "index.html",
    "Overview.html",
    "spec.bs",
    "w3c.json",
    "package.json",
    "echidna.json",
]

@dataclass(frozen=True)
class RepoCandidate:
    owner: str
    repo: str
    resolved_from: str

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.repo}"

    @property
    def html_url(self) -> str:
        return f"https://github.com/{self.owner}/{self.repo}"


STRATEGY_CHARTER_REPO = RepoCandidate(
    owner="w3c",
    repo="strategy",
    resolved_from="https://github.com/w3c/strategy/issues?q=label%3Acharter",
)


class GitHubDraftContextClient:
    def __init__(self, settings: Settings) -> None:
        self.enabled = settings.github_context_enabled
        self.api_base_url = settings.github_api_base_url.rstrip("/")
        self.timeout = settings.github_context_timeout_seconds
        self.cache_ttl = settings.github_context_cache_ttl_seconds
        self.allowed_orgs = set(settings.github_allowed_orgs)
        self.max_files = max(1, settings.github_context_max_files)
        self.max_file_bytes = max(1000, settings.github_context_max_file_bytes)
        self.token = settings.github_token
        self._cache: dict[str, tuple[float, DraftContext | None]] = {}

    def resolve_contexts(
        self,
        query: str,
        entities: list[W3CEntity],
        task_plan: TaskPlan,
        limit: int = 2,
    ) -> list[DraftContext]:
        if not self.enabled:
            return []

        contexts: list[DraftContext] = []
        seen: set[str] = set()
        if _should_resolve_strategy_charter_context(query, task_plan):
            context = self._strategy_charter_context(query)
            if context:
                contexts.append(context)
                seen.add(context.repo_full_name)

        if not _should_resolve_draft_context(query, entities, task_plan):
            return contexts[:limit]

        for candidate in _repo_candidates_from_entities(entities, self.allowed_orgs):
            if candidate.full_name in seen:
                continue
            seen.add(candidate.full_name)
            context = self._context_for_repo(candidate)
            if context:
                contexts.append(context)
            if len(contexts) >= limit:
                break
        return contexts

    def _strategy_charter_context(self, query: str) -> DraftContext | None:
        cache_key = "w3c/strategy:label=charter"
        cached = self._cache.get(cache_key)
        if cached and time.time() - cached[0] < self.cache_ttl:
            return _rerank_strategy_context(cached[1], query) if cached[1] else None

        try:
            repo_payload = self._get(f"/repos/{STRATEGY_CHARTER_REPO.full_name}")
            issues_payload = self._get_list(
                f"/repos/{STRATEGY_CHARTER_REPO.full_name}/issues?state=all&labels=charter&sort=updated&direction=desc&per_page=30"
            )
        except Exception:
            self._cache[cache_key] = (time.time(), None)
            return None

        snippets: list[DraftSnippet] = []
        for issue in _rank_strategy_issues(issues_payload, query)[:8]:
            if not isinstance(issue, dict) or issue.get("pull_request"):
                continue
            number = issue.get("number")
            title = str(issue.get("title") or "")
            if not number or not title:
                continue
            labels = [
                str(label.get("name"))
                for label in issue.get("labels", [])
                if isinstance(label, dict) and label.get("name")
            ]
            status = _strategy_issue_status(issue, labels)
            snippets.append(
                DraftSnippet(
                    path=f"issues/{number}",
                    title=title,
                    text=_strategy_issue_text(issue, labels, status),
                    url=str(issue.get("html_url")) if issue.get("html_url") else None,
                )
            )

        context = DraftContext(
            repo_full_name=STRATEGY_CHARTER_REPO.full_name,
            repo_url=str(repo_payload.get("html_url") or STRATEGY_CHARTER_REPO.html_url),
            resolved_from=STRATEGY_CHARTER_REPO.resolved_from,
            default_branch=str(repo_payload.get("default_branch")) if repo_payload.get("default_branch") else None,
            description=str(repo_payload.get("description")) if repo_payload.get("description") else None,
            homepage=str(repo_payload.get("homepage")) if repo_payload.get("homepage") else None,
            open_issues_count=(
                int(repo_payload["open_issues_count"])
                if isinstance(repo_payload.get("open_issues_count"), int)
                else None
            ),
            snippets=snippets,
            retrieval_hints=[
                "w3c/strategy",
                "charter label",
                "charter review issue tracker",
                "recharter issue tracker",
                "closed charter issues",
                "charter renewal timing",
                "Horizontal review requested",
                "horizontal review completed labels",
                "TiLT review readiness",
                "https://github.com/w3c/strategy/issues?q=label%3Acharter",
            ],
            confidence=0.88,
        )
        self._cache[cache_key] = (time.time(), context)
        return context

    def _context_for_repo(self, candidate: RepoCandidate) -> DraftContext | None:
        cached = self._cache.get(candidate.full_name)
        if cached and time.time() - cached[0] < self.cache_ttl:
            return cached[1]

        try:
            repo_payload = self._get(f"/repos/{candidate.full_name}")
        except Exception:
            self._cache[candidate.full_name] = (time.time(), None)
            return None

        default_branch = str(repo_payload.get("default_branch") or "main")
        snippets = self._snippets(candidate, default_branch)
        latest_commit_sha = self._latest_commit_sha(candidate, default_branch)
        retrieval_hints = _retrieval_hints(candidate, repo_payload, snippets)

        context = DraftContext(
            repo_full_name=candidate.full_name,
            repo_url=str(repo_payload.get("html_url") or candidate.html_url),
            resolved_from=candidate.resolved_from,
            default_branch=default_branch,
            description=str(repo_payload.get("description")) if repo_payload.get("description") else None,
            homepage=str(repo_payload.get("homepage")) if repo_payload.get("homepage") else None,
            latest_commit_sha=latest_commit_sha,
            open_issues_count=(
                int(repo_payload["open_issues_count"])
                if isinstance(repo_payload.get("open_issues_count"), int)
                else None
            ),
            snippets=snippets,
            retrieval_hints=retrieval_hints,
            confidence=0.82 if snippets else 0.62,
        )
        self._cache[candidate.full_name] = (time.time(), context)
        return context

    def _snippets(self, candidate: RepoCandidate, branch: str) -> list[DraftSnippet]:
        snippets: list[DraftSnippet] = []
        for path in SPEC_SOURCE_FILES:
            if len(snippets) >= self.max_files:
                break
            try:
                payload = self._get(f"/repos/{candidate.full_name}/contents/{path}?ref={branch}")
            except Exception:
                continue
            if not isinstance(payload, dict) or payload.get("type") != "file":
                continue
            text = _decode_content(payload, self.max_file_bytes)
            if not text:
                continue
            snippets.append(
                DraftSnippet(
                    path=path,
                    title=_title_from_text(path, text),
                    text=_compact_text(text, 1400),
                    url=str(payload.get("html_url")) if payload.get("html_url") else None,
                )
            )
        return snippets

    def _latest_commit_sha(self, candidate: RepoCandidate, branch: str) -> str | None:
        try:
            payload = self._get(f"/repos/{candidate.full_name}/commits/{branch}")
        except Exception:
            return None
        sha = payload.get("sha") if isinstance(payload, dict) else None
        return str(sha)[:12] if sha else None

    def _get(self, path: str) -> dict[str, object]:
        payload = self._get_json(path)
        return payload if isinstance(payload, dict) else {}

    def _get_list(self, path: str) -> list[object]:
        payload = self._get_json(path)
        return payload if isinstance(payload, list) else []

    def _get_json(self, path: str) -> object:
        parsed_base = urlparse(self.api_base_url)
        if parsed_base.netloc != "api.github.com":
            raise ValueError("GitHub draft context only accepts api.github.com")
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        response = httpx.get(f"{self.api_base_url}{path}", headers=headers, timeout=self.timeout)
        response.raise_for_status()
        return response.json()


def build_draft_context_augmented_query(query: str, contexts: list[DraftContext]) -> str:
    if not contexts:
        return query

    lines = [
        query,
        "",
        "Resolved official GitHub draft context for retrieval only; not normative Process rules:",
    ]
    for context in contexts:
        lines.append(
            "; ".join(
                part
                for part in [
                    context.repo_full_name,
                    str(context.repo_url),
                    f"default_branch={context.default_branch}" if context.default_branch else "",
                    f"latest_commit={context.latest_commit_sha}" if context.latest_commit_sha else "",
                    context.description or "",
                    f"retrieval_hints={', '.join(context.retrieval_hints)}" if context.retrieval_hints else "",
                ]
                if part
            )
        )
        for snippet in context.snippets[:3]:
            lines.append(f"- {snippet.path}: {snippet.title or _compact_text(snippet.text, 120)}")
    return "\n".join(lines)


def _should_resolve_draft_context(query: str, entities: list[W3CEntity], task_plan: TaskPlan) -> bool:
    if not any(entity.entity_type == "specification" for entity in entities):
        return False
    text = f"{query} {task_plan.user_goal} {task_plan.spec_or_group or ''}".lower()
    return any(marker in text for marker in CONTEXT_QUERY_MARKERS)


def _should_resolve_strategy_charter_context(query: str, task_plan: TaskPlan) -> bool:
    text = f"{query} {task_plan.user_goal} {task_plan.spec_or_group or ''}".lower()
    return task_plan.intent_type == "charter_or_recharter" or any(
        marker in text for marker in ["charter", "recharter", "章程"]
    )


def _rank_strategy_issues(issues: list[object], query: str) -> list[dict[str, object]]:
    query_terms = {
        term
        for term in re.findall(r"[a-z0-9][a-z0-9-]{1,}", query.lower())
        if term not in {"charter", "recharter", "review", "group", "working", "issue", "track", "tracking"}
    }
    ranked: list[tuple[int, dict[str, object]]] = []
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        title = str(issue.get("title") or "").lower()
        labels = " ".join(
            str(label.get("name"))
            for label in issue.get("labels", [])
            if isinstance(label, dict) and label.get("name")
        ).lower()
        score = 0
        score += sum(4 for term in query_terms if term in title)
        score += sum(1 for term in query_terms if term in labels)
        if issue.get("state") == "open":
            score += 1
        ranked.append((score, issue))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [issue for _, issue in ranked]


def _strategy_issue_status(issue: dict[str, object], labels: list[str]) -> dict[str, object]:
    lowered = {label.lower() for label in labels}
    completed_reviews = [
        label
        for label in labels
        if "review completed" in label.lower()
    ]
    needs_resolution = [
        label
        for label in labels
        if label.lower().endswith("-needs-resolution")
    ]
    horizontal_requested = "horizontal review requested" in lowered
    tilt_candidate = horizontal_requested and bool(completed_reviews) and not needs_resolution
    return {
        "created_days_ago": _days_ago(issue.get("created_at")),
        "updated_days_ago": _days_ago(issue.get("updated_at")),
        "closed_days_ago": _days_ago(issue.get("closed_at")),
        "horizontal_requested": horizontal_requested,
        "completed_reviews": completed_reviews,
        "needs_resolution": needs_resolution,
        "in_charter_refinement": "in charter refinement" in lowered,
        "ac_review": "ac review" in lowered,
        "tilt_candidate": tilt_candidate,
    }


def _strategy_issue_text(
    issue: dict[str, object],
    labels: list[str],
    status: dict[str, object],
) -> str:
    completed = status["completed_reviews"]
    needs_resolution = status["needs_resolution"]
    timing = [
        f"created_at={issue.get('created_at') or '(unknown)'}",
        f"updated_at={issue.get('updated_at') or '(unknown)'}",
    ]
    if issue.get("closed_at"):
        timing.append(f"closed_at={issue.get('closed_at')}")
    day_counts = [
        f"created_days_ago={status['created_days_ago']}",
        f"updated_days_ago={status['updated_days_ago']}",
    ]
    if status["closed_days_ago"] is not None:
        day_counts.append(f"closed_days_ago={status['closed_days_ago']}")
    return (
        f"w3c/strategy issue #{issue.get('number')}; state={issue.get('state')}; "
        f"labels={', '.join(labels) or '(none)'}; "
        f"{'; '.join(timing)}; {'; '.join(day_counts)}; "
        f"horizontal_review_requested={status['horizontal_requested']}; "
        f"completed_horizontal_reviews={', '.join(completed) if completed else '(none)'}; "
        f"needs_resolution_labels={', '.join(needs_resolution) if needs_resolution else '(none)'}; "
        f"in_charter_refinement={status['in_charter_refinement']}; "
        f"ac_review={status['ac_review']}; "
        f"tilt_readiness_signal={'possible_staff_contact_tilt_check' if status['tilt_candidate'] else 'not_yet_clear'}."
    )


def _days_ago(value: object) -> int | None:
    if not value:
        return None
    try:
        parsed = time.strptime(str(value), "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None
    timestamp = time.mktime(parsed)
    return max(0, int((time.time() - timestamp) // 86400))


def _rerank_strategy_context(context: DraftContext, query: str) -> DraftContext:
    if context.repo_full_name != STRATEGY_CHARTER_REPO.full_name:
        return context
    ranked_snippets = [
        snippet
        for _, snippet in sorted(
            (
                (
                    sum(
                        1
                        for term in re.findall(r"[a-z0-9][a-z0-9-]{1,}", query.lower())
                        if term in f"{snippet.title} {snippet.text}".lower()
                    ),
                    snippet,
                )
                for snippet in context.snippets
            ),
            key=lambda item: item[0],
            reverse=True,
        )
    ]
    context.snippets = ranked_snippets
    return context


def _repo_candidates_from_entities(entities: list[W3CEntity], allowed_orgs: set[str]) -> list[RepoCandidate]:
    candidates: list[RepoCandidate] = []
    for entity in entities:
        for value in [entity.editor_draft_url, entity.public_url, *entity.retrieval_hints]:
            if not value:
                continue
            candidate = _repo_candidate_from_url(str(value), allowed_orgs)
            if candidate:
                candidates.append(candidate)
    return candidates


def _repo_candidate_from_url(value: str, allowed_orgs: set[str]) -> RepoCandidate | None:
    parsed = urlparse(value)
    host = parsed.netloc.lower()
    path_parts = [part for part in parsed.path.split("/") if part]

    if host == "github.com" and len(path_parts) >= 2:
        owner, repo = path_parts[0].lower(), path_parts[1]
        if owner in allowed_orgs:
            return RepoCandidate(owner=owner, repo=repo, resolved_from=value)

    if host.endswith(".github.io") and path_parts:
        owner = host.removesuffix(".github.io").lower()
        repo = path_parts[0]
        if owner in allowed_orgs:
            return RepoCandidate(owner=owner, repo=repo, resolved_from=value)

    return None


def _decode_content(payload: dict[str, object], max_file_bytes: int) -> str:
    if payload.get("encoding") != "base64" or not payload.get("content"):
        return ""
    try:
        raw = base64.b64decode(str(payload["content"]), validate=False)
    except Exception:
        return ""
    if not raw:
        return ""
    return raw[:max_file_bytes].decode("utf-8", errors="ignore")


def _title_from_text(path: str, text: str) -> str | None:
    for line in text.splitlines()[:80]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()[:140]
        if stripped.lower().startswith("title:"):
            return stripped.split(":", 1)[1].strip()[:140]
        match = re.search(r"<title>(.*?)</title>", stripped, re.I)
        if match:
            return match.group(1).strip()[:140]
    return path


def _compact_text(text: str, limit: int) -> str:
    compacted = " ".join(text.split())
    if len(compacted) <= limit:
        return compacted
    return f"{compacted[:limit].rsplit(' ', 1)[0]}..."


def _retrieval_hints(
    candidate: RepoCandidate,
    repo_payload: dict[str, object],
    snippets: list[DraftSnippet],
) -> list[str]:
    hints = [candidate.full_name]
    if repo_payload.get("homepage"):
        hints.append(str(repo_payload["homepage"]))
    for snippet in snippets[:4]:
        hints.append(snippet.path)
        if snippet.title:
            hints.append(snippet.title)
    return list(dict.fromkeys(hints))[:12]
