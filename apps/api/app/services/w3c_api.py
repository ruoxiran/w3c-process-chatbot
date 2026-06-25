from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from urllib.parse import urlparse

import httpx

from app.core.config import Settings
from app.models.schemas import W3CEntity


@dataclass(frozen=True)
class CatalogItem:
    entity_type: str
    title: str
    api_url: str
    token_score: float


class W3CAPIClient:
    def __init__(self, settings: Settings) -> None:
        self.enabled = settings.w3c_api_enabled
        self.base_url = settings.w3c_api_base_url.rstrip("/")
        self.timeout = settings.w3c_api_timeout_seconds
        self.cache_ttl = settings.w3c_api_cache_ttl_seconds
        self.catalog_pages = max(1, settings.w3c_api_catalog_pages)
        self.persistent_cache_enabled = settings.w3c_api_persistent_cache_enabled
        self.cache_path = Path(settings.w3c_api_cache_path)
        # Instance-level caches + lock. Class-level mutable dicts were a
        # threading race under multi-worker / async deployments.
        self._catalog_cache: dict[str, tuple[float, list[CatalogItem]]] = {}
        self._detail_cache: dict[str, tuple[float, dict[str, object]]] = {}
        self._persistent_loaded_paths: set[str] = set()
        self._cache_lock = threading.Lock()
        if self.persistent_cache_enabled:
            self._load_persistent_cache()

    def resolve_entities(self, query: str, limit: int = 5) -> list[W3CEntity]:
        if not self.enabled:
            return []

        terms = _terms(query)
        if not terms:
            return []

        entities: list[W3CEntity] = []
        seen: set[str] = set()
        for shortname in _direct_shortname_candidates(terms):
            api_url = f"{self.base_url}/specifications/{shortname}"
            try:
                detail = self._detail(api_url)
            except Exception:
                continue
            if not detail or api_url in seen:
                continue
            seen.add(api_url)
            entity = self._entity_from_item(
                CatalogItem(
                    entity_type="specification",
                    title=str(detail.get("title") or shortname),
                    api_url=api_url,
                    token_score=4.0,
                ),
                confidence=0.95,
            )
            if entity:
                entities.append(entity)
            if len(entities) >= limit:
                return entities

        if entities:
            return entities

        candidates: list[tuple[float, CatalogItem]] = []
        for item in self._catalog():
            score = _match_score(terms, item.title, item.api_url) + item.token_score
            if score >= 4 and _has_meaningful_overlap(terms, item.title, item.api_url):
                candidates.append((score, item))

        candidates.sort(key=lambda hit: hit[0], reverse=True)
        for score, item in candidates[: limit * 2]:
            if item.api_url in seen:
                continue
            seen.add(item.api_url)
            entity = self._entity_from_item(item, min(0.95, score / 12))
            if entity:
                entities.append(entity)
            if len(entities) >= limit:
                break
        return entities

    def _catalog(self) -> list[CatalogItem]:
        cache_key = f"{self.base_url}:catalog:{self.catalog_pages}"
        # Atomic TTL check under the lock so two concurrent callers can't both
        # miss and trigger duplicate catalog fetches.
        with self._cache_lock:
            cached = self._catalog_cache.get(cache_key)
            if cached and time.time() - cached[0] < self.cache_ttl:
                return cached[1]

        items: list[CatalogItem] = []
        items.extend(self._catalog_endpoint("specifications", "specification"))
        items.extend(self._catalog_endpoint("groups", "group"))
        with self._cache_lock:
            self._catalog_cache[cache_key] = (time.time(), items)
        self._save_persistent_cache()
        return items

    def _catalog_endpoint(self, endpoint: str, entity_type: str) -> list[CatalogItem]:
        output: list[CatalogItem] = []
        for page in range(1, self.catalog_pages + 1):
            payload = self._get(f"/{endpoint}?items=1000&page={page}")
            links = ((payload.get("_links") or {}).get(endpoint) or []) if isinstance(payload, dict) else []
            for link in links:
                if not isinstance(link, dict):
                    continue
                href = str(link.get("href") or "")
                title = str(link.get("title") or "")
                if not href or not title:
                    continue
                output.append(
                    CatalogItem(
                        entity_type=entity_type,
                        title=title,
                        api_url=href,
                        token_score=_specificity_score(title, href),
                    )
                )
        return output

    def _entity_from_item(self, item: CatalogItem, confidence: float) -> W3CEntity | None:
        detail = self._detail(item.api_url)
        title = str(detail.get("title") or detail.get("name") or item.title)
        shortname = detail.get("shortname")
        description = _clean_description(detail.get("description"))
        links = detail.get("_links") if isinstance(detail.get("_links"), dict) else {}
        latest_version_url = None
        latest_version_date = None
        process_rules_url = None
        deliverers: list[str] = []
        charter_url = None
        charter_end = None
        patent_policy_url = None
        team_contacts: list[str] = []

        if item.entity_type == "specification":
            latest = links.get("latest-version") if isinstance(links.get("latest-version"), dict) else {}
            editor_draft_url = str(detail.get("editor-draft")) if detail.get("editor-draft") else None
            public_url = detail.get("shortlink") or editor_draft_url
            status = latest.get("title") if isinstance(latest, dict) else None
            latest_version_url = latest.get("href") if isinstance(latest, dict) else None
            if latest_version_url:
                latest_detail = self._detail(str(latest_version_url))
                latest_version_date = str(latest_detail.get("date")) if latest_detail.get("date") else None
                process_rules_url = (
                    str(latest_detail.get("process-rules")) if latest_detail.get("process-rules") else None
                )
                deliverer_link = ((latest_detail.get("_links") or {}).get("deliverers") or {}) if isinstance(latest_detail.get("_links"), dict) else {}
                if isinstance(deliverer_link, dict) and deliverer_link.get("href"):
                    deliverers = self._link_titles(str(deliverer_link["href"]), "deliverers")
        else:
            homepage = links.get("homepage") if isinstance(links.get("homepage"), dict) else {}
            public_url = homepage.get("href") if isinstance(homepage, dict) else None
            status = "closed" if detail.get("is_closed") else "active"
            active_charter = links.get("active-charter") if isinstance(links.get("active-charter"), dict) else {}
            if isinstance(active_charter, dict) and active_charter.get("href"):
                charter_detail = self._detail(str(active_charter["href"]))
                charter_url = str(charter_detail.get("uri")) if charter_detail.get("uri") else None
                charter_end = str(charter_detail.get("end")) if charter_detail.get("end") else None
                patent_policy_url = (
                    str(charter_detail.get("patent-policy")) if charter_detail.get("patent-policy") else None
                )
            team_contacts_link = links.get("team-contacts") if isinstance(links.get("team-contacts"), dict) else {}
            if isinstance(team_contacts_link, dict) and team_contacts_link.get("href"):
                team_contacts = self._link_titles(str(team_contacts_link["href"]), "team-contacts")

        resolved_shortname = str(shortname) if shortname else _shortname_from_api_url(item.api_url)
        resolved_status = str(status) if status else None
        resolved_public_url = str(public_url) if public_url else None
        resolved_process_rules_url = process_rules_url
        resolved_group_type = str(detail.get("type")) if detail.get("type") else None
        retrieval_hints = _retrieval_hints(
            shortname=resolved_shortname,
            status=resolved_status,
            latest_version_date=latest_version_date,
            group_type=resolved_group_type,
            charter_end=charter_end,
            public_url=resolved_public_url,
            process_rules_url=str(resolved_process_rules_url) if resolved_process_rules_url else None,
            deliverers=deliverers,
            team_contacts=team_contacts,
            charter_url=charter_url,
            patent_policy_url=patent_policy_url,
        )
        return W3CEntity(
            entity_type=item.entity_type,
            title=title,
            shortname=resolved_shortname,
            api_url=item.api_url,
            public_url=resolved_public_url,
            editor_draft_url=editor_draft_url if item.entity_type == "specification" else None,
            status=resolved_status,
            latest_version_url=str(latest_version_url) if latest_version_url else None,
            latest_version_date=latest_version_date,
            process_rules_url=resolved_process_rules_url,
            deliverers=deliverers,
            charter_url=charter_url,
            charter_end=charter_end,
            patent_policy_url=patent_policy_url,
            team_contacts=team_contacts,
            group_type=resolved_group_type,
            description=description,
            confidence=confidence,
            retrieval_hints=retrieval_hints,
        )

    def _link_titles(self, api_url: str, relation: str) -> list[str]:
        payload = self._detail(api_url)
        links = payload.get("_links") if isinstance(payload.get("_links"), dict) else {}
        values = links.get(relation) if isinstance(links, dict) else []
        if not isinstance(values, list):
            return []
        return [str(item.get("title")) for item in values if isinstance(item, dict) and item.get("title")][:5]

    def _detail(self, api_url: str) -> dict[str, object]:
        with self._cache_lock:
            cached = self._detail_cache.get(api_url)
            if cached and time.time() - cached[0] < self.cache_ttl:
                return cached[1]
        payload = self._get(api_url)
        detail = payload if isinstance(payload, dict) else {}
        with self._cache_lock:
            self._detail_cache[api_url] = (time.time(), detail)
        self._save_persistent_cache()
        return detail

    def _load_persistent_cache(self) -> None:
        cache_id = str(self.cache_path.resolve())
        if cache_id in self._persistent_loaded_paths:
            return
        self._persistent_loaded_paths.add(cache_id)
        if not self.cache_path.exists():
            return
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return

        catalog = payload.get("catalog") if isinstance(payload, dict) else {}
        if isinstance(catalog, dict):
            for key, value in catalog.items():
                if not isinstance(value, dict):
                    continue
                timestamp = float(value.get("timestamp") or 0)
                raw_items = value.get("items") if isinstance(value.get("items"), list) else []
                items = [
                    CatalogItem(
                        entity_type=str(item.get("entity_type")),
                        title=str(item.get("title")),
                        api_url=str(item.get("api_url")),
                        token_score=float(item.get("token_score") or 0),
                    )
                    for item in raw_items
                    if isinstance(item, dict) and item.get("entity_type") and item.get("title") and item.get("api_url")
                ]
                if items:
                    self._catalog_cache[str(key)] = (timestamp, items)

        detail = payload.get("detail") if isinstance(payload, dict) else {}
        if isinstance(detail, dict):
            for url, value in detail.items():
                if not isinstance(value, dict) or not isinstance(value.get("payload"), dict):
                    continue
                self._detail_cache[str(url)] = (float(value.get("timestamp") or 0), value["payload"])

    def _save_persistent_cache(self) -> None:
        if not self.persistent_cache_enabled:
            return
        with self._cache_lock:
            payload = {
                "catalog": {
                    key: {
                        "timestamp": timestamp,
                        "items": [asdict(item) for item in items],
                    }
                    for key, (timestamp, items) in self._catalog_cache.items()
                },
                "detail": {
                    url: {
                        "timestamp": timestamp,
                        "payload": detail,
                    }
                    for url, (timestamp, detail) in self._detail_cache.items()
                },
            }
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp_path.replace(self.cache_path)
        except OSError:
            return

    def _get(self, path_or_url: str) -> dict[str, object]:
        url = path_or_url if path_or_url.startswith("https://") else f"{self.base_url}{path_or_url}"
        parsed = urlparse(url)
        if parsed.netloc != "api.w3.org":
            raise ValueError("W3C API client only accepts api.w3.org URLs")
        response = httpx.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.json()


def _terms(query: str) -> set[str]:
    raw = re.findall(r"[a-z0-9][a-z0-9-]{1,}|[\u4e00-\u9fff]{2,}", query.lower())
    ignored = {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "for",
        "about",
        "how",
        "now",
        "to",
        "in",
        "it",
        "what",
        "should",
        "next",
        "from",
        "with",
        "does",
        "need",
        "process",
        "guidebook",
        "w3c",
        "wai",
        "specification",
        "standard",
        "working",
        "group",
        "do",
        "publish",
        "published",
        "publication",
        "cr",
        "rec",
    }
    terms: set[str] = set()
    for term in raw:
        if term not in ignored:
            terms.add(term)
        if "-" in term:
            terms.update(part for part in term.split("-") if part and part not in ignored)
    return terms


def _match_score(terms: set[str], title: str, api_url: str) -> float:
    shortname = _shortname_from_api_url(api_url).lower()
    title_tokens = set(re.findall(r"[a-z0-9]+", title.lower()))
    short_tokens = set(shortname.replace("-", " ").split())
    all_tokens = title_tokens | short_tokens | {shortname}
    score = 0.0
    for term in terms:
        if term == shortname:
            score += 12
        elif "-" in term and term in shortname:
            score += 8
        elif term in title_tokens:
            score += 3
        elif term in short_tokens:
            score += 1.5
    if terms and all(term in all_tokens for term in terms if len(term) > 2):
        score += 2
    return score


def _has_meaningful_overlap(terms: set[str], title: str, api_url: str) -> bool:
    shortname = _shortname_from_api_url(api_url).lower()
    title_tokens = set(re.findall(r"[a-z0-9]+", title.lower()))
    short_tokens = set(shortname.replace("-", " ").split())
    meaningful_terms = {term for term in terms if len(term) > 2}
    if not meaningful_terms:
        return False
    if any(term == shortname or ("-" in term and term in shortname) for term in meaningful_terms):
        return True
    searchable_tokens = _token_variants(title_tokens | short_tokens)
    overlap = _token_variants(meaningful_terms).intersection(searchable_tokens)
    return len(overlap) >= 2 or (len(overlap) == 1 and len(meaningful_terms) == 1)


def _token_variants(tokens: set[str]) -> set[str]:
    variants = set(tokens)
    for token in tokens:
        if token.endswith("s") and len(token) > 3:
            variants.add(token[:-1])
        elif len(token) > 3:
            variants.add(f"{token}s")
    return variants


def _direct_shortname_candidates(terms: set[str]) -> list[str]:
    candidates: list[str] = []
    for term in terms:
        if "-" in term and len(term) > 3:
            candidates.append(term)

    if ("wai-adapt" in terms or "adapt" in terms) and ("symbol" in terms or "symbols" in terms):
        candidates.append("adapt-symbols")
    if ("wai-adapt" in terms or "adapt" in terms) and "content" in terms:
        candidates.append("adapt-content")
    if ("wai-adapt" in terms or "adapt" in terms) and "help" in terms:
        candidates.append("adapt-help")
    if ("wai-adapt" in terms or "adapt" in terms) and "tools" in terms:
        candidates.append("adapt-tools")

    tokens = sorted(term for term in terms if len(term) > 2 and "-" not in term and term not in {"wai"})
    for left in tokens:
        for right in tokens:
            if left == right:
                continue
            candidates.append(f"{left}-{right}")
            candidates.append(f"{left}-{right}s")

    return list(dict.fromkeys(candidates))[:24]


def _specificity_score(title: str, api_url: str) -> float:
    shortname = _shortname_from_api_url(api_url)
    score = 0.0
    if "-" in shortname:
        score += 0.5
    if any(char.isdigit() for char in shortname):
        score += 0.3
    if len(title.split()) <= 6:
        score += 0.2
    return score


def _shortname_from_api_url(api_url: str) -> str:
    return urlparse(api_url).path.rstrip("/").split("/")[-1]


def _clean_description(value: object) -> str | None:
    if not value:
        return None
    text = unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = " ".join(text.split())
    if len(text) > 320:
        return f"{text[:320].rsplit(' ', 1)[0]}..."
    return text or None


def _retrieval_hints(
    *,
    shortname: str | None,
    status: str | None,
    latest_version_date: str | None,
    group_type: str | None,
    charter_end: str | None,
    public_url: str | None,
    process_rules_url: str | None,
    deliverers: list[str],
    team_contacts: list[str],
    charter_url: str | None,
    patent_policy_url: str | None,
) -> list[str]:
    hints: list[str] = []
    values: list[str | None] = [
        shortname,
        status,
        latest_version_date,
        group_type,
        charter_end,
        public_url,
        process_rules_url,
        *deliverers,
        *team_contacts,
    ]
    status_lc = (status or "").lower()
    if "candidate recommendation" in status_lc:
        values.extend(["transitioning to Recommendation", "AC Review", "Exclusion Opportunity"])
    if "working draft" in status_lc:
        values.extend(["wide review", "horizontal review"])
    if charter_url:
        values.append("active charter")
    if patent_policy_url:
        values.append("Patent Policy")

    seen: set[str] = set()
    for value in values:
        if not value:
            continue
        normalized = str(value).strip()
        key = normalized.lower()
        if normalized and key not in seen:
            hints.append(normalized)
            seen.add(key)
    return hints[:12]
