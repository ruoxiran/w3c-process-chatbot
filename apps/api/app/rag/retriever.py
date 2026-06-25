import json
import logging
import math
import re
import threading
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from app.core.config import Settings, get_settings
from app.core.paths import resolve_data_path
from app.models.schemas import Citation, SourceType
from app.rag.guide_topics import matching_guide_topics
from app.services.embeddings import OllamaEmbeddingClient


logger = logging.getLogger(__name__)
_QUERY_EMBEDDING_CACHE_LIMIT = 256


# BM25 tuning parameters
_BM25_K1 = 1.45
_BM25_B = 0.72

# Rerank score weights
_SEMANTIC_WEIGHT = 20
_SOURCE_PRIORITY_PROCESS = 4
_SOURCE_PRIORITY_GUIDE = 2
_HEADING_OVERLAP_PER_TOKEN = 2
_HEADING_OVERLAP_CAP = 10


DEFAULT_PROCESS_CITATION = Citation(
    title="W3C Process Document",
    url="https://www.w3.org/policies/process/",
    source_type=SourceType.process,
    heading_path="Latest operative W3C Process Document",
)

DEFAULT_GUIDE_CITATION = Citation(
    title="The Art of Consensus: W3C Guidebook",
    url="https://www.w3.org/guide/",
    source_type=SourceType.guide,
    heading_path="W3C Guidebook",
)


@dataclass(frozen=True)
class CorpusRecord:
    chunk: dict[str, object]
    title: str
    heading: str
    body: str
    url: str
    source_type: str
    tokens: Counter[str]
    length: int


@dataclass(frozen=True)
class CorpusIndex:
    records: list[CorpusRecord]
    document_frequency: Counter[str]
    average_length: float
    mtime: float


@dataclass(frozen=True)
class DenseEmbeddingCache:
    vectors: dict[str, list[float]]
    model: str
    mtime: float


class Retriever:
    """Retrieval facade.

    The first implementation returns safe authoritative entry points when Qdrant
    has not yet been populated. The interface is deliberately small so a
    LlamaIndex/Qdrant-backed retriever can replace this without changing the
    workflow contract.
    """

    def __init__(
        self,
        corpus_path: str | None = None,
        *,
        settings: Settings | None = None,
        embedding_client: OllamaEmbeddingClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.corpus_path = resolve_data_path(corpus_path or self.settings.corpus_path)
        self.embedding_cache_path = resolve_data_path(self.settings.retrieval_embedding_cache_path)
        self.embedding_model = self.settings.ollama_embedding_model or self.settings.embedding_model
        self.embedding_client = embedding_client or OllamaEmbeddingClient(
            self.settings.ollama_base_url,
            self.settings.ollama_timeout_seconds,
        )
        self._index: CorpusIndex | None = None
        self._dense_cache: DenseEmbeddingCache | None = None
        # Bounded LRU. Workflow is a singleton now, so without an upper bound
        # this would grow with every distinct query for the life of the process.
        self._query_embeddings: OrderedDict[str, list[float] | None] = OrderedDict()
        self._index_lock = threading.Lock()
        self._dense_cache_lock = threading.Lock()
        self._query_embedding_lock = threading.Lock()

    def retrieve(self, query: str) -> list[Citation]:
        corpus_hits = self._retrieve_from_corpus(query)
        if corpus_hits:
            return corpus_hits

        text = query.lower()
        citations = [DEFAULT_PROCESS_CITATION]
        if "guide" in text or "指南" in text or "practice" in text or "怎么" in text:
            citations.append(DEFAULT_GUIDE_CITATION)
        return citations

    def _retrieve_from_corpus(self, query: str, limit: int = 10) -> list[Citation]:
        index = self._load_index()
        if not index.records:
            return []

        query_terms = _query_terms(query)
        if not query_terms:
            return []

        query_vector = _tfidf_vector(Counter(_tokenize(query)), index.document_frequency, len(index.records))
        dense_cache = self._load_dense_cache() if self.settings.retrieval_dense_enabled else None
        dense_query = self._query_embedding(query) if dense_cache and dense_cache.vectors else None
        candidates: list[tuple[float, CorpusRecord, dict[str, float]]] = []
        for record in index.records:
            bm25 = _bm25_score(query_terms, record, index)
            dense = 0.0
            if dense_query and dense_cache:
                dense = _dense_cosine(dense_query, dense_cache.vectors.get(_hit_id(record.chunk)))
            # BM25=0 implies semantic=0 (identical vocabulary), so skip both when neither scores
            if bm25 <= 0 and dense <= 0:
                continue
            # Only compute the expensive TF-IDF cosine when BM25 found term overlap
            semantic = (
                _cosine(query_vector, _tfidf_vector(record.tokens, index.document_frequency, len(index.records)))
                if bm25 > 0
                else 0.0
            )
            rerank = _rerank_score(
                query,
                record,
                bm25,
                semantic,
                dense,
                self.settings.retrieval_dense_weight if dense_query else 0.0,
            )
            if rerank > 0:
                candidates.append((rerank, record, {"bm25": bm25, "semantic": semantic, "dense": dense}))

        candidates.sort(key=lambda item: item[0], reverse=True)
        selected = _balanced_hits(candidates, limit)
        selected = _ensure_topic_coverage(query, selected, candidates, limit)
        citations: list[Citation] = []
        seen: set[str] = set()
        for _, record, _scores in selected:
            chunk = record.chunk
            url = str(chunk.get("source_url") or "")
            if not url or url in seen:
                continue
            seen.add(url)
            citations.append(
                Citation(
                    title=str(chunk.get("title") or _title_from_url(url)),
                    url=url,
                    source_type=_source_type(str(chunk.get("source_type") or "repo")),
                    section_id=chunk.get("section_id") if isinstance(chunk.get("section_id"), str) else None,
                    heading_path=chunk.get("heading_path") if isinstance(chunk.get("heading_path"), str) else None,
                    commit_sha=chunk.get("commit_sha") if isinstance(chunk.get("commit_sha"), str) else None,
                    published_version_date=(
                        chunk.get("published_version_date")
                        if isinstance(chunk.get("published_version_date"), str)
                        else None
                    ),
                    quote=str(chunk.get("text") or "")[:420],
                )
            )
        return _ensure_topic_entrypoint_citations(query, citations, limit)

    def _query_embedding(self, query: str) -> list[float] | None:
        with self._query_embedding_lock:
            if query in self._query_embeddings:
                # LRU touch
                self._query_embeddings.move_to_end(query)
                return self._query_embeddings[query]
        try:
            embedding = self.embedding_client.embed(model=self.embedding_model, text=_embedding_text(query, max_chars=4000))
        except Exception as exc:
            logger.warning("Query embedding failed; falling back to lexical retrieval only", exc_info=exc)
            embedding = None
        with self._query_embedding_lock:
            self._query_embeddings[query] = embedding
            self._query_embeddings.move_to_end(query)
            while len(self._query_embeddings) > _QUERY_EMBEDDING_CACHE_LIMIT:
                self._query_embeddings.popitem(last=False)
        return embedding

    def _load_dense_cache(self) -> DenseEmbeddingCache | None:
        if not self.embedding_cache_path.exists():
            return None
        mtime = self.embedding_cache_path.stat().st_mtime
        if self._dense_cache and self._dense_cache.mtime == mtime:
            return self._dense_cache
        with self._dense_cache_lock:
            if self._dense_cache and self._dense_cache.mtime == mtime:
                return self._dense_cache
            vectors: dict[str, list[float]] = {}
            model = ""
            with self.embedding_cache_path.open("r", encoding="utf-8") as cache:
                for line in cache:
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if payload.get("model") and not model:
                        model = str(payload["model"])
                    if model and payload.get("model") != model:
                        continue
                    chunk_id = payload.get("chunk_id")
                    vector = payload.get("embedding")
                    if not isinstance(chunk_id, str) or not isinstance(vector, list):
                        continue
                    values = [float(value) for value in vector if isinstance(value, (int, float))]
                    if values:
                        vectors[chunk_id] = values
            if model and model != self.embedding_model:
                return None
            self._dense_cache = DenseEmbeddingCache(vectors=vectors, model=model or self.embedding_model, mtime=mtime)
            return self._dense_cache

    def _load_index(self) -> CorpusIndex:
        if not self.corpus_path.exists():
            return CorpusIndex(records=[], document_frequency=Counter(), average_length=0, mtime=0)

        mtime = self.corpus_path.stat().st_mtime
        # Fast path: index is already loaded and the file hasn't changed.
        if self._index and self._index.mtime == mtime:
            return self._index
        # Slow path: take the lock and re-check before doing the expensive parse.
        with self._index_lock:
            if self._index and self._index.mtime == mtime:
                return self._index
            records: list[CorpusRecord] = []
            document_frequency: Counter[str] = Counter()
            with self.corpus_path.open("r", encoding="utf-8") as corpus:
                for line in corpus:
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    if _is_toc_chunk(chunk) or _is_low_quality_chunk(chunk):
                        continue
                    title = str(chunk.get("title", "")).lower()
                    heading = str(chunk.get("heading_path", "")).lower()
                    body = str(chunk.get("text", "")).lower()
                    url = str(chunk.get("source_url") or "")
                    source_type = str(chunk.get("source_type") or "repo")
                    weighted_text = f"{title} {title} {heading} {heading} {body} {source_type} {url}"
                    tokens = Counter(_tokenize(weighted_text))
                    if not tokens:
                        continue
                    document_frequency.update(tokens.keys())
                    records.append(
                        CorpusRecord(
                            chunk=chunk,
                            title=title,
                            heading=heading,
                            body=body,
                            url=url,
                            source_type=source_type,
                            tokens=tokens,
                            length=sum(tokens.values()),
                        )
                    )
            average_length = sum(record.length for record in records) / len(records) if records else 0
            self._index = CorpusIndex(
                records=records,
                document_frequency=document_frequency,
                average_length=average_length,
                mtime=mtime,
            )
            return self._index


def _query_terms(query: str) -> list[str]:
    words = _tokenize(query)
    chinese_terms = {
        mapped
        for needle, mapped in {
            "推进": "advance",
            "标准": "specification",
            "下一步": "next step",
            "流程": "process",
            "章程": "charter",
            "工作组": "working group",
            "异议": "formal objection",
            "审查": "review",
            "横向审查": "horizontal review",
            "专利": "patent",
            "候选推荐": "candidate recommendation",
            "推荐标准": "recommendation",
        }.items()
        if needle in query
    }
    phrases = {
        phrase
        for phrase in [
            "formal objection",
            "candidate recommendation",
            "working draft",
            "wide review",
            "horizontal review",
            "working group",
            "advisory committee",
            "patent policy",
        ]
        if phrase in query.lower()
    }
    return list(dict.fromkeys([*words, *phrases, *chinese_terms]))


def _tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9][a-z0-9-]{1,}|[\u4e00-\u9fff]{2,}", text.lower())
    normalized: list[str] = []
    for word in words:
        normalized.append(word)
        if "-" in word:
            normalized.extend(part for part in word.split("-") if len(part) > 1)
    return normalized


def _bm25_score(query_terms: list[str], record: CorpusRecord, index: CorpusIndex) -> float:
    if not record.length or not index.average_length:
        return 0
    total = 0.0
    document_count = len(index.records)
    for term in query_terms:
        frequency = record.tokens.get(term, 0)
        if not frequency:
            continue
        document_frequency = index.document_frequency.get(term, 0)
        idf = math.log(1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))
        denominator = frequency + _BM25_K1 * (1 - _BM25_B + _BM25_B * record.length / index.average_length)
        total += idf * (frequency * (_BM25_K1 + 1) / denominator)
    return total


def _tfidf_vector(tokens: Counter[str], document_frequency: Counter[str], document_count: int) -> dict[str, float]:
    if not tokens or document_count <= 0:
        return {}
    vector: dict[str, float] = {}
    total = sum(tokens.values()) or 1
    for term, count in tokens.items():
        df = document_frequency.get(term, 0)
        idf = math.log(1 + document_count / (1 + df))
        vector[term] = (count / total) * idf
    return vector


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0
    shared = set(left).intersection(right)
    numerator = sum(left[term] * right[term] for term in shared)
    if numerator <= 0:
        return 0
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0
    return numerator / (left_norm * right_norm)


def _dense_cosine(left: list[float], right: list[float] | None) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    score = sum(l_value * r_value for l_value, r_value in zip(left, right, strict=True))
    return max(0.0, score)


def _embedding_text(value: str, max_chars: int = 1800) -> str:
    compact = " ".join(value.split())
    return compact[:max_chars]


def _rerank_score(
    query: str,
    record: CorpusRecord,
    bm25: float,
    semantic: float,
    dense: float = 0.0,
    dense_weight: float = 0.0,
) -> float:
    topic = _topic_bonus(query, record.title, record.heading, record.body)
    priority = _source_priority(record.source_type)
    quality = _quality_bonus(record.chunk)
    adjustment = _relevance_adjustment(query, record.chunk, record.title, record.heading, record.body)
    heading_overlap = _heading_overlap(query, record.heading)
    return bm25 + (semantic * _SEMANTIC_WEIGHT) + (dense * dense_weight) + topic + priority + quality + adjustment + heading_overlap


def _heading_overlap(query: str, heading: str) -> int:
    query_tokens = set(_tokenize(query))
    heading_tokens = set(_tokenize(heading))
    if not query_tokens or not heading_tokens:
        return 0
    overlap = len(query_tokens.intersection(heading_tokens))
    return min(overlap * _HEADING_OVERLAP_PER_TOKEN, _HEADING_OVERLAP_CAP)


def _topic_bonus(query: str, title: str, heading: str, body: str) -> int:
    text = query.lower()
    combined = f"{heading} {body}"
    score = 0
    if "formal objection" in text and "formal objection" in heading:
        score += 10
    elif "formal objection" in text and "formal objection" in combined:
        score += 5
    if ("cr" in text or "rec" in text) and "transitioning to recommendation" in heading:
        score += 12
    if (
        "cr" in text
        or "rec" in text
        or "candidate recommendation" in text
        or "recommendation" in text
        or "候选推荐" in text
        or "推荐标准" in text
        or "推进" in text
    ) and "advancing on the recommendation track" in heading:
        score += 12
    if "charter" in text and "charter review and approval" in heading:
        score += 12
    if ("charter" in text or "章程" in text) and "starting a group" in heading:
        score += 10
    if ("working group" in text or "工作组" in text) and "groups" in heading:
        score += 5
    if "patent" in text and "patent" in heading:
        score += 8
    if "wide review" in text and "wide review" in heading:
        score += 8
    if _is_horizontal_review_query(text):
        if "reviews and review responsibilities" in heading:
            score += 34
        if "#doc-reviews" in combined:
            score += 22
        if "how to get horizontal review" in heading:
            score += 30
        if "working with horizontal review labels" in heading:
            score += 28
        if "needs-resolution" in heading or "needs-resolution" in body:
            score += 22
        if "issue trackers" in heading or "tracker boards" in body:
            score += 18
        if "horizontal groups" in heading:
            score += 18
        if "labels and other metadata" in title and "horizontal reviews" in heading:
            score += 18
        if "organize a technical report transition" in title and "horizontal" in body:
            score += 14
    if ("formal objection" in text or "异议" in text) and "formal objection" in combined:
        score += 6
    if ("appeal" in text or "申诉" in text) and ("appeal" in heading or "appeal" in title):
        score += 8
    if ("recharter" in text or "rechartering" in text) and ("charter" in heading or "rechartering" in combined):
        score += 10
    if (
        "fpwd" in text or "first public working draft" in text
    ) and ("first public working draft" in combined or "fpwd" in combined):
        score += 10
    if ("ac review" in text or "advisory committee" in text) and (
        "advisory committee" in heading or "ac review" in combined
    ):
        score += 10
    if ("next step" in text or "下一步" in text) and "next step finder" in combined:
        score += 8
    return score


def _balanced_hits(
    scored: list[tuple[float, CorpusRecord, dict[str, float]]],
    limit: int,
) -> list[tuple[float, CorpusRecord, dict[str, float]]]:
    """Prefer Process authority while still surfacing Guidebook practice guidance."""
    by_source: dict[str, list[tuple[float, CorpusRecord, dict[str, float]]]] = {
        "process": [],
        "guide": [],
        "repo": [],
        "related_policy": [],
    }
    for hit in scored:
        source_type = hit[1].source_type
        by_source.setdefault(source_type, []).append(hit)

    selected: list[tuple[float, CorpusRecord, dict[str, float]]] = []
    selected.extend(by_source.get("process", [])[:4])
    selected.extend(by_source.get("guide", [])[:2])

    seen_ids = {_hit_id(hit[1].chunk) for hit in selected}
    for hit in scored:
        if len(selected) >= limit:
            break
        hit_id = _hit_id(hit[1].chunk)
        if hit_id in seen_ids:
            continue
        selected.append(hit)
        seen_ids.add(hit_id)

    selected.sort(key=lambda item: item[0], reverse=True)
    return selected[:limit]


def _ensure_topic_coverage(
    query: str,
    selected: list[tuple[float, CorpusRecord, dict[str, float]]],
    candidates: list[tuple[float, CorpusRecord, dict[str, float]]],
    limit: int,
) -> list[tuple[float, CorpusRecord, dict[str, float]]]:
    """Keep topic-critical Guidebook pages represented after reranking."""
    topics = matching_guide_topics(query)
    if not topics:
        return selected

    required_url_needles = []
    required_text_needles: list[str] = []
    for topic in topics:
        required_url_needles.extend(topic.required_url_needles)
        required_text_needles.extend(topic.optional_text_needles)

    lowered_query = query.lower()
    if "i18n" in lowered_query or "internationalization" in lowered_query:
        required_text_needles.append("i18n-request")
    if "privacy" in lowered_query:
        required_text_needles.append("privacy-request")
    if "security" in lowered_query:
        required_text_needles.append("security-request")
    if "a11y" in lowered_query or "accessibility" in lowered_query:
        required_text_needles.append("a11y-request")
    if "tag" in lowered_query:
        required_text_needles.append("w3ctag/design-reviews")
    required_url_needles = _dedupe_text(required_url_needles)
    required_text_needles = _dedupe_text(required_text_needles)
    enriched = list(selected)
    seen_ids = {_hit_id(hit[1].chunk) for hit in enriched}
    seen_urls = {str(hit[1].chunk.get("source_url") or "").lower() for hit in enriched}

    for needle in required_url_needles:
        if any(needle in url for url in seen_urls):
            continue
        replacement = next(
            (
                hit
                for hit in candidates
                if needle in str(hit[1].chunk.get("source_url") or "").lower()
                and _hit_id(hit[1].chunk) not in seen_ids
            ),
            None,
        )
        if replacement is None:
            continue
        if len(enriched) < limit:
            enriched.append(replacement)
        else:
            victim_index = _replacement_victim_index(enriched, required_url_needles)
            enriched[victim_index] = replacement
        seen_ids.add(_hit_id(replacement[1].chunk))
        seen_urls.add(str(replacement[1].chunk.get("source_url") or "").lower())
        enriched.sort(key=lambda item: item[0], reverse=True)

    for needle in required_text_needles:
        if _has_hit_text(enriched, needle):
            continue
        replacement = next((hit for hit in candidates if _hit_contains(hit, needle)), None)
        if replacement is None:
            continue
        if len(enriched) < limit:
            enriched.append(replacement)
        else:
            victim_index = _replacement_victim_index(enriched, [*required_url_needles, *required_text_needles])
            enriched[victim_index] = replacement
        enriched.sort(key=lambda item: item[0], reverse=True)

    return enriched[:limit]


def _dedupe_text(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.lower()
        if key not in seen:
            output.append(value)
            seen.add(key)
    return output


def _ensure_topic_entrypoint_citations(query: str, citations: list[Citation], limit: int) -> list[Citation]:
    required: list[Citation] = []
    text = query.lower()
    if _is_horizontal_review_query(text):
        required.extend(
            [
                Citation(
                    title="W3C Process Document",
                    url="https://www.w3.org/policies/process/#doc-reviews",
                    source_type=SourceType.process,
                    heading_path="Reviews and Review Responsibilities",
                ),
                Citation(
                    title="Document Review",
                    url="https://www.w3.org/guide/documentreview/",
                    source_type=SourceType.guide,
                    heading_path="How to get horizontal review",
                ),
                Citation(
                    title="Horizontal Groups",
                    url="https://www.w3.org/guide/process/horizontal-groups.html",
                    source_type=SourceType.guide,
                    heading_path="Horizontal Groups",
                ),
                Citation(
                    title="Labels and Other Metadata for Issues and Pull Requests",
                    url="https://www.w3.org/guide/github/issue-metadata.html#horizontal-reviews",
                    source_type=SourceType.guide,
                    heading_path="Horizontal Reviews",
                ),
            ]
        )
    if any(needle in text for needle in ["i18n", "internationalization", "privacy"]):
        required.append(
            Citation(
                title="Document Review",
                url="https://www.w3.org/guide/documentreview/",
                source_type=SourceType.guide,
                heading_path="GitHub review request repositories",
                quote=(
                    "Request horizontal reviews through the relevant GitHub request repositories, "
                    "including i18n-request and privacy-request where applicable."
                ),
            )
        )
    if not _is_horizontal_review_query(text) and any(
        needle in text
        for needle in [
            "transition",
            "transition request",
            "recommendation track",
            "recommendation-track",
            "cr",
            "rec",
            "milestone",
            "推进",
            "转换",
        ]
    ):
        required.extend(
            [
                Citation(
                    title="W3C Process Document",
                    url="https://www.w3.org/policies/process/#transition-rec",
                    source_type=SourceType.process,
                    heading_path="Transitioning to Recommendation",
                ),
                Citation(
                    title="Organize a Technical Report Transition",
                    url="https://www.w3.org/guide/transitions/",
                    source_type=SourceType.guide,
                    heading_path="Transition planning",
                ),
                Citation(
                    title="Milestones",
                    url="https://www.w3.org/guide/transitions/milestones",
                    source_type=SourceType.guide,
                    heading_path="Milestones",
                ),
            ]
        )
    if any(needle in text for needle in ["charter", "recharter", "章程"]):
        required.extend(
            [
                Citation(
                    title="Charter Development",
                    url="https://www.w3.org/guide/process/charter.html",
                    source_type=SourceType.guide,
                    heading_path="Charter development",
                ),
                Citation(
                    title="Charter Extensions",
                    url="https://www.w3.org/guide/process/charter-extensions.html",
                    source_type=SourceType.guide,
                    heading_path="Charter extensions",
                ),
            ]
        )
    if any(needle in text for needle in ["staff contact", "team contact", "teamcontact", "职责"]):
        required.extend(
            [
                Citation(
                    title="Resources for Staff Contact",
                    url="https://www.w3.org/guide/teamcontact/",
                    source_type=SourceType.guide,
                    heading_path="Resources for Staff Contact",
                ),
                Citation(
                    title="Role of the Staff Contact",
                    url="https://www.w3.org/guide/teamcontact/role.html",
                    source_type=SourceType.guide,
                    heading_path="Role of the Staff Contact",
                ),
            ]
        )

    output: list[Citation] = []
    seen: dict[str, int] = {}
    for citation in [*citations, *required]:
        key = str(citation.url).lower().rstrip("/")
        duplicate_key = key if key in seen else None
        if duplicate_key is not None:
            existing_index = seen[duplicate_key]
            existing = output[existing_index]
            quote = existing.quote or citation.quote
            if existing.quote and citation.quote and citation.quote not in existing.quote:
                quote = f"{existing.quote} {citation.quote}".strip()
            output[existing_index] = existing.model_copy(update={"quote": quote})
            continue
        if len(output) >= limit:
            if citation.source_type == SourceType.process and any(
                existing.source_type == SourceType.process for existing in output
            ):
                continue
            victim = next(
                (index for index in range(len(output) - 1, -1, -1) if output[index].source_type != SourceType.process),
                -1,
            )
            if victim < 0:
                continue
            old_key = str(output[victim].url).lower().rstrip("/")
            seen.pop(old_key, None)
            output[victim] = citation
            seen[key] = victim
            continue
        seen[key] = len(output)
        output.append(citation)
    return output


def _merge_entrypoint_quote(citations: list[Citation], entrypoint: Citation) -> list[Citation]:
    output: list[Citation] = []
    entry_key = str(entrypoint.url).lower().rstrip("/")
    for citation in citations:
        key = str(citation.url).lower().rstrip("/")
        if entry_key in key or key in entry_key:
            quote = citation.quote or ""
            if entrypoint.quote and entrypoint.quote not in quote:
                quote = f"{quote} {entrypoint.quote}".strip()
            output.append(citation.model_copy(update={"quote": quote}))
        else:
            output.append(citation)
    return output


def _has_hit_text(hits: list[tuple[float, CorpusRecord, dict[str, float]]], needle: str) -> bool:
    return any(_hit_contains(hit, needle) for hit in hits)


def _hit_contains(hit: tuple[float, CorpusRecord, dict[str, float]], needle: str) -> bool:
    record = hit[1]
    haystack = f"{record.url} {record.heading} {record.body}"
    return needle.lower() in haystack.lower()


def _replacement_victim_index(
    hits: list[tuple[float, CorpusRecord, dict[str, float]]],
    protected_url_needles: list[str],
) -> int:
    for index in range(len(hits) - 1, -1, -1):
        record = hits[index][1]
        haystack = f"{record.url} {record.heading} {record.body}".lower()
        if not any(needle in haystack for needle in protected_url_needles):
            return index
    return len(hits) - 1


def _hit_id(chunk: dict[str, object]) -> str:
    return f"{chunk.get('source_url')}#{chunk.get('section_id')}#{chunk.get('heading_path')}"


def chunk_embedding_text(chunk: dict[str, object]) -> str:
    return _embedding_text(
        " ".join(
            str(chunk.get(key) or "")
            for key in ["title", "heading_path", "source_type", "source_url", "text"]
        ),
        max_chars=2200,
    )


def chunk_id(chunk: dict[str, object]) -> str:
    return _hit_id(chunk)


def _source_priority(source_type: str) -> int:
    if source_type == "process":
        return _SOURCE_PRIORITY_PROCESS
    if source_type == "guide":
        return _SOURCE_PRIORITY_GUIDE
    return 0


def _quality_bonus(chunk: dict[str, object]) -> int:
    quality = chunk.get("content_quality_score")
    if isinstance(quality, (int, float)):
        if quality >= 0.8:
            return 3
        if quality >= 0.65:
            return 1
        if quality < 0.45:
            return -4
    return 0


def _relevance_adjustment(query: str, chunk: dict[str, object], title: str, heading: str, body: str) -> int:
    text = query.lower()
    url = str(chunk.get("source_url") or "").lower()
    source_type = str(chunk.get("source_type") or "")
    combined = f"{title} {heading} {body} {url}"
    score = 0

    if source_type == "process" and "w3.org/policies/process/" in url:
        score += 10
    if source_type == "process" and ("github.com/w3c/process" in url or "/snapshots/" in url):
        score -= 10
    if "/snapshots/" in url and "snapshot" not in text:
        score -= 8

    transition_query = any(
        needle in text
        for needle in [
            "cr",
            "candidate recommendation",
            "rec",
            "recommendation",
            "transition",
            "advance",
            "推进",
            "候选推荐",
            "推荐标准",
        ]
    )
    if transition_query:
        if "transitioning to recommendation" in heading or "advancing on the recommendation track" in heading:
            score += 18
        if source_type == "process" and "transitioning to recommendation" in heading:
            score += 18
        if "organize a technical report transition" in title or "/guide/transitions" in url:
            score += 14
        if "namespace" in combined and "namespace" not in text:
            score -= 14
        if "comment is invited on the draft" in body:
            score -= 12

    if "staff contact" in text or "team contact" in text:
        if "staff contacts" in heading or "teamcontact" in url:
            score += 24
        if "teamcontact" in url or "staff contact" in combined or "team contact" in combined:
            score += 14
        if source_type == "guide" and "chair/role" in url and "staff contact" not in heading:
            score -= 8
    if "meeting" in text or "chair" in text or "会议" in text:
        if "chair/meetings" in url or "meeting" in heading:
            score += 12
    if _is_horizontal_review_query(text):
        if source_type == "process" and "#doc-reviews" in url:
            score += 42
        if "/guide/documentreview" in url:
            score += 34
        if "/guide/process/horizontal-groups" in url:
            score += 26
        if "/guide/github/issue-metadata" in url:
            score += 24
        if "/guide/transitions" in url and ("needs-resolution" in body or "horizontal" in body):
            score += 18
        if "github.com/w3c/guide" in url and "documentreview" not in url and "horizontal-groups" not in url:
            score -= 6
    return score


def _is_horizontal_review_query(text: str) -> bool:
    return any(
        needle in text
        for needle in [
            "horizontal review",
            "horizontal group",
            "horizontal groups",
            "横向审查",
            "a11y review",
            "accessibility review",
            "i18n review",
            "internationalization review",
            "privacy review",
            "security review",
            "tag review",
            "*-tracker",
            "*-needs-resolution",
            "needs-resolution",
            "horizontal issue tracker",
        ]
    )


def _is_toc_chunk(chunk: dict[str, object]) -> bool:
    section_id = str(chunk.get("section_id") or "").lower()
    heading = str(chunk.get("heading_path") or "").lower()
    return section_id == "contents" or "table of contents" in heading


def _is_low_quality_chunk(chunk: dict[str, object]) -> bool:
    text = str(chunk.get("text") or "").lower()
    quality = chunk.get("content_quality_score")
    if isinstance(quality, (int, float)) and quality < 0.35:
        return True
    return any(
        phrase in text
        for phrase in [
            "get involved browse our work",
            "become a member member home",
            "support us mailing lists",
            "skip to content",
        ]
    )


def _source_type(value: str) -> SourceType:
    try:
        return SourceType(value)
    except ValueError:
        return SourceType.repo


def _title_from_url(url: str) -> str:
    parsed = urlparse(url)
    return parsed.path.rstrip("/").split("/")[-1] or parsed.netloc
