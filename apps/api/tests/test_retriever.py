import json

from app.core.config import Settings
from app.models.schemas import SourceType
from app.rag.retriever import Retriever, chunk_id


def test_retriever_balances_process_and_guide_for_rec_track_question() -> None:
    citations = Retriever().retrieve("CSS spec 从 CR 到 REC 下一步做什么？")

    assert any(citation.source_type == SourceType.process for citation in citations)
    assert any(citation.source_type == SourceType.guide for citation in citations)
    assert any("transition" in (citation.heading_path or "").lower() for citation in citations)
    assert any("recommendation track" in (citation.heading_path or "").lower() for citation in citations)
    assert "w3.org/policies/process/#transition-rec" in str(citations[0].url)
    assert not any("/snapshots/" in str(citation.url) for citation in citations[:3])
    assert not any("namespace" in (citation.heading_path or "").lower() for citation in citations[:3])


def test_retriever_finds_guide_for_general_standards_progress_question() -> None:
    citations = Retriever().retrieve("我要推进一个标准，下一步流程是什么？")

    assert any(citation.source_type == SourceType.process for citation in citations)
    assert any(citation.source_type == SourceType.guide for citation in citations)
    assert any(citation.quote for citation in citations)


def test_retriever_treats_guide_repo_files_as_guidebook_content() -> None:
    citations = Retriever().retrieve("teamcontact index md Staff Contact")

    assert any(
        citation.source_type == SourceType.guide and "github.com/w3c/guide" in str(citation.url)
        for citation in citations
    )
    assert any("teamcontact" in (citation.heading_path or "").lower() for citation in citations)


def test_retriever_finds_deep_guidebook_web_pages() -> None:
    citations = Retriever().retrieve("What does the Guidebook say about preparing hybrid group meetings?")

    assert any(
        citation.source_type == SourceType.guide
        and "w3.org/guide/meetings/hybrid-meeting" in str(citation.url)
        for citation in citations
    )


def test_retriever_reranks_staff_contact_sources() -> None:
    citations = Retriever().retrieve("Staff Contact 的职责是什么？")

    assert any("staff contact" in (citation.heading_path or "").lower() for citation in citations)
    assert any("teamcontact" in str(citation.url).lower() for citation in citations)


def test_retriever_prioritizes_horizontal_review_guidebook_sources() -> None:
    citations = Retriever().retrieve("How should we request horizontal review and handle *-needs-resolution labels before CR?")

    assert any("guide/documentreview" in str(citation.url) for citation in citations[:3])
    assert any("policies/process/#doc-reviews" in str(citation.url) for citation in citations)
    assert any("guide/process/horizontal-groups" in str(citation.url) for citation in citations)
    assert any("guide/github/issue-metadata" in str(citation.url) for citation in citations)
    assert any("needs-resolution" in ((citation.quote or "") + " " + (citation.heading_path or "")).lower() for citation in citations)


def test_retriever_finds_specific_horizontal_review_request_guidance() -> None:
    citations = Retriever().retrieve("How should we request i18n review or privacy review through GitHub?")
    combined = " ".join(f"{citation.url} {citation.quote}" for citation in citations).lower()

    assert "documentreview" in combined
    assert "i18n-request" in combined or "privacy-request" in combined


def test_retriever_prioritises_meeting_tooling_entry_points_for_scribe_question() -> None:
    """A "how to scribe?" answer used to lead with the ``#pickvictim``
    Zakim feature (random scribe selection) instead of the IRC /
    invite-Zakim / RRSAgent entry points. Boost rules in
    guide_topics.RELEVANCE_RULES counterweight the lexical pull of
    the niche feature chunk; pin the resulting ordering here."""
    citations = Retriever().retrieve("how to scribe a meeting?")
    urls = [str(c.url).lower() for c in citations]
    combined = " ".join(urls)

    # The entry-point chapters must surface in the top retrieval pool.
    assert any("/guide/meetings/irc.html" in url for url in urls), (
        "IRC chapter missing from top results: " + combined
    )
    assert any("/guide/meetings/zakim.html" in url for url in urls), (
        "Zakim chapter missing from top results: " + combined
    )
    assert any("/guide/meetings/rrsagent.html" in url for url in urls), (
        "RRSAgent chapter missing from top results: " + combined
    )
    # The niche ``#pickvictim`` chunk should not lead the results for
    # a general "how to scribe" query — at best mid-pack.
    pickvictim_indices = [i for i, u in enumerate(urls) if "#pickvictim" in u]
    if pickvictim_indices:
        assert min(pickvictim_indices) >= 3, (
            f"#pickvictim should not lead general scribe results; "
            f"appeared at index {min(pickvictim_indices)}"
        )


def test_retriever_keeps_transition_guidebook_topic_pages() -> None:
    citations = Retriever().retrieve("How should a specification prepare a CR to REC transition request and milestones?")
    combined = " ".join(str(citation.url) for citation in citations).lower()

    assert "/policies/process/#transition-rec" in combined
    assert "/guide/transitions" in combined
    assert "/guide/transitions/milestones" in combined or "/guide/#rec-track" in combined


def test_retriever_keeps_charter_guidebook_topic_pages() -> None:
    citations = Retriever().retrieve("How should we prepare a recharter and charter review?")
    combined = " ".join(str(citation.url) for citation in citations).lower()

    assert "/guide/process/charter" in combined
    assert "/guide/process/charter-extensions" in combined or "/guide/tools/new-group" in combined


def test_retriever_keeps_staff_contact_guidebook_topic_pages() -> None:
    citations = Retriever().retrieve("What should the Staff Contact do for a Working Group process question?")
    combined = " ".join(str(citation.url) for citation in citations).lower()

    assert "/guide/teamcontact" in combined
    assert "/guide/teamcontact/role" in combined


def test_retriever_uses_dense_embedding_cache_when_enabled(tmp_path) -> None:
    relevant = {
        "title": "Dense only relevant page",
        "source_url": "https://www.w3.org/guide/dense-relevant/",
        "source_type": "guide",
        "heading_path": "Dense target",
        "section_id": "target",
        "text": "Opaque content that has no lexical overlap.",
        "content_quality_score": 0.9,
    }
    distractor = {
        "title": "Lexical distractor",
        "source_url": "https://www.w3.org/guide/dense-distractor/",
        "source_type": "guide",
        "heading_path": "Horizontal review distractor",
        "section_id": "distractor",
        "text": "horizontal review horizontal review horizontal review",
        "content_quality_score": 0.9,
    }
    corpus_path = tmp_path / "chunks.jsonl"
    corpus_path.write_text(
        "\n".join(json.dumps(item) for item in [relevant, distractor]) + "\n",
        encoding="utf-8",
    )

    cache_path = tmp_path / "embeddings.jsonl"
    cache_path.write_text(
        "\n".join(
            [
                json.dumps({"chunk_id": chunk_id(relevant), "model": "fake-embed", "embedding": [1.0, 0.0]}),
                json.dumps({"chunk_id": chunk_id(distractor), "model": "fake-embed", "embedding": [0.0, 1.0]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeEmbeddingClient:
        def embed(self, *, model: str, text: str) -> list[float]:
            return [1.0, 0.0]

    settings = Settings(
        corpus_path=str(corpus_path),
        retrieval_dense_enabled=True,
        retrieval_embedding_cache_path=str(cache_path),
        retrieval_dense_weight=80,
        ollama_embedding_model="fake-embed",
    )

    citations = Retriever(settings=settings, embedding_client=FakeEmbeddingClient()).retrieve("horizontal review")

    assert citations[0].url == relevant["source_url"]
