from app.core.config import Settings
from app.services.w3c_api import CatalogItem, W3CAPIClient, _direct_shortname_candidates, _has_meaningful_overlap, _terms


def test_w3c_api_persistent_cache_round_trip(tmp_path) -> None:
    cache_path = tmp_path / "w3c_api_cache.json"
    settings = Settings(
        w3c_api_enabled=True,
        w3c_api_cache_path=str(cache_path),
        w3c_api_persistent_cache_enabled=True,
    )

    client = W3CAPIClient(settings)
    catalog_key = f"{settings.w3c_api_base_url}:catalog:{settings.w3c_api_catalog_pages}"
    client._catalog_cache[catalog_key] = (
        123.0,
        [
            CatalogItem(
                entity_type="specification",
                title="CSS Grid Layout Module Level 1",
                api_url="https://api.w3.org/specifications/css-grid-1",
                token_score=0.8,
            )
        ],
    )
    client._detail_cache["https://api.w3.org/specifications/css-grid-1"] = (
        456.0,
        {"title": "CSS Grid Layout Module Level 1", "shortname": "css-grid-1"},
    )
    client._save_persistent_cache()

    W3CAPIClient._catalog_cache.clear()
    W3CAPIClient._detail_cache.clear()
    W3CAPIClient._persistent_loaded_paths.clear()

    restored = W3CAPIClient(settings)

    assert catalog_key in restored._catalog_cache
    assert restored._catalog_cache[catalog_key][1][0].api_url == "https://api.w3.org/specifications/css-grid-1"
    assert restored._detail_cache["https://api.w3.org/specifications/css-grid-1"][1]["shortname"] == "css-grid-1"


def test_w3c_api_terms_ignore_process_stopwords_and_split_shortname() -> None:
    terms = _terms("now wai-adapt symbol in CR, how to publish it in REC")

    assert "wai-adapt" in terms
    assert "adapt" in terms
    assert "symbol" in terms
    assert "wai" not in terms
    assert "how" not in terms
    assert "to" not in terms
    assert "in" not in terms
    assert "publish" not in terms


def test_w3c_api_direct_shortname_candidates_include_adapt_symbols() -> None:
    terms = _terms("wai-adapt symbol")

    assert "adapt-symbols" in _direct_shortname_candidates(terms)


def test_w3c_api_rejects_weak_entity_overlap() -> None:
    terms = _terms("now wai-adapt symbol in CR, how to publish it in REC")

    assert not _has_meaningful_overlap(
        terms,
        "Mobile Accessibility: How WCAG 2.0 and Other W3C/WAI Guidelines Apply to Mobile",
        "https://api.w3.org/specifications/mobile-accessibility-mapping",
    )
    assert _has_meaningful_overlap(
        terms,
        "WAI-Adapt: Symbols Module",
        "https://api.w3.org/specifications/adapt-symbols",
    )


class FakeW3CAPIClient(W3CAPIClient):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.catalog_called = False

    def _catalog(self):  # type: ignore[no-untyped-def]
        self.catalog_called = True
        return [
            CatalogItem(
                entity_type="specification",
                title="Mobile Accessibility: How WCAG 2.0 and Other W3C/WAI Guidelines Apply to Mobile",
                api_url="https://api.w3.org/specifications/mobile-accessibility-mapping",
                token_score=0.5,
            )
        ]

    def _detail(self, api_url: str) -> dict[str, object]:
        if api_url == "https://api.w3.org/specifications/adapt-symbols":
            return {
                "title": "WAI-Adapt: Symbols Module",
                "shortname": "adapt-symbols",
                "shortlink": "https://www.w3.org/TR/adapt-symbols/",
                "editor-draft": "https://w3c.github.io/personalization-semantics/content/",
                "_links": {
                    "latest-version": {
                        "href": "https://api.w3.org/specifications/adapt-symbols/versions/20230105",
                        "title": "Candidate Recommendation Snapshot",
                    }
                },
            }
        if api_url == "https://api.w3.org/specifications/adapt-symbols/versions/20230105":
            return {"date": "2023-01-05"}
        raise RuntimeError("not found")


def test_w3c_api_resolves_wai_adapt_symbols_before_catalog_fallback() -> None:
    settings = Settings(w3c_api_enabled=True)
    client = FakeW3CAPIClient(settings)

    entities = client.resolve_entities("now wai-adapt symbol in CR, how to publish it in rec")

    assert entities
    assert entities[0].shortname == "adapt-symbols"
    assert entities[0].title == "WAI-Adapt: Symbols Module"
    assert entities[0].status == "Candidate Recommendation Snapshot"
    assert not client.catalog_called
