import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "import_w3c_sources.py"
spec = importlib.util.spec_from_file_location("import_w3c_sources", SCRIPT_PATH)
assert spec and spec.loader
import_w3c_sources = importlib.util.module_from_spec(spec)
sys.modules["import_w3c_sources"] = import_w3c_sources
spec.loader.exec_module(import_w3c_sources)


def test_canonical_url_removes_fragments_queries_and_duplicate_slashes() -> None:
    assert (
        import_w3c_sources.canonical_url("https://www.w3.org/guide/?utm=1#top")
        == "https://www.w3.org/guide/"
    )
    assert (
        import_w3c_sources.canonical_url("https://www.w3.org/policies/process/#transition-rec")
        == "https://www.w3.org/policies/process/"
    )
    assert (
        import_w3c_sources.canonical_url("https://www.w3.org/guide/teamcontact/?utm=1#role")
        == "https://www.w3.org/guide/teamcontact"
    )
    assert (
        import_w3c_sources.canonical_url("https://www.w3.org/guide/teamcontact/index.html#role")
        == "https://www.w3.org/guide/teamcontact"
    )


def test_discover_internal_links_stays_inside_guide_and_dedupes() -> None:
    html = """
    <main>
      <a href="/guide/teamcontact/#role">Team Contact</a>
      <a href="/guide/teamcontact/">Team Contact duplicate</a>
      <a href="/TR/css-grid-1/">TR</a>
      <a href="/guide/assets/logo.svg">SVG</a>
      <a href="/guide/process/@@@">Placeholder</a>
      <a href="/guide/,validate">Validator</a>
      <a href="/guide/process/charter.md">Markdown source</a>
      <a href="mailto:public@example.org">Email</a>
    </main>
    """

    links = import_w3c_sources.discover_internal_links(
        html,
        base_url="https://www.w3.org/guide/",
        root_url="https://www.w3.org/guide/",
    )

    assert links == ["https://www.w3.org/guide/teamcontact"]


def test_canonical_url_preserves_policies_landing_trailing_slash() -> None:
    # /policies/ must keep its trailing slash — www.w3.org rejects the
    # slashless /policies form mid-handshake, so stripping it broke the
    # crawl seed. Sub-pages still get the slash stripped (canonical form).
    assert (
        import_w3c_sources.canonical_url("https://www.w3.org/policies/#contrib")
        == "https://www.w3.org/policies/"
    )
    assert (
        import_w3c_sources.canonical_url("https://www.w3.org/policies/email/")
        == "https://www.w3.org/policies/email"
    )


def test_discover_internal_links_honours_exclude_prefixes() -> None:
    # The /policies/ crawl excludes pages that already have their own
    # dedicated source (process, patent-policy, code-of-conduct,
    # antitrust) so they are not re-ingested under repo_name "policies".
    html = """
    <main>
      <a href="/policies/email/">Email policy</a>
      <a href="/policies/process/#intro">Process (excluded)</a>
      <a href="/policies/patent-policy/">Patent Policy (excluded)</a>
      <a href="/policies/code-of-conduct/">CoC (excluded)</a>
      <a href="/policies/antitrust-2024/">Antitrust versioned (excluded)</a>
      <a href="/policies/logos/">Logos policy</a>
    </main>
    """

    links = import_w3c_sources.discover_internal_links(
        html,
        base_url="https://www.w3.org/policies/",
        root_url="https://www.w3.org/policies/",
        exclude_prefixes=[
            "/policies/process",
            "/policies/patent-policy",
            "/policies/code-of-conduct",
            "/policies/antitrust",
        ],
    )

    assert links == [
        "https://www.w3.org/policies/email",
        "https://www.w3.org/policies/logos",
    ]
