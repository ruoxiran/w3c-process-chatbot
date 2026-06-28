#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES_DIR = ROOT / "data" / "sources"
DEFAULT_CORPUS_DIR = ROOT / "data" / "corpus"

WEB_SOURCES = [
    {
        "name": "W3C Process Document",
        "url": "https://www.w3.org/policies/process/",
        "source_type": "process",
        "repo_name": "process",
    },
    {
        "name": "W3C Guidebook",
        "url": "https://www.w3.org/guide/",
        "source_type": "guide",
        "repo_name": "guide",
    },
    # Tier-1 policy documents the Process + Guidebook reference but
    # never define in full. Each is a single HTML page (same shape as
    # the Process Document) so depth=0 / max_pages=1 — the default for
    # non-guide sources — is correct. Tagged ``process`` so the prompt
    # treats them as normative authority alongside the main Process
    # Document, not as practice guidance.
    {
        "name": "W3C Patent Policy",
        "url": "https://www.w3.org/policies/patent-policy/",
        "source_type": "process",
        "repo_name": "patent-policy",
    },
    {
        "name": "W3C Code of Conduct",
        "url": "https://www.w3.org/policies/code-of-conduct/",
        "source_type": "process",
        "repo_name": "code-of-conduct",
    },
    {
        "name": "W3C Antitrust Policy",
        "url": "https://www.w3.org/policies/antitrust/",
        "source_type": "process",
        "repo_name": "antitrust",
    },
    # Tier-2 horizontal-review reference doc the Guidebook
    # documentreview chapter links to but never inlines. Single-page
    # TR with ~50 sections; gives the model concrete questionnaire
    # content to cite alongside the "file a privacy/security review
    # request" step instead of only naming the tracker URL.
    #
    # Source type is ``guide`` so internal-section anchors get
    # followed (depth=4 by default), but ``is_crawlable_internal_url``
    # keeps the crawl scoped to ``/TR/security-privacy-questionnaire/``
    # — no fan-out into the broader ``/TR/`` tree.
    {
        "name": "W3C Security and Privacy Questionnaire",
        "url": "https://www.w3.org/TR/security-privacy-questionnaire/",
        "source_type": "guide",
        "repo_name": "security-privacy-questionnaire",
    },
    # WAI standards landing was tried and reverted in round 27:
    # ``/WAI/standards-guidelines/`` fans out into ACT (~14k chunks),
    # WCAG (~1k), and 10+ language translations — would dominate the
    # corpus and bias retrieval toward a11y-only content. The TAG
    # design-reviews + i18n-request action surfaces already cover the
    # operational entry points; the per-axis review specifics belong
    # on the external sites, not in this corpus.
]

REPOS = [
    {
        "name": "w3c/process",
        "url": "https://github.com/w3c/process.git",
        "directory": "process",
        "source_type": "process",
    },
    {
        "name": "w3c/guide",
        "url": "https://github.com/w3c/guide.git",
        "directory": "guide",
        "source_type": "guide",
    },
]

TEXT_EXTENSIONS = {
    ".bs",
    ".css",
    ".html",
    ".js",
    ".json",
    ".md",
    ".py",
    ".txt",
    ".yml",
    ".yaml",
}

SKIP_DIRS = {".git", "node_modules", ".cache", "build", "dist", ".next"}


@dataclass
class Chunk:
    id: str
    title: str
    text: str
    source_url: str
    source_type: str
    heading_path: str | None = None
    section_id: str | None = None
    repo_url: str | None = None
    repo_name: str | None = None
    file_path: str | None = None
    commit_sha: str | None = None
    published_version_date: str | None = None
    indexed_at: str | None = None
    content_quality_score: float | None = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Import W3C Process and Guidebook sources.")
    parser.add_argument("--sources-dir", type=Path, default=DEFAULT_SOURCES_DIR)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--skip-repos", action="store_true")
    parser.add_argument(
        "--guide-depth",
        type=int,
        default=4,
        help="How many levels of same-guide web links to crawl from https://www.w3.org/guide/.",
    )
    parser.add_argument(
        "--guide-max-pages",
        type=int,
        default=500,
        help="Maximum same-guide web pages to crawl. This prevents deep crawls from expanding unexpectedly.",
    )
    args = parser.parse_args()

    args.sources_dir.mkdir(parents=True, exist_ok=True)
    args.corpus_dir.mkdir(parents=True, exist_ok=True)

    repo_info = {}
    if not args.skip_repos:
        for repo in REPOS:
            repo_path = args.sources_dir / repo["directory"]
            sync_repo(repo["url"], repo_path)
            repo_info[repo["directory"]] = {
                "commit_sha": git_output(repo_path, "rev-parse", "HEAD"),
                "remote_url": repo["url"].removesuffix(".git"),
            }

    indexed_at = datetime.now(timezone.utc).isoformat()
    chunks: list[Chunk] = []

    for source in WEB_SOURCES:
        info = repo_info.get(source["repo_name"], {})
        web_pages = crawl_web_source(
            source=source,
            sources_dir=args.sources_dir,
            max_depth=args.guide_depth if source["source_type"] == "guide" else 0,
            max_pages=args.guide_max_pages if source["source_type"] == "guide" else 1,
        )
        for page in web_pages:
            chunks.extend(
                html_chunks(
                    html=page["html"],
                    title=page["title"],
                    url=page["url"],
                    source_type=source["source_type"],
                    repo_name=source["repo_name"],
                    repo_url=info.get("remote_url"),
                    commit_sha=info.get("commit_sha"),
                    indexed_at=indexed_at,
                    page_depth=page["depth"],
                    parent_url=page.get("parent_url"),
                )
            )

    for repo in REPOS:
        repo_path = args.sources_dir / repo["directory"]
        if not repo_path.exists():
            continue
        info = repo_info.get(repo["directory"], {})
        chunks.extend(
            repo_chunks(
                repo_path=repo_path,
                repo_name=repo["name"],
                repo_url=info.get("remote_url", repo["url"].removesuffix(".git")),
                commit_sha=info.get("commit_sha"),
                source_type=repo["source_type"],
                indexed_at=indexed_at,
            )
        )

    corpus_path = args.corpus_dir / "chunks.jsonl"
    with corpus_path.open("w", encoding="utf-8") as output:
        for chunk in chunks:
            output.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    manifest = {
        "indexed_at": indexed_at,
        "chunk_count": len(chunks),
        "corpus_path": str(corpus_path.relative_to(ROOT)),
        "web_sources": WEB_SOURCES,
        "guide_depth": args.guide_depth,
        "guide_max_pages": args.guide_max_pages,
        "repos": repo_info,
    }
    (args.corpus_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def sync_repo(url: str, path: Path) -> None:
    if path.exists():
        subprocess.run(["git", "-C", str(path), "pull", "--ff-only"], check=True)
    else:
        subprocess.run(["git", "clone", url, str(path)], check=True)


def git_output(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def fetch(url: str) -> str:
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.text


def crawl_web_source(
    *,
    source: dict[str, str],
    sources_dir: Path,
    max_depth: int,
    max_pages: int,
) -> list[dict[str, str | int | None]]:
    """Fetch a web source and same-guide internal links up to max_depth.

    The crawl is intentionally narrow: it follows only HTML-like links under the
    original source path, so Guidebook expansion cannot wander into arbitrary
    W3C pages or user-provided URLs.
    """
    queue: list[tuple[str, int, str | None]] = [(canonical_url(source["url"]), 0, None)]
    visited: set[str] = set()
    pages: list[dict[str, str | int | None]] = []

    while queue and len(pages) < max_pages:
        url, depth, parent_url = queue.pop(0)
        if url in visited or depth > max_depth:
            continue
        visited.add(url)

        try:
            html = fetch(url)
        except httpx.HTTPError as exc:
            print(f"Skipping {url}: {exc}", flush=True)
            continue
        slug = _page_slug(source["repo_name"], url)
        (sources_dir / f"{slug}.html").write_text(html, encoding="utf-8")

        soup = BeautifulSoup(html, "html.parser")
        title = page_title(soup) or source["name"]
        pages.append(
            {
                "html": html,
                "title": title,
                "url": url,
                "depth": depth,
                "parent_url": parent_url,
            }
        )

        if depth >= max_depth:
            continue

        for link in discover_internal_links(html, base_url=url, root_url=source["url"]):
            if link not in visited and all(queued[0] != link for queued in queue):
                queue.append((link, depth + 1, url))

    return pages


def discover_internal_links(html: str, *, base_url: str, root_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: list[str] = []
    seen: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        href = str(anchor.get("href") or "").strip()
        url = canonical_url(urljoin(base_url, href))
        if not is_crawlable_internal_url(url, root_url):
            continue
        if url in seen:
            continue
        seen.add(url)
        links.append(url)
    return links


def is_crawlable_internal_url(url: str, root_url: str) -> bool:
    parsed = urlparse(url)
    root = urlparse(root_url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if parsed.netloc.lower() != root.netloc.lower():
        return False
    root_path = root.path if root.path.endswith("/") else f"{root.path}/"
    if not parsed.path.startswith(root_path):
        return False
    if "," in parsed.path or "@@" in parsed.path:
        return False
    if Path(parsed.path).suffix.lower() in {".css", ".js", ".json", ".md", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".pdf", ".zip"}:
        return False
    return True


def canonical_url(url: str) -> str:
    clean_url, _fragment = urldefrag(url)
    parsed = urlparse(clean_url)
    path = re.sub(r"/index\.html?$", "/", parsed.path)
    if path not in {"/", "/guide/", "/policies/process/"} and path.endswith("/"):
        path = path.rstrip("/")
    return parsed._replace(path=path, query="", fragment="").geturl()


def page_title(soup: BeautifulSoup) -> str | None:
    h1 = soup.find("h1")
    if h1:
        text = clean_text(h1.get_text(" ", strip=True))
        if text:
            return text
    if soup.title:
        text = clean_text(soup.title.get_text(" ", strip=True))
        if text:
            return text
    return None


def _page_slug(repo_name: str, url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.strip("/").replace("/", "-") or "root"
    return f"{repo_name}-{path}"


def html_chunks(
    *,
    html: str,
    title: str,
    url: str,
    source_type: str,
    repo_name: str,
    repo_url: str | None,
    commit_sha: str | None,
    indexed_at: str,
    page_depth: int = 0,
    parent_url: str | None = None,
) -> list[Chunk]:
    soup = BeautifulSoup(html, "html.parser")
    content = main_content(soup)
    version_date = find_version_date(soup)
    headings: list[str] = [title]
    chunks: list[Chunk] = []
    index = 0

    for element in content.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = clean_text(element.get_text(" ", strip=True))
        if not text:
            continue
        if element.name in {"h1", "h2", "h3", "h4"}:
            level = int(element.name[1])
            headings = headings[: max(1, level - 1)]
            headings.append(text)
            continue
        if len(text) < 40:
            continue
        quality = content_quality(text)
        if quality < 0.35:
            continue
        index += 1
        section_id = nearest_section_id(element)
        source_url = f"{url}#{section_id}" if section_id else url
        chunks.append(
            Chunk(
                id=f"{repo_name}:web:{page_depth}:{index}:{stable_hash(url)}",
                title=title,
                text=text,
                source_url=source_url,
                source_type=source_type,
                heading_path=heading_path(headings, page_depth, parent_url),
                section_id=section_id,
                repo_url=repo_url,
                repo_name=repo_name,
                commit_sha=commit_sha,
                published_version_date=version_date,
                indexed_at=indexed_at,
                content_quality_score=quality,
            )
        )
    return chunks


def main_content(soup: BeautifulSoup):
    for selector in ["script", "style", "noscript", "svg", "header", "footer", "nav", "aside", "[role='navigation']", ".breadcrumbs", ".breadcrumb"]:
        for element in soup.select(selector):
            element.decompose()

    for selector in ["main", "article", "#main", "#content", ".content", ".main"]:
        element = soup.select_one(selector)
        if element:
            return element
    return soup.body or soup


def content_quality(text: str) -> float:
    lower = text.lower()
    words = re.findall(r"[a-zA-Z][a-zA-Z-]+|[\u4e00-\u9fff]", text)
    unique_ratio = len(set(words)) / max(len(words), 1)
    score = 0.55
    if len(text) >= 120:
        score += 0.15
    if len(text) >= 260:
        score += 0.1
    if any(token in lower for token in ["must", "should", "process", "guidebook", "working group", "charter", "review", "transition", "staff contact"]):
        score += 0.15
    if unique_ratio < 0.35:
        score -= 0.2
    if any(
        phrase in lower
        for phrase in [
            "get involved browse our work",
            "become a member member home",
            "support us mailing lists",
            "news & events",
            "skip to content",
        ]
    ):
        score -= 0.45
    return max(0.0, min(1.0, score))


def repo_chunks(
    *,
    repo_path: Path,
    repo_name: str,
    repo_url: str,
    commit_sha: str,
    source_type: str,
    indexed_at: str,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    index = 0
    for file_path in iter_text_files(repo_path):
        rel = file_path.relative_to(repo_path)
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for part_index, text_chunk in enumerate(split_text(text), start=1):
            index += 1
            file_url = f"{repo_url}/blob/{commit_sha}/{rel.as_posix()}"
            chunks.append(
                Chunk(
                    id=f"{repo_name}:repo:{index}",
                    title=f"{repo_name}: {rel.as_posix()}",
                    text=text_chunk,
                    source_url=file_url,
                    source_type=source_type,
                    heading_path=rel.as_posix(),
                    repo_url=repo_url,
                    repo_name=repo_name,
                    file_path=rel.as_posix(),
                    commit_sha=commit_sha,
                    indexed_at=indexed_at,
                )
            )
    return chunks


def iter_text_files(root: Path):
    for current_root, dirs, files in os.walk(root):
        dirs[:] = [directory for directory in dirs if directory not in SKIP_DIRS]
        for filename in files:
            path = Path(current_root) / filename
            if path.suffix.lower() in TEXT_EXTENSIONS:
                yield path


def split_text(text: str, max_chars: int = 1400) -> list[str]:
    blocks = [clean_text(block) for block in re.split(r"\n\s*\n", text)]
    blocks = [block for block in blocks if len(block) >= 40]
    chunks: list[str] = []
    current = ""
    for block in blocks:
        if len(current) + len(block) + 2 > max_chars and current:
            chunks.append(current)
            current = block
        else:
            current = f"{current}\n\n{block}".strip()
    if current:
        chunks.append(current)
    return chunks


def find_version_date(soup: BeautifulSoup) -> str | None:
    body_text = soup.get_text("\n", strip=True)
    match = re.search(r"\b\d{1,2}\s+[A-Z][a-z]+\s+\d{4}\b", body_text)
    return match.group(0) if match else None


def nearest_section_id(element) -> str | None:
    current = element
    while current is not None:
        if current.get("id"):
            return current.get("id")
        current = current.find_previous(["section", "h1", "h2", "h3", "h4"])
    return None


def heading_path(headings: list[str], page_depth: int, parent_url: str | None) -> str:
    parts = list(headings)
    if page_depth and parent_url:
        parts.insert(1, f"Linked from {parent_url}")
    return " > ".join(parts)


def stable_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


if __name__ == "__main__":
    main()
