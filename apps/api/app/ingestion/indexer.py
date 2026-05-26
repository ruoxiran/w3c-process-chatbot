from bs4 import BeautifulSoup
import httpx

from app.ingestion.sources import AUTHORITATIVE_SOURCES, SourceConfig


class IndexedChunk(dict):
    pass


async def fetch_source(source: SourceConfig) -> str:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(source.url)
        response.raise_for_status()
        return response.text


def html_to_chunks(html: str, source: SourceConfig) -> list[IndexedChunk]:
    soup = BeautifulSoup(html, "html.parser")
    chunks: list[IndexedChunk] = []
    current_heading = source.name

    for element in soup.find_all(["h1", "h2", "h3", "p", "li"]):
        text = " ".join(element.get_text(" ", strip=True).split())
        if not text:
            continue
        if element.name in {"h1", "h2", "h3"}:
            current_heading = text
            continue
        chunks.append(
            IndexedChunk(
                text=text,
                source_url=source.url,
                repo_url=source.repo,
                source_type=source.source_type,
                heading_path=current_heading,
                section_id=element.get("id"),
            )
        )
    return chunks


async def build_preview_index() -> dict[str, int]:
    counts: dict[str, int] = {}
    for source in AUTHORITATIVE_SOURCES:
        html = await fetch_source(source)
        counts[source.name] = len(html_to_chunks(html, source))
    return counts

