from pydantic import BaseModel


class SourceConfig(BaseModel):
    name: str
    url: str
    repo: str | None = None
    source_type: str


AUTHORITATIVE_SOURCES = [
    SourceConfig(
        name="W3C Process Document",
        url="https://www.w3.org/policies/process/",
        repo="https://github.com/w3c/process",
        source_type="process",
    ),
    SourceConfig(
        name="W3C Guidebook",
        url="https://www.w3.org/guide/",
        repo="https://github.com/w3c/guide",
        source_type="guide",
    ),
]

